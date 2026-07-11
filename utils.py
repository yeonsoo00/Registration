"""Utility functions for affine registration.

Coordinate convention
---------------------
`torch.nn.functional.affine_grid` expects a matrix that maps output
coordinates to input coordinates. Therefore, if `M` is used in
`apply_affine_transform(image, M)`, the output pixel at coordinate `x`
samples the input image at `M @ x`.

The registration model predicts a matrix that maps fixed/target output
coordinates to moving-image input coordinates. Applying that matrix to the
moving image produces an image in the fixed/target coordinate system.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

import cv2
import numpy as np
from PIL import Image
import torch
import torch.nn.functional as F


def load_image(path: str, grayscale: bool = False) -> np.ndarray:
    """Load an image as float32 in [0, 1].

    RGB is the default because the histology signals are already
    pseudo-coloured. Grayscale loading should be requested explicitly only
    for mask generation or grayscale experiments.
    """
    mode = "L" if grayscale else "RGB"
    with Image.open(path) as img:
        arr = np.asarray(img.convert(mode), dtype=np.float32) / 255.0
    return arr


def resize_image(image: np.ndarray, size: Tuple[int, int], *, nearest: bool = False) -> np.ndarray:
    """Resize an image to `(height, width)` while preserving channel count."""
    interpolation = cv2.INTER_NEAREST if nearest else cv2.INTER_LINEAR
    return cv2.resize(image, (int(size[1]), int(size[0])), interpolation=interpolation)


def normalize_image(image: np.ndarray, per_channel: bool = True, eps: float = 1e-6) -> np.ndarray:
    """Z-score normalize a grayscale or RGB image.

    RGB images are normalized independently per channel by default so the
    pseudo-colour channels retain their relative spatial patterns.
    """
    image = image.astype(np.float32, copy=False)
    if image.ndim == 2 or not per_channel:
        mean = float(image.mean())
        std = float(image.std())
        return (image - mean) / max(std, eps)

    mean = image.mean(axis=(0, 1), keepdims=True)
    std = image.std(axis=(0, 1), keepdims=True)
    std = np.maximum(std, eps)
    return (image - mean) / std


def compute_mineral_mask(mineral_image: np.ndarray) -> np.ndarray:
    """Compute a binary mineral mask with Otsu thresholding."""
    if mineral_image.ndim == 3:
        mineral_image = cv2.cvtColor(
            np.clip(mineral_image * 255.0, 0, 255).astype(np.uint8),
            cv2.COLOR_RGB2GRAY,
        ).astype(np.float32) / 255.0
    img_uint8 = np.clip(mineral_image * 255.0, 0, 255).astype(np.uint8)
    _, thresh = cv2.threshold(img_uint8, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    mask = (thresh > 0).astype(np.float32)
    return mask


def compute_boundary_mask(mask: np.ndarray, dilation_iter: int = 3) -> np.ndarray:
    """Compute a band around the mineral boundary."""
    kernel = np.ones((3, 3), np.uint8)
    mask_u8 = (mask > 0.5).astype(np.uint8)
    dilated = cv2.dilate(mask_u8, kernel, iterations=dilation_iter)
    eroded = cv2.erode(mask_u8, kernel, iterations=dilation_iter)
    return ((dilated - eroded) > 0).astype(np.float32)


def compute_exterior_mask(mask: np.ndarray) -> np.ndarray:
    """Return the region outside the mineral mask."""
    return (1.0 - (mask > 0.5).astype(np.float32)).astype(np.float32)


@dataclass(frozen=True)
class PreprocessGeometry:
    """Geometry mapping original fixed-space images to model space."""

    original_height: int
    original_width: int
    crop_y0: int
    crop_y1: int
    crop_x0: int
    crop_x1: int
    resized_height: int
    resized_width: int
    pad_top: int
    pad_left: int
    output_height: int
    output_width: int
    scale: float

    def original_to_model_matrix(self) -> np.ndarray:
        """Return a 3×3 matrix mapping original pixels to model pixels."""
        return np.array(
            [
                [self.scale, 0.0, self.pad_left - self.scale * self.crop_x0],
                [0.0, self.scale, self.pad_top - self.scale * self.crop_y0],
                [0.0, 0.0, 1.0],
            ],
            dtype=np.float64,
        )


def compute_preprocess_geometry(
    mineral_mask: np.ndarray,
    output_size: Tuple[int, int],
    crop_mode: str = "full",
    crop_margin: int = 32,
) -> PreprocessGeometry:
    """Compute crop, aspect-preserving resize, and padding geometry.

    `crop_mode='full'` is the safest default for registration because it does
    not discard displaced moving signal before the network sees it. The
    mineral mask is still used to restrict the losses.
    """
    h, w = mineral_mask.shape[:2]
    out_h, out_w = int(output_size[0]), int(output_size[1])

    if crop_mode == "full":
        y0, y1, x0, x1 = 0, h, 0, w
    elif crop_mode == "mineral_bbox":
        ys, xs = np.where(mineral_mask > 0.5)
        if len(ys) == 0 or len(xs) == 0:
            y0, y1, x0, x1 = 0, h, 0, w
        else:
            y0 = max(0, int(ys.min()) - int(crop_margin))
            y1 = min(h, int(ys.max()) + 1 + int(crop_margin))
            x0 = max(0, int(xs.min()) - int(crop_margin))
            x1 = min(w, int(xs.max()) + 1 + int(crop_margin))
    else:
        raise ValueError(f"Unsupported crop_mode: {crop_mode}")

    crop_h = max(1, y1 - y0)
    crop_w = max(1, x1 - x0)
    scale = min(out_h / crop_h, out_w / crop_w)
    new_h = max(1, min(out_h, int(round(crop_h * scale))))
    new_w = max(1, min(out_w, int(round(crop_w * scale))))
    pad_top = (out_h - new_h) // 2
    pad_left = (out_w - new_w) // 2

    return PreprocessGeometry(
        original_height=h,
        original_width=w,
        crop_y0=y0,
        crop_y1=y1,
        crop_x0=x0,
        crop_x1=x1,
        resized_height=new_h,
        resized_width=new_w,
        pad_top=pad_top,
        pad_left=pad_left,
        output_height=out_h,
        output_width=out_w,
        scale=float(scale),
    )


def apply_preprocess_geometry(
    image: np.ndarray,
    geometry: PreprocessGeometry,
    *,
    nearest: bool = False,
) -> np.ndarray:
    """Apply a fixed-space crop/letterbox transform to an image.

    If the input size differs from the fixed mineral size, it is first resized
    to the fixed mineral canvas. This is required so moving, target, and fixed
    share one coordinate system before affine registration.
    """
    if image.shape[0] != geometry.original_height or image.shape[1] != geometry.original_width:
        image = resize_image(
            image,
            (geometry.original_height, geometry.original_width),
            nearest=nearest,
        )

    crop = image[
        geometry.crop_y0 : geometry.crop_y1,
        geometry.crop_x0 : geometry.crop_x1,
        ...,
    ]
    resized = resize_image(
        crop,
        (geometry.resized_height, geometry.resized_width),
        nearest=nearest,
    )

    if image.ndim == 2:
        canvas = np.zeros((geometry.output_height, geometry.output_width), dtype=np.float32)
        canvas[
            geometry.pad_top : geometry.pad_top + geometry.resized_height,
            geometry.pad_left : geometry.pad_left + geometry.resized_width,
        ] = resized
    else:
        canvas = np.zeros(
            (geometry.output_height, geometry.output_width, image.shape[2]),
            dtype=np.float32,
        )
        canvas[
            geometry.pad_top : geometry.pad_top + geometry.resized_height,
            geometry.pad_left : geometry.pad_left + geometry.resized_width,
            :,
        ] = resized
    return canvas.astype(np.float32, copy=False)


def affine_parameters_to_matrix(params: torch.Tensor) -> torch.Tensor:
    """Convert `(tx, ty, theta, sx, sy)` to affine-grid matrices.

    Translation is in normalized coordinates and rotation is in radians.
    The returned matrix maps output coordinates to input coordinates.
    """
    if params.ndim != 2 or params.shape[1] != 5:
        raise ValueError(f"params must have shape (B,5), got {tuple(params.shape)}")
    tx, ty, theta, sx, sy = torch.unbind(params, dim=1)
    cos = torch.cos(theta)
    sin = torch.sin(theta)
    return torch.stack(
        [
            torch.stack([sx * cos, -sy * sin, tx], dim=1),
            torch.stack([sx * sin, sy * cos, ty], dim=1),
        ],
        dim=1,
    )


def invert_affine_matrix(matrix: torch.Tensor) -> torch.Tensor:
    """Invert a batch of 2×3 affine matrices."""
    if matrix.ndim != 3 or matrix.shape[1:] != (2, 3):
        raise ValueError(f"matrix must have shape (B,2,3), got {tuple(matrix.shape)}")
    batch = matrix.shape[0]
    bottom = torch.tensor([0.0, 0.0, 1.0], dtype=matrix.dtype, device=matrix.device)
    bottom = bottom.view(1, 1, 3).expand(batch, -1, -1)
    homogeneous = torch.cat([matrix, bottom], dim=1)
    inverse = torch.linalg.inv(homogeneous)
    return inverse[:, :2, :]


def apply_affine_transform(
    image: torch.Tensor,
    matrix: torch.Tensor,
    mode: str = "bilinear",
    padding_mode: str = "zeros",
) -> torch.Tensor:
    """Warp an image with an affine-grid matrix."""
    if image.ndim != 4:
        raise ValueError(f"image must have shape (B,C,H,W), got {tuple(image.shape)}")
    if matrix.ndim != 3 or matrix.shape[1:] != (2, 3):
        raise ValueError(f"matrix must have shape (B,2,3), got {tuple(matrix.shape)}")
    grid = F.affine_grid(matrix, size=image.shape, align_corners=True)
    return F.grid_sample(
        image,
        grid,
        mode=mode,
        padding_mode=padding_mode,
        align_corners=True,
    )


def sample_registration_parameters(
    tx_range: Tuple[float, float],
    ty_range: Tuple[float, float],
    rot_range: Tuple[float, float],
    scale_range: Tuple[float, float],
    image_size: Tuple[int, int],
    *,
    rng: Optional[np.random.Generator] = None,
) -> torch.Tensor:
    """Sample the transform that should register moving → target.

    Translation ranges are specified in pixels of model space. Rotation is
    specified in degrees and converted exactly once to radians.
    """
    rng = rng if rng is not None else np.random.default_rng()
    h, w = int(image_size[0]), int(image_size[1])
    tx_px = float(rng.uniform(*tx_range))
    ty_px = float(rng.uniform(*ty_range))
    theta = float(np.deg2rad(rng.uniform(*rot_range)))
    scale = float(rng.uniform(*scale_range))
    tx = tx_px / max(w / 2.0, 1.0)
    ty = ty_px / max(h / 2.0, 1.0)
    return torch.tensor([tx, ty, theta, scale, scale], dtype=torch.float32)


def random_affine_matrix(
    tx_range: Tuple[float, float],
    ty_range: Tuple[float, float],
    rot_range: Tuple[float, float],
    scale_range: Tuple[float, float],
    batch_size: int = 1,
    device: Optional[torch.device] = None,
) -> torch.Tensor:
    """Backward-compatible random parameter sampler.

    Returns translation in pixels, rotation in radians, and uniform scales.
    New code should prefer `sample_registration_parameters` because it makes
    the image-size normalization explicit.
    """
    device = device or torch.device("cpu")
    tx = torch.empty(batch_size, device=device).uniform_(*tx_range)
    ty = torch.empty(batch_size, device=device).uniform_(*ty_range)
    theta = torch.empty(batch_size, device=device).uniform_(*rot_range) * np.pi / 180.0
    scale = torch.empty(batch_size, device=device).uniform_(*scale_range)
    return torch.stack([tx, ty, theta, scale, scale], dim=1)


def normalized_affine_to_pixel_matrix(
    matrix: torch.Tensor | np.ndarray,
    height: int,
    width: int,
) -> np.ndarray:
    """Convert an affine-grid matrix to a 3×3 pixel dst→src matrix.

    This conversion respects `align_corners=True`, including rotation and
    scaling around the image center. It is required when applying predicted
    PyTorch transforms to original-resolution images with OpenCV.
    """
    if torch.is_tensor(matrix):
        arr = matrix.detach().cpu().numpy()
    else:
        arr = np.asarray(matrix)
    if arr.shape == (1, 2, 3):
        arr = arr[0]
    if arr.shape != (2, 3):
        raise ValueError(f"Expected (2,3) matrix, got {arr.shape}")

    homogeneous = np.eye(3, dtype=np.float64)
    homogeneous[:2, :] = arr.astype(np.float64)

    sx = 2.0 / max(width - 1, 1)
    sy = 2.0 / max(height - 1, 1)
    pixel_to_norm = np.array(
        [[sx, 0.0, -1.0], [0.0, sy, -1.0], [0.0, 0.0, 1.0]],
        dtype=np.float64,
    )
    norm_to_pixel = np.linalg.inv(pixel_to_norm)
    return norm_to_pixel @ homogeneous @ pixel_to_norm
