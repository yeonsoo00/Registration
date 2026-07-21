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


STRUCTURAL_CHANNEL_NAMES = (
    "tissue",
    "foreground",
    "edge",
    "boundary_distance",
    "skeleton",
    "context",
)


def resolve_device(
    device_spec: str, gpu_ids_spec: str
) -> Tuple[torch.device, list[int]]:
    """Resolve one primary device and optional DataParallel device IDs."""
    requested = torch.device(device_spec)
    if requested.type != "cuda":
        return requested, []
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available")

    gpu_ids = [int(value) for value in gpu_ids_spec.split(",") if value.strip()]
    if not gpu_ids:
        gpu_ids = [requested.index if requested.index is not None else 0]
    if len(set(gpu_ids)) != len(gpu_ids):
        raise ValueError("gpu_ids must not contain duplicates")
    device_count = torch.cuda.device_count()
    if any(gpu_id < 0 or gpu_id >= device_count for gpu_id in gpu_ids):
        raise ValueError(
            f"gpu_ids {gpu_ids} are invalid for {device_count} visible CUDA devices"
        )
    return torch.device(f"cuda:{gpu_ids[0]}"), gpu_ids


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


def resize_image(
    image: np.ndarray, size: Tuple[int, int], *, nearest: bool = False
) -> np.ndarray:
    """Resize an image to `(height, width)` while preserving channel count."""
    interpolation = cv2.INTER_NEAREST if nearest else cv2.INTER_LINEAR
    return cv2.resize(image, (int(size[1]), int(size[0])), interpolation=interpolation)


def normalize_image(
    image: np.ndarray, per_channel: bool = True, eps: float = 1e-6
) -> np.ndarray:
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


def structural_likelihood(
    image: np.ndarray,
    *,
    color_space: str = "rgb",
    valid_mask: Optional[np.ndarray] = None,
    lower_percentile: float = 1.0,
    upper_percentile: float = 99.5,
    eps: float = 1e-6,
) -> np.ndarray:
    """Convert a raw model canvas to one stain-invariant signal likelihood.

    RGB fluorescence uses its maximum colour channel, while HSV input uses
    only value, so hue and saturation are never interpreted as brightness.
    Robust percentile scaling maps the valid field of view to ``[0, 1]``
    without z-scoring black background or padding.
    """
    if color_space not in {"rgb", "hsv"}:
        raise ValueError("color_space must be rgb or hsv")
    if not 0.0 <= lower_percentile < upper_percentile <= 100.0:
        raise ValueError("structural likelihood percentiles are invalid")

    source = np.asarray(image, dtype=np.float32)
    if source.ndim == 2:
        intensity = source
    elif source.ndim == 3 and source.shape[2] == 1:
        intensity = source[..., 0]
    elif source.ndim == 3 and source.shape[2] == 3:
        intensity = source[..., 2] if color_space == "hsv" else source.max(axis=2)
    else:
        raise ValueError("structural likelihood expects HxW, HxWx1, or HxWx3 input")

    intensity = np.nan_to_num(intensity, nan=0.0, posinf=1.0, neginf=0.0)
    intensity = np.clip(intensity, 0.0, 1.0).astype(np.float32, copy=False)
    if valid_mask is None:
        valid = np.ones(intensity.shape, dtype=bool)
    else:
        mask = np.asarray(valid_mask)
        if mask.shape != intensity.shape:
            raise ValueError(
                f"valid_mask shape {mask.shape} does not match image {intensity.shape}"
            )
        valid = mask > 0.5

    output = np.zeros_like(intensity, dtype=np.float32)
    values = intensity[valid]
    if values.size == 0:
        return output

    low = float(np.percentile(values, lower_percentile))
    high = float(np.percentile(values, upper_percentile))
    maximum = float(values.max())
    if high - low <= eps and maximum > low + eps:
        high = maximum
    if high - low > eps:
        output = np.clip((intensity - low) / (high - low), 0.0, 1.0)
    elif maximum > eps:
        # A spatially constant positive canvas is a valid full foreground,
        # whereas an all-zero/empty canvas must remain exactly zero.
        output = np.clip(intensity / maximum, 0.0, 1.0)
    output[~valid] = 0.0
    return output.astype(np.float32, copy=False)


def build_structural_descriptor(
    likelihood: np.ndarray,
    *,
    valid_mask: Optional[np.ndarray] = None,
    foreground_threshold: Optional[float] = None,
    distance_scale: float = 0.03,
    context_scale: float = 0.03,
    skeleton_radius: int = 4,
) -> np.ndarray:
    """Build a bounded HWC structural representation from raw likelihood.

    Channel order is :data:`STRUCTURAL_CHANNEL_NAMES`. Distances and Gaussian
    context scales are fractions of the shorter image side, which keeps their
    physical support comparable across model resolutions. Empty and full
    foregrounds deliberately have zero boundary-distance and skeleton maps.
    """
    tissue = np.asarray(likelihood, dtype=np.float32)
    if tissue.ndim != 2:
        raise ValueError("structural likelihood must be an HxW array")
    if foreground_threshold is not None and not 0.0 <= foreground_threshold <= 1.0:
        raise ValueError("foreground_threshold must be in [0, 1] or None")
    if distance_scale <= 0.0:
        raise ValueError("distance_scale must be positive")
    if context_scale <= 0.0:
        raise ValueError("context_scale must be positive")
    if skeleton_radius < 0:
        raise ValueError("skeleton_radius cannot be negative")

    tissue = np.nan_to_num(tissue, nan=0.0, posinf=1.0, neginf=0.0)
    tissue = np.clip(tissue, 0.0, 1.0)
    if valid_mask is None:
        valid = np.ones(tissue.shape, dtype=bool)
    else:
        mask = np.asarray(valid_mask)
        if mask.shape != tissue.shape:
            raise ValueError(
                f"valid_mask shape {mask.shape} does not match likelihood {tissue.shape}"
            )
        valid = mask > 0.5
    valid_float = valid.astype(np.float32)
    tissue = (tissue * valid_float).astype(np.float32, copy=False)

    empty_channel = np.zeros_like(tissue, dtype=np.float32)
    valid_values = tissue[valid]
    if valid_values.size == 0:
        return np.stack([empty_channel] * len(STRUCTURAL_CHANNEL_NAMES), axis=-1)

    if foreground_threshold is None:
        if float(valid_values.max()) <= 1e-6:
            threshold = 1.0
        elif float(valid_values.max() - valid_values.min()) <= 1e-6:
            threshold = 0.0
        else:
            valid_u8 = np.clip(valid_values * 255.0, 0, 255).astype(np.uint8)
            threshold_u8, _ = cv2.threshold(
                valid_u8.reshape(-1, 1),
                0,
                255,
                cv2.THRESH_BINARY + cv2.THRESH_OTSU,
            )
            threshold = float(threshold_u8) / 255.0
    else:
        threshold = float(foreground_threshold)
    foreground_bool = (tissue > threshold) & valid
    foreground = foreground_bool.astype(np.float32)
    foreground_count = int(foreground_bool.sum())
    valid_count = int(valid.sum())
    empty_foreground = foreground_count == 0
    full_foreground = foreground_count == valid_count

    gx = cv2.Sobel(tissue, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(tissue, cv2.CV_32F, 0, 1, ksize=3)
    edge = cv2.magnitude(gx, gy)
    valid_u8 = valid.astype(np.uint8)
    edge_valid = cv2.erode(valid_u8, np.ones((3, 3), np.uint8)) > 0
    if not edge_valid.any():
        edge_valid = valid
    edge_values = edge[edge_valid]
    edge_scale = float(np.percentile(edge_values, 99.0)) if edge_values.size else 0.0
    if edge_scale <= 1e-6 and edge_values.size:
        edge_scale = float(edge_values.max())
    if edge_scale > 1e-6:
        edge = np.clip(edge / edge_scale, 0.0, 1.0)
    else:
        edge = np.zeros_like(tissue)
    edge = (edge * edge_valid.astype(np.float32)).astype(np.float32)

    boundary_distance = np.zeros_like(tissue, dtype=np.float32)
    skeleton = np.zeros_like(tissue, dtype=np.float32)
    if not empty_foreground and not full_foreground:
        foreground_u8 = foreground_bool.astype(np.uint8)
        distance_inside = cv2.distanceTransform(
            foreground_u8, cv2.DIST_L2, cv2.DIST_MASK_PRECISE
        )
        distance_outside = cv2.distanceTransform(
            1 - foreground_u8, cv2.DIST_L2, cv2.DIST_MASK_PRECISE
        )
        signed_distance = distance_inside - distance_outside
        distance_pixels = max(float(distance_scale) * float(min(tissue.shape)), 1.0)
        boundary_distance = np.exp(-np.abs(signed_distance) / distance_pixels).astype(
            np.float32
        )
        boundary_distance *= valid_float

        local_maximum = cv2.dilate(
            distance_inside,
            cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3)),
        )
        ridge = foreground_bool & (distance_inside >= local_maximum - 1e-6)
        if ridge.any():
            if skeleton_radius == 0:
                skeleton = ridge.astype(np.float32)
            else:
                distance_to_ridge = cv2.distanceTransform(
                    (~ridge).astype(np.uint8),
                    cv2.DIST_L2,
                    cv2.DIST_MASK_PRECISE,
                )
                skeleton = np.clip(
                    1.0 - distance_to_ridge / float(skeleton_radius), 0.0, 1.0
                ).astype(np.float32)
            skeleton *= foreground * valid_float

    def masked_gaussian(source: np.ndarray, sigma: float) -> np.ndarray:
        sigma = max(float(sigma), 0.5)
        numerator = cv2.GaussianBlur(
            source * valid_float, (0, 0), sigmaX=sigma, sigmaY=sigma
        )
        denominator = cv2.GaussianBlur(valid_float, (0, 0), sigmaX=sigma, sigmaY=sigma)
        return np.divide(
            numerator,
            np.maximum(denominator, 1e-6),
            out=np.zeros_like(numerator, dtype=np.float32),
            where=denominator > 1e-6,
        )

    context_sigma = max(float(context_scale) * float(min(tissue.shape)), 0.5)
    context = 0.5 * masked_gaussian(tissue, context_sigma)
    context += 0.5 * masked_gaussian(foreground, 2.0 * context_sigma)
    context = np.clip(context * valid_float, 0.0, 1.0).astype(np.float32)

    descriptor = np.stack(
        (tissue, foreground, edge, boundary_distance, skeleton, context), axis=-1
    )
    descriptor = np.nan_to_num(descriptor, nan=0.0, posinf=1.0, neginf=0.0)
    return np.ascontiguousarray(np.clip(descriptor, 0.0, 1.0), dtype=np.float32)


def compute_mineral_mask(mineral_image: np.ndarray) -> np.ndarray:
    """Compute a binary mineral mask with Otsu thresholding."""
    if mineral_image.ndim == 3:
        mineral_image = (
            cv2.cvtColor(
                np.clip(mineral_image * 255.0, 0, 255).astype(np.uint8),
                cv2.COLOR_RGB2GRAY,
            ).astype(np.float32)
            / 255.0
        )
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


def compute_preprocess_valid_mask(geometry: PreprocessGeometry) -> np.ndarray:
    """Return the binary model-space field of view for letterbox geometry."""
    mask = np.zeros((geometry.output_height, geometry.output_width), dtype=np.float32)
    y0 = int(geometry.pad_top)
    x0 = int(geometry.pad_left)
    y1 = y0 + int(geometry.resized_height)
    x1 = x0 + int(geometry.resized_width)
    if y0 < 0 or x0 < 0 or y1 > geometry.output_height or x1 > geometry.output_width:
        raise ValueError("PreprocessGeometry places resized data outside its canvas")
    mask[y0:y1, x0:x1] = 1.0
    return mask


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
    if (
        image.shape[0] != geometry.original_height
        or image.shape[1] != geometry.original_width
    ):
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
        canvas = np.zeros(
            (geometry.output_height, geometry.output_width), dtype=np.float32
        )
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


def compose_affine_grid_warps(
    first_matrix: torch.Tensor,
    second_matrix: torch.Tensor,
) -> torch.Tensor:
    """Compose two sequential ``affine_grid`` warps without resampling.

    ``apply_affine_transform(apply_affine_transform(image, first), second)``
    samples the original image with ``first @ second`` because affine-grid
    matrices map output coordinates back to input coordinates. Returning the
    composed 2x3 matrix lets callers apply that geometry to the original image
    once, avoiding clipping and interpolation from an intermediate canvas.
    """
    if first_matrix.ndim != 3 or first_matrix.shape[1:] != (2, 3):
        raise ValueError(
            "first_matrix must have shape (B,2,3), got " f"{tuple(first_matrix.shape)}"
        )
    if second_matrix.shape != first_matrix.shape:
        raise ValueError(
            "second_matrix must match first_matrix; got "
            f"{tuple(second_matrix.shape)} and {tuple(first_matrix.shape)}"
        )
    batch = first_matrix.shape[0]
    bottom = first_matrix.new_tensor((0.0, 0.0, 1.0)).view(1, 1, 3)
    bottom = bottom.expand(batch, -1, -1)
    first_h = torch.cat((first_matrix, bottom), dim=1)
    second_h = torch.cat((second_matrix, bottom), dim=1)
    return torch.bmm(first_h, second_h)[:, :2, :]


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


def warp_group_with_matrix(
    moving_group: torch.Tensor,
    matrices: torch.Tensor,
) -> torch.Tensor:
    """Warp every stain slot with one explicit affine-grid matrix per item."""
    if moving_group.ndim != 5:
        raise ValueError(
            "moving_group must have shape BxKxCxHxW, got "
            f"{tuple(moving_group.shape)}"
        )
    batch_size, slots, channels, height, width = moving_group.shape
    if tuple(matrices.shape) != (batch_size, 2, 3):
        raise ValueError(
            f"matrices must have shape {(batch_size, 2, 3)}, got "
            f"{tuple(matrices.shape)}"
        )
    repeated_matrices = matrices[:, None].expand(batch_size, slots, 2, 3)
    flattened = moving_group.reshape(batch_size * slots, channels, height, width)
    warped = apply_affine_transform(
        flattened,
        repeated_matrices.reshape(batch_size * slots, 2, 3),
    )
    return warped.reshape(batch_size, slots, channels, height, width)


def warp_model_space_group(
    moving_group: torch.Tensor,
    predicted_params: torch.Tensor,
) -> torch.Tensor:
    """Apply the canonical deployable model-space warp to a grouped tensor."""
    return warp_group_with_matrix(
        moving_group,
        affine_parameters_to_matrix(predicted_params),
    )


def warp_group(moving_group: torch.Tensor, params: torch.Tensor) -> torch.Tensor:
    """Backward-compatible alias for the canonical model-space group warp."""
    return warp_model_space_group(moving_group, params)


def synthetic_full_correction_matrices(
    predicted_params: torch.Tensor,
    params_true: torch.Tensor,
) -> torch.Tensor:
    """Compose prediction and known synthesis for a one-pass source warp."""
    if predicted_params.shape != params_true.shape:
        raise ValueError(
            "predicted_params and params_true must have identical shapes; got "
            f"{tuple(predicted_params.shape)} and {tuple(params_true.shape)}"
        )
    predicted_matrix = affine_parameters_to_matrix(predicted_params)
    registration_matrix = affine_parameters_to_matrix(params_true)
    synthesis_matrix = invert_affine_matrix(registration_matrix)
    return compose_affine_grid_warps(synthesis_matrix, predicted_matrix)


def supervision_source_and_matrices(
    moving_group: torch.Tensor,
    target_group: torch.Tensor,
    params: torch.Tensor,
    params_true: torch.Tensor,
    has_params: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Select one-pass synthetic or ordinary real supervision geometry.

    Synthetic items use the original registered ``target_group`` with
    ``A_syn @ A_pred``. Real items have no known synthesis transform, so they
    retain ``warp(moving_group, A_pred)``. Selection is per item, allowing
    mixed batches while keeping targets out of prediction.
    """
    if moving_group.shape != target_group.shape:
        raise ValueError(
            "moving_group and target_group must have identical shapes; got "
            f"{tuple(moving_group.shape)} and {tuple(target_group.shape)}"
        )
    batch_size = moving_group.shape[0]
    if tuple(params.shape) != (batch_size, 5):
        raise ValueError(
            f"params must have shape {(batch_size, 5)}, got {tuple(params.shape)}"
        )
    if tuple(params_true.shape) != (batch_size, 5):
        raise ValueError(
            "params_true must have the same Bx5 shape as params; got "
            f"{tuple(params_true.shape)}"
        )
    synthetic_items = has_params.reshape(-1).bool()
    if synthetic_items.numel() != batch_size:
        raise ValueError(
            f"has_params must contain {batch_size} values, got "
            f"{synthetic_items.numel()}"
        )

    predicted_matrices = affine_parameters_to_matrix(params)
    if not synthetic_items.any():
        return moving_group, predicted_matrices

    synthetic_params_true = params_true[synthetic_items]
    if not torch.isfinite(synthetic_params_true).all():
        raise ValueError("Every has_params=True sample must provide finite params_true")
    synthetic_matrices = synthetic_full_correction_matrices(
        params[synthetic_items], synthetic_params_true
    )
    matrices = predicted_matrices.clone()
    matrices[synthetic_items] = synthetic_matrices
    source_mask = synthetic_items.view(batch_size, 1, 1, 1, 1)
    sources = torch.where(source_mask, target_group, moving_group)
    return sources, matrices


def warp_group_for_supervision(
    moving_group: torch.Tensor,
    target_group: torch.Tensor,
    params: torch.Tensor,
    params_true: torch.Tensor,
    has_params: torch.Tensor,
) -> torch.Tensor:
    """Correct synthetic sources once and real moving images normally."""
    sources, matrices = supervision_source_and_matrices(
        moving_group,
        target_group,
        params,
        params_true,
        has_params,
    )
    return warp_group_with_matrix(sources, matrices)


def _hsv_to_rgb(image: np.ndarray) -> np.ndarray:
    """Convert a normalized H/S/V image to RGB for a readable overlay."""
    hue = (image[..., 0] % 1.0) * 6.0
    saturation = np.clip(image[..., 1], 0.0, 1.0)
    value = np.clip(image[..., 2], 0.0, 1.0)
    sector = np.floor(hue).astype(np.int64) % 6
    fraction = hue - np.floor(hue)
    p = value * (1.0 - saturation)
    q = value * (1.0 - fraction * saturation)
    t = value * (1.0 - (1.0 - fraction) * saturation)
    candidates = (
        np.stack((value, t, p), axis=-1),
        np.stack((q, value, p), axis=-1),
        np.stack((p, value, t), axis=-1),
        np.stack((p, q, value), axis=-1),
        np.stack((t, p, value), axis=-1),
        np.stack((value, p, q), axis=-1),
    )
    rgb = np.zeros_like(image)
    for index, candidate in enumerate(candidates):
        rgb[sector == index] = candidate[sector == index]
    return rgb


def _to_display_rgb(
    image: torch.Tensor,
    group_id: int,
    sfo_mode: str,
) -> np.ndarray:
    """Convert one normalized model-space tensor to display RGB."""
    display = image.detach().float().clamp(0.0, 1.0).cpu().permute(1, 2, 0).numpy()
    if group_id == 5 and sfo_mode == "hsv":
        display = _hsv_to_rgb(display)
    return np.clip(display, 0.0, 1.0)


def save_group_overlay(
    path: str,
    fixed_mineral: torch.Tensor,
    warped_group: torch.Tensor,
    valid_group: torch.Tensor,
    group_id: int,
    sfo_mode: str,
) -> None:
    """Save the validation-style model-space Mineral/group screen overlay."""
    components = [_to_display_rgb(fixed_mineral, 1, "rgb")]
    for slot, is_valid in enumerate(valid_group.tolist()):
        if is_valid:
            components.append(_to_display_rgb(warped_group[slot], group_id, sfo_mode))
    stacked = np.stack(components, axis=0)
    overlay = 1.0 - np.prod(1.0 - stacked, axis=0)
    pixels = np.rint(np.clip(overlay, 0.0, 1.0) * 255.0).astype(np.uint8)
    Image.fromarray(pixels, mode="RGB").save(path)


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
