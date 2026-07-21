"""Dataset for deployable stain-group affine registration.

Each item contains a fixed Mineral image, one padded stack of unregistered moving
stains, and (when explicitly required) the matching registered stain stack used
only as supervision. Missing group members are zero-filled. Consequently one
model prediction can be shared by every valid stain acquired in that group.

Images remain raw float32 values in [0, 1] after the common geometric
preprocessing. Structural conversion belongs to the model, so prediction
never depends on a registered target-group image.
"""

from __future__ import annotations

import os
import re
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset

from utils import (
    PreprocessGeometry,
    affine_parameters_to_matrix,
    apply_affine_transform,
    apply_preprocess_geometry,
    compute_mineral_mask,
    compute_preprocess_geometry,
    invert_affine_matrix,
    load_image,
    sample_registration_parameters,
)


STAIN_TO_GROUP = {1: 1, 2: 1, 3: 1, 4: 2, 5: 3, 6: 3, 7: 3, 8: 4, 9: 5}
GROUP_TO_STAINS = {1: (2, 3), 2: (4,), 3: (5, 6, 7), 4: (8,), 5: (9,)}
MAX_GROUP_STAINS = max(len(stains) for stains in GROUP_TO_STAINS.values())

_STAIN_INDEX_RE = re.compile(r"(?<![FLfl])_([1-9])(?!\d)")


def parse_stain_index(filename: str) -> Optional[int]:
    """Return the final stain index encoded in a filename."""
    stem = os.path.splitext(os.path.basename(filename))[0]
    matches = list(_STAIN_INDEX_RE.finditer(stem))
    if matches:
        return int(matches[-1].group(1))
    if stem.isdigit() and 1 <= int(stem) <= 9:
        return int(stem)
    tokens = re.split(r"[_\-\s]+", stem)
    numeric = [
        int(token) for token in tokens if token.isdigit() and 1 <= int(token) <= 9
    ]
    return numeric[-1] if numeric else None


def _base_id_cartilage(name: str) -> str:
    cut_points = [
        index for marker in ("_org", "_aligned") if (index := name.find(marker)) >= 0
    ]
    return name[: min(cut_points)] if cut_points else name


def _find_stains(directory: str) -> Dict[int, str]:
    stains: Dict[int, str] = {}
    if not directory or not os.path.isdir(directory):
        return stains
    for filename in sorted(os.listdir(directory)):
        if not filename.lower().endswith((".png", ".jpg", ".jpeg", ".tif", ".tiff")):
            continue
        stain_index = parse_stain_index(filename)
        if stain_index is not None:
            stains[stain_index] = os.path.join(directory, filename)
    return stains


def _convert_sfo(image_rgb: np.ndarray, mode: str) -> np.ndarray:
    """Convert an RGB SFO canvas without changing its numeric range."""
    if mode == "rgb":
        return image_rgb
    if mode == "hsv":
        hsv = cv2.cvtColor(
            np.clip(image_rgb * 255.0, 0, 255).astype(np.uint8),
            cv2.COLOR_RGB2HSV,
        ).astype(np.float32)
        hsv[..., 0] /= 179.0
        hsv[..., 1:] /= 255.0
        return hsv
    if mode == "gray":
        gray = (
            cv2.cvtColor(
                np.clip(image_rgb * 255.0, 0, 255).astype(np.uint8),
                cv2.COLOR_RGB2GRAY,
            ).astype(np.float32)
            / 255.0
        )
        return np.repeat(gray[..., None], 3, axis=2)
    raise ValueError(f"Unsupported sfo_mode: {mode}")


class CartilageDataset(Dataset):
    """Return one fixed-size stack for each sample/acquisition group.

    When require_registered_targets is true, only matching moving/target stain
    pairs are valid for real samples. Synthetic samples use a registered target
    as their source and therefore do not require an unregistered file. When it
    is false, item discovery and validity depend only on fixed Mineral and
    unregistered moving stains; target slots are returned as zeros and
    target-group files are never decoded. fixed_mineral_root defaults to
    registered_root for training compatibility, but inference may point it at
    the deployment input root independently of optional evaluation targets.
    """

    def __init__(
        self,
        registered_root: Optional[str],
        unregistered_root: str,
        size: Tuple[int, int] = (512, 512),
        image_mode: str = "rgb",
        sfo_idx: int = 9,
        sfo_mode: str = "rgb",
        crop_mode: str = "full",
        crop_margin: int = 32,
        synthetic_prob: float = 0.0,
        tx_range: Tuple[float, float] = (-32.0, 32.0),
        ty_range: Tuple[float, float] = (-32.0, 32.0),
        rot_range: Tuple[float, float] = (-10.0, 10.0),
        scale_range: Tuple[float, float] = (0.9, 1.1),
        deterministic_synthetic: bool = False,
        synthetic_seed: int = 1234,
        include_group1: bool = True,
        require_registered_targets: bool = True,
        fixed_mineral_root: Optional[str] = None,
    ) -> None:
        super().__init__()
        if image_mode not in {"rgb", "gray"}:
            raise ValueError("image_mode must be 'rgb' or 'gray'")
        if sfo_mode not in {"rgb", "hsv", "gray"}:
            raise ValueError("sfo_mode must be rgb, hsv, or gray")
        if crop_mode not in {"full", "mineral_bbox"}:
            raise ValueError("crop_mode must be full or mineral_bbox")
        if not 0.0 <= synthetic_prob <= 1.0:
            raise ValueError("synthetic_prob must be in [0, 1]")
        if synthetic_prob > 0.0 and not require_registered_targets:
            raise ValueError(
                "Synthetic samples require registered targets; set "
                "require_registered_targets=True"
            )
        if len(size) != 2 or any(int(value) < 1 for value in size):
            raise ValueError("size must contain two positive integers")
        if require_registered_targets and not registered_root:
            raise ValueError(
                "registered_root is required when registered targets are requested"
            )
        if registered_root and not os.path.isdir(registered_root):
            raise FileNotFoundError(
                f"Registered target root does not exist: {registered_root}"
            )
        resolved_fixed_mineral_root = fixed_mineral_root or registered_root
        if not resolved_fixed_mineral_root:
            raise ValueError(
                "fixed_mineral_root is required when registered_root is omitted"
            )
        if not os.path.isdir(resolved_fixed_mineral_root):
            raise FileNotFoundError(
                "Fixed Mineral root does not exist: " f"{resolved_fixed_mineral_root}"
            )

        self.registered_root = registered_root
        self.fixed_mineral_root = resolved_fixed_mineral_root
        self.unregistered_root = unregistered_root
        self.size = tuple(map(int, size))
        self.image_mode = image_mode
        self.channels = 3 if image_mode == "rgb" else 1
        self.sfo_idx = int(sfo_idx)
        self.sfo_mode = sfo_mode
        self.crop_mode = crop_mode
        self.crop_margin = int(crop_margin)
        self.synthetic_prob = float(synthetic_prob)
        self.tx_range = tuple(map(float, tx_range))
        self.ty_range = tuple(map(float, ty_range))
        self.rot_range = tuple(map(float, rot_range))
        self.scale_range = tuple(map(float, scale_range))
        self.deterministic_synthetic = bool(deterministic_synthetic)
        self.synthetic_seed = int(synthetic_seed)
        self.include_group1 = bool(include_group1)
        self.require_registered_targets = bool(require_registered_targets)

        # Cache only resized tensors. Persistent workers then avoid repeatedly
        # decoding large source images without retaining full source arrays.
        self._canvas_cache: Dict[Tuple[str, int, PreprocessGeometry], torch.Tensor] = {}
        self._synthetic_raw_cache: Dict[
            Tuple[str, int, PreprocessGeometry], torch.Tensor
        ] = {}
        self._fixed_cache: Dict[str, Tuple[torch.Tensor, PreprocessGeometry]] = {}

        registered_directories: Dict[str, str] = {}
        if registered_root:
            registered_directories = {
                _base_id_cartilage(name): name
                for name in os.listdir(registered_root)
                if os.path.isdir(os.path.join(registered_root, name))
            }
        fixed_mineral_directories = {
            _base_id_cartilage(name): name
            for name in os.listdir(self.fixed_mineral_root)
            if os.path.isdir(os.path.join(self.fixed_mineral_root, name))
        }
        unregistered_directories: Dict[str, str] = {}
        if os.path.isdir(unregistered_root):
            for name in os.listdir(unregistered_root):
                path = os.path.join(unregistered_root, name)
                if not os.path.isdir(path):
                    continue
                base_id = _base_id_cartilage(name)
                if (
                    base_id not in unregistered_directories
                    or "ano" in unregistered_directories[base_id].lower()
                ):
                    unregistered_directories[base_id] = name

        self.samples: Dict[str, Dict[str, object]] = {}
        # Keep the historical three-field item shape because inference uses
        # this metadata to reconstruct original-resolution output paths.
        self.items: List[Tuple[str, int, None]] = []
        missing_target_pairs: List[str] = []

        for base_id, fixed_mineral_name in sorted(fixed_mineral_directories.items()):
            fixed_mineral_stains = _find_stains(
                os.path.join(self.fixed_mineral_root, fixed_mineral_name)
            )
            mineral_path = fixed_mineral_stains.get(1)
            if mineral_path is None:
                continue

            registered_name = registered_directories.get(base_id)
            all_registered_stains = _find_stains(
                os.path.join(registered_root, registered_name)
                if registered_root and registered_name
                else ""
            )

            unregistered_name = unregistered_directories.get(base_id)
            unregistered_stains = _find_stains(
                os.path.join(unregistered_root, unregistered_name)
                if unregistered_name
                else ""
            )
            registered_stains = (
                all_registered_stains
                if self.require_registered_targets
                else {1: mineral_path}
            )
            self.samples[base_id] = {
                "reg_stains": registered_stains,
                "unreg_stains": unregistered_stains,
                "mineral": mineral_path,
            }

            for group_id, stain_indices in GROUP_TO_STAINS.items():
                if group_id == 1 and not self.include_group1:
                    continue
                if self.require_registered_targets and self.synthetic_prob >= 1.0:
                    available_targets = [
                        stain for stain in stain_indices if stain in registered_stains
                    ]
                    usable = available_targets
                else:
                    usable = [
                        stain for stain in stain_indices if stain in unregistered_stains
                    ]
                    if self.require_registered_targets:
                        missing_target_pairs.extend(
                            f"{base_id}/G{group_id}/stain{stain}"
                            for stain in usable
                            if stain not in registered_stains
                        )
                if usable:
                    self.items.append((base_id, group_id, None))

        if missing_target_pairs:
            preview = ", ".join(missing_target_pairs[:20])
            if len(missing_target_pairs) > 20:
                preview += f", ... ({len(missing_target_pairs)} total)"
            raise RuntimeError(
                "Registered supervision targets are missing for unregistered "
                f"moving stains: {preview}"
            )
        if not self.items:
            target_note = (
                " with matching registered targets"
                if self.require_registered_targets
                else ""
            )
            raise RuntimeError(
                "No usable grouped registration items were found from fixed "
                f"Mineral and moving stains{target_note}"
            )

    def _rng(self, index: int) -> np.random.Generator:
        if self.deterministic_synthetic:
            return np.random.default_rng(self.synthetic_seed + index * 1009)
        return np.random.default_rng()

    def _load_signal(self, path: str, stain_index: int) -> np.ndarray:
        del stain_index
        return load_image(path, grayscale=self.image_mode == "gray")

    @staticmethod
    def _to_tensor(image: np.ndarray) -> torch.Tensor:
        image = np.clip(image, 0.0, 1.0).astype(np.float32, copy=False)
        if image.ndim == 2:
            return torch.from_numpy(image.copy()).unsqueeze(0)
        return torch.from_numpy(image.copy()).permute(2, 0, 1).contiguous()

    def _signal_mode(self, stain_index: int) -> str:
        if self.image_mode == "rgb" and stain_index == self.sfo_idx:
            return self.sfo_mode
        return "rgb"

    def _model_canvas(
        self,
        image: np.ndarray,
        geometry: PreprocessGeometry,
        signal_mode: str,
    ) -> np.ndarray:
        # Resize RGB before converting SFO to HSV. Interpolating hue can create
        # false cyan/green pixels because hue is circular.
        canvas = apply_preprocess_geometry(image, geometry)
        if self.image_mode == "rgb" and signal_mode != "rgb":
            canvas = _convert_sfo(canvas, signal_mode)
        return np.clip(canvas, 0.0, 1.0).astype(np.float32, copy=False)

    def _prepare_path(
        self,
        path: str,
        stain_index: int,
        geometry: PreprocessGeometry,
    ) -> torch.Tensor:
        key = (path, stain_index, geometry)
        cached = self._canvas_cache.get(key)
        if cached is None:
            canvas = self._model_canvas(
                self._load_signal(path, stain_index),
                geometry,
                self._signal_mode(stain_index),
            )
            cached = self._to_tensor(canvas).contiguous()
            self._canvas_cache[key] = cached
        return cached.clone()

    def _prepare_raw_synthetic_path(
        self,
        path: str,
        stain_index: int,
        geometry: PreprocessGeometry,
    ) -> torch.Tensor:
        """Prepare RGB/gray target before the synthetic affine and SFO conversion."""
        key = (path, stain_index, geometry)
        cached = self._synthetic_raw_cache.get(key)
        if cached is None:
            canvas = apply_preprocess_geometry(
                self._load_signal(path, stain_index),
                geometry,
            )
            cached = self._to_tensor(canvas).contiguous()
            self._synthetic_raw_cache[key] = cached
        return cached.clone()

    def _prepare_fixed(
        self, base_id: str, mineral_path: str
    ) -> Tuple[torch.Tensor, PreprocessGeometry]:
        cached = self._fixed_cache.get(base_id)
        if cached is None:
            fixed_gray = load_image(mineral_path, grayscale=True)
            geometry = compute_preprocess_geometry(
                compute_mineral_mask(fixed_gray),
                self.size,
                crop_mode=self.crop_mode,
                crop_margin=self.crop_margin,
            )
            fixed_mineral = self._prepare_path(mineral_path, 1, geometry)
            cached = (fixed_mineral.contiguous(), geometry)
            self._fixed_cache[base_id] = cached
        return cached[0].clone(), cached[1]

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, index: int) -> Dict[str, torch.Tensor]:
        base_id, group_id, _ = self.items[index]
        sample = self.samples[base_id]
        registered_stains: Dict[int, str] = sample["reg_stains"]  # type: ignore[assignment]
        unregistered_stains: Dict[int, str] = sample["unreg_stains"]  # type: ignore[assignment]
        fixed_mineral, geometry = self._prepare_fixed(
            base_id,
            str(sample["mineral"]),
        )

        rng = self._rng(index)
        synthetic = bool(rng.random() < self.synthetic_prob)
        moving_list: List[torch.Tensor] = []
        target_list: List[torch.Tensor] = []
        valid_list: List[bool] = []
        output_stain_indices: List[int] = []

        for stain_index in GROUP_TO_STAINS[group_id]:
            target_path = (
                registered_stains.get(stain_index)
                if self.require_registered_targets
                else None
            )
            moving_path = unregistered_stains.get(stain_index)
            if synthetic:
                valid = target_path is not None
            elif self.require_registered_targets:
                if moving_path is not None and target_path is None:
                    raise AssertionError(
                        "Dataset pairing validation missed a registered target"
                    )
                valid = moving_path is not None
            else:
                valid = moving_path is not None

            zero = torch.zeros((self.channels, *self.size), dtype=torch.float32)
            if valid and target_path is not None:
                target = self._prepare_path(target_path, stain_index, geometry)
            else:
                target = zero.clone()

            if synthetic and valid:
                moving = target.clone()
            elif valid and moving_path is not None:
                moving = self._prepare_path(moving_path, stain_index, geometry)
            else:
                moving = zero.clone()

            moving_list.append(moving)
            target_list.append(target)
            valid_list.append(valid)
            output_stain_indices.append(stain_index)

        # Real observations have no affine label. NaN is an explicit undefined
        # sentinel, not an identity target; every consumer must gate this tensor
        # with has_params before using it as parameter supervision.
        params_true = torch.full((5,), float("nan"), dtype=torch.float32)
        has_params = False
        if synthetic:
            params_true = sample_registration_parameters(
                self.tx_range,
                self.ty_range,
                self.rot_range,
                self.scale_range,
                self.size,
                rng=rng,
            )
            registration_matrix = affine_parameters_to_matrix(params_true.unsqueeze(0))
            synthesis_matrix = invert_affine_matrix(registration_matrix)
            for slot, valid in enumerate(valid_list):
                if not valid:
                    continue
                stain_index = GROUP_TO_STAINS[group_id][slot]
                target_path = registered_stains.get(stain_index)
                if target_path is None:
                    raise AssertionError(
                        "Synthetic valid slot has no registered target"
                    )
                raw_target = self._prepare_raw_synthetic_path(
                    target_path,
                    stain_index,
                    geometry,
                )
                warped_raw = apply_affine_transform(
                    raw_target.unsqueeze(0),
                    synthesis_matrix,
                    padding_mode="zeros",
                ).squeeze(0)
                warped_canvas = warped_raw.permute(1, 2, 0).contiguous().cpu().numpy()
                signal_mode = self._signal_mode(stain_index)
                if self.image_mode == "rgb" and signal_mode != "rgb":
                    warped_canvas = _convert_sfo(warped_canvas, signal_mode)
                moving_list[slot] = self._to_tensor(warped_canvas)
            has_params = True

        while len(moving_list) < MAX_GROUP_STAINS:
            moving_list.append(
                torch.zeros((self.channels, *self.size), dtype=torch.float32)
            )
            target_list.append(
                torch.zeros((self.channels, *self.size), dtype=torch.float32)
            )
            valid_list.append(False)
            output_stain_indices.append(0)

        moving_group = torch.stack(
            moving_list[:MAX_GROUP_STAINS],
            dim=0,
        ).contiguous()
        target_group = torch.stack(
            target_list[:MAX_GROUP_STAINS],
            dim=0,
        ).contiguous()
        valid_group = torch.tensor(
            valid_list[:MAX_GROUP_STAINS],
            dtype=torch.bool,
        )
        stain_indices = torch.tensor(
            output_stain_indices[:MAX_GROUP_STAINS],
            dtype=torch.long,
        )

        if not bool(valid_group.any()):
            raise RuntimeError(
                f"Item {base_id}/G{group_id} has no valid group stain slots"
            )
        if (
            not torch.isfinite(fixed_mineral).all()
            or not torch.isfinite(moving_group).all()
        ):
            raise RuntimeError(f"Non-finite model input in {base_id}/G{group_id}")
        if (
            fixed_mineral.min() < 0.0
            or fixed_mineral.max() > 1.0
            or moving_group.min() < 0.0
            or moving_group.max() > 1.0
            or target_group.min() < 0.0
            or target_group.max() > 1.0
        ):
            raise RuntimeError(f"Model input escaped [0, 1] in {base_id}/G{group_id}")

        return {
            "fixed_mineral": fixed_mineral.contiguous().clone(),
            "moving_group": moving_group.clone(),
            "target_group": target_group.clone(),
            "valid_group": valid_group,
            "group_id": torch.tensor(group_id, dtype=torch.long),
            "stain_indices": stain_indices,
            "params_true": params_true,
            "has_params": torch.tensor(has_params, dtype=torch.bool),
        }
