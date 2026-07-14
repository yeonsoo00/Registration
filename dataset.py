"""Datasets for stain-group affine registration.

Grouped mode returns one item per sample/group and therefore guarantees that a
single predicted affine matrix is applied to every stain acquired in that group.

Input modes
-----------
``single``
    Backward-compatible one-stain-per-item mode.
``stack``
    Concatenate all stains in a group into fixed channel slots. Missing stains
    are zero-filled and accompanied by binary presence channels.
``overlay``
    Create one RGB/grayscale group overlay by taking the channel-wise maximum
    across the available group stains.
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
    affine_parameters_to_matrix,
    apply_affine_transform,
    apply_preprocess_geometry,
    compute_mineral_mask,
    compute_preprocess_geometry,
    enhance_signal_for_registration,
    invert_affine_matrix,
    PreprocessGeometry,
    load_image,
    normalize_image,
    sample_registration_parameters,
)


STAIN_TO_GROUP = {1: 1, 2: 1, 3: 1, 4: 2, 5: 3, 6: 3, 7: 3, 8: 4, 9: 5}
GROUP_TO_STAINS = {1: (2, 3), 2: (4,), 3: (5, 6, 7), 4: (8,), 5: (9,)}
MAX_GROUP_STAINS = max(len(v) for v in GROUP_TO_STAINS.values())

_STAIN_INDEX_RE = re.compile(r"(?<![FLfl])_([1-9])(?!\d)")


def parse_stain_index(filename: str) -> Optional[int]:
    stem = os.path.splitext(os.path.basename(filename))[0]
    matches = list(_STAIN_INDEX_RE.finditer(stem))
    if matches:
        return int(matches[-1].group(1))
    if stem.isdigit() and 1 <= int(stem) <= 9:
        return int(stem)
    tokens = re.split(r"[_\-\s]+", stem)
    numeric = [int(t) for t in tokens if t.isdigit() and 1 <= int(t) <= 9]
    return numeric[-1] if numeric else None


def _base_id_cartilage(name: str) -> str:
    cut_points = [
        idx for marker in ("_org", "_aligned") if (idx := name.find(marker)) >= 0
    ]
    return name[: min(cut_points)] if cut_points else name


def _find_stains(directory: str) -> Dict[int, str]:
    stains: Dict[int, str] = {}
    if not directory or not os.path.isdir(directory):
        return stains
    for filename in sorted(os.listdir(directory)):
        if not filename.lower().endswith((".png", ".jpg", ".jpeg", ".tif", ".tiff")):
            continue
        idx = parse_stain_index(filename)
        if idx is not None:
            stains[idx] = os.path.join(directory, filename)
    return stains


def _convert_sfo(image_rgb: np.ndarray, mode: str) -> np.ndarray:
    if mode == "rgb":
        return image_rgb
    if mode == "hsv":
        hsv = cv2.cvtColor(
            np.clip(image_rgb * 255.0, 0, 255).astype(np.uint8), cv2.COLOR_RGB2HSV
        ).astype(np.float32)
        hsv[..., 0] /= 179.0
        hsv[..., 1:] /= 255.0
        return hsv
    if mode == "gray":
        gray = (
            cv2.cvtColor(
                np.clip(image_rgb * 255.0, 0, 255).astype(np.uint8), cv2.COLOR_RGB2GRAY
            ).astype(np.float32)
            / 255.0
        )
        return np.repeat(gray[..., None], 3, axis=2)
    raise ValueError(f"Unsupported sfo_mode: {mode}")


class CartilageDataset(Dataset):
    """Cartilage registration dataset supporting stain-group items."""

    def __init__(
        self,
        registered_root: str,
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
        group_input_mode: str = "single",
        include_group1: bool = True,
        enhance_groups: Tuple[int, ...] = (),
        enhancement_strength: float = 0.7,
        enhancement_mode: str = "legacy",
        trap_enhancement_strength: float = 0.8,
        cfo_enhancement_strength: float = 0.5,
        sfo_enhancement_strength: float = 1.0,
        sfo_hue_range_degrees: Tuple[float, float] = (70.0, 200.0),
        trap_morphology_kernel_size: int = 3,
        trap_morphology_iterations: int = 1,
    ) -> None:
        super().__init__()
        if image_mode not in {"rgb", "gray"}:
            raise ValueError("image_mode must be 'rgb' or 'gray'")
        if sfo_mode not in {"rgb", "hsv", "gray"}:
            raise ValueError("sfo_mode must be rgb, hsv, or gray")
        if group_input_mode not in {"single", "stack", "overlay"}:
            raise ValueError("group_input_mode must be single, stack, or overlay")
        if not 0.0 <= synthetic_prob <= 1.0:
            raise ValueError("synthetic_prob must be in [0,1]")

        self.registered_root = registered_root
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
        self.group_input_mode = group_input_mode
        self.include_group1 = include_group1
        self.enhance_groups = tuple(sorted(set(map(int, enhance_groups))))
        self.enhancement_strength = float(enhancement_strength)
        self.enhancement_mode = enhancement_mode
        self.group_enhancement_strengths = {
            2: float(trap_enhancement_strength),
            4: float(cfo_enhancement_strength),
            5: float(sfo_enhancement_strength),
        }
        self.sfo_hue_range_degrees = tuple(map(float, sfo_hue_range_degrees))
        self.trap_morphology_kernel_size = int(trap_morphology_kernel_size)
        self.trap_morphology_iterations = int(trap_morphology_iterations)
        if any(group < 1 or group > 5 for group in self.enhance_groups):
            raise ValueError("enhance_groups must contain group IDs from 1 to 5")
        if self.image_mode != "rgb" and self.enhance_groups:
            raise ValueError("Group enhancement requires image_mode=rgb")
        if self.enhancement_mode not in {"legacy", "group_specific"}:
            raise ValueError("enhancement_mode must be legacy or group_specific")
        strengths = [
            self.enhancement_strength,
            *self.group_enhancement_strengths.values(),
        ]
        if any(strength < 0.0 or strength > 1.0 for strength in strengths):
            raise ValueError("enhancement strengths must be in [0, 1]")
        if len(self.sfo_hue_range_degrees) != 2:
            raise ValueError("sfo_hue_range_degrees must contain two values")
        hue_low, hue_high = self.sfo_hue_range_degrees
        if not 0.0 <= hue_low < hue_high <= 360.0:
            raise ValueError("SFO hue range must satisfy 0 <= low < high <= 360")
        if (
            self.trap_morphology_kernel_size < 1
            or self.trap_morphology_kernel_size % 2 == 0
        ):
            raise ValueError("TRAP morphology kernel must be a positive odd integer")
        if self.trap_morphology_iterations < 1:
            raise ValueError("TRAP morphology iterations must be positive")
        if (
            self.enhancement_mode == "group_specific"
            and 5 in self.enhance_groups
            and self.sfo_mode not in {"rgb", "hsv"}
        ):
            raise ValueError(
                "Group-specific SFO selection requires sfo_mode=rgb or hsv"
            )
        # Cache resized tensors, not multi-megapixel source arrays. Persistent
        # DataLoader workers then pay the large PNG decode cost only once.
        self._canvas_cache: Dict[
            Tuple[str, int, PreprocessGeometry, bool, int], torch.Tensor
        ] = {}
        self._synthetic_raw_cache: Dict[
            Tuple[str, int, PreprocessGeometry], torch.Tensor
        ] = {}
        self._fixed_cache: Dict[str, Tuple[torch.Tensor, PreprocessGeometry]] = {}

        if group_input_mode == "stack":
            self.group_input_channels = (
                MAX_GROUP_STAINS * self.channels + MAX_GROUP_STAINS
            )
        else:
            self.group_input_channels = self.channels

        reg_map = {
            _base_id_cartilage(name): name
            for name in os.listdir(registered_root)
            if os.path.isdir(os.path.join(registered_root, name))
        }
        unreg_map: Dict[str, str] = {}
        for name in os.listdir(unregistered_root):
            path = os.path.join(unregistered_root, name)
            if not os.path.isdir(path):
                continue
            base = _base_id_cartilage(name)
            if base not in unreg_map or "ano" in unreg_map[base].lower():
                unreg_map[base] = name

        self.samples: Dict[str, Dict[str, object]] = {}
        self.items: List[Tuple[str, int, Optional[int]]] = []

        for base_id, reg_name in sorted(reg_map.items()):
            reg_stains = _find_stains(os.path.join(registered_root, reg_name))
            if 1 not in reg_stains:
                continue
            unreg_name = unreg_map.get(base_id)
            unreg_stains = _find_stains(
                os.path.join(unregistered_root, unreg_name) if unreg_name else ""
            )
            self.samples[base_id] = {
                "reg_stains": reg_stains,
                "unreg_stains": unreg_stains,
                "mineral": reg_stains[1],
            }

            if group_input_mode == "single":
                for stain_idx in range(2, 10):
                    if stain_idx in reg_stains and (
                        stain_idx in unreg_stains or synthetic_prob >= 1.0
                    ):
                        self.items.append(
                            (base_id, STAIN_TO_GROUP[stain_idx], stain_idx)
                        )
            else:
                for group_id, stain_indices in GROUP_TO_STAINS.items():
                    if group_id == 1 and not include_group1:
                        continue
                    available_target = [s for s in stain_indices if s in reg_stains]
                    available_moving = [
                        s for s in available_target if s in unreg_stains
                    ]
                    if not available_target:
                        continue
                    if synthetic_prob >= 1.0 or available_moving:
                        self.items.append((base_id, group_id, None))

        if not self.items:
            raise RuntimeError("No usable registration items were found")

    def _rng(self, idx: int) -> np.random.Generator:
        if self.deterministic_synthetic:
            return np.random.default_rng(self.synthetic_seed + idx * 1009)
        return np.random.default_rng()

    def _load_signal(self, path: str, stain_idx: int) -> np.ndarray:
        if self.image_mode == "gray":
            return load_image(path, grayscale=True)
        return load_image(path, grayscale=False)

    def _to_tensor(self, image: np.ndarray) -> torch.Tensor:
        if image.ndim == 2:
            return torch.from_numpy(image).unsqueeze(0).float()
        return torch.from_numpy(image).permute(2, 0, 1).contiguous().float()

    def _signal_mode(self, stain_idx: int) -> str:
        if self.image_mode == "rgb" and stain_idx == self.sfo_idx:
            return self.sfo_mode
        return "rgb"

    def _model_canvas(
        self,
        image: np.ndarray,
        geometry: PreprocessGeometry,
        signal_mode: str,
    ) -> np.ndarray:
        # Resize in RGB before converting SFO to HSV. Interpolating hue itself
        # can create false cyan/green pixels because hue is circular.
        canvas = apply_preprocess_geometry(image, geometry)
        if self.image_mode == "rgb" and signal_mode != "rgb":
            canvas = _convert_sfo(canvas, signal_mode)
        return canvas

    def _enhancement_config(self, group_id: int) -> Tuple[str, float]:
        method = "contrast_edges"
        strength = self.enhancement_strength
        if self.enhancement_mode == "group_specific":
            method = {
                2: "trap_morphology",
                4: "contrast_edges",
                5: "sfo_hue_selection",
            }.get(group_id, "contrast_edges")
            strength = self.group_enhancement_strengths.get(
                group_id, self.enhancement_strength
            )
        return method, strength

    def _normalize_model_canvas(
        self,
        canvas: np.ndarray,
        *,
        enhance: bool,
        signal_mode: str,
        group_id: int,
    ) -> torch.Tensor:
        if enhance and self.image_mode == "rgb":
            method, strength = self._enhancement_config(group_id)
            canvas = enhance_signal_for_registration(
                canvas,
                color_space="hsv" if signal_mode == "hsv" else "rgb",
                strength=strength,
                method=method,
                sfo_hue_range_degrees=self.sfo_hue_range_degrees,
                morphology_kernel_size=self.trap_morphology_kernel_size,
                morphology_iterations=self.trap_morphology_iterations,
            )
        return self._to_tensor(normalize_image(canvas, per_channel=True))

    def _prepare_canvas(
        self,
        image: np.ndarray,
        geometry: PreprocessGeometry,
        *,
        enhance: bool = False,
        signal_mode: str = "rgb",
        group_id: int = 0,
    ) -> torch.Tensor:
        canvas = self._model_canvas(image, geometry, signal_mode)
        return self._normalize_model_canvas(
            canvas,
            enhance=enhance,
            signal_mode=signal_mode,
            group_id=group_id,
        )

    def _prepare_raw_synthetic_path(
        self,
        path: str,
        stain_idx: int,
        geometry: PreprocessGeometry,
    ) -> torch.Tensor:
        key = (path, stain_idx, geometry)
        cached = self._synthetic_raw_cache.get(key)
        if cached is None:
            canvas = apply_preprocess_geometry(
                self._load_signal(path, stain_idx), geometry
            )
            cached = self._to_tensor(canvas).contiguous()
            self._synthetic_raw_cache[key] = cached
        return cached.clone()

    def _prepare_path(
        self,
        path: str,
        stain_idx: int,
        geometry: PreprocessGeometry,
        *,
        enhance: bool = False,
        group_id: int = 0,
    ) -> torch.Tensor:
        key = (path, stain_idx, geometry, enhance, group_id if enhance else 0)
        cached = self._canvas_cache.get(key)
        if cached is None:
            signal_mode = self._signal_mode(stain_idx)
            cached = self._prepare_canvas(
                self._load_signal(path, stain_idx),
                geometry,
                enhance=enhance,
                signal_mode=signal_mode,
                group_id=group_id,
            ).contiguous()
            self._canvas_cache[key] = cached
        return cached.clone()

    def _prepare_fixed(self, base_id: str, mineral_path: str):
        cached = self._fixed_cache.get(base_id)
        if cached is None:
            fixed_gray = load_image(mineral_path, grayscale=True)
            mineral_mask = compute_mineral_mask(fixed_gray)
            geometry = compute_preprocess_geometry(
                mineral_mask,
                self.size,
                crop_mode=self.crop_mode,
                crop_margin=self.crop_margin,
            )
            fixed_tensor = self._prepare_path(mineral_path, 1, geometry)
            cached = (fixed_tensor, geometry)
            self._fixed_cache[base_id] = cached
        return cached[0].clone(), cached[1]

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        base_id, group_id, single_stain = self.items[idx]
        sample = self.samples[base_id]
        reg_stains: Dict[int, str] = sample["reg_stains"]  # type: ignore[assignment]
        unreg_stains: Dict[int, str] = sample["unreg_stains"]  # type: ignore[assignment]
        mineral_path = str(sample["mineral"])

        fixed_tensor, geometry = self._prepare_fixed(base_id, mineral_path)

        rng = self._rng(idx)
        synthetic = bool(rng.random() < self.synthetic_prob)

        stain_indices = (
            (single_stain,) if single_stain is not None else GROUP_TO_STAINS[group_id]
        )
        moving_list: List[torch.Tensor] = []
        input_list: List[torch.Tensor] = []
        target_list: List[torch.Tensor] = []
        valid_list: List[bool] = []
        output_stain_indices: List[int] = []

        for stain_idx in stain_indices:
            target_path = reg_stains.get(stain_idx)
            moving_path = unreg_stains.get(stain_idx)
            if target_path is None:
                target = torch.zeros((self.channels, *self.size), dtype=torch.float32)
                moving = target.clone()
                valid = False
            else:
                target = self._prepare_path(target_path, stain_idx, geometry)
                if synthetic:
                    moving = target.clone()
                    valid = True
                elif moving_path is not None:
                    moving = self._prepare_path(moving_path, stain_idx, geometry)
                    valid = True
                else:
                    moving = torch.zeros_like(target)
                    valid = False
            if valid:
                input_path = target_path if synthetic else moving_path
                assert input_path is not None
                model_input = self._prepare_path(
                    input_path,
                    stain_idx,
                    geometry,
                    enhance=(group_id in self.enhance_groups and not synthetic),
                    group_id=group_id,
                )
            else:
                model_input = torch.zeros_like(target)
            target_list.append(target)
            moving_list.append(moving)
            input_list.append(model_input)
            valid_list.append(valid)
            output_stain_indices.append(stain_idx)

        params_true = torch.tensor([0.0, 0.0, 0.0, 1.0, 1.0], dtype=torch.float32)
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
            for i, valid in enumerate(valid_list):
                if not valid:
                    continue
                stain_idx = int(stain_indices[i])
                target_path = reg_stains.get(stain_idx)
                assert target_path is not None
                raw_input = self._prepare_raw_synthetic_path(
                    target_path, stain_idx, geometry
                )
                warped_raw = apply_affine_transform(
                    raw_input.unsqueeze(0),
                    synthesis_matrix,
                    padding_mode="zeros",
                ).squeeze(0)
                warped_canvas = warped_raw.permute(1, 2, 0).contiguous().numpy()
                signal_mode = self._signal_mode(stain_idx)
                if self.image_mode == "rgb" and signal_mode != "rgb":
                    warped_canvas = _convert_sfo(warped_canvas, signal_mode)
                moving_list[i] = self._normalize_model_canvas(
                    warped_canvas,
                    enhance=False,
                    signal_mode=signal_mode,
                    group_id=group_id,
                )
                if group_id in self.enhance_groups:
                    input_list[i] = self._normalize_model_canvas(
                        warped_canvas,
                        enhance=True,
                        signal_mode=signal_mode,
                        group_id=group_id,
                    )
                else:
                    input_list[i] = moving_list[i].clone()
            has_params = True

        # Pad every group to a fixed number of stain slots. Groups contain
        # different numbers of stains (for example, G2 has one stain while G3
        # has three), and PyTorch's default DataLoader collate function cannot
        # batch variable-length tensors. Keeping a fixed slot dimension also
        # makes the downstream loss and inference code deterministic.
        while len(moving_list) < MAX_GROUP_STAINS:
            moving_list.append(
                torch.zeros((self.channels, *self.size), dtype=torch.float32)
            )
            input_list.append(
                torch.zeros((self.channels, *self.size), dtype=torch.float32)
            )
            target_list.append(
                torch.zeros((self.channels, *self.size), dtype=torch.float32)
            )
            valid_list.append(False)
            # Zero denotes a padded slot, not a real stain index.
            output_stain_indices.append(0)

        moving_group = torch.stack(moving_list[:MAX_GROUP_STAINS], dim=0).contiguous()
        input_group = torch.stack(input_list[:MAX_GROUP_STAINS], dim=0).contiguous()
        target_group = torch.stack(target_list[:MAX_GROUP_STAINS], dim=0).contiguous()
        valid_group = torch.tensor(valid_list[:MAX_GROUP_STAINS], dtype=torch.bool)
        stain_indices_tensor = torch.tensor(
            output_stain_indices[:MAX_GROUP_STAINS], dtype=torch.long
        )

        if self.group_input_mode == "single":
            group_input = input_group[0]
        elif self.group_input_mode == "overlay":
            if valid_group.any():
                group_input = input_group[valid_group].amax(dim=0)
            else:
                group_input = torch.zeros_like(input_group[0])
        else:  # stack
            slots: List[torch.Tensor] = []
            presence_maps: List[torch.Tensor] = []
            for slot in range(MAX_GROUP_STAINS):
                if bool(valid_group[slot]):
                    slots.append(input_group[slot])
                    presence = torch.ones((1, *self.size), dtype=torch.float32)
                else:
                    slots.append(
                        torch.zeros((self.channels, *self.size), dtype=torch.float32)
                    )
                    presence = torch.zeros((1, *self.size), dtype=torch.float32)
                presence_maps.append(presence)
            group_input = torch.cat(slots + presence_maps, dim=0)

        return {
            # Clone returned tensors so their storage is owned by PyTorch.
            # This avoids shared-memory collation failures with tensors backed
            # by NumPy/PIL buffers in multi-worker DataLoaders.
            "fixed_mineral": fixed_tensor.contiguous().clone(),
            "group_input": group_input.contiguous().clone(),
            "moving_group": moving_group.clone(),
            "target_group": target_group.clone(),
            "valid_group": valid_group,
            "stain_indices": stain_indices_tensor,
            "group": torch.tensor(group_id, dtype=torch.long),
            "params_true": params_true,
            "has_params": torch.tensor(has_params, dtype=torch.bool),
            "sample_index": torch.tensor(idx, dtype=torch.long),
        }
