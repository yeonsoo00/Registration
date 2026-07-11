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
    invert_affine_matrix,
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
    cut_points = [idx for marker in ("_org", "_aligned") if (idx := name.find(marker)) >= 0]
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
        gray = cv2.cvtColor(
            np.clip(image_rgb * 255.0, 0, 255).astype(np.uint8), cv2.COLOR_RGB2GRAY
        ).astype(np.float32) / 255.0
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
    ) -> None:
        super().__init__()
        if image_mode not in {"rgb", "gray"}:
            raise ValueError("image_mode must be 'rgb' or 'gray'")
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

        if group_input_mode == "stack":
            self.group_input_channels = MAX_GROUP_STAINS * self.channels + MAX_GROUP_STAINS
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
                        self.items.append((base_id, STAIN_TO_GROUP[stain_idx], stain_idx))
            else:
                for group_id, stain_indices in GROUP_TO_STAINS.items():
                    if group_id == 1 and not include_group1:
                        continue
                    available_target = [s for s in stain_indices if s in reg_stains]
                    available_moving = [s for s in available_target if s in unreg_stains]
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
        rgb = load_image(path, grayscale=False)
        return _convert_sfo(rgb, self.sfo_mode) if stain_idx == self.sfo_idx else rgb

    def _to_tensor(self, image: np.ndarray) -> torch.Tensor:
        if image.ndim == 2:
            return torch.from_numpy(image).unsqueeze(0).float()
        return torch.from_numpy(image).permute(2, 0, 1).contiguous().float()

    def _prepare_canvas(self, image: np.ndarray, geometry) -> torch.Tensor:
        arr = apply_preprocess_geometry(image, geometry)
        arr = normalize_image(arr, per_channel=True)
        return self._to_tensor(arr)

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        base_id, group_id, single_stain = self.items[idx]
        sample = self.samples[base_id]
        reg_stains: Dict[int, str] = sample["reg_stains"]  # type: ignore[assignment]
        unreg_stains: Dict[int, str] = sample["unreg_stains"]  # type: ignore[assignment]
        mineral_path = str(sample["mineral"])

        fixed_gray = load_image(mineral_path, grayscale=True)
        fixed_signal = load_image(mineral_path, grayscale=self.image_mode == "gray")
        mineral_mask = compute_mineral_mask(fixed_gray)
        geometry = compute_preprocess_geometry(
            mineral_mask, self.size, crop_mode=self.crop_mode, crop_margin=self.crop_margin
        )
        fixed_tensor = self._prepare_canvas(fixed_signal, geometry)

        rng = self._rng(idx)
        synthetic = bool(rng.random() < self.synthetic_prob)

        stain_indices = (single_stain,) if single_stain is not None else GROUP_TO_STAINS[group_id]
        moving_list: List[torch.Tensor] = []
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
                target = self._prepare_canvas(self._load_signal(target_path, stain_idx), geometry)
                if synthetic:
                    moving = target.clone()
                    valid = True
                elif moving_path is not None:
                    moving = self._prepare_canvas(self._load_signal(moving_path, stain_idx), geometry)
                    valid = True
                else:
                    moving = torch.zeros_like(target)
                    valid = False
            target_list.append(target)
            moving_list.append(moving)
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
                if valid:
                    moving_list[i] = apply_affine_transform(
                        target_list[i].unsqueeze(0), synthesis_matrix, padding_mode="zeros"
                    ).squeeze(0)
            has_params = True

        # Pad every group to a fixed number of stain slots. Groups contain
        # different numbers of stains (for example, G2 has one stain while G3
        # has three), and PyTorch's default DataLoader collate function cannot
        # batch variable-length tensors. Keeping a fixed slot dimension also
        # makes the downstream loss and inference code deterministic.
        while len(moving_list) < MAX_GROUP_STAINS:
            moving_list.append(torch.zeros((self.channels, *self.size), dtype=torch.float32))
            target_list.append(torch.zeros((self.channels, *self.size), dtype=torch.float32))
            valid_list.append(False)
            # Zero denotes a padded slot, not a real stain index.
            output_stain_indices.append(0)

        moving_group = torch.stack(moving_list[:MAX_GROUP_STAINS], dim=0).contiguous()
        target_group = torch.stack(target_list[:MAX_GROUP_STAINS], dim=0).contiguous()
        valid_group = torch.tensor(valid_list[:MAX_GROUP_STAINS], dtype=torch.bool)
        stain_indices_tensor = torch.tensor(
            output_stain_indices[:MAX_GROUP_STAINS], dtype=torch.long
        )

        if self.group_input_mode == "single":
            group_input = moving_group[0]
        elif self.group_input_mode == "overlay":
            if valid_group.any():
                group_input = moving_group[valid_group].amax(dim=0)
            else:
                group_input = torch.zeros_like(moving_group[0])
        else:  # stack
            slots: List[torch.Tensor] = []
            presence_maps: List[torch.Tensor] = []
            for slot in range(MAX_GROUP_STAINS):
                if bool(valid_group[slot]):
                    slots.append(moving_group[slot])
                    presence = torch.ones((1, *self.size), dtype=torch.float32)
                else:
                    slots.append(torch.zeros((self.channels, *self.size), dtype=torch.float32))
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
