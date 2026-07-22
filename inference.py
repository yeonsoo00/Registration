"""Run deployable student inference with a TeacherStudent checkpoint.

One predicted affine is shared by all valid stains in each group item. See
README.md for the current command and output layout. The training-only teacher
is deliberately never constructed or loaded here.
"""

from __future__ import annotations

import argparse
import csv
import os
import re
from collections import OrderedDict, defaultdict
from typing import Any, Dict, List

import cv2
import numpy as np
from PIL import Image
import torch
from torch.utils.data import DataLoader, Subset

from dataset import MAX_GROUP_STAINS, CartilageDataset
from models import (
    AFFINE_HEAD_MODES,
    DEFAULT_GROUP_SLOTS,
    ENCODER_ARCHES,
    FRONTEND_MODES,
    GROUP_INPUT_MODES,
    CorrelationVolumeAffineRegistrationModel,
    canonicalize_model_config,
)
from utils import (
    STRUCTURAL_CHANNEL_NAMES,
    affine_parameters_to_matrix,
    compute_mineral_mask,
    compute_preprocess_geometry,
    load_image,
    normalized_affine_to_pixel_matrix,
    resolve_device,
    save_group_overlay as _save_group_overlay,
    warp_model_space_group,
)


ARCHITECTURE = "correlation_volume_teacher_student_affine_v1"
STRUCTURAL_DESCRIPTOR_VERSION = "torch_structural_no_enhancement_v3"
STRUCTURAL_CHANNELS = len(STRUCTURAL_CHANNEL_NAMES)
INPUT_CONTRACT_VERSION = "fixed_mineral_moving_group_raw_v1"


def safe_collate(batch: List[Dict[str, Any]]) -> Dict[str, Any]:
    if not batch:
        raise RuntimeError("Cannot collate an empty batch")
    result: Dict[str, Any] = {}
    for key in batch[0].keys():
        values = [sample[key] for sample in batch]
        first = values[0]
        if torch.is_tensor(first):
            shapes = [tuple(value.shape) for value in values]
            if any(shape != shapes[0] for shape in shapes[1:]):
                raise RuntimeError(f"Variable tensor shape for key '{key}': {shapes}")
            result[key] = torch.stack(
                [value.detach().contiguous().clone() for value in values], dim=0
            )
        else:
            result[key] = values
    return result


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", required=True)
    p.add_argument(
        "--registered_root",
        default=None,
        help=(
            "Optional registered target root. Supplying it automatically enables "
            "registered-target MAE/NCC evaluation; it is never a model input."
        ),
    )
    p.add_argument("--unregistered_root", required=True)
    p.add_argument("--output_dir", required=True)
    p.add_argument("--batch_size", type=int, default=4)
    p.add_argument("--n_workers", type=int, default=2)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--gpu_ids", default="0,1")
    p.add_argument("--no_multi_gpu", action="store_true")
    p.add_argument(
        "--include_group1",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Override checkpoint Group 1 inclusion; default reconstructs checkpoint preprocessing.",
    )
    p.add_argument(
        "--max_items",
        type=int,
        default=0,
        help="Smoke-only inference cap; 0 processes every grouped item",
    )
    p.add_argument(
        "--eval_with_registered_targets",
        action=argparse.BooleanOptionalAction,
        default=None,
        help=(
            "Deprecated compatibility alias. Evaluation is enabled automatically "
            "whenever --registered_root is supplied."
        ),
    )
    return p.parse_args()


def _registered_evaluation_enabled(a) -> bool:
    registered_root = getattr(a, "registered_root", None)
    legacy_choice = getattr(a, "eval_with_registered_targets", None)
    if legacy_choice is True and not registered_root:
        raise ValueError(
            "--eval_with_registered_targets requires --registered_root; the flag "
            "is otherwise no longer needed"
        )
    if legacy_choice is False and registered_root:
        raise ValueError(
            "--no-eval_with_registered_targets conflicts with --registered_root; "
            "remove registered_root for target-free inference"
        )
    return bool(registered_root)


def save_rgb(path, img):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    Image.fromarray(img.astype(np.uint8), "RGB").save(path)


def make_overlay(images: List[np.ndarray]) -> np.ndarray:
    if not images:
        raise ValueError("No images for overlay")
    shapes = [image.shape for image in images]
    if any(shape != shapes[0] for shape in shapes[1:]):
        raise ValueError(f"Overlay images must have identical shapes; got {shapes}")
    arr = np.stack([x.astype(np.float32) for x in images], axis=0)
    return np.clip(arr.max(axis=0), 0, 255).astype(np.uint8)


def make_mineral_group_overlay(
    mineral_rgb: np.ndarray, aligned_group_images: List[np.ndarray]
) -> np.ndarray:
    """Overlay original RGB Mineral with every available aligned group signal."""
    if not aligned_group_images:
        raise ValueError("No aligned group images for Mineral overlap")
    return make_overlay([mineral_rgb, *aligned_group_images])


def _rectangular_signal_support(
    images: torch.Tensor, valid_group: torch.Tensor, eps: float = 1e-6
) -> torch.Tensor:
    """Infer non-padded rectangular support for each raw stain slot."""
    if images.ndim != 5 or valid_group.shape != images.shape[:2]:
        raise ValueError("images and valid_group must have shapes BKCHW and BK")
    batch_size, slots, _, height, width = images.shape
    evidence = images.detach().abs().amax(dim=2) > eps
    row_has_signal = evidence.any(dim=3)
    column_has_signal = evidence.any(dim=2)
    has_signal = evidence.flatten(2).any(dim=2) & valid_group.bool()
    y_coordinates = torch.arange(height, device=images.device).view(1, 1, height)
    x_coordinates = torch.arange(width, device=images.device).view(1, 1, width)
    first_y = torch.where(row_has_signal, y_coordinates, height).amin(dim=2)
    last_y = torch.where(row_has_signal, y_coordinates, -1).amax(dim=2)
    first_x = torch.where(column_has_signal, x_coordinates, width).amin(dim=2)
    last_x = torch.where(column_has_signal, x_coordinates, -1).amax(dim=2)
    yy = y_coordinates.unsqueeze(-1)
    xx = x_coordinates.unsqueeze(-2)
    support = (
        (yy >= first_y[..., None, None])
        & (yy <= last_y[..., None, None])
        & (xx >= first_x[..., None, None])
        & (xx <= last_x[..., None, None])
        & has_signal[..., None, None]
    )
    return support.unsqueeze(2).to(dtype=images.dtype)


def registered_target_metrics(
    warped_group: torch.Tensor,
    target_group: torch.Tensor,
    valid_group: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return support-masked per-slot MAE and NCC; invalid slots are NaN."""
    if warped_group.shape != target_group.shape:
        raise ValueError("warped_group and target_group must have identical shapes")
    if valid_group.shape != warped_group.shape[:2]:
        raise ValueError("valid_group must have shape BxK")
    overlap = _rectangular_signal_support(warped_group, valid_group)
    overlap = overlap * _rectangular_signal_support(target_group, valid_group)
    weight = overlap.expand(-1, -1, warped_group.shape[2], -1, -1)
    denominator = weight.sum(dim=(2, 3, 4))
    safe_denominator = denominator.clamp_min(1.0)
    mae = ((warped_group - target_group).abs() * weight).sum(
        dim=(2, 3, 4)
    ) / safe_denominator
    warped_mean = (warped_group * weight).sum(dim=(2, 3, 4)) / safe_denominator
    target_mean = (target_group * weight).sum(dim=(2, 3, 4)) / safe_denominator
    warped_centered = warped_group - warped_mean[:, :, None, None, None]
    target_centered = target_group - target_mean[:, :, None, None, None]
    numerator = (warped_centered * target_centered * weight).sum(dim=(2, 3, 4))
    ncc_denominator = torch.sqrt(
        (warped_centered.square() * weight).sum(dim=(2, 3, 4))
        * (target_centered.square() * weight).sum(dim=(2, 3, 4))
    ).clamp_min(1e-8)
    ncc = numerator / ncc_denominator
    metric_valid = valid_group.bool() & (denominator > 0.0)
    nan = torch.full_like(mae, float("nan"))
    return torch.where(metric_valid, mae, nan), torch.where(metric_valid, ncc, nan)


def load_student_checkpoint(checkpoint_path: str):
    """Load only the deployable student portion of a teacher-student checkpoint.

    Teacher configuration, weights, and optimizer state are intentionally left
    unused. This keeps inference independent of the registered target-group
    branch used during training.
    """
    checkpoint = torch.load(
        checkpoint_path,
        map_location="cpu",
        weights_only=False,
    )
    if not isinstance(checkpoint, dict):
        raise ValueError("TeacherStudent checkpoint must be a dictionary")
    architecture = checkpoint.get("architecture")
    if architecture != ARCHITECTURE:
        raise ValueError(
            "This inference script only accepts TeacherStudent checkpoints with "
            f"architecture={ARCHITECTURE!r}; received {architecture!r}. The "
            "original Correlation_Vol_Net checkpoints cannot be loaded because "
            "their checkpoint and model contracts differ."
        )
    checkpoint_type = checkpoint.get("checkpoint_type")
    if checkpoint_type not in {"deployable_student", "full_training"}:
        raise ValueError(
            "Inference requires either a full training checkpoint or its derived "
            "student-only artifact"
        )
    required_keys = {
        "preprocess_config",
        "student_model_config",
        "student_model_state_dict",
    }
    missing = sorted(required_keys - set(checkpoint))
    if missing:
        raise ValueError(
            "TeacherStudent checkpoint is missing deployable student fields: "
            + ", ".join(missing)
        )
    preprocess_config = dict(checkpoint["preprocess_config"])
    student_config = canonicalize_model_config(checkpoint["student_model_config"])
    student_state = OrderedDict(
        (key[7:] if key.startswith("module.") else key, value)
        for key, value in checkpoint["student_model_state_dict"].items()
    )
    return preprocess_config, student_config, student_state


def predict_model_space_group(model, batch):
    """Run the deployable student and the canonical model-space warp."""
    predicted_params = model(
        fixed_mineral=batch["fixed_mineral"],
        moving_group=batch["moving_group"],
        group=batch["group_id"],
    )
    warped_group = warp_model_space_group(
        batch["moving_group"],
        predicted_params,
    )
    return predicted_params, warped_group


def main(a):
    if a.max_items < 0:
        raise ValueError("max_items cannot be negative")
    evaluate_registered_targets = _registered_evaluation_enabled(a)
    registered_root = getattr(a, "registered_root", None)
    device, gpu_ids = resolve_device(a.device, a.gpu_ids)
    # Deserialize on CPU so teacher and optimizer tensors never occupy inference
    # GPU memory. Only the extracted student state is copied to the model.
    pre, cfg, state = load_student_checkpoint(a.checkpoint)
    required_base_config = {
        "affine_head_mode",
        "encoder_arch",
        "encoder_channels",
        "encoder_blocks_per_stage",
        "correlation_feature_width",
        "input_channels",
        "frontend_mode",
        "group_input_mode",
        "group_slots",
        "force_group1_identity",
    }
    missing_base_config = sorted(required_base_config - set(cfg))
    if missing_base_config:
        raise ValueError(
            "Deployable checkpoint is missing student_model_config fields: "
            + ", ".join(missing_base_config)
        )
    encoder_arch = cfg["encoder_arch"]
    if encoder_arch not in ENCODER_ARCHES:
        raise ValueError(
            f"Unsupported encoder_arch {encoder_arch!r}; expected one of "
            + ", ".join(ENCODER_ARCHES)
        )
    encoder_channels = tuple(cfg["encoder_channels"])
    if len(encoder_channels) < 3 or any(channel < 1 for channel in encoder_channels):
        raise ValueError(
            "Checkpoint encoder_channels must contain positive stage widths"
        )
    encoder_blocks = cfg["encoder_blocks_per_stage"]
    if encoder_arch == "residual":
        if encoder_blocks is None or len(tuple(encoder_blocks)) != len(
            encoder_channels
        ):
            raise ValueError(
                "Residual checkpoint must store one block count per encoder stage"
            )
        if any(block < 1 for block in encoder_blocks):
            raise ValueError("Residual checkpoint block counts must be positive")
    elif encoder_blocks is not None:
        raise ValueError(
            "Current encoder checkpoint must not contain residual block counts"
        )
    if cfg["correlation_feature_width"] < 1:
        raise ValueError("Checkpoint correlation_feature_width must be positive")

    frontend_mode = cfg["frontend_mode"]
    if frontend_mode not in FRONTEND_MODES:
        raise ValueError(
            f"Unsupported frontend_mode {frontend_mode!r}; expected one of "
            f"{', '.join(FRONTEND_MODES)}"
        )
    if cfg["group_input_mode"] not in GROUP_INPUT_MODES:
        raise ValueError(
            f"Unsupported group_input_mode {cfg['group_input_mode']!r}; expected "
            f"one of {', '.join(GROUP_INPUT_MODES)}"
        )
    affine_head_mode = cfg["affine_head_mode"]
    if affine_head_mode not in AFFINE_HEAD_MODES:
        choices = ", ".join(AFFINE_HEAD_MODES)
        raise ValueError(
            f"Unsupported affine_head_mode {affine_head_mode!r}; expected one of "
            f"{choices}"
        )
    if cfg["group_slots"] != MAX_GROUP_STAINS or (
        MAX_GROUP_STAINS != DEFAULT_GROUP_SLOTS
    ):
        raise ValueError(
            "Checkpoint/dataset/model group-slot contract differs: "
            f"checkpoint={cfg['group_slots']}, dataset={MAX_GROUP_STAINS}, "
            f"model_default={DEFAULT_GROUP_SLOTS}"
        )
    if frontend_mode in {"structural", "hybrid"}:
        required_structure_config = {
            "structural_channels",
            "structural_descriptor_version",
            "structural_foreground_threshold",
            "structural_distance_scale",
            "structural_context_scale",
            "structural_skeleton_radius",
        }
        missing_structure_config = sorted(required_structure_config - set(cfg))
        if missing_structure_config:
            raise ValueError(
                "Deployable structural/hybrid checkpoint is missing "
                "student_model_config fields: " + ", ".join(missing_structure_config)
            )
        if cfg["structural_descriptor_version"] != STRUCTURAL_DESCRIPTOR_VERSION:
            raise ValueError(
                "Unsupported structural descriptor version: "
                f"{cfg['structural_descriptor_version']!r}"
            )
        if cfg["structural_channels"] != STRUCTURAL_CHANNELS:
            raise ValueError(
                "Structural checkpoint channel contract differs from this code: "
                f"{cfg['structural_channels']} vs {STRUCTURAL_CHANNELS}"
            )
    if cfg.get("force_group1_identity") is not True:
        raise ValueError(
            "Grouped affine inference requires force_group1_identity=True so AC and "
            "Calcein cannot move"
        )
    if pre.get("input_contract_version") != INPUT_CONTRACT_VERSION:
        raise ValueError(
            "Checkpoint does not use the deployable raw Mineral/moving-group "
            f"input contract {INPUT_CONTRACT_VERSION!r}"
        )
    ds = CartilageDataset(
        registered_root=registered_root,
        unregistered_root=a.unregistered_root,
        size=(pre["height"], pre["width"]),
        image_mode=pre["image_mode"],
        sfo_mode=pre["sfo_mode"],
        crop_mode=pre["crop_mode"],
        crop_margin=pre["crop_margin"],
        synthetic_prob=0.0,
        deterministic_synthetic=True,
        include_group1=(
            pre.get("include_group1", True)
            if a.include_group1 is None
            else a.include_group1
        ),
        require_registered_targets=evaluate_registered_targets,
        fixed_mineral_root=a.unregistered_root,
    )
    if ds.channels != cfg["input_channels"]:
        raise ValueError(
            "Dataset/checkpoint input channel mismatch: "
            f"{ds.channels} vs {cfg['input_channels']}"
        )
    selected_indices = list(range(len(ds)))
    if a.max_items:
        selected_indices = selected_indices[: a.max_items]
    inference_data = Subset(ds, selected_indices)
    dl = DataLoader(
        inference_data,
        batch_size=a.batch_size,
        shuffle=False,
        num_workers=a.n_workers,
        pin_memory=device.type == "cuda",
        collate_fn=safe_collate,
        persistent_workers=a.n_workers > 0,
    )
    model = CorrelationVolumeAffineRegistrationModel(**cfg).to(device)
    model.load_state_dict(state, strict=True)
    del state
    if device.type == "cuda" and not a.no_multi_gpu and len(gpu_ids) > 1:
        model = torch.nn.DataParallel(model, device_ids=gpu_ids)
    model.eval()
    aligned_root = os.path.join(a.output_dir, "aligned_original_rgb")
    overlay_root = os.path.join(a.output_dir, "group_overlays")
    model_space_overlay_root = os.path.join(a.output_dir, "model_space_group_overlays")
    os.makedirs(aligned_root, exist_ok=True)
    os.makedirs(overlay_root, exist_ok=True)
    os.makedirs(model_space_overlay_root, exist_ok=True)
    rows = []
    metric_rows = []
    offset = 0
    grouped_outputs = defaultdict(lambda: defaultdict(list))
    mineral_images: Dict[str, np.ndarray] = {}
    with torch.no_grad():
        for batch in dl:
            for key in (
                "fixed_mineral",
                "moving_group",
                "target_group",
                "valid_group",
                "group_id",
                "params_true",
                "has_params",
            ):
                batch[key] = batch[key].to(device, non_blocking=True)
            if bool(batch["has_params"].any()):
                raise RuntimeError(
                    "Inference must use real moving groups with has_params=False"
                )
            params, warped_model_group = predict_model_space_group(model, batch)
            # Matrix reconstruction below is only for the separate
            # original-resolution OpenCV export. Model-space rendering is
            # complete above and shares no independent affine logic here.
            mats = affine_parameters_to_matrix(params)
            bs = batch["fixed_mineral"].shape[0]
            if evaluate_registered_targets:
                batch_mae, batch_ncc = registered_target_metrics(
                    warped_model_group,
                    batch["target_group"],
                    batch["valid_group"],
                )
                batch_mae = batch_mae.cpu()
                batch_ncc = batch_ncc.cpu()
            for j in range(bs):
                ds_idx = selected_indices[offset + j]
                sample_name, group_id, _ = ds.items[ds_idx]
                predicted_group_id = int(batch["group_id"][j])
                if predicted_group_id != int(group_id):
                    raise AssertionError(
                        "Inference loader order no longer matches dataset items"
                    )
                safe_sample = re.sub(r"[^A-Za-z0-9_.-]+", "_", sample_name)
                model_space_filename = (
                    f"{offset + j:05d}_{safe_sample}_G{predicted_group_id}.png"
                )
                _save_group_overlay(
                    os.path.join(model_space_overlay_root, model_space_filename),
                    batch["fixed_mineral"][j],
                    warped_model_group[j],
                    batch["valid_group"][j],
                    predicted_group_id,
                    pre["sfo_mode"],
                )
                sample = ds.samples[sample_name]
                unreg_stains = sample["unreg_stains"]
                mineral_path = str(sample["mineral"])
                fixed_gray = load_image(mineral_path, grayscale=True)
                if sample_name not in mineral_images:
                    mineral_rgb = load_image(mineral_path, grayscale=False)
                    mineral_images[sample_name] = np.clip(
                        mineral_rgb * 255.0, 0, 255
                    ).astype(np.uint8)
                fixed_h, fixed_w = fixed_gray.shape
                mask = compute_mineral_mask(fixed_gray)
                geom = compute_preprocess_geometry(
                    mask,
                    (pre["height"], pre["width"]),
                    crop_mode=pre["crop_mode"],
                    crop_margin=pre["crop_margin"],
                )
                pre_m = geom.original_to_model_matrix()
                pre_inv = np.linalg.inv(pre_m)
                model_px = normalized_affine_to_pixel_matrix(
                    mats[j], pre["height"], pre["width"]
                )
                stain_indices = batch["stain_indices"][j].tolist()
                valid = batch["valid_group"][j].tolist()
                for slot, (stain_idx, is_valid) in enumerate(zip(stain_indices, valid)):
                    if not is_valid:
                        continue
                    moving_path = unreg_stains.get(int(stain_idx))
                    if moving_path is None:
                        continue
                    moving_rgb = load_image(str(moving_path), grayscale=False)
                    mh, mw = moving_rgb.shape[:2]
                    fixed_canvas_to_moving = np.array(
                        [[mw / fixed_w, 0, 0], [0, mh / fixed_h, 0], [0, 0, 1]],
                        dtype=np.float64,
                    )
                    dst_to_src = fixed_canvas_to_moving @ pre_inv @ model_px @ pre_m
                    aligned = cv2.warpAffine(
                        np.clip(moving_rgb * 255, 0, 255).astype(np.uint8),
                        dst_to_src[:2],
                        (fixed_w, fixed_h),
                        flags=cv2.INTER_LINEAR | cv2.WARP_INVERSE_MAP,
                        borderMode=cv2.BORDER_CONSTANT,
                        borderValue=(0, 0, 0),
                    )
                    out = os.path.join(
                        aligned_root,
                        sample_name,
                        f"group{int(group_id)}_stain{int(stain_idx)}_aligned.png",
                    )
                    save_rgb(out, aligned)
                    grouped_outputs[sample_name][int(group_id)].append(aligned)
                    p = params[j].cpu().numpy()
                    rows.append(
                        {
                            "sample": sample_name,
                            "group": int(group_id),
                            "stain_idx": int(stain_idx),
                            "tx_normalized": float(p[0]),
                            "ty_normalized": float(p[1]),
                            "rotation_degrees": float(np.rad2deg(p[2])),
                            "scale_x": float(p[3]),
                            "scale_y": float(p[4]),
                            "output": out,
                        }
                    )
                    if evaluate_registered_targets:
                        metric_rows.append(
                            {
                                "sample": sample_name,
                                "group": int(group_id),
                                "stain_idx": int(stain_idx),
                                "mae": float(batch_mae[j, slot]),
                                "ncc": float(batch_ncc[j, slot]),
                            }
                        )
            offset += bs
    for sample, groups in grouped_outputs.items():
        for gid, imgs in groups.items():
            save_rgb(
                os.path.join(overlay_root, sample, f"group{gid}_overlay.png"),
                make_overlay(imgs),
            )
            save_rgb(
                os.path.join(
                    overlay_root,
                    sample,
                    f"group{gid}_with_mineral_overlay.png",
                ),
                make_mineral_group_overlay(mineral_images[sample], imgs),
            )
    with open(
        os.path.join(a.output_dir, "predicted_group_affine_parameters.csv"),
        "w",
        newline="",
    ) as f:
        field_names = list(rows[0]) if rows else ["sample"]
        writer = csv.DictWriter(f, fieldnames=field_names)
        writer.writeheader()
        writer.writerows(rows)
    if evaluate_registered_targets:
        metric_path = os.path.join(a.output_dir, "registered_target_metrics.csv")
        with open(metric_path, "w", newline="") as f:
            field_names = list(metric_rows[0]) if metric_rows else ["sample"]
            writer = csv.DictWriter(f, fieldnames=field_names)
            writer.writeheader()
            writer.writerows(metric_rows)
        if metric_rows:
            mean_mae = float(np.mean([row["mae"] for row in metric_rows]))
            mean_ncc = float(np.mean([row["ncc"] for row in metric_rows]))
            print(
                f"Registered-target diagnostics: MAE={mean_mae:.6f}, "
                f"NCC={mean_ncc:.6f} ({len(metric_rows)} stains)"
            )
    print(
        f"Saved {len(rows)} aligned original-resolution stains, {offset} "
        "validation-style model-space overlays, original-resolution group "
        f"overlays, and Mineral overlaps to {a.output_dir}"
    )


if __name__ == "__main__":
    main(parse_args())
