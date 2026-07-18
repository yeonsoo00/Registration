"""Deterministically replay the Stage-1 teacher/student registration pipeline.

This script intentionally builds a synthetic-only dataset without an
unregistered-data root.  It loads a full teacher/student checkpoint, predicts
with both branches, prints the relevant affine errors, and saves image panels
that make a transform-direction mistake visible.  This recreates the saved
pipeline and transform ranges; it does not claim to recover historical random
draws that were not stored during training.
"""

from __future__ import annotations

import argparse
import csv
import os
import re
from collections import OrderedDict
from typing import Iterable

import numpy as np
import torch
from PIL import Image, ImageDraw

from dataset import CartilageDataset
from losses import affine_control_point_loss
from models import (
    TeacherStudentAffineRegistrationModel,
    canonicalize_model_config,
)
from train import ARCHITECTURE, INPUT_CONTRACT_VERSION, warp_group
from utils import (
    STRUCTURAL_CHANNEL_NAMES,
    affine_parameters_to_matrix,
    apply_affine_transform,
    invert_affine_matrix,
    resolve_device,
)


DEFAULT_OUTPUT_DIR = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    "Debug",
    "teacher_transform_audit",
)
DIRECTION_ABSOLUTE_TOLERANCE = 1e-5
DIRECTION_RELATIVE_TOLERANCE = 1e-3


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Print and visualize the TeacherStudent Stage-1 sources, affine "
            "directions, predictions, and control-point errors."
        )
    )
    parser.add_argument(
        "--checkpoint",
        required=True,
        help="Full TeacherStudent checkpoint containing both branches",
    )
    parser.add_argument(
        "--registered_root",
        help=("Registered-data root; defaults to the path saved in the checkpoint"),
    )
    parser.add_argument("--output_dir", default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--max_items", type=int, default=8)
    parser.add_argument("--synthetic_seed", type=int, default=9090)
    parser.add_argument(
        "--device", default="cuda" if torch.cuda.is_available() else "cpu"
    )
    parser.add_argument("--gpu_ids", default="0")
    return parser.parse_args()


def _state_without_data_parallel_prefix(state: dict) -> OrderedDict:
    return OrderedDict(
        (key[7:] if key.startswith("module.") else key, value)
        for key, value in state.items()
    )


def load_full_teacher_student_checkpoint(checkpoint_path: str, device: torch.device):
    """Strictly load the full checkpoint needed for a teacher-path audit."""
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    if checkpoint.get("architecture") != ARCHITECTURE:
        raise ValueError(
            f"Expected architecture {ARCHITECTURE!r}, got "
            f"{checkpoint.get('architecture')!r}"
        )
    if checkpoint.get("checkpoint_type") != "full_training":
        raise ValueError(
            "The audit requires the full checkpoint, not a *_student.pt "
            "inference artifact."
        )
    if not checkpoint.get("use_teacher_branch", False):
        raise ValueError("The checkpoint does not contain an enabled teacher branch")
    required = {
        "student_model_config",
        "student_model_state_dict",
        "teacher_model_state_dict",
        "preprocess_config",
    }
    missing = sorted(required.difference(checkpoint))
    if missing:
        raise ValueError("Checkpoint is missing: " + ", ".join(missing))

    preprocess = checkpoint["preprocess_config"]
    if preprocess.get("input_contract_version") != INPUT_CONTRACT_VERSION:
        raise ValueError(
            "Checkpoint input contract differs from this audit: "
            f"{preprocess.get('input_contract_version')!r}"
        )

    student_model_config = canonicalize_model_config(checkpoint["student_model_config"])
    checkpoint = dict(checkpoint)
    checkpoint["student_model_config"] = student_model_config
    model = TeacherStudentAffineRegistrationModel(
        student_config=student_model_config,
        use_teacher_branch=True,
    )
    model.student.load_state_dict(
        _state_without_data_parallel_prefix(checkpoint["student_model_state_dict"]),
        strict=True,
    )
    assert model.teacher is not None
    model.teacher.load_state_dict(
        _state_without_data_parallel_prefix(checkpoint["teacher_model_state_dict"]),
        strict=True,
    )
    model.to(device).eval()
    return checkpoint, model


def warp_group_with_matrix(
    moving_group: torch.Tensor, matrix: torch.Tensor
) -> torch.Tensor:
    """Warp every stain slot with an explicit Bx2x3 affine-grid matrix."""
    if moving_group.ndim != 5:
        raise ValueError("moving_group must have shape BxKxCxHxW")
    batch_size, slots, channels, height, width = moving_group.shape
    if tuple(matrix.shape) != (batch_size, 2, 3):
        raise ValueError(
            f"matrix must have shape {(batch_size, 2, 3)}, got {tuple(matrix.shape)}"
        )
    repeated = matrix[:, None].expand(batch_size, slots, 2, 3)
    warped = apply_affine_transform(
        moving_group.reshape(batch_size * slots, channels, height, width),
        repeated.reshape(batch_size * slots, 2, 3),
    )
    return warped.reshape(batch_size, slots, channels, height, width)


def shared_signal_union_support(
    tensors: Iterable[torch.Tensor],
    valid_group: torch.Tensor,
    epsilon: float = 1e-6,
) -> torch.Tensor:
    """Return one BxKx1xHxW support shared by every compared image tensor."""
    tensors = list(tensors)
    if not tensors:
        raise ValueError("At least one tensor is required to build signal support")
    reference = tensors[0]
    if reference.ndim != 5:
        raise ValueError("Signal tensors must have shape BxKxCxHxW")
    batch_size, slots, _, height, width = reference.shape
    if tuple(valid_group.shape) != (batch_size, slots):
        raise ValueError("valid_group shape does not match the grouped images")
    support = torch.zeros(
        (batch_size, slots, 1, height, width),
        dtype=torch.bool,
        device=reference.device,
    )
    for tensor in tensors:
        if tensor.shape != reference.shape:
            raise ValueError("All signal tensors must have identical BxKxCxHxW shapes")
        if tensor.device != reference.device:
            raise ValueError("All signal tensors must be on the same device")
        support |= tensor.detach().abs().amax(dim=2, keepdim=True) > epsilon
    support &= valid_group[:, :, None, None, None].bool()
    if not bool(support.any()):
        support = (
            valid_group[:, :, None, None, None]
            .bool()
            .expand(batch_size, slots, 1, height, width)
        )
    return support


def signal_union_mae(
    prediction: torch.Tensor,
    target: torch.Tensor,
    valid_group: torch.Tensor,
    epsilon: float = 1e-6,
    shared_support: torch.Tensor | None = None,
) -> torch.Tensor:
    """MAE over valid slots using an optional fixed comparison support."""
    if prediction.shape != target.shape or prediction.ndim != 5:
        raise ValueError("prediction and target must be equal BxKxCxHxW tensors")
    batch_size, slots, channels, height, width = prediction.shape
    if tuple(valid_group.shape) != (batch_size, slots):
        raise ValueError("valid_group shape does not match the grouped images")
    if shared_support is None:
        weight = shared_signal_union_support(
            (prediction, target), valid_group, epsilon=epsilon
        )
    else:
        expected_shape = (batch_size, slots, 1, height, width)
        if tuple(shared_support.shape) != expected_shape:
            raise ValueError(
                f"shared_support must have shape {expected_shape}, got "
                f"{tuple(shared_support.shape)}"
            )
        if shared_support.device != prediction.device:
            raise ValueError("shared_support and image tensors must share one device")
        weight = shared_support.bool().clone()
        weight &= valid_group[:, :, None, None, None].bool()
        if not bool(weight.any()):
            weight = (
                valid_group[:, :, None, None, None]
                .bool()
                .expand(batch_size, slots, 1, height, width)
            )
    denominator = weight.sum().clamp_min(1).to(prediction.dtype) * channels
    return ((prediction - target).abs() * weight).sum() / denominator


def _hsv_group_to_rgb_tensor(group_stack: torch.Tensor) -> torch.Tensor:
    """Convert normalized BxKx3xHxW HSV tensors to RGB for image metrics."""
    if group_stack.ndim != 5 or group_stack.shape[2] != 3:
        raise ValueError("HSV group stack must have shape BxKx3xHxW")
    hue = torch.remainder(group_stack[:, :, 0], 1.0) * 6.0
    saturation = group_stack[:, :, 1].clamp(0.0, 1.0)
    value = group_stack[:, :, 2].clamp(0.0, 1.0)
    sector = torch.floor(hue).long() % 6
    fraction = hue - torch.floor(hue)
    p = value * (1.0 - saturation)
    q = value * (1.0 - fraction * saturation)
    t = value * (1.0 - (1.0 - fraction) * saturation)
    candidates = (
        torch.stack((value, t, p), dim=2),
        torch.stack((q, value, p), dim=2),
        torch.stack((p, value, t), dim=2),
        torch.stack((p, q, value), dim=2),
        torch.stack((t, p, value), dim=2),
        torch.stack((value, p, q), dim=2),
    )
    rgb = torch.zeros_like(group_stack)
    for index, candidate in enumerate(candidates):
        rgb = torch.where((sector == index).unsqueeze(2), candidate, rgb)
    return rgb.clamp(0.0, 1.0)


def _metric_group_stack(
    group_stack: torch.Tensor, group_id: int, sfo_mode: str
) -> torch.Tensor:
    """Return a metric-space stack; G5 HSV is compared in visible RGB space."""
    if group_id == 5 and sfo_mode == "hsv":
        return _hsv_group_to_rgb_tensor(group_stack)
    return group_stack


def _direction_status(
    true_error: torch.Tensor,
    moving_error: torch.Tensor,
    inverse_error: torch.Tensor,
) -> str:
    """Classify direction evidence with absolute and relative tie tolerances."""

    def tolerance(first: float, second: float) -> float:
        return max(
            DIRECTION_ABSOLUTE_TOLERANCE,
            DIRECTION_RELATIVE_TOLERANCE * max(abs(first), abs(second)),
        )

    true_value = float(true_error)
    moving_value = float(moving_error)
    inverse_value = float(inverse_error)
    moving_margin = moving_value - true_value
    inverse_margin = inverse_value - true_value
    moving_tolerance = tolerance(moving_value, true_value)
    inverse_tolerance = tolerance(inverse_value, true_value)
    if moving_margin > moving_tolerance and inverse_margin > inverse_tolerance:
        return "PASS"
    if (
        moving_margin >= -moving_tolerance
        and inverse_margin >= -inverse_tolerance
        and (
            abs(moving_margin) <= moving_tolerance
            or abs(inverse_margin) <= inverse_tolerance
        )
    ):
        return "INCONCLUSIVE"
    return "REVIEW"


def _allocate_run_directory(
    output_root: str,
    checkpoint_path: str,
    checkpoint_epoch: object,
    synthetic_seed: int,
) -> str:
    """Create a collision-safe run directory without touching earlier audits."""
    os.makedirs(output_root, exist_ok=True)
    checkpoint_stem = os.path.splitext(os.path.basename(checkpoint_path))[0]
    checkpoint_stem = re.sub(r"[^A-Za-z0-9_.-]+", "_", checkpoint_stem)
    epoch_text = re.sub(r"[^A-Za-z0-9_.-]+", "_", str(checkpoint_epoch))
    base_name = f"{checkpoint_stem}_epoch{epoch_text}_seed{synthetic_seed}"
    suffix = 0
    while True:
        name = base_name if suffix == 0 else f"{base_name}_{suffix:03d}"
        candidate = os.path.join(output_root, name)
        try:
            os.makedirs(candidate, exist_ok=False)
        except FileExistsError:
            suffix += 1
            continue
        return candidate


def _hsv_to_rgb(image: np.ndarray) -> np.ndarray:
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
    return np.clip(rgb, 0.0, 1.0)


def _display_rgb(image: torch.Tensor, group_id: int, sfo_mode: str) -> np.ndarray:
    image = image.detach().float().clamp(0.0, 1.0).cpu()
    if image.shape[0] == 1:
        image = image.expand(3, -1, -1)
    if image.shape[0] != 3:
        raise ValueError(
            f"Display image must have 1 or 3 channels, got {image.shape[0]}"
        )
    array = image.permute(1, 2, 0).numpy()
    if group_id == 5 and sfo_mode == "hsv":
        array = _hsv_to_rgb(array)
    return np.clip(array, 0.0, 1.0)


def _group_union(
    group_stack: torch.Tensor,
    valid_group: torch.Tensor,
    group_id: int,
    sfo_mode: str,
) -> np.ndarray:
    components = [
        _display_rgb(group_stack[slot], group_id, sfo_mode)
        for slot, valid in enumerate(valid_group.tolist())
        if valid
    ]
    if not components:
        height, width = group_stack.shape[-2:]
        return np.zeros((height, width, 3), dtype=np.float32)
    return 1.0 - np.prod(1.0 - np.stack(components, axis=0), axis=0)


def _save_rgb(path: str, image: np.ndarray) -> None:
    pixels = np.rint(np.clip(image, 0.0, 1.0) * 255.0).astype(np.uint8)
    Image.fromarray(pixels, mode="RGB").save(path)


def _save_labeled_panel(path: str, entries: Iterable[tuple[str, np.ndarray]]) -> None:
    entries = list(entries)
    if not entries:
        raise ValueError("A debug panel needs at least one image")
    label_height = 30
    width = max(image.shape[1] for _, image in entries)
    height = max(image.shape[0] for _, image in entries)
    panel = Image.new("RGB", (width * len(entries), height + label_height), "black")
    draw = ImageDraw.Draw(panel)
    for index, (label, image) in enumerate(entries):
        pixels = np.rint(np.clip(image, 0.0, 1.0) * 255.0).astype(np.uint8)
        tile = Image.fromarray(pixels, mode="RGB")
        panel.paste(tile, (index * width, label_height))
        draw.text((index * width + 4, 7), label, fill="white")
    panel.save(path)


def _save_structural_descriptor(
    path: str, descriptor: torch.Tensor, validity: torch.Tensor
) -> None:
    descriptor = descriptor.detach().float().clamp(0.0, 1.0).cpu()
    validity = validity.detach().bool().cpu()
    entries = []
    for channel, name in enumerate(STRUCTURAL_CHANNEL_NAMES):
        image = descriptor[channel] * validity[0]
        rgb_tensor = image.unsqueeze(0).expand(3, -1, -1).permute(1, 2, 0)
        rgb = np.from_dlpack(rgb_tensor.contiguous())
        entries.append((name, rgb))
    _save_labeled_panel(path, entries)


def _save_frontend_representation(
    path: str,
    representation: torch.Tensor,
    validity: torch.Tensor,
    max_channels: int = 8,
) -> None:
    """Save bounded, independently normalized channels from a learned frontend."""
    representation = representation.detach().float().cpu()
    validity = validity.detach().bool().cpu()
    if representation.ndim != 3 or validity.shape != (1, *representation.shape[-2:]):
        raise ValueError("Frontend panel expects CxHxW features and 1xHxW validity")
    valid_pixels = validity[0]
    entries = []
    for channel in range(min(max_channels, representation.shape[0])):
        feature = representation[channel]
        normalized = torch.zeros_like(feature)
        values = feature[valid_pixels]
        if values.numel():
            minimum = values.amin()
            maximum = values.amax()
            if float(maximum - minimum) > 1e-8:
                normalized = (feature - minimum) / (maximum - minimum)
            elif float(maximum.abs()) > 1e-8:
                normalized[valid_pixels] = 0.5
        normalized = normalized.clamp(0.0, 1.0) * valid_pixels.to(normalized.dtype)
        rgb_tensor = normalized.unsqueeze(0).expand(3, -1, -1).permute(1, 2, 0)
        rgb = np.from_dlpack(rgb_tensor.contiguous())
        entries.append((f"feature_{channel:02d}", rgb))
    _save_labeled_panel(path, entries)


def _format_vector(values: torch.Tensor) -> str:
    return "[" + ", ".join(f"{float(value):+.6f}" for value in values) + "]"


def _select_indices(dataset: CartilageDataset, maximum: int) -> list[int]:
    """Cover G2-G5 once before filling remaining deterministic items."""
    selected = []
    for desired_group in (2, 3, 4, 5):
        match = next(
            (
                index
                for index, (_, group_id, _) in enumerate(dataset.items)
                if group_id == desired_group and index not in selected
            ),
            None,
        )
        if match is not None:
            selected.append(match)
            if len(selected) == maximum:
                return selected
    for index in range(len(dataset)):
        if index not in selected:
            selected.append(index)
            if len(selected) == maximum:
                break
    return selected


def _build_synthetic_dataset(
    checkpoint: dict, registered_root: str, synthetic_seed: int
) -> CartilageDataset:
    preprocess = checkpoint["preprocess_config"]
    train_config = checkpoint.get("train_config", {})
    return CartilageDataset(
        registered_root=registered_root,
        # Deliberately empty: this audit proves Stage 1 does not read real
        # unregistered images when synthetic_prob is exactly one.
        unregistered_root="",
        size=(int(preprocess["height"]), int(preprocess["width"])),
        image_mode=preprocess["image_mode"],
        sfo_mode=preprocess["sfo_mode"],
        crop_mode=preprocess["crop_mode"],
        crop_margin=int(preprocess["crop_margin"]),
        synthetic_prob=1.0,
        tx_range=tuple(train_config.get("tx_range", (-64.0, 64.0))),
        ty_range=tuple(train_config.get("ty_range", (-64.0, 64.0))),
        rot_range=tuple(train_config.get("rot_range", (-15.0, 15.0))),
        scale_range=tuple(train_config.get("scale_range", (0.85, 1.15))),
        deterministic_synthetic=True,
        synthetic_seed=synthetic_seed,
        include_group1=False,
        require_registered_targets=True,
    )


def run_audit(args: argparse.Namespace) -> list[dict]:
    if args.max_items < 1:
        raise ValueError("max_items must be positive")
    device, _ = resolve_device(args.device, args.gpu_ids)
    checkpoint, model = load_full_teacher_student_checkpoint(args.checkpoint, device)
    frontend_mode = checkpoint["student_model_config"]["frontend_mode"]
    train_config = checkpoint.get("train_config", {})
    registered_root = args.registered_root or train_config.get("registered_root")
    if not registered_root:
        raise ValueError(
            "--registered_root is required because the checkpoint has no saved path"
        )

    dataset = _build_synthetic_dataset(checkpoint, registered_root, args.synthetic_seed)
    selected_indices = _select_indices(dataset, min(args.max_items, len(dataset)))
    run_dir = _allocate_run_directory(
        args.output_dir,
        args.checkpoint,
        checkpoint.get("epoch", "unknown"),
        args.synthetic_seed,
    )
    log_lines: list[str] = []

    def emit(message: str = "") -> None:
        print(message)
        log_lines.append(message)

    emit("Teacher/student deterministic Stage-1 pipeline replay")
    emit(f"checkpoint = {os.path.abspath(args.checkpoint)}")
    emit(f"checkpoint_epoch = {checkpoint.get('epoch')}")
    emit(f"registered_root = {os.path.abspath(registered_root)}")
    emit(f"output_run_dir = {os.path.abspath(run_dir)}")
    emit("real_unregistered_root = NOT USED (the audit passes an empty path)")
    emit(f"frontend_mode = {frontend_mode}")
    emit(
        "teacher_fixed_source = target_group "
        f"{frontend_mode} frontend representation, NOT Mineral"
    )
    emit(
        "moving_source = synthetic inverse-matrix warp of target_group, "
        "NOT real unregistered"
    )
    emit(
        "params_true convention = moving->target registration; its affine_grid "
        "matrix maps target/output coordinates to moving/input coordinates"
    )
    emit("")

    rows = []
    direction_status_counts = {"PASS": 0, "INCONCLUSIVE": 0, "REVIEW": 0}
    with torch.no_grad():
        for output_index, dataset_index in enumerate(selected_indices):
            sample = dataset[dataset_index]
            base_id, metadata_group_id, _ = dataset.items[dataset_index]
            group_id = int(sample["group_id"])
            if group_id != metadata_group_id:
                raise AssertionError("Dataset item metadata and group tensor differ")
            if not bool(sample["has_params"]):
                raise AssertionError("Synthetic audit item has no ground-truth affine")
            sfo_mode = checkpoint["preprocess_config"]["sfo_mode"]

            batch = {
                key: value.unsqueeze(0).to(device)
                for key, value in sample.items()
                if torch.is_tensor(value)
            }
            params_true = batch["params_true"]
            teacher_params = model.forward_teacher(
                target_group=batch["target_group"],
                moving_group=batch["moving_group"],
                group=batch["group_id"],
            )
            student_params = model(
                fixed_mineral=batch["fixed_mineral"],
                moving_group=batch["moving_group"],
                group=batch["group_id"],
            )

            true_matrix = affine_parameters_to_matrix(params_true)
            inverse_true_matrix = invert_affine_matrix(true_matrix)
            warped_true = warp_group(batch["moving_group"], params_true)
            warped_inverse = warp_group_with_matrix(
                batch["moving_group"], inverse_true_matrix
            )
            warped_teacher = warp_group(batch["moving_group"], teacher_params)
            warped_student = warp_group(batch["moving_group"], student_params)

            target = batch["target_group"]
            valid = batch["valid_group"]
            metric_stacks = {
                "target": target,
                "moving": batch["moving_group"],
                "true": warped_true,
                "inverse": warped_inverse,
                "teacher": warped_teacher,
                "student": warped_student,
            }
            metric_stacks = {
                name: _metric_group_stack(stack, group_id, sfo_mode)
                for name, stack in metric_stacks.items()
            }
            comparison_support = shared_signal_union_support(
                metric_stacks.values(), valid
            )

            def metric_error(name: str) -> torch.Tensor:
                return signal_union_mae(
                    metric_stacks[name],
                    metric_stacks["target"],
                    valid,
                    shared_support=comparison_support,
                )

            moving_error = metric_error("moving")
            true_error = metric_error("true")
            inverse_error = metric_error("inverse")
            teacher_image_error = metric_error("teacher")
            student_image_error = metric_error("student")
            teacher_cp_error = affine_control_point_loss(teacher_params, params_true)
            student_cp_error = affine_control_point_loss(student_params, params_true)
            direction_status = _direction_status(
                true_error, moving_error, inverse_error
            )
            direction_status_counts[direction_status] += 1

            homogeneous_true = torch.eye(
                3, dtype=true_matrix.dtype, device=true_matrix.device
            ).unsqueeze(0)
            homogeneous_true[:, :2] = true_matrix
            homogeneous_inverse = torch.eye(
                3, dtype=true_matrix.dtype, device=true_matrix.device
            ).unsqueeze(0)
            homogeneous_inverse[:, :2] = inverse_true_matrix
            inverse_residual = (
                (
                    homogeneous_true @ homogeneous_inverse
                    - torch.eye(3, dtype=true_matrix.dtype, device=true_matrix.device)
                )
                .abs()
                .amax()
            )

            teacher_branch = model.teacher
            assert teacher_branch is not None
            (
                teacher_fixed_representation,
                teacher_fixed_valid,
            ) = teacher_branch.group_frontend_representation(
                batch["target_group"].float(), batch["group_id"]
            )

            safe_base = re.sub(r"[^A-Za-z0-9_.-]+", "_", str(base_id))
            item_dir = os.path.join(
                run_dir, f"{output_index:03d}_{safe_base}_G{group_id}"
            )
            os.makedirs(item_dir, exist_ok=True)
            valid_single = sample["valid_group"]
            images = {
                "fixed_mineral_not_teacher_source": _display_rgb(
                    sample["fixed_mineral"], 1, "rgb"
                ),
                "teacher_fixed_source_target_group_union": _group_union(
                    sample["target_group"], valid_single, group_id, sfo_mode
                ),
                "moving_source_synthetic_warped_target_group": _group_union(
                    sample["moving_group"], valid_single, group_id, sfo_mode
                ),
                "warp_moving_group_params_true": _group_union(
                    warped_true[0], valid_single, group_id, sfo_mode
                ),
                "warp_moving_group_inverse_params_true_matrix": _group_union(
                    warped_inverse[0], valid_single, group_id, sfo_mode
                ),
                "warp_moving_group_teacher_params": _group_union(
                    warped_teacher[0], valid_single, group_id, sfo_mode
                ),
                "warp_moving_group_student_params": _group_union(
                    warped_student[0], valid_single, group_id, sfo_mode
                ),
                "target_group": _group_union(
                    sample["target_group"], valid_single, group_id, sfo_mode
                ),
            }
            for name, image in images.items():
                _save_rgb(os.path.join(item_dir, f"{name}.png"), image)
            for slot, valid_slot in enumerate(valid_single.tolist()):
                if not valid_slot:
                    continue
                stain_index = int(sample["stain_indices"][slot])
                _save_rgb(
                    os.path.join(item_dir, f"target_group_stain{stain_index}.png"),
                    _display_rgb(sample["target_group"][slot], group_id, sfo_mode),
                )
            if frontend_mode == "structural":
                _save_structural_descriptor(
                    os.path.join(item_dir, "teacher_fixed_structural_descriptor.png"),
                    teacher_fixed_representation[0],
                    teacher_fixed_valid[0],
                )
            else:
                _save_frontend_representation(
                    os.path.join(
                        item_dir,
                        "teacher_fixed_frontend_representation.png",
                    ),
                    teacher_fixed_representation[0],
                    teacher_fixed_valid[0],
                )
            _save_labeled_panel(
                os.path.join(item_dir, "flow_panel.png"),
                [
                    (
                        "Mineral (not teacher)",
                        images["fixed_mineral_not_teacher_source"],
                    ),
                    (
                        f"teacher fixed: {frontend_mode}",
                        images["teacher_fixed_source_target_group_union"],
                    ),
                    (
                        "moving: synthetic",
                        images["moving_source_synthetic_warped_target_group"],
                    ),
                    ("warp true", images["warp_moving_group_params_true"]),
                    (
                        "warp inverse (wrong)",
                        images["warp_moving_group_inverse_params_true_matrix"],
                    ),
                    (
                        "warp teacher",
                        images["warp_moving_group_teacher_params"],
                    ),
                    (
                        "warp student",
                        images["warp_moving_group_student_params"],
                    ),
                    ("target_group", images["target_group"]),
                ],
            )

            valid_stains = [
                int(stain)
                for stain, valid_slot in zip(
                    sample["stain_indices"], sample["valid_group"]
                )
                if bool(valid_slot)
            ]
            emit(f"[{output_index:03d}] sample={base_id} group=G{group_id}")
            emit(
                "  teacher_fixed_source = target_group "
                f"{frontend_mode} frontend, not Mineral"
            )
            emit(
                "  moving_source = synthetic warped target_group, "
                "not real unregistered"
            )
            emit(f"  valid_stains = {valid_stains}; has_params = True")
            emit(f"  params_true = {_format_vector(params_true[0])}")
            emit(f"  teacher_params = {_format_vector(teacher_params[0])}")
            emit(f"  student_params = {_format_vector(student_params[0])}")
            emit(
                "  unwarped moving_group -> target_group shared-support RGB MAE "
                f"= {float(moving_error):.8f}"
            )
            emit(
                "  warp(moving_group, params_true) -> target_group "
                "shared-support RGB MAE "
                f"= {float(true_error):.8f}"
            )
            emit(
                "  warp(moving_group, inverse(params_true matrix)) -> "
                "target_group shared-support RGB MAE "
                f"= {float(inverse_error):.8f}"
            )
            emit(
                "  warp(moving_group, teacher_params) -> target_group "
                "shared-support RGB MAE "
                f"= {float(teacher_image_error):.8f}"
            )
            emit(
                "  warp(moving_group, student_params) -> target_group "
                "shared-support RGB MAE "
                f"= {float(student_image_error):.8f}"
            )
            emit(
                f"  target_group shape = {tuple(sample['target_group'].shape)}; "
                f"range = [{float(sample['target_group'].min()):.4f}, "
                f"{float(sample['target_group'].max()):.4f}]"
            )
            emit(
                "  teacher control-point error vs params_true = "
                f"{float(teacher_cp_error):.8f}"
            )
            emit(
                "  student control-point error vs params_true = "
                f"{float(student_cp_error):.8f}"
            )
            emit(
                "  direction check (true warp beats unwarped and inverse) = "
                f"{direction_status}"
            )
            emit(f"  affine inverse residual = {float(inverse_residual):.3e}")
            emit(f"  images = {item_dir}")
            emit("")

            row = {
                "item": output_index,
                "dataset_index": dataset_index,
                "sample": str(base_id),
                "group": group_id,
                "valid_stains": ";".join(map(str, valid_stains)),
                "moving_vs_target_signal_mae": float(moving_error),
                "true_warp_vs_target_signal_mae": float(true_error),
                "inverse_warp_vs_target_signal_mae": float(inverse_error),
                "teacher_warp_vs_target_signal_mae": float(teacher_image_error),
                "student_warp_vs_target_signal_mae": float(student_image_error),
                "teacher_control_point_error": float(teacher_cp_error),
                "student_control_point_error": float(student_cp_error),
                "direction_status": direction_status,
                "true_direction_better_than_inverse": bool(true_error < inverse_error),
                "affine_inverse_residual": float(inverse_residual),
            }
            parameter_names = ("tx", "ty", "theta", "sx", "sy")
            for prefix, values in (
                ("true", params_true[0]),
                ("teacher", teacher_params[0]),
                ("student", student_params[0]),
            ):
                for name, value in zip(parameter_names, values):
                    row[f"{prefix}_{name}"] = float(value)
            rows.append(row)

    metrics_path = os.path.join(run_dir, "metrics.csv")
    with open(metrics_path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    emit(
        "Direction summary: "
        + ", ".join(
            f"{status}={direction_status_counts[status]}"
            for status in ("PASS", "INCONCLUSIVE", "REVIEW")
        )
        + f" across {len(rows)} items."
    )
    emit(f"Metrics CSV = {metrics_path}")
    log_path = os.path.join(run_dir, "audit.txt")
    with open(log_path, "w", encoding="utf-8") as handle:
        handle.write("\n".join(log_lines) + "\n")
    print(f"Audit run directory = {run_dir}")
    print(f"Audit log = {log_path}")
    return rows


def main() -> None:
    run_audit(parse_args())


if __name__ == "__main__":
    main()
