"""Train the teacher/student correlation-volume affine registration model.

Stage 1 trains from scratch on synthetic transforms. Stage 2 resumes the
Correlation_Vol_Net Stage 1 checkpoint and fine-tunes on real unregistered
images. Exact commands are documented in README.md.
"""

from __future__ import annotations

import argparse
import os
import random
import re
import warnings
from collections import OrderedDict
from contextlib import nullcontext
from typing import Any, Dict, List

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset
from tqdm import tqdm

from dataset import MAX_GROUP_STAINS, CartilageDataset
from losses import (
    affine_control_point_loss,
    charbonnier_loss,
    gradient_ncc_loss,
    multiscale_gradient_loss,
    multiscale_local_ncc_loss,
    regularisation_loss,
    soft_foreground_dice_loss,
)
from models import (
    AFFINE_HEAD_MODES,
    DEFAULT_GROUP_SLOTS,
    FRONTEND_MODES,
    GROUP_INPUT_MODES,
    TEACHER_FIXED_INPUT_VERSION,
    TeacherStudentAffineRegistrationModel,
    canonicalize_model_config,
)
from utils import (
    affine_parameters_to_matrix,
    apply_affine_transform,
    resolve_device,
    save_group_overlay as _save_group_overlay,
    supervision_source_and_matrices,
    synthetic_full_correction_matrices,
    warp_group,
    warp_group_for_supervision,
    warp_group_with_matrix,
    warp_model_space_group,
)


ARCHITECTURE = "correlation_volume_teacher_student_affine_v1"
STRUCTURAL_DESCRIPTOR_VERSION = "torch_structural_no_enhancement_v3"
INPUT_CONTRACT_VERSION = "fixed_mineral_moving_group_raw_v1"
AFFINE_ERROR_METRIC_NAMES = (
    "tx_mae_px",
    "ty_mae_px",
    "theta_mae_deg",
    "sx_mae",
    "sy_mae",
    "control_point_error_px",
)
AFFINE_ERROR_COMPONENT_NAMES = frozenset(
    f"{path_name}_{metric_name}"
    for path_name in ("student", "teacher")
    for metric_name in AFFINE_ERROR_METRIC_NAMES
)

# Only these tensors participate in model prediction or loss computation.
# ``target_group`` is used only by the training teacher and supervision loss;
# it is never passed to the deployable student.
DEVICE_BATCH_KEYS = frozenset(
    {
        "fixed_mineral",
        "moving_group",
        "target_group",
        "valid_group",
        "group_id",
        "params_true",
        "has_params",
    }
)


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--registered_root", required=True)
    p.add_argument("--unregistered_root", required=True)
    p.add_argument("--output_dir", required=True)
    p.add_argument("--epochs", type=int, default=200)
    p.add_argument("--batch_size", type=int, default=4)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--weight_decay", type=float, default=1e-5)
    p.add_argument("--n_workers", type=int, default=4)
    p.add_argument("--seed", type=int, default=1234)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--gpu_ids", default="0,1")
    p.add_argument("--no_multi_gpu", action="store_true")
    p.add_argument("--amp", action="store_true")
    p.add_argument("--height", type=int, default=512)
    p.add_argument("--width", type=int, default=512)
    p.add_argument("--image_mode", choices=["rgb", "gray"], default="rgb")
    p.add_argument("--sfo_mode", choices=["rgb", "hsv", "gray"], default="rgb")
    p.add_argument("--crop_mode", choices=["full", "mineral_bbox"], default="full")
    p.add_argument("--crop_margin", type=int, default=32)
    p.add_argument(
        "--frontend_mode",
        choices=FRONTEND_MODES,
        default="structural",
        help=(
            "Model frontend: structural preserves the existing six-channel "
            "descriptor, raw uses normalized images, and hybrid uses both"
        ),
    )
    p.add_argument(
        "--group_input_mode",
        choices=GROUP_INPUT_MODES,
        default="overlay",
        help=(
            "Moving-group representation for raw/hybrid frontends: stack keeps "
            "padded stain slots, overlay takes the valid-stain maximum union"
        ),
    )
    p.add_argument(
        "--affine_head_mode",
        choices=AFFINE_HEAD_MODES,
        default="joint",
        help=(
            "Affine regressor: joint predicts all parameters together, separated "
            "uses geometry-specific heads, and separated_residual refines a "
            "cost-volume displacement estimate"
        ),
    )
    p.add_argument(
        "--structural_foreground_threshold",
        type=float,
        default=None,
        help="Foreground threshold in [0,1]; omit for an adaptive per-image threshold",
    )
    p.add_argument("--structural_distance_scale", type=float, default=0.03)
    p.add_argument("--structural_context_scale", type=float, default=0.03)
    p.add_argument("--structural_skeleton_radius", type=int, default=4)
    p.add_argument("--use_group_embedding", action="store_true")
    p.add_argument(
        "--include_group1", action=argparse.BooleanOptionalAction, default=False
    )
    p.add_argument("--encoder_base_channels", type=int, default=24)
    p.add_argument("--encoder_depth", type=int, default=5)
    p.add_argument("--feature_width", type=int, default=48)
    p.add_argument("--cost_hidden_channels", type=int, default=48)
    p.add_argument("--cost_volume_radii", type=int, nargs="+", default=[4, 4, 4])
    p.add_argument("--cost_pool_size", type=int, default=4)
    p.add_argument("--correlation_temperature", type=float, default=0.07)
    p.add_argument("--latent_dim", type=int, default=384)
    p.add_argument("--group_embedding_dim", type=int, default=32)
    p.add_argument(
        "--norm_type", choices=["group", "batch", "instance"], default="group"
    )
    p.add_argument(
        "--force_group1_identity",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Required safety contract: AC and Calcein stay fixed with Mineral",
    )
    p.add_argument(
        "--separate_group_heads", action=argparse.BooleanOptionalAction, default=True
    )
    p.add_argument(
        "--separate_group_adapters",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    p.add_argument("--model_scale_range", type=float, nargs=2, default=[0.8, 1.2])
    p.add_argument("--translation_limit", type=float, default=0.5)
    p.add_argument("--max_rotation_deg", type=float, default=20)
    p.add_argument("--synthetic_prob", type=float, default=1.0)
    p.add_argument("--val_synthetic_prob", type=float, default=1.0)
    p.add_argument("--tx_range", type=float, nargs=2, default=[-64, 64])
    p.add_argument("--ty_range", type=float, nargs=2, default=[-64, 64])
    p.add_argument("--rot_range", type=float, nargs=2, default=[-20, 20])
    p.add_argument("--scale_range", type=float, nargs=2, default=[0.8, 1.2])
    p.add_argument("--param_weight", type=float, default=10.0)
    p.add_argument("--ncc_weight", type=float, default=1.0)
    p.add_argument("--edge_weight", type=float, default=0.25)
    p.add_argument("--charbonnier_weight", type=float, default=0.1)
    p.add_argument("--gradient_weight", type=float, default=0.1)
    p.add_argument("--overlap_weight", type=float, default=0.5)
    p.add_argument("--reg_weight", type=float, default=0.0)
    p.add_argument(
        "--use_teacher_branch",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Train the target-group teacher; the student remains deployable alone",
    )
    p.add_argument("--teacher_distill_weight", type=float, default=1.0)
    p.add_argument(
        "--detach_teacher",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Stop distillation gradients at the teacher prediction",
    )
    p.add_argument(
        "--teacher_warmup_epochs",
        type=int,
        default=0,
        help=(
            "Train only the teacher for epochs 1..N using image losses on all "
            "samples and affine labels only where has_params=True; the student "
            "is frozen and distillation starts at epoch N+1"
        ),
    )
    p.add_argument(
        "--freeze_teacher",
        action=argparse.BooleanOptionalAction,
        default=False,
        help=(
            "Keep a resumed teacher frozen/eval and detach its Stage-2 "
            "distillation targets"
        ),
    )
    p.add_argument("--val_split", type=float, default=0.15)
    p.add_argument("--split_seed", type=int, default=2026)
    p.add_argument("--grad_clip", type=float, default=1.0)
    p.add_argument("--resume_checkpoint")
    p.add_argument("--best_checkpoint_name", default="best_model.pt")
    p.add_argument("--last_checkpoint_name", default="last_model.pt")
    p.add_argument("--wandb_project")
    p.add_argument("--wandb_run_name")
    p.add_argument(
        "--max_train_items",
        type=int,
        default=0,
        help="Smoke-only cap after the sample split; 0 uses all training items",
    )
    p.add_argument(
        "--max_val_items",
        type=int,
        default=0,
        help="Smoke-only cap after the sample split; 0 uses all validation items",
    )
    return p.parse_args()


def safe_collate(batch: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Stack fixed-shape grouped samples without shared-memory resize.

    PyTorch's default worker-side collate preallocates non-resizable shared
    storage. If any item has a different shape, the resulting error is the
    opaque ``Trying to resize storage that is not resizable``. This collate
    function checks every tensor key explicitly and then calls ``torch.stack``
    without the shared ``out=`` buffer, producing an informative shape error.
    """
    if not batch:
        raise RuntimeError("Cannot collate an empty batch")

    result: Dict[str, Any] = {}
    keys = batch[0].keys()
    for key in keys:
        values = [sample[key] for sample in batch]
        first = values[0]
        if torch.is_tensor(first):
            shapes = [tuple(value.shape) for value in values]
            dtypes = [value.dtype for value in values]
            if any(shape != shapes[0] for shape in shapes[1:]):
                raise RuntimeError(
                    f"Variable tensor shape for key '{key}': {shapes}. "
                    "All grouped samples must use fixed padded dimensions."
                )
            if any(dtype != dtypes[0] for dtype in dtypes[1:]):
                raise RuntimeError(f"Variable tensor dtype for key '{key}': {dtypes}")
            result[key] = torch.stack(
                [value.detach().contiguous().clone() for value in values], dim=0
            )
        else:
            result[key] = values
    return result


def set_seed(s):
    random.seed(s)
    np.random.seed(s)
    torch.manual_seed(s)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(s)


def move_required_tensors(batch, device, non_blocking=False):
    """Move only tensors used by the structural model and registration loss."""
    return {
        key: (
            value.to(device, non_blocking=non_blocking)
            if key in DEVICE_BATCH_KEYS and torch.is_tensor(value)
            else value
        )
        for key, value in batch.items()
    }


def build_preprocess_config(a):
    """Return the exact raw-input construction contract saved in checkpoints."""
    return {
        "input_contract_version": INPUT_CONTRACT_VERSION,
        "input_value_range": [0.0, 1.0],
        "height": a.height,
        "width": a.width,
        "image_mode": a.image_mode,
        "sfo_mode": a.sfo_mode,
        "crop_mode": a.crop_mode,
        "crop_margin": a.crop_margin,
        "include_group1": a.include_group1,
    }


def build_dataset(a, validation=False):
    synthetic_probability = a.val_synthetic_prob if validation else a.synthetic_prob
    return CartilageDataset(
        registered_root=a.registered_root,
        unregistered_root=a.unregistered_root,
        size=(a.height, a.width),
        image_mode=a.image_mode,
        sfo_mode=a.sfo_mode,
        crop_mode=a.crop_mode,
        crop_margin=a.crop_margin,
        synthetic_prob=synthetic_probability,
        tx_range=tuple(a.tx_range),
        ty_range=tuple(a.ty_range),
        rot_range=tuple(a.rot_range),
        scale_range=tuple(a.scale_range),
        deterministic_synthetic=validation,
        synthetic_seed=9090,
        include_group1=a.include_group1,
        require_registered_targets=True,
    )


def split_by_sample(
    train_dataset: CartilageDataset,
    validation_dataset: CartilageDataset,
    validation_fraction: float,
    seed: int,
):
    """Split by tissue sample and index each dataset independently."""
    train_samples = {item[0] for item in train_dataset.items}
    validation_samples = {item[0] for item in validation_dataset.items}
    common_samples = sorted(train_samples & validation_samples)
    if len(common_samples) < 2:
        raise RuntimeError("At least two common samples are required for a split")

    rng = np.random.default_rng(seed)
    rng.shuffle(common_samples)
    validation_count = max(1, int(round(len(common_samples) * validation_fraction)))
    validation_count = min(validation_count, len(common_samples) - 1)
    validation_ids = set(common_samples[:validation_count])
    training_ids = set(common_samples[validation_count:])

    train_indices = [
        index
        for index, item in enumerate(train_dataset.items)
        if item[0] in training_ids
    ]
    validation_indices = [
        index
        for index, item in enumerate(validation_dataset.items)
        if item[0] in validation_ids
    ]
    if not train_indices or not validation_indices:
        raise RuntimeError("The sample-level split produced an empty partition")
    if training_ids & validation_ids:
        raise AssertionError("Training and validation sample IDs overlap")
    return train_indices, validation_indices, training_ids, validation_ids


def affine_error_metrics(params, params_true, height, width):
    """Return interpretable affine errors for labeled model-space transforms.

    Translation MAE follows the dataset label convention, which normalizes
    configured pixel translations by ``width / 2`` and ``height / 2``.
    Control-point error instead measures the actual ``align_corners=True``
    sampling geometry at four corners and the image center.
    """
    if params.ndim != 2 or params.shape[1] != 5:
        raise ValueError(f"params must have shape (B,5), got {tuple(params.shape)}")
    if params_true.shape != params.shape:
        raise ValueError(
            "params_true must have the same shape as params; got "
            f"{tuple(params_true.shape)} and {tuple(params.shape)}"
        )
    if params.shape[0] == 0:
        raise ValueError("Affine metrics require at least one labeled transform")
    if height < 1 or width < 1:
        raise ValueError("Affine metric image dimensions must be positive")

    predicted = params.detach().float()
    target = params_true.detach().float()
    delta = predicted - target
    wrapped_angle = torch.atan2(torch.sin(delta[:, 2]), torch.cos(delta[:, 2]))

    points = predicted.new_tensor(
        (
            (-1.0, -1.0, 1.0),
            (1.0, -1.0, 1.0),
            (-1.0, 1.0, 1.0),
            (1.0, 1.0, 1.0),
            (0.0, 0.0, 1.0),
        )
    )
    predicted_points = torch.einsum(
        "bij,pj->bpi", affine_parameters_to_matrix(predicted), points
    )
    target_points = torch.einsum(
        "bij,pj->bpi", affine_parameters_to_matrix(target), points
    )
    point_delta = predicted_points - target_points
    pixel_scale = predicted.new_tensor(
        (max(width - 1, 1) / 2.0, max(height - 1, 1) / 2.0)
    )
    point_error_px = torch.linalg.vector_norm(
        point_delta * pixel_scale.view(1, 1, 2), dim=2
    )

    return {
        "tx_mae_px": delta[:, 0].abs().mean() * max(width / 2.0, 1.0),
        "ty_mae_px": delta[:, 1].abs().mean() * max(height / 2.0, 1.0),
        "theta_mae_deg": wrapped_angle.abs().mean() * (180.0 / np.pi),
        "sx_mae": delta[:, 3].abs().mean(),
        "sy_mae": delta[:, 4].abs().mean(),
        "control_point_error_px": point_error_px.mean(),
    }


def _rectangular_signal_support(images, valid_group, eps=1e-6):
    """Infer a conservative rectangular FOV from raw zero-padded group images."""
    if images.ndim != 5:
        raise ValueError(f"images must have shape BxKxCxHxW, got {tuple(images.shape)}")
    batch_size, slots, _, height, width = images.shape
    if tuple(valid_group.shape) != (batch_size, slots):
        raise ValueError(
            f"valid_group must have shape {(batch_size, slots)}, got "
            f"{tuple(valid_group.shape)}"
        )

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


def build_group_valid_overlap(
    moving_group,
    target_group,
    valid_group,
    params,
    *,
    params_true=None,
    has_params=None,
):
    """Return detached target/source FOV overlap for every grouped stain slot.

    Registered targets define only the supervised output region. They never
    participate in affine prediction. Synthetic supervision uses untouched
    target support with the one-pass composed matrix; real supervision warps
    moving support normally. Both remove newly exposed borders from image loss.
    """
    if moving_group.shape != target_group.shape:
        raise ValueError(
            "moving_group and target_group must have identical shapes; got "
            f"{tuple(moving_group.shape)} and {tuple(target_group.shape)}"
        )
    if (params_true is None) != (has_params is None):
        raise ValueError("params_true and has_params must be provided together")
    if params_true is None:
        sources = moving_group
        matrices = affine_parameters_to_matrix(params)
    else:
        sources, matrices = supervision_source_and_matrices(
            moving_group,
            target_group,
            params,
            params_true,
            has_params,
        )

    batch_size, slots, _, height, width = moving_group.shape
    source_support = _rectangular_signal_support(sources, valid_group)
    target_support = _rectangular_signal_support(target_group, valid_group)
    matrices = matrices[:, None].expand(batch_size, slots, 2, 3)
    warped_source_support = apply_affine_transform(
        source_support.reshape(batch_size * slots, 1, height, width),
        matrices.reshape(batch_size * slots, 2, 3),
        mode="nearest",
    ).reshape(batch_size, slots, 1, height, width)
    return (warped_source_support * target_support).clamp(0.0, 1.0).detach()


def parameter_supervision_mask(params, params_true, has_params):
    """Validate and return the per-sample affine-label availability mask."""
    if params.ndim != 2 or params.shape[1] != 5:
        raise ValueError(f"params must have shape Bx5, got {tuple(params.shape)}")
    batch_size = params.shape[0]
    if tuple(params_true.shape) != (batch_size, 5):
        raise ValueError(
            f"params_true must have shape {(batch_size, 5)}, got "
            f"{tuple(params_true.shape)}"
        )
    mask = has_params.reshape(-1).bool()
    if mask.numel() != batch_size:
        raise ValueError(
            f"has_params must contain {batch_size} values, got {mask.numel()}"
        )
    if mask.any() and not torch.isfinite(params_true[mask]).all():
        raise ValueError("Every has_params=True sample must provide finite params_true")
    return mask


def grouped_loss(
    a,
    *,
    params,
    warped_group,
    target_group,
    valid_group,
    group_id,
    params_true,
    has_params,
    valid_overlap,
):
    """Compare warped moving stains with train-only registered targets."""
    valid = valid_group.bool()
    target = target_group
    warped = warped_group
    batch_size, slots, channels, height, width = warped.shape
    if target.shape != warped.shape:
        raise ValueError(
            f"warped_group and target_group must match: {warped.shape} vs {target.shape}"
        )
    flat_valid = valid.reshape(-1)
    if not flat_valid.any():
        raise RuntimeError("Batch contains no valid group members")
    expected_weight_shape = (batch_size, slots, 1, height, width)
    if tuple(valid_overlap.shape) != expected_weight_shape:
        raise ValueError(
            f"valid_overlap must have shape {expected_weight_shape}, got "
            f"{tuple(valid_overlap.shape)}"
        )

    warped_valid = warped.reshape(batch_size * slots, channels, height, width)[
        flat_valid
    ]
    target_valid = target.reshape(batch_size * slots, channels, height, width)[
        flat_valid
    ]
    overlap_valid = valid_overlap.reshape(batch_size * slots, 1, height, width)[
        flat_valid
    ]
    group_valid = group_id[:, None].expand(batch_size, slots).reshape(-1)[flat_valid]

    losses = {}
    if a.ncc_weight:
        losses["ncc"] = multiscale_local_ncc_loss(
            warped_valid, target_valid, weight=overlap_valid
        )
    if a.edge_weight:
        losses["edge"] = gradient_ncc_loss(
            warped_valid, target_valid, weight=overlap_valid
        )
    if a.charbonnier_weight:
        losses["charbonnier"] = charbonnier_loss(
            warped_valid, target_valid, weight=overlap_valid
        )

    sparse_mask = (group_valid == 2) | (group_valid == 4) | (group_valid == 5)
    if a.gradient_weight and sparse_mask.any():
        losses["gradient"] = multiscale_gradient_loss(
            warped_valid[sparse_mask],
            target_valid[sparse_mask],
            weight=overlap_valid[sparse_mask],
        )
    if a.overlap_weight and sparse_mask.any():
        losses["overlap"] = soft_foreground_dice_loss(
            warped_valid[sparse_mask],
            target_valid[sparse_mask],
            weight=overlap_valid[sparse_mask],
        )

    has_parameters = parameter_supervision_mask(params, params_true, has_params)
    if a.param_weight and has_parameters.any():
        losses["param"] = affine_control_point_loss(
            params[has_parameters], params_true[has_parameters]
        )
    if a.reg_weight and (~has_parameters).any():
        losses["reg"] = regularisation_loss(params[~has_parameters])

    weights = {
        "ncc": a.ncc_weight,
        "edge": a.edge_weight,
        "charbonnier": a.charbonnier_weight,
        "gradient": a.gradient_weight,
        "overlap": a.overlap_weight,
        "param": a.param_weight,
        "reg": a.reg_weight,
    }
    if not losses:
        raise ValueError("At least one applicable loss weight must be non-zero")
    total = torch.stack([weights[name] * value for name, value in losses.items()]).sum()
    losses["total"] = total
    return total, losses


def evaluate_path(a, model, loader, device, *, path_name, teacher=False):
    """Evaluate one path using only the inputs available to that path."""
    if teacher and not a.use_teacher_branch:
        raise ValueError("Teacher evaluation requested while its branch is disabled")
    model.eval()
    sums = {}
    metric_counts = {}
    affine_metric_sums = {name: 0.0 for name in AFFINE_ERROR_METRIC_NAMES}
    affine_metric_count = 0
    group_sums = {group_id: 0.0 for group_id in range(1, 6)}
    group_counts = {group_id: 0 for group_id in range(1, 6)}

    with torch.no_grad():
        for batch in loader:
            batch = move_required_tensors(batch, device)
            if teacher:
                params = model.forward_teacher(
                    fixed_mineral=batch["fixed_mineral"],
                    target_group=batch["target_group"],
                    moving_group=batch["moving_group"],
                    group=batch["group_id"],
                )
            else:
                params = model(
                    fixed_mineral=batch["fixed_mineral"],
                    moving_group=batch["moving_group"],
                    group=batch["group_id"],
                )
            warped = warp_group_for_supervision(
                batch["moving_group"],
                batch["target_group"],
                params,
                batch["params_true"],
                batch["has_params"],
            )
            valid_overlap = build_group_valid_overlap(
                batch["moving_group"],
                batch["target_group"],
                batch["valid_group"],
                params,
                params_true=batch["params_true"],
                has_params=batch["has_params"],
            )
            _, losses = grouped_loss(
                a,
                params=params,
                warped_group=warped,
                target_group=batch["target_group"],
                valid_group=batch["valid_group"],
                group_id=batch["group_id"],
                params_true=batch["params_true"],
                has_params=batch["has_params"],
                valid_overlap=valid_overlap,
            )
            batch_size = params.shape[0]
            has_parameters = parameter_supervision_mask(
                params, batch["params_true"], batch["has_params"]
            )
            labeled_count = int(has_parameters.sum())
            unlabeled_count = batch_size - labeled_count
            for name, value in losses.items():
                metric_name = f"val_{path_name}_{name}"
                if name == "param":
                    metric_count = labeled_count
                elif name == "reg":
                    metric_count = unlabeled_count
                else:
                    metric_count = batch_size
                if metric_count == 0:
                    continue
                sums[metric_name] = (
                    sums.get(metric_name, 0.0) + float(value) * metric_count
                )
                metric_counts[metric_name] = (
                    metric_counts.get(metric_name, 0) + metric_count
                )
            if labeled_count:
                height, width = batch["moving_group"].shape[-2:]
                affine_metrics = affine_error_metrics(
                    params[has_parameters],
                    batch["params_true"][has_parameters],
                    height,
                    width,
                )
                for name, value in affine_metrics.items():
                    affine_metric_sums[name] += float(value) * labeled_count
                affine_metric_count += labeled_count

            for group_id in range(1, 6):
                item_mask = batch["group_id"] == group_id
                if not item_mask.any():
                    continue
                group_warped = warped[item_mask]
                group_target = batch["target_group"][item_mask]
                group_valid = batch["valid_group"][item_mask].reshape(-1)
                _, slots, channels, height, width = group_warped.shape
                warped_valid = group_warped.reshape(-1, channels, height, width)[
                    group_valid
                ]
                target_valid = group_target.reshape(-1, channels, height, width)[
                    group_valid
                ]
                overlap_valid = valid_overlap[item_mask].reshape(-1, 1, height, width)[
                    group_valid
                ]
                if warped_valid.shape[0] == 0:
                    continue
                value = multiscale_local_ncc_loss(
                    warped_valid, target_valid, weight=overlap_valid
                )
                count = int(item_mask.sum())
                group_sums[group_id] += float(value) * count
                group_counts[group_id] += count

    metrics = {name: value / metric_counts[name] for name, value in sums.items()}
    if affine_metric_count:
        metrics.update(
            {
                f"val_{path_name}_{name}": value / affine_metric_count
                for name, value in affine_metric_sums.items()
            }
        )
    for group_id in range(1, 6):
        if group_counts[group_id]:
            metrics[f"val_{path_name}_group{group_id}_ncc"] = (
                group_sums[group_id] / group_counts[group_id]
            )
    return metrics


def evaluate(a, model, loader, device):
    """Report student and optional teacher validation metrics separately."""
    metrics = evaluate_path(
        a, model, loader, device, path_name="student", teacher=False
    )
    if a.use_teacher_branch:
        metrics.update(
            evaluate_path(a, model, loader, device, path_name="teacher", teacher=True)
        )
    return metrics


def _require_matching_config(name, saved, expected):
    if expected is None or saved == expected:
        return
    saved = saved or {}
    changed = sorted(
        key for key in set(saved) | set(expected) if saved.get(key) != expected.get(key)
    )

    raise ValueError(
        f"Correlation checkpoint {name} differs from this run: " + ", ".join(changed)
    )


def load_initial_weights(
    model,
    checkpoint_path,
    device,
    expected_student_model_config=None,
    expected_preprocess_config=None,
    expected_use_teacher_branch=None,
):
    """Strictly restore a full teacher/student training checkpoint."""
    del device
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    architecture = checkpoint.get("architecture")
    if architecture != ARCHITECTURE:
        raise ValueError(
            "--resume_checkpoint must be a full TeacherStudent checkpoint with "
            f"architecture='{ARCHITECTURE}', got {architecture!r}"
        )
    if checkpoint.get("checkpoint_type") != "full_training":
        raise ValueError(
            "--resume_checkpoint requires the full training checkpoint, not the "
            "student-only inference artifact"
        )

    saved_teacher_presence = bool(checkpoint.get("use_teacher_branch", False))
    saved_teacher_input_version = (checkpoint.get("model_config") or {}).get(
        "teacher_fixed_input_version"
    )
    if saved_teacher_presence and saved_teacher_input_version is None:
        warnings.warn(
            "This checkpoint predates the Mineral-aware teacher input contract. "
            "Its state dict is compatible, but an unfrozen teacher warmup is "
            "recommended before --freeze_teacher, especially for raw/hybrid "
            "frontends.",
            UserWarning,
            stacklevel=2,
        )
    elif (
        saved_teacher_presence
        and saved_teacher_input_version != TEACHER_FIXED_INPUT_VERSION
    ):
        raise ValueError(
            "Teacher fixed-input contract differs between the resume checkpoint "
            f"and this code: {saved_teacher_input_version!r} vs "
            f"{TEACHER_FIXED_INPUT_VERSION!r}"
        )
    if (
        expected_use_teacher_branch is not None
        and saved_teacher_presence != expected_use_teacher_branch
    ):
        raise ValueError(
            "Teacher-branch presence differs between the resume checkpoint and "
            "this run; retain the Stage 1 --use_teacher_branch setting"
        )
    saved_student_model_config = canonicalize_model_config(
        checkpoint.get("student_model_config")
    )
    canonical_expected_student_model_config = (
        None
        if expected_student_model_config is None
        else canonicalize_model_config(expected_student_model_config)
    )
    _require_matching_config(
        "student_model_config",
        saved_student_model_config,
        canonical_expected_student_model_config,
    )
    _require_matching_config(
        "preprocess_config",
        checkpoint.get("preprocess_config"),
        expected_preprocess_config,
    )
    if "student_model_state_dict" not in checkpoint:
        raise ValueError("Full checkpoint is missing student_model_state_dict")
    if saved_teacher_presence and "teacher_model_state_dict" not in checkpoint:
        raise ValueError("Full teacher checkpoint is missing teacher_model_state_dict")

    student_state = OrderedDict(
        (key[7:] if key.startswith("module.") else key, value)
        for key, value in checkpoint["student_model_state_dict"].items()
    )
    try:
        _branch_module(model.student).load_state_dict(student_state, strict=True)
        if saved_teacher_presence:
            teacher_state = OrderedDict(
                (key[7:] if key.startswith("module.") else key, value)
                for key, value in checkpoint["teacher_model_state_dict"].items()
            )
            _branch_module(model.teacher).load_state_dict(teacher_state, strict=True)
    except RuntimeError as error:
        raise RuntimeError(
            "Teacher/student checkpoint mismatch. Stage 2 must retain teacher "
            "presence and every student architecture/descriptor flag from Stage 1."
        ) from error
    return checkpoint


def validate_args(a):
    if a.frontend_mode not in FRONTEND_MODES:
        raise ValueError(f"frontend_mode must be one of {', '.join(FRONTEND_MODES)}")
    if a.group_input_mode not in GROUP_INPUT_MODES:
        raise ValueError(
            f"group_input_mode must be one of {', '.join(GROUP_INPUT_MODES)}"
        )
    if a.affine_head_mode not in AFFINE_HEAD_MODES:
        raise ValueError(
            f"affine_head_mode must be one of {', '.join(AFFINE_HEAD_MODES)}"
        )
    if MAX_GROUP_STAINS != DEFAULT_GROUP_SLOTS:
        raise ValueError(
            "Dataset/model group-slot contract differs: "
            f"{MAX_GROUP_STAINS} vs {DEFAULT_GROUP_SLOTS}"
        )
    if a.image_mode != "rgb":
        raise ValueError(
            "Deployable Correlation_Vol_Net requires raw RGB/HSV three-channel input"
        )
    if not a.force_group1_identity:
        raise ValueError(
            "Grouped Correlation_Vol_Net requires force_group1_identity=True; "
            "AC and Calcein must not move relative to Mineral"
        )
    if not 0.0 < a.val_split < 1.0:
        raise ValueError("val_split must be strictly between 0 and 1")
    if a.epochs < 1:
        raise ValueError("epochs must be positive")
    if a.grad_clip <= 0.0:
        raise ValueError("grad_clip must be positive")
    if a.max_train_items < 0 or a.max_val_items < 0:
        raise ValueError("max_train_items and max_val_items cannot be negative")
    if a.encoder_base_channels < 8:
        raise ValueError("encoder_base_channels must be at least 8")
    if a.feature_width < 8 or a.cost_hidden_channels < 8:
        raise ValueError("feature_width and cost_hidden_channels must be at least 8")
    if not a.cost_volume_radii:
        raise ValueError("cost_volume_radii cannot be empty")
    if any(radius < 1 or radius > 8 for radius in a.cost_volume_radii):
        raise ValueError("Every cost-volume radius must be in [1, 8]")
    if a.encoder_depth < len(a.cost_volume_radii) + 2:
        raise ValueError("encoder_depth must be at least len(cost_volume_radii) + 2")
    if a.cost_pool_size < 1:
        raise ValueError("cost_pool_size must be positive")
    if a.correlation_temperature <= 0.0:
        raise ValueError("correlation_temperature must be positive")
    if not 0.0 < a.model_scale_range[0] < 1.0 < a.model_scale_range[1]:
        raise ValueError("model_scale_range must strictly contain identity scale 1")
    if (
        a.structural_foreground_threshold is not None
        and not 0.0 <= a.structural_foreground_threshold <= 1.0
    ):
        raise ValueError("structural_foreground_threshold must be in [0, 1]")
    if a.structural_distance_scale <= 0.0:
        raise ValueError("structural_distance_scale must be positive")
    if a.structural_context_scale <= 0.0:
        raise ValueError("structural_context_scale must be positive")
    if a.structural_skeleton_radius < 1:
        raise ValueError("structural_skeleton_radius must be positive")
    if not 0.0 <= a.synthetic_prob <= 1.0:
        raise ValueError("synthetic_prob must be in [0, 1]")
    if not 0.0 <= a.val_synthetic_prob <= 1.0:
        raise ValueError("val_synthetic_prob must be in [0, 1]")
    if a.teacher_distill_weight < 0.0:
        raise ValueError("teacher_distill_weight cannot be negative")
    if a.teacher_warmup_epochs < 0:
        raise ValueError("teacher_warmup_epochs cannot be negative")
    if a.teacher_warmup_epochs:
        # Warmup accepts all-real, mixed, and all-synthetic loaders. Per-sample
        # label integrity is enforced by parameter_supervision_mask at runtime.
        if not a.use_teacher_branch:
            raise ValueError("teacher_warmup_epochs requires --use_teacher_branch")
        if a.freeze_teacher:
            raise ValueError(
                "--freeze_teacher conflicts with teacher-only warmup; set "
                "--teacher_warmup_epochs 0 for frozen-teacher Stage 2"
            )
    if a.freeze_teacher:
        if not a.use_teacher_branch:
            raise ValueError("--freeze_teacher requires --use_teacher_branch")
        if not a.resume_checkpoint:
            raise ValueError(
                "--freeze_teacher requires a full pretrained " "--resume_checkpoint"
            )
    if (
        a.use_teacher_branch
        and (a.detach_teacher or a.freeze_teacher)
        and (a.synthetic_prob == 0.0 or a.param_weight == 0.0)
        and a.teacher_warmup_epochs == 0
        and not a.resume_checkpoint
    ):
        raise ValueError(
            "A detached teacher with neither parameter supervision nor an "
            "image-supervised warmup must resume a full checkpoint containing "
            "a trained teacher"
        )
    loss_weights = (
        a.param_weight,
        a.ncc_weight,
        a.edge_weight,
        a.charbonnier_weight,
        a.gradient_weight,
        a.overlap_weight,
        a.reg_weight,
    )
    if any(weight < 0 for weight in loss_weights):
        raise ValueError("Loss weights cannot be negative")
    for name in (a.best_checkpoint_name, a.last_checkpoint_name):
        if os.path.basename(name) != name or not name.endswith(".pt"):
            raise ValueError(
                "Checkpoint names must be .pt basenames without directories"
            )
    if a.best_checkpoint_name == a.last_checkpoint_name:
        raise ValueError("Best and last checkpoint names must differ")
    max_tx = max(abs(x) for x in a.tx_range) / max(a.width / 2.0, 1.0)
    max_ty = max(abs(y) for y in a.ty_range) / max(a.height / 2.0, 1.0)
    if max(max_tx, max_ty) > a.translation_limit + 1e-8:
        raise ValueError(
            "Synthetic translation range exceeds model capacity: "
            f"required normalized limit={max(max_tx, max_ty):.3f}, "
            f"model limit={a.translation_limit:.3f}"
        )
    if max(abs(r) for r in a.rot_range) > a.max_rotation_deg:
        raise ValueError("Synthetic rotation range exceeds model capacity")
    if min(a.scale_range) < min(a.model_scale_range) or max(a.scale_range) > max(
        a.model_scale_range
    ):
        raise ValueError("Synthetic scale range exceeds model capacity")
    if (
        a.force_group1_identity
        and a.include_group1
        and max(a.synthetic_prob, a.val_synthetic_prob) > 0
    ):
        raise ValueError(
            "Cannot force Group 1 identity while applying synthetic transforms to it"
        )


def _branch_module(branch):
    """Return a branch without a DataParallel container."""
    if isinstance(branch, torch.nn.DataParallel):
        return branch.module
    return branch


def _branch_state_dict(branch):
    return _branch_module(branch).state_dict()


def _student_checkpoint_name(full_checkpoint_name):
    stem, extension = os.path.splitext(full_checkpoint_name)
    return f"{stem}_student{extension}"


def save_checkpoint_pair(
    *,
    output_dir,
    checkpoint_name,
    full_payload,
    model,
    student_model_config,
    preprocess_config,
):
    """Save full resume state and a teacher-free deployable student artifact."""
    student_model_config = canonicalize_model_config(student_model_config)
    student_state = _branch_state_dict(model.student)
    teacher_state = (
        _branch_state_dict(model.teacher) if model.teacher is not None else None
    )
    full_payload = dict(full_payload)
    full_model_config = dict(full_payload.get("model_config") or {})
    full_model_config.update(
        {
            "frontend_mode": student_model_config["frontend_mode"],
            "group_input_mode": student_model_config["group_input_mode"],
            "affine_head_mode": student_model_config["affine_head_mode"],
            "student_config": student_model_config,
            "use_teacher_branch": model.teacher is not None,
        }
    )
    if teacher_state is not None:
        full_model_config["teacher_fixed_input_version"] = TEACHER_FIXED_INPUT_VERSION
    full_payload.update(
        {
            "checkpoint_type": "full_training",
            "use_teacher_branch": model.teacher is not None,
            "student_model_state_dict": student_state,
            "student_model_config": student_model_config,
            "model_config": full_model_config,
        }
    )
    if teacher_state is not None:
        full_payload["teacher_model_state_dict"] = teacher_state
    torch.save(full_payload, os.path.join(output_dir, checkpoint_name))

    student_payload = {
        "architecture": ARCHITECTURE,
        "checkpoint_type": "deployable_student",
        "student_model_state_dict": student_state,
        "student_model_config": student_model_config,
        "model_config": dict(student_model_config),
        "preprocess_config": preprocess_config,
        "epoch": full_payload["epoch"],
        "metrics": full_payload["metrics"],
    }
    torch.save(
        student_payload,
        os.path.join(output_dir, _student_checkpoint_name(checkpoint_name)),
    )


def teacher_warmup_active(a, epoch):
    """Return whether this epoch is the teacher-only optimization phase."""
    return bool(a.use_teacher_branch and epoch <= a.teacher_warmup_epochs)


def configure_training_phase(a, model, epoch):
    """Set branch gradients and modes for warmup or student training.

    ``model.train()`` alone would also train BatchNorm buffers in a frozen
    branch. The explicit branch modes below make "not updated" include both
    parameters and running state.
    """
    model.train()
    teacher_only = teacher_warmup_active(a, epoch)
    model.student.train(not teacher_only)
    for parameter in model.student.parameters():
        parameter.requires_grad_(not teacher_only)

    if model.teacher is not None:
        teacher_trainable = not a.freeze_teacher
        model.teacher.train(teacher_trainable)
        for parameter in model.teacher.parameters():
            parameter.requires_grad_(teacher_trainable)
    elif teacher_only or a.freeze_teacher:
        raise RuntimeError("The requested training phase requires a teacher branch")
    return "teacher_warmup" if teacher_only else "student"


def teacher_distillation_weight(a, epoch):
    """Return the scheduled control-point distillation weight for this epoch."""
    if not a.use_teacher_branch or teacher_warmup_active(a, epoch):
        return 0.0
    return float(a.teacher_distill_weight)


def compute_training_loss(a, model, batch, epoch):
    """Compute the phase-correct student/teacher training objective."""
    teacher_only = teacher_warmup_active(a, epoch)
    student_context = torch.no_grad() if teacher_only else nullcontext()
    with student_context:
        student_params = model(
            fixed_mineral=batch["fixed_mineral"],
            moving_group=batch["moving_group"],
            group=batch["group_id"],
        )

    components = {}
    total = None
    if not teacher_only:
        student_warped = warp_group_for_supervision(
            batch["moving_group"],
            batch["target_group"],
            student_params,
            batch["params_true"],
            batch["has_params"],
        )
        student_overlap = build_group_valid_overlap(
            batch["moving_group"],
            batch["target_group"],
            batch["valid_group"],
            student_params,
            params_true=batch["params_true"],
            has_params=batch["has_params"],
        )
        student_total, student_components = grouped_loss(
            a,
            params=student_params,
            warped_group=student_warped,
            target_group=batch["target_group"],
            valid_group=batch["valid_group"],
            group_id=batch["group_id"],
            params_true=batch["params_true"],
            has_params=batch["has_params"],
            valid_overlap=student_overlap,
        )
        components.update(
            {f"student_{name}": value for name, value in student_components.items()}
        )
        total = student_total

    has_parameters = parameter_supervision_mask(
        student_params,
        batch["params_true"],
        batch["has_params"],
    )
    if has_parameters.any():
        height, width = batch["moving_group"].shape[-2:]
        student_affine_metrics = affine_error_metrics(
            student_params[has_parameters],
            batch["params_true"][has_parameters],
            height,
            width,
        )
        components.update(
            {f"student_{name}": value for name, value in student_affine_metrics.items()}
        )

    if a.use_teacher_branch:
        teacher_has_supervision = bool(a.param_weight and has_parameters.any())
        teacher_without_graph = a.freeze_teacher or (
            not teacher_only and a.detach_teacher and not teacher_has_supervision
        )
        teacher_context = torch.no_grad() if teacher_without_graph else nullcontext()
        with teacher_context:
            teacher_params = model.forward_teacher(
                fixed_mineral=batch["fixed_mineral"],
                target_group=batch["target_group"],
                moving_group=batch["moving_group"],
                group=batch["group_id"],
            )

        if teacher_only:
            teacher_warped = warp_group_for_supervision(
                batch["moving_group"],
                batch["target_group"],
                teacher_params,
                batch["params_true"],
                batch["has_params"],
            )
            teacher_overlap = build_group_valid_overlap(
                batch["moving_group"],
                batch["target_group"],
                batch["valid_group"],
                teacher_params,
                params_true=batch["params_true"],
                has_params=batch["has_params"],
            )
            teacher_total, teacher_components = grouped_loss(
                a,
                params=teacher_params,
                warped_group=teacher_warped,
                target_group=batch["target_group"],
                valid_group=batch["valid_group"],
                group_id=batch["group_id"],
                params_true=batch["params_true"],
                has_params=batch["has_params"],
                valid_overlap=teacher_overlap,
            )
            components.update(
                {f"teacher_{name}": value for name, value in teacher_components.items()}
            )
            total = teacher_total
        elif teacher_has_supervision:
            teacher_param = affine_control_point_loss(
                teacher_params[has_parameters], batch["params_true"][has_parameters]
            )
            teacher_total = a.param_weight * teacher_param
            components["teacher_param"] = teacher_param
            components["teacher_total"] = teacher_total
            if not a.freeze_teacher:
                total = teacher_total if total is None else total + teacher_total

        if has_parameters.any():
            teacher_affine_metrics = affine_error_metrics(
                teacher_params[has_parameters],
                batch["params_true"][has_parameters],
                height,
                width,
            )
            components.update(
                {
                    f"teacher_{name}": value
                    for name, value in teacher_affine_metrics.items()
                }
            )

        if not teacher_only:
            detach_target = a.detach_teacher or a.freeze_teacher
            teacher_target = (
                teacher_params.detach() if detach_target else teacher_params
            )
            distillation = affine_control_point_loss(student_params, teacher_target)
            components["teacher_distill"] = distillation
            distill_weight = teacher_distillation_weight(a, epoch)
            if distill_weight:
                total = total + distill_weight * distillation

    if total is None or not total.requires_grad:
        raise RuntimeError(
            "This training phase has no differentiable objective. Configure at "
            "least one applicable image, parameter, or regularization loss."
        )
    components["total"] = total
    return total, components, student_params


def save_validation_overlays(
    a,
    model,
    loader,
    validation_dataset,
    validation_indices,
    device,
):
    """Save best-checkpoint model-space overlays for every validation item."""
    paths = ["student"] + (["teacher"] if a.use_teacher_branch else [])
    roots = {
        name: os.path.join(a.output_dir, "validation_overlays", name) for name in paths
    }
    for root in roots.values():
        os.makedirs(root, exist_ok=True)

    model.eval()
    item_offset = 0
    with torch.no_grad():
        for batch in loader:
            batch = move_required_tensors(batch, device)
            predictions = {
                "student": model(
                    fixed_mineral=batch["fixed_mineral"],
                    moving_group=batch["moving_group"],
                    group=batch["group_id"],
                )
            }
            if a.use_teacher_branch:
                predictions["teacher"] = model.forward_teacher(
                    fixed_mineral=batch["fixed_mineral"],
                    target_group=batch["target_group"],
                    moving_group=batch["moving_group"],
                    group=batch["group_id"],
                )
            warped_predictions = {
                name: warp_model_space_group(batch["moving_group"], params)
                for name, params in predictions.items()
            }

            batch_size = batch["fixed_mineral"].shape[0]
            for local_index in range(batch_size):
                dataset_index = validation_indices[item_offset + local_index]
                base_id, metadata_group_id, _ = validation_dataset.items[dataset_index]
                group_id = int(batch["group_id"][local_index])
                if group_id != metadata_group_id:
                    raise AssertionError(
                        "Validation loader order no longer matches items"
                    )
                safe_base = re.sub(r"[^A-Za-z0-9_.-]+", "_", str(base_id))
                filename = (
                    f"{item_offset + local_index:05d}_{safe_base}_G{group_id}.png"
                )
                for path_name, warped in warped_predictions.items():
                    _save_group_overlay(
                        os.path.join(roots[path_name], filename),
                        batch["fixed_mineral"][local_index],
                        warped[local_index],
                        batch["valid_group"][local_index],
                        group_id,
                        a.sfo_mode,
                    )
            item_offset += batch_size
    if item_offset != len(validation_indices):
        raise AssertionError("Validation overlay count does not match split indices")


def main(a):
    validate_args(a)
    preprocess_config = build_preprocess_config(a)
    set_seed(a.seed)
    os.makedirs(a.output_dir, exist_ok=True)
    device, gpu_ids = resolve_device(a.device, a.gpu_ids)
    training_dataset = build_dataset(a, False)
    validation_dataset = build_dataset(a, True)
    (
        training_indices,
        validation_indices,
        training_sample_ids,
        validation_sample_ids,
    ) = split_by_sample(training_dataset, validation_dataset, a.val_split, a.split_seed)
    if a.max_train_items:
        training_indices = training_indices[: a.max_train_items]
    if a.max_val_items:
        validation_indices = validation_indices[: a.max_val_items]
    if not training_indices or not validation_indices:
        raise RuntimeError("Debug item caps produced an empty partition")
    print(
        f"Split: {len(training_sample_ids)} train samples/{len(training_indices)} items, "
        f"{len(validation_sample_ids)} validation samples/{len(validation_indices)} items"
    )
    training_loader = DataLoader(
        Subset(training_dataset, training_indices),
        batch_size=a.batch_size,
        shuffle=True,
        num_workers=a.n_workers,
        pin_memory=device.type == "cuda",
        collate_fn=safe_collate,
        persistent_workers=a.n_workers > 0,
    )
    validation_loader = DataLoader(
        Subset(validation_dataset, validation_indices),
        batch_size=a.batch_size,
        shuffle=False,
        num_workers=a.n_workers,
        pin_memory=device.type == "cuda",
        collate_fn=safe_collate,
        persistent_workers=a.n_workers > 0,
    )

    student_model_config = canonicalize_model_config(
        dict(
            input_channels=3,
            frontend_mode=a.frontend_mode,
            group_input_mode=a.group_input_mode,
            affine_head_mode=a.affine_head_mode,
            group_slots=MAX_GROUP_STAINS,
            sfo_mode=a.sfo_mode,
            structural_channels=6,
            structural_descriptor_version=STRUCTURAL_DESCRIPTOR_VERSION,
            structural_foreground_threshold=a.structural_foreground_threshold,
            structural_distance_scale=a.structural_distance_scale,
            structural_context_scale=a.structural_context_scale,
            structural_skeleton_radius=a.structural_skeleton_radius,
            latent_dim=a.latent_dim,
            group_embedding_dim=a.group_embedding_dim,
            use_group_embedding=a.use_group_embedding,
            scale_range=tuple(a.model_scale_range),
            translation_limit=a.translation_limit,
            max_rotation_degrees=a.max_rotation_deg,
            encoder_base_channels=a.encoder_base_channels,
            encoder_depth=a.encoder_depth,
            feature_width=a.feature_width,
            cost_hidden_channels=a.cost_hidden_channels,
            cost_volume_radii=tuple(a.cost_volume_radii),
            cost_pool_size=a.cost_pool_size,
            correlation_temperature=a.correlation_temperature,
            norm_type=a.norm_type,
            force_group1_identity=a.force_group1_identity,
            separate_group_heads=a.separate_group_heads,
            separate_group_adapters=a.separate_group_adapters,
        )
    )
    model = TeacherStudentAffineRegistrationModel(
        student_config=student_model_config,
        use_teacher_branch=a.use_teacher_branch,
    ).to(device)
    if a.resume_checkpoint:
        load_initial_weights(
            model,
            a.resume_checkpoint,
            device,
            expected_student_model_config=student_model_config,
            expected_preprocess_config=preprocess_config,
            expected_use_teacher_branch=a.use_teacher_branch,
        )

    if device.type == "cuda" and not a.no_multi_gpu and len(gpu_ids) > 1:
        model.student = torch.nn.DataParallel(model.student, device_ids=gpu_ids)
        if model.teacher is not None:
            model.teacher = torch.nn.DataParallel(model.teacher, device_ids=gpu_ids)

    # Apply permanent teacher freezing before the optimizer is first used.
    # Student parameters remain in the optimizer so they can be enabled after
    # a teacher-only warmup without rebuilding optimizer state.
    configure_training_phase(a, model, epoch=1)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=a.lr, weight_decay=a.weight_decay
    )
    student_training_epochs = max(a.epochs - min(a.teacher_warmup_epochs, a.epochs), 1)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=student_training_epochs, eta_min=a.lr * 0.05
    )
    scaler = torch.amp.GradScaler(
        "cuda",
        enabled=a.amp and device.type == "cuda",
        init_scale=4096.0,
    )
    best = float("inf")
    best_selection_phase = None
    use_wandb = bool(a.wandb_project)
    if use_wandb:
        import wandb

        wandb.init(project=a.wandb_project, name=a.wandb_run_name, config=vars(a))

    for epoch in range(1, a.epochs + 1):
        training_phase = configure_training_phase(a, model, epoch)
        sums = {}
        metric_counts = {}
        optimizer_steps = 0
        skipped_steps = 0
        for batch in tqdm(
            training_loader, desc=f"Epoch {epoch}/{a.epochs}", leave=False
        ):
            batch = move_required_tensors(batch, device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            with torch.amp.autocast(
                device_type=device.type, enabled=a.amp and device.type == "cuda"
            ):
                total_loss, losses, student_params = compute_training_loss(
                    a, model, batch, epoch
                )
            if not bool(torch.isfinite(total_loss)):
                raise FloatingPointError("Encountered a non-finite training loss")
            scaler.scale(total_loss).backward()
            scaler.unscale_(optimizer)
            gradient_norm = torch.nn.utils.clip_grad_norm_(
                model.parameters(), a.grad_clip
            )
            gradients_are_finite = bool(torch.isfinite(gradient_norm))
            if scaler.is_enabled():
                scaler.step(optimizer)
                scaler.update()
                if gradients_are_finite:
                    optimizer_steps += 1
                else:
                    skipped_steps += 1
            else:
                if not gradients_are_finite:
                    raise FloatingPointError("Encountered non-finite gradients")
                optimizer.step()
                optimizer_steps += 1

            batch_size = student_params.shape[0]
            labeled_count = int(batch["has_params"].reshape(-1).bool().sum())
            unlabeled_count = batch_size - labeled_count
            for name, value in losses.items():
                metric_name = f"train_{name}"
                if (
                    name in AFFINE_ERROR_COMPONENT_NAMES
                    or name.endswith("_param")
                    or (name == "teacher_total" and training_phase == "student")
                ):
                    metric_count = labeled_count
                elif name.endswith("_reg"):
                    metric_count = unlabeled_count
                else:
                    metric_count = batch_size
                if metric_count == 0:
                    continue
                sums[metric_name] = (
                    sums.get(metric_name, 0.0) + float(value.detach()) * metric_count
                )
                metric_counts[metric_name] = (
                    metric_counts.get(metric_name, 0) + metric_count
                )

        if optimizer_steps == 0:
            raise FloatingPointError(
                "AMP skipped every optimizer step; lower the loss weights or disable AMP"
            )
        # Do not consume the student's cosine schedule while it is frozen.
        if training_phase == "student":
            scheduler.step()
        metrics = {name: value / metric_counts[name] for name, value in sums.items()}
        metrics["train_optimizer_steps"] = optimizer_steps
        metrics["train_skipped_steps"] = skipped_steps
        metrics["teacher_warmup_active"] = float(training_phase == "teacher_warmup")
        metrics["teacher_frozen"] = float(a.freeze_teacher)
        metrics["teacher_distillation_active"] = float(
            teacher_distillation_weight(a, epoch) > 0.0
        )
        metrics["learning_rate"] = float(optimizer.param_groups[0]["lr"])
        metrics.update(evaluate(a, model, validation_loader, device))
        metrics["epoch"] = epoch
        print_keys = {
            "train_total",
            "train_student_param",
            "train_teacher_param",
            "train_teacher_total",
            "train_teacher_distill",
            "val_student_total",
            "val_student_param",
            "val_teacher_total",
            "val_teacher_param",
        }
        print(
            " ".join(
                f"{name}={value:.6f}"
                for name, value in metrics.items()
                if name in print_keys
            )
        )
        if use_wandb:
            import wandb

            wandb.log(metrics)

        full_payload = {
            "architecture": ARCHITECTURE,
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict(),
            "epoch": epoch,
            "metrics": metrics,
            "model_config": {
                "frontend_mode": student_model_config["frontend_mode"],
                "group_input_mode": student_model_config["group_input_mode"],
                "affine_head_mode": student_model_config["affine_head_mode"],
                "student_config": student_model_config,
                "use_teacher_branch": a.use_teacher_branch,
            },
            "train_config": dict(vars(a)),
            "debug_config": {
                "max_train_items": a.max_train_items,
                "max_val_items": a.max_val_items,
            },
            "split_config": {
                "training_sample_ids": sorted(training_sample_ids),
                "validation_sample_ids": sorted(validation_sample_ids),
                "split_seed": a.split_seed,
                "validation_fraction": a.val_split,
            },
            "preprocess_config": preprocess_config,
        }
        save_checkpoint_pair(
            output_dir=a.output_dir,
            checkpoint_name=a.last_checkpoint_name,
            full_payload=full_payload,
            model=model,
            student_model_config=student_model_config,
            preprocess_config=preprocess_config,
        )
        selection_phase = training_phase
        if selection_phase != best_selection_phase:
            best = float("inf")
            best_selection_phase = selection_phase
        if selection_phase == "teacher_warmup":
            selection_metric = metrics.get(
                "val_teacher_total", metrics["train_teacher_total"]
            )
        else:
            selection_metric = metrics.get(
                "val_student_total", metrics["train_student_total"]
            )
        if selection_metric < best:
            best = selection_metric
            save_checkpoint_pair(
                output_dir=a.output_dir,
                checkpoint_name=a.best_checkpoint_name,
                full_payload=full_payload,
                model=model,
                student_model_config=student_model_config,
                preprocess_config=preprocess_config,
            )

    # Reload the selected pair, not the final epoch, before visual review.
    model.student = _branch_module(model.student)
    if model.teacher is not None:
        model.teacher = _branch_module(model.teacher)
    load_initial_weights(
        model,
        os.path.join(a.output_dir, a.best_checkpoint_name),
        device,
        expected_student_model_config=student_model_config,
        expected_preprocess_config=preprocess_config,
        expected_use_teacher_branch=a.use_teacher_branch,
    )
    model.to(device)
    save_validation_overlays(
        a,
        model,
        validation_loader,
        validation_dataset,
        validation_indices,
        device,
    )

    if use_wandb:
        import wandb

        wandb.finish()


if __name__ == "__main__":
    main(parse_args())
