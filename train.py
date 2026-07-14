"""Train one affine transform per stain acquisition group.

STAGE 1: synthetic warm-start with group-specific structural inputs
python train.py \
    --registered_root /home/yec23006/projects/research/Registration/Data/Cartilage/Registered \
    --unregistered_root /home/yec23006/projects/research/Registration/Data/Cartilage/Unregistered \
    --output_dir /home/yec23006/projects/research/Registration/Grouped/ckpt/group_stack_structural_rgb \
    --resume_checkpoint /home/yec23006/projects/research/Registration/Grouped/ckpt/group_stack_fixed/best_model.pt \
    --best_checkpoint_name best_model.pt --last_checkpoint_name last_model.pt \
    --group_input_mode stack --include_group1 \
    --enhance_groups 2 4 5 --enhancement_mode group_specific \
    --trap_enhancement_strength 0.8 --cfo_enhancement_strength 0.5 \
    --sfo_enhancement_strength 1.0 --sfo_hue_range_degrees 70 200 \
    --trap_morphology_kernel_size 3 --trap_morphology_iterations 1 \
    --separate_group_heads --separate_group_adapters --use_group_embedding \
    --height 1024 --width 1024 --image_mode rgb --sfo_mode rgb \
    --crop_mode full --crop_margin 32 \
    --fusion_mode intermediate --depth 5 --base_channels 48 \
    --latent_dim 384 --group_embedding_dim 32 --spatial_pool_size 4 \
    --norm_type group \
    --synthetic_prob 1.0 --val_synthetic_prob 1.0 \
    --tx_range -64 64 --ty_range -64 64 --rot_range -15 15 \
    --scale_range 0.85 1.15 --model_scale_range 0.8 1.2 \
    --translation_limit 0.5 --max_rotation_deg 20 \
    --param_weight 10.0 --ncc_weight 1.0 --edge_weight 0.25 \
    --charbonnier_weight 0.1 --gradient_weight 0.1 \
    --overlap_weight 0.5 --reg_weight 0.0 \
    --epochs 400 --batch_size 8 --lr 0.0003 --weight_decay 0.00001 \
    --grad_clip 1.0 --val_split 0.15 --split_seed 2026 \
    --n_workers 8 --amp --gpu_ids 0,1 \
    --wandb_project registration --wandb_run_name group_stack_affine_stage1_rgb

STAGE 2: real unregistered-data fine-tuning
python train.py \
    --registered_root /home/yec23006/projects/research/Registration/Data/Cartilage/Registered \
    --unregistered_root /home/yec23006/projects/research/Registration/Data/Cartilage/Unregistered \
    --output_dir /home/yec23006/projects/research/Registration/Grouped/ckpt/group_stack_fixed \
    --resume_checkpoint /home/yec23006/projects/research/Registration/Grouped/ckpt/group_stack_structural_rgb/best_model.pt \
    --best_checkpoint_name stage2_best_model.pt \
    --last_checkpoint_name stage2_last_model.pt \
    --group_input_mode stack --include_group1 \
    --enhance_groups 2 4 5 --enhancement_mode group_specific \
    --trap_enhancement_strength 0.8 --cfo_enhancement_strength 0.5 \
    --sfo_enhancement_strength 1.0 --sfo_hue_range_degrees 70 200 \
    --trap_morphology_kernel_size 3 --trap_morphology_iterations 1 \
    --separate_group_heads --separate_group_adapters --use_group_embedding \
    --height 1024 --width 1024 --image_mode rgb --sfo_mode rgb \
    --crop_mode full --crop_margin 32 \
    --fusion_mode intermediate --depth 5 --base_channels 48 \
    --latent_dim 384 --group_embedding_dim 32 --spatial_pool_size 4 \
    --norm_type group \
    --synthetic_prob 0.0 --val_synthetic_prob 0.0 \
    --tx_range -64 64 --ty_range -64 64 --rot_range -15 15 \
    --scale_range 0.85 1.15 --model_scale_range 0.8 1.2 \
    --translation_limit 0.5 --max_rotation_deg 20 \
    --param_weight 0.0 --ncc_weight 1.0 --edge_weight 0.25 \
    --charbonnier_weight 0.1 --gradient_weight 0.1 \
    --overlap_weight 0.5 --reg_weight 0.01 \
    --epochs 400 --batch_size 8 --lr 0.00001 --weight_decay 0.00001 \
    --grad_clip 1.0 --val_split 0.15 --split_seed 2026 \
    --n_workers 8 --amp --gpu_ids 0,1 \
    --wandb_project registration --wandb_run_name group_stack_affine_stage2_rgb


STATE OF THE ART

python train.py \
--registered_root /home/yec23006/projects/research/Registration/Data/Cartilage/Registered \
--unregistered_root /home/yec23006/projects/research/Registration/Data/Cartilage/Unregistered \
--output_dir /home/yec23006/projects/research/Registration/Grouped/ckpt/group_stack_fixed_hsv_1024 \
--group_input_mode stack \
--use_group_embedding --height 1024 --width 1024 --image_mode rgb --sfo_mode hsv \
--crop_mode full --fusion_mode intermediate --depth 6 --base_channels 48 --latent_dim 384 --group_embedding_dim 32 \
--spatial_pool_size 4 --norm_type group \
--synthetic_prob 1.0 --val_synthetic_prob 1.0 --tx_range -64 64 --ty_range -64 64 --rot_range -20 20 --scale_range 0.80 1.20 \
--model_scale_range 0.8 1.2 --translation_limit 0.5 --max_rotation_deg 20 \
--param_weight 10.0 --ncc_weight 1.0 --edge_weight 0.25 --charbonnier_weight 0.1 \
--epochs 400 --batch_size 8 --lr 0.0003 --weight_decay 0.00001 --grad_clip 1.0 \
--n_workers 8 --amp --gpu_ids 0,1 \
--wandb_project registration --wandb_run_name group_stack_affine_hsv_stage1

python /train.py \
    --registered_root /home/yec23006/projects/research/Registration/Data/Cartilage/Registered \
    --unregistered_root /home/yec23006/projects/research/Registration/Data/Cartilage/Unregistered \
    --output_dir /home/yec23006/projects/research/Registration/Grouped/ckpt/group_stack_fixed \
    --resume_checkpoint /home/yec23006/projects/research/Registration/Grouped/ckpt/group_stack_fixed_hsv/best_model.pt \
    --best_checkpoint_name stage2_best_model.pt --last_checkpoint_name stage2_last_model.pt --group_input_mode stack \
    --use_group_embedding \
    --height 1024 --width 1024 \ 
    --image_mode rgb --sfo_mode rgb --sfo_mode hsv \
    --crop_mode full --crop_margin 32 --fusion_mode intermediate --depth 5 --base_channels 48 --latent_dim 384 --group_embedding_dim 32 \
    --spatial_pool_size 4 --norm_type group \
    --synthetic_prob 0.0 --val_synthetic_prob 0.0 \
    --model_scale_range 0.8 1.2 --translation_limit 0.5 --max_rotation_deg 20 --param_weight 0.0 \
    --ncc_weight 1.0 --edge_weight 0.25 --charbonnier_weight 0.1 --reg_weight 0.0 --epochs 400 --batch_size 8 \
    --lr 0.00001 --weight_decay 0.00001 --grad_clip 1.0 --n_workers 8 \
    --amp --gpu_ids 0,1 --wandb_project registration --wandb_run_name group_stack_affine_stage2
"""


from __future__ import annotations

import argparse
import os
import random
from collections import OrderedDict
from typing import Any, Dict, List

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset
from tqdm import tqdm

from dataset import CartilageDataset
from losses import (
    affine_control_point_loss,
    charbonnier_loss,
    gradient_ncc_loss,
    multiscale_gradient_loss,
    multiscale_local_ncc_loss,
    regularisation_loss,
    soft_foreground_dice_loss,
)
from models import GroupAffineRegistrationModel
from utils import (
    affine_parameters_to_matrix,
    apply_affine_transform,
    resolve_device,
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
        "--group_input_mode", choices=["single", "stack", "overlay"], default="stack"
    )
    p.add_argument("--enhance_groups", type=int, nargs="*", default=[2, 4, 5])
    p.add_argument("--enhancement_strength", type=float, default=0.7)
    p.add_argument(
        "--enhancement_mode",
        choices=["legacy", "group_specific"],
        default="group_specific",
    )
    p.add_argument("--trap_enhancement_strength", type=float, default=0.8)
    p.add_argument("--cfo_enhancement_strength", type=float, default=0.5)
    p.add_argument("--sfo_enhancement_strength", type=float, default=1.0)
    p.add_argument(
        "--sfo_hue_range_degrees", type=float, nargs=2, default=[70.0, 200.0]
    )
    p.add_argument("--trap_morphology_kernel_size", type=int, default=3)
    p.add_argument("--trap_morphology_iterations", type=int, default=1)
    p.add_argument("--use_group_embedding", action="store_true")
    p.add_argument(
        "--include_group1", action=argparse.BooleanOptionalAction, default=True
    )
    p.add_argument("--depth", type=int, default=4)
    p.add_argument("--base_channels", type=int, default=32)
    p.add_argument("--latent_dim", type=int, default=256)
    p.add_argument("--group_embedding_dim", type=int, default=32)
    p.add_argument("--spatial_pool_size", type=int, default=4)
    p.add_argument(
        "--norm_type", choices=["group", "batch", "instance"], default="group"
    )
    p.add_argument(
        "--fusion_mode", choices=["concat", "intermediate"], default="intermediate"
    )
    p.add_argument("--disable_coordconv", action="store_true")
    p.add_argument("--force_group1_identity", action="store_true")
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
    p.add_argument("--val_split", type=float, default=0.15)
    p.add_argument("--split_seed", type=int, default=2026)
    p.add_argument("--grad_clip", type=float, default=1.0)
    p.add_argument("--resume_checkpoint")
    p.add_argument("--best_checkpoint_name", default="best_model.pt")
    p.add_argument("--last_checkpoint_name", default="last_model.pt")
    p.add_argument("--wandb_project")
    p.add_argument("--wandb_run_name")
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
        group_input_mode=a.group_input_mode,
        include_group1=a.include_group1,
        enhance_groups=tuple(a.enhance_groups),
        enhancement_strength=a.enhancement_strength,
        enhancement_mode=a.enhancement_mode,
        trap_enhancement_strength=a.trap_enhancement_strength,
        cfo_enhancement_strength=a.cfo_enhancement_strength,
        sfo_enhancement_strength=a.sfo_enhancement_strength,
        sfo_hue_range_degrees=tuple(a.sfo_hue_range_degrees),
        trap_morphology_kernel_size=a.trap_morphology_kernel_size,
        trap_morphology_iterations=a.trap_morphology_iterations,
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


def warp_group(moving, params):
    # moving: B,K,C,H,W ; same matrix repeated for all K members
    batch_size, slots, channels, height, width = moving.shape
    matrices = affine_parameters_to_matrix(params)
    matrices = matrices[:, None].expand(batch_size, slots, 2, 3)
    matrices = matrices.reshape(batch_size * slots, 2, 3)
    flattened = moving.reshape(batch_size * slots, channels, height, width)
    warped = apply_affine_transform(flattened, matrices)
    return warped.reshape(batch_size, slots, channels, height, width)


def grouped_loss(a, params, warped, batch):
    """Compute dense losses plus sparse-signal losses for Groups 2, 4, and 5."""
    valid = batch["valid_group"].bool()
    target = batch["target_group"]
    batch_size, slots, channels, height, width = warped.shape
    flat_valid = valid.reshape(-1)
    if not flat_valid.any():
        raise RuntimeError("Batch contains no valid group members")

    warped_valid = warped.reshape(batch_size * slots, channels, height, width)[
        flat_valid
    ]
    target_valid = target.reshape(batch_size * slots, channels, height, width)[
        flat_valid
    ]
    group_valid = (
        batch["group"][:, None].expand(batch_size, slots).reshape(-1)[flat_valid]
    )

    losses = {}
    if a.ncc_weight:
        losses["ncc"] = multiscale_local_ncc_loss(warped_valid, target_valid)
    if a.edge_weight:
        losses["edge"] = gradient_ncc_loss(warped_valid, target_valid)
    if a.charbonnier_weight:
        losses["charbonnier"] = charbonnier_loss(warped_valid, target_valid)

    sparse_mask = (group_valid == 2) | (group_valid == 4) | (group_valid == 5)
    if a.gradient_weight and sparse_mask.any():
        losses["gradient"] = multiscale_gradient_loss(
            warped_valid[sparse_mask], target_valid[sparse_mask]
        )
    if a.overlap_weight and sparse_mask.any():
        losses["overlap"] = soft_foreground_dice_loss(
            warped_valid[sparse_mask], target_valid[sparse_mask]
        )

    has_parameters = batch["has_params"].reshape(-1).bool()
    if a.param_weight and has_parameters.any():
        losses["param"] = affine_control_point_loss(
            params[has_parameters], batch["params_true"][has_parameters]
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


def evaluate(a, model, loader, device):
    """Evaluate total losses and report alignment separately for every group."""
    model.eval()
    sums = {}
    sample_count = 0
    group_sums = {group_id: 0.0 for group_id in range(1, 6)}
    group_counts = {group_id: 0 for group_id in range(1, 6)}

    with torch.no_grad():
        for batch in loader:
            batch = {
                key: (value.to(device) if torch.is_tensor(value) else value)
                for key, value in batch.items()
            }
            params = model(batch["group_input"], batch["fixed_mineral"], batch["group"])
            warped = warp_group(batch["moving_group"], params)
            _, losses = grouped_loss(a, params, warped, batch)
            batch_size = params.shape[0]
            for name, value in losses.items():
                metric_name = f"val_{name}"
                sums[metric_name] = (
                    sums.get(metric_name, 0.0) + float(value) * batch_size
                )
            sample_count += batch_size

            for group_id in range(1, 6):
                item_mask = batch["group"] == group_id
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
                if warped_valid.shape[0] == 0:
                    continue
                value = multiscale_local_ncc_loss(warped_valid, target_valid)
                count = int(item_mask.sum())
                group_sums[group_id] += float(value) * count
                group_counts[group_id] += count

    metrics = {name: value / max(sample_count, 1) for name, value in sums.items()}
    for group_id in range(1, 6):
        if group_counts[group_id]:
            metrics[f"val_group{group_id}_ncc"] = (
                group_sums[group_id] / group_counts[group_id]
            )
    return metrics


def load_initial_weights(model, checkpoint_path, device):
    """Load a checkpoint, expanding one legacy head into five group heads."""
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    raw_state = checkpoint.get("model_state_dict", checkpoint)
    state = OrderedDict(
        (key[7:] if key.startswith("module.") else key, value)
        for key, value in raw_state.items()
    )

    if model.separate_group_heads and any(key.startswith("head.") for key in state):
        upgraded_state = OrderedDict()
        for key, value in state.items():
            if key.startswith("head."):
                suffix = key[len("head.") :]
                for head_index in range(model.num_groups):
                    upgraded_state[f"heads.{head_index}.{suffix}"] = value.clone()
            else:
                upgraded_state[key] = value
        state = upgraded_state
        print("Initialized all five group heads from the legacy shared head")

    if model.group_adapters is not None and not any(
        key.startswith("group_adapters.") for key in state
    ):
        for key, value in model.state_dict().items():
            if key.startswith("group_adapters."):
                state[key] = value
        print("Initialized five new group adapters as identity mappings")

    model.load_state_dict(state, strict=True)
    return checkpoint


def validate_args(a):
    if not 0.0 < a.val_split < 1.0:
        raise ValueError("val_split must be strictly between 0 and 1")
    if a.grad_clip <= 0.0:
        raise ValueError("grad_clip must be positive")
    if a.image_mode != "rgb" and a.enhance_groups:
        raise ValueError("Group enhancement requires image_mode=rgb")
    if any(group < 1 or group > 5 for group in a.enhance_groups):
        raise ValueError("enhance_groups must contain IDs from 1 to 5")
    if (
        a.enhancement_mode == "group_specific"
        and 5 in a.enhance_groups
        and a.sfo_mode not in {"rgb", "hsv"}
    ):
        raise ValueError("Group-specific SFO selection requires --sfo_mode rgb or hsv")
    strengths = (
        a.enhancement_strength,
        a.trap_enhancement_strength,
        a.cfo_enhancement_strength,
        a.sfo_enhancement_strength,
    )
    if any(strength < 0.0 or strength > 1.0 for strength in strengths):
        raise ValueError("Enhancement strengths must be in [0, 1]")
    hue_low, hue_high = a.sfo_hue_range_degrees
    if not 0.0 <= hue_low < hue_high <= 360.0:
        raise ValueError("SFO hue range must satisfy 0 <= low < high <= 360")
    if a.trap_morphology_kernel_size < 1 or a.trap_morphology_kernel_size % 2 == 0:
        raise ValueError("TRAP morphology kernel must be a positive odd integer")
    if a.trap_morphology_iterations < 1:
        raise ValueError("TRAP morphology iterations must be positive")
    if not 0.0 <= a.synthetic_prob <= 1.0:
        raise ValueError("synthetic_prob must be in [0, 1]")
    if not 0.0 <= a.val_synthetic_prob <= 1.0:
        raise ValueError("val_synthetic_prob must be in [0, 1]")
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


def main(a):
    validate_args(a)
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
    model_config = dict(
        group_input_channels=training_dataset.group_input_channels,
        fixed_channels=training_dataset.channels,
        latent_dim=a.latent_dim,
        group_embedding_dim=a.group_embedding_dim,
        use_group_embedding=a.use_group_embedding,
        scale_range=tuple(a.model_scale_range),
        translation_limit=a.translation_limit,
        max_rotation_degrees=a.max_rotation_deg,
        depth=a.depth,
        base_channels=a.base_channels,
        spatial_pool_size=a.spatial_pool_size,
        norm_type=a.norm_type,
        use_coordconv=not a.disable_coordconv,
        fusion_mode=a.fusion_mode,
        force_group1_identity=a.force_group1_identity,
        separate_group_heads=a.separate_group_heads,
        separate_group_adapters=a.separate_group_adapters,
    )
    model = GroupAffineRegistrationModel(**model_config).to(device)
    if device.type == "cuda" and not a.no_multi_gpu and len(gpu_ids) > 1:
        model = torch.nn.DataParallel(model, device_ids=gpu_ids)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=a.lr, weight_decay=a.weight_decay
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=max(a.epochs, 1), eta_min=a.lr * 0.05
    )
    if a.resume_checkpoint:
        target_model = (
            model.module if isinstance(model, torch.nn.DataParallel) else model
        )
        load_initial_weights(target_model, a.resume_checkpoint, device)
    scaler = torch.amp.GradScaler(
        "cuda",
        enabled=a.amp and device.type == "cuda",
        init_scale=4096.0,
    )
    best = float("inf")
    use_wandb = bool(a.wandb_project)
    if use_wandb:
        import wandb

        wandb.init(project=a.wandb_project, name=a.wandb_run_name, config=vars(a))
    for epoch in range(1, a.epochs + 1):
        model.train()
        sums = {}
        n = 0
        optimizer_steps = 0
        skipped_steps = 0
        for batch in tqdm(
            training_loader, desc=f"Epoch {epoch}/{a.epochs}", leave=False
        ):
            batch = {
                k: (v.to(device, non_blocking=True) if torch.is_tensor(v) else v)
                for k, v in batch.items()
            }
            optimizer.zero_grad(set_to_none=True)
            with torch.amp.autocast(
                device_type=device.type, enabled=a.amp and device.type == "cuda"
            ):
                params = model(
                    batch["group_input"], batch["fixed_mineral"], batch["group"]
                )
                warped = warp_group(batch["moving_group"], params)
                total_loss, losses = grouped_loss(a, params, warped, batch)
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
            batch_size = params.shape[0]
            n += batch_size
            for k, v in losses.items():
                sums["train_" + k] = (
                    sums.get("train_" + k, 0) + float(v.detach()) * batch_size
                )
        if optimizer_steps == 0:
            raise FloatingPointError(
                "AMP skipped every optimizer step; lower the loss weights or disable AMP"
            )
        scheduler.step()
        metrics = {k: v / max(n, 1) for k, v in sums.items()}
        metrics["train_optimizer_steps"] = optimizer_steps
        metrics["train_skipped_steps"] = skipped_steps
        metrics.update(evaluate(a, model, validation_loader, device))
        metrics["epoch"] = epoch
        print(
            " ".join(
                f"{k}={v:.6f}"
                for k, v in metrics.items()
                if k
                in {"train_total", "train_param", "val_total", "val_param", "val_ncc"}
            )
        )
        if use_wandb:
            import wandb

            wandb.log(metrics)
        unwrapped_model = (
            model.module if isinstance(model, torch.nn.DataParallel) else model
        )
        payload = {
            "model_state_dict": unwrapped_model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict(),
            "epoch": epoch,
            "metrics": metrics,
            "model_config": model_config,
            "split_config": {
                "training_sample_ids": sorted(training_sample_ids),
                "validation_sample_ids": sorted(validation_sample_ids),
                "split_seed": a.split_seed,
                "validation_fraction": a.val_split,
            },
            "preprocess_config": {
                "height": a.height,
                "width": a.width,
                "image_mode": a.image_mode,
                "sfo_mode": a.sfo_mode,
                "crop_mode": a.crop_mode,
                "crop_margin": a.crop_margin,
                "group_input_mode": a.group_input_mode,
                "include_group1": a.include_group1,
                "enhance_groups": list(a.enhance_groups),
                "enhancement_strength": a.enhancement_strength,
                "enhancement_mode": a.enhancement_mode,
                "trap_enhancement_strength": a.trap_enhancement_strength,
                "cfo_enhancement_strength": a.cfo_enhancement_strength,
                "sfo_enhancement_strength": a.sfo_enhancement_strength,
                "sfo_hue_range_degrees": list(a.sfo_hue_range_degrees),
                "trap_morphology_kernel_size": a.trap_morphology_kernel_size,
                "trap_morphology_iterations": a.trap_morphology_iterations,
            },
        }
        torch.save(payload, os.path.join(a.output_dir, a.last_checkpoint_name))
        metric = metrics.get("val_total", metrics["train_total"])
        if metric < best:
            best = metric
            torch.save(payload, os.path.join(a.output_dir, a.best_checkpoint_name))
    if use_wandb:
        import wandb

        wandb.finish()


if __name__ == "__main__":
    main(parse_args())
