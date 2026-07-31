"""Correlation-volume model with selectable image frontends.

The student consumes Mineral reference and unregistered grouped
stains through a structural, raw, or concatenated hybrid frontend. Within each branch, learnable domain adapters feed
one weight-shared FPN, so both sides of every cost volume occupy the same
feature space. At every selected pyramid level, the model builds an explicit
local cost volume

    C(x, d) = cosine_similarity(F_mineral(x), F_group(x + d)).

Thus, a positive displacement directly means that a fixed/output location
samples the moving/group feature at ``x + d``.  This is the same output-to-input
direction used by :func:`torch.nn.functional.affine_grid` and by ``utils.py``.

The finest cost level is 1/8 resolution by construction.  Correlations are
computed with a displacement loop, so the largest temporary/output volume is
``B x (2r+1)^2 x H/8 x W/8`` rather than an all-pairs or unfolded tensor.
"""

from __future__ import annotations

import copy
import math
from typing import Sequence, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


STRUCTURAL_DESCRIPTOR_VERSION = "torch_structural_no_enhancement_v3"
STRUCTURAL_CHANNEL_NAMES = (
    "tissue",
    "foreground",
    "edge",
    "boundary_distance",
    "skeleton",
    "context",
)
STRUCTURAL_EVIDENCE_EPSILON = 1e-6
MAX_STRUCTURAL_RADIUS = 32
FRONTEND_MODES = ("structural", "raw", "hybrid")
GROUP_INPUT_MODES = ("stack", "overlay")
AFFINE_HEAD_MODES = ("joint", "separated", "separated_residual")
ENCODER_ARCHES = ("current", "residual")
STUDENT_FIXED_ADAPTER_MODES = ("none", "separate")
STUDENT_FIXED_ADAPTER_STATE_PREFIX = "student_fixed_group_adapters."
DEFAULT_RESIDUAL_ENCODER_CHANNELS = (64, 96, 128, 192, 256)
DEFAULT_RESIDUAL_BLOCKS_PER_STAGE = (2, 2, 2, 2, 2)
DEFAULT_CORRELATION_FEATURE_WIDTH = 96
DEFAULT_GROUP_SLOTS = 3
TEACHER_FIXED_INPUT_VERSION = "mineral_target_validity_mean_v1"


def _positive_int_tuple(values: int | Sequence[int], name: str) -> tuple[int, ...]:
    """Normalize one-or-more positive integer configuration values."""
    if isinstance(values, bool):
        raise ValueError(f"{name} must contain positive integers")
    if isinstance(values, int):
        parsed = (values,)
    else:
        parsed = tuple(values)
    if not parsed or any(
        isinstance(value, bool) or int(value) != value or int(value) < 1
        for value in parsed
    ):
        raise ValueError(f"{name} must contain positive integers")
    return tuple(int(value) for value in parsed)


def canonicalize_model_config(config: dict | None) -> dict:
    """Return a copied model config with backward-compatible frontend defaults.

    Checkpoints created before the frontend ablation always used the online
    structural descriptor and a stain-union representation. Supplying these
    defaults at every checkpoint boundary preserves that exact interpretation
    without mutating the serialized dictionary.
    """
    canonical = dict(config or {})
    canonical.setdefault("frontend_mode", "structural")
    canonical.setdefault("group_input_mode", "overlay")
    canonical.setdefault("group_slots", DEFAULT_GROUP_SLOTS)
    # Checkpoints written before geometry-head ablations used AffineHead. Keep
    # that exact module/state-dict interpretation when the field is absent.
    canonical.setdefault("affine_head_mode", "joint")
    student_fixed_adapter = str(canonical.setdefault("student_fixed_adapter", "none"))
    if student_fixed_adapter not in STUDENT_FIXED_ADAPTER_MODES:
        choices = ", ".join(STUDENT_FIXED_ADAPTER_MODES)
        raise ValueError(f"student_fixed_adapter must be one of: {choices}")
    canonical["student_fixed_adapter"] = student_fixed_adapter

    # Encoder fields were added after the original FPN checkpoints. Missing
    # fields must reconstruct that exact architecture: in particular, an old
    # checkpoint must not gain a learned correlation projection merely because
    # the default for new command-line runs is wider.
    encoder_arch = str(canonical.setdefault("encoder_arch", "current"))
    if encoder_arch not in ENCODER_ARCHES:
        choices = ", ".join(ENCODER_ARCHES)
        raise ValueError(f"encoder_arch must be one of: {choices}")

    configured_channels = canonical.get("encoder_channels")
    if configured_channels is None:
        if encoder_arch == "residual":
            channels = DEFAULT_RESIDUAL_ENCODER_CHANNELS
        else:
            channels = tuple(
                _stage_channels(
                    int(canonical.get("encoder_base_channels", 24)),
                    int(canonical.get("encoder_depth", 5)),
                )
            )
    else:
        channels = _positive_int_tuple(configured_channels, "encoder_channels")
    if len(channels) < 3:
        raise ValueError("encoder_channels must contain at least three stages")
    canonical["encoder_channels"] = tuple(channels)
    canonical["encoder_depth"] = len(channels)

    configured_blocks = canonical.get("encoder_blocks_per_stage")
    if configured_blocks is None:
        blocks = (
            (
                DEFAULT_RESIDUAL_BLOCKS_PER_STAGE
                if tuple(channels) == DEFAULT_RESIDUAL_ENCODER_CHANNELS
                else tuple(2 for _ in channels)
            )
            if encoder_arch == "residual"
            else None
        )
    else:
        parsed_blocks = _positive_int_tuple(
            configured_blocks, "encoder_blocks_per_stage"
        )
        if len(parsed_blocks) == 1:
            blocks = parsed_blocks * len(channels)
        elif len(parsed_blocks) == len(channels):
            blocks = parsed_blocks
        else:
            raise ValueError(
                "encoder_blocks_per_stage must contain one value or exactly one "
                "value per encoder stage"
            )
    canonical["encoder_blocks_per_stage"] = blocks

    correlation_width = canonical.get("correlation_feature_width")
    if correlation_width is None:
        # Legacy cost volumes correlated the FPN features directly.
        correlation_width = int(canonical.get("feature_width", 48))
    if (
        isinstance(correlation_width, bool)
        or int(correlation_width) != correlation_width
        or int(correlation_width) < 1
    ):
        raise ValueError("correlation_feature_width must be a positive integer")
    canonical["correlation_feature_width"] = int(correlation_width)
    return canonical


class OnlineStructuralFrontend(nn.Module):
    """Construct six-channel structural maps from deployable image inputs.

    All operations are batched PyTorch operations. No stain-specific
    enhancement is performed: RGB and HSV affect only how signal intensity is
    interpreted before the same structural descriptor is constructed.
    """

    def __init__(
        self,
        *,
        input_channels: int,
        sfo_mode: str,
        foreground_threshold: float | None,
        distance_scale: float,
        context_scale: float,
        skeleton_radius: int,
    ) -> None:
        super().__init__()
        if input_channels not in (1, 3):
            raise ValueError("input_channels must be 1 or 3")
        if sfo_mode not in {"rgb", "hsv", "gray"}:
            raise ValueError("sfo_mode must be rgb, hsv, or gray")
        if foreground_threshold is not None and (
            not math.isfinite(float(foreground_threshold))
            or not 0.0 <= float(foreground_threshold) <= 1.0
        ):
            raise ValueError("foreground_threshold must be None or in [0, 1]")
        if not math.isfinite(float(distance_scale)) or float(distance_scale) <= 0.0:
            raise ValueError("distance_scale must be positive and finite")
        if not math.isfinite(float(context_scale)) or float(context_scale) <= 0.0:
            raise ValueError("context_scale must be positive and finite")
        if (
            isinstance(skeleton_radius, bool)
            or int(skeleton_radius) != skeleton_radius
            or int(skeleton_radius) < 1
        ):
            raise ValueError("skeleton_radius must be a positive integer")

        self.input_channels = int(input_channels)
        self.sfo_mode = sfo_mode
        self.foreground_threshold = (
            None if foreground_threshold is None else float(foreground_threshold)
        )
        self.distance_scale = float(distance_scale)
        self.context_scale = float(context_scale)
        self.skeleton_radius = int(skeleton_radius)

        sobel_x = torch.tensor(
            [[-1.0, 0.0, 1.0], [-2.0, 0.0, 2.0], [-1.0, 0.0, 1.0]],
            dtype=torch.float32,
        ).view(1, 1, 3, 3)
        self.register_buffer("sobel_x", sobel_x, persistent=True)
        self.register_buffer(
            "sobel_y", sobel_x.transpose(-1, -2).contiguous(), persistent=True
        )

    @staticmethod
    def _unit_interval(image: torch.Tensor) -> torch.Tensor:
        """Scale each single-channel image while preserving padded zeros."""
        if image.ndim != 4 or image.shape[1] != 1:
            raise ValueError("Structural intensity must have shape Nx1xHxW")
        maximum = image.amax(dim=(2, 3), keepdim=True)
        scaled = torch.where(
            maximum > STRUCTURAL_EVIDENCE_EPSILON,
            image / maximum.clamp_min(STRUCTURAL_EVIDENCE_EPSILON),
            torch.zeros_like(image),
        )
        return scaled.clamp(0.0, 1.0) * (image > STRUCTURAL_EVIDENCE_EPSILON).to(
            image.dtype
        )

    def _sobel_magnitude(self, image: torch.Tensor) -> torch.Tensor:
        kernel_x = self.sobel_x.to(device=image.device, dtype=image.dtype)
        kernel_y = self.sobel_y.to(device=image.device, dtype=image.dtype)
        gx = F.conv2d(image, kernel_x, padding=1)
        gy = F.conv2d(image, kernel_y, padding=1)
        return self._unit_interval(torch.sqrt(gx.square() + gy.square() + 1e-12))

    def _slot_likelihoods(
        self, moving_group: torch.Tensor, group_ids: torch.Tensor
    ) -> torch.Tensor:
        batch_size, slots, channels, height, width = moving_group.shape
        flat = moving_group.reshape(batch_size * slots, channels, height, width)
        if channels == 1:
            intensity = flat
        else:
            intensity = flat.amax(dim=1, keepdim=True)
            if self.sfo_mode == "hsv":
                hsv_value = flat[:, 2:3]
                sample_is_group5 = (
                    (group_ids == 5)[:, None]
                    .expand(batch_size, slots)
                    .reshape(batch_size * slots, 1, 1, 1)
                )
                intensity = torch.where(sample_is_group5, hsv_value, intensity)

        likelihood = self._unit_interval(intensity).view(
            batch_size, slots, 1, height, width
        )
        slot_present = (
            moving_group.detach().abs().amax(dim=(2, 3, 4))
            > STRUCTURAL_EVIDENCE_EPSILON
        )
        return likelihood * slot_present[:, :, None, None, None].to(likelihood.dtype)

    @staticmethod
    def _distance_proxy(boundary: torch.Tensor, radius: int) -> torch.Tensor:
        """Approximate distance with four separable compact dilation bands."""
        radius = min(max(int(radius), 1), MAX_STRUCTURAL_RADIUS)
        radii = sorted({1, max(1, radius // 4), max(1, radius // 2), radius})
        proximity = boundary
        for sample_radius in radii:
            kernel = 2 * sample_radius + 1
            # A direct KxK max pool is prohibitively expensive at 1024 pixels.
            # Separable horizontal/vertical pools produce the same rectangular
            # dilation with O(K), rather than O(K squared), local comparisons.
            expanded = F.max_pool2d(
                boundary,
                kernel_size=(1, kernel),
                stride=1,
                padding=(0, sample_radius),
            )
            expanded = F.max_pool2d(
                expanded,
                kernel_size=(kernel, 1),
                stride=1,
                padding=(sample_radius, 0),
            )
            proximity = torch.maximum(
                proximity,
                expanded * math.exp(-float(sample_radius) / float(radius)),
            )
        return proximity.clamp(0.0, 1.0)

    @staticmethod
    def _box_blur(image: torch.Tensor, radius: int) -> torch.Tensor:
        """Separable box context with bounded 1024-scale runtime."""
        radius = min(max(int(radius), 1), MAX_STRUCTURAL_RADIUS)
        kernel = 2 * radius + 1
        blurred = F.avg_pool2d(
            image,
            kernel_size=(1, kernel),
            stride=1,
            padding=(0, radius),
        )
        return F.avg_pool2d(
            blurred,
            kernel_size=(kernel, 1),
            stride=1,
            padding=(radius, 0),
        )

    @staticmethod
    def _erode(image: torch.Tensor) -> torch.Tensor:
        padded = F.pad(image, (1, 1, 1, 1), value=0.0)
        return -F.max_pool2d(-padded, kernel_size=3, stride=1)

    def _descriptor(
        self, likelihood: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        tissue = self._unit_interval(likelihood.float())
        positive = tissue > STRUCTURAL_EVIDENCE_EPSILON
        if self.foreground_threshold is None:
            positive_float = positive.to(tissue.dtype)
            count = positive_float.sum(dim=(2, 3), keepdim=True).clamp_min(1.0)
            positive_mean = (tissue * positive_float).sum(
                dim=(2, 3), keepdim=True
            ) / count
            threshold = (0.5 * positive_mean).clamp(0.05, 0.5)
        else:
            threshold = tissue.new_tensor(self.foreground_threshold)
        foreground = (tissue > threshold).to(tissue.dtype)
        edge = self._sobel_magnitude(tissue)

        dilated = F.max_pool2d(foreground, 3, stride=1, padding=1)
        eroded = self._erode(foreground)
        boundary = (dilated - eroded).clamp(0.0, 1.0)
        height, width = tissue.shape[-2:]
        distance_radius = min(
            MAX_STRUCTURAL_RADIUS,
            max(1, int(round(self.distance_scale * min(height, width)))),
        )
        boundary_distance = self._distance_proxy(boundary, distance_radius)

        erosion_steps = min(max(distance_radius, 2), 12)
        current = foreground
        inside_distance = torch.zeros_like(foreground)
        for _ in range(erosion_steps):
            current = self._erode(current).clamp(0.0, 1.0)
            inside_distance = inside_distance + current
        local_maximum = F.max_pool2d(inside_distance, 3, stride=1, padding=1)
        skeleton = (
            (inside_distance > 0.0) & (inside_distance >= local_maximum - 1e-6)
        ).to(tissue.dtype)
        radius = self.skeleton_radius
        skeleton = F.max_pool2d(
            skeleton,
            kernel_size=2 * radius + 1,
            stride=1,
            padding=radius,
        )
        foreground_count = foreground.sum(dim=(2, 3), keepdim=True)
        nontrivial = (foreground_count > 0.0) & (
            foreground_count < float(height * width)
        )
        skeleton = skeleton * nontrivial.to(skeleton.dtype)
        boundary_distance = boundary_distance * nontrivial.to(boundary_distance.dtype)

        context_radius = min(
            MAX_STRUCTURAL_RADIUS,
            max(1, int(round(self.context_scale * min(height, width)))),
        )
        context = 0.5 * self._box_blur(tissue, context_radius)
        context = context + 0.5 * self._box_blur(foreground, context_radius)
        descriptor = torch.cat(
            (
                tissue,
                foreground,
                edge,
                boundary_distance,
                skeleton,
                context.clamp(0.0, 1.0),
            ),
            dim=1,
        )
        descriptor = torch.where(
            descriptor.abs() > STRUCTURAL_EVIDENCE_EPSILON,
            descriptor,
            torch.zeros_like(descriptor),
        )
        validity = (
            descriptor.detach().abs().amax(dim=1, keepdim=True)
            > STRUCTURAL_EVIDENCE_EPSILON
        )
        return descriptor, validity

    def mineral_descriptor(
        self, fixed_mineral: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Build a descriptor for registered Mineral without stain enhancement."""
        if fixed_mineral.ndim != 4:
            raise ValueError("fixed_mineral must have shape BxCxHxW")
        if fixed_mineral.shape[1] != self.input_channels:
            raise ValueError(
                f"Expected {self.input_channels} input channels, got "
                f"{fixed_mineral.shape[1]}"
            )
        likelihood = (
            fixed_mineral
            if self.input_channels == 1
            else fixed_mineral.amax(dim=1, keepdim=True)
        )
        return self._descriptor(likelihood)

    def group_descriptor(
        self,
        group_stack: torch.Tensor,
        group_ids: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Build one structural union descriptor from all present stain slots."""
        if group_stack.ndim != 5:
            raise ValueError("group_stack must have shape BxKxCxHxW")
        batch_size, slots, channels, _, _ = group_stack.shape
        if slots < 1:
            raise ValueError("group_stack must contain at least one stain slot")
        if channels != self.input_channels:
            raise ValueError(
                f"Expected {self.input_channels} input channels, got {channels}"
            )
        if group_ids.reshape(-1).numel() != batch_size:
            raise ValueError("group_stack and group_ids must have equal batches")
        slot_likelihoods = self._slot_likelihoods(
            group_stack, group_ids.reshape(-1).long()
        )
        return self._descriptor(slot_likelihoods.amax(dim=1))

    def forward(
        self,
        fixed_mineral: torch.Tensor,
        moving_group: torch.Tensor,
        group_ids: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        batch_size, _, channels, height, width = moving_group.shape
        if tuple(fixed_mineral.shape) != (batch_size, channels, height, width):
            raise ValueError(
                "fixed_mineral and moving_group image shapes are incompatible"
            )
        fixed_structure, fixed_valid = self.mineral_descriptor(fixed_mineral)
        group_structure, group_valid = self.group_descriptor(moving_group, group_ids)
        return fixed_structure, group_structure, fixed_valid, group_valid


def _make_norm(channels: int, norm_type: str) -> nn.Module:
    if norm_type == "batch":
        return nn.BatchNorm2d(channels)
    if norm_type == "instance":
        return nn.InstanceNorm2d(channels, affine=True)
    if norm_type == "group":
        groups = min(8, channels)
        while channels % groups and groups > 1:
            groups -= 1
        return nn.GroupNorm(groups, channels)
    raise ValueError(f"Unsupported norm_type: {norm_type}")


def _stage_channels(base_channels: int, depth: int) -> list[int]:
    """Return memory-conscious channel widths, rounded to multiples of eight."""
    if base_channels < 8:
        raise ValueError("encoder_base_channels must be at least 8")
    if depth < 3:
        raise ValueError("encoder_depth must be at least 3")
    channels = []
    for level in range(depth):
        value = base_channels * (2.0 ** (0.5 * level))
        channels.append(max(8, int(round(value / 8.0)) * 8))
    return channels


def _canonical_validity_mask(
    mask: torch.Tensor | None,
    reference: torch.Tensor,
    name: str,
) -> torch.Tensor | None:
    """Validate a support mask and return it as ``B x 1 x H x W`` boolean."""
    if mask is None:
        return None
    if mask.ndim == 3:
        mask = mask.unsqueeze(1)
    if mask.ndim != 4 or mask.shape[1] != 1:
        raise ValueError(f"{name} must have shape Bx1xHxW or BxHxW")
    expected = (reference.shape[0], 1, *reference.shape[-2:])
    if tuple(mask.shape) != expected:
        raise ValueError(
            f"{name} shape {tuple(mask.shape)} does not match expected {expected}"
        )
    if mask.is_floating_point() and not bool(torch.isfinite(mask).all()):
        raise ValueError(f"{name} contains non-finite values")
    canonical = mask.bool() if mask.dtype == torch.bool else mask > 0.5
    return canonical.to(device=reference.device)


def _validity_with_structural_evidence(
    descriptor: torch.Tensor,
    validity: torch.Tensor | None,
    name: str,
) -> torch.Tensor:
    """Intersect FOV support with locations containing descriptor evidence.

    Normalization and adapter offsets can otherwise turn an all-zero descriptor
    into non-zero features whose cost-volume pattern reveals only crop/support
    geometry. Deriving evidence from the descriptor before those learned
    offsets guarantees that empty structure cannot become a correspondence.
    """
    canonical = _canonical_validity_mask(validity, descriptor, name)
    evidence = (
        descriptor.detach().abs().amax(dim=1, keepdim=True)
        > STRUCTURAL_EVIDENCE_EPSILON
    )
    return evidence if canonical is None else canonical & evidence


def _raw_rectangular_fov(image: torch.Tensor, name: str) -> torch.Tensor:
    """Infer a conservative rectangular FOV from a raw ``B x C x H x W`` tensor.

    Raw canvases are non-negative and zero padded. The rectangle deliberately
    includes dark pixels between observed signal extrema while an all-zero
    canvas remains invalid. It is computed before any learnable adapter, so
    adapter offsets cannot manufacture evidence in padded or empty inputs.
    """
    if image.ndim != 4:
        raise ValueError(f"{name} must have shape BxCxHxW")
    evidence = image.detach().abs().amax(dim=1) > STRUCTURAL_EVIDENCE_EPSILON
    batch_size, height, width = evidence.shape
    row_has_signal = evidence.any(dim=2)
    column_has_signal = evidence.any(dim=1)
    has_signal = evidence.flatten(1).any(dim=1)

    y_coordinates = torch.arange(height, device=image.device).view(1, height)
    x_coordinates = torch.arange(width, device=image.device).view(1, width)
    first_y = torch.where(row_has_signal, y_coordinates, height).amin(dim=1)
    last_y = torch.where(row_has_signal, y_coordinates, -1).amax(dim=1)
    first_x = torch.where(column_has_signal, x_coordinates, width).amin(dim=1)
    last_x = torch.where(column_has_signal, x_coordinates, -1).amax(dim=1)

    yy = y_coordinates.view(1, height, 1)
    xx = x_coordinates.view(1, 1, width)
    support = (
        (yy >= first_y[:, None, None])
        & (yy <= last_y[:, None, None])
        & (xx >= first_x[:, None, None])
        & (xx <= last_x[:, None, None])
        & has_signal[:, None, None]
    )
    expected = (batch_size, 1, height, width)
    support = support.unsqueeze(1)
    if tuple(support.shape) != expected:
        raise AssertionError(
            f"Internal raw FOV shape {tuple(support.shape)} != {expected}"
        )
    return support


class ConvStage(nn.Module):
    """Stride-two CNN stage followed by a residual refinement block."""

    def __init__(self, in_channels: int, out_channels: int, norm_type: str) -> None:
        super().__init__()
        self.down = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, stride=2, padding=1, bias=False),
            _make_norm(out_channels, norm_type),
            nn.GELU(),
        )
        self.refine = nn.Sequential(
            nn.Conv2d(out_channels, out_channels, 3, padding=1, bias=False),
            _make_norm(out_channels, norm_type),
            nn.GELU(),
            nn.Conv2d(out_channels, out_channels, 3, padding=1, bias=False),
            _make_norm(out_channels, norm_type),
        )
        self.activation = nn.GELU()

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        feature = self.down(image)
        return self.activation(feature + self.refine(feature))


class ResidualEncoderBlock(nn.Module):
    """Two-convolution residual block with GroupNorm and a learned skip."""

    def __init__(self, in_channels: int, out_channels: int, *, stride: int = 1) -> None:
        super().__init__()
        if stride not in (1, 2):
            raise ValueError("ResidualEncoderBlock stride must be 1 or 2")
        self.conv1 = nn.Conv2d(
            in_channels,
            out_channels,
            3,
            stride=stride,
            padding=1,
            bias=False,
        )
        self.norm1 = _make_norm(out_channels, "group")
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, padding=1, bias=False)
        self.norm2 = _make_norm(out_channels, "group")
        self.projection = (
            nn.Identity()
            if stride == 1 and in_channels == out_channels
            else nn.Conv2d(in_channels, out_channels, 1, stride=stride, bias=False)
        )
        self.activation = nn.GELU()

        for module in (self.conv1, self.conv2, self.projection):
            if isinstance(module, nn.Conv2d):
                nn.init.kaiming_normal_(
                    module.weight, mode="fan_out", nonlinearity="relu"
                )
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
        # The residual branch begins as zero while the learned projection still
        # carries channel changes and downsampling. This is the appropriate
        # identity initialization for same-shape blocks.
        nn.init.zeros_(self.norm2.weight)
        nn.init.zeros_(self.norm2.bias)

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        residual = self.projection(image)
        feature = self.activation(self.norm1(self.conv1(image)))
        feature = self.norm2(self.conv2(feature))
        return self.activation(residual + feature)


class ResidualEncoderStage(nn.Module):
    """Stride-two residual stage followed by same-resolution refinements."""

    def __init__(self, in_channels: int, out_channels: int, blocks: int) -> None:
        super().__init__()
        if blocks < 1:
            raise ValueError("Every residual encoder stage needs at least one block")
        modules = [ResidualEncoderBlock(in_channels, out_channels, stride=2)]
        modules.extend(
            ResidualEncoderBlock(out_channels, out_channels) for _ in range(blocks - 1)
        )
        self.blocks = nn.ModuleList(modules)

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        feature = image
        for block in self.blocks:
            feature = block(feature)
        return feature


class FeaturePyramidEncoder(nn.Module):
    """Weight-shared selectable CNN encoder with top-down FPN fusion."""

    def __init__(
        self,
        input_channels: int,
        encoder_base_channels: int,
        encoder_depth: int,
        feature_width: int,
        norm_type: str,
        num_cost_levels: int,
        encoder_arch: str = "current",
        encoder_channels: Sequence[int] | None = None,
        encoder_blocks_per_stage: int | Sequence[int] | None = None,
    ) -> None:
        super().__init__()
        if feature_width < 8:
            raise ValueError("feature_width must be at least 8")
        if num_cost_levels < 1:
            raise ValueError("At least one cost-volume level is required")
        if encoder_arch not in ENCODER_ARCHES:
            choices = ", ".join(ENCODER_ARCHES)
            raise ValueError(f"encoder_arch must be one of: {choices}")
        if encoder_channels is None:
            channels = (
                DEFAULT_RESIDUAL_ENCODER_CHANNELS
                if encoder_arch == "residual"
                else tuple(_stage_channels(encoder_base_channels, encoder_depth))
            )
        else:
            channels = _positive_int_tuple(encoder_channels, "encoder_channels")
        if len(channels) < num_cost_levels + 2:
            raise ValueError(
                "The encoder must have at least len(cost_volume_radii) + 2 "
                "stages so the finest cost volume is no finer than 1/8"
            )

        block_counts: tuple[int, ...] | None
        if encoder_blocks_per_stage is None:
            block_counts = (
                (
                    DEFAULT_RESIDUAL_BLOCKS_PER_STAGE
                    if tuple(channels) == DEFAULT_RESIDUAL_ENCODER_CHANNELS
                    else tuple(2 for _ in channels)
                )
                if encoder_arch == "residual"
                else None
            )
        else:
            parsed_blocks = _positive_int_tuple(
                encoder_blocks_per_stage, "encoder_blocks_per_stage"
            )
            if len(parsed_blocks) == 1:
                block_counts = parsed_blocks * len(channels)
            elif len(parsed_blocks) == len(channels):
                block_counts = parsed_blocks
            else:
                raise ValueError(
                    "encoder_blocks_per_stage must contain one value or exactly "
                    "one value per encoder stage"
                )

        self.encoder_arch = encoder_arch
        self.encoder_channels = tuple(channels)
        self.encoder_blocks_per_stage = block_counts
        stages = []
        in_channels = input_channels
        for index, out_channels in enumerate(channels):
            if encoder_arch == "current":
                stage = ConvStage(in_channels, out_channels, norm_type)
            else:
                assert block_counts is not None
                stage = ResidualEncoderStage(
                    in_channels, out_channels, block_counts[index]
                )
            stages.append(stage)
            in_channels = out_channels
        self.stages = nn.ModuleList(stages)

        # Stages 0 and 1 remain bottom-up only. Matching always begins at stage
        # 2 (1/8), irrespective of channel widths or residual block counts.
        fpn_indices = range(2, len(channels))
        self.lateral = nn.ModuleDict(
            {
                str(index): nn.Conv2d(channels[index], feature_width, 1, bias=False)
                for index in fpn_indices
            }
        )
        fpn_norm_type = "group" if encoder_arch == "residual" else norm_type
        self.smooth = nn.ModuleDict(
            {
                str(index): nn.Sequential(
                    nn.Conv2d(feature_width, feature_width, 3, padding=1, bias=False),
                    _make_norm(feature_width, fpn_norm_type),
                    nn.GELU(),
                )
                for index in range(2, len(channels))
            }
        )
        if encoder_arch == "residual":
            for collection in (self.lateral, self.smooth):
                for module in collection.modules():
                    if isinstance(module, nn.Conv2d):
                        nn.init.kaiming_normal_(
                            module.weight, mode="fan_out", nonlinearity="relu"
                        )
                        if module.bias is not None:
                            nn.init.zeros_(module.bias)

        self.cost_indices = tuple(range(2, 2 + num_cost_levels))
        self.min_cost_index = self.cost_indices[0]

    def forward(self, image: torch.Tensor) -> list[torch.Tensor]:
        bottom_up = []
        feature = image
        for stage in self.stages:
            feature = stage(feature)
            bottom_up.append(feature)

        pyramid: list[torch.Tensor] = [bottom_up[-1].new_empty(0)] * len(bottom_up)
        last_index = len(bottom_up) - 1
        top_down = self.lateral[str(last_index)](bottom_up[-1])
        pyramid[-1] = self.smooth[str(last_index)](top_down)
        for index in range(len(bottom_up) - 2, self.min_cost_index - 1, -1):
            lateral = self.lateral[str(index)](bottom_up[index])
            top_down = lateral + F.interpolate(
                top_down, size=lateral.shape[-2:], mode="bilinear", align_corners=False
            )
            pyramid[index] = self.smooth[str(index)](top_down)
        return [pyramid[index] for index in self.cost_indices]


class ResidualInputAdapter(nn.Module):
    """Identity-initialized group-specific frontend adapter."""

    def __init__(self, channels: int, norm_type: str) -> None:
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(channels, channels, 3, padding=1, bias=False),
            _make_norm(channels, norm_type),
            nn.GELU(),
            nn.Conv2d(channels, channels, 1, bias=True),
        )
        nn.init.zeros_(self.block[-1].weight)
        nn.init.zeros_(self.block[-1].bias)

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        return image + self.block(image)


class StudentFixedGroupAdapter(nn.Module):
    """Near-identity fixed-side adapter used only by the student branch.

    The second GroupNorm scale starts at zero, so the learned residual branch
    initially contributes exactly zero before the prescribed final GELU.
    GroupNorm is unconditional because effective registration batches are small.
    """

    def __init__(self, channels: int) -> None:
        super().__init__()
        if channels < 1:
            raise ValueError("Student fixed adapter channels must be positive")
        self.conv1 = nn.Conv2d(channels, channels, 3, padding=1, bias=False)
        self.norm1 = _make_norm(channels, "group")
        self.activation1 = nn.GELU()
        self.conv2 = nn.Conv2d(channels, channels, 3, padding=1, bias=False)
        self.norm2 = _make_norm(channels, "group")
        self.activation2 = nn.GELU()

        nn.init.kaiming_normal_(self.conv1.weight, mode="fan_out", nonlinearity="relu")
        nn.init.kaiming_normal_(self.conv2.weight, mode="fan_out", nonlinearity="relu")
        if not isinstance(self.norm2, nn.GroupNorm):
            raise AssertionError("Student fixed adapters require GroupNorm")
        nn.init.zeros_(self.norm2.weight)
        nn.init.zeros_(self.norm2.bias)

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        residual = self.activation1(self.norm1(self.conv1(image)))
        residual = self.norm2(self.conv2(residual))
        return self.activation2(image + residual)


class FrontendInputAdapter(nn.Module):
    """Map one raw or hybrid domain into the shared FPN input width."""

    def __init__(
        self, input_channels: int, output_channels: int, norm_type: str
    ) -> None:
        super().__init__()
        if input_channels < 1 or output_channels < 1:
            raise ValueError("Frontend adapter channel counts must be positive")
        self.block = nn.Sequential(
            nn.Conv2d(input_channels, output_channels, 1, bias=False),
            _make_norm(output_channels, norm_type),
            nn.GELU(),
            nn.Conv2d(
                output_channels,
                output_channels,
                3,
                padding=1,
                bias=False,
            ),
            _make_norm(output_channels, norm_type),
            nn.GELU(),
        )

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        return self.block(image)


class LocalCorrelationVolume(nn.Module):
    """Compute ``fixed(x)`` versus ``moving(x+d)`` cosine similarities."""

    def __init__(self, radius: int, temperature: float) -> None:
        super().__init__()
        if radius < 1:
            raise ValueError("Every cost-volume radius must be at least 1")
        if radius > 8:
            raise ValueError("Cost-volume radii above 8 are intentionally disallowed")
        if temperature <= 0.0:
            raise ValueError("correlation_temperature must be positive")
        self.radius = int(radius)
        self.temperature = float(temperature)
        displacements = [
            (float(dx), float(dy))
            for dy in range(-self.radius, self.radius + 1)
            for dx in range(-self.radius, self.radius + 1)
        ]
        self.register_buffer(
            "displacements", torch.tensor(displacements, dtype=torch.float32)
        )

    @property
    def channels(self) -> int:
        return (2 * self.radius + 1) ** 2

    def forward(
        self,
        fixed: torch.Tensor,
        moving: torch.Tensor,
        fixed_valid: torch.Tensor | None = None,
        moving_valid: torch.Tensor | None = None,
        return_aux: bool = False,
        _batch_aux_metadata: bool = False,
    ) -> (
        tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]
        | tuple[
            torch.Tensor,
            torch.Tensor,
            torch.Tensor,
            torch.Tensor,
            dict[str, object],
        ]
    ):
        if fixed.shape != moving.shape:
            raise ValueError(
                f"Correlation feature mismatch: {tuple(fixed.shape)} vs "
                f"{tuple(moving.shape)}"
            )
        fixed_valid = _canonical_validity_mask(fixed_valid, fixed, "fixed_valid")
        moving_valid = _canonical_validity_mask(moving_valid, moving, "moving_valid")
        # Correlation and softmax remain float32 under AMP for stable matching.
        fixed_unit = F.normalize(fixed.float(), p=2, dim=1, eps=1e-6)
        moving_unit = F.normalize(moving.float(), p=2, dim=1, eps=1e-6)
        batch_size, _, height, width = fixed_unit.shape
        radius = self.radius
        padded = F.pad(moving_unit, (radius, radius, radius, radius))
        if fixed_valid is None:
            fixed_valid = torch.ones(
                (batch_size, 1, height, width),
                device=fixed_unit.device,
                dtype=torch.bool,
            )
        else:
            fixed_valid = fixed_valid.to(device=fixed_unit.device)
        if moving_valid is None:
            moving_valid = torch.ones_like(fixed_valid)
        else:
            moving_valid = moving_valid.to(device=fixed_unit.device)
        # Padding the moving validity mask with false simultaneously enforces
        # image borders and excludes structureless/padded moving locations.
        valid_padded = F.pad(moving_valid, (radius, radius, radius, radius))
        correlations = []
        validities = []
        for dy in range(-radius, radius + 1):
            y0 = radius + dy
            for dx in range(-radius, radius + 1):
                x0 = radius + dx
                shifted = padded[..., y0 : y0 + height, x0 : x0 + width]
                correlations.append((fixed_unit * shifted).sum(dim=1))
                shifted_valid = valid_padded[..., y0 : y0 + height, x0 : x0 + width]
                validities.append((fixed_valid & shifted_valid).squeeze(1))
        raw_volume = torch.stack(correlations, dim=1)
        valid_volume = torch.stack(validities, dim=1)
        # Invalid shifts must be neutral in the tensor seen by the learned
        # aggregator. Encoding them as a sentinel (for example, -1) exposes
        # the shape and position of the validity rectangle even when both
        # structural descriptors are identically zero. They are still
        # excluded completely from the correspondence softmax below.
        volume = raw_volume.masked_fill(~valid_volume, 0.0)

        logits = (raw_volume / self.temperature).masked_fill(
            ~valid_volume, torch.finfo(raw_volume.dtype).min
        )
        probability = torch.softmax(logits, dim=1) * valid_volume.to(volume.dtype)
        probability_sum = probability.sum(dim=1, keepdim=True)
        probability = torch.where(
            probability_sum > 0.0,
            probability / probability_sum.clamp_min(1e-8),
            torch.zeros_like(probability),
        )
        displacement_table = self.displacements.to(
            device=volume.device, dtype=volume.dtype
        )
        probability_expected = torch.einsum(
            "bkhw,kd->bdhw", probability, displacement_table
        )
        confidence = probability.amax(dim=1, keepdim=True)
        entropy = -(probability * torch.log(probability.clamp_min(1e-8))).sum(
            dim=1, keepdim=True
        )
        valid_count = valid_volume.sum(dim=1, keepdim=True).to(volume.dtype)
        max_entropy = torch.log(valid_count.clamp_min(2.0))
        normalized_entropy = torch.where(
            valid_count > 1.0, entropy / max_entropy, torch.ones_like(entropy)
        )
        certainty = (1.0 - normalized_entropy).clamp(0.0, 1.0)
        certainty = torch.where(
            certainty > 1e-6, certainty, torch.zeros_like(certainty)
        )
        # A uniform distribution over asymmetric valid border offsets has a
        # non-zero raw displacement. Certainty gating prevents that geometry
        # artefact from becoming a false correspondence.
        expected = probability_expected * certainty
        if not return_aux:
            return volume, expected, confidence, certainty

        if _batch_aux_metadata:
            # nn.DataParallel gathers tensors along their leading dimension.
            # Repeat non-batched metadata temporarily; the outer wrapper turns
            # it back into the canonical Kx2 table and [height, width] list.
            aux_displacements: torch.Tensor = displacement_table.unsqueeze(0).expand(
                batch_size, -1, -1
            )
            aux_feature_size: object = (
                torch.tensor((height, width), device=volume.device, dtype=torch.long)
                .unsqueeze(0)
                .expand(batch_size, -1)
            )
        else:
            aux_displacements = displacement_table
            aux_feature_size = [height, width]
        auxiliary: dict[str, object] = {
            "volume": raw_volume,
            "probability": probability,
            "expected_displacement": probability_expected,
            "confidence": confidence,
            "certainty": certainty,
            "displacements": aux_displacements,
            "fixed_valid": fixed_valid,
            "moving_valid": moving_valid,
            "feature_size": aux_feature_size,
            # Training needs the per-candidate border/evidence support when it
            # normalizes a Gaussian correspondence target. The requested public
            # fields remain present verbatim; this extra mask prevents callers
            # from reverse-engineering validity from floating probabilities.
            "candidate_valid": valid_volume,
        }
        return volume, expected, confidence, certainty, auxiliary


class CostVolumeAggregator(nn.Module):
    """Turn one cost volume into spatial and affine-moment descriptors."""

    stats_dim = 8

    def __init__(
        self,
        volume_channels: int,
        hidden_channels: int,
        output_dim: int,
        pool_size: int,
        norm_type: str,
        radius: int,
    ) -> None:
        super().__init__()
        if hidden_channels < 8:
            raise ValueError("cost_hidden_channels must be at least 8")
        if pool_size < 1:
            raise ValueError("cost_pool_size must be positive")
        self.radius = float(radius)
        # volume + expected dx/dy + confidence + certainty + x/y coordinates
        input_channels = volume_channels + 6
        self.encoder = nn.Sequential(
            nn.Conv2d(input_channels, hidden_channels, 1, bias=False),
            _make_norm(hidden_channels, norm_type),
            nn.GELU(),
            nn.Conv2d(
                hidden_channels,
                hidden_channels,
                3,
                padding=1,
                groups=hidden_channels,
                bias=False,
            ),
            _make_norm(hidden_channels, norm_type),
            nn.GELU(),
            nn.Conv2d(hidden_channels, hidden_channels, 1, bias=False),
            _make_norm(hidden_channels, norm_type),
            nn.GELU(),
        )
        self.pool = nn.AdaptiveAvgPool2d((pool_size, pool_size))
        self.projection = nn.Sequential(
            nn.Flatten(),
            nn.Linear(hidden_channels * pool_size * pool_size, output_dim),
            nn.LayerNorm(output_dim),
            nn.GELU(),
        )

    @staticmethod
    def _coordinates(reference: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        batch, _, height, width = reference.shape
        yy = torch.linspace(
            -1.0, 1.0, height, device=reference.device, dtype=reference.dtype
        )
        xx = torch.linspace(
            -1.0, 1.0, width, device=reference.device, dtype=reference.dtype
        )
        yy, xx = torch.meshgrid(yy, xx, indexing="ij")
        coords = torch.stack((xx, yy), dim=0).unsqueeze(0).expand(batch, -1, -1, -1)
        return coords[:, 0:1], coords[:, 1:2]

    def forward(
        self,
        volume: torch.Tensor,
        displacement_pixels: torch.Tensor,
        confidence: torch.Tensor,
        certainty: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        x_coord, y_coord = self._coordinates(volume)
        displacement_for_map = displacement_pixels / self.radius
        # A softmax over equally uninformative features can have high maximum
        # probability merely because only a few shifts are valid. Certainty is
        # zero for that uniform distribution, so gate confidence before it
        # reaches either the learned maps or explicit summary statistics.
        # This makes empty structural inputs invariant to validity geometry.
        confidence = confidence * certainty
        augmented = torch.cat(
            [
                volume,
                displacement_for_map,
                confidence,
                certainty,
                x_coord,
                y_coord,
            ],
            dim=1,
        )
        descriptor = self.projection(self.pool(self.encoder(augmented)))

        # Convert feature-grid displacement to align_corners normalized units.
        height, width = volume.shape[-2:]
        dx = displacement_pixels[:, 0:1] * (2.0 / max(width - 1, 1))
        dy = displacement_pixels[:, 1:2] * (2.0 / max(height - 1, 1))
        weight = confidence
        denominator = weight.sum(dim=(2, 3)).clamp_min(1e-6)

        def weighted_mean(value: torch.Tensor) -> torch.Tensor:
            return (weight * value).sum(dim=(2, 3)) / denominator

        stats = torch.cat(
            [
                weighted_mean(dx),
                weighted_mean(dy),
                weighted_mean(dx * x_coord),
                weighted_mean(dx * y_coord),
                weighted_mean(dy * x_coord),
                weighted_mean(dy * y_coord),
                confidence.mean(dim=(2, 3)),
                certainty.mean(dim=(2, 3)),
            ],
            dim=1,
        )
        return descriptor, stats


class CorrelationVolumePairEncoder(nn.Module):
    """One shared FPN followed by multi-scale local cost volumes."""

    def __init__(
        self,
        structural_channels: int,
        latent_dim: int,
        encoder_base_channels: int,
        encoder_depth: int,
        feature_width: int,
        cost_hidden_channels: int,
        cost_volume_radii: Sequence[int],
        cost_pool_size: int,
        correlation_temperature: float,
        norm_type: str,
        require_input_evidence: bool = True,
        encoder_arch: str = "current",
        encoder_channels: Sequence[int] | None = None,
        encoder_blocks_per_stage: int | Sequence[int] | None = None,
        correlation_feature_width: int = DEFAULT_CORRELATION_FEATURE_WIDTH,
    ) -> None:
        super().__init__()
        if structural_channels < 1:
            raise ValueError("structural_channels must be positive")
        radii = tuple(int(radius) for radius in cost_volume_radii)
        if not radii:
            raise ValueError("cost_volume_radii cannot be empty")
        if (
            isinstance(correlation_feature_width, bool)
            or int(correlation_feature_width) != correlation_feature_width
            or int(correlation_feature_width) < 1
        ):
            raise ValueError("correlation_feature_width must be a positive integer")
        self.cost_volume_radii = radii
        self.structural_channels = int(structural_channels)
        self.require_input_evidence = bool(require_input_evidence)
        self.shared_encoder = FeaturePyramidEncoder(
            input_channels=self.structural_channels,
            encoder_base_channels=encoder_base_channels,
            encoder_depth=encoder_depth,
            feature_width=feature_width,
            norm_type=norm_type,
            num_cost_levels=len(radii),
            encoder_arch=encoder_arch,
            encoder_channels=encoder_channels,
            encoder_blocks_per_stage=encoder_blocks_per_stage,
        )
        self.correlation_feature_width = int(correlation_feature_width)
        if self.correlation_feature_width == feature_width:
            self.correlation_projections = nn.ModuleList(nn.Identity() for _ in radii)
        else:
            projections = [
                nn.Conv2d(
                    feature_width,
                    self.correlation_feature_width,
                    1,
                    bias=False,
                )
                for _ in radii
            ]
            for projection in projections:
                nn.init.kaiming_normal_(
                    projection.weight, mode="fan_out", nonlinearity="relu"
                )
            self.correlation_projections = nn.ModuleList(projections)
        self.correlations = nn.ModuleList(
            [
                LocalCorrelationVolume(radius, correlation_temperature)
                for radius in radii
            ]
        )
        self.aggregators = nn.ModuleList(
            [
                CostVolumeAggregator(
                    volume_channels=correlation.channels,
                    hidden_channels=cost_hidden_channels,
                    output_dim=feature_width,
                    pool_size=cost_pool_size,
                    norm_type=norm_type,
                    radius=correlation.radius,
                )
                for correlation in self.correlations
            ]
        )
        fused_dim = len(radii) * (feature_width + CostVolumeAggregator.stats_dim)
        self.fusion = nn.Sequential(
            nn.Linear(fused_dim, latent_dim),
            nn.LayerNorm(latent_dim),
            nn.GELU(),
            nn.Linear(latent_dim, latent_dim),
            nn.LayerNorm(latent_dim),
            nn.GELU(),
        )

    @staticmethod
    def _downsample_validity(
        validity: torch.Tensor | None,
        pyramid: Sequence[torch.Tensor],
    ) -> list[torch.Tensor | None]:
        if validity is None:
            return [None] * len(pyramid)
        return [
            F.adaptive_max_pool2d(validity.float(), feature.shape[-2:]) > 0.5
            for feature in pyramid
        ]

    def forward(
        self,
        group_structure: torch.Tensor,
        fixed_structure: torch.Tensor,
        group_structure_valid: torch.Tensor | None = None,
        fixed_structure_valid: torch.Tensor | None = None,
        return_displacement_stats: bool = False,
        return_aux: bool = False,
        _batch_aux_metadata: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, object]:
        if group_structure.ndim != 4 or fixed_structure.ndim != 4:
            raise ValueError("Structural descriptors must be BCHW tensors")
        if group_structure.shape != fixed_structure.shape:
            raise ValueError(
                "Group and fixed structural descriptors must have identical shapes; "
                f"got {tuple(group_structure.shape)} and "
                f"{tuple(fixed_structure.shape)}"
            )
        if group_structure.shape[1] != self.structural_channels:
            raise ValueError(
                f"Expected {self.structural_channels} structural channels, got "
                f"{group_structure.shape[1]}"
            )

        if self.require_input_evidence:
            # Preserve the original structural path exactly: descriptor
            # magnitude is checked again immediately before shared encoding.
            group_structure_valid = _validity_with_structural_evidence(
                group_structure,
                group_structure_valid,
                "group_structure_valid",
            )
            fixed_structure_valid = _validity_with_structural_evidence(
                fixed_structure,
                fixed_structure_valid,
                "fixed_structure_valid",
            )
        else:
            # Raw/hybrid masks were derived from the unadapted image tensors.
            # Never let learned adapter magnitude redefine their evidence.
            group_structure_valid = _canonical_validity_mask(
                group_structure_valid,
                group_structure,
                "group_structure_valid",
            )
            fixed_structure_valid = _canonical_validity_mask(
                fixed_structure_valid,
                fixed_structure,
                "fixed_structure_valid",
            )
            if group_structure_valid is None:
                group_structure_valid = torch.ones_like(
                    group_structure[:, :1], dtype=torch.bool
                )
            if fixed_structure_valid is None:
                fixed_structure_valid = torch.ones_like(
                    fixed_structure[:, :1], dtype=torch.bool
                )
        if group_structure_valid is not None:
            group_structure = group_structure * group_structure_valid.to(
                dtype=group_structure.dtype
            )
        if fixed_structure_valid is not None:
            fixed_structure = fixed_structure * fixed_structure_valid.to(
                dtype=fixed_structure.dtype
            )

        # Encode both domains in one call through the exact same module. In
        # addition to tying weights, the joint batch gives BatchNorm the same
        # statistics for group and fixed structures instead of making its
        # running state depend on call order.
        batch_size = group_structure.shape[0]
        joint_pyramid = self.shared_encoder(
            torch.cat((group_structure, fixed_structure), dim=0)
        )
        group_pyramid = [feature[:batch_size] for feature in joint_pyramid]
        fixed_pyramid = [feature[batch_size:] for feature in joint_pyramid]
        group_valid_pyramid = self._downsample_validity(
            group_structure_valid, group_pyramid
        )
        fixed_valid_pyramid = self._downsample_validity(
            fixed_structure_valid, fixed_pyramid
        )
        descriptors = []
        displacement_stats = []
        levels: list[dict[str, object]] = []
        for (
            group_feature,
            fixed_feature,
            group_valid,
            fixed_valid,
            projection,
            correlation,
            aggregator,
        ) in zip(
            group_pyramid,
            fixed_pyramid,
            group_valid_pyramid,
            fixed_valid_pyramid,
            self.correlation_projections,
            self.correlations,
            self.aggregators,
        ):
            # The one projection module is shared between both Siamese sides.
            # Joint projection also keeps any future stateful projection layer
            # independent of fixed/moving call order.
            projected = projection(torch.cat((group_feature, fixed_feature), dim=0))
            group_feature = projected[:batch_size]
            fixed_feature = projected[batch_size:]
            if group_valid is not None:
                group_feature = group_feature * group_valid.to(group_feature.dtype)
            if fixed_valid is not None:
                fixed_feature = fixed_feature * fixed_valid.to(fixed_feature.dtype)
            correlation_output = correlation(
                fixed_feature,
                group_feature,
                fixed_valid=fixed_valid,
                moving_valid=group_valid,
                return_aux=return_aux,
                _batch_aux_metadata=_batch_aux_metadata,
            )
            if return_aux:
                (
                    volume,
                    displacement,
                    confidence,
                    certainty,
                    level_aux,
                ) = correlation_output
                levels.append(level_aux)
            else:
                volume, displacement, confidence, certainty = correlation_output
            descriptor, stats = aggregator(volume, displacement, confidence, certainty)
            descriptors.extend((descriptor, stats))
            displacement_stats.append(stats)
        latent = self.fusion(torch.cat(descriptors, dim=1))
        stacked_stats = torch.stack(displacement_stats, dim=1)
        if return_aux:
            auxiliary: dict[str, object] = {"levels": levels}
            if return_displacement_stats:
                auxiliary["displacement_stats"] = stacked_stats
            return latent, auxiliary
        if return_displacement_stats:
            return latent, stacked_stats
        return latent


def coarse_similarity_from_cost_stats(
    stats: torch.Tensor,
    scale_range: Tuple[float, float],
    translation_limit: float,
    max_rotation_degrees: float,
) -> torch.Tensor:
    """Estimate a bounded fixed->moving similarity from cost-volume moments.

    ``stats`` is ``BxLx8`` in :class:`CostVolumeAggregator` order. Under a
    centered, approximately uniform support, the four displacement/coordinate
    moments are the least-squares similarity coefficients. Confidence-weighted
    fusion across pyramid levels makes an uninformative volume return identity.
    The coarse scale is deliberately isotropic: ``coarse @ residual`` then
    remains exactly representable by the five-parameter ``R @ diag`` family.
    """
    if stats.ndim != 3 or stats.shape[-1] != CostVolumeAggregator.stats_dim:
        raise ValueError("stats must have shape BxLx8")
    scale_min, scale_max = map(float, scale_range)
    if not 0.0 < scale_min < 1.0 < scale_max:
        raise ValueError("scale_range must strictly contain identity scale 1")
    translation_limit = float(translation_limit)
    max_rotation = math.radians(float(max_rotation_degrees))
    if translation_limit <= 0.0 or max_rotation <= 0.0:
        raise ValueError("translation and rotation limits must be positive")

    finite = torch.where(torch.isfinite(stats), stats, torch.zeros_like(stats))
    reliability = finite[..., 6].clamp_min(0.0)
    denominator = reliability.sum(dim=1, keepdim=True)
    normalized_weight = torch.where(
        denominator > 1e-8,
        reliability / denominator.clamp_min(1e-8),
        torch.zeros_like(reliability),
    )
    moments = (normalized_weight[..., None] * finite[..., :6]).sum(dim=1)
    tx, ty, dx_x, dx_y, dy_x, dy_y = moments.unbind(dim=1)

    # For x,y uniform on [-1,1], E[x^2]=E[y^2]=1/3. Project the
    # displacement field onto u=(sR-I)x+t without introducing shear.
    a = 1.0 + 1.5 * (dx_x + dy_y)
    b = 1.5 * (dy_x - dx_y)
    # Leave a small learnable margin on both sides. A proposal exactly at a
    # hard limit would otherwise give a zero-initialized residual branch no
    # room (and therefore no local gradient) in one direction.
    interior_fraction = 0.01
    coarse_translation_limit = translation_limit * (1.0 - interior_fraction)
    coarse_rotation_limit = max_rotation * (1.0 - interior_fraction)
    scale_margin = (scale_max - scale_min) * interior_fraction
    theta = torch.atan2(b, a).clamp(-coarse_rotation_limit, coarse_rotation_limit)
    scale = torch.sqrt(a.square() + b.square()).clamp(
        scale_min + scale_margin, scale_max - scale_margin
    )
    coarse = torch.stack(
        (
            tx.clamp(-coarse_translation_limit, coarse_translation_limit),
            ty.clamp(-coarse_translation_limit, coarse_translation_limit),
            theta,
            scale,
            scale,
        ),
        dim=1,
    )
    identity = coarse.new_tensor((0.0, 0.0, 0.0, 1.0, 1.0))
    valid = (denominator[:, 0] > 1e-8) & torch.isfinite(coarse).all(dim=1)
    return torch.where(valid[:, None], coarse, identity[None, :])


def _parameters_to_matrix(params: torch.Tensor) -> torch.Tensor:
    tx, ty, theta, sx, sy = params.unbind(dim=1)
    cosine = torch.cos(theta)
    sine = torch.sin(theta)
    return torch.stack(
        (
            torch.stack((sx * cosine, -sy * sine, tx), dim=1),
            torch.stack((sx * sine, sy * cosine, ty), dim=1),
        ),
        dim=1,
    )


def compose_similarity_and_residual_affine(
    coarse_params: torch.Tensor,
    residual_params: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return ``coarse @ residual`` as five parameters and a 2x3 matrix.

    The coarse linear transform must be a similarity (``sx == sy``). This
    order applies the residual in output/fixed coordinates before the coarse
    fixed-to-moving map and cannot create shear, even when the residual scale
    is anisotropic.
    """
    if coarse_params.ndim != 2 or coarse_params.shape[1] != 5:
        raise ValueError("coarse_params must have shape Bx5")
    if residual_params.shape != coarse_params.shape:
        raise ValueError("residual_params must match coarse_params shape")
    if not torch.allclose(
        coarse_params[:, 3], coarse_params[:, 4], rtol=1e-5, atol=1e-6
    ):
        raise ValueError("coarse affine must have isotropic scale")

    coarse_matrix = _parameters_to_matrix(coarse_params)
    residual_matrix = _parameters_to_matrix(residual_params)
    batch_size = coarse_params.shape[0]
    bottom = coarse_params.new_tensor((0.0, 0.0, 1.0)).view(1, 1, 3)
    bottom = bottom.expand(batch_size, -1, -1)
    coarse_h = torch.cat((coarse_matrix, bottom), dim=1)
    residual_h = torch.cat((residual_matrix, bottom), dim=1)
    matrix = torch.bmm(coarse_h, residual_h)[:, :2]

    sx = torch.linalg.vector_norm(matrix[:, :, 0], dim=1)
    sy = torch.linalg.vector_norm(matrix[:, :, 1], dim=1)
    theta = torch.atan2(matrix[:, 1, 0], matrix[:, 0, 0])
    params = torch.stack((matrix[:, 0, 2], matrix[:, 1, 2], theta, sx, sy), dim=1)
    return params, matrix


class AffineHead(nn.Module):
    """Bounded affine regressor with exact identity initialization."""

    def __init__(
        self,
        input_dim: int,
        scale_range: Tuple[float, float],
        translation_limit: float,
        max_rotation_degrees: float,
    ) -> None:
        super().__init__()
        self.scale_min, self.scale_max = map(float, scale_range)
        if not 0.0 < self.scale_min < 1.0 < self.scale_max:
            raise ValueError("scale_range must strictly contain identity scale 1")
        self.translation_limit = float(translation_limit)
        self.max_rotation = math.radians(float(max_rotation_degrees))
        if self.translation_limit <= 0.0 or self.max_rotation <= 0.0:
            raise ValueError("translation and rotation limits must be positive")
        self.body = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.LayerNorm(256),
            nn.GELU(),
            nn.Linear(256, 128),
            nn.LayerNorm(128),
            nn.GELU(),
        )
        self.output = nn.Linear(128, 5)
        nn.init.zeros_(self.output.weight)
        nn.init.zeros_(self.output.bias)
        identity_fraction = (1.0 - self.scale_min) / (self.scale_max - self.scale_min)
        identity_logit = math.log(identity_fraction / (1.0 - identity_fraction))
        with torch.no_grad():
            self.output.bias[3:5].fill_(identity_logit)

    def forward(self, feature: torch.Tensor) -> torch.Tensor:
        raw = self.output(self.body(feature))
        translation = self.translation_limit * torch.tanh(raw[:, :2])
        rotation = self.max_rotation * torch.tanh(raw[:, 2:3])
        scales = torch.sigmoid(raw[:, 3:5])
        scales = self.scale_min + (self.scale_max - self.scale_min) * scales
        return torch.cat((translation, rotation, scales), dim=1)


class _AffineComponentHead(nn.Module):
    """Independent geometry-specific MLP with a zero-initialized output."""

    def __init__(self, input_dim: int, output_dim: int) -> None:
        super().__init__()
        self.body = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.LayerNorm(128),
            nn.GELU(),
        )
        self.output = nn.Linear(128, output_dim)
        nn.init.zeros_(self.output.weight)
        nn.init.zeros_(self.output.bias)

    def forward(self, feature: torch.Tensor) -> torch.Tensor:
        return self.output(self.body(feature))


class SeparatedAffineHead(nn.Module):
    """Independent bounded translation, rotation, and scale regressors."""

    def __init__(
        self,
        input_dim: int,
        scale_range: Tuple[float, float],
        translation_limit: float,
        max_rotation_degrees: float,
    ) -> None:
        super().__init__()
        self.scale_min, self.scale_max = map(float, scale_range)
        if not 0.0 < self.scale_min < 1.0 < self.scale_max:
            raise ValueError("scale_range must strictly contain identity scale 1")
        self.translation_limit = float(translation_limit)
        self.max_rotation = math.radians(float(max_rotation_degrees))
        if self.translation_limit <= 0.0 or self.max_rotation <= 0.0:
            raise ValueError("translation and rotation limits must be positive")
        self.translation_head = _AffineComponentHead(input_dim, 2)
        self.rotation_head = _AffineComponentHead(input_dim, 1)
        self.scale_head = _AffineComponentHead(input_dim, 2)
        identity_fraction = (1.0 - self.scale_min) / (self.scale_max - self.scale_min)
        identity_logit = math.log(identity_fraction / (1.0 - identity_fraction))
        with torch.no_grad():
            self.scale_head.output.bias.fill_(identity_logit)

    def forward(self, feature: torch.Tensor) -> torch.Tensor:
        translation = self.translation_limit * torch.tanh(
            self.translation_head(feature)
        )
        rotation = self.max_rotation * torch.tanh(self.rotation_head(feature))
        scales = torch.sigmoid(self.scale_head(feature))
        scales = self.scale_min + (self.scale_max - self.scale_min) * scales
        return torch.cat((translation, rotation, scales), dim=1)


class SeparatedResidualAffineHead(nn.Module):
    """Separated corrections around a bounded coarse similarity proposal."""

    def __init__(
        self,
        input_dim: int,
        scale_range: Tuple[float, float],
        translation_limit: float,
        max_rotation_degrees: float,
    ) -> None:
        super().__init__()
        self.scale_min, self.scale_max = map(float, scale_range)
        if not 0.0 < self.scale_min < 1.0 < self.scale_max:
            raise ValueError("scale_range must strictly contain identity scale 1")
        self.translation_limit = float(translation_limit)
        self.max_rotation = math.radians(float(max_rotation_degrees))
        if self.translation_limit <= 0.0 or self.max_rotation <= 0.0:
            raise ValueError("translation and rotation limits must be positive")
        self.translation_head = _AffineComponentHead(input_dim, 2)
        self.rotation_head = _AffineComponentHead(input_dim, 1)
        self.scale_head = _AffineComponentHead(input_dim, 2)

    @staticmethod
    def _bounded_update(
        center: torch.Tensor,
        raw: torch.Tensor,
        lower: float,
        upper: float,
    ) -> torch.Tensor:
        """Use only the available room on the selected side of ``center``."""
        direction = torch.tanh(raw)
        positive_room = upper - center
        negative_room = center - lower
        # This algebra is equivalent to the two directional branches away
        # from zero, but gives the exact zero initialization the average-room
        # derivative instead of arbitrarily choosing only one side.
        average_room = 0.5 * (positive_room + negative_room)
        room_asymmetry = 0.5 * (positive_room - negative_room)
        delta = direction * average_room + direction.abs() * room_asymmetry
        return (center + delta).clamp(lower, upper)

    def _predict(
        self, feature: torch.Tensor, coarse_params: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if coarse_params.ndim != 2 or coarse_params.shape != (feature.shape[0], 5):
            raise ValueError("coarse_params must have shape Bx5")

        final_translation = self._bounded_update(
            coarse_params[:, :2],
            self.translation_head(feature),
            -self.translation_limit,
            self.translation_limit,
        )
        final_rotation = self._bounded_update(
            coarse_params[:, 2:3],
            self.rotation_head(feature),
            -self.max_rotation,
            self.max_rotation,
        )
        final_scales = self._bounded_update(
            coarse_params[:, 3:5],
            self.scale_head(feature),
            self.scale_min,
            self.scale_max,
        )
        final_params = torch.cat(
            (final_translation, final_rotation, final_scales), dim=1
        )

        # Derive the residual in coarse coordinates, then compose explicitly.
        # With isotropic coarse scale this is exactly shear-free.
        delta = final_translation - coarse_params[:, :2]
        coarse_theta = coarse_params[:, 2:3]
        cosine = torch.cos(coarse_theta)
        sine = torch.sin(coarse_theta)
        coarse_scale = coarse_params[:, 3:4].clamp_min(1e-6)
        residual_tx = (cosine * delta[:, 0:1] + sine * delta[:, 1:2]) / coarse_scale
        residual_ty = (-sine * delta[:, 0:1] + cosine * delta[:, 1:2]) / coarse_scale
        residual_params = torch.cat(
            (
                residual_tx,
                residual_ty,
                final_rotation - coarse_theta,
                final_scales / coarse_scale,
            ),
            dim=1,
        )
        _, final_matrix = compose_similarity_and_residual_affine(
            coarse_params, residual_params
        )
        return final_params, final_matrix

    def forward(
        self, feature: torch.Tensor, coarse_params: torch.Tensor
    ) -> torch.Tensor:
        params, _ = self._predict(feature, coarse_params)
        return params

    def forward_with_matrix(
        self, feature: torch.Tensor, coarse_params: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Expose the exact composed matrix without changing model.forward."""
        return self._predict(feature, coarse_params)


class CorrelationVolumeAffineRegistrationModel(nn.Module):
    """Predict one group-wide affine through a structural/raw/hybrid frontend."""

    def __init__(
        self,
        input_channels: int = 3,
        structural_channels: int = 6,
        frontend_mode: str = "structural",
        group_input_mode: str = "overlay",
        group_slots: int = DEFAULT_GROUP_SLOTS,
        latent_dim: int = 384,
        group_embedding_dim: int = 32,
        use_group_embedding: bool = True,
        num_groups: int = 5,
        scale_range: Tuple[float, float] = (0.8, 1.2),
        translation_limit: float = 0.5,
        max_rotation_degrees: float = 20.0,
        encoder_base_channels: int = 24,
        encoder_depth: int = 5,
        encoder_arch: str = "current",
        encoder_channels: Sequence[int] | None = None,
        encoder_blocks_per_stage: int | Sequence[int] | None = None,
        feature_width: int = 48,
        correlation_feature_width: int = DEFAULT_CORRELATION_FEATURE_WIDTH,
        cost_hidden_channels: int = 48,
        cost_volume_radii: Sequence[int] = (4, 4, 4),
        cost_pool_size: int = 4,
        correlation_temperature: float = 0.07,
        norm_type: str = "group",
        structural_descriptor_version: str = STRUCTURAL_DESCRIPTOR_VERSION,
        structural_foreground_threshold: float | None = None,
        structural_distance_scale: float = 0.03,
        structural_context_scale: float = 0.03,
        structural_skeleton_radius: int = 4,
        sfo_mode: str = "rgb",
        force_group1_identity: bool = True,
        affine_head_mode: str = "joint",
        separate_group_heads: bool = True,
        separate_group_adapters: bool = True,
        student_fixed_adapter: str = "none",
    ) -> None:
        super().__init__()
        if num_groups < 1:
            raise ValueError("num_groups must be positive")
        if frontend_mode not in FRONTEND_MODES:
            choices = ", ".join(FRONTEND_MODES)
            raise ValueError(f"frontend_mode must be one of: {choices}")
        if group_input_mode not in GROUP_INPUT_MODES:
            choices = ", ".join(GROUP_INPUT_MODES)
            raise ValueError(f"group_input_mode must be one of: {choices}")
        if affine_head_mode not in AFFINE_HEAD_MODES:
            choices = ", ".join(AFFINE_HEAD_MODES)
            raise ValueError(f"affine_head_mode must be one of: {choices}")
        if encoder_arch not in ENCODER_ARCHES:
            choices = ", ".join(ENCODER_ARCHES)
            raise ValueError(f"encoder_arch must be one of: {choices}")
        if student_fixed_adapter not in STUDENT_FIXED_ADAPTER_MODES:
            choices = ", ".join(STUDENT_FIXED_ADAPTER_MODES)
            raise ValueError(f"student_fixed_adapter must be one of: {choices}")
        if (
            isinstance(group_slots, bool)
            or int(group_slots) != group_slots
            or int(group_slots) < 1
        ):
            raise ValueError("group_slots must be a positive integer")
        if structural_descriptor_version != STRUCTURAL_DESCRIPTOR_VERSION:
            raise ValueError(
                "structural_descriptor_version must be "
                f"{STRUCTURAL_DESCRIPTOR_VERSION!r}"
            )
        if structural_channels != len(STRUCTURAL_CHANNEL_NAMES):
            raise ValueError(
                f"{STRUCTURAL_DESCRIPTOR_VERSION} requires exactly "
                f"{len(STRUCTURAL_CHANNEL_NAMES)} structural channels"
            )

        self.input_channels = int(input_channels)
        self.num_groups = int(num_groups)
        self.structural_channels = int(structural_channels)
        self.frontend_mode = frontend_mode
        self.group_input_mode = group_input_mode
        self.group_slots = int(group_slots)
        self.structural_descriptor_version = structural_descriptor_version
        self.structural_foreground_threshold = (
            None
            if structural_foreground_threshold is None
            else float(structural_foreground_threshold)
        )
        self.structural_distance_scale = float(structural_distance_scale)
        self.structural_context_scale = float(structural_context_scale)
        self.structural_skeleton_radius = int(structural_skeleton_radius)
        self.sfo_mode = sfo_mode
        self.use_group_embedding = bool(use_group_embedding)
        self.force_group1_identity = bool(force_group1_identity)
        self.affine_head_mode = affine_head_mode
        self.encoder_arch = encoder_arch
        self.scale_range = tuple(map(float, scale_range))
        self.translation_limit = float(translation_limit)
        self.max_rotation_degrees = float(max_rotation_degrees)
        self.separate_group_heads = bool(separate_group_heads)
        self.separate_group_adapters = bool(separate_group_adapters)
        self.student_fixed_adapter = student_fixed_adapter

        self.structural_frontend = OnlineStructuralFrontend(
            input_channels=self.input_channels,
            sfo_mode=self.sfo_mode,
            foreground_threshold=self.structural_foreground_threshold,
            distance_scale=self.structural_distance_scale,
            context_scale=self.structural_context_scale,
            skeleton_radius=self.structural_skeleton_radius,
        )
        if self.frontend_mode == "structural":
            # These attributes intentionally register no modules or parameters.
            # Consequently old structural checkpoints retain exactly their
            # original state-dict keys and tensor shapes.
            self.frontend_channels = self.structural_channels
            self.mineral_frontend_adapter = None
            self.group_frontend_adapter = None
        else:
            self.frontend_channels = int(feature_width)
            raw_group_channels = (
                self.input_channels
                if self.group_input_mode == "overlay"
                else self.group_slots * self.input_channels
            )
            descriptor_channels = (
                self.structural_channels if self.frontend_mode == "hybrid" else 0
            )
            self.mineral_frontend_adapter = FrontendInputAdapter(
                self.input_channels + descriptor_channels,
                self.frontend_channels,
                norm_type,
            )
            self.group_frontend_adapter = FrontendInputAdapter(
                raw_group_channels + descriptor_channels,
                self.frontend_channels,
                norm_type,
            )
        self.group_adapters = (
            nn.ModuleList(
                [
                    ResidualInputAdapter(self.frontend_channels, norm_type)
                    for _ in range(self.num_groups)
                ]
            )
            if self.separate_group_adapters
            else None
        )
        self.student_fixed_group_adapters = nn.ModuleDict(
            {
                f"G{group_id}": StudentFixedGroupAdapter(self.frontend_channels)
                for group_id in range(2, 6)
            }
            if self.student_fixed_adapter == "separate"
            else {}
        )
        self.encoder = CorrelationVolumePairEncoder(
            structural_channels=self.frontend_channels,
            latent_dim=latent_dim,
            encoder_base_channels=encoder_base_channels,
            encoder_depth=encoder_depth,
            feature_width=feature_width,
            encoder_arch=encoder_arch,
            encoder_channels=encoder_channels,
            encoder_blocks_per_stage=encoder_blocks_per_stage,
            correlation_feature_width=correlation_feature_width,
            cost_hidden_channels=cost_hidden_channels,
            cost_volume_radii=cost_volume_radii,
            cost_pool_size=cost_pool_size,
            correlation_temperature=correlation_temperature,
            norm_type=norm_type,
            require_input_evidence=self.frontend_mode == "structural",
        )
        self.encoder_channels = self.encoder.shared_encoder.encoder_channels
        self.encoder_blocks_per_stage = (
            self.encoder.shared_encoder.encoder_blocks_per_stage
        )
        self.correlation_feature_width = self.encoder.correlation_feature_width
        if self.use_group_embedding:
            self.group_embedding = nn.Embedding(
                self.num_groups + 1, group_embedding_dim
            )
            head_dim = latent_dim + group_embedding_dim
        else:
            self.group_embedding = None
            head_dim = latent_dim

        head_class = {
            "joint": AffineHead,
            "separated": SeparatedAffineHead,
            "separated_residual": SeparatedResidualAffineHead,
        }[self.affine_head_mode]
        if self.separate_group_heads:
            self.heads = nn.ModuleList(
                [
                    head_class(
                        head_dim,
                        scale_range,
                        translation_limit,
                        max_rotation_degrees,
                    )
                    for _ in range(self.num_groups)
                ]
            )
            self.head = None
        else:
            self.heads = None
            self.head = head_class(
                head_dim, scale_range, translation_limit, max_rotation_degrees
            )

    @staticmethod
    def _route_by_group(
        values: torch.Tensor,
        group_ids: torch.Tensor,
        modules: nn.ModuleList,
        auxiliary: torch.Tensor | None = None,
    ) -> torch.Tensor:
        chunks = []
        indices_per_chunk = []
        for group_id in torch.unique(group_ids, sorted=True).tolist():
            indices = torch.nonzero(group_ids == int(group_id), as_tuple=False).reshape(
                -1
            )
            module = modules[int(group_id) - 1]
            if auxiliary is None:
                chunks.append(module(values[indices]))
            else:
                chunks.append(module(values[indices], auxiliary[indices]))
            indices_per_chunk.append(indices)
        combined_indices = torch.cat(indices_per_chunk)
        return torch.cat(chunks, dim=0)[torch.argsort(combined_indices)]

    def _route_student_fixed_by_group(
        self, values: torch.Tensor, group_ids: torch.Tensor
    ) -> torch.Tensor:
        """Apply independent G2-G5 student fixed adapters to mixed batches."""
        if len(self.student_fixed_group_adapters) == 0:
            return values
        chunks = []
        indices_per_chunk = []
        for group_id in torch.unique(group_ids, sorted=True).tolist():
            numeric_group_id = int(group_id)
            indices = torch.nonzero(
                group_ids == numeric_group_id, as_tuple=False
            ).reshape(-1)
            if numeric_group_id == 1:
                adapted = values[indices]
            else:
                key = f"G{numeric_group_id}"
                if key not in self.student_fixed_group_adapters:
                    raise ValueError(
                        "student fixed group adapters are defined only for G2-G5; "
                        f"received {key}"
                    )
                adapted = self.student_fixed_group_adapters[key](values[indices])
            chunks.append(adapted)
            indices_per_chunk.append(indices)
        combined_indices = torch.cat(indices_per_chunk)
        return torch.cat(chunks, dim=0)[torch.argsort(combined_indices)]

    def _validate_group_stack(
        self, group_stack: torch.Tensor, name: str
    ) -> tuple[int, int, int, int, int]:
        if group_stack.ndim != 5:
            raise ValueError(f"{name} must have shape BxKxCxHxW")
        if not group_stack.is_floating_point():
            raise TypeError(f"{name} must be a floating tensor")
        batch_size, slots, channels, height, width = group_stack.shape
        if slots < 1:
            raise ValueError(f"{name} must contain at least one stain slot")
        if self.group_input_mode == "stack" and slots != self.group_slots:
            raise ValueError(
                f"{name} has {slots} stain slots, but stack mode requires "
                f"group_slots={self.group_slots}"
            )
        if channels != self.input_channels:
            raise ValueError(
                f"Expected {self.input_channels} image channels, got {channels}"
            )
        if not bool(torch.isfinite(group_stack).all()):
            raise ValueError(f"{name} contains non-finite values")
        tolerance = 1e-4
        if (
            float(group_stack.detach().amin()) < -tolerance
            or float(group_stack.detach().amax()) > 1.0 + tolerance
        ):
            raise ValueError(f"{name} must contain model-space images in [0, 1]")
        return batch_size, slots, channels, height, width

    def _validate_group_ids(
        self,
        group: torch.Tensor,
        *,
        batch_size: int,
        device: torch.device,
    ) -> torch.Tensor:
        group_ids = group.reshape(-1)
        if group_ids.numel() != batch_size:
            raise ValueError("Image inputs and group must have equal batches")
        if group_ids.dtype not in (torch.int32, torch.int64):
            raise TypeError("group IDs must use an integer tensor dtype")
        if group_ids.device != device:
            raise ValueError("group IDs and images must share one device")
        group_ids = group_ids.long()
        if torch.any((group_ids < 1) | (group_ids > self.num_groups)):
            raise ValueError(f"group IDs must be in [1, {self.num_groups}]")
        return group_ids

    @staticmethod
    def _validate_fixed_mineral(
        fixed_mineral: torch.Tensor,
        *,
        expected_shape: tuple[int, int, int, int],
        device: torch.device,
    ) -> None:
        if fixed_mineral.ndim != 4:
            raise ValueError("fixed_mineral must have shape BxCxHxW")
        if not fixed_mineral.is_floating_point():
            raise TypeError("fixed_mineral must be a floating tensor")
        if tuple(fixed_mineral.shape) != expected_shape:
            raise ValueError(
                f"fixed_mineral shape {tuple(fixed_mineral.shape)} does not "
                f"match moving-group image shape {expected_shape}"
            )
        if fixed_mineral.device != device:
            raise ValueError("fixed_mineral and moving_group must share one device")
        if not bool(torch.isfinite(fixed_mineral).all()):
            raise ValueError("fixed_mineral contains non-finite values")
        tolerance = 1e-4
        if (
            float(fixed_mineral.detach().amin()) < -tolerance
            or float(fixed_mineral.detach().amax()) > 1.0 + tolerance
        ):
            raise ValueError("fixed_mineral must contain a model-space image in [0, 1]")

    def _raw_group_representation(
        self, group_stack: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Return raw stack/overlay channels and pre-adapter rectangular FOV."""
        batch_size, slots, channels, height, width = group_stack.shape
        flattened = group_stack.reshape(batch_size * slots, channels, height, width)
        slot_fov = _raw_rectangular_fov(flattened, "group_stack")
        slot_fov = slot_fov.reshape(batch_size, slots, 1, height, width)
        group_fov = slot_fov.any(dim=1)
        slot_present = slot_fov.flatten(2).any(dim=2)
        present_stack = group_stack * slot_present[:, :, None, None, None].to(
            group_stack.dtype
        )
        if self.group_input_mode == "overlay":
            raw_representation = present_stack.amax(dim=1)
        else:
            raw_representation = present_stack.reshape(
                batch_size, slots * channels, height, width
            )
        return raw_representation, group_fov

    def mineral_frontend_representation(
        self, fixed_mineral: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Build the configured student fixed representation and validity mask."""
        if fixed_mineral.ndim != 4:
            raise ValueError("fixed_mineral must have shape BxCxHxW")
        if fixed_mineral.shape[1] != self.input_channels:
            raise ValueError(
                f"Expected {self.input_channels} input channels, got "
                f"{fixed_mineral.shape[1]}"
            )
        if self.frontend_mode == "structural":
            return self.structural_frontend.mineral_descriptor(fixed_mineral)

        raw_fov = _raw_rectangular_fov(fixed_mineral, "fixed_mineral")
        frontend_input = fixed_mineral
        validity = raw_fov
        if self.frontend_mode == "hybrid":
            structural, structural_valid = self.structural_frontend.mineral_descriptor(
                fixed_mineral
            )
            frontend_input = torch.cat((fixed_mineral, structural), dim=1)
            validity = raw_fov | structural_valid
        if self.mineral_frontend_adapter is None:
            raise AssertionError("Raw/hybrid Mineral adapter is not initialized")
        representation = self.mineral_frontend_adapter(frontend_input)
        representation = representation * validity.to(representation.dtype)
        return representation, validity

    def group_frontend_representation(
        self,
        group_stack: torch.Tensor,
        group_ids: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Build the configured group representation and pre-adapter validity."""
        if group_stack.ndim != 5:
            raise ValueError("group_stack must have shape BxKxCxHxW")
        batch_size, slots, channels, _, _ = group_stack.shape
        if slots < 1:
            raise ValueError("group_stack must contain at least one stain slot")
        if channels != self.input_channels:
            raise ValueError(
                f"Expected {self.input_channels} input channels, got {channels}"
            )
        if group_ids.reshape(-1).numel() != batch_size:
            raise ValueError("group_stack and group_ids must have equal batches")
        if self.group_input_mode == "stack" and slots != self.group_slots:
            raise ValueError(
                f"group_stack has {slots} stain slots, but stack mode requires "
                f"group_slots={self.group_slots}"
            )
        if self.frontend_mode == "structural":
            return self.structural_frontend.group_descriptor(group_stack, group_ids)

        raw_representation, raw_fov = self._raw_group_representation(group_stack)
        frontend_input = raw_representation
        validity = raw_fov
        if self.frontend_mode == "hybrid":
            structural, structural_valid = self.structural_frontend.group_descriptor(
                group_stack, group_ids
            )
            frontend_input = torch.cat((raw_representation, structural), dim=1)
            validity = raw_fov | structural_valid
        if self.group_frontend_adapter is None:
            raise AssertionError("Raw/hybrid group adapter is not initialized")
        representation = self.group_frontend_adapter(frontend_input)
        representation = representation * validity.to(representation.dtype)
        return representation, validity

    def _predict_from_representations(
        self,
        *,
        fixed_representation: torch.Tensor,
        moving_representation: torch.Tensor,
        fixed_valid: torch.Tensor,
        moving_valid: torch.Tensor,
        group_ids: torch.Tensor,
        adapt_fixed_group: bool,
        return_aux: bool = False,
        _batch_aux_metadata: bool = False,
    ) -> torch.Tensor | dict[str, object]:
        if self.frontend_mode == "structural":
            # Keep the original pre-adapter descriptor-evidence gate verbatim.
            moving_valid = _validity_with_structural_evidence(
                moving_representation, moving_valid, "moving_structure_valid"
            )
            fixed_valid = _validity_with_structural_evidence(
                fixed_representation, fixed_valid, "fixed_structure_valid"
            )
        else:
            moving_valid = _canonical_validity_mask(
                moving_valid, moving_representation, "moving_frontend_valid"
            )
            fixed_valid = _canonical_validity_mask(
                fixed_valid, fixed_representation, "fixed_frontend_valid"
            )
            if moving_valid is None or fixed_valid is None:
                raise AssertionError("Raw/hybrid frontend validity is required")
        moving_representation = moving_representation * moving_valid.to(
            moving_representation.dtype
        )
        fixed_representation = fixed_representation * fixed_valid.to(
            fixed_representation.dtype
        )
        if self.group_adapters is not None:
            moving_representation = self._route_by_group(
                moving_representation, group_ids, self.group_adapters
            )
            moving_representation = moving_representation * moving_valid.to(
                moving_representation.dtype
            )
            if adapt_fixed_group:
                fixed_representation = self._route_by_group(
                    fixed_representation, group_ids, self.group_adapters
                )
                fixed_representation = fixed_representation * fixed_valid.to(
                    fixed_representation.dtype
                )

        if not adapt_fixed_group and len(self.student_fixed_group_adapters) > 0:
            fixed_representation = self._route_student_fixed_by_group(
                fixed_representation, group_ids
            )
            fixed_representation = fixed_representation * fixed_valid.to(
                fixed_representation.dtype
            )

        needs_displacement_stats = self.affine_head_mode == "separated_residual"
        encoder_result = self.encoder(
            moving_representation,
            fixed_representation,
            group_structure_valid=moving_valid,
            fixed_structure_valid=fixed_valid,
            return_displacement_stats=needs_displacement_stats,
            return_aux=return_aux,
            _batch_aux_metadata=_batch_aux_metadata,
        )
        encoder_aux: dict[str, object] | None = None
        if return_aux:
            latent, encoder_aux_value = encoder_result
            if not isinstance(encoder_aux_value, dict):
                raise AssertionError("Encoder auxiliary result must be a dictionary")
            encoder_aux = encoder_aux_value
            cost_stats = encoder_aux.get("displacement_stats")
        elif needs_displacement_stats:
            latent, cost_stats = encoder_result
        else:
            latent = encoder_result
            cost_stats = None

        if needs_displacement_stats:
            if not isinstance(cost_stats, torch.Tensor):
                raise AssertionError("Separated-residual mode requires cost statistics")
            coarse_params = coarse_similarity_from_cost_stats(
                cost_stats,
                self.scale_range,
                self.translation_limit,
                self.max_rotation_degrees,
            )
        else:
            coarse_params = None
        if self.group_embedding is not None:
            latent = torch.cat((latent, self.group_embedding(group_ids)), dim=1)
        if self.heads is not None:
            params = self._route_by_group(
                latent, group_ids, self.heads, auxiliary=coarse_params
            )
        else:
            assert self.head is not None
            if coarse_params is None:
                params = self.head(latent)
            else:
                params = self.head(latent, coarse_params)

        identity = params.new_tensor((0.0, 0.0, 0.0, 1.0, 1.0))
        has_pair_evidence = moving_valid.flatten(1).any(dim=1)
        has_pair_evidence = has_pair_evidence & fixed_valid.flatten(1).any(dim=1)
        force_identity = ~has_pair_evidence
        if self.force_group1_identity:
            force_identity = force_identity | (group_ids == 1)
        final_params = torch.where(force_identity[:, None], identity[None, :], params)
        if not return_aux:
            return final_params
        if encoder_aux is None:
            raise AssertionError("return_aux=True requires encoder auxiliary output")
        levels = encoder_aux.get("levels")
        if not isinstance(levels, list):
            raise AssertionError("Encoder auxiliary output is missing levels")
        return {"params": final_params, "levels": levels}

    def forward(
        self,
        fixed_mineral: torch.Tensor,
        moving_group: torch.Tensor,
        group: torch.Tensor,
        return_aux: bool = False,
        _batch_aux_metadata: bool = False,
    ) -> torch.Tensor | dict[str, object]:
        """Predict the deployable student affine without registered targets."""
        batch_size, _, channels, height, width = self._validate_group_stack(
            moving_group, "moving_group"
        )
        expected_fixed_shape = (batch_size, channels, height, width)
        self._validate_fixed_mineral(
            fixed_mineral,
            expected_shape=expected_fixed_shape,
            device=moving_group.device,
        )
        group_ids = self._validate_group_ids(
            group, batch_size=batch_size, device=moving_group.device
        )
        fixed_representation, fixed_valid = self.mineral_frontend_representation(
            fixed_mineral.float()
        )
        moving_representation, moving_valid = self.group_frontend_representation(
            moving_group.float(), group_ids
        )
        return self._predict_from_representations(
            fixed_representation=fixed_representation,
            moving_representation=moving_representation,
            fixed_valid=fixed_valid,
            moving_valid=moving_valid,
            group_ids=group_ids,
            adapt_fixed_group=False,
            return_aux=return_aux,
            _batch_aux_metadata=_batch_aux_metadata,
        )

    def fuse_teacher_fixed_representations(
        self,
        target_representation: torch.Tensor,
        mineral_representation: torch.Tensor,
        target_valid: torch.Tensor,
        mineral_valid: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Fuse the teacher registered-group and Mineral fixed evidence.

        The evidence-aware mean preserves either source where it is the only
        valid source and averages them where both are valid. This operation is
        deliberately parameter-free so existing teacher checkpoints retain
        exactly the same state-dict contract.
        """
        if tuple(target_representation.shape) != tuple(mineral_representation.shape):
            raise ValueError(
                "Teacher target and Mineral representations must have identical "
                "BxCxHxW shapes"
            )
        expected_valid_shape = (
            target_representation.shape[0],
            1,
            target_representation.shape[-2],
            target_representation.shape[-1],
        )
        if tuple(target_valid.shape) != expected_valid_shape:
            raise ValueError(
                "target_valid must match the representation batch and spatial shape"
            )
        if tuple(mineral_valid.shape) != expected_valid_shape:
            raise ValueError(
                "mineral_valid must match the representation batch and spatial shape"
            )

        target_weight = target_valid.to(dtype=target_representation.dtype)
        mineral_weight = mineral_valid.to(dtype=target_representation.dtype)
        denominator = (target_weight + mineral_weight).clamp_min(1.0)
        fused = (
            target_representation * target_weight
            + mineral_representation * mineral_weight
        ) / denominator
        fused_valid = target_valid.bool() | mineral_valid.bool()
        return fused * fused_valid.to(dtype=fused.dtype), fused_valid

    def teacher_fixed_frontend_representation(
        self,
        fixed_mineral: torch.Tensor,
        target_group: torch.Tensor,
        group_ids: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Build the teacher fixed side from Mineral and registered targets."""
        target_representation, target_valid = self.group_frontend_representation(
            target_group.float(), group_ids
        )
        mineral_representation, mineral_valid = self.mineral_frontend_representation(
            fixed_mineral.float()
        )
        return self.fuse_teacher_fixed_representations(
            target_representation,
            mineral_representation,
            target_valid,
            mineral_valid,
        )

    def forward_group_pair(
        self,
        fixed_mineral: torch.Tensor,
        target_group: torch.Tensor,
        moving_group: torch.Tensor,
        group: torch.Tensor,
        return_aux: bool = False,
        _batch_aux_metadata: bool = False,
    ) -> torch.Tensor | dict[str, object]:
        """Predict from Mineral, registered targets, and the moving group.

        This method is training-only. It is intentionally not called by the
        deployable student forward path.
        """
        batch_size, _, channels, height, width = self._validate_group_stack(
            moving_group, "moving_group"
        )
        self._validate_group_stack(target_group, "target_group")
        if tuple(target_group.shape) != tuple(moving_group.shape):
            raise ValueError(
                "target_group and moving_group must have identical BxKxCxHxW shapes"
            )
        if target_group.device != moving_group.device:
            raise ValueError("target_group and moving_group must share one device")
        self._validate_fixed_mineral(
            fixed_mineral,
            expected_shape=(batch_size, channels, height, width),
            device=moving_group.device,
        )
        group_ids = self._validate_group_ids(
            group, batch_size=batch_size, device=moving_group.device
        )
        fixed_representation, fixed_valid = self.teacher_fixed_frontend_representation(
            fixed_mineral,
            target_group,
            group_ids,
        )
        moving_representation, moving_valid = self.group_frontend_representation(
            moving_group.float(), group_ids
        )
        return self._predict_from_representations(
            fixed_representation=fixed_representation,
            moving_representation=moving_representation,
            fixed_valid=fixed_valid,
            moving_valid=moving_valid,
            group_ids=group_ids,
            adapt_fixed_group=True,
            return_aux=return_aux,
            _batch_aux_metadata=_batch_aux_metadata,
        )


class GroupPairCorrelationVolumeAffineRegistrationModel(
    CorrelationVolumeAffineRegistrationModel
):
    """Training-only teacher with a DataParallel-compatible forward.

    It deliberately omits student-only fixed-side adapters. Every remaining
    shared parameter name and tensor shape is initialized strictly from the student.
    """

    def __init__(
        self,
        *args: object,
        student_fixed_adapter: str = "none",
        **kwargs: object,
    ) -> None:
        if student_fixed_adapter not in STUDENT_FIXED_ADAPTER_MODES:
            choices = ", ".join(STUDENT_FIXED_ADAPTER_MODES)
            raise ValueError(f"student_fixed_adapter must be one of: {choices}")
        super().__init__(*args, student_fixed_adapter="none", **kwargs)

    def forward(
        self,
        fixed_mineral: torch.Tensor,
        target_group: torch.Tensor,
        moving_group: torch.Tensor,
        group: torch.Tensor,
        return_aux: bool = False,
        _batch_aux_metadata: bool = False,
    ) -> torch.Tensor | dict[str, object]:
        return self.forward_group_pair(
            fixed_mineral,
            target_group,
            moving_group,
            group,
            return_aux=return_aux,
            _batch_aux_metadata=_batch_aux_metadata,
        )


def _canonicalize_auxiliary_metadata(
    output: torch.Tensor | dict[str, object],
) -> torch.Tensor | dict[str, object]:
    """Undo batch-shaped metadata used for safe ``nn.DataParallel`` gather."""
    if not isinstance(output, dict):
        return output
    levels = output.get("levels")
    if not isinstance(levels, list):
        raise ValueError("Auxiliary branch output must contain a levels list")
    canonical_levels: list[dict[str, object]] = []
    for level in levels:
        if not isinstance(level, dict):
            raise ValueError("Every auxiliary correlation level must be a dictionary")
        canonical = dict(level)
        displacements = canonical.get("displacements")
        if isinstance(displacements, torch.Tensor) and displacements.ndim == 3:
            if displacements.shape[0] < 1 or not torch.equal(
                displacements, displacements[:1].expand_as(displacements)
            ):
                raise ValueError(
                    "DataParallel displacement metadata differs across samples"
                )
            canonical["displacements"] = displacements[0]
        feature_size = canonical.get("feature_size")
        if isinstance(feature_size, torch.Tensor):
            if feature_size.ndim != 2 or feature_size.shape[1] != 2:
                raise ValueError("Batched feature_size metadata must have shape Bx2")
            if feature_size.shape[0] < 1 or not torch.equal(
                feature_size, feature_size[:1].expand_as(feature_size)
            ):
                raise ValueError(
                    "DataParallel feature sizes differ across branch replicas"
                )
            canonical["feature_size"] = [
                int(feature_size[0, 0]),
                int(feature_size[0, 1]),
            ]
        canonical_levels.append(canonical)
    result = dict(output)
    result["levels"] = canonical_levels
    return result


def _data_parallel_branch_for_batch(
    branch: nn.Module,
    batch_size: int,
) -> tuple[nn.Module, bool]:
    """Avoid empty replicas when a batch is smaller than the GPU count.

    Auxiliary forwards include scalar keyword arguments. ``nn.DataParallel``
    replicates those arguments to every device even when the tensor inputs
    produce fewer chunks, which can invoke an empty replica without the
    required image tensors. Small final train/validation batches therefore run
    directly on the primary module; gradients still reach the same parameters.
    """
    if not isinstance(branch, nn.DataParallel):
        return branch, False
    device_count = len(branch.device_ids)
    if device_count <= 1 or batch_size < device_count:
        return branch.module, False
    return branch, True


class TeacherStudentAffineRegistrationModel(nn.Module):
    """Own an inference-safe student and an optional independent teacher."""

    def __init__(
        self,
        *,
        student_config: dict | None = None,
        use_teacher_branch: bool = True,
        **student_config_overrides: object,
    ) -> None:
        super().__init__()
        config = dict(student_config or {})
        duplicate_keys = set(config).intersection(student_config_overrides)
        if duplicate_keys:
            names = ", ".join(sorted(duplicate_keys))
            raise ValueError(f"Duplicate student configuration keys: {names}")
        config.update(student_config_overrides)
        config = canonicalize_model_config(config)
        self.student_config = copy.deepcopy(config)
        self.use_teacher_branch = bool(use_teacher_branch)
        self.student = CorrelationVolumeAffineRegistrationModel(**config)
        self.teacher: GroupPairCorrelationVolumeAffineRegistrationModel | None
        if self.use_teacher_branch:
            self.teacher = GroupPairCorrelationVolumeAffineRegistrationModel(**config)
            self.initialize_teacher_from_student()
        else:
            self.teacher = None

    def initialize_teacher_from_student(self) -> None:
        """Strictly copy all shared weights while excluding student-only adapters."""
        if self.teacher is None:
            raise RuntimeError("Cannot initialize a disabled teacher branch")
        student_state = self.student.state_dict()
        shared_state = {
            key: value
            for key, value in student_state.items()
            if not key.startswith(STUDENT_FIXED_ADAPTER_STATE_PREFIX)
        }
        teacher_state = self.teacher.state_dict()
        missing = sorted(set(teacher_state).difference(shared_state))
        unexpected = sorted(set(shared_state).difference(teacher_state))
        shape_mismatches = sorted(
            key
            for key in set(shared_state).intersection(teacher_state)
            if tuple(shared_state[key].shape) != tuple(teacher_state[key].shape)
        )
        if missing or unexpected or shape_mismatches:
            details = []
            if missing:
                details.append(f"missing shared teacher keys: {missing}")
            if unexpected:
                details.append(f"unexpected shared student keys: {unexpected}")
            if shape_mismatches:
                details.append(f"shared shape mismatches: {shape_mismatches}")
            raise RuntimeError(
                "Teacher initialization rejected a non-adapter architecture "
                "mismatch; " + "; ".join(details)
            )
        self.teacher.load_state_dict(shared_state, strict=True)

    def forward(
        self,
        fixed_mineral: torch.Tensor,
        moving_group: torch.Tensor,
        group: torch.Tensor,
        return_aux: bool = False,
    ) -> torch.Tensor | dict[str, object]:
        """Run only the deployable student path."""
        student, batch_metadata = _data_parallel_branch_for_batch(
            self.student,
            fixed_mineral.shape[0],
        )
        if not return_aux:
            return student(fixed_mineral, moving_group, group)
        output = student(
            fixed_mineral,
            moving_group,
            group,
            return_aux=True,
            _batch_aux_metadata=batch_metadata,
        )
        return _canonicalize_auxiliary_metadata(output)

    def forward_teacher(
        self,
        fixed_mineral: torch.Tensor,
        target_group: torch.Tensor,
        moving_group: torch.Tensor,
        group: torch.Tensor,
        return_aux: bool = False,
    ) -> torch.Tensor | dict[str, object]:
        """Run the training-only Mineral + same-group teacher path."""
        if self.teacher is None:
            raise RuntimeError(
                "Teacher branch is disabled; construct with use_teacher_branch=True"
            )
        teacher, batch_metadata = _data_parallel_branch_for_batch(
            self.teacher,
            fixed_mineral.shape[0],
        )
        if not return_aux:
            return teacher(fixed_mineral, target_group, moving_group, group)
        output = teacher(
            fixed_mineral,
            target_group,
            moving_group,
            group,
            return_aux=True,
            _batch_aux_metadata=batch_metadata,
        )
        return _canonicalize_auxiliary_metadata(output)


# A short compatibility alias keeps external scripts that expect the original
# public class name easy to adapt while checkpoints use the explicit class.
GroupAffineRegistrationModel = CorrelationVolumeAffineRegistrationModel


__all__ = [
    "AFFINE_HEAD_MODES",
    "DEFAULT_CORRELATION_FEATURE_WIDTH",
    "DEFAULT_RESIDUAL_BLOCKS_PER_STAGE",
    "DEFAULT_RESIDUAL_ENCODER_CHANNELS",
    "ENCODER_ARCHES",
    "STUDENT_FIXED_ADAPTER_MODES",
    "STUDENT_FIXED_ADAPTER_STATE_PREFIX",
    "AffineHead",
    "CorrelationVolumeAffineRegistrationModel",
    "DEFAULT_GROUP_SLOTS",
    "FRONTEND_MODES",
    "FrontendInputAdapter",
    "GROUP_INPUT_MODES",
    "GroupPairCorrelationVolumeAffineRegistrationModel",
    "CorrelationVolumePairEncoder",
    "FeaturePyramidEncoder",
    "GroupAffineRegistrationModel",
    "LocalCorrelationVolume",
    "OnlineStructuralFrontend",
    "ResidualEncoderBlock",
    "ResidualEncoderStage",
    "SeparatedAffineHead",
    "SeparatedResidualAffineHead",
    "STRUCTURAL_CHANNEL_NAMES",
    "TEACHER_FIXED_INPUT_VERSION",
    "STRUCTURAL_DESCRIPTOR_VERSION",
    "STRUCTURAL_EVIDENCE_EPSILON",
    "StudentFixedGroupAdapter",
    "TeacherStudentAffineRegistrationModel",
    "canonicalize_model_config",
    "coarse_similarity_from_cost_stats",
    "compose_similarity_and_residual_affine",
]
