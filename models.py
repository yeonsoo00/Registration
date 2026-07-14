"""Affine registration model for one transform per stain acquisition group."""

from __future__ import annotations

import math
from typing import Tuple

import torch
import torch.nn as nn


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


class ConvStage(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, norm_type: str) -> None:
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, stride=2, padding=1, bias=False),
            _make_norm(out_channels, norm_type),
            nn.GELU(),
            nn.Conv2d(out_channels, out_channels, 3, padding=1, bias=False),
            _make_norm(out_channels, norm_type),
            nn.GELU(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class ResidualInputAdapter(nn.Module):
    """Small identity-initialized adapter specialized for one signal group."""

    def __init__(self, channels: int, norm_type: str) -> None:
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(channels, channels, 3, padding=1, bias=False),
            _make_norm(channels, norm_type),
            nn.GELU(),
            nn.Conv2d(channels, channels, 1, bias=True),
        )
        final = self.block[-1]
        nn.init.zeros_(final.weight)
        nn.init.zeros_(final.bias)

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        return image + self.block(image)


class GroupPairEncoder(nn.Module):
    """Encode a group representation against the fixed mineral image."""

    def __init__(
        self,
        group_input_channels: int,
        fixed_channels: int,
        latent_dim: int = 256,
        depth: int = 4,
        base_channels: int = 32,
        spatial_pool_size: int = 4,
        norm_type: str = "group",
        use_coordconv: bool = True,
        fusion_mode: str = "intermediate",
    ) -> None:
        super().__init__()
        if fusion_mode not in {"concat", "intermediate"}:
            raise ValueError("fusion_mode must be concat or intermediate")
        self.fusion_mode = fusion_mode
        self.use_coordconv = use_coordconv
        coord_channels = 2 if use_coordconv else 0
        channels = [base_channels * 2**i for i in range(depth)]

        if fusion_mode == "concat":
            self.joint = nn.ModuleList()
            in_ch = group_input_channels + fixed_channels + coord_channels
            for out_ch in channels:
                self.joint.append(ConvStage(in_ch, out_ch, norm_type))
                in_ch = out_ch
        else:
            self.group_stages = nn.ModuleList()
            self.fixed_stages = nn.ModuleList()
            self.fuse_stages = nn.ModuleList()
            group_in = group_input_channels + coord_channels
            fixed_in = fixed_channels + coord_channels
            for out_ch in channels:
                self.group_stages.append(ConvStage(group_in, out_ch, norm_type))
                self.fixed_stages.append(ConvStage(fixed_in, out_ch, norm_type))
                self.fuse_stages.append(
                    nn.Sequential(
                        nn.Conv2d(out_ch * 4, out_ch, 1, bias=False),
                        _make_norm(out_ch, norm_type),
                        nn.GELU(),
                    )
                )
                group_in = fixed_in = out_ch

        self.pool = nn.AdaptiveAvgPool2d((spatial_pool_size, spatial_pool_size))
        self.projection = nn.Sequential(
            nn.Flatten(),
            nn.Linear(channels[-1] * spatial_pool_size * spatial_pool_size, latent_dim),
            nn.LayerNorm(latent_dim),
            nn.GELU(),
            nn.Identity(),
        )

    @staticmethod
    def _coords(x: torch.Tensor) -> torch.Tensor:
        b, _, h, w = x.shape
        yy = torch.linspace(-1, 1, h, device=x.device, dtype=x.dtype)
        xx = torch.linspace(-1, 1, w, device=x.device, dtype=x.dtype)
        yy, xx = torch.meshgrid(yy, xx, indexing="ij")
        return torch.stack([xx, yy], dim=0).unsqueeze(0).expand(b, -1, -1, -1)

    def forward(self, group_input: torch.Tensor, fixed: torch.Tensor) -> torch.Tensor:
        coords = self._coords(group_input) if self.use_coordconv else None
        if self.fusion_mode == "concat":
            inputs = [group_input, fixed]
            if coords is not None:
                inputs.append(coords)
            feat = torch.cat(inputs, dim=1)
            for stage in self.joint:
                feat = stage(feat)
        else:
            gf = (
                torch.cat([group_input, coords], dim=1)
                if coords is not None
                else group_input
            )
            ff = torch.cat([fixed, coords], dim=1) if coords is not None else fixed
            fused = None
            for gs, fs, fuse in zip(
                self.group_stages, self.fixed_stages, self.fuse_stages
            ):
                gf = gs(gf)
                ff = fs(ff)
                fused = fuse(torch.cat([gf, ff, torch.abs(gf - ff), gf * ff], dim=1))
                gf = gf + fused
                ff = ff + fused
            assert fused is not None
            feat = fused
        return self.projection(self.pool(feat))


class AffineHead(nn.Module):
    def __init__(
        self,
        input_dim: int,
        scale_range: Tuple[float, float],
        translation_limit: float,
        max_rotation_degrees: float,
    ) -> None:
        super().__init__()
        self.scale_min, self.scale_max = scale_range
        self.translation_limit = float(translation_limit)
        self.max_rotation = math.radians(max_rotation_degrees)
        self.body = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.LayerNorm(256),
            nn.GELU(),
            nn.Identity(),
            nn.Linear(256, 128),
            nn.LayerNorm(128),
            nn.GELU(),
        )
        self.output = nn.Linear(128, 5)
        nn.init.zeros_(self.output.weight)
        nn.init.zeros_(self.output.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        raw = self.output(self.body(x))
        t = self.translation_limit * torch.tanh(raw[:, :2])
        r = self.max_rotation * torch.tanh(raw[:, 2:3])
        s = torch.sigmoid(raw[:, 3:5])
        s = self.scale_min + (self.scale_max - self.scale_min) * s
        return torch.cat([t, r, s], dim=1)


class GroupAffineRegistrationModel(nn.Module):
    """Predict exactly one affine matrix for each sample/group item.

    ``use_group_embedding`` optionally inserts a learned group-ID embedding into
    the affine regressor. The output is still one parameter vector per grouped
    dataset item, so every member stain receives the same transform.
    """

    def __init__(
        self,
        group_input_channels: int,
        fixed_channels: int = 3,
        latent_dim: int = 256,
        group_embedding_dim: int = 32,
        use_group_embedding: bool = False,
        num_groups: int = 5,
        scale_range: Tuple[float, float] = (0.8, 1.2),
        translation_limit: float = 0.5,
        max_rotation_degrees: float = 20.0,
        depth: int = 4,
        base_channels: int = 32,
        spatial_pool_size: int = 4,
        norm_type: str = "group",
        use_coordconv: bool = True,
        fusion_mode: str = "intermediate",
        force_group1_identity: bool = False,
        separate_group_heads: bool = False,
        separate_group_adapters: bool = False,
    ) -> None:
        super().__init__()
        self.use_group_embedding = use_group_embedding
        self.force_group1_identity = force_group1_identity
        self.separate_group_heads = separate_group_heads
        self.separate_group_adapters = separate_group_adapters
        self.num_groups = int(num_groups)
        self.group_adapters = (
            nn.ModuleList(
                [
                    ResidualInputAdapter(group_input_channels, norm_type)
                    for _ in range(self.num_groups)
                ]
            )
            if separate_group_adapters
            else None
        )
        self.encoder = GroupPairEncoder(
            group_input_channels=group_input_channels,
            fixed_channels=fixed_channels,
            latent_dim=latent_dim,
            depth=depth,
            base_channels=base_channels,
            spatial_pool_size=spatial_pool_size,
            norm_type=norm_type,
            use_coordconv=use_coordconv,
            fusion_mode=fusion_mode,
        )
        if use_group_embedding:
            self.group_embedding = nn.Embedding(num_groups + 1, group_embedding_dim)
            head_dim = latent_dim + group_embedding_dim
        else:
            self.group_embedding = None
            head_dim = latent_dim
        if separate_group_heads:
            self.heads = nn.ModuleList(
                [
                    AffineHead(
                        head_dim, scale_range, translation_limit, max_rotation_degrees
                    )
                    for _ in range(self.num_groups)
                ]
            )
            self.head = None
        else:
            self.heads = None
            self.head = AffineHead(
                head_dim, scale_range, translation_limit, max_rotation_degrees
            )

    def forward(
        self,
        group_input: torch.Tensor,
        fixed_mineral: torch.Tensor,
        group: torch.Tensor,
    ) -> torch.Tensor:
        if group_input.ndim != 4 or fixed_mineral.ndim != 4:
            raise ValueError("group_input and fixed_mineral must be BCHW tensors")
        group_ids = group.reshape(-1)
        batch_size = group_input.shape[0]
        if fixed_mineral.shape[0] != batch_size or group_ids.numel() != batch_size:
            raise ValueError(
                "group_input, fixed_mineral, and group must have equal batch sizes"
            )
        if group_ids.dtype not in (torch.int32, torch.int64):
            raise TypeError("group IDs must use an integer tensor dtype")
        group_ids = group_ids.long()
        if torch.any((group_ids < 1) | (group_ids > self.num_groups)):
            raise ValueError(f"group IDs must be in [1, {self.num_groups}]")
        if self.group_adapters is not None:
            adapted_chunks = []
            index_chunks = []
            for group_id in torch.unique(group_ids).tolist():
                indices = torch.nonzero(
                    group_ids == int(group_id), as_tuple=False
                ).reshape(-1)
                adapted_chunks.append(
                    self.group_adapters[int(group_id) - 1](group_input[indices])
                )
                index_chunks.append(indices)
            combined_indices = torch.cat(index_chunks)
            restore_order = torch.argsort(combined_indices)
            group_input = torch.cat(adapted_chunks, dim=0)[restore_order]
        latent = self.encoder(group_input, fixed_mineral)
        if self.group_embedding is not None:
            latent = torch.cat([latent, self.group_embedding(group_ids)], dim=1)
        if self.heads is None:
            assert self.head is not None
            params = self.head(latent)
        else:
            parameter_chunks = []
            index_chunks = []
            for group_id in torch.unique(group_ids).tolist():
                indices = torch.nonzero(
                    group_ids == int(group_id), as_tuple=False
                ).reshape(-1)
                parameter_chunks.append(self.heads[int(group_id) - 1](latent[indices]))
                index_chunks.append(indices)
            combined_indices = torch.cat(index_chunks)
            restore_order = torch.argsort(combined_indices)
            params = torch.cat(parameter_chunks, dim=0)[restore_order]
        if self.force_group1_identity:
            identity = params.new_tensor([0.0, 0.0, 0.0, 1.0, 1.0])
            mask = group.reshape(-1) == 1
            if mask.any():
                params = torch.where(mask[:, None], identity[None, :], params)
        return params
