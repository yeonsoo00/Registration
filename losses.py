"""Losses for affine registration of pseudo-coloured histology images."""

from __future__ import annotations

from typing import Optional, Sequence

import torch
import torch.nn.functional as F


def _expand_weight(weight: torch.Tensor, reference: torch.Tensor) -> torch.Tensor:
    weight = weight.to(device=reference.device, dtype=reference.dtype)
    if weight.shape[1] == 1 and reference.shape[1] > 1:
        weight = weight.expand(-1, reference.shape[1], -1, -1)
    return weight


def _masked_mean(values: torch.Tensor, weight: Optional[torch.Tensor], eps: float = 1e-8) -> torch.Tensor:
    if weight is None:
        return values.mean()
    weight = _expand_weight(weight, values)
    return (values * weight).sum() / (weight.sum() + eps)


def ssd_loss(moving: torch.Tensor, target: torch.Tensor, weight: Optional[torch.Tensor] = None) -> torch.Tensor:
    """Mean squared difference, optionally restricted to the mineral mask."""
    return _masked_mean((moving - target).square(), weight)


def ncc_loss(
    moving: torch.Tensor,
    target: torch.Tensor,
    eps: float = 1e-8,
    weight: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Return `1 - NCC`, averaged over batch and RGB channels."""
    if moving.shape != target.shape:
        raise ValueError(f"Shape mismatch: {moving.shape} vs {target.shape}")
    b, c, _, _ = moving.shape
    m = moving.reshape(b, c, -1)
    t = target.reshape(b, c, -1)

    if weight is None:
        w = torch.ones((b, c, m.shape[-1]), device=m.device, dtype=m.dtype)
    else:
        w4 = _expand_weight(weight, moving)
        w = w4.reshape(b, c, -1)
    w = w / (w.sum(dim=-1, keepdim=True) + eps)

    m_mean = (w * m).sum(dim=-1, keepdim=True)
    t_mean = (w * t).sum(dim=-1, keepdim=True)
    mc = m - m_mean
    tc = t - t_mean
    covariance = (w * mc * tc).sum(dim=-1)
    variance_m = (w * mc.square()).sum(dim=-1)
    variance_t = (w * tc.square()).sum(dim=-1)
    correlation = covariance / (torch.sqrt(variance_m * variance_t) + eps)
    correlation = correlation.clamp(-1.0, 1.0)
    return 1.0 - correlation.mean()


def gradient_ncc_loss(
    moving: torch.Tensor,
    target: torch.Tensor,
    weight: Optional[torch.Tensor] = None,
    eps: float = 1e-8,
) -> torch.Tensor:
    """NCC on Sobel-like gradient magnitudes for structural alignment."""
    def gradient_magnitude(x: torch.Tensor) -> torch.Tensor:
        gx = F.pad(x[..., 1:] - x[..., :-1], (0, 1, 0, 0), mode="replicate")
        gy = F.pad(x[..., 1:, :] - x[..., :-1, :], (0, 0, 0, 1), mode="replicate")
        return torch.sqrt(gx.square() + gy.square() + eps)

    return ncc_loss(gradient_magnitude(moving), gradient_magnitude(target), weight=weight)


def mutual_information_loss(
    moving: torch.Tensor,
    target: torch.Tensor,
    num_bins: int = 32,
    sigma_ratio: float = 0.5,
    eps: float = 1e-7,
    weight: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Differentiable negative mutual information using soft histograms.

    RGB channels are averaged to one intensity image to limit memory use.
    """
    b = moving.shape[0]
    m = moving.mean(dim=1).reshape(b, -1)
    t = target.mean(dim=1).reshape(b, -1)
    m = (m - m.min(dim=1, keepdim=True).values) / (
        m.max(dim=1, keepdim=True).values - m.min(dim=1, keepdim=True).values + eps
    )
    t = (t - t.min(dim=1, keepdim=True).values) / (
        t.max(dim=1, keepdim=True).values - t.min(dim=1, keepdim=True).values + eps
    )

    if weight is None:
        w = torch.full_like(m, 1.0 / m.shape[1])
    else:
        w = weight[:, 0].reshape(b, -1).to(m)
        w = w / (w.sum(dim=1, keepdim=True) + eps)

    bins = torch.linspace(0.0, 1.0, num_bins, device=m.device, dtype=m.dtype)
    sigma = sigma_ratio / max(num_bins - 1, 1)
    m_soft = torch.exp(-0.5 * ((m.unsqueeze(-1) - bins) / (sigma + eps)).square())
    t_soft = torch.exp(-0.5 * ((t.unsqueeze(-1) - bins) / (sigma + eps)).square())
    m_soft = m_soft / (m_soft.sum(dim=-1, keepdim=True) + eps)
    t_soft = t_soft / (t_soft.sum(dim=-1, keepdim=True) + eps)

    joint = torch.bmm((m_soft * w.unsqueeze(-1)).transpose(1, 2), t_soft)
    joint = joint / (joint.sum(dim=(1, 2), keepdim=True) + eps)
    px = joint.sum(dim=2, keepdim=True)
    py = joint.sum(dim=1, keepdim=True)
    mi = (joint * (torch.log(joint + eps) - torch.log(px + eps) - torch.log(py + eps))).sum((1, 2))
    return -mi.mean()


def correlation_ratio_loss(
    moving: torch.Tensor,
    target: torch.Tensor,
    num_bins: int = 32,
    eps: float = 1e-8,
    weight: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Differentiable `1 - correlation ratio` for multimodal alignment."""
    b = moving.shape[0]
    y = moving.mean(dim=1).reshape(b, -1)
    x = target.mean(dim=1).reshape(b, -1)
    y = (y - y.min(dim=1, keepdim=True).values) / (
        y.max(dim=1, keepdim=True).values - y.min(dim=1, keepdim=True).values + eps
    )
    x = (x - x.min(dim=1, keepdim=True).values) / (
        x.max(dim=1, keepdim=True).values - x.min(dim=1, keepdim=True).values + eps
    )

    if weight is None:
        w = torch.full_like(y, 1.0 / y.shape[1])
    else:
        w = weight[:, 0].reshape(b, -1).to(y)
        w = w / (w.sum(dim=1, keepdim=True) + eps)

    bins = torch.linspace(0.0, 1.0, num_bins, device=x.device, dtype=x.dtype)
    sigma = 0.5 / max(num_bins - 1, 1)
    memberships = torch.exp(-0.5 * ((x.unsqueeze(-1) - bins) / (sigma + eps)).square())
    memberships = memberships / (memberships.sum(dim=-1, keepdim=True) + eps)
    weighted_memberships = memberships * w.unsqueeze(-1)
    bin_mass = weighted_memberships.sum(dim=1) + eps
    conditional_mean = torch.bmm(
        weighted_memberships.transpose(1, 2), y.unsqueeze(-1)
    ).squeeze(-1) / bin_mass
    global_mean = (w * y).sum(dim=1, keepdim=True)
    between = (bin_mass * (conditional_mean - global_mean).square()).sum(dim=1)
    total = (w * (y - global_mean).square()).sum(dim=1) + eps
    eta = (between / total).clamp(0.0, 1.0)
    return 1.0 - eta.mean()


def _unit_interval(x: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    flat = x.flatten(2)
    minimum = flat.min(dim=2, keepdim=True).values
    maximum = flat.max(dim=2, keepdim=True).values
    normalized = (flat - minimum) / (maximum - minimum + eps)
    return normalized.view_as(x)


def dice_loss(
    moving: torch.Tensor,
    target: torch.Tensor,
    weight: Optional[torch.Tensor] = None,
    eps: float = 1e-8,
) -> torch.Tensor:
    """Soft Dice loss in [0,1]; use as a validation diagnostic."""
    m = _unit_interval(moving, eps)
    t = _unit_interval(target, eps)
    w = torch.ones_like(m) if weight is None else _expand_weight(weight, m)
    intersection = (w * m * t).sum(dim=(2, 3))
    denominator = (w * m).sum(dim=(2, 3)) + (w * t).sum(dim=(2, 3))
    score = (2.0 * intersection + eps) / (denominator + eps)
    return 1.0 - score.mean()


def jaccard_distance_loss(
    moving: torch.Tensor,
    target: torch.Tensor,
    weight: Optional[torch.Tensor] = None,
    eps: float = 1e-8,
) -> torch.Tensor:
    """Soft Jaccard distance in [0,1]; use mainly as a diagnostic."""
    m = _unit_interval(moving, eps)
    t = _unit_interval(target, eps)
    w = torch.ones_like(m) if weight is None else _expand_weight(weight, m)
    intersection = (w * m * t).sum(dim=(2, 3))
    union = (w * (m + t - m * t)).sum(dim=(2, 3))
    score = (intersection + eps) / (union + eps)
    return 1.0 - score.mean()


def param_loss(
    pred: torch.Tensor,
    true: torch.Tensor,
    parameter_weights: Optional[Sequence[float] | torch.Tensor] = None,
) -> torch.Tensor:
    """Weighted MSE for `(tx, ty, theta, sx, sy)` registration labels."""
    error = (pred - true).square()
    if parameter_weights is not None:
        weights = torch.as_tensor(parameter_weights, dtype=pred.dtype, device=pred.device)
        if weights.numel() != 5:
            raise ValueError("parameter_weights must contain five values")
        error = error * weights.view(1, 5)
    return error.mean()


def regularisation_loss(params: torch.Tensor) -> torch.Tensor:
    """Penalize transforms far from identity."""
    identity = torch.tensor([0.0, 0.0, 0.0, 1.0, 1.0], device=params.device, dtype=params.dtype)
    return (params - identity).square().mean()


def bio_loss(
    warped: torch.Tensor,
    group: torch.Tensor,
    mineral_mask: torch.Tensor,
    boundary_mask: torch.Tensor,
    exterior_mask: torch.Tensor,
    eps: float = 1e-8,
) -> torch.Tensor:
    """Weak anatomical prior based on stain group.

    The penalty uses channel-averaged squared magnitude and should remain a
    small auxiliary term, not the main registration objective.
    """
    intensity = warped.square().mean(dim=1, keepdim=True)
    penalties = []
    for i, group_id in enumerate(group.reshape(-1).tolist()):
        if int(group_id) == 1:
            penalty_region = 1.0 - mineral_mask[i : i + 1]
        elif int(group_id) == 2:
            penalty_region = 1.0 - boundary_mask[i : i + 1]
        elif int(group_id) == 3:
            penalty_region = mineral_mask[i : i + 1]
        else:
            penalties.append(intensity.new_zeros(()))
            continue
        penalties.append(
            (intensity[i : i + 1] * penalty_region).sum()
            / (penalty_region.sum() + eps)
        )
    return torch.stack(penalties).mean()
