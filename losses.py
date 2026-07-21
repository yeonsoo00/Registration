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


def _masked_mean(
    values: torch.Tensor, weight: Optional[torch.Tensor], eps: float = 1e-8
) -> torch.Tensor:
    if weight is None:
        return values.mean()
    weight = _expand_weight(weight, values)
    return (values * weight).sum() / (weight.sum() + eps)


def _masked_downsample_pair(
    moving: torch.Tensor,
    target: torch.Tensor,
    weight: torch.Tensor,
    scale: int,
    eps: float = 1e-8,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Downsample without averaging invalid FOV padding into valid pixels."""
    expanded_weight = _expand_weight(weight, moving)
    pooled_weight = F.avg_pool2d(expanded_weight, scale, scale)
    moving_sum = F.avg_pool2d(moving * expanded_weight, scale, scale)
    target_sum = F.avg_pool2d(target * expanded_weight, scale, scale)
    denominator = pooled_weight.clamp_min(eps)
    pooled_moving = torch.where(
        pooled_weight > eps, moving_sum / denominator, torch.zeros_like(moving_sum)
    )
    pooled_target = torch.where(
        pooled_weight > eps, target_sum / denominator, torch.zeros_like(target_sum)
    )
    return pooled_moving, pooled_target, pooled_weight


def ssd_loss(
    moving: torch.Tensor, target: torch.Tensor, weight: Optional[torch.Tensor] = None
) -> torch.Tensor:
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
    variance_product = (variance_m * variance_t).clamp_min(0.0)
    correlation = covariance / torch.sqrt(variance_product + eps * eps)
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

    gradient_weight = weight
    if weight is not None:
        expanded_weight = _expand_weight(weight, moving)
        gx_weight = F.pad(
            torch.minimum(expanded_weight[..., 1:], expanded_weight[..., :-1]),
            (0, 1, 0, 0),
            mode="replicate",
        )
        gy_weight = F.pad(
            torch.minimum(expanded_weight[..., 1:, :], expanded_weight[..., :-1, :]),
            (0, 0, 0, 1),
            mode="replicate",
        )
        gradient_weight = torch.minimum(gx_weight, gy_weight)

    return ncc_loss(
        gradient_magnitude(moving), gradient_magnitude(target), weight=gradient_weight
    )


def local_ncc_loss(moving, target, window_size=9, eps=1e-5, weight=None):
    """Local NCC over informative regions, optionally excluding invalid FOV pixels."""
    if moving.shape != target.shape:
        raise ValueError(f"Shape mismatch: {moving.shape} vs {target.shape}")
    if window_size < 3 or window_size % 2 == 0:
        raise ValueError("window_size must be an odd integer >= 3")
    pad = window_size // 2

    if weight is not None:
        weight = _expand_weight(weight, moving)
        local_weight = F.avg_pool2d(weight, window_size, 1, pad)
        denominator = local_weight.clamp_min(eps)
        mm = F.avg_pool2d(moving * weight, window_size, 1, pad) / denominator
        mt = F.avg_pool2d(target * weight, window_size, 1, pad) / denominator
        vm = (
            F.avg_pool2d(moving.square() * weight, window_size, 1, pad) / denominator
            - mm.square()
        ).clamp_min(0)
        vt = (
            F.avg_pool2d(target.square() * weight, window_size, 1, pad) / denominator
            - mt.square()
        ).clamp_min(0)
        cov = (
            F.avg_pool2d(moving * target * weight, window_size, 1, pad) / denominator
            - mm * mt
        )
        corr = cov / torch.sqrt(vm * vt + eps)
        informative = (vm + vt) > eps
        informative_weight = weight * informative * (local_weight > eps)
        if informative_weight.sum() <= eps:
            return corr.new_tensor(1.0)
        return 1.0 - _masked_mean(corr, informative_weight, eps)

    mm = F.avg_pool2d(moving, window_size, 1, pad)
    mt = F.avg_pool2d(target, window_size, 1, pad)
    vm = (F.avg_pool2d(moving.square(), window_size, 1, pad) - mm.square()).clamp_min(0)
    vt = (F.avg_pool2d(target.square(), window_size, 1, pad) - mt.square()).clamp_min(0)
    cov = F.avg_pool2d(moving * target, window_size, 1, pad) - mm * mt
    corr = cov / torch.sqrt(vm * vt + eps)
    informative = (vm + vt) > eps
    return 1.0 - corr[informative].mean() if informative.any() else corr.new_tensor(1.0)


def multiscale_local_ncc_loss(
    moving, target, scales=(1, 2, 4), window_size=9, weight=None
):
    """Coarse-to-fine local NCC with mask-normalized image pooling."""
    terms = []
    for scale in scales:
        if scale < 1:
            raise ValueError("scales must contain positive integers")
        if weight is None:
            m = moving if scale == 1 else F.avg_pool2d(moving, scale, scale)
            t = target if scale == 1 else F.avg_pool2d(target, scale, scale)
            pooled_weight = None
        elif scale == 1:
            m, t = moving, target
            pooled_weight = _expand_weight(weight, moving)
        else:
            m, t, pooled_weight = _masked_downsample_pair(moving, target, weight, scale)
        terms.append(local_ncc_loss(m, t, window_size, weight=pooled_weight))
    return torch.stack(terms).mean()


def charbonnier_loss(moving, target, eps=1e-3, weight=None):
    """Robust photometric loss, optionally excluding invalid FOV padding."""
    error = torch.sqrt((moving - target).square() + eps * eps)
    return _masked_mean(error, weight)


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
    mi = (
        joint * (torch.log(joint + eps) - torch.log(px + eps) - torch.log(py + eps))
    ).sum((1, 2))
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
    conditional_mean = (
        torch.bmm(weighted_memberships.transpose(1, 2), y.unsqueeze(-1)).squeeze(-1)
        / bin_mass
    )
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


def soft_foreground_dice_loss(
    moving: torch.Tensor,
    target: torch.Tensor,
    threshold: float = 0.35,
    temperature: float = 12.0,
    eps: float = 1e-6,
    weight: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Align sparse foreground support while optionally excluding invalid FOV."""

    spatial_weight = None
    if weight is not None:
        spatial_weight = weight.to(device=moving.device, dtype=moving.dtype)
        if spatial_weight.ndim != 4:
            raise ValueError("weight must have shape [B, C, H, W]")
        if (
            spatial_weight.shape[0] != moving.shape[0]
            or spatial_weight.shape[2:] != moving.shape[2:]
        ):
            raise ValueError(
                f"Weight shape {spatial_weight.shape} is incompatible with {moving.shape}"
            )
        if spatial_weight.shape[1] != 1:
            spatial_weight = spatial_weight.mean(dim=1, keepdim=True)

    def foreground_response(image: torch.Tensor) -> torch.Tensor:
        positive = F.relu(image).amax(dim=1, keepdim=True)
        if spatial_weight is not None:
            positive = positive * (spatial_weight > 0).to(positive.dtype)
        maximum = positive.amax(dim=(2, 3), keepdim=True)
        return positive / (maximum + eps)

    moving_unit = foreground_response(moving)
    target_unit = foreground_response(target)
    moving_mask = torch.sigmoid((moving_unit - threshold) * temperature)
    target_mask = torch.sigmoid((target_unit - threshold) * temperature)
    dice_weight = 1.0 if spatial_weight is None else spatial_weight
    intersection = (dice_weight * moving_mask * target_mask).sum(dim=(1, 2, 3))
    denominator = (dice_weight * moving_mask).sum(dim=(1, 2, 3)) + (
        dice_weight * target_mask
    ).sum(dim=(1, 2, 3))
    score = (2.0 * intersection + eps) / (denominator + eps)
    if spatial_weight is not None:
        support = spatial_weight.sum(dim=(1, 2, 3))
        score = torch.where(support > eps, score, torch.zeros_like(score))
    return 1.0 - score.mean()


def multiscale_gradient_loss(
    moving: torch.Tensor,
    target: torch.Tensor,
    scales: Sequence[int] = (1, 2, 4),
    eps: float = 1e-3,
    weight: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Robust edge loss whose masks require both gradient-neighbor pixels."""

    def gradients(
        image: torch.Tensor, valid: Optional[torch.Tensor]
    ) -> tuple[
        torch.Tensor,
        torch.Tensor,
        Optional[torch.Tensor],
        Optional[torch.Tensor],
    ]:
        gx = F.pad(image[..., 1:] - image[..., :-1], (0, 1, 0, 0))
        gy = F.pad(image[..., 1:, :] - image[..., :-1, :], (0, 0, 0, 1))
        if valid is None:
            return gx, gy, None, None
        gx_weight = F.pad(torch.minimum(valid[..., 1:], valid[..., :-1]), (0, 1, 0, 0))
        gy_weight = F.pad(
            torch.minimum(valid[..., 1:, :], valid[..., :-1, :]),
            (0, 0, 0, 1),
        )
        return gx, gy, gx_weight, gy_weight

    terms = []
    for scale in scales:
        if scale < 1:
            raise ValueError("scales must contain positive integers")
        if weight is None:
            m = moving if scale == 1 else F.avg_pool2d(moving, scale, scale)
            t = target if scale == 1 else F.avg_pool2d(target, scale, scale)
            pooled_weight = None
        elif scale == 1:
            m, t = moving, target
            pooled_weight = _expand_weight(weight, moving)
        else:
            m, t, pooled_weight = _masked_downsample_pair(moving, target, weight, scale)
        mgx, mgy, gx_weight, gy_weight = gradients(m, pooled_weight)
        tgx, tgy, _, _ = gradients(t, pooled_weight)
        terms.append(
            _masked_mean(torch.sqrt((mgx - tgx).square() + eps * eps), gx_weight)
        )
        terms.append(
            _masked_mean(torch.sqrt((mgy - tgy).square() + eps * eps), gy_weight)
        )
    return torch.stack(terms).mean()


def param_loss(
    pred: torch.Tensor,
    true: torch.Tensor,
    parameter_weights: Optional[Sequence[float] | torch.Tensor] = None,
) -> torch.Tensor:
    """Weighted MSE for `(tx, ty, theta, sx, sy)` registration labels."""
    error = (pred - true).square()
    if parameter_weights is not None:
        weights = torch.as_tensor(
            parameter_weights, dtype=pred.dtype, device=pred.device
        )
        if weights.numel() != 5:
            raise ValueError("parameter_weights must contain five values")
        error = error * weights.view(1, 5)
    return error.mean()


def affine_control_point_loss(
    pred: torch.Tensor, true: torch.Tensor, beta: float = 0.02
):
    """Compare transforms by displacement instead of incompatible raw units."""
    from utils import affine_parameters_to_matrix

    points = pred.new_tensor(
        [
            [-1.0, -1.0, 1.0],
            [1.0, -1.0, 1.0],
            [-1.0, 1.0, 1.0],
            [1.0, 1.0, 1.0],
            [0.0, 0.0, 1.0],
        ]
    )
    pp = torch.einsum("bij,pj->bpi", affine_parameters_to_matrix(pred), points)
    tp = torch.einsum("bij,pj->bpi", affine_parameters_to_matrix(true), points)
    return F.smooth_l1_loss(pp, tp, beta=beta)


def regularisation_loss(params: torch.Tensor) -> torch.Tensor:
    """Penalize transforms far from identity."""
    identity = torch.tensor(
        [0.0, 0.0, 0.0, 1.0, 1.0], device=params.device, dtype=params.dtype
    )
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
            (intensity[i : i + 1] * penalty_region).sum() / (penalty_region.sum() + eps)
        )
    return torch.stack(penalties).mean()
