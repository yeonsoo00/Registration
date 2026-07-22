"""Losses for affine registration of pseudo-coloured histology images."""

from __future__ import annotations

import math

from typing import Mapping, Optional, Sequence

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


def _cost_volume_displacements(
    displacements: torch.Tensor,
    *,
    batch_size: int,
    candidates: int,
    device: torch.device,
) -> torch.Tensor:
    """Return a DP-safe BxKx2 displacement table in float32."""
    if not isinstance(displacements, torch.Tensor):
        raise TypeError("cost-volume displacements must be a tensor")
    if displacements.ndim == 2:
        if tuple(displacements.shape) != (candidates, 2):
            raise ValueError(
                "displacements must have shape Kx2 or BxKx2; got "
                f"{tuple(displacements.shape)} for K={candidates}"
            )
        displacements = displacements.unsqueeze(0).expand(batch_size, -1, -1)
    elif displacements.ndim == 3:
        if tuple(displacements.shape) != (batch_size, candidates, 2):
            raise ValueError(
                "batched displacements must have shape BxKx2; got "
                f"{tuple(displacements.shape)} for B={batch_size}, K={candidates}"
            )
    else:
        raise ValueError(
            "displacements must have shape Kx2 or BxKx2, got "
            f"{tuple(displacements.shape)}"
        )
    if not displacements.is_floating_point():
        raise TypeError("cost-volume displacements must be floating point")
    displacements = displacements.detach().to(device=device, dtype=torch.float32)
    if not bool(torch.isfinite(displacements).all()):
        raise ValueError("cost-volume displacements contain non-finite values")
    return displacements


def _cost_volume_mask(
    mask: torch.Tensor,
    *,
    shape: tuple[int, ...],
    name: str,
    device: torch.device,
) -> torch.Tensor:
    """Validate one auxiliary validity tensor and return a boolean mask."""
    if not isinstance(mask, torch.Tensor):
        raise TypeError(f"{name} must be a tensor")
    if tuple(mask.shape) != shape:
        raise ValueError(f"{name} must have shape {shape}, got {tuple(mask.shape)}")
    if mask.is_floating_point() and not bool(torch.isfinite(mask).all()):
        raise ValueError(f"{name} contains non-finite values")
    canonical = mask if mask.dtype == torch.bool else mask > 0.5
    return canonical.detach().to(device=device)


def _candidate_valid_from_level_masks(
    fixed_valid: torch.Tensor,
    moving_valid: torch.Tensor,
    displacements: torch.Tensor,
) -> torch.Tensor:
    """Rebuild the integer-shift validity used by local correlation.

    Model auxiliary output normally supplies candidate_valid directly. This
    fallback follows the same fixed(x) versus moving(x+d) convention when
    consuming an older or hand-built auxiliary record.
    """
    batch_size, _, height, width = fixed_valid.shape
    candidates = displacements.shape[1]
    identity = torch.eye(2, 3, device=fixed_valid.device, dtype=torch.float32)
    fixed_grid = F.affine_grid(
        identity.unsqueeze(0).expand(batch_size, -1, -1),
        size=(batch_size, 1, height, width),
        align_corners=True,
    )
    pixel_to_normalized = fixed_grid.new_tensor(
        (2.0 / max(width - 1, 1), 2.0 / max(height - 1, 1))
    )
    candidate_grid = fixed_grid[:, None] + displacements[:, :, None, None] * (
        pixel_to_normalized.view(1, 1, 1, 1, 2)
    )
    sampled = F.grid_sample(
        moving_valid[:, None]
        .expand(batch_size, candidates, 1, height, width)
        .reshape(batch_size * candidates, 1, height, width)
        .float(),
        candidate_grid.reshape(batch_size * candidates, height, width, 2),
        mode="nearest",
        padding_mode="zeros",
        align_corners=True,
    ).reshape(batch_size, candidates, height, width)
    in_bounds = candidate_grid.abs().amax(dim=-1) <= 1.0 + 1e-6
    return fixed_valid[:, 0, None] & in_bounds & (sampled > 0.5)


def cost_volume_correspondence_targets(
    level: Mapping[str, object],
    params_true: torch.Tensor,
    has_params: torch.Tensor,
    sigma: float,
) -> dict[str, torch.Tensor]:
    """Construct direct local-cost-volume targets for one FPN level.

    params_true is the existing moving-to-target registration parameter vector.
    Its affine-grid matrix maps fixed/output coordinates to moving/input
    coordinates, exactly matching C(x,d) = similarity(fixed(x), moving(x+d)).
    Real samples are never passed through affine conversion, so their NaN label
    sentinels remain unused behind has_params=False.
    """
    if (
        not isinstance(sigma, (int, float))
        or not math.isfinite(float(sigma))
        or not float(sigma) > 0.0
    ):
        raise ValueError("corr_target_sigma must be positive")
    sigma = float(sigma)
    required = {
        "probability",
        "expected_displacement",
        "confidence",
        "displacements",
        "fixed_valid",
        "moving_valid",
    }
    missing = sorted(required.difference(level))
    if missing:
        raise KeyError("Cost-volume auxiliary level is missing: " + ", ".join(missing))

    probability = level["probability"]
    expected = level["expected_displacement"]
    confidence = level["confidence"]
    if not all(
        isinstance(value, torch.Tensor) for value in (probability, expected, confidence)
    ):
        raise TypeError(
            "probability, expected_displacement, and confidence must be tensors"
        )
    assert isinstance(probability, torch.Tensor)
    assert isinstance(expected, torch.Tensor)
    assert isinstance(confidence, torch.Tensor)
    if probability.ndim != 4:
        raise ValueError("probability must have shape BxKxHxW")
    batch_size, candidates, height, width = probability.shape
    if tuple(expected.shape) != (batch_size, 2, height, width):
        raise ValueError(
            "expected_displacement must have shape "
            f"{(batch_size, 2, height, width)}, got {tuple(expected.shape)}"
        )
    if tuple(confidence.shape) != (batch_size, 1, height, width):
        raise ValueError(
            "confidence must have shape "
            f"{(batch_size, 1, height, width)}, got {tuple(confidence.shape)}"
        )
    if not probability.is_floating_point() or not expected.is_floating_point():
        raise TypeError("cost-volume predictions must be floating point")
    if not confidence.is_floating_point():
        raise TypeError("cost-volume confidence must be floating point")
    if not bool(torch.isfinite(probability).all()):
        raise ValueError("cost-volume probability contains non-finite values")
    if not bool(torch.isfinite(expected).all()):
        raise ValueError("expected_displacement contains non-finite values")
    if not bool(torch.isfinite(confidence).all()):
        raise ValueError("confidence contains non-finite values")

    device = probability.device
    if expected.device != device or confidence.device != device:
        raise ValueError("All cost-volume auxiliary predictions must share one device")
    if tuple(params_true.shape) != (batch_size, 5):
        raise ValueError(
            f"params_true must have shape {(batch_size, 5)}, got {tuple(params_true.shape)}"
        )
    synthetic = has_params.reshape(-1).bool()
    if synthetic.numel() != batch_size:
        raise ValueError(
            f"has_params must contain {batch_size} values, got {synthetic.numel()}"
        )
    synthetic = synthetic.detach().to(device=device)
    params_true = params_true.detach().to(device=device)
    if synthetic.any() and not bool(torch.isfinite(params_true[synthetic]).all()):
        raise ValueError("Every has_params=True sample must provide finite params_true")

    fixed_valid = _cost_volume_mask(
        level["fixed_valid"],
        shape=(batch_size, 1, height, width),
        name="fixed_valid",
        device=device,
    )
    moving_valid = _cost_volume_mask(
        level["moving_valid"],
        shape=(batch_size, 1, height, width),
        name="moving_valid",
        device=device,
    )
    displacements = _cost_volume_displacements(
        level["displacements"],
        batch_size=batch_size,
        candidates=candidates,
        device=device,
    )

    identity = torch.eye(2, 3, device=device, dtype=torch.float32)
    fixed_grid = F.affine_grid(
        identity.unsqueeze(0).expand(batch_size, -1, -1),
        size=(batch_size, 1, height, width),
        align_corners=True,
    )
    moving_grid = fixed_grid.clone()
    if synthetic.any():
        from utils import affine_parameters_to_matrix

        synthetic_indices = torch.nonzero(synthetic, as_tuple=False).reshape(-1)
        matrices = affine_parameters_to_matrix(params_true[synthetic_indices].float())
        moving_grid[synthetic_indices] = F.affine_grid(
            matrices,
            size=(synthetic_indices.numel(), 1, height, width),
            align_corners=True,
        )

    normalized_to_pixel = fixed_grid.new_tensor(
        (max(width - 1, 1) / 2.0, max(height - 1, 1) / 2.0)
    )
    true_displacement = (moving_grid - fixed_grid) * normalized_to_pixel
    true_displacement = true_displacement.permute(0, 3, 1, 2).contiguous()
    in_bounds = (moving_grid.abs().amax(dim=-1) <= 1.0 + 1e-6).unsqueeze(1)
    sampled_moving_valid = (
        F.grid_sample(
            moving_valid.float(),
            moving_grid,
            mode="nearest",
            padding_mode="zeros",
            align_corners=True,
        )
        > 0.5
    )
    geometric_mask = (
        synthetic[:, None, None, None] & fixed_valid & in_bounds & sampled_moving_valid
    )

    minimum = displacements.amin(dim=1)
    maximum = displacements.amax(dim=1)
    dx = true_displacement[:, 0:1]
    dy = true_displacement[:, 1:2]
    within_radius = (
        (dx >= minimum[:, 0, None, None, None] - 1e-6)
        & (dx <= maximum[:, 0, None, None, None] + 1e-6)
        & (dy >= minimum[:, 1, None, None, None] - 1e-6)
        & (dy <= maximum[:, 1, None, None, None] + 1e-6)
    )
    in_radius_mask = geometric_mask & within_radius

    supplied_candidate_valid = level.get("candidate_valid")
    if supplied_candidate_valid is None:
        candidate_valid = _candidate_valid_from_level_masks(
            fixed_valid, moving_valid, displacements
        )
    else:
        candidate_valid = _cost_volume_mask(
            supplied_candidate_valid,
            shape=(batch_size, candidates, height, width),
            name="candidate_valid",
            device=device,
        )
        candidate_valid = candidate_valid & fixed_valid

    delta = displacements[:, :, :, None, None] - true_displacement[:, None]
    target_logits = -delta.square().sum(dim=2) / (2.0 * sigma * sigma)
    negative_large = torch.finfo(target_logits.dtype).min
    masked_logits = target_logits.masked_fill(~candidate_valid, negative_large)
    maximum_logit = masked_logits.amax(dim=1, keepdim=True)
    target_weight = torch.exp(masked_logits - maximum_logit) * candidate_valid
    target_mass = target_weight.sum(dim=1, keepdim=True)
    distribution_mask = in_radius_mask & (target_mass > 0.0)
    target_probability = torch.where(
        target_mass > 0.0,
        target_weight / target_mass.clamp_min(torch.finfo(target_weight.dtype).tiny),
        torch.zeros_like(target_weight),
    )
    target_probability = target_probability * distribution_mask.to(
        target_probability.dtype
    )

    return {
        "true_displacement": true_displacement,
        "geometric_mask": geometric_mask,
        "in_radius_mask": in_radius_mask,
        "distribution_mask": distribution_mask,
        "confidence_mask": geometric_mask,
        "target_probability": target_probability,
        "confidence_target": in_radius_mask.to(dtype=torch.float32),
    }


def cost_volume_correspondence_loss(
    levels: Sequence[Mapping[str, object]],
    params_true: torch.Tensor,
    has_params: torch.Tensor,
    sigma: float,
) -> tuple[dict[str, torch.Tensor], dict[str, object]]:
    """Return stable direct-correspondence losses and exact validity counts.

    Each component is normalized by all of its valid pixels across the supplied
    pyramid levels. Empty components return a finite graph-connected zero. The
    per-level counts make validation aggregation and diagnostics explicit.
    """
    if not isinstance(levels, Sequence) or isinstance(levels, (str, bytes)):
        raise TypeError("levels must be a sequence of auxiliary level mappings")

    level_zeros = []
    for level_index, level in enumerate(levels):
        if not isinstance(level, Mapping):
            raise TypeError(f"levels[{level_index}] must be a mapping")
        tensors = []
        for name in ("probability", "expected_displacement", "confidence"):
            value = level.get(name)
            if not isinstance(value, torch.Tensor):
                raise TypeError(f"levels[{level_index}][{name!r}] must be a tensor")
            tensors.append(value)
        level_zeros.append(sum((value.sum() * 0.0 for value in tensors)))

    if level_zeros:
        differentiable_zero = torch.stack(level_zeros).sum()
    else:
        differentiable_zero = torch.zeros(
            (),
            device=params_true.device,
            dtype=torch.float32,
            requires_grad=True,
        )

    names = ("displacement", "distribution", "confidence")
    denominators = {name: 0 for name in names}
    active_levels = {name: 0 for name in names}
    per_level = {name: [0] * len(levels) for name in names}
    numerators: dict[str, torch.Tensor] = {}
    synthetic_samples = int(has_params.reshape(-1).bool().sum().item())

    if synthetic_samples:
        for level_index, level in enumerate(levels):
            targets = cost_volume_correspondence_targets(
                level,
                params_true=params_true,
                has_params=has_params,
                sigma=sigma,
            )
            expected = level["expected_displacement"]
            probability = level["probability"]
            confidence = level["confidence"]
            assert isinstance(expected, torch.Tensor)
            assert isinstance(probability, torch.Tensor)
            assert isinstance(confidence, torch.Tensor)

            displacement_mask = targets["in_radius_mask"]
            displacement_count = int(displacement_mask.sum().item())
            per_level["displacement"][level_index] = displacement_count
            if displacement_count:
                error = F.smooth_l1_loss(
                    expected.float(),
                    targets["true_displacement"],
                    beta=1.0,
                    reduction="none",
                ).mean(dim=1, keepdim=True)
                numerator = (error * displacement_mask).sum()
                numerators["displacement"] = (
                    numerators.get("displacement", numerator.new_zeros(())) + numerator
                )
                denominators["displacement"] += displacement_count
                active_levels["displacement"] += 1

            distribution_mask = targets["distribution_mask"]
            distribution_count = int(distribution_mask.sum().item())
            per_level["distribution"][level_index] = distribution_count
            if distribution_count:
                probability_float = probability.float()
                epsilon = torch.finfo(probability_float.dtype).eps
                cross_entropy = -(
                    targets["target_probability"]
                    * torch.log(probability_float.clamp_min(epsilon))
                ).sum(dim=1, keepdim=True)
                numerator = (cross_entropy * distribution_mask).sum()
                numerators["distribution"] = (
                    numerators.get("distribution", numerator.new_zeros(())) + numerator
                )
                denominators["distribution"] += distribution_count
                active_levels["distribution"] += 1

            confidence_mask = targets["confidence_mask"]
            confidence_count = int(confidence_mask.sum().item())
            per_level["confidence"][level_index] = confidence_count
            if confidence_count:
                confidence_float = confidence.float()
                epsilon = torch.finfo(confidence_float.dtype).eps
                # ``confidence`` is already a probability (the maximum local
                # correspondence probability), so there is no upstream sigmoid
                # logit to consume directly. Convert the clamped probability to
                # its exact logit and use the autocast-safe fused loss. This is
                # mathematically equivalent to BCE(probability, target),
                # including its gradient with respect to the probability.
                confidence_probability = confidence_float.clamp(epsilon, 1.0 - epsilon)
                confidence_logits = torch.logit(confidence_probability)
                confidence_error = F.binary_cross_entropy_with_logits(
                    confidence_logits,
                    targets["confidence_target"].float(),
                    reduction="none",
                )
                numerator = (confidence_error * confidence_mask).sum()
                numerators["confidence"] = (
                    numerators.get("confidence", numerator.new_zeros(())) + numerator
                )
                denominators["confidence"] += confidence_count
                active_levels["confidence"] += 1

    losses = {
        name: (
            numerators[name] / denominators[name]
            if denominators[name]
            else differentiable_zero
        )
        for name in names
    }
    counts: dict[str, object] = {
        **denominators,
        **{f"{name}_levels": count for name, count in active_levels.items()},
        **{f"{name}_per_level": values for name, values in per_level.items()},
        "synthetic_samples": synthetic_samples,
        "levels": len(levels),
    }
    return losses, counts
