"""Fast CPU contracts for deployable grouped affine registration.

Registered target stains may supervise the image loss, but every prediction in
this file is made from registered Mineral and an unregistered moving group.
"""

from __future__ import annotations

import argparse
import inspect
import math
import sys
import tempfile
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import numpy as np
from PIL import Image
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset

import inference as inference_module
import train as train_module
import utils as utils_module
from dataset import CartilageDataset, MAX_GROUP_STAINS
from debug_teacher import (
    _direction_status,
    _metric_group_stack,
    shared_signal_union_support,
    signal_union_mae,
)
from inference import ARCHITECTURE, load_student_checkpoint
from losses import (
    affine_control_point_loss,
    cost_volume_correspondence_loss,
    cost_volume_correspondence_targets,
)
from models import (
    AFFINE_HEAD_MODES,
    AffineHead,
    CorrelationVolumeAffineRegistrationModel,
    DEFAULT_CORRELATION_FEATURE_WIDTH,
    DEFAULT_GROUP_SLOTS,
    DEFAULT_RESIDUAL_BLOCKS_PER_STAGE,
    DEFAULT_RESIDUAL_ENCODER_CHANNELS,
    ENCODER_ARCHES,
    FeaturePyramidEncoder,
    FRONTEND_MODES,
    GROUP_INPUT_MODES,
    LocalCorrelationVolume,
    OnlineStructuralFrontend,
    ResidualEncoderBlock,
    ResidualEncoderStage,
    SeparatedAffineHead,
    SeparatedResidualAffineHead,
    STRUCTURAL_DESCRIPTOR_VERSION,
    TEACHER_FIXED_INPUT_VERSION,
    TeacherStudentAffineRegistrationModel,
    canonicalize_model_config,
    coarse_similarity_from_cost_stats,
    compose_similarity_and_residual_affine,
)
from train import (
    AFFINE_ERROR_METRIC_NAMES,
    affine_error_metrics,
    build_group_valid_overlap,
    compute_training_loss,
    configure_training_phase,
    evaluate_path,
    grouped_loss,
    load_initial_weights,
    parse_args as parse_train_args,
    safe_collate,
    save_checkpoint_pair,
    save_validation_overlays,
    supervision_source_and_matrices,
    synthetic_full_correction_matrices,
    teacher_warmup_active,
    teacher_distillation_weight,
    validate_args,
    warp_group,
    warp_group_for_supervision,
    warp_group_with_matrix,
)
from utils import affine_parameters_to_matrix, invert_affine_matrix


DATASET_KEYS = {
    "fixed_mineral",
    "moving_group",
    "target_group",
    "valid_group",
    "group_id",
    "stain_indices",
    "params_true",
    "has_params",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--height", type=int, default=40)
    parser.add_argument("--width", type=int, default=40)
    return parser.parse_args()


def build_config(
    frontend_mode: str = "structural",
    group_input_mode: str = "overlay",
    group_slots: int = MAX_GROUP_STAINS,
    affine_head_mode: str = "joint",
) -> dict:
    """Return the smallest valid v3 model used by these CPU contracts."""
    return {
        "input_channels": 3,
        "structural_channels": 6,
        "frontend_mode": frontend_mode,
        "group_input_mode": group_input_mode,
        "group_slots": group_slots,
        "affine_head_mode": affine_head_mode,
        "structural_descriptor_version": STRUCTURAL_DESCRIPTOR_VERSION,
        "structural_foreground_threshold": 0.2,
        "structural_distance_scale": 0.05,
        "structural_context_scale": 0.05,
        "structural_skeleton_radius": 2,
        "sfo_mode": "rgb",
        "latent_dim": 24,
        "group_embedding_dim": 8,
        "use_group_embedding": True,
        "num_groups": 5,
        "scale_range": (0.8, 1.2),
        "translation_limit": 0.5,
        "max_rotation_degrees": 20.0,
        "encoder_base_channels": 8,
        "encoder_depth": 3,
        "encoder_arch": "current",
        "encoder_channels": (8, 8, 16),
        "encoder_blocks_per_stage": None,
        "feature_width": 8,
        "correlation_feature_width": 8,
        "cost_hidden_channels": 8,
        "cost_volume_radii": (1,),
        "cost_pool_size": 1,
        "correlation_temperature": 0.07,
        "norm_type": "group",
        "force_group1_identity": True,
        "separate_group_heads": True,
        "separate_group_adapters": True,
    }


def _parse_train_cli(*extra: str) -> argparse.Namespace:
    argv = [
        "train.py",
        "--registered_root",
        "registered",
        "--unregistered_root",
        "unregistered",
        "--output_dir",
        "output",
        *extra,
    ]
    with mock.patch.object(sys, "argv", argv):
        return parse_train_args()


def assert_frontend_cli_contract() -> None:
    defaults = _parse_train_cli()
    assert defaults.frontend_mode == "structural"
    assert defaults.group_input_mode == "overlay"
    assert defaults.affine_head_mode == "joint"
    assert defaults.freeze_teacher is False
    assert FRONTEND_MODES == ("structural", "raw", "hybrid")
    assert GROUP_INPUT_MODES == ("stack", "overlay")
    assert AFFINE_HEAD_MODES == ("joint", "separated", "separated_residual")
    assert DEFAULT_GROUP_SLOTS == MAX_GROUP_STAINS
    custom = build_config("raw", "stack", group_slots=2)
    assert custom["frontend_mode"] == "raw"
    assert custom["group_input_mode"] == "stack"
    assert custom["group_slots"] == 2

    for frontend_mode in sorted(FRONTEND_MODES):
        parsed = _parse_train_cli("--frontend_mode", frontend_mode)
        assert parsed.frontend_mode == frontend_mode
    for group_input_mode in sorted(GROUP_INPUT_MODES):
        parsed = _parse_train_cli("--group_input_mode", group_input_mode)
        assert parsed.group_input_mode == group_input_mode
    for affine_head_mode in AFFINE_HEAD_MODES:
        parsed = _parse_train_cli("--affine_head_mode", affine_head_mode)
        assert parsed.affine_head_mode == affine_head_mode
    assert _parse_train_cli("--freeze_teacher").freeze_teacher is True
    assert _parse_train_cli("--no-freeze_teacher").freeze_teacher is False

    inference_argv = [
        "inference.py",
        "--checkpoint",
        "student.pt",
        "--unregistered_root",
        "unregistered",
        "--output_dir",
        "output",
    ]
    with mock.patch.object(sys, "argv", inference_argv):
        target_free = inference_module.parse_args()
    assert target_free.registered_root is None
    assert target_free.eval_with_registered_targets is None
    assert not inference_module._registered_evaluation_enabled(target_free)

    with mock.patch.object(
        sys, "argv", [*inference_argv, "--registered_root", "registered"]
    ):
        diagnostic = inference_module.parse_args()
    assert diagnostic.registered_root == "registered"
    assert inference_module._registered_evaluation_enabled(diagnostic)
    assert inference_module._registered_evaluation_enabled(
        SimpleNamespace(registered_root="registered", eval_with_registered_targets=True)
    )
    try:
        inference_module._registered_evaluation_enabled(
            SimpleNamespace(registered_root=None, eval_with_registered_targets=True)
        )
    except ValueError as error:
        assert "--registered_root" in str(error)
    else:
        raise AssertionError("Legacy evaluation flag accepted no registered root")

    for flag, invalid in (
        ("--frontend_mode", "pixels"),
        ("--group_input_mode", "single"),
        ("--affine_head_mode", "split"),
    ):
        with mock.patch("sys.stderr"):
            try:
                _parse_train_cli(flag, invalid)
            except SystemExit as error:
                assert error.code == 2
            else:
                raise AssertionError(f"{flag} accepted invalid value {invalid!r}")

    for key, invalid in (
        ("frontend_mode", "pixels"),
        ("group_input_mode", "single"),
        ("affine_head_mode", "split"),
    ):
        config = build_config()
        config[key] = invalid
        try:
            CorrelationVolumeAffineRegistrationModel(**config)
        except ValueError as error:
            assert key in str(error)
        else:
            raise AssertionError(f"Model accepted invalid {key}={invalid!r}")


def _set_head_output(head, value: float) -> None:
    with torch.no_grad():
        if isinstance(head, AffineHead):
            head.output.weight.zero_()
            head.output.bias.fill_(value)
            return
        for component in (
            head.translation_head,
            head.rotation_head,
            head.scale_head,
        ):
            component.output.weight.zero_()
            component.output.bias.fill_(value)


def _assert_affine_bounds(params: torch.Tensor) -> None:
    tolerance = 1e-6
    assert bool((params[:, :2].abs() <= 0.5 + tolerance).all())
    assert bool((params[:, 2].abs() <= math.radians(20.0) + tolerance).all())
    assert bool(
        ((params[:, 3:] >= 0.8 - tolerance) & (params[:, 3:] <= 1.2 + tolerance)).all()
    )


def assert_affine_head_geometry_contract(device: torch.device) -> None:
    feature = torch.randn((3, 12), device=device)
    identity = feature.new_tensor((0.0, 0.0, 0.0, 1.0, 1.0)).expand(3, -1)
    for head_class in (AffineHead, SeparatedAffineHead):
        head = head_class(12, (0.8, 1.2), 0.5, 20.0).to(device)
        torch.testing.assert_close(head(feature), identity, atol=1e-7, rtol=1e-7)
        if isinstance(head, SeparatedAffineHead):
            component_parameter_ids = [
                {id(parameter) for parameter in component.parameters()}
                for component in (
                    head.translation_head,
                    head.rotation_head,
                    head.scale_head,
                )
            ]
            assert all(
                left.isdisjoint(right)
                for index, left in enumerate(component_parameter_ids)
                for right in component_parameter_ids[index + 1 :]
            )
        for saturation in (-100.0, 100.0):
            _set_head_output(head, saturation)
            params = head(feature)
            _assert_affine_bounds(params)
            expected = params.new_tensor(
                (
                    -0.5 if saturation < 0 else 0.5,
                    -0.5 if saturation < 0 else 0.5,
                    math.radians(-20.0 if saturation < 0 else 20.0),
                    0.8 if saturation < 0 else 1.2,
                    0.8 if saturation < 0 else 1.2,
                )
            ).expand_as(params)
            torch.testing.assert_close(params, expected, atol=1e-6, rtol=1e-6)

    residual_head = SeparatedResidualAffineHead(12, (0.8, 1.2), 0.5, 20.0).to(device)
    coarse = feature.new_tensor(((0.10, -0.08, math.radians(5.0), 1.05, 1.05),)).expand(
        3, -1
    )
    params, matrix = residual_head.forward_with_matrix(feature, coarse)
    torch.testing.assert_close(params, coarse, atol=1e-7, rtol=1e-7)
    torch.testing.assert_close(
        matrix, affine_parameters_to_matrix(params), atol=2e-6, rtol=2e-6
    )
    ((params - identity) ** 2).sum().backward()
    for component in (
        residual_head.translation_head,
        residual_head.rotation_head,
        residual_head.scale_head,
    ):
        gradient = component.output.bias.grad
        assert gradient is not None
        assert bool(torch.isfinite(gradient).all())
        assert bool((gradient.abs() > 0.0).all())
    residual_head.zero_grad(set_to_none=True)
    for saturation in (-100.0, 100.0):
        _set_head_output(residual_head, saturation)
        params, matrix = residual_head.forward_with_matrix(feature, coarse)
        _assert_affine_bounds(params)
        torch.testing.assert_close(
            matrix, affine_parameters_to_matrix(params), atol=2e-6, rtol=2e-6
        )

    expected_head_class = {
        "joint": AffineHead,
        "separated": SeparatedAffineHead,
        "separated_residual": SeparatedResidualAffineHead,
    }
    group_ids = torch.tensor((5, 2, 4, 3), dtype=torch.long, device=device)
    latent = torch.zeros((4, 24), device=device)
    coarse_identity = latent.new_tensor((0.0, 0.0, 0.0, 1.0, 1.0)).expand(4, -1)
    for mode in AFFINE_HEAD_MODES:
        config = build_config(affine_head_mode=mode)
        config.update(
            force_group1_identity=False,
            use_group_embedding=False,
            separate_group_adapters=False,
        )
        model = CorrelationVolumeAffineRegistrationModel(**config).to(device)
        assert model.heads is not None and len(model.heads) == 5
        assert all(isinstance(head, expected_head_class[mode]) for head in model.heads)
        for index, head in enumerate(model.heads, start=1):
            _set_head_output(head, 0.05 * index)
        auxiliary = coarse_identity if mode == "separated_residual" else None
        routed = model._route_by_group(
            latent, group_ids, model.heads, auxiliary=auxiliary
        )
        expected_rows = []
        for row, group_id in enumerate(group_ids.tolist()):
            head = model.heads[group_id - 1]
            if auxiliary is None:
                expected_rows.append(head(latent[row : row + 1])[0])
            else:
                expected_rows.append(
                    head(latent[row : row + 1], auxiliary[row : row + 1])[0]
                )
        torch.testing.assert_close(routed, torch.stack(expected_rows))
        assert len({float(value) for value in routed[:, 0]}) == len(group_ids)

        shared_config = dict(config)
        shared_config["separate_group_heads"] = False
        shared_model = CorrelationVolumeAffineRegistrationModel(**shared_config).to(
            device
        )
        assert shared_model.heads is None
        assert isinstance(shared_model.head, expected_head_class[mode])


def assert_coarse_affine_and_composition_contract(device: torch.device) -> None:
    stats = torch.zeros((2, 2, 8), device=device)
    desired_scale = 1.08
    desired_theta = math.radians(7.0)
    coefficient_a = desired_scale * math.cos(desired_theta)
    coefficient_b = desired_scale * math.sin(desired_theta)
    stats[1, :, 0] = 0.12
    stats[1, :, 1] = -0.09
    stats[1, :, 2] = (coefficient_a - 1.0) / 3.0
    stats[1, :, 5] = (coefficient_a - 1.0) / 3.0
    stats[1, :, 3] = -coefficient_b / 3.0
    stats[1, :, 4] = coefficient_b / 3.0
    stats[1, :, 6] = stats.new_tensor((0.25, 0.75))
    stats[1, :, 7] = 1.0
    coarse = coarse_similarity_from_cost_stats(stats, (0.8, 1.2), 0.5, 20.0)
    identity = coarse.new_tensor((0.0, 0.0, 0.0, 1.0, 1.0))
    torch.testing.assert_close(coarse[0], identity)
    expected = coarse.new_tensor(
        (0.12, -0.09, desired_theta, desired_scale, desired_scale)
    )
    torch.testing.assert_close(coarse[1], expected, atol=2e-6, rtol=2e-6)

    extreme = torch.zeros((1, 1, 8), device=device)
    extreme[..., :6] = 100.0
    extreme[..., 6:] = 1.0
    _assert_affine_bounds(
        coarse_similarity_from_cost_stats(extreme, (0.8, 1.2), 0.5, 20.0)
    )

    coarse_params = coarse.new_tensor(((0.10, -0.08, math.radians(5.0), 1.05, 1.05),))
    residual_params = coarse.new_tensor(((0.03, 0.02, math.radians(-2.0), 0.95, 1.10),))
    final_params, final_matrix = compose_similarity_and_residual_affine(
        coarse_params, residual_params
    )
    bottom = coarse.new_tensor((0.0, 0.0, 1.0)).view(1, 1, 3)
    coarse_h = torch.cat((affine_parameters_to_matrix(coarse_params), bottom), dim=1)
    residual_h = torch.cat(
        (affine_parameters_to_matrix(residual_params), bottom), dim=1
    )
    expected_matrix = (coarse_h @ residual_h)[:, :2]
    torch.testing.assert_close(final_matrix, expected_matrix, atol=2e-6, rtol=2e-6)
    torch.testing.assert_close(
        affine_parameters_to_matrix(final_params),
        final_matrix,
        atol=2e-6,
        rtol=2e-6,
    )
    invalid_coarse = coarse_params.clone()
    invalid_coarse[:, 4] = 0.97
    try:
        compose_similarity_and_residual_affine(invalid_coarse, residual_params)
    except ValueError as error:
        assert "isotropic" in str(error)
    else:
        raise AssertionError("Residual composition accepted anisotropic coarse scale")


def _frontend_inputs(
    device: torch.device, height: int, width: int
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    fixed = torch.zeros((1, 3, height, width), device=device)
    fixed[:, :, 4:-5, 5:-6] = 0.65
    moving = torch.zeros((1, MAX_GROUP_STAINS, 3, height, width), device=device)
    target = torch.zeros_like(moving)
    moving[:, 0, :, 7:-5, 4:-8] = 0.75
    moving[:, 1, :, 5:-8, 9:-5] = 0.45
    target[:, 0, :, 5:-7, 7:-5] = 0.85
    target[:, 1, :, 8:-5, 5:-9] = 0.55
    group = torch.tensor((3,), dtype=torch.long, device=device)
    return fixed, moving, target, group


def assert_structural_frontend_compatibility(
    device: torch.device, height: int, width: int
) -> None:
    legacy_config = build_config()
    for key in (
        "frontend_mode",
        "group_input_mode",
        "group_slots",
        "affine_head_mode",
    ):
        legacy_config.pop(key)
    explicit_config = build_config(
        frontend_mode="structural",
        group_input_mode="overlay",
        group_slots=MAX_GROUP_STAINS,
    )

    torch.manual_seed(101)
    legacy = CorrelationVolumeAffineRegistrationModel(**legacy_config).to(device)
    torch.manual_seed(101)
    explicit = CorrelationVolumeAffineRegistrationModel(**explicit_config).to(device)
    legacy_state = legacy.state_dict()
    explicit_state = explicit.state_dict()
    assert legacy_state.keys() == explicit_state.keys()
    assert not any("frontend_adapter" in key for key in legacy_state)
    assert all(
        torch.equal(legacy_state[key], explicit_state[key]) for key in legacy_state
    )
    explicit.load_state_dict(legacy_state, strict=True)
    torch.manual_seed(102)
    _randomize_affine_outputs(legacy, 0.15)
    explicit.load_state_dict(legacy.state_dict(), strict=True)

    fixed, moving, _, group = _frontend_inputs(device, height, width)
    legacy.eval()
    explicit.eval()
    with torch.no_grad():
        legacy_params = legacy(fixed, moving, group)
        explicit_params = explicit(fixed, moving, group)
    assert torch.equal(legacy_params, explicit_params)


def assert_frontend_forward_matrix(
    device: torch.device, height: int, width: int
) -> None:
    cases = (
        ("structural", "overlay"),
        ("raw", "overlay"),
        ("hybrid", "overlay"),
        ("raw", "stack"),
        ("hybrid", "stack"),
    )
    fixed, moving, target, group = _frontend_inputs(device, height, width)
    for frontend_mode, group_input_mode in cases:
        wrapper = TeacherStudentAffineRegistrationModel(
            student_config=build_config(frontend_mode, group_input_mode),
            use_teacher_branch=True,
        ).to(device)
        assert wrapper.teacher is not None
        assert tuple(inspect.signature(wrapper.forward).parameters) == (
            "fixed_mineral",
            "moving_group",
            "group",
            "return_aux",
        )
        assert tuple(inspect.signature(wrapper.student.forward).parameters) == (
            "fixed_mineral",
            "moving_group",
            "group",
            "return_aux",
            "_batch_aux_metadata",
        )
        assert tuple(inspect.signature(wrapper.teacher.forward).parameters) == (
            "fixed_mineral",
            "target_group",
            "moving_group",
            "group",
            "return_aux",
            "_batch_aux_metadata",
        )
        assert tuple(inspect.signature(wrapper.forward_teacher).parameters) == (
            "fixed_mineral",
            "target_group",
            "moving_group",
            "group",
            "return_aux",
        )
        wrapper.eval()
        with torch.no_grad():
            student_params = wrapper(fixed, moving, group)
            teacher_params = wrapper.forward_teacher(fixed, target, moving, group)
        assert student_params.shape == teacher_params.shape == (1, 5)
        assert torch.isfinite(student_params).all()
        assert torch.isfinite(teacher_params).all()
        try:
            wrapper(
                fixed_mineral=fixed,
                moving_group=moving,
                group=group,
                target_group=target,
            )
        except TypeError:
            pass
        else:
            raise AssertionError(
                f"{frontend_mode}/{group_input_mode} student accepted target_group"
            )


def assert_teacher_mineral_fusion_contract(
    device: torch.device, height: int, width: int
) -> None:
    """Prove that both registered fixed sources feed every teacher frontend."""
    toy_target = torch.full((1, 2, 2, 2), 2.0, device=device)
    toy_mineral = torch.full_like(toy_target, 4.0)
    toy_target_valid = torch.tensor(
        (((True, True), (False, False)),), device=device
    ).unsqueeze(1)
    toy_mineral_valid = torch.tensor(
        (((True, False), (True, False)),), device=device
    ).unsqueeze(1)
    fusion_model = CorrelationVolumeAffineRegistrationModel(**build_config()).to(device)
    toy_fused, toy_valid = fusion_model.fuse_teacher_fixed_representations(
        toy_target, toy_mineral, toy_target_valid, toy_mineral_valid
    )
    expected_fused = toy_target.new_tensor(
        [[[[3.0, 2.0], [4.0, 0.0]], [[3.0, 2.0], [4.0, 0.0]]]]
    )
    expected_valid = toy_target_valid | toy_mineral_valid
    torch.testing.assert_close(toy_fused, expected_fused)
    assert torch.equal(toy_valid, expected_valid)

    fixed, moving, target, group = _frontend_inputs(device, height, width)
    alternate_fixed = torch.zeros_like(fixed)
    alternate_fixed[:, 0, 2 : height // 2, width // 2 : width - 2] = 0.95
    alternate_fixed[:, 1, height // 2 : height - 2, 2 : width // 3] = 0.35
    alternate_target = torch.roll(target, shifts=(4, -5), dims=(-2, -1))

    for frontend_mode, group_input_mode in (
        ("structural", "overlay"),
        ("raw", "overlay"),
        ("hybrid", "stack"),
    ):
        wrapper = TeacherStudentAffineRegistrationModel(
            student_config=build_config(frontend_mode, group_input_mode),
            use_teacher_branch=True,
        ).to(device)
        assert wrapper.teacher is not None
        teacher = wrapper.teacher
        _randomize_affine_outputs(teacher, -0.17)

        target_representation, target_valid = teacher.group_frontend_representation(
            target, group
        )
        mineral_representation, mineral_valid = teacher.mineral_frontend_representation(
            fixed
        )
        fused, fused_valid = teacher.teacher_fixed_frontend_representation(
            fixed, target, group
        )
        target_weight = target_valid.to(target_representation.dtype)
        mineral_weight = mineral_valid.to(target_representation.dtype)
        expected = (
            target_representation * target_weight
            + mineral_representation * mineral_weight
        ) / (target_weight + mineral_weight).clamp_min(1.0)
        expected_valid = target_valid | mineral_valid
        expected = expected * expected_valid.to(expected.dtype)
        torch.testing.assert_close(fused, expected)
        assert torch.equal(fused_valid, expected_valid)

        altered_mineral_fused, _ = teacher.teacher_fixed_frontend_representation(
            alternate_fixed, target, group
        )
        altered_target_fused, _ = teacher.teacher_fixed_frontend_representation(
            fixed, alternate_target, group
        )
        assert not torch.equal(fused, altered_mineral_fused)
        assert not torch.equal(fused, altered_target_fused)

        teacher.eval()
        with torch.no_grad():
            base_params = wrapper.forward_teacher(fixed, target, moving, group)
            mineral_changed_params = wrapper.forward_teacher(
                alternate_fixed, target, moving, group
            )
            target_changed_params = wrapper.forward_teacher(
                fixed, alternate_target, moving, group
            )
        assert torch.isfinite(base_params).all()
        assert not torch.equal(base_params, mineral_changed_params)
        assert not torch.equal(base_params, target_changed_params)

    try:
        wrapper.forward_teacher(fixed[..., :-1], target, moving, group)
    except ValueError as error:
        assert "fixed_mineral shape" in str(error)
    else:
        raise AssertionError("Teacher accepted a mismatched Mineral shape")


def _zero_mineral_descriptor(
    fixed_mineral: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    batch, _, height, width = fixed_mineral.shape
    descriptor = fixed_mineral.new_zeros((batch, 6, height, width))
    validity = torch.zeros(
        (batch, 1, height, width), dtype=torch.bool, device=fixed_mineral.device
    )
    return descriptor, validity


def _zero_group_descriptor(
    group_stack: torch.Tensor, group_ids: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    del group_ids
    batch, _, _, height, width = group_stack.shape
    descriptor = group_stack.new_zeros((batch, 6, height, width))
    validity = torch.zeros(
        (batch, 1, height, width), dtype=torch.bool, device=group_stack.device
    )
    return descriptor, validity


def _set_frontend_affine_offsets(model) -> None:
    modules = [
        model.mineral_frontend_adapter,
        model.group_frontend_adapter,
        *(list(model.group_adapters) if model.group_adapters is not None else []),
    ]
    with torch.no_grad():
        for module in modules:
            if module is None:
                continue
            for name, parameter in module.named_parameters():
                if name.endswith("bias"):
                    parameter.fill_(0.75)


def assert_raw_and_hybrid_frontend_contract(
    device: torch.device, height: int, width: int
) -> None:
    small_height = small_width = 12
    group_stack = torch.zeros(
        (1, MAX_GROUP_STAINS, 3, small_height, small_width), device=device
    )
    group_stack[:, 0, :, 2:5, 3:7] = 0.25
    group_stack[:, 1, :, 6:9, 7:10] = 0.8
    expected_valid = torch.zeros(
        (1, 1, small_height, small_width), dtype=torch.bool, device=device
    )
    expected_valid[:, :, 2:5, 3:7] = True
    expected_valid[:, :, 6:9, 7:10] = True

    overlay_model = CorrelationVolumeAffineRegistrationModel(
        **build_config("raw", "overlay")
    ).to(device)
    overlay_raw, overlay_valid = overlay_model._raw_group_representation(group_stack)
    assert torch.equal(overlay_raw, group_stack.amax(dim=1))
    assert torch.equal(overlay_valid, expected_valid)
    assert bool(overlay_valid.any()) and not bool(overlay_valid.all())
    assert torch.count_nonzero(overlay_raw[:, :, ~expected_valid[0, 0]]) == 0

    stack_model = CorrelationVolumeAffineRegistrationModel(
        **build_config("raw", "stack")
    ).to(device)
    stack_raw, stack_valid = stack_model._raw_group_representation(group_stack)
    expected_stack = group_stack.reshape(
        1, MAX_GROUP_STAINS * 3, small_height, small_width
    )
    assert torch.equal(stack_raw, expected_stack)
    assert torch.equal(stack_valid, expected_valid)
    assert torch.count_nonzero(stack_raw[:, 6:]) == 0

    fixed, moving, target, group = _frontend_inputs(device, height, width)
    raw_wrapper = TeacherStudentAffineRegistrationModel(
        student_config=build_config("raw", "overlay"),
        use_teacher_branch=True,
    ).to(device)
    assert raw_wrapper.teacher is not None
    with (
        mock.patch.object(
            raw_wrapper.student.structural_frontend,
            "mineral_descriptor",
            side_effect=AssertionError("raw student built a structural descriptor"),
        ),
        mock.patch.object(
            raw_wrapper.student.structural_frontend,
            "group_descriptor",
            side_effect=AssertionError("raw student built a structural descriptor"),
        ),
        mock.patch.object(
            raw_wrapper.teacher.structural_frontend,
            "mineral_descriptor",
            side_effect=AssertionError("raw teacher built a structural descriptor"),
        ),
        mock.patch.object(
            raw_wrapper.teacher.structural_frontend,
            "group_descriptor",
            side_effect=AssertionError("raw teacher built a structural descriptor"),
        ),
        torch.no_grad(),
    ):
        assert torch.isfinite(raw_wrapper(fixed, moving, group)).all()
        assert torch.isfinite(
            raw_wrapper.forward_teacher(fixed, target, moving, group)
        ).all()

    hsv_stack = torch.zeros_like(moving)
    hsv_stack[:, 0, 0, 6:-7, 5:-8] = 0.48
    hsv_stack[:, 0, 1, 6:-7, 5:-8] = 0.82
    hsv_stack[:, 0, 2, 6:-7, 5:-8] = 0.73
    hsv_model = CorrelationVolumeAffineRegistrationModel(
        **{**build_config("raw", "overlay"), "sfo_mode": "hsv"}
    ).to(device)
    captured_inputs = []

    def capture_raw_input(module, inputs):
        del module
        captured_inputs.append(inputs[0].detach().clone())

    assert hsv_model.group_frontend_adapter is not None
    handle = hsv_model.group_frontend_adapter.register_forward_pre_hook(
        capture_raw_input
    )
    hsv_model.group_frontend_representation(
        hsv_stack, torch.tensor((5,), dtype=torch.long, device=device)
    )
    handle.remove()
    assert len(captured_inputs) == 1
    assert torch.equal(captured_inputs[0], hsv_stack.amax(dim=1))

    hybrid = CorrelationVolumeAffineRegistrationModel(
        **build_config("hybrid", "overlay")
    ).to(device)
    _randomize_affine_outputs(hybrid, 0.25)
    with (
        mock.patch.object(
            hybrid.structural_frontend,
            "mineral_descriptor",
            side_effect=_zero_mineral_descriptor,
        ),
        mock.patch.object(
            hybrid.structural_frontend,
            "group_descriptor",
            side_effect=_zero_group_descriptor,
        ),
    ):
        _, fixed_valid = hybrid.mineral_frontend_representation(fixed)
        _, moving_valid = hybrid.group_frontend_representation(moving, group)
        assert bool(fixed_valid.any()) and not bool(fixed_valid.all())
        assert bool(moving_valid.any()) and not bool(moving_valid.all())
        hybrid.eval()
        with torch.no_grad():
            hybrid_params = hybrid(fixed, moving, group)
    identity = hybrid_params.new_tensor(((0.0, 0.0, 0.0, 1.0, 1.0),))
    assert not torch.equal(hybrid_params, identity)

    for frontend_mode, group_input_mode in (
        ("raw", "overlay"),
        ("hybrid", "stack"),
    ):
        wrapper = TeacherStudentAffineRegistrationModel(
            student_config=build_config(frontend_mode, group_input_mode),
            use_teacher_branch=True,
        ).to(device)
        assert wrapper.teacher is not None
        _set_frontend_affine_offsets(wrapper.student)
        _set_frontend_affine_offsets(wrapper.teacher)
        _randomize_affine_outputs(wrapper.student, 0.25)
        _randomize_affine_outputs(wrapper.teacher, -0.25)
        zero_fixed = torch.zeros_like(fixed)
        zero_group = torch.zeros_like(moving)
        wrapper.eval()
        with torch.no_grad():
            zero_student = wrapper(zero_fixed, zero_group, group)
            zero_teacher = wrapper.forward_teacher(
                zero_fixed, zero_group, zero_group, group
            )
        assert torch.equal(zero_student, identity)
        assert torch.equal(zero_teacher, identity)


def _write_signal(path: Path, *, shift_x: int, color: tuple[int, int, int]) -> None:
    height = width = 48
    yy, xx = np.mgrid[:height, :width]
    center_x = width // 2 + shift_x
    radius_squared = (xx - center_x) ** 2 + (yy - height // 2) ** 2
    disk = radius_squared <= 9**2
    ring = (radius_squared >= 5**2) & disk
    image = np.zeros((height, width, 3), dtype=np.uint8)
    image[disk] = np.asarray(color, dtype=np.uint8)
    image[ring] = np.maximum(image[ring], np.asarray((80, 180, 220), np.uint8))
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(image, mode="RGB").save(path)


def _dataset(
    registered_root: Path | None,
    unregistered_root: Path,
    *,
    height: int,
    width: int,
    synthetic_prob: float,
    require_registered_targets: bool,
    fixed_mineral_root: Path | None = None,
) -> CartilageDataset:
    return CartilageDataset(
        None if registered_root is None else str(registered_root),
        str(unregistered_root),
        size=(height, width),
        image_mode="rgb",
        sfo_mode="rgb",
        crop_mode="full",
        synthetic_prob=synthetic_prob,
        tx_range=(2.0, 2.0),
        ty_range=(0.0, 0.0),
        rot_range=(0.0, 0.0),
        scale_range=(1.0, 1.0),
        deterministic_synthetic=True,
        synthetic_seed=17,
        include_group1=False,
        require_registered_targets=require_registered_targets,
        fixed_mineral_root=(
            None if fixed_mineral_root is None else str(fixed_mineral_root)
        ),
    )


def _assert_sample(sample: dict[str, torch.Tensor], height: int, width: int) -> None:
    assert set(sample) == DATASET_KEYS
    assert sample["fixed_mineral"].shape == (3, height, width)
    group_shape = (MAX_GROUP_STAINS, 3, height, width)
    assert sample["moving_group"].shape == group_shape
    assert sample["target_group"].shape == group_shape
    assert sample["valid_group"].shape == (MAX_GROUP_STAINS,)
    assert sample["stain_indices"].shape == (MAX_GROUP_STAINS,)
    assert sample["params_true"].shape == (5,)
    assert sample["group_id"].shape == ()
    assert sample["has_params"].shape == ()
    assert sample["fixed_mineral"].dtype == torch.float32
    assert sample["moving_group"].dtype == torch.float32
    assert sample["target_group"].dtype == torch.float32
    assert sample["valid_group"].dtype == torch.bool
    for key in ("fixed_mineral", "moving_group", "target_group"):
        tensor = sample[key]
        assert torch.isfinite(tensor).all(), key
        assert float(tensor.min()) >= 0.0, key
        assert float(tensor.max()) <= 1.0, key


def assert_dataset_contract(model, device, height: int, width: int) -> None:
    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory)
        mineral_only = root / "registered_mineral_only"
        registered = root / "registered_targets"
        unregistered = root / "unregistered"
        no_unregistered = root / "no_unregistered"
        no_unregistered.mkdir(parents=True)

        _write_signal(
            mineral_only / "case" / "case_1.png",
            shift_x=0,
            color=(210, 210, 210),
        )
        _write_signal(
            registered / "case" / "case_1.png",
            shift_x=0,
            color=(210, 210, 210),
        )
        _write_signal(
            registered / "case" / "case_4.png",
            shift_x=0,
            color=(230, 40, 30),
        )
        _write_signal(
            unregistered / "case" / "case_1.png",
            shift_x=2,
            color=(40, 170, 210),
        )
        _write_signal(
            unregistered / "case" / "case_4.png",
            shift_x=4,
            color=(230, 40, 30),
        )

        # The training-compatible default sources fixed Mineral from registered_root.
        deployment_dataset = _dataset(
            mineral_only,
            unregistered,
            height=height,
            width=width,
            synthetic_prob=0.0,
            require_registered_targets=False,
        )
        assert len(deployment_dataset) == 1
        deployment = deployment_dataset[0]
        _assert_sample(deployment, height, width)
        assert int(deployment["group_id"]) == 2
        assert deployment["valid_group"].tolist() == [True, False, False]
        assert deployment["stain_indices"].tolist() == [4, 0, 0]
        assert torch.count_nonzero(deployment["target_group"]) == 0
        assert not bool(deployment["has_params"])
        model.eval()
        with torch.no_grad():
            params = model(
                fixed_mineral=deployment["fixed_mineral"].unsqueeze(0).to(device),
                moving_group=deployment["moving_group"].unsqueeze(0).to(device),
                group=deployment["group_id"].reshape(1).to(device),
            )
        assert params.shape == (1, 5) and torch.isfinite(params).all()

        root_free_dataset = _dataset(
            None,
            unregistered,
            height=height,
            width=width,
            synthetic_prob=0.0,
            require_registered_targets=False,
            fixed_mineral_root=unregistered,
        )
        root_free = root_free_dataset[0]
        _assert_sample(root_free, height, width)
        assert int(root_free["group_id"]) == 2
        assert torch.count_nonzero(root_free["target_group"]) == 0
        assert not torch.equal(root_free["fixed_mineral"], deployment["fixed_mineral"])
        assert root_free_dataset.samples["case"]["mineral"] == str(
            unregistered / "case" / "case_1.png"
        )

        try:
            _dataset(
                mineral_only,
                unregistered,
                height=height,
                width=width,
                synthetic_prob=0.0,
                require_registered_targets=True,
            )
        except RuntimeError as error:
            assert "target" in str(error).lower()
        else:
            raise AssertionError("Missing registered supervision target was accepted")

        # Real Stage 2 has paired images but no affine parameter label.
        stage2 = _dataset(
            registered,
            unregistered,
            height=height,
            width=width,
            synthetic_prob=0.0,
            require_registered_targets=True,
        )[0]
        _assert_sample(stage2, height, width)
        assert not bool(stage2["has_params"])
        assert torch.isnan(stage2["params_true"]).all()
        assert not torch.equal(stage2["moving_group"], stage2["target_group"])

        # Stage 1 creates moving from target and labels moving -> target.
        stage1 = _dataset(
            registered,
            no_unregistered,
            height=height,
            width=width,
            synthetic_prob=1.0,
            require_registered_targets=True,
        )[0]
        _assert_sample(stage1, height, width)
        assert bool(stage1["has_params"])
        expected = torch.tensor((2.0 / (width / 2.0), 0.0, 0.0, 1.0, 1.0))
        assert torch.allclose(stage1["params_true"], expected, atol=1e-7)
        assert not torch.equal(stage1["moving_group"], stage1["target_group"])
        moving_error = F.mse_loss(stage1["moving_group"], stage1["target_group"])
        registered_group = warp_group(
            stage1["moving_group"].unsqueeze(0),
            stage1["params_true"].unsqueeze(0),
        ).squeeze(0)
        registered_error = F.mse_loss(registered_group, stage1["target_group"])
        assert float(registered_error) < float(moving_error)


def assert_local_correlation_contract(device: torch.device) -> None:
    correlation = LocalCorrelationVolume(radius=1, temperature=0.05).to(device)
    fixed = torch.zeros((1, 2, 7, 7), device=device)
    moving = torch.zeros_like(fixed)
    fixed[0, 0, 3, 3] = 1.0
    moving[0, 0, 3, 4] = 1.0
    volume, expected, confidence, certainty = correlation(fixed, moving)
    assert volume.shape == (1, 9, 7, 7)
    # C(x,d) compares fixed(x) with moving(x+d), so this match is dx=+1.
    assert float(expected[0, 0, 3, 3]) > 0.95
    assert abs(float(expected[0, 1, 3, 3])) < 0.05
    assert torch.isfinite(confidence).all()
    assert torch.isfinite(certainty).all()

    fixed_valid = torch.zeros((1, 1, 7, 7), device=device)
    fixed_valid[..., 3, 3] = 1.0
    moving_valid = torch.ones_like(fixed_valid)
    moving_valid[..., 3, 4] = 0.0
    excluded, _, _, _ = correlation(
        fixed,
        moving,
        fixed_valid=fixed_valid,
        moving_valid=moving_valid,
    )
    displacement_index = int(
        torch.nonzero(
            (correlation.displacements[:, 0] == 1)
            & (correlation.displacements[:, 1] == 0),
            as_tuple=False,
        )[0]
    )
    assert float(excluded[0, displacement_index, 3, 3]) == 0.0
    invalid = torch.zeros_like(fixed_valid)
    for output in correlation(fixed, moving, fixed_valid=invalid, moving_valid=invalid):
        assert torch.equal(output, torch.zeros_like(output))


def _loss_args() -> SimpleNamespace:
    return SimpleNamespace(
        ncc_weight=0.0,
        edge_weight=0.0,
        charbonnier_weight=1.0,
        gradient_weight=0.0,
        overlap_weight=0.0,
        param_weight=0.0,
        reg_weight=0.0,
    )


def _image_loss(moving, target, valid, params) -> torch.Tensor:
    warped = warp_group(moving, params)
    overlap = build_group_valid_overlap(moving, target, valid, params)
    total, _ = grouped_loss(
        _loss_args(),
        params=params,
        warped_group=warped,
        target_group=target,
        valid_group=valid,
        group_id=torch.full(valid.shape[:1], 2, dtype=torch.long, device=moving.device),
        params_true=params.detach(),
        has_params=torch.zeros(valid.shape[0], dtype=torch.bool, device=moving.device),
        valid_overlap=overlap,
    )
    return total


def assert_target_is_supervision_only(model, device, height: int, width: int) -> None:
    fixed = torch.full((1, 3, height, width), 0.15, device=device)
    fixed[..., 7:-7, 9:-9] = 0.85
    moving = torch.full((1, MAX_GROUP_STAINS, 3, height, width), 0.2, device=device)
    moving[:, 0, :, 6:-8, 8:-6] = 0.75
    moving[:, 1:] = 0.0
    group = torch.tensor((2,), dtype=torch.long, device=device)
    valid = torch.tensor(((True, False, False),), dtype=torch.bool, device=device)

    model.eval()
    with torch.no_grad():
        before = model(fixed_mineral=fixed, moving_group=moving, group=group)
        # There is intentionally no target argument to vary between calls.
        after = model(fixed_mineral=fixed, moving_group=moving, group=group)
    assert torch.equal(before, after)

    identity = moving.new_tensor(((0.0, 0.0, 0.0, 1.0, 1.0),))
    target_a = moving.clone()
    target_b = target_a.clone()
    target_b[:, 0] = 0.9
    baseline = _image_loss(moving, target_a, valid, identity)
    changed_valid = _image_loss(moving, target_b, valid, identity)
    assert float(changed_valid) > float(baseline) + 0.1

    target_invalid_changed = target_a.clone()
    target_invalid_changed[:, 1:] = 1.0
    changed_invalid = _image_loss(moving, target_invalid_changed, valid, identity)
    assert torch.equal(changed_invalid, baseline)


def assert_model_contract(model, device, height: int, width: int) -> None:
    signature = inspect.signature(model.forward)
    assert tuple(signature.parameters) == (
        "fixed_mineral",
        "moving_group",
        "group",
        "return_aux",
        "_batch_aux_metadata",
    )
    assert all(
        parameter.kind is inspect.Parameter.POSITIONAL_OR_KEYWORD
        for parameter in signature.parameters.values()
    )

    fixed = torch.zeros((2, 3, height, width), device=device)
    fixed[:, :, 6:-6, 7:-7] = 0.8
    moving = torch.zeros((2, MAX_GROUP_STAINS, 3, height, width), device=device)
    moving[0, 0, :, 8:-5, 5:-9] = 0.9
    moving[1, 0, :, 5:-9, 9:-5] = 0.7
    groups = torch.tensor((2, 3), dtype=torch.long, device=device)

    try:
        model(
            fixed_mineral=fixed,
            moving_group=moving,
            group=groups,
            target_group=moving,
        )
    except TypeError:
        pass
    else:
        raise AssertionError("model.forward accepted a registered target")

    shared_encoders = [
        module
        for module in model.modules()
        if isinstance(module, FeaturePyramidEncoder)
    ]
    assert shared_encoders == [model.encoder.shared_encoder]

    model.train()
    assert model.heads is not None
    _randomize_affine_outputs(model, 0.0)
    model.zero_grad(set_to_none=True)
    params = model(fixed_mineral=fixed, moving_group=moving, group=groups)
    assert params.shape == (2, 5) and torch.isfinite(params).all()
    target_params = params.detach().new_tensor(
        (
            (0.08, -0.04, math.radians(4.0), 0.95, 1.05),
            (-0.05, 0.06, math.radians(-3.0), 1.04, 0.96),
        )
    )
    F.smooth_l1_loss(params, target_params).backward()
    gradients = [
        parameter.grad
        for parameter in model.parameters()
        if parameter.requires_grad and parameter.grad is not None
    ]
    assert gradients
    assert all(torch.isfinite(gradient).all() for gradient in gradients)
    assert sum(float(gradient.abs().sum()) for gradient in gradients) > 0.0

    identity = params.new_tensor((0.0, 0.0, 0.0, 1.0, 1.0))
    model.eval()
    with torch.no_grad():
        no_moving = model(
            fixed_mineral=fixed[:1],
            moving_group=torch.zeros_like(moving[:1]),
            group=torch.tensor((2,), dtype=torch.long, device=device),
        )
        group1 = model(
            fixed_mineral=fixed[:1],
            moving_group=moving[:1],
            group=torch.tensor((1,), dtype=torch.long, device=device),
        )
    assert torch.equal(no_moving[0], identity)
    assert torch.equal(group1[0], identity)

    # One Bx5 vector is expanded across all K slots by warp_group.
    base = torch.rand((1, 1, 3, 20, 20), device=device)
    repeated = base.expand(-1, MAX_GROUP_STAINS, -1, -1, -1).clone()
    shared_params = repeated.new_tensor(((0.12, -0.06, math.radians(3.0), 1.03, 0.97),))
    warped = warp_group(repeated, shared_params)
    assert torch.equal(warped[:, 0], warped[:, 1])
    assert torch.equal(warped[:, 1], warped[:, 2])


def _randomize_affine_outputs(model, translation_bias: float) -> None:
    assert model.heads is not None
    with torch.no_grad():
        for head in model.heads:
            translation_output = (
                head.output
                if isinstance(head, AffineHead)
                else head.translation_head.output
            )
            torch.nn.init.normal_(translation_output.weight, mean=0.0, std=1e-3)
            translation_output.bias[0] += translation_bias


def _gradient_sum(model) -> float:
    return sum(
        float(parameter.grad.abs().sum())
        for parameter in model.parameters()
        if parameter.grad is not None
    )


def _state_snapshot(module) -> dict[str, torch.Tensor]:
    """Clone parameters and persistent buffers for phase-transition checks."""
    return {name: value.detach().clone() for name, value in module.state_dict().items()}


def _assert_state_unchanged(module, before: dict[str, torch.Tensor]) -> None:
    after = module.state_dict()
    assert after.keys() == before.keys()
    for name, expected in before.items():
        assert torch.equal(after[name], expected), f"Unexpected state update: {name}"


def _assert_parameter_update(module, before: dict[str, torch.Tensor]) -> None:
    current_parameters = dict(module.named_parameters())
    changed = [
        name
        for name, parameter in current_parameters.items()
        if not torch.equal(parameter.detach(), before[name])
    ]
    assert changed, "Expected at least one trainable parameter to update"


def _teacher_loss_args(
    *,
    detach_teacher: bool,
    warmup_epochs: int = 0,
    freeze_teacher: bool = False,
):
    return SimpleNamespace(
        ncc_weight=0.0,
        edge_weight=0.0,
        charbonnier_weight=0.1,
        gradient_weight=0.0,
        overlap_weight=0.0,
        param_weight=0.0,
        reg_weight=0.0,
        use_teacher_branch=True,
        teacher_distill_weight=1.5,
        detach_teacher=detach_teacher,
        teacher_warmup_epochs=warmup_epochs,
        freeze_teacher=freeze_teacher,
    )


def _teacher_batch(device: torch.device, height: int, width: int):
    fixed = torch.zeros((1, 3, height, width), device=device)
    fixed[..., 5:-5, 7:-7] = 0.8
    moving = torch.zeros((1, MAX_GROUP_STAINS, 3, height, width), device=device)
    target = torch.zeros_like(moving)
    moving[:, 0, :, 8:-6, 5:-9] = 0.7
    target[:, 0, :, 6:-8, 8:-6] = 0.9
    return {
        "fixed_mineral": fixed,
        "moving_group": moving,
        "target_group": target,
        "valid_group": torch.tensor(
            ((True, False, False),), dtype=torch.bool, device=device
        ),
        "group_id": torch.tensor((2,), dtype=torch.long, device=device),
        "params_true": torch.full((1, 5), float("nan"), device=device),
        "has_params": torch.tensor((False,), dtype=torch.bool, device=device),
    }


def assert_no_enhancement_contract() -> None:
    forbidden = ("enhance", "morphology", "hue_range", "cfo", "trap_")
    for constructor in (
        OnlineStructuralFrontend.__init__,
        CorrelationVolumeAffineRegistrationModel.__init__,
    ):
        parameter_names = tuple(inspect.signature(constructor).parameters)
        assert not any(token in name for name in parameter_names for token in forbidden)
    with mock.patch.object(
        sys,
        "argv",
        [
            "train.py",
            "--registered_root",
            "registered",
            "--unregistered_root",
            "unregistered",
            "--output_dir",
            "output",
        ],
    ):
        parsed = parse_train_args()
    assert not any(token in name for name in vars(parsed) for token in forbidden)
    frontend = OnlineStructuralFrontend(
        input_channels=3,
        sfo_mode="rgb",
        foreground_threshold=0.2,
        distance_scale=0.05,
        context_scale=0.05,
        skeleton_radius=2,
    )
    assert not hasattr(frontend, "enhance_groups")


def assert_teacher_student_contract(
    device: torch.device, height: int, width: int
) -> None:
    config = build_config()
    wrapper = TeacherStudentAffineRegistrationModel(
        student_config=config, use_teacher_branch=True
    ).to(device)
    assert wrapper.teacher is not None
    student_state = wrapper.student.state_dict()
    teacher_state = wrapper.teacher.state_dict()
    assert student_state.keys() == teacher_state.keys()
    assert all(
        torch.equal(student_state[name], teacher_state[name]) for name in student_state
    )
    student_parameters = dict(wrapper.student.named_parameters())
    teacher_parameters = dict(wrapper.teacher.named_parameters())
    assert student_parameters.keys() == teacher_parameters.keys()
    assert all(
        student_parameters[name].data_ptr() != teacher_parameters[name].data_ptr()
        for name in student_parameters
    )

    batch = _teacher_batch(device, height, width)
    wrapper.eval()
    with torch.no_grad():
        positional = wrapper(
            batch["fixed_mineral"], batch["moving_group"], batch["group_id"]
        )
        keyword = wrapper(
            fixed_mineral=batch["fixed_mineral"],
            moving_group=batch["moving_group"],
            group=batch["group_id"],
        )
        teacher_params = wrapper.forward_teacher(
            batch["fixed_mineral"],
            batch["target_group"],
            batch["moving_group"],
            batch["group_id"],
        )
    assert torch.equal(positional, keyword)
    assert positional.shape == teacher_params.shape == (1, 5)
    assert torch.isfinite(positional).all() and torch.isfinite(teacher_params).all()

    try:
        wrapper(
            batch["fixed_mineral"],
            batch["moving_group"],
            batch["group_id"],
            batch["target_group"],
        )
    except (TypeError, RuntimeError, ValueError):
        pass
    else:
        raise AssertionError("Deployable student accepted target_group")

    without_teacher = TeacherStudentAffineRegistrationModel(
        student_config=config, use_teacher_branch=False
    ).to(device)
    try:
        without_teacher.forward_teacher(
            batch["fixed_mineral"],
            batch["target_group"],
            batch["moving_group"],
            batch["group_id"],
        )
    except RuntimeError:
        pass
    else:
        raise AssertionError("Disabled teacher branch produced a prediction")

    shared_encoders = [
        module
        for module in wrapper.modules()
        if isinstance(module, FeaturePyramidEncoder)
    ]
    assert len(shared_encoders) == 2
    known_params = teacher_params.new_tensor(((0.05, 0.0, 0.0, 1.0, 1.0),))
    supervised = affine_control_point_loss(teacher_params, known_params)
    assert torch.isfinite(supervised)


def assert_distillation_contract(device: torch.device, height: int, width: int) -> None:
    warmup_args = _teacher_loss_args(detach_teacher=True, warmup_epochs=2)
    assert teacher_distillation_weight(warmup_args, 1) == 0.0
    assert teacher_distillation_weight(warmup_args, 2) == 0.0
    assert teacher_distillation_weight(warmup_args, 3) == 1.5

    for detach_teacher in (True, False):
        wrapper = TeacherStudentAffineRegistrationModel(
            student_config=build_config(), use_teacher_branch=True
        ).to(device)
        assert wrapper.teacher is not None
        _randomize_affine_outputs(wrapper.student, 0.20)
        _randomize_affine_outputs(wrapper.teacher, -0.20)
        batch = _teacher_batch(device, height, width)
        args = _teacher_loss_args(detach_teacher=detach_teacher)
        wrapper.zero_grad(set_to_none=True)
        total, components, _ = compute_training_loss(args, wrapper, batch, epoch=1)
        assert torch.isfinite(total)
        assert "teacher_distill" in components
        assert not any(
            name.endswith(metric_name)
            for name in components
            for metric_name in AFFINE_ERROR_METRIC_NAMES
        )
        assert float(components["teacher_distill"]) > 0.0
        total.backward()
        assert _gradient_sum(wrapper.student) > 0.0
        teacher_gradient = _gradient_sum(wrapper.teacher)
        if detach_teacher:
            assert teacher_gradient == 0.0
        else:
            assert teacher_gradient > 0.0

    wrapper = TeacherStudentAffineRegistrationModel(
        student_config=build_config(), use_teacher_branch=True
    ).to(device)
    _randomize_affine_outputs(wrapper.student, 0.10)
    assert wrapper.teacher is not None
    _randomize_affine_outputs(wrapper.teacher, -0.10)
    batch = _teacher_batch(device, height, width)
    batch["has_params"] = torch.tensor((True,), dtype=torch.bool, device=device)
    batch["params_true"] = batch["moving_group"].new_tensor(
        ((0.05, 0.0, 0.0, 1.0, 1.0),)
    )
    args = _teacher_loss_args(detach_teacher=True)
    _, components, _ = compute_training_loss(args, wrapper, batch, epoch=1)
    expected_components = {
        f"{path_name}_{metric_name}"
        for path_name in ("student", "teacher")
        for metric_name in AFFINE_ERROR_METRIC_NAMES
    }
    assert expected_components.issubset(components)
    assert all(torch.isfinite(components[name]) for name in expected_components)
    for path_name, teacher in (("student", False), ("teacher", True)):
        validation_metrics = evaluate_path(
            args,
            wrapper,
            [batch],
            device,
            path_name=path_name,
            teacher=teacher,
        )
        expected_validation = {
            f"val_{path_name}_{metric_name}"
            for metric_name in AFFINE_ERROR_METRIC_NAMES
        }
        assert expected_validation.issubset(validation_metrics)


def assert_teacher_phase_cli_and_validation_contract() -> None:
    """Accept mixed warmup and reject only structurally invalid phases."""
    defaults = _parse_train_cli()
    assert defaults.teacher_warmup_epochs == 0
    assert defaults.freeze_teacher is False
    validate_args(defaults)

    for synthetic_prob, param_weight in (
        (0.0, 0.0),
        (0.0, 2.0),
        (0.5, 0.0),
        (0.5, 2.0),
        (1.0, 0.0),
        (1.0, 2.0),
    ):
        parsed = _parse_train_cli(
            "--use_teacher_branch",
            "--teacher_warmup_epochs",
            "1",
            "--synthetic_prob",
            str(synthetic_prob),
            "--val_synthetic_prob",
            "0.0",
            "--param_weight",
            str(param_weight),
        )
        validate_args(parsed)

    invalid_cases = (
        (
            _parse_train_cli("--freeze_teacher"),
            "requires --use_teacher_branch",
        ),
        (
            _parse_train_cli("--use_teacher_branch", "--freeze_teacher"),
            "requires a full pretrained",
        ),
        (
            _parse_train_cli(
                "--use_teacher_branch",
                "--freeze_teacher",
                "--resume_checkpoint",
                "resume.pt",
                "--teacher_warmup_epochs",
                "1",
            ),
            "conflicts with teacher-only warmup",
        ),
        (
            _parse_train_cli("--teacher_warmup_epochs", "1"),
            "requires --use_teacher_branch",
        ),
        (
            _parse_train_cli(
                "--use_teacher_branch",
                "--teacher_warmup_epochs",
                "1",
                "--synthetic_prob",
                "1.1",
            ),
            "synthetic_prob must be in [0, 1]",
        ),
        (
            _parse_train_cli(
                "--use_teacher_branch",
                "--teacher_warmup_epochs",
                "1",
                "--param_weight",
                "-0.1",
            ),
            "Loss weights cannot be negative",
        ),
    )
    for parsed, expected_message in invalid_cases:
        try:
            validate_args(parsed)
        except ValueError as error:
            assert expected_message in str(error)
        else:
            raise AssertionError(
                f"Invalid teacher configuration was accepted: {vars(parsed)}"
            )


def assert_teacher_only_warmup_contract(
    device: torch.device, height: int, width: int
) -> None:
    """Warmup updates only the teacher, then releases the student at N+1."""
    config = build_config()
    config["norm_type"] = "batch"
    wrapper = TeacherStudentAffineRegistrationModel(
        student_config=config, use_teacher_branch=True
    ).to(device)
    assert wrapper.teacher is not None
    assert any(
        isinstance(module, torch.nn.BatchNorm2d) for module in wrapper.student.modules()
    )
    assert any(
        isinstance(module, torch.nn.BatchNorm2d) for module in wrapper.teacher.modules()
    )

    args = _teacher_loss_args(detach_teacher=True, warmup_epochs=1)
    args.param_weight = 1.0
    batch = _teacher_batch(device, height, width)
    batch["has_params"] = torch.ones((1,), dtype=torch.bool, device=device)
    batch["params_true"] = batch["moving_group"].new_tensor(
        ((0.12, -0.07, math.radians(4.0), 0.96, 1.04),)
    )
    optimizer = torch.optim.SGD(wrapper.parameters(), lr=0.1)

    student_before = _state_snapshot(wrapper.student)
    teacher_before = _state_snapshot(wrapper.teacher)
    assert teacher_warmup_active(args, 1)
    assert configure_training_phase(args, wrapper, epoch=1) == "teacher_warmup"
    assert not wrapper.student.training
    assert wrapper.teacher.training
    assert all(
        not parameter.requires_grad for parameter in wrapper.student.parameters()
    )
    assert all(parameter.requires_grad for parameter in wrapper.teacher.parameters())

    optimizer.zero_grad(set_to_none=True)
    total, components, _ = compute_training_loss(args, wrapper, batch, epoch=1)
    assert torch.isfinite(total)
    assert "teacher_param" in components
    assert "teacher_charbonnier" in components
    assert "teacher_distill" not in components
    student_loss_names = {
        "student_ncc",
        "student_edge",
        "student_charbonnier",
        "student_gradient",
        "student_overlap",
        "student_param",
        "student_reg",
        "student_total",
    }
    assert student_loss_names.isdisjoint(components)
    assert teacher_distillation_weight(args, 1) == 0.0
    total.backward()
    assert _gradient_sum(wrapper.student) == 0.0
    assert _gradient_sum(wrapper.teacher) > 0.0
    optimizer.step()
    _assert_state_unchanged(wrapper.student, student_before)
    _assert_parameter_update(wrapper.teacher, teacher_before)

    assert not teacher_warmup_active(args, 2)
    assert configure_training_phase(args, wrapper, epoch=2) == "student"
    assert wrapper.student.training and wrapper.teacher.training
    assert all(parameter.requires_grad for parameter in wrapper.student.parameters())
    student_before_release = _state_snapshot(wrapper.student)
    optimizer.zero_grad(set_to_none=True)
    released_total, released_components, _ = compute_training_loss(
        args, wrapper, batch, epoch=2
    )
    assert "teacher_distill" in released_components
    assert teacher_distillation_weight(args, 2) == args.teacher_distill_weight
    released_total.backward()
    assert _gradient_sum(wrapper.student) > 0.0
    optimizer.step()
    _assert_parameter_update(wrapper.student, student_before_release)


def assert_real_and_mixed_teacher_warmup_contract(
    device: torch.device, height: int, width: int
) -> None:
    """Warmup learns from real images and masks affine labels per sample."""
    args = _teacher_loss_args(detach_teacher=True, warmup_epochs=1)
    args.ncc_weight = 0.1
    args.edge_weight = 0.1
    args.charbonnier_weight = 0.1
    args.gradient_weight = 0.1
    args.overlap_weight = 0.1
    args.param_weight = 0.0
    args.reg_weight = 0.1

    real_model = TeacherStudentAffineRegistrationModel(
        student_config=build_config(), use_teacher_branch=True
    ).to(device)
    assert real_model.teacher is not None
    real_batch = _teacher_batch(device, height, width)
    assert not bool(real_batch["has_params"].any())
    assert torch.isnan(real_batch["params_true"]).all()
    assert configure_training_phase(args, real_model, epoch=1) == "teacher_warmup"
    optimizer = torch.optim.SGD(real_model.parameters(), lr=0.1)
    student_before = _state_snapshot(real_model.student)
    teacher_before = _state_snapshot(real_model.teacher)
    optimizer.zero_grad(set_to_none=True)
    real_total, real_components, _ = compute_training_loss(
        args, real_model, real_batch, epoch=1
    )
    expected_real_losses = {
        "teacher_ncc",
        "teacher_edge",
        "teacher_charbonnier",
        "teacher_gradient",
        "teacher_overlap",
        "teacher_reg",
        "teacher_total",
    }
    assert expected_real_losses.issubset(real_components)
    assert "teacher_param" not in real_components
    assert "teacher_distill" not in real_components
    assert not any(name.startswith("student_") for name in real_components)
    assert torch.isfinite(real_total)
    assert all(torch.isfinite(real_components[name]) for name in expected_real_losses)
    real_total.backward()
    assert _gradient_sum(real_model.student) == 0.0
    assert _gradient_sum(real_model.teacher) > 0.0
    optimizer.step()
    _assert_state_unchanged(real_model.student, student_before)
    _assert_parameter_update(real_model.teacher, teacher_before)

    # A fully real validation loader (val_synthetic_prob=0.0) has no affine
    # metrics or parameter loss but retains finite image-registration metrics.
    for path_name, teacher in (("student", False), ("teacher", True)):
        metrics = evaluate_path(
            args,
            real_model,
            [real_batch],
            device,
            path_name=path_name,
            teacher=teacher,
        )
        assert math.isfinite(metrics[f"val_{path_name}_total"])
        assert f"val_{path_name}_charbonnier" in metrics
        assert f"val_{path_name}_reg" in metrics
        assert f"val_{path_name}_param" not in metrics
        assert not any(
            name == f"val_{path_name}_{metric_name}"
            for name in metrics
            for metric_name in AFFINE_ERROR_METRIC_NAMES
        )

    args.param_weight = 1.0
    mixed_model = TeacherStudentAffineRegistrationModel(
        student_config=build_config(), use_teacher_branch=True
    ).to(device)
    assert mixed_model.teacher is not None
    base_batch = _teacher_batch(device, height, width)
    mixed_batch = {
        name: value.repeat((2,) + (1,) * (value.ndim - 1))
        for name, value in base_batch.items()
    }
    mixed_batch["has_params"] = torch.tensor(
        (True, False), dtype=torch.bool, device=device
    )
    params_true = mixed_batch["moving_group"].new_tensor(
        (0.12, -0.07, math.radians(4.0), 0.96, 1.04)
    )
    mixed_batch["params_true"][0] = params_true
    synthesis = invert_affine_matrix(
        affine_parameters_to_matrix(params_true.unsqueeze(0))
    )
    mixed_batch["moving_group"][:1] = warp_group_with_matrix(
        mixed_batch["target_group"][:1], synthesis
    )
    assert torch.isfinite(mixed_batch["params_true"][0]).all()
    assert torch.isnan(mixed_batch["params_true"][1]).all()
    assert configure_training_phase(args, mixed_model, epoch=1) == "teacher_warmup"
    mixed_model.zero_grad(set_to_none=True)
    mixed_total, mixed_components, _ = compute_training_loss(
        args, mixed_model, mixed_batch, epoch=1
    )
    expected_mixed_losses = expected_real_losses | {"teacher_param"}
    assert expected_mixed_losses.issubset(mixed_components)
    assert "teacher_distill" not in mixed_components
    assert torch.isfinite(mixed_total)
    assert all(torch.isfinite(mixed_components[name]) for name in expected_mixed_losses)
    expected_param = affine_control_point_loss(
        mixed_model.forward_teacher(
            fixed_mineral=mixed_batch["fixed_mineral"],
            target_group=mixed_batch["target_group"],
            moving_group=mixed_batch["moving_group"],
            group=mixed_batch["group_id"],
        )[:1],
        mixed_batch["params_true"][:1],
    )
    torch.testing.assert_close(mixed_components["teacher_param"], expected_param)
    mixed_total.backward()
    assert _gradient_sum(mixed_model.student) == 0.0
    assert _gradient_sum(mixed_model.teacher) > 0.0

    invalid_batch = {name: value.clone() for name, value in real_batch.items()}
    invalid_batch["has_params"] = torch.ones((1,), dtype=torch.bool, device=device)
    try:
        compute_training_loss(args, real_model, invalid_batch, epoch=1)
    except ValueError as error:
        assert "has_params=True" in str(error)
    else:
        raise AssertionError("Synthetic sample without params_true was accepted")


def assert_frozen_teacher_contract(
    device: torch.device, height: int, width: int
) -> None:
    """A Stage-2 teacher is eval-only and supplies a detached target."""
    config = build_config()
    config["norm_type"] = "batch"
    wrapper = TeacherStudentAffineRegistrationModel(
        student_config=config, use_teacher_branch=True
    ).to(device)
    assert wrapper.teacher is not None
    _randomize_affine_outputs(wrapper.student, 0.20)
    _randomize_affine_outputs(wrapper.teacher, -0.20)
    args = _teacher_loss_args(detach_teacher=False, freeze_teacher=True)
    batch = _teacher_batch(device, height, width)
    optimizer = torch.optim.SGD(wrapper.parameters(), lr=0.1)

    assert configure_training_phase(args, wrapper, epoch=1) == "student"
    assert wrapper.student.training
    assert not wrapper.teacher.training
    assert all(parameter.requires_grad for parameter in wrapper.student.parameters())
    assert all(
        not parameter.requires_grad for parameter in wrapper.teacher.parameters()
    )
    teacher_before = _state_snapshot(wrapper.teacher)
    student_before = _state_snapshot(wrapper.student)
    teacher_predictions = []
    hook = wrapper.teacher.register_forward_hook(
        lambda _module, _inputs, output: teacher_predictions.append(output)
    )
    optimizer.zero_grad(set_to_none=True)
    total, components, _ = compute_training_loss(args, wrapper, batch, epoch=1)
    hook.remove()
    assert teacher_predictions
    assert all(
        not prediction.requires_grad and prediction.grad_fn is None
        for prediction in teacher_predictions
    )
    assert float(components["teacher_distill"]) > 0.0
    assert components["teacher_distill"].requires_grad
    total.backward()
    assert _gradient_sum(wrapper.teacher) == 0.0
    assert _gradient_sum(wrapper.student) > 0.0
    optimizer.step()
    _assert_state_unchanged(wrapper.teacher, teacher_before)
    _assert_parameter_update(wrapper.student, student_before)


def _module_gradient_sum(module) -> float:
    gradients = [
        parameter.grad
        for parameter in module.parameters()
        if parameter.grad is not None
    ]
    assert gradients
    assert all(torch.isfinite(gradient).all() for gradient in gradients)
    return sum(float(gradient.abs().sum()) for gradient in gradients)


def assert_frontend_adapter_training_contract(
    device: torch.device, height: int, width: int
) -> None:
    for frontend_mode, group_input_mode in (
        ("raw", "overlay"),
        ("hybrid", "stack"),
    ):
        wrapper = TeacherStudentAffineRegistrationModel(
            student_config=build_config(frontend_mode, group_input_mode),
            use_teacher_branch=True,
        ).to(device)
        assert wrapper.teacher is not None
        _randomize_affine_outputs(wrapper.student, 0.20)
        _randomize_affine_outputs(wrapper.teacher, -0.20)
        batch = _teacher_batch(device, height, width)
        args = _teacher_loss_args(detach_teacher=False)
        wrapper.zero_grad(set_to_none=True)
        total, components, params = compute_training_loss(args, wrapper, batch, epoch=1)
        assert total.isfinite()
        assert params.shape == (1, 5)
        assert float(components["teacher_distill"]) > 0.0
        total.backward()
        assert wrapper.student.mineral_frontend_adapter is not None
        assert wrapper.student.group_frontend_adapter is not None
        assert wrapper.teacher.mineral_frontend_adapter is not None
        assert wrapper.teacher.group_frontend_adapter is not None
        assert _module_gradient_sum(wrapper.student.mineral_frontend_adapter) > 0.0
        assert _module_gradient_sum(wrapper.student.group_frontend_adapter) > 0.0
        assert _module_gradient_sum(wrapper.teacher.mineral_frontend_adapter) > 0.0
        assert _module_gradient_sum(wrapper.teacher.group_frontend_adapter) > 0.0


class _KeywordOnlyRecordingStudent(CorrelationVolumeAffineRegistrationModel):
    """Inference student that rejects positional or extra model inputs."""

    calls: list[dict[str, torch.Tensor]] = []

    def forward(
        self,
        *,
        fixed_mineral: torch.Tensor,
        moving_group: torch.Tensor,
        group: torch.Tensor,
    ) -> torch.Tensor:
        params = super().forward(
            fixed_mineral=fixed_mineral,
            moving_group=moving_group,
            group=group,
        )
        type(self).calls.append(
            {
                "fixed_mineral": fixed_mineral.detach().cpu().clone(),
                "moving_group": moving_group.detach().cpu().clone(),
                "group": group.detach().cpu().clone(),
                "params": params.detach().cpu().clone(),
            }
        )
        return params


def _run_recorded_inference(
    *,
    checkpoint: Path,
    registered_root: Path | None,
    unregistered_root: Path,
    output_dir: Path,
    eval_with_registered_targets: bool | None = None,
) -> dict[str, list[dict[str, object]]]:
    """Run one CPU inference item while recording shared parity helpers."""
    _KeywordOnlyRecordingStudent.calls = []
    warp_calls: list[dict[str, object]] = []
    overlay_calls: list[dict[str, object]] = []

    def recording_warp(moving_group, predicted_params):
        warped = utils_module.warp_model_space_group(moving_group, predicted_params)
        warp_calls.append(
            {
                "moving_group": moving_group.detach().cpu().clone(),
                "params": predicted_params.detach().cpu().clone(),
                "warped_group": warped.detach().cpu().clone(),
            }
        )
        return warped

    def recording_overlay(
        path,
        fixed_mineral,
        warped_group,
        valid_group,
        group_id,
        sfo_mode,
    ):
        overlay_calls.append(
            {
                "path": Path(path),
                "fixed_mineral": fixed_mineral.detach().cpu().clone(),
                "warped_group": warped_group.detach().cpu().clone(),
                "valid_group": valid_group.detach().cpu().clone(),
                "group_id": int(group_id),
                "sfo_mode": str(sfo_mode),
            }
        )
        return utils_module.save_group_overlay(
            path,
            fixed_mineral,
            warped_group,
            valid_group,
            group_id,
            sfo_mode,
        )

    inference_args = SimpleNamespace(
        checkpoint=str(checkpoint),
        registered_root=(None if registered_root is None else str(registered_root)),
        unregistered_root=str(unregistered_root),
        output_dir=str(output_dir),
        batch_size=1,
        n_workers=0,
        device="cpu",
        gpu_ids="0",
        no_multi_gpu=True,
        include_group1=False,
        max_items=0,
        eval_with_registered_targets=eval_with_registered_targets,
    )
    with (
        mock.patch.object(
            inference_module,
            "CorrelationVolumeAffineRegistrationModel",
            _KeywordOnlyRecordingStudent,
        ),
        mock.patch.object(
            inference_module,
            "warp_model_space_group",
            side_effect=recording_warp,
        ),
        mock.patch.object(
            inference_module,
            "_save_group_overlay",
            side_effect=recording_overlay,
        ),
    ):
        inference_module.main(inference_args)

    return {
        "model": list(_KeywordOnlyRecordingStudent.calls),
        "warp": warp_calls,
        "overlay": overlay_calls,
    }


def assert_checkpoint_and_overlay_contract(
    device: torch.device, height: int, width: int
) -> None:
    config = build_config()
    model = TeacherStudentAffineRegistrationModel(
        student_config=config, use_teacher_branch=True
    ).to(device)
    assert model.teacher is not None
    _randomize_affine_outputs(model.student, 0.12)
    _randomize_affine_outputs(model.teacher, -0.12)
    preprocess = {
        "input_contract_version": "fixed_mineral_moving_group_raw_v1",
        "input_value_range": [0.0, 1.0],
        "height": height,
        "width": width,
        "image_mode": "rgb",
        "sfo_mode": "rgb",
        "crop_mode": "full",
        "crop_margin": 4,
        "include_group1": False,
    }
    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory)
        full_payload = {
            "architecture": ARCHITECTURE,
            "epoch": 1,
            "metrics": {"val_student_total": 0.5},
            "preprocess_config": preprocess,
        }
        save_checkpoint_pair(
            output_dir=str(root),
            checkpoint_name="best_model.pt",
            full_payload=full_payload,
            model=model,
            student_model_config=config,
            preprocess_config=preprocess,
        )
        full_path = root / "best_model.pt"
        student_path = root / "best_model_student.pt"
        full_checkpoint = torch.load(full_path, map_location="cpu", weights_only=False)
        student_checkpoint = torch.load(
            student_path, map_location="cpu", weights_only=False
        )
        assert full_checkpoint["checkpoint_type"] == "full_training"
        assert "teacher_model_state_dict" in full_checkpoint
        assert (
            full_checkpoint["model_config"]["teacher_fixed_input_version"]
            == TEACHER_FIXED_INPUT_VERSION
        )
        assert student_checkpoint["checkpoint_type"] == "deployable_student"
        assert "teacher_fixed_input_version" not in student_checkpoint["model_config"]
        assert (
            "teacher_fixed_input_version"
            not in student_checkpoint["student_model_config"]
        )
        for checkpoint in (full_checkpoint, student_checkpoint):
            assert checkpoint["student_model_config"]["frontend_mode"] == "structural"
            assert checkpoint["student_model_config"]["group_input_mode"] == "overlay"
            assert checkpoint["student_model_config"]["group_slots"] == MAX_GROUP_STAINS
            assert checkpoint["student_model_config"]["affine_head_mode"] == "joint"
            assert checkpoint["model_config"]["frontend_mode"] == "structural"
            assert checkpoint["model_config"]["group_input_mode"] == "overlay"
            assert checkpoint["model_config"]["affine_head_mode"] == "joint"
        assert set(key for key in student_checkpoint if "state_dict" in key) == {
            "student_model_state_dict"
        }
        assert not any(
            "teacher" in key or "optimizer" in key for key in student_checkpoint
        )

        full_preprocess, full_config, full_state = load_student_checkpoint(
            str(full_path)
        )
        loaded_preprocess, loaded_config, loaded_state = load_student_checkpoint(
            str(student_path)
        )
        assert full_preprocess == loaded_preprocess == preprocess
        assert full_config == loaded_config == config
        assert full_state.keys() == loaded_state.keys()
        assert all(
            torch.equal(full_state[name], loaded_state[name]) for name in full_state
        )
        deployable = CorrelationVolumeAffineRegistrationModel(**loaded_config).to(
            device
        )
        deployable.load_state_dict(loaded_state, strict=True)

        restored = TeacherStudentAffineRegistrationModel(
            student_config=config, use_teacher_branch=True
        ).to(device)
        load_initial_weights(
            restored,
            str(full_path),
            device,
            expected_student_model_config=config,
            expected_preprocess_config=preprocess,
            expected_use_teacher_branch=True,
        )
        assert all(
            torch.equal(
                model.student.state_dict()[name], restored.student.state_dict()[name]
            )
            for name in model.student.state_dict()
        )
        assert restored.teacher is not None and model.teacher is not None
        assert all(
            torch.equal(
                model.teacher.state_dict()[name], restored.teacher.state_dict()[name]
            )
            for name in model.teacher.state_dict()
        )

        mineral_only = root / "registered_mineral_only"
        registered = root / "registered"
        unregistered = root / "unregistered"
        _write_signal(
            mineral_only / "case" / "case_1.png",
            shift_x=0,
            color=(210, 210, 210),
        )
        _write_signal(
            registered / "case" / "case_1.png",
            shift_x=0,
            color=(210, 210, 210),
        )
        _write_signal(
            registered / "case" / "case_4.png",
            shift_x=0,
            color=(230, 40, 30),
        )
        _write_signal(
            unregistered / "case" / "case_1.png",
            shift_x=2,
            color=(40, 170, 210),
        )
        _write_signal(
            unregistered / "case" / "case_4.png",
            shift_x=3,
            color=(230, 40, 30),
        )
        dataset = _dataset(
            registered,
            unregistered,
            height=height,
            width=width,
            synthetic_prob=0.0,
            require_registered_targets=True,
        )
        loader = DataLoader(
            Subset(dataset, [0]),
            batch_size=1,
            shuffle=False,
            collate_fn=safe_collate,
        )
        overlay_args = SimpleNamespace(
            output_dir=str(root / "output"),
            use_teacher_branch=True,
            sfo_mode="rgb",
        )
        save_validation_overlays(
            overlay_args,
            restored,
            loader,
            dataset,
            [0],
            device,
        )
        for branch in ("student", "teacher"):
            images = list(
                (root / "output" / "validation_overlays" / branch).glob("*.png")
            )
            assert len(images) == 1
            pixels = np.asarray(Image.open(images[0]))
            assert pixels.shape == (height, width, 3)
            assert np.count_nonzero(pixels) > 0

        # The deployable inference route must be the exact validation route for
        # a real item (has_params=False), including target-free deployment.
        assert tuple(
            inspect.signature(utils_module.warp_model_space_group).parameters
        ) == ("moving_group", "predicted_params")
        assert (
            inference_module.warp_model_space_group
            is utils_module.warp_model_space_group
        )
        assert (
            train_module.warp_model_space_group is utils_module.warp_model_space_group
        )
        assert (
            train_module.warp_group_for_supervision
            is utils_module.warp_group_for_supervision
        )
        assert inference_module._save_group_overlay is utils_module.save_group_overlay
        assert train_module._save_group_overlay is utils_module.save_group_overlay

        deployment_dataset = _dataset(
            None,
            unregistered,
            height=height,
            width=width,
            synthetic_prob=0.0,
            require_registered_targets=False,
            fixed_mineral_root=unregistered,
        )
        assert len(deployment_dataset) == 1
        assert deployment_dataset.samples["case"]["mineral"] == str(
            unregistered / "case" / "case_1.png"
        )
        assert not torch.equal(
            deployment_dataset[0]["fixed_mineral"], dataset[0]["fixed_mineral"]
        )
        deployment_loader = DataLoader(
            Subset(deployment_dataset, [0]),
            batch_size=1,
            shuffle=False,
            collate_fn=safe_collate,
        )
        parity_student = CorrelationVolumeAffineRegistrationModel(**loaded_config).cpu()
        parity_student.load_state_dict(loaded_state, strict=True)
        validation_warp_calls: list[dict[str, torch.Tensor]] = []
        validation_overlay_calls: list[dict[str, object]] = []

        def recording_validation_warp(moving_group, predicted_params):
            warped = utils_module.warp_model_space_group(moving_group, predicted_params)
            validation_warp_calls.append(
                {
                    "moving_group": moving_group.detach().cpu().clone(),
                    "params": predicted_params.detach().cpu().clone(),
                    "warped_group": warped.detach().cpu().clone(),
                }
            )
            return warped

        def recording_validation_overlay(
            path,
            fixed_mineral,
            warped_group,
            valid_group,
            group_id,
            sfo_mode,
        ):
            validation_overlay_calls.append(
                {
                    "path": Path(path),
                    "fixed_mineral": fixed_mineral.detach().cpu().clone(),
                    "warped_group": warped_group.detach().cpu().clone(),
                    "valid_group": valid_group.detach().cpu().clone(),
                    "group_id": int(group_id),
                    "sfo_mode": str(sfo_mode),
                }
            )
            return utils_module.save_group_overlay(
                path,
                fixed_mineral,
                warped_group,
                valid_group,
                group_id,
                sfo_mode,
            )

        parity_validation_args = SimpleNamespace(
            output_dir=str(root / "parity_validation"),
            use_teacher_branch=False,
            sfo_mode=preprocess["sfo_mode"],
        )
        with (
            mock.patch.object(
                train_module,
                "warp_model_space_group",
                side_effect=recording_validation_warp,
            ),
            mock.patch.object(
                train_module,
                "_save_group_overlay",
                side_effect=recording_validation_overlay,
            ),
        ):
            save_validation_overlays(
                parity_validation_args,
                parity_student,
                deployment_loader,
                deployment_dataset,
                [0],
                torch.device("cpu"),
            )
        assert len(validation_warp_calls) == 1
        assert len(validation_overlay_calls) == 1

        target_free_output = root / "parity_inference_target_free"
        diagnostic_output = root / "parity_inference_diagnostic"
        target_free = _run_recorded_inference(
            checkpoint=full_path,
            registered_root=None,
            unregistered_root=unregistered,
            output_dir=target_free_output,
        )
        diagnostic = _run_recorded_inference(
            checkpoint=full_path,
            registered_root=registered,
            unregistered_root=unregistered,
            output_dir=diagnostic_output,
        )

        def only_call(records, name):
            calls = records[name]
            assert len(calls) == 1, (name, len(calls))
            return calls[0]

        target_free_model = only_call(target_free, "model")
        target_free_warp = only_call(target_free, "warp")
        target_free_overlay = only_call(target_free, "overlay")
        diagnostic_model = only_call(diagnostic, "model")
        diagnostic_warp = only_call(diagnostic, "warp")
        diagnostic_overlay = only_call(diagnostic, "overlay")
        validation_warp = validation_warp_calls[0]
        validation_overlay = validation_overlay_calls[0]

        # The keyword-only subclass makes any positional call, wrong keyword,
        # or accidental target_group model input fail before producing output.
        assert torch.equal(
            target_free_model["moving_group"], target_free_warp["moving_group"]
        )
        assert torch.equal(target_free_model["params"], target_free_warp["params"])
        direct_real_warp = utils_module.warp_model_space_group(
            target_free_warp["moving_group"],
            target_free_warp["params"],
        )
        assert torch.equal(target_free_warp["warped_group"], direct_real_warp)
        identity = target_free_warp["params"].new_tensor(((0.0, 0.0, 0.0, 1.0, 1.0),))
        assert not torch.equal(target_free_warp["params"], identity)

        assert torch.equal(
            target_free_overlay["fixed_mineral"],
            target_free_model["fixed_mineral"][0],
        )
        assert torch.equal(
            target_free_overlay["warped_group"],
            target_free_warp["warped_group"][0],
        )
        assert target_free_overlay["group_id"] == int(target_free_model["group"][0])
        assert target_free_overlay["group_id"] == 2
        assert target_free_overlay["sfo_mode"] == preprocess["sfo_mode"]
        expected_model_space_path = (
            target_free_output / "model_space_group_overlays" / "00000_case_G2.png"
        )
        assert target_free_overlay["path"] == expected_model_space_path
        assert expected_model_space_path.is_file()

        # Training validation and deployable inference receive identical real
        # tensors and predictions, then invoke the same shared warp and overlay.
        for key in ("moving_group", "params", "warped_group"):
            assert torch.equal(target_free_warp[key], validation_warp[key]), key
        for key in (
            "fixed_mineral",
            "warped_group",
            "valid_group",
        ):
            assert torch.equal(target_free_overlay[key], validation_overlay[key]), key
        assert target_free_overlay["group_id"] == validation_overlay["group_id"]
        assert target_free_overlay["sfo_mode"] == validation_overlay["sfo_mode"]

        # Loading registered targets for optional metrics must not alter any
        # student input, prediction, real warp, or model-space visualization.
        for key in ("fixed_mineral", "moving_group", "group", "params"):
            assert torch.equal(target_free_model[key], diagnostic_model[key]), key
        for key in ("moving_group", "params", "warped_group"):
            assert torch.equal(target_free_warp[key], diagnostic_warp[key]), key
        for key in (
            "fixed_mineral",
            "warped_group",
            "valid_group",
        ):
            assert torch.equal(target_free_overlay[key], diagnostic_overlay[key]), key
        assert target_free_overlay["group_id"] == diagnostic_overlay["group_id"]
        assert target_free_overlay["sfo_mode"] == diagnostic_overlay["sfo_mode"]

        def read_rgb(path):
            with Image.open(path) as image:
                return np.asarray(image.convert("RGB")).copy()

        validation_pixels = read_rgb(validation_overlay["path"])
        target_free_pixels = read_rgb(target_free_overlay["path"])
        diagnostic_pixels = read_rgb(diagnostic_overlay["path"])
        assert validation_pixels.shape == (height, width, 3)
        assert np.array_equal(target_free_pixels, validation_pixels)
        assert np.array_equal(diagnostic_pixels, target_free_pixels)
        assert not (target_free_output / "registered_target_metrics.csv").exists()
        assert (diagnostic_output / "registered_target_metrics.csv").is_file()

        # Stage-1 parity uses one literal full checkpoint and one identical
        # deterministic synthetic dataset item for both rendering paths.
        stage1_dataset = _dataset(
            registered,
            unregistered,
            height=height,
            width=width,
            synthetic_prob=1.0,
            require_registered_targets=True,
        )
        stage1_batch = safe_collate([stage1_dataset[0]])
        assert bool(stage1_batch["has_params"].all())
        stage1_validation_warps = []
        stage1_validation_overlays = []

        def record_stage1_validation_warp(moving_group, predicted_params):
            warped = utils_module.warp_model_space_group(moving_group, predicted_params)
            stage1_validation_warps.append(
                {
                    "moving_group": moving_group.detach().cpu().clone(),
                    "params": predicted_params.detach().cpu().clone(),
                    "warped_group": warped.detach().cpu().clone(),
                }
            )
            return warped

        def record_stage1_validation_overlay(
            path,
            fixed_mineral,
            warped_group,
            valid_group,
            group_id,
            sfo_mode,
        ):
            stage1_validation_overlays.append(
                {
                    "path": Path(path),
                    "fixed_mineral": fixed_mineral.detach().cpu().clone(),
                    "warped_group": warped_group.detach().cpu().clone(),
                    "valid_group": valid_group.detach().cpu().clone(),
                    "group_id": int(group_id),
                    "sfo_mode": str(sfo_mode),
                }
            )
            return utils_module.save_group_overlay(
                path,
                fixed_mineral,
                warped_group,
                valid_group,
                group_id,
                sfo_mode,
            )

        stage1_validation_args = SimpleNamespace(
            output_dir=str(root / "stage1_exact_validation"),
            use_teacher_branch=False,
            sfo_mode=preprocess["sfo_mode"],
        )
        with (
            mock.patch.object(
                train_module,
                "warp_model_space_group",
                side_effect=record_stage1_validation_warp,
            ),
            mock.patch.object(
                train_module,
                "_save_group_overlay",
                side_effect=record_stage1_validation_overlay,
            ),
        ):
            save_validation_overlays(
                stage1_validation_args,
                restored,
                [stage1_batch],
                stage1_dataset,
                [0],
                device,
            )
        assert len(stage1_validation_warps) == 1
        assert len(stage1_validation_overlays) == 1

        same_preprocess, same_config, same_state = load_student_checkpoint(
            str(full_path)
        )
        assert same_preprocess == preprocess
        stage1_inference_model = CorrelationVolumeAffineRegistrationModel(
            **same_config
        ).to(device)
        stage1_inference_model.load_state_dict(same_state, strict=True)
        stage1_inference_model.eval()
        stage1_inference_batch = train_module.move_required_tensors(
            stage1_batch, device
        )
        stage1_inference_warps = []

        def record_stage1_inference_warp(moving_group, predicted_params):
            warped = utils_module.warp_model_space_group(moving_group, predicted_params)
            stage1_inference_warps.append(
                {
                    "moving_group": moving_group.detach().cpu().clone(),
                    "params": predicted_params.detach().cpu().clone(),
                    "warped_group": warped.detach().cpu().clone(),
                }
            )
            return warped

        with (
            torch.no_grad(),
            mock.patch.object(
                inference_module,
                "warp_model_space_group",
                side_effect=record_stage1_inference_warp,
            ),
        ):
            stage1_params, stage1_warped = inference_module.predict_model_space_group(
                stage1_inference_model, stage1_inference_batch
            )
        assert len(stage1_inference_warps) == 1
        validation_warp = stage1_validation_warps[0]
        inference_warp = stage1_inference_warps[0]
        for key in ("moving_group", "params", "warped_group"):
            assert torch.equal(validation_warp[key], inference_warp[key]), key
        assert tuple(stage1_warped.shape) == tuple(stage1_batch["moving_group"].shape)
        torch.testing.assert_close(
            stage1_params.detach().cpu(), validation_warp["params"], atol=0, rtol=0
        )

        stage1_overlay = stage1_validation_overlays[0]
        stage1_inference_overlay_path = (
            root / "stage1_exact_inference" / stage1_overlay["path"].name
        )
        stage1_inference_overlay_path.parent.mkdir(parents=True, exist_ok=True)
        inference_module._save_group_overlay(
            str(stage1_inference_overlay_path),
            stage1_inference_batch["fixed_mineral"][0],
            stage1_warped[0],
            stage1_inference_batch["valid_group"][0],
            int(stage1_inference_batch["group_id"][0]),
            preprocess["sfo_mode"],
        )
        assert torch.equal(
            stage1_overlay["fixed_mineral"],
            stage1_inference_batch["fixed_mineral"][0].detach().cpu(),
        )
        assert torch.equal(
            stage1_overlay["warped_group"], stage1_warped[0].detach().cpu()
        )
        assert torch.equal(
            stage1_overlay["valid_group"],
            stage1_inference_batch["valid_group"][0].detach().cpu(),
        )
        assert stage1_overlay["group_id"] == int(stage1_inference_batch["group_id"][0])
        assert np.array_equal(
            read_rgb(stage1_overlay["path"]),
            read_rgb(stage1_inference_overlay_path),
        )

        supervision_only_warp = utils_module.warp_group_for_supervision(
            stage1_inference_batch["moving_group"],
            stage1_inference_batch["target_group"],
            stage1_params,
            stage1_inference_batch["params_true"],
            stage1_inference_batch["has_params"],
        )
        assert not torch.equal(stage1_warped, supervision_only_warp)


def assert_frontend_checkpoint_contract(
    device: torch.device, height: int, width: int
) -> None:
    preprocess = {
        "input_contract_version": "fixed_mineral_moving_group_raw_v1",
        "input_value_range": [0.0, 1.0],
        "height": height,
        "width": width,
        "image_mode": "rgb",
        "sfo_mode": "rgb",
        "crop_mode": "full",
        "crop_margin": 4,
        "include_group1": False,
    }
    fixed, moving, _, group = _frontend_inputs(device, height, width)
    cases = (
        ("structural", "overlay"),
        ("raw", "overlay"),
        ("hybrid", "stack"),
    )
    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory)
        saved = {}
        for frontend_mode, group_input_mode in cases:
            config = build_config(frontend_mode, group_input_mode)
            model = TeacherStudentAffineRegistrationModel(
                student_config=config, use_teacher_branch=False
            ).to(device)
            mode_root = root / f"{frontend_mode}_{group_input_mode}"
            mode_root.mkdir()
            save_checkpoint_pair(
                output_dir=str(mode_root),
                checkpoint_name="best_model.pt",
                full_payload={
                    "architecture": ARCHITECTURE,
                    "epoch": 1,
                    "metrics": {"val_student_total": 0.5},
                    "preprocess_config": preprocess,
                },
                model=model,
                student_model_config=config,
                preprocess_config=preprocess,
            )
            full_path = mode_root / "best_model.pt"
            student_path = mode_root / "best_model_student.pt"
            full_checkpoint = torch.load(
                full_path, map_location="cpu", weights_only=False
            )
            student_checkpoint = torch.load(
                student_path, map_location="cpu", weights_only=False
            )
            saved[(frontend_mode, group_input_mode)] = (
                full_path,
                student_path,
                full_checkpoint,
                student_checkpoint,
            )
            for checkpoint in (full_checkpoint, student_checkpoint):
                assert checkpoint["student_model_config"]["frontend_mode"] == (
                    frontend_mode
                )
                assert checkpoint["student_model_config"]["group_input_mode"] == (
                    group_input_mode
                )
                assert checkpoint["student_model_config"]["group_slots"] == (
                    MAX_GROUP_STAINS
                )
                assert checkpoint["student_model_config"]["affine_head_mode"] == "joint"
                assert checkpoint["model_config"]["frontend_mode"] == frontend_mode
                assert (
                    checkpoint["model_config"]["group_input_mode"] == group_input_mode
                )
                assert checkpoint["model_config"]["affine_head_mode"] == "joint"

            loaded_preprocess, loaded_config, loaded_state = load_student_checkpoint(
                str(student_path)
            )
            assert loaded_preprocess == preprocess
            assert loaded_config == config
            deployable = CorrelationVolumeAffineRegistrationModel(**loaded_config).to(
                device
            )
            deployable.load_state_dict(loaded_state, strict=True)
            deployable.eval()
            with torch.no_grad():
                params = deployable(
                    fixed_mineral=fixed,
                    moving_group=moving,
                    group=group,
                )
            assert params.shape == (1, 5) and torch.isfinite(params).all()

        raw_full_path = saved[("raw", "overlay")][0]
        incompatible_config = build_config("hybrid", "overlay")
        incompatible = TeacherStudentAffineRegistrationModel(
            student_config=incompatible_config, use_teacher_branch=False
        ).to(device)
        try:
            load_initial_weights(
                incompatible,
                str(raw_full_path),
                device,
                expected_student_model_config=incompatible_config,
                expected_preprocess_config=preprocess,
                expected_use_teacher_branch=False,
            )
        except ValueError as error:
            assert "frontend_mode" in str(error)
        else:
            raise AssertionError("Resume accepted a mismatched frontend_mode")

        (
            _,
            _,
            structural_full,
            structural_student,
        ) = saved[("structural", "overlay")]
        legacy_student = dict(structural_student)
        legacy_student["student_model_config"] = dict(
            structural_student["student_model_config"]
        )
        legacy_student["model_config"] = dict(structural_student["model_config"])
        legacy_full = dict(structural_full)
        legacy_full["student_model_config"] = dict(
            structural_full["student_model_config"]
        )
        legacy_full["model_config"] = dict(structural_full["model_config"])
        for checkpoint in (legacy_student, legacy_full):
            for config_key in ("student_model_config", "model_config"):
                for key in (
                    "frontend_mode",
                    "group_input_mode",
                    "group_slots",
                    "affine_head_mode",
                ):
                    checkpoint[config_key].pop(key, None)
        nested = legacy_full["model_config"].get("student_config")
        if nested is not None:
            legacy_full["model_config"]["student_config"] = dict(nested)
            for key in (
                "frontend_mode",
                "group_input_mode",
                "group_slots",
                "affine_head_mode",
            ):
                legacy_full["model_config"]["student_config"].pop(key, None)

        legacy_student_path = root / "legacy_student.pt"
        legacy_full_path = root / "legacy_full.pt"
        torch.save(legacy_student, legacy_student_path)
        torch.save(legacy_full, legacy_full_path)
        _, legacy_config, legacy_state = load_student_checkpoint(
            str(legacy_student_path)
        )
        assert legacy_config["frontend_mode"] == "structural"
        assert legacy_config["group_input_mode"] == "overlay"
        assert legacy_config["group_slots"] == MAX_GROUP_STAINS
        assert legacy_config["affine_head_mode"] == "joint"
        legacy_deployable = CorrelationVolumeAffineRegistrationModel(
            **legacy_config
        ).to(device)
        legacy_deployable.load_state_dict(legacy_state, strict=True)

        explicit_structural_config = build_config("structural", "overlay")
        restored = TeacherStudentAffineRegistrationModel(
            student_config=explicit_structural_config,
            use_teacher_branch=False,
        ).to(device)
        load_initial_weights(
            restored,
            str(legacy_full_path),
            device,
            expected_student_model_config=explicit_structural_config,
            expected_preprocess_config=preprocess,
            expected_use_teacher_branch=False,
        )


def assert_affine_head_checkpoint_contract(
    device: torch.device, height: int, width: int
) -> None:
    preprocess = {
        "input_contract_version": "fixed_mineral_moving_group_raw_v1",
        "input_value_range": [0.0, 1.0],
        "height": height,
        "width": width,
        "image_mode": "rgb",
        "sfo_mode": "rgb",
        "crop_mode": "full",
        "crop_margin": 4,
        "include_group1": False,
    }
    fixed, moving, _, group = _frontend_inputs(device, height, width)
    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory)
        full_paths = {}
        for mode in AFFINE_HEAD_MODES:
            config = build_config(affine_head_mode=mode)
            model = TeacherStudentAffineRegistrationModel(
                student_config=config, use_teacher_branch=False
            ).to(device)
            mode_root = root / mode
            mode_root.mkdir()
            save_checkpoint_pair(
                output_dir=str(mode_root),
                checkpoint_name="best_model.pt",
                full_payload={
                    "architecture": ARCHITECTURE,
                    "epoch": 1,
                    "metrics": {"val_student_total": 0.5},
                    "preprocess_config": preprocess,
                },
                model=model,
                student_model_config=config,
                preprocess_config=preprocess,
            )
            full_path = mode_root / "best_model.pt"
            student_path = mode_root / "best_model_student.pt"
            full_paths[mode] = full_path
            full_checkpoint = torch.load(
                full_path, map_location="cpu", weights_only=False
            )
            student_checkpoint = torch.load(
                student_path, map_location="cpu", weights_only=False
            )
            for checkpoint in (full_checkpoint, student_checkpoint):
                assert checkpoint["student_model_config"]["affine_head_mode"] == mode
                assert checkpoint["model_config"]["affine_head_mode"] == mode

            loaded_preprocess, loaded_config, loaded_state = load_student_checkpoint(
                str(student_path)
            )
            assert loaded_preprocess == preprocess
            assert loaded_config == config
            deployable = CorrelationVolumeAffineRegistrationModel(**loaded_config).to(
                device
            )
            deployable.load_state_dict(loaded_state, strict=True)
            deployable.eval()
            with torch.no_grad():
                params = deployable(fixed, moving, group)
            assert params.shape == (1, 5) and torch.isfinite(params).all()
            _assert_affine_bounds(params)

        incompatible_config = build_config(affine_head_mode="separated")
        incompatible = TeacherStudentAffineRegistrationModel(
            student_config=incompatible_config, use_teacher_branch=False
        ).to(device)
        try:
            load_initial_weights(
                incompatible,
                str(full_paths["joint"]),
                device,
                expected_student_model_config=incompatible_config,
                expected_preprocess_config=preprocess,
                expected_use_teacher_branch=False,
            )
        except ValueError as error:
            assert "affine_head_mode" in str(error)
        else:
            raise AssertionError("Resume accepted a mismatched affine_head_mode")


def assert_affine_error_metric_contract(device: torch.device) -> None:
    assert AFFINE_ERROR_METRIC_NAMES == (
        "tx_mae_px",
        "ty_mae_px",
        "theta_mae_deg",
        "sx_mae",
        "sy_mae",
        "control_point_error_px",
    )
    height, width = 80, 100
    target = torch.tensor(((0.0, 0.0, 0.0, 1.0, 1.0),), device=device)
    predicted = torch.tensor(
        ((0.20, -0.10, math.radians(10.0), 1.10, 0.80),), device=device
    )
    metrics = affine_error_metrics(predicted, target, height, width)
    assert tuple(metrics) == AFFINE_ERROR_METRIC_NAMES
    expected_scalars = {
        "tx_mae_px": 10.0,
        "ty_mae_px": 4.0,
        "theta_mae_deg": 10.0,
        "sx_mae": 0.10,
        "sy_mae": 0.20,
    }
    for name, expected in expected_scalars.items():
        torch.testing.assert_close(
            metrics[name],
            metrics[name].new_tensor(expected),
            atol=2e-5,
            rtol=2e-5,
        )

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
    pixel_scale = predicted.new_tensor(((width - 1) / 2.0, (height - 1) / 2.0))
    expected_control_point_error = torch.linalg.vector_norm(
        (predicted_points - target_points) * pixel_scale.view(1, 1, 2), dim=2
    ).mean()
    torch.testing.assert_close(
        metrics["control_point_error_px"], expected_control_point_error
    )

    wrapped_target = target.clone()
    wrapped_predicted = target.clone()
    wrapped_target[:, 2] = math.radians(179.0)
    wrapped_predicted[:, 2] = math.radians(-179.0)
    wrapped = affine_error_metrics(wrapped_predicted, wrapped_target, height, width)
    torch.testing.assert_close(
        wrapped["theta_mae_deg"],
        wrapped["theta_mae_deg"].new_tensor(2.0),
        atol=2e-5,
        rtol=2e-5,
    )
    zero_metrics = affine_error_metrics(target, target, height, width)
    assert all(float(value) == 0.0 for value in zero_metrics.values())


def assert_full_source_supervision_contract(
    device: torch.device, height: int, width: int
) -> None:
    """Lock one-pass synthetic correction and mixed Stage-1/Stage-2 routing."""
    target = torch.zeros(
        (2, MAX_GROUP_STAINS, 3, height, width), dtype=torch.float32, device=device
    )
    # The first signal touches the right edge, so an intermediate synthetic
    # canvas loses content that a one-pass warp of the original can retain.
    target[0, 0, :, height // 4 : 3 * height // 4, width - 7 : width - 1] = 1.0
    target[0, 1, 0, 5 : height // 2, 4 : width // 3] = 0.7
    target[1, 0, :, 5 : height - 8, 7 : width // 2] = 0.8
    target[1, 1, 1, height // 2 : height - 4, width // 2 : width - 5] = 0.6

    params_true = target.new_tensor(
        (
            (0.34, -0.08, math.radians(13.0), 0.88, 1.12),
            (0.0, 0.0, 0.0, 1.0, 1.0),
        )
    )
    synthesis = invert_affine_matrix(affine_parameters_to_matrix(params_true))
    moving = target.clone()
    moving[:1] = warp_group_with_matrix(target[:1], synthesis[:1])
    moving[1:] = torch.roll(target[1:], shifts=(3, -4), dims=(-2, -1))

    predicted = target.new_tensor(
        (
            (-0.11, 0.09, math.radians(-7.0), 1.08, 0.94),
            (0.07, -0.05, math.radians(5.0), 0.97, 1.03),
        )
    )
    predicted_matrix = affine_parameters_to_matrix(predicted)
    composed = synthetic_full_correction_matrices(predicted, params_true)

    bottom = target.new_tensor((0.0, 0.0, 1.0)).view(1, 1, 3).expand(2, -1, -1)
    synthesis_h = torch.cat((synthesis, bottom), dim=1)
    predicted_h = torch.cat((predicted_matrix, bottom), dim=1)
    expected = torch.bmm(synthesis_h, predicted_h)[:, :2]
    wrong_order = torch.bmm(predicted_h, synthesis_h)[:, :2]
    torch.testing.assert_close(composed, expected, atol=2e-6, rtol=2e-6)
    assert float((composed[0] - wrong_order[0]).abs().amax()) > 1e-3

    # Perfect moving-to-target parameters cancel A_syn exactly. The one-pass
    # result preserves the original, while correcting the clipped intermediate
    # cannot recreate signal that was already outside its canvas.
    perfect_matrix = synthetic_full_correction_matrices(
        params_true[:1], params_true[:1]
    )
    identity = target.new_tensor(((1.0, 0.0, 0.0), (0.0, 1.0, 0.0))).unsqueeze(0)
    torch.testing.assert_close(perfect_matrix, identity, atol=2e-6, rtol=2e-6)
    corrected_full = warp_group_with_matrix(target[:1], perfect_matrix)
    corrected_intermediate = warp_group(moving[:1], params_true[:1])
    full_error = F.mse_loss(corrected_full, target[:1])
    intermediate_error = F.mse_loss(corrected_intermediate, target[:1])
    assert float(full_error) < 1e-10
    assert float(intermediate_error) > float(full_error) + 1e-5

    perfect_via_router = warp_group_for_supervision(
        moving[:1],
        target[:1],
        params_true[:1],
        params_true[:1],
        torch.ones((1,), dtype=torch.bool, device=device),
    )
    torch.testing.assert_close(perfect_via_router, target[:1], atol=2e-5, rtol=2e-5)

    differentiable_prediction = predicted[:1].clone().detach().requires_grad_(True)
    differentiable_warp = warp_group_for_supervision(
        moving[:1],
        target[:1],
        differentiable_prediction,
        params_true[:1],
        torch.ones((1,), dtype=torch.bool, device=device),
    )
    differentiable_loss = F.mse_loss(differentiable_warp, target[:1])
    differentiable_loss.backward()
    assert differentiable_prediction.grad is not None
    assert torch.isfinite(differentiable_prediction.grad).all()
    assert float(differentiable_prediction.grad.abs().sum()) > 0.0

    mixed_has_params = torch.tensor((True, False), dtype=torch.bool, device=device)
    mixed_params_true = params_true.clone()
    mixed_params_true[1] = float("nan")
    sources, matrices = supervision_source_and_matrices(
        moving, target, predicted, mixed_params_true, mixed_has_params
    )
    assert torch.equal(sources[0], target[0])
    assert torch.equal(sources[1], moving[1])
    torch.testing.assert_close(matrices[0], composed[0])
    torch.testing.assert_close(matrices[1], predicted_matrix[1])
    mixed_warp = warp_group_for_supervision(
        moving, target, predicted, mixed_params_true, mixed_has_params
    )
    expected_mixed_warp = torch.cat(
        (
            warp_group_with_matrix(target[:1], composed[:1]),
            warp_group(moving[1:], predicted[1:]),
        ),
        dim=0,
    )
    torch.testing.assert_close(mixed_warp, expected_mixed_warp, atol=0.0, rtol=0.0)

    # Real Stage 2 has no synthesis matrix and remains the original direct warp.
    real_only = torch.zeros((2,), dtype=torch.bool, device=device)
    real_params_true = torch.full_like(params_true, float("nan"))
    stage2_warp = warp_group_for_supervision(
        moving, target, predicted, real_params_true, real_only
    )
    direct_warp = warp_group(moving, predicted)
    torch.testing.assert_close(stage2_warp, direct_warp, atol=0.0, rtol=0.0)

    valid = torch.tensor(
        ((True, True, False), (True, True, False)), dtype=torch.bool, device=device
    )
    overlap = build_group_valid_overlap(
        moving,
        target,
        valid,
        predicted,
        params_true=mixed_params_true,
        has_params=mixed_has_params,
    )
    synthetic_overlap = build_group_valid_overlap(
        moving[:1],
        target[:1],
        valid[:1],
        predicted[:1],
        params_true=params_true[:1],
        has_params=mixed_has_params[:1],
    )
    real_overlap = build_group_valid_overlap(
        moving[1:],
        target[1:],
        valid[1:],
        predicted[1:],
        params_true=mixed_params_true[1:],
        has_params=mixed_has_params[1:],
    )
    assert not overlap.requires_grad
    assert torch.equal(overlap, torch.cat((synthetic_overlap, real_overlap), dim=0))
    assert not bool(overlap[:, 2].any())
    assert bool(overlap[:, :2].any())


def assert_affine_inverse_diagnostic(device: torch.device) -> None:
    """Lock the synthetic moving-to-target matrix and inverse conventions."""
    height = width = 65
    yy, xx = torch.meshgrid(
        torch.linspace(-1.0, 1.0, height, device=device),
        torch.linspace(-1.0, 1.0, width, device=device),
        indexing="ij",
    )

    def gaussian(center_x: float, center_y: float, sigma: float) -> torch.Tensor:
        squared_distance = (xx - center_x).square() + (yy - center_y).square()
        return torch.exp(-squared_distance / (2.0 * sigma * sigma))

    target_group = torch.zeros((1, MAX_GROUP_STAINS, 3, height, width), device=device)
    target_group[0, 0, 0] = gaussian(-0.38, -0.22, 0.08)
    target_group[0, 0, 1] = 0.8 * gaussian(0.27, -0.08, 0.11)
    target_group[0, 0, 2] = 0.6 * gaussian(-0.05, 0.41, 0.07)
    target_group[0, 1, 0] = 0.9 * gaussian(0.36, 0.32, 0.09)
    target_group[0, 1, 1] = 0.7 * gaussian(-0.24, 0.15, 0.06)
    target_group[0, 1, 2] = 0.5 * gaussian(0.12, -0.37, 0.10)
    valid_group = torch.tensor(((True, True, False),), dtype=torch.bool, device=device)

    params_true = target_group.new_tensor(((0.16, -0.12, 0.18, 0.92, 1.08),))
    registration_matrix = affine_parameters_to_matrix(params_true)
    synthesis_matrix = invert_affine_matrix(registration_matrix)
    bottom_row = registration_matrix.new_tensor((0.0, 0.0, 1.0))
    bottom_row = bottom_row.view(1, 1, 3)
    registration_h = torch.cat((registration_matrix, bottom_row), dim=1)
    synthesis_h = torch.cat((synthesis_matrix, bottom_row), dim=1)
    identity_h = torch.eye(3, device=device, dtype=target_group.dtype).unsqueeze(0)
    torch.testing.assert_close(
        registration_h @ synthesis_h, identity_h, atol=2e-6, rtol=2e-6
    )
    torch.testing.assert_close(
        synthesis_h @ registration_h, identity_h, atol=2e-6, rtol=2e-6
    )

    moving_group = warp_group_with_matrix(target_group, synthesis_matrix)
    correctly_registered = warp_group_with_matrix(moving_group, registration_matrix)
    wrong_inverse_warp = warp_group_with_matrix(moving_group, synthesis_matrix)
    shared_support = shared_signal_union_support(
        (target_group, moving_group, correctly_registered, wrong_inverse_warp),
        valid_group,
    )
    assert shared_support.shape == (1, MAX_GROUP_STAINS, 1, height, width)
    assert not bool(shared_support[:, 2].any())
    unwarped_error = signal_union_mae(
        moving_group,
        target_group,
        valid_group,
        shared_support=shared_support,
    )
    correct_error = signal_union_mae(
        correctly_registered,
        target_group,
        valid_group,
        shared_support=shared_support,
    )
    wrong_error = signal_union_mae(
        wrong_inverse_warp,
        target_group,
        valid_group,
        shared_support=shared_support,
    )
    assert float(correct_error) < 0.1 * float(unwarped_error)
    assert float(correct_error) < 0.1 * float(wrong_error)

    same_direction_cp = affine_control_point_loss(params_true, params_true)
    assert float(same_direction_cp) == 0.0
    control_points = params_true.new_tensor(
        (
            (-1.0, -1.0, 1.0),
            (1.0, -1.0, 1.0),
            (-1.0, 1.0, 1.0),
            (1.0, 1.0, 1.0),
            (0.0, 0.0, 1.0),
        )
    )
    registration_points = torch.einsum(
        "bij,pj->bpi", registration_matrix, control_points
    )
    inverse_points = torch.einsum("bij,pj->bpi", synthesis_matrix, control_points)
    inverse_direction_cp = F.smooth_l1_loss(
        registration_points, inverse_points, beta=0.02
    )
    assert float(inverse_direction_cp) > 0.05


def assert_audit_metric_helpers() -> None:
    """Exercise visible-color metrics and direction-status edge cases."""
    near_zero_hue = torch.tensor((0.001, 1.0, 1.0)).view(1, 1, 3, 1, 1)
    near_one_hue = torch.tensor((0.999, 1.0, 1.0)).view(1, 1, 3, 1, 1)
    near_zero_rgb = _metric_group_stack(near_zero_hue, group_id=5, sfo_mode="hsv")
    near_one_rgb = _metric_group_stack(near_one_hue, group_id=5, sfo_mode="hsv")
    assert float((near_zero_rgb - near_one_rgb).abs().amax()) < 0.01

    black_hsv = torch.zeros((1, 2, 3, 2, 2))
    black_hsv[:, :, 0] = 0.73
    black_hsv[:, :, 1] = 1.0
    black_rgb = _metric_group_stack(black_hsv, group_id=5, sfo_mode="hsv")
    assert torch.equal(black_rgb, torch.zeros_like(black_rgb))

    assert (
        _direction_status(torch.tensor(0.0), torch.tensor(0.1), torch.tensor(0.2))
        == "PASS"
    )
    assert (
        _direction_status(torch.tensor(0.1), torch.tensor(0.1), torch.tensor(0.2))
        == "INCONCLUSIVE"
    )
    assert (
        _direction_status(torch.tensor(0.2), torch.tensor(0.1), torch.tensor(0.3))
        == "REVIEW"
    )


def _small_residual_config() -> dict:
    config = build_config()
    config.update(
        {
            "encoder_arch": "residual",
            "encoder_channels": (8, 16, 24),
            "encoder_blocks_per_stage": (2,),
            "correlation_feature_width": 12,
        }
    )
    return config


def assert_residual_encoder_and_cli_contract(
    device: torch.device, height: int, width: int
) -> None:
    assert ENCODER_ARCHES == ("current", "residual")
    assert DEFAULT_RESIDUAL_ENCODER_CHANNELS == (64, 96, 128, 192, 256)
    assert DEFAULT_RESIDUAL_BLOCKS_PER_STAGE == (2, 2, 2, 2, 2)
    assert DEFAULT_CORRELATION_FEATURE_WIDTH == 96

    defaults = _parse_train_cli()
    assert defaults.encoder_arch == "current"
    assert defaults.encoder_channels is None
    assert defaults.encoder_blocks_per_stage is None
    assert defaults.correlation_feature_width is None
    train_module.resolve_encoder_args(defaults)
    assert defaults.encoder_channels == (24, 32, 48, 64, 96)
    assert defaults.encoder_blocks_per_stage is None
    assert defaults.correlation_feature_width == 96

    residual_defaults = _parse_train_cli("--encoder_arch", "residual")
    train_module.resolve_encoder_args(residual_defaults)
    assert residual_defaults.encoder_channels == DEFAULT_RESIDUAL_ENCODER_CHANNELS
    assert (
        residual_defaults.encoder_blocks_per_stage == DEFAULT_RESIDUAL_BLOCKS_PER_STAGE
    )
    assert residual_defaults.correlation_feature_width == 96

    custom = _parse_train_cli(
        "--encoder_arch",
        "residual",
        "--encoder_channels",
        "8",
        "16",
        "24",
        "--encoder_blocks_per_stage",
        "3",
        "--correlation_feature_width",
        "12",
    )
    train_module.resolve_encoder_args(custom)
    assert custom.encoder_channels == (8, 16, 24)
    assert custom.encoder_blocks_per_stage == (3, 3, 3)
    assert custom.encoder_depth == 3
    assert custom.correlation_feature_width == 12

    legacy = canonicalize_model_config(
        {"encoder_base_channels": 8, "encoder_depth": 3, "feature_width": 8}
    )
    assert legacy["encoder_arch"] == "current"
    assert legacy["encoder_channels"] == (8, 8, 16)
    assert legacy["encoder_blocks_per_stage"] is None
    assert legacy["correlation_feature_width"] == 8

    residual = TeacherStudentAffineRegistrationModel(
        student_config=_small_residual_config(), use_teacher_branch=True
    ).to(device)
    current = TeacherStudentAffineRegistrationModel(
        student_config=build_config(), use_teacher_branch=False
    ).to(device)
    assert residual.teacher is not None
    assert residual.student.encoder_channels == (8, 16, 24)
    assert residual.student.encoder_blocks_per_stage == (2, 2, 2)
    assert residual.student.correlation_feature_width == 12
    assert sum(parameter.numel() for parameter in residual.student.parameters()) > sum(
        parameter.numel() for parameter in current.student.parameters()
    )
    student_encoder = residual.student.encoder.shared_encoder
    teacher_encoder = residual.teacher.encoder.shared_encoder
    assert student_encoder is not teacher_encoder
    assert residual.student.state_dict().keys() == residual.teacher.state_dict().keys()
    assert all(
        isinstance(stage, ResidualEncoderStage) for stage in student_encoder.stages
    )
    for stage in student_encoder.stages:
        assert len(stage.blocks) == 2
        for block_index, block in enumerate(stage.blocks):
            assert isinstance(block, ResidualEncoderBlock)
            assert isinstance(block.norm1, torch.nn.GroupNorm)
            assert isinstance(block.norm2, torch.nn.GroupNorm)
            if block_index == 0:
                assert isinstance(block.projection, torch.nn.Conv2d)
                assert block.projection.kernel_size == (1, 1)
                assert block.projection.stride == (2, 2)

    encoded = student_encoder(
        torch.rand(2, residual.student.frontend_channels, height, width, device=device)
    )
    assert len(encoded) == 1
    assert encoded[0].shape == (
        2,
        build_config()["feature_width"],
        math.ceil(height / 8),
        math.ceil(width / 8),
    )
    projection = residual.student.encoder.correlation_projections[0]
    assert isinstance(projection, torch.nn.Conv2d)
    assert projection.weight.shape == (12, 8, 1, 1)


def assert_cost_volume_aux_contract(
    device: torch.device, height: int, width: int
) -> None:
    fixed, moving, target, group = _frontend_inputs(device, height, width)
    model = TeacherStudentAffineRegistrationModel(
        student_config=build_config(), use_teacher_branch=True
    ).to(device)
    model.eval()
    with torch.no_grad():
        params = model(fixed, moving, group)
        student_aux = model(fixed, moving, group, return_aux=True)
        teacher_aux = model.forward_teacher(
            fixed, target, moving, group, return_aux=True
        )
    assert isinstance(params, torch.Tensor) and params.shape == (1, 5)
    assert isinstance(student_aux, dict) and isinstance(teacher_aux, dict)
    assert torch.equal(params, student_aux["params"])
    required = {
        "volume",
        "probability",
        "expected_displacement",
        "confidence",
        "certainty",
        "displacements",
        "fixed_valid",
        "moving_valid",
        "feature_size",
    }
    for output in (student_aux, teacher_aux):
        assert output["params"].shape == (1, 5)
        assert len(output["levels"]) == len(build_config()["cost_volume_radii"])
        for level in output["levels"]:
            assert required <= set(level)
            volume = level["volume"]
            probability = level["probability"]
            expected = level["expected_displacement"]
            confidence = level["confidence"]
            certainty = level["certainty"]
            displacements = level["displacements"]
            fixed_valid = level["fixed_valid"]
            moving_valid = level["moving_valid"]
            assert volume.shape == probability.shape
            batch_size, candidates, level_height, level_width = volume.shape
            assert expected.shape == (batch_size, 2, level_height, level_width)
            assert (
                confidence.shape
                == certainty.shape
                == (batch_size, 1, level_height, level_width)
            )
            assert displacements.shape == (candidates, 2)
            assert (
                fixed_valid.shape
                == moving_valid.shape
                == (batch_size, 1, level_height, level_width)
            )
            assert level["feature_size"] == [level_height, level_width]
            assert all(
                torch.isfinite(value).all()
                for value in (volume, probability, expected, confidence, certainty)
            )
            candidate_valid = level["candidate_valid"]
            probability_sum = probability.sum(dim=1)
            has_candidate = candidate_valid.any(dim=1)
            assert torch.allclose(
                probability_sum[has_candidate],
                torch.ones_like(probability_sum[has_candidate]),
                atol=1e-6,
            )
            assert torch.equal(
                probability_sum[~has_candidate],
                torch.zeros_like(probability_sum[~has_candidate]),
            )


def _correspondence_level(
    batch_size: int, height: int, width: int, radius: int = 1
) -> tuple[dict[str, object], tuple[torch.Tensor, ...]]:
    displacements = LocalCorrelationVolume(
        radius=radius, temperature=0.07
    ).displacements.clone()
    candidates = displacements.shape[0]
    logits = torch.zeros(batch_size, candidates, height, width, requires_grad=True)
    probability = torch.softmax(logits, dim=1)
    expected = torch.zeros(batch_size, 2, height, width, requires_grad=True)
    confidence = torch.full((batch_size, 1, height, width), 0.7, requires_grad=True)
    validity = torch.ones(batch_size, 1, height, width, dtype=torch.bool)
    level: dict[str, object] = {
        "volume": torch.zeros_like(probability),
        "probability": probability,
        "expected_displacement": expected,
        "confidence": confidence,
        "certainty": torch.ones_like(confidence),
        "displacements": displacements,
        "fixed_valid": validity,
        "moving_valid": validity.clone(),
        "feature_size": [height, width],
    }
    return level, (logits, expected, confidence)


def assert_correspondence_geometry_and_empty_contract() -> None:
    height = width = 7
    center = (height // 2, width // 2)
    identity_level, _ = _correspondence_level(1, height, width)
    identity_params = torch.tensor(((0.0, 0.0, 0.0, 1.0, 1.0),))
    identity_targets = cost_volume_correspondence_targets(
        identity_level, identity_params, torch.tensor((True,)), sigma=0.75
    )
    assert torch.equal(
        identity_targets["true_displacement"][0, :, center[0], center[1]],
        torch.zeros(2),
    )
    assert bool(identity_targets["in_radius_mask"][0, 0, center[0], center[1]])
    identity_peak = int(
        identity_targets["target_probability"][0, :, center[0], center[1]].argmax()
    )
    assert torch.equal(
        identity_level["displacements"][identity_peak], torch.tensor((0.0, 0.0))
    )

    mixed_level, _ = _correspondence_level(2, height, width)
    plus_one_x = 2.0 / (width - 1)
    mixed_params = torch.tensor(
        (
            (plus_one_x, 0.0, 0.0, 1.0, 1.0),
            (float("nan"),) * 5,
        )
    )
    mixed_targets = cost_volume_correspondence_targets(
        mixed_level, mixed_params, torch.tensor((True, False)), sigma=0.75
    )
    assert torch.allclose(
        mixed_targets["true_displacement"][0, :, center[0], center[1]],
        torch.tensor((1.0, 0.0)),
        atol=1e-6,
    )
    assert not bool(mixed_targets["geometric_mask"][1].any())
    plus_one_peak = int(
        mixed_targets["target_probability"][0, :, center[0], center[1]].argmax()
    )
    assert torch.equal(
        mixed_level["displacements"][plus_one_peak], torch.tensor((1.0, 0.0))
    )

    outside_level, _ = _correspondence_level(1, height, width, radius=1)
    outside_params = torch.tensor(((4.0 / (width - 1), 0.0, 0.0, 1.0, 1.0),))
    outside_targets = cost_volume_correspondence_targets(
        outside_level, outside_params, torch.tensor((True,)), sigma=1.0
    )
    confidence_mask = outside_targets["confidence_mask"]
    assert bool(confidence_mask.any())
    assert not bool(outside_targets["in_radius_mask"].any())
    assert torch.equal(
        outside_targets["confidence_target"][confidence_mask],
        torch.zeros_like(outside_targets["confidence_target"][confidence_mask]),
    )
    outside_losses, outside_counts = cost_volume_correspondence_loss(
        [outside_level], outside_params, torch.tensor((True,)), sigma=1.0
    )
    assert outside_counts["displacement"] == 0
    assert outside_counts["distribution"] == 0
    assert outside_counts["confidence"] > 0
    assert all(torch.isfinite(value) for value in outside_losses.values())

    real_level, leaves = _correspondence_level(1, height, width)
    real_losses, real_counts = cost_volume_correspondence_loss(
        [real_level],
        torch.full((1, 5), float("nan")),
        torch.tensor((False,)),
        sigma=1.0,
    )
    assert real_counts["synthetic_samples"] == 0
    assert all(
        count == 0
        for count in (
            real_counts["displacement"],
            real_counts["distribution"],
            real_counts["confidence"],
        )
    )
    real_total = torch.stack(tuple(real_losses.values())).sum()
    assert torch.isfinite(real_total) and float(real_total) == 0.0
    real_total.backward()
    for leaf in leaves:
        assert leaf.grad is not None and torch.isfinite(leaf.grad).all()

    empty_losses, empty_counts = cost_volume_correspondence_loss(
        [],
        torch.full((1, 5), float("nan")),
        torch.tensor((False,)),
        sigma=1.0,
    )
    empty_total = torch.stack(tuple(empty_losses.values())).sum()
    assert empty_counts["levels"] == 0
    assert torch.isfinite(empty_total) and empty_total.requires_grad
    empty_total.backward()


def assert_correspondence_autocast_contract(device: torch.device) -> None:
    """Confidence supervision must remain valid inside CPU/CUDA autocast."""
    level, _ = _correspondence_level(1, 7, 7)
    moved_level: dict[str, object] = {}
    for name, value in level.items():
        if not torch.is_tensor(value):
            moved_level[name] = value
            continue
        moved = value.detach().to(device)
        if value.requires_grad:
            moved.requires_grad_(True)
        moved_level[name] = moved

    params_true = torch.tensor(((0.0, 0.0, 0.0, 1.0, 1.0),), device=device)
    has_params = torch.tensor((True,), dtype=torch.bool, device=device)
    autocast_dtype = torch.float16 if device.type == "cuda" else torch.bfloat16
    with torch.autocast(device_type=device.type, dtype=autocast_dtype):
        losses, counts = cost_volume_correspondence_loss(
            [moved_level], params_true, has_params, sigma=1.0
        )
        total = torch.stack(tuple(losses.values())).sum()
    assert counts["confidence"] > 0
    assert torch.isfinite(total)
    total.backward()
    confidence = moved_level["confidence"]
    assert isinstance(confidence, torch.Tensor)
    assert confidence.grad is not None and torch.isfinite(confidence.grad).all()


def _has_nonzero_finite_gradient(module: torch.nn.Module) -> bool:
    gradients = [
        parameter.grad
        for parameter in module.parameters()
        if parameter.grad is not None
    ]
    return (
        bool(gradients)
        and all(torch.isfinite(gradient).all() for gradient in gradients)
        and any(bool(torch.count_nonzero(gradient)) for gradient in gradients)
    )


def assert_student_teacher_correspondence_gradient_contract(
    device: torch.device, height: int, width: int
) -> None:
    fixed, moving, target, group = _frontend_inputs(device, height, width)
    model = TeacherStudentAffineRegistrationModel(
        student_config=build_config(), use_teacher_branch=True
    ).to(device)
    params_true = torch.tensor(((0.0, 0.0, 0.0, 1.0, 1.0),), device=device)
    has_params = torch.tensor((True,), dtype=torch.bool, device=device)

    model.zero_grad(set_to_none=True)
    student_aux = model(fixed, moving, group, return_aux=True)
    student_losses, student_counts = cost_volume_correspondence_loss(
        student_aux["levels"], params_true, has_params, sigma=1.0
    )
    assert student_counts["synthetic_samples"] == 1
    torch.stack(tuple(student_losses.values())).sum().backward()
    assert _has_nonzero_finite_gradient(model.student.encoder)

    model.zero_grad(set_to_none=True)
    teacher_aux = model.forward_teacher(fixed, target, moving, group, return_aux=True)
    teacher_losses, teacher_counts = cost_volume_correspondence_loss(
        teacher_aux["levels"], params_true, has_params, sigma=1.0
    )
    assert teacher_counts["synthetic_samples"] == 1
    torch.stack(tuple(teacher_losses.values())).sum().backward()
    assert model.teacher is not None
    assert _has_nonzero_finite_gradient(model.teacher.encoder)


def assert_training_correspondence_phase_contract(
    device: torch.device, height: int, width: int
) -> None:
    args = _teacher_loss_args(detach_teacher=True, warmup_epochs=0)
    args.corr_displacement_weight = 1.0
    args.corr_distribution_weight = 1.0
    args.corr_confidence_weight = 1.0
    args.corr_target_sigma = 1.0
    args.student_corr_weight = 1.0
    args.teacher_corr_weight = 0.5

    batch = _teacher_batch(device, height, width)
    batch["moving_group"] = batch["target_group"].clone()
    batch["params_true"] = batch["moving_group"].new_tensor(
        ((0.0, 0.0, 0.0, 1.0, 1.0),)
    )
    batch["has_params"] = torch.tensor((True,), dtype=torch.bool, device=device)

    model = TeacherStudentAffineRegistrationModel(
        student_config=build_config(), use_teacher_branch=True
    ).to(device)
    assert configure_training_phase(args, model, epoch=1) == "student"
    model.zero_grad(set_to_none=True)
    total, components, _ = compute_training_loss(args, model, batch, epoch=1)
    student_corr_names = {
        "student_corr_displacement",
        "student_corr_distribution",
        "student_corr_confidence",
        "student_corr_total",
    }
    teacher_corr_names = {
        "teacher_corr_displacement",
        "teacher_corr_distribution",
        "teacher_corr_confidence",
        "teacher_corr_total",
    }
    assert student_corr_names | teacher_corr_names <= set(components)
    assert torch.isfinite(total)
    total.backward()
    assert _gradient_sum(model.student) > 0.0
    assert model.teacher is not None and _gradient_sum(model.teacher) > 0.0

    warmup_args = _teacher_loss_args(detach_teacher=True, warmup_epochs=1)
    for name in (
        "corr_displacement_weight",
        "corr_distribution_weight",
        "corr_confidence_weight",
        "corr_target_sigma",
        "student_corr_weight",
        "teacher_corr_weight",
    ):
        setattr(warmup_args, name, getattr(args, name))
    warmup_model = TeacherStudentAffineRegistrationModel(
        student_config=build_config(), use_teacher_branch=True
    ).to(device)
    assert (
        configure_training_phase(warmup_args, warmup_model, epoch=1) == "teacher_warmup"
    )
    warmup_model.zero_grad(set_to_none=True)
    warmup_total, warmup_components, _ = compute_training_loss(
        warmup_args, warmup_model, batch, epoch=1
    )
    assert teacher_corr_names <= set(warmup_components)
    assert student_corr_names.isdisjoint(warmup_components)
    assert torch.isfinite(warmup_total)
    warmup_total.backward()
    assert _gradient_sum(warmup_model.student) == 0.0
    assert warmup_model.teacher is not None
    assert _gradient_sum(warmup_model.teacher) > 0.0

    real_batch = _teacher_batch(device, height, width)
    assert torch.isnan(real_batch["params_true"]).all()
    assert not bool(real_batch["has_params"].any())
    assert configure_training_phase(args, model, epoch=1) == "student"
    model.zero_grad(set_to_none=True)
    real_total, real_components, _ = compute_training_loss(
        args, model, real_batch, epoch=1
    )
    assert "student_charbonnier" in real_components
    assert not any("_corr_" in name for name in real_components)
    assert torch.isfinite(real_total)
    real_total.backward()
    assert _gradient_sum(model.student) > 0.0


def _checkpoint_preprocess(height: int, width: int) -> dict:
    return {
        "input_contract_version": "fixed_mineral_moving_group_raw_v1",
        "input_value_range": [0.0, 1.0],
        "height": height,
        "width": width,
        "image_mode": "rgb",
        "sfo_mode": "rgb",
        "crop_mode": "full",
        "crop_margin": 4,
        "include_group1": False,
    }


def assert_legacy_encoder_checkpoint_migration_contract(
    device: torch.device, height: int, width: int
) -> None:
    current_config = build_config()
    preprocess = _checkpoint_preprocess(height, width)
    source = TeacherStudentAffineRegistrationModel(
        student_config=current_config, use_teacher_branch=False
    ).to(device)
    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory)
        save_checkpoint_pair(
            output_dir=str(root),
            checkpoint_name="current.pt",
            full_payload={
                "architecture": ARCHITECTURE,
                "epoch": 1,
                "metrics": {"val_student_total": 0.5},
                "preprocess_config": preprocess,
            },
            model=source,
            student_model_config=current_config,
            preprocess_config=preprocess,
        )
        checkpoint = torch.load(
            root / "current.pt", map_location="cpu", weights_only=False
        )
        legacy = dict(checkpoint)
        legacy["student_model_config"] = dict(checkpoint["student_model_config"])
        legacy["model_config"] = dict(checkpoint["model_config"])
        encoder_keys = (
            "encoder_arch",
            "encoder_channels",
            "encoder_blocks_per_stage",
            "correlation_feature_width",
        )
        for key in encoder_keys:
            legacy["student_model_config"].pop(key, None)
            legacy["model_config"].pop(key, None)
        nested = legacy["model_config"].get("student_config")
        if nested is not None:
            legacy["model_config"]["student_config"] = dict(nested)
            for key in encoder_keys:
                legacy["model_config"]["student_config"].pop(key, None)
        legacy_path = root / "legacy_current.pt"
        torch.save(legacy, legacy_path)

        strict_current = TeacherStudentAffineRegistrationModel(
            student_config=current_config, use_teacher_branch=False
        ).to(device)
        load_initial_weights(
            strict_current,
            str(legacy_path),
            device,
            expected_student_model_config=current_config,
            expected_preprocess_config=preprocess,
            expected_use_teacher_branch=False,
        )
        assert all(
            torch.equal(value, strict_current.student.state_dict()[name])
            for name, value in source.student.state_dict().items()
        )

        residual_config = _small_residual_config()
        migrated = TeacherStudentAffineRegistrationModel(
            student_config=residual_config, use_teacher_branch=False
        ).to(device)
        with mock.patch("builtins.print") as printed:
            load_initial_weights(
                migrated,
                str(legacy_path),
                device,
                expected_student_model_config=residual_config,
                expected_preprocess_config=preprocess,
                expected_use_teacher_branch=False,
            )
        report = "\n".join(
            " ".join(str(argument) for argument in call.args)
            for call in printed.call_args_list
        )
        assert "current -> residual" in report
        assert "newly initialized destination keys" in report
        assert "shape-mismatched checkpoint keys" in report
        assert torch.equal(
            migrated.student.group_embedding.weight,
            source.student.group_embedding.weight,
        )
        residual_weight = (
            migrated.student.encoder.shared_encoder.stages[0].blocks[0].conv1.weight
        )
        assert torch.isfinite(residual_weight).all()


def main(args: argparse.Namespace) -> None:
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available")
    if args.height < 32 or args.width < 32:
        raise ValueError("Smoke-test dimensions must be at least 32")

    torch.manual_seed(7)
    assert_affine_inverse_diagnostic(device)
    assert_affine_error_metric_contract(device)
    assert_full_source_supervision_contract(device, args.height, args.width)
    assert_audit_metric_helpers()
    assert_frontend_cli_contract()
    assert_residual_encoder_and_cli_contract(device, args.height, args.width)
    assert_cost_volume_aux_contract(device, args.height, args.width)
    assert_correspondence_geometry_and_empty_contract()
    assert_correspondence_autocast_contract(device)
    assert_student_teacher_correspondence_gradient_contract(
        device, args.height, args.width
    )
    assert_training_correspondence_phase_contract(device, args.height, args.width)
    assert_legacy_encoder_checkpoint_migration_contract(device, args.height, args.width)
    assert_teacher_phase_cli_and_validation_contract()
    assert_affine_head_geometry_contract(device)
    assert_coarse_affine_and_composition_contract(device)
    assert_structural_frontend_compatibility(device, args.height, args.width)
    assert_frontend_forward_matrix(device, args.height, args.width)
    assert_teacher_mineral_fusion_contract(device, args.height, args.width)
    assert_raw_and_hybrid_frontend_contract(device, args.height, args.width)
    student = CorrelationVolumeAffineRegistrationModel(**build_config()).to(device)
    assert_no_enhancement_contract()
    assert_dataset_contract(student, device, args.height, args.width)
    assert_local_correlation_contract(device)
    assert_target_is_supervision_only(student, device, args.height, args.width)
    assert_model_contract(student, device, args.height, args.width)
    assert_teacher_student_contract(device, args.height, args.width)
    assert_distillation_contract(device, args.height, args.width)
    assert_teacher_only_warmup_contract(device, args.height, args.width)
    assert_real_and_mixed_teacher_warmup_contract(device, args.height, args.width)
    assert_frozen_teacher_contract(device, args.height, args.width)
    assert_frontend_adapter_training_contract(device, args.height, args.width)
    assert_checkpoint_and_overlay_contract(device, args.height, args.width)
    assert_frontend_checkpoint_contract(device, args.height, args.width)
    assert_affine_head_checkpoint_contract(device, args.height, args.width)
    print("Teacher/student structural correlation-volume smoke test passed")


if __name__ == "__main__":
    main(parse_args())
