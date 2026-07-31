"""Train the exact TeacherStudent student architecture without a teacher.

All implementation details are delegated to the canonical parent trainer so
the comparison uses the same data, model, losses, warps, validation, and
checkpoint format.  This entrypoint only enforces the student-only regime.
"""

from __future__ import annotations

import sys
from argparse import Namespace
from collections.abc import Sequence

from _core import load_core_module


_CORE_TRAIN = load_core_module("train")

_TEACHER_ONLY_OPTIONS = (
    "--use_teacher_branch",
    "--teacher_distill_weight",
    "--detach_teacher",
    "--teacher_warmup_epochs",
    "--freeze_teacher",
    "--teacher_corr_weight",
)


def _provided_option(arguments: Sequence[str], option: str) -> bool:
    return any(token == option or token.startswith(f"{option}=") for token in arguments)


def _reject_teacher_options(arguments: Sequence[str]) -> None:
    provided = [
        option
        for option in _TEACHER_ONLY_OPTIONS
        if _provided_option(arguments, option)
    ]
    if not provided:
        return
    formatted = ", ".join(provided)
    raise SystemExit(
        "StudentOnly does not accept active teacher settings: "
        f"{formatted}. Remove them; this entrypoint always constructs and "
        "optimizes only the student."
    )


def _parse_core_args(arguments: Sequence[str] | None) -> Namespace:
    if arguments is None:
        return _CORE_TRAIN.parse_args()

    original_argv = sys.argv
    try:
        sys.argv = [original_argv[0], *arguments]
        return _CORE_TRAIN.parse_args()
    finally:
        sys.argv = original_argv


def enforce_student_only(args: Namespace) -> Namespace:
    """Normalize teacher settings before the shared trainer sees the config."""
    args.use_teacher_branch = False
    args.teacher_distill_weight = 0.0
    args.detach_teacher = False
    args.teacher_warmup_epochs = 0
    args.freeze_teacher = False
    args.teacher_corr_weight = 0.0
    # Saved in train_config and sent to W&B for unambiguous experiment metadata.
    args.training_regime = "student_only"
    return args


def parse_args(arguments: Sequence[str] | None = None) -> Namespace:
    command_arguments = list(sys.argv[1:] if arguments is None else arguments)
    _reject_teacher_options(command_arguments)
    return enforce_student_only(_parse_core_args(arguments))


def main(args: Namespace) -> None:
    args = enforce_student_only(args)
    print(
        "Training regime: student_only "
        "(teacher branch absent; no warmup or distillation)"
    )
    _CORE_TRAIN.main(args)


if __name__ == "__main__":
    main(parse_args())
