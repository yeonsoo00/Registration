"""Run inference for a StudentOnly checkpoint.

The shared inference implementation already constructs only the deployable
student and never loads or requires a teacher.
"""

from __future__ import annotations

import sys
from argparse import Namespace
from collections.abc import Sequence

from _core import load_core_module


_CORE_INFERENCE = load_core_module("inference")


def parse_args(arguments: Sequence[str] | None = None) -> Namespace:
    if arguments is None:
        return _CORE_INFERENCE.parse_args()

    original_argv = sys.argv
    try:
        sys.argv = [original_argv[0], *arguments]
        return _CORE_INFERENCE.parse_args()
    finally:
        sys.argv = original_argv


def main(args: Namespace) -> None:
    _CORE_INFERENCE.main(args)


if __name__ == "__main__":
    main(parse_args())
