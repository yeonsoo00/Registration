"""Load the canonical TeacherStudent implementation without copying it.

The student-only experiment intentionally shares dataset, model, loss, warp,
checkpoint, and visualization code with its teacher/student baseline.  Keeping
one implementation makes the comparison differ only in whether the training
teacher exists.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from types import ModuleType


CORE_DIRECTORY = Path(__file__).resolve().parent.parent
_MODULE_CACHE: dict[str, ModuleType] = {}


def load_core_module(module_stem: str) -> ModuleType:
    """Load one module from the parent TeacherStudent experiment."""
    if module_stem in _MODULE_CACHE:
        return _MODULE_CACHE[module_stem]

    module_path = CORE_DIRECTORY / f"{module_stem}.py"
    if not module_path.is_file():
        raise FileNotFoundError(f"Missing shared core module: {module_path}")

    core_path = str(CORE_DIRECTORY)
    if core_path not in sys.path:
        # Parent modules use local imports (for example, ``from models import``).
        sys.path.insert(0, core_path)

    qualified_name = f"_student_only_shared_{module_stem}"
    spec = importlib.util.spec_from_file_location(qualified_name, module_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load shared core module from {module_path}")

    module = importlib.util.module_from_spec(spec)
    sys.modules[qualified_name] = module
    try:
        spec.loader.exec_module(module)
    except Exception:
        sys.modules.pop(qualified_name, None)
        raise
    _MODULE_CACHE[module_stem] = module
    return module
