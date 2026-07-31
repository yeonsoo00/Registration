"""Compatibility alias for :mod:`infer`.

The requested entrypoint is ``infer.py``; this alias also supports the naming
used by the parent TeacherStudent experiment.
"""

from infer import main, parse_args


if __name__ == "__main__":
    main(parse_args())
