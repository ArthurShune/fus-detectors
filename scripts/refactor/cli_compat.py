"""Shared compatibility helpers for legacy script entrypoints."""

from __future__ import annotations

import importlib


def run_module_main(module_path: str) -> int:
    module = importlib.import_module(module_path)
    main = getattr(module, "main", None)
    if main is None:
        raise AttributeError(f"{module_path} does not define main()")
    result = main()
    if result is None:
        return 0
    if isinstance(result, int):
        return result
    raise TypeError(f"{module_path}.main() must return int|None, got {type(result).__name__}")

