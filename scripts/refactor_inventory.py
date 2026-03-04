#!/usr/bin/env python3
"""Backward-compatible entrypoint for refactor inventory generation."""

from scripts.refactor.cli_compat import run_module_main

if __name__ == "__main__":
    raise SystemExit(run_module_main("scripts.refactor.inventory"))
