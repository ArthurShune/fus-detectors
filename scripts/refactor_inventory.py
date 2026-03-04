#!/usr/bin/env python3
"""Backward-compatible entrypoint for refactor inventory generation."""

from scripts.refactor.inventory import main

if __name__ == "__main__":
    raise SystemExit(main())

