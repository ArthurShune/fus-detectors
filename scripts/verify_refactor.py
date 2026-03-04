#!/usr/bin/env python3
"""Backward-compatible entrypoint for refactor verification gates."""

from scripts.refactor.verify_gate import main

if __name__ == "__main__":
    raise SystemExit(main())

