#!/usr/bin/env python3
"""Backward-compatible entrypoint for phase-close report generation."""

from scripts.refactor.phase_close import main

if __name__ == "__main__":
    raise SystemExit(main())

