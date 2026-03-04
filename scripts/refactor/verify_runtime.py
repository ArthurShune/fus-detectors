"""Runtime/execution helpers for refactor verification gates."""

from __future__ import annotations

import shutil
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence


@dataclass(frozen=True)
class Step:
    key: str
    description: str
    cmd: list[str]
    requires: tuple[Path, ...] = ()
    optional: bool = False


def have_conda() -> bool:
    return shutil.which("conda") is not None


def build_py_cmd(
    *,
    conda_env: str,
    use_conda: bool,
    script: str,
    args: Sequence[str],
) -> list[str]:
    if use_conda and have_conda():
        return ["conda", "run", "-n", conda_env, "python", script, *args]
    return [sys.executable, script, *args]


def format_cmd(cmd: Sequence[str]) -> str:
    return " ".join(subprocess.list2cmdline([part]) if " " in part else part for part in cmd)


def execute_steps(
    *,
    steps: Sequence[Step],
    repo: Path,
    env: dict[str, str],
) -> tuple[int, set[str], set[str]]:
    completed_steps: set[str] = set()
    skipped_steps: set[str] = set()

    for step in steps:
        missing = [p for p in step.requires if not p.exists()]
        if missing:
            missing_paths = ", ".join(str(p) for p in missing)
            msg = (
                f"[verify-refactor] missing required path(s) for {step.key}: "
                f"{missing_paths}"
            )
            if step.optional:
                print(msg)
                print(f"[verify-refactor] skipping optional step: {step.key}")
                skipped_steps.add(step.key)
                continue
            print(msg, file=sys.stderr)
            return 3, completed_steps, skipped_steps

        print(f"[verify-refactor] running: {step.key}")
        proc = subprocess.run(step.cmd, cwd=str(repo), env=env)
        if proc.returncode != 0:
            print(
                f"[verify-refactor] step failed: {step.key} (code={proc.returncode})",
                file=sys.stderr,
            )
            return proc.returncode, completed_steps, skipped_steps
        completed_steps.add(step.key)

    return 0, completed_steps, skipped_steps

