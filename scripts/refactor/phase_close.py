#!/usr/bin/env python3
"""Create a standard phase-close report scaffold.

Usage:
  PYTHONPATH=. python scripts/phase_close_report.py --phase 1
"""

from __future__ import annotations

import argparse
import subprocess
from datetime import datetime, timezone
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
DEFAULT_DIR = REPO / "docs" / "refactor" / "phase_reports"


def _git(cmd: list[str]) -> str:
    proc = subprocess.run(
        ["git", *cmd],
        cwd=str(REPO),
        check=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    return proc.stdout.strip()


def _git_info() -> dict[str, str]:
    info: dict[str, str] = {}
    info["commit"] = _git(["rev-parse", "HEAD"])
    info["commit_short"] = _git(["rev-parse", "--short", "HEAD"])
    info["branch"] = _git(["rev-parse", "--abbrev-ref", "HEAD"])
    info["status"] = "dirty" if _git(["status", "--porcelain"]) else "clean"
    return info


def _render(*, phase: str, verification_mode: str, git_info: dict[str, str]) -> str:
    ts = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%SZ")
    verify_cmd = (
        "PYTHONPATH=. python scripts/verify_refactor.py "
        f"--mode {verification_mode} --execute"
    )
    lines: list[str] = []
    lines.append(f"# Phase {phase} Close Report")
    lines.append("")
    lines.append(f"- Generated: {ts} (UTC)")
    lines.append(f"- Branch: `{git_info['branch']}`")
    lines.append(f"- Commit: `{git_info['commit_short']}` (`{git_info['commit']}`)")
    lines.append(f"- Working tree: `{git_info['status']}`")
    lines.append("")
    lines.append("## Summary")
    lines.append("")
    lines.append("- Outcome:")
    lines.append("- Scope closed:")
    lines.append("- Not in scope / deferred:")
    lines.append("")
    lines.append("## Verification")
    lines.append("")
    lines.append(f"- Gate mode: `{verification_mode}`")
    lines.append(f"- Command: `{verify_cmd}`")
    lines.append("- Result: PASS/FAIL")
    lines.append("- Duration:")
    lines.append("")
    lines.append("## Metrics And Drift")
    lines.append("")
    lines.append("- TPR/FPR drift vs baseline:")
    lines.append("- Latency drift vs baseline:")
    lines.append("- Threshold contract updates:")
    lines.append("")
    lines.append("## Reproducibility")
    lines.append("")
    lines.append("- Manifest files updated:")
    lines.append("- New/retired scripts:")
    lines.append("- Data roots required:")
    lines.append("")
    lines.append("## Risks")
    lines.append("")
    lines.append("- Remaining technical risks:")
    lines.append("- Operational risks:")
    lines.append("")
    lines.append("## Phase Handoff")
    lines.append("")
    lines.append("- Ready for next phase: yes/no")
    lines.append("- Next phase first tasks:")
    lines.append("")
    lines.append("## Commit Record")
    lines.append("")
    lines.append("- Phase close commit:")
    lines.append("- Additional commits in phase:")
    lines.append("")
    return "\n".join(lines) + "\n"


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Generate a phase-close report scaffold")
    ap.add_argument("--phase", required=True, help="Phase identifier (e.g., 1, 2, 2a)")
    ap.add_argument(
        "--verification-mode",
        choices=["quick", "phase", "full"],
        default="phase",
        help="Verification mode expected at phase close.",
    )
    ap.add_argument(
        "--out",
        type=Path,
        default=None,
        help="Output markdown path. Default: docs/refactor/phase_reports/phase<phase>_close.md",
    )
    ap.add_argument(
        "--force",
        action="store_true",
        help="Overwrite output file if it already exists.",
    )
    return ap.parse_args()


def main() -> int:
    args = parse_args()
    git_info = _git_info()
    out_path = args.out or (DEFAULT_DIR / f"phase{args.phase}_close.md")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    if out_path.exists() and not args.force:
        raise SystemExit(f"refusing to overwrite existing file: {out_path} (use --force)")

    content = _render(
        phase=str(args.phase),
        verification_mode=str(args.verification_mode),
        git_info=git_info,
    )
    out_path.write_text(content, encoding="utf-8")
    print(f"[phase-close-report] wrote {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

