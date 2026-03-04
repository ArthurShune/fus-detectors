#!/usr/bin/env python3
"""Generate a Phase 0 file inventory for refactor planning.

Outputs:
  - docs/refactor/REPO_CLASSIFICATION_PHASE0.csv
  - docs/refactor/REPO_CLASSIFICATION_PHASE0.md
"""

from __future__ import annotations

import argparse
import csv
import subprocess
from collections import Counter, defaultdict
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]

CORE_PREFIXES = (
    "pipeline/",
    "sim/",
    "eval/",
    "tests/",
    "configs/",
)

CORE_SCRIPT_EXACT = {
    "brain_baseline_sanity_table.py",
    "brain_cov_train_ablation_table.py",
    "brain_crosswindow_calibration.py",
    "brain_detector_ablation_table.py",
    "brain_detector_swap_table.py",
    "brain_kwave_vnext_baselines_table.py",
    "fair_filter_comparison.py",
    "generate_repro_manifest.py",
    "latency_realdata_rerun_check.py",
    "latency_rerun_check.py",
    "replay_stap_from_run.py",
    "reproduce_table5_brain_kwave.sh",
    "refactor_inventory.py",
    "shin_ratbrain_make_bundle.py",
    "twinkling_make_bundles.py",
    "ulm_zenodo_7883227_make_bundle.py",
    "verify_gpu.py",
    "verify_refactor.py",
}

LEGACY_FILE_EXACT = {
    "foo.txt",
    "foo_conda.txt",
    "test_url_spaces.aux",
    "test_url_spaces.log",
    "tmp_ratio_current.py",
    "tmp_ratio_mix.py",
}

LEGACY_PREFIXES = (
    "__pycache__/",
    "build/",
    "tmp/",
)

LEGACY_EXTENSIONS = (
    ".aux",
    ".fdb_latexmk",
    ".fls",
    ".log",
    ".out",
)

CORE_INFRA_FILES = {
    ".gitattributes",
    ".gitignore",
    ".pre-commit-config.yaml",
    "Dockerfile",
    "LICENSE",
    "LATENCY_OPTIMIZATION_SPEC.md",
    "Makefile",
    "README.md",
    "build_context.sh",
    "environment.yml",
    "pyproject.toml",
}


@dataclass(frozen=True)
class Row:
    path: str
    track: str
    reason: str


def _git_ls_files() -> list[str]:
    proc = subprocess.run(
        ["git", "ls-files"],
        cwd=str(REPO),
        check=True,
        stdout=subprocess.PIPE,
        text=True,
    )
    files = [line.strip() for line in proc.stdout.splitlines() if line.strip()]
    files.sort()
    return files


def _classify(path: str) -> tuple[str, str]:
    name = Path(path).name

    if path in LEGACY_FILE_EXACT:
        return "legacy_candidate", "temporary/ad-hoc file"

    if path.startswith(LEGACY_PREFIXES):
        return "legacy_candidate", "build/tmp/cache directory"

    if path.endswith(LEGACY_EXTENSIONS):
        return "legacy_candidate", "tracked build artifact"

    if path.startswith(CORE_PREFIXES):
        return "core", "core source/test/config directory"

    if path.startswith("scripts/"):
        script_name = Path(path).name
        if script_name in CORE_SCRIPT_EXACT:
            return "core", "repro/runtime orchestration script"
        return "experiments", "analysis/experiment script"

    if path.startswith("docs/refactor/"):
        return "core", "refactor governance doc"

    if path.startswith("notebooks/"):
        return "experiments", "exploratory notebook"

    if path.startswith("figs/"):
        return "experiments", "figure artifact/source"

    if name in CORE_INFRA_FILES:
        return "core", "repo runtime/build interface"

    if name.startswith("stap_fus_") and name.endswith(".pdf"):
        return "experiments", "compiled manuscript artifact"

    return "experiments", "paper/supporting artifact"


def _write_csv(rows: list[Row], out_csv: Path) -> None:
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    with out_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["path", "track", "reason"])
        writer.writeheader()
        for row in rows:
            writer.writerow({"path": row.path, "track": row.track, "reason": row.reason})


def _write_markdown(rows: list[Row], out_md: Path) -> None:
    out_md.parent.mkdir(parents=True, exist_ok=True)
    counts = Counter(row.track for row in rows)
    by_track: dict[str, list[Row]] = defaultdict(list)
    for row in rows:
        by_track[row.track].append(row)

    def _top_dirs(track: str, limit: int = 12) -> list[tuple[str, int]]:
        ctr: Counter[str] = Counter()
        for row in by_track.get(track, []):
            p = Path(row.path)
            key = p.parts[0] if p.parts else row.path
            ctr[key] += 1
        return ctr.most_common(limit)

    generated = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%SZ")

    lines: list[str] = []
    lines.append("# Repo Classification (Phase 0)")
    lines.append("")
    lines.append(f"Generated: {generated} (UTC)")
    lines.append("")
    lines.append("## Summary")
    lines.append("")
    lines.append(f"- core: {counts.get('core', 0)} files")
    lines.append(f"- experiments: {counts.get('experiments', 0)} files")
    lines.append(f"- legacy_candidate: {counts.get('legacy_candidate', 0)} files")
    lines.append("")
    lines.append("## Core Candidates (keep + refactor first)")
    lines.append("")
    for key, value in _top_dirs("core"):
        lines.append(f"- {key}: {value}")
    lines.append("")
    lines.append("## Experiments Candidates (modularize / move under experiments)")
    lines.append("")
    for key, value in _top_dirs("experiments"):
        lines.append(f"- {key}: {value}")
    lines.append("")
    lines.append("## Legacy Candidates (review remove/archive)")
    lines.append("")
    legacy = by_track.get("legacy_candidate", [])
    if legacy:
        for row in legacy[:80]:
            lines.append(f"- `{row.path}` ({row.reason})")
        if len(legacy) > 80:
            lines.append(f"- ... {len(legacy) - 80} more (see CSV)")
    else:
        lines.append("- none")
    lines.append("")
    lines.append("## Notes")
    lines.append("")
    lines.append("- This is a planning inventory, not a deletion list.")
    lines.append(
        "- Final delete/archive decisions should be tied to "
        "Phase 1/2 verification gates."
    )
    lines.append(
        "- Full path-level mapping is in "
        "`docs/refactor/REPO_CLASSIFICATION_PHASE0.csv`."
    )

    out_md.write_text("\n".join(lines) + "\n", encoding="utf-8")


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Generate Phase 0 core/experiments/legacy inventory")
    ap.add_argument(
        "--out-csv",
        type=Path,
        default=REPO / "docs" / "refactor" / "REPO_CLASSIFICATION_PHASE0.csv",
        help="Path for CSV inventory output.",
    )
    ap.add_argument(
        "--out-md",
        type=Path,
        default=REPO / "docs" / "refactor" / "REPO_CLASSIFICATION_PHASE0.md",
        help="Path for markdown summary output.",
    )
    return ap.parse_args()


def main() -> int:
    args = parse_args()
    files = _git_ls_files()
    rows: list[Row] = []
    for path in files:
        track, reason = _classify(path)
        rows.append(Row(path=path, track=track, reason=reason))
    _write_csv(rows, args.out_csv)
    _write_markdown(rows, args.out_md)
    print(f"[refactor-inventory] wrote {args.out_csv}")
    print(f"[refactor-inventory] wrote {args.out_md}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
