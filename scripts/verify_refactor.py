#!/usr/bin/env python3
"""Tiered verification harness for refactor phases.

This script defines three gate levels:
  - quick: fast regression guard for day-to-day refactor changes
  - phase: broader audit run at phase boundaries
  - full: paper-level reproduction gate

By default the script prints the execution plan without running commands.
Use `--execute` to run steps.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

REPO = Path(__file__).resolve().parents[1]
DEFAULT_THRESHOLDS = REPO / "configs" / "refactor_verify_thresholds.json"


@dataclass(frozen=True)
class Step:
    key: str
    description: str
    cmd: list[str]
    requires: tuple[Path, ...] = ()
    optional: bool = False


def _have_conda() -> bool:
    return shutil.which("conda") is not None


def _py_cmd(
    *,
    conda_env: str,
    use_conda: bool,
    script: str,
    args: Sequence[str],
) -> list[str]:
    if use_conda and _have_conda():
        return ["conda", "run", "-n", conda_env, "python", script, *args]
    return [sys.executable, script, *args]


def _format_cmd(cmd: Sequence[str]) -> str:
    return " ".join(subprocess.list2cmdline([part]) if " " in part else part for part in cmd)


def _load_thresholds(path: Path) -> dict:
    if not path.exists():
        raise FileNotFoundError(f"Threshold config not found: {path}")
    data = json.loads(path.read_text())
    if not isinstance(data, dict):
        raise ValueError(f"Threshold config must be a JSON object: {path}")
    return data


def _quick_metric_check(*, matrix_json: Path, thresholds: dict) -> list[str]:
    errors: list[str] = []
    if not matrix_json.exists():
        return [f"missing quick matrix JSON: {matrix_json}"]

    records = json.loads(matrix_json.read_text())
    if not isinstance(records, list):
        return [f"quick matrix JSON is not a list: {matrix_json}"]

    quick = thresholds.get("quick", {})
    regime = str(quick.get("regime", "open"))
    window_offset = int(quick.get("window_offset", 0))
    stap_method = str(quick.get("stap_method", "STAP (MC-SVD+STAP, full)"))
    baseline_method = str(quick.get("baseline_method", "MC-SVD"))

    def _pick(method: str) -> dict | None:
        for rec in records:
            if not isinstance(rec, dict):
                continue
            if str(rec.get("method")) != method:
                continue
            if str(rec.get("regime")) != regime:
                continue
            if int(rec.get("window_offset", -1)) != window_offset:
                continue
            return rec
        return None

    stap = _pick(stap_method)
    base = _pick(baseline_method)
    if stap is None:
        errors.append(
            "missing quick record for method="
            f"{stap_method}, regime={regime}, window_offset={window_offset}"
        )
    if base is None:
        errors.append(
            "missing quick record for method="
            f"{baseline_method}, regime={regime}, window_offset={window_offset}"
        )
    if errors:
        return errors

    stap_min = quick.get("stap_min", {})
    for metric, floor in stap_min.items():
        observed = float(stap.get(metric, float("nan")))
        if observed < float(floor):
            errors.append(f"{stap_method} {metric}={observed:.4f} < required {float(floor):.4f}")

    baseline_max = quick.get("baseline_max", {})
    for metric, cap in baseline_max.items():
        observed = float(base.get(metric, float("nan")))
        if observed > float(cap):
            errors.append(f"{baseline_method} {metric}={observed:.4f} > allowed {float(cap):.4f}")

    fpr_tol_rel = float(quick.get("fpr_tolerance_rel", 0.25))
    for metric in ("fpr@0.001", "fpr@0.0003", "fpr@0.0001"):
        target = float(metric.split("@", 1)[1])
        observed = float(stap.get(metric, float("nan")))
        if target <= 0:
            continue
        rel = abs(observed - target) / target
        if rel > fpr_tol_rel:
            errors.append(
                f"{stap_method} {metric}={observed:.6g} drifts from target {target:.6g} "
                f"by {rel:.2%} (limit {fpr_tol_rel:.2%})"
            )

    return errors


def _build_steps(*, mode: str, conda_env: str, use_conda: bool, device: str) -> list[Step]:
    steps_quick = [
        Step(
            key="tests_quick",
            description="Fast unit/regression subset for temporal STAP core",
            cmd=[
                "pytest",
                "-q",
                "tests/test_temporal_msd.py",
                "tests/test_tyler_covariance_batched.py",
                "tests/test_stap_fastpath_batch_wrapper.py",
                "tests/test_covariance.py",
            ],
        ),
        Step(
            key="brain_quick_matrix",
            description="Single-window Brain-OpenSkull matrix replay (MC-SVD vs STAP)",
            cmd=_py_cmd(
                conda_env=conda_env,
                use_conda=use_conda,
                script="scripts/fair_filter_comparison.py",
                args=[
                    "--mode",
                    "matrix",
                    "--eval-score",
                    "vnext",
                    "--matrix-regimes",
                    "open",
                    "--matrix-seeds-open",
                    "1",
                    "--window-length",
                    "64",
                    "--window-offsets",
                    "0",
                    "--matrix-use-profile",
                    "--matrix-mcsvd-energy-frac",
                    "0.90",
                    "--matrix-mcsvd-baseline-support",
                    "window",
                    "--methods",
                    "mcsvd,stap_full",
                    "--generated-root",
                    "runs/pilot/fair_filter_matrix_phase1_quick",
                    "--autogen-missing",
                    "--stap-device",
                    device,
                    "--out-csv",
                    "reports/refactor/refactor_quick_fair_matrix.csv",
                    "--out-json",
                    "reports/refactor/refactor_quick_fair_matrix.json",
                ],
            ),
            requires=(REPO / "runs" / "pilot" / "r4c_kwave_seed1",),
            optional=True,
        ),
    ]

    steps_phase = [
        Step(
            key="brain_crosswindow",
            description="Cross-window threshold transfer audit refresh",
            cmd=_py_cmd(
                conda_env=conda_env,
                use_conda=use_conda,
                script="scripts/brain_crosswindow_calibration.py",
                args=[
                    "--runs-root",
                    "runs/pilot/fair_filter_matrix_pd_r3_localbaselines",
                    "--out-csv",
                    "reports/refactor/refactor_phase_crosswindow.csv",
                    "--out-json",
                    "reports/refactor/refactor_phase_crosswindow.json",
                    "--out-tex",
                    "reports/refactor/refactor_phase_crosswindow_table.tex",
                ],
            ),
            requires=(REPO / "runs" / "pilot" / "fair_filter_matrix_pd_r3_localbaselines",),
            optional=True,
        ),
        Step(
            key="latency_open_steady",
            description="Brain-OpenSkull latency replay with steady-state windows 2..N",
            cmd=_py_cmd(
                conda_env=conda_env,
                use_conda=use_conda,
                script="scripts/latency_rerun_check.py",
                args=[
                    "--src",
                    "runs/latency_pilot_open",
                    "--out-root",
                    "runs/latency_phase_gate_open",
                    "--profile",
                    "Brain-OpenSkull",
                    "--window-length",
                    "64",
                    "--window-offset",
                    "0",
                    "--window-offsets",
                    "0,64,128",
                    "--stap-device",
                    device,
                    "--stap-debug-samples",
                    "0",
                    "--tile-batch",
                    "192",
                    "--cuda-warmup-heavy",
                ],
            ),
            requires=(REPO / "runs" / "latency_pilot_open",),
            optional=True,
        ),
    ]

    steps_full = [
        Step(
            key="paper_table5",
            description="Full Table 5 reproduction script",
            cmd=["bash", "scripts/reproduce_table5_brain_kwave.sh"],
        ),
        Step(
            key="manifest_refresh",
            description="Refresh central reproducibility manifest",
            cmd=_py_cmd(
                conda_env=conda_env,
                use_conda=use_conda,
                script="scripts/generate_repro_manifest.py",
                args=[],
            ),
        ),
    ]

    if mode == "quick":
        return steps_quick
    if mode == "phase":
        return [*steps_quick, *steps_phase]
    if mode == "full":
        return [*steps_quick, *steps_phase, *steps_full]
    raise ValueError(f"unsupported mode: {mode}")


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(
        description="Refactor verification harness (quick / phase / full)"
    )
    ap.add_argument("--mode", choices=["quick", "phase", "full"], default="quick")
    ap.add_argument(
        "--execute",
        action="store_true",
        help="Execute steps. Default is dry-run plan output.",
    )
    ap.add_argument("--conda-env", default="stap-fus", help="Conda env name when conda is used.")
    ap.add_argument(
        "--python-runner",
        choices=["auto", "conda", "local"],
        default="auto",
        help="How to invoke Python commands.",
    )
    ap.add_argument("--stap-device", default="cuda", help="Device forwarded to replay scripts.")
    ap.add_argument(
        "--thresholds",
        type=Path,
        default=DEFAULT_THRESHOLDS,
        help="Threshold JSON for quick gate checks.",
    )
    ap.add_argument(
        "--allow-missing-data-gates",
        action="store_true",
        help=(
            "Allow optional data-backed steps (and metric checks that depend on them) "
            "to be skipped without failing. Intended for CI environments without local data roots."
        ),
    )
    return ap.parse_args()


def main() -> int:
    args = parse_args()
    use_conda = args.python_runner in {"auto", "conda"}
    if args.python_runner == "local":
        use_conda = False
    if args.python_runner == "conda" and not _have_conda():
        print(
            "[verify-refactor] ERROR: --python-runner=conda but conda was not found",
            file=sys.stderr,
        )
        return 2

    steps = _build_steps(
        mode=args.mode,
        conda_env=args.conda_env,
        use_conda=use_conda,
        device=args.stap_device,
    )

    print(f"[verify-refactor] mode={args.mode} execute={args.execute} steps={len(steps)}")
    for idx, step in enumerate(steps, start=1):
        print(f"  {idx:02d}. {step.key}: {step.description}")
        print(f"      cmd: {_format_cmd(step.cmd)}")
        if step.requires:
            req = ", ".join(str(p) for p in step.requires)
            print(f"      requires: {req}")

    if not args.execute:
        print("[verify-refactor] dry-run only. Re-run with --execute to run steps.")
        return 0

    env = os.environ.copy()
    env["PYTHONPATH"] = "."

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
            return 3

        print(f"[verify-refactor] running: {step.key}")
        proc = subprocess.run(step.cmd, cwd=str(REPO), env=env)
        if proc.returncode != 0:
            print(
                f"[verify-refactor] step failed: {step.key} (code={proc.returncode})",
                file=sys.stderr,
            )
            return proc.returncode
        completed_steps.add(step.key)

    thresholds = _load_thresholds(args.thresholds)
    quick_json = REPO / "reports" / "refactor" / "refactor_quick_fair_matrix.json"
    quick_step_needed = any(step.key == "brain_quick_matrix" for step in steps)
    quick_step_attempted = "brain_quick_matrix" in completed_steps
    quick_step_skipped = "brain_quick_matrix" in skipped_steps
    if quick_step_needed and not quick_step_attempted:
        if args.allow_missing_data_gates and quick_step_skipped:
            print(
                "[verify-refactor] quick metric checks skipped "
                "(brain_quick_matrix was skipped and --allow-missing-data-gates is set)"
            )
        else:
            print(
                "[verify-refactor] quick metric checks unavailable because "
                "brain_quick_matrix did not run",
                file=sys.stderr,
            )
            return 4
    else:
        errors = _quick_metric_check(matrix_json=quick_json, thresholds=thresholds)
        if errors:
            print("[verify-refactor] quick metric checks failed:", file=sys.stderr)
            for err in errors:
                print(f"  - {err}", file=sys.stderr)
            return 4

    print("[verify-refactor] all checks passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
