from __future__ import annotations

import argparse
from pathlib import Path

from sim.simus.bundle import SUPPORTED_SIMUS_STAP_PROFILES, derive_bundle_from_run


def _discover_runs(sim_root: Path) -> list[Path]:
    root = Path(sim_root)
    if (root / "dataset" / "meta.json").is_file():
        return [root]
    runs = [p for p in sorted(root.iterdir()) if p.is_dir() and (p / "dataset" / "meta.json").is_file()]
    if not runs:
        raise FileNotFoundError(f"{root}: no run directories with dataset/meta.json")
    return runs


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Derive acceptance bundles for one or more canonical SIMUS runs.")
    ap.add_argument("--sim-root", type=Path, default=None, help="Root containing per-run subdirectories.")
    ap.add_argument("--run", type=Path, action="append", default=None, help="Explicit run directory (repeatable).")
    ap.add_argument("--out-root", type=Path, default=None, help="Optional output root for derived bundles.")
    ap.add_argument(
        "--stap-profile",
        type=str,
        default="Brain-SIMUS-Clin",
        choices=list(SUPPORTED_SIMUS_STAP_PROFILES),
    )
    ap.add_argument("--stap-device", type=str, default="cpu")
    ap.add_argument("--baseline-type", type=str, default="mc_svd")
    ap.add_argument("--no-run-stap", action="store_true", help="Only derive baseline bundle (run_stap=False).")
    return ap.parse_args()


def main() -> None:
    args = parse_args()
    runs: list[Path] = []
    if args.sim_root is not None:
        runs.extend(_discover_runs(Path(args.sim_root)))
    for run in args.run or []:
        runs.append(Path(run))
    if not runs:
        raise ValueError("Provide --sim-root or at least one --run")

    seen: set[str] = set()
    for run in runs:
        key = str(Path(run).resolve())
        if key in seen:
            continue
        seen.add(key)
        bundle_root = (Path(args.out_root) / Path(run).name) if args.out_root is not None else (Path(run) / "bundle")
        bundle_dir = derive_bundle_from_run(
            run_dir=Path(run),
            out_root=bundle_root,
            dataset_name=Path(run).name,
            stap_profile=str(args.stap_profile),
            baseline_type=str(args.baseline_type),
            run_stap=not bool(args.no_run_stap),
            stap_device=str(args.stap_device),
        )
        print(f"[simus-cli-bundle] wrote {bundle_dir}", flush=True)


if __name__ == "__main__":
    main()
