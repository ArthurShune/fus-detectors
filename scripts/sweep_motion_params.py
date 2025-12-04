#!/usr/bin/env python3
"""
Sweep motion-stress hyperparameters (msd_ridge, ka_target_shrink_perp, ka_alpha, contrast)
on the r3 k-Wave pilot, replaying both alias-only and Pf-trace variants.

Each combo is written under:
  <out_root>/<tag>/{alias,pftrace}/pw_7.5MHz_5ang_3ens_192T_seed1/...

The script extracts PD-mask telemetry (global + ≥20/50/80%) directly from meta.json
for quick comparison.
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from dataclasses import dataclass
from itertools import product
from pathlib import Path
from typing import Dict, Optional

REPO = Path(__file__).resolve().parents[1]
DEFAULT_SRC = REPO / "runs/pilot/r3_kwave"
DEFAULT_PRIOR = REPO / "runs/motion/priors/ka_prior_lt4_prf3k.npy"
DEFAULT_OUT = REPO / "runs/motion/sweeps"


@dataclass(frozen=True)
class Combo:
    msd_ridge: float
    shrink_perp: float
    ka_alpha: float
    contrast: float

    def tag(self) -> str:
        def fmt(x: float) -> str:
            return f"{x:.3g}".replace(".", "p")

        return (
            f"ridge{fmt(self.msd_ridge)}_shp{fmt(self.shrink_perp)}_"
            f"alpha{fmt(self.ka_alpha)}_c{fmt(self.contrast)}"
        )


def _bundle_dir(root: Path) -> Optional[Path]:
    cands = sorted(root.glob("pw_*"))
    return cands[0] if cands else None


def _run_replay(
    out_dir: Path, combo: Combo, src: Path, pftrace: bool, base_args: Dict[str, str]
) -> Path:
    bundle = _bundle_dir(out_dir)
    if bundle and (bundle / "meta.json").exists():
        return bundle

    cmd = [
        "python",
        str(REPO / "scripts/replay_stap_from_run.py"),
        "--src",
        str(src),
        "--out",
        str(out_dir),
        "--tile-h",
        "10",
        "--tile-w",
        "10",
        "--tile-stride",
        "5",
        "--lt",
        "4",
        "--diag-load",
        "1e-2",
        "--cov-estimator",
        "tyler_pca",
        "--huber-c",
        "5.0",
        "--fd-span-mode",
        "psd",
        "--fd-span-rel",
        "0.30,1.10",
        "--grid-step-rel",
        "0.12",
        "--max-pts",
        "5",
        "--fd-min-pts",
        "3",
        "--msd-lambda",
        "5e-2",
        "--msd-ridge",
        f"{combo.msd_ridge}",
        "--msd-agg",
        "median",
        "--msd-ratio-rho",
        "0.05",
        "--motion-half-span-rel",
        "0.20",
        "--msd-contrast-alpha",
        f"{combo.contrast}",
        "--constraint-mode",
        "exp+deriv",
        "--constraint-ridge",
        "0.10",
        "--mvdr-load-mode",
        "auto",
        "--mvdr-auto-kappa",
        "50",
        "--ka-mode",
        "library",
        "--ka-prior-path",
        base_args["ka_prior_path"],
        "--ka-directional-beta",
        "--ka-kappa",
        "30",
        "--ka-beta-bounds",
        "0.05,0.5",
        "--ka-alpha",
        f"{combo.ka_alpha}",
        "--ka-target-retain-f",
        "1.0",
        "--ka-target-shrink-perp",
        f"{combo.shrink_perp}",
        "--stap-device",
        base_args.get("stap_device", "cuda"),
        "--stap-debug-samples",
        base_args.get("debug_samples", "32"),
    ]
    if pftrace:
        cmd.append("--ka-equalize-pf-trace")

    print("[run]", " ".join(cmd))
    env = os.environ.copy()
    env.setdefault("PYTHONPATH", str(REPO))
    subprocess.run(["conda", "run", "-n", "stap-fus", *cmd], check=True, env=env, cwd=str(REPO))

    bundle = _bundle_dir(out_dir)
    if not bundle:
        raise RuntimeError(f"No bundle emitted under {out_dir}")
    return bundle


def _load_metrics(bundle: Path) -> Dict[str, object]:
    meta = json.loads((bundle / "meta.json").read_text())
    tele = meta.get("stap_fallback_telemetry", {})
    fields = [
        "flow_pdmask_ratio_median",
        "flow_pdmask_cov_ge_20_ratio_median",
        "flow_pdmask_cov_ge_50_ratio_median",
        "flow_pdmask_cov_ge_80_ratio_median",
        "flow_mu_ratio_actual",
    ]
    data = {k: tele.get(k) for k in fields}
    data["flow_pdmask_fraction"] = tele.get("flow_pdmask_fraction")
    data["ka_trace_scale_lock_hist"] = tele.get("ka_trace_scale_lock_hist")
    data["ka_pf_trace_alpha_median"] = tele.get("ka_pf_trace_alpha_median")
    data["bundle"] = str(bundle)
    return data


def main() -> None:
    ap = argparse.ArgumentParser(description="Sweep motion hyperparameters on r3 k-Wave pilot.")
    ap.add_argument("--src", type=Path, default=DEFAULT_SRC)
    ap.add_argument("--out", type=Path, default=DEFAULT_OUT)
    ap.add_argument(
        "--msd-ridge",
        type=str,
        default="0.06,0.08,0.10",
        help="Comma-separated msd_ridge values.",
    )
    ap.add_argument(
        "--ka-shrink-perp",
        type=str,
        default="0.95,0.975,1.0",
        help="Comma-separated ka_target_shrink_perp values.",
    )
    ap.add_argument(
        "--ka-alpha",
        type=str,
        default="0.0,0.1",
        help="Comma-separated ka_alpha values.",
    )
    ap.add_argument(
        "--contrast",
        type=str,
        default="0.0,0.8",
        help="Comma-separated MSD contrast alphas.",
    )
    ap.add_argument(
        "--ka-prior-path",
        type=Path,
        default=DEFAULT_PRIOR,
        help="Path to KA prior npy file.",
    )
    ap.add_argument(
        "--run-pftrace",
        action="store_true",
        help="Also run Pf-trace variant for each combo.",
    )
    args = ap.parse_args()

    out_root = Path(args.out)
    out_root.mkdir(parents=True, exist_ok=True)

    msd_ridges = [float(x) for x in args.msd_ridge.split(",") if x.strip()]
    shrink_vals = [float(x) for x in args.ka_shrink_perp.split(",") if x.strip()]
    alphas = [float(x) for x in args.ka_alpha.split(",") if x.strip()]
    contrasts = [float(x) for x in args.contrast.split(",") if x.strip()]
    combos = [
        Combo(m, s, a, c) for m, s, a, c in product(msd_ridges, shrink_vals, alphas, contrasts)
    ]

    base_args = {
        "ka_prior_path": str(args.ka_prior_path),
        "stap_device": "cuda",
        "debug_samples": "16",
    }

    summary = []
    for combo in combos:
        tag_dir = out_root / combo.tag()
        results = {}
        for variant in ["alias", "pftrace" if args.run_pftrace else None]:
            if variant is None:
                continue
            variant_dir = tag_dir / variant
            variant_dir.mkdir(parents=True, exist_ok=True)
            bundle = _run_replay(
                variant_dir,
                combo,
                args.src,
                pftrace=(variant == "pftrace"),
                base_args=base_args,
            )
            metrics = _load_metrics(bundle)
            results[variant] = metrics
            print(
                f"[metrics] {combo.tag()} / {variant} -> "
                f"pdmask={metrics['flow_pdmask_ratio_median']:.3f} "
                f"(cov50={metrics['flow_pdmask_cov_ge_50_ratio_median']:.3f})"
            )
        summary.append({"combo": combo.__dict__, "results": results})

    out_json = out_root / "sweep_summary.json"
    out_json.write_text(json.dumps(summary, indent=2))
    print("[summary]", out_json)


if __name__ == "__main__":
    main()
