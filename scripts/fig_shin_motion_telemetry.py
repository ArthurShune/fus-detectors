#!/usr/bin/env python3
"""
Shin RatBrain Fig3: summarize measured rigid-motion telemetry from the same
phase-correlation registration used in baseline preprocessing.

This is a small "measured motion" audit that complements the synthetic motion
injection ladders elsewhere: it reports the distribution of estimated rigid
shifts (in pixels) across the Shin baseline-matrix scenarios, and relates it
to the proxy-separation uplift (e.g., ΔAUC for STAP vs MC--SVD).
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


def _load_meta(bundle_dir: Path) -> dict:
    return json.loads((bundle_dir / "meta.json").read_text())


def main() -> None:
    ap = argparse.ArgumentParser(description="Plot measured rigid-motion telemetry for Shin baseline-matrix scenarios.")
    ap.add_argument(
        "--in-csv",
        type=Path,
        default=Path("reports/shin_ratbrain_baseline_matrix_shinU_e970_Lt64_nomaskunion_k80.csv"),
        help="Scenario-matrix CSV (default: %(default)s).",
    )
    ap.add_argument(
        "--out-pdf",
        type=Path,
        default=Path("figs/paper/shin_motion_telemetry.pdf"),
        help="Output PDF path (default: %(default)s).",
    )
    ap.add_argument("--also-png", action="store_true", help="Also write a matching PNG next to the PDF.")
    ap.add_argument(
        "--out-csv",
        type=Path,
        default=Path("reports/shin_motion_telemetry.csv"),
        help="Write the extracted telemetry table here (default: %(default)s).",
    )
    args = ap.parse_args()

    df = pd.read_csv(Path(args.in_csv))
    if df.empty:
        raise SystemExit(f"Empty CSV: {args.in_csv}")

    req = {"iq_file", "frame_tag", "frames_spec", "bundle_dir_stap", "method_key", "auc_flow_vs_bg"}
    missing = sorted([c for c in req if c not in df.columns])
    if missing:
        raise SystemExit(f"Missing required columns in {args.in_csv}: {missing}")

    # One row per scenario (clip + window). bundle_dir_stap is shared across method rows.
    scen = (
        df.drop_duplicates(subset=["iq_file", "frame_tag"])
        .loc[:, ["iq_file", "frame_tag", "frames_spec", "bundle_dir_stap"]]
        .copy()
    )
    shifts_rms: list[float] = []
    shifts_p90: list[float] = []
    failures: list[float] = []
    psr_med: list[float] = []
    for _, row in scen.iterrows():
        bdir = Path(str(row["bundle_dir_stap"]))
        meta = _load_meta(bdir)
        bs = meta.get("baseline_stats") or {}
        shifts_rms.append(float(bs.get("reg_shift_rms") or 0.0))
        shifts_p90.append(float(bs.get("reg_shift_p90") or 0.0))
        failures.append(float(bs.get("reg_failed_fraction") or 0.0))
        psr_med.append(float(bs.get("reg_psr_median")) if bs.get("reg_psr_median") is not None else float("nan"))
    scen["reg_shift_rms_px"] = shifts_rms
    scen["reg_shift_p90_px"] = shifts_p90
    scen["reg_failed_fraction"] = failures
    scen["reg_psr_median"] = psr_med

    # Proxy-separation uplift: ΔAUC (STAP - MC--SVD) per scenario.
    auc = (
        df.loc[df["method_key"].isin(["mc_svd", "stap"]), ["iq_file", "frame_tag", "method_key", "auc_flow_vs_bg"]]
        .copy()
        .pivot(index=["iq_file", "frame_tag"], columns="method_key", values="auc_flow_vs_bg")
    )
    auc = auc.rename(columns={"mc_svd": "auc_mc_svd", "stap": "auc_stap"})
    auc["delta_auc_stap_minus_mcsvd"] = auc["auc_stap"] - auc["auc_mc_svd"]

    scen = scen.merge(auc.reset_index(), on=["iq_file", "frame_tag"], how="left")
    scen.to_csv(Path(args.out_csv), index=False)

    try:
        import matplotlib.pyplot as plt
    except Exception as exc:  # pragma: no cover
        raise SystemExit(f"matplotlib required to plot: {exc}")

    plt.rcParams.update(
        {
            "font.family": "serif",
            "mathtext.fontset": "cm",
            "font.size": 10,
            "axes.titlesize": 11,
            "axes.labelsize": 10,
            "legend.fontsize": 9,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )

    fig, axes = plt.subplots(1, 2, figsize=(12.8, 3.8), constrained_layout=True)
    ax0, ax1 = axes

    x = pd.to_numeric(scen["reg_shift_rms_px"], errors="coerce").to_numpy(dtype=np.float64, copy=False)
    x = x[np.isfinite(x)]
    ax0.hist(x, bins=12, color="#666666", alpha=0.75, edgecolor="#444444", linewidth=0.6)
    ax0.set_xlabel("Estimated rigid shift RMS (px)")
    ax0.set_ylabel("Scenario count")
    ax0.set_title("Measured rigid-motion telemetry (Shin)")
    ax0.grid(True, axis="y", alpha=0.25)
    if x.size:
        ax0.text(
            0.98,
            0.95,
            f"median={np.median(x):.4f}px\np90={np.quantile(x,0.90):.4f}px",
            transform=ax0.transAxes,
            ha="right",
            va="top",
            fontsize=9,
            color="#333333",
        )

    y = pd.to_numeric(scen["delta_auc_stap_minus_mcsvd"], errors="coerce").to_numpy(dtype=np.float64, copy=False)
    m = np.isfinite(x) & np.isfinite(y)
    if m.any():
        ax1.scatter(x[m], y[m], s=22, color="#1f77b4", alpha=0.85, edgecolor="none")
        ax1.axhline(float(np.median(y[m])), color="#1f77b4", alpha=0.35, linewidth=1.2)
    ax1.set_xlabel("Estimated rigid shift RMS (px)")
    ax1.set_ylabel(r"$\Delta$AUC (STAP $-$ MC--SVD)")
    ax1.set_title("Proxy-separation uplift vs measured motion")
    ax1.grid(True, alpha=0.25)

    out_pdf = Path(args.out_pdf)
    out_pdf.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_pdf, bbox_inches="tight", pad_inches=0.06)
    if bool(args.also_png):
        fig.savefig(out_pdf.with_suffix(".png"), dpi=300, bbox_inches="tight", pad_inches=0.06)
    plt.close(fig)

    print(f"[shin-motion-telemetry] wrote {out_pdf} and {Path(args.out_csv)}")


if __name__ == "__main__":
    main()

