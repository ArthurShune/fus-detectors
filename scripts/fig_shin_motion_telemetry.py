#!/usr/bin/env python3
"""
Shin RatBrain Fig3: summarize measured rigid-motion telemetry from the same
phase-correlation registration used in baseline preprocessing.

This is a small "measured motion" audit that complements the synthetic motion
injection ladders elsewhere: it reports the distribution of estimated rigid
shifts (in pixels) across the Shin baseline-matrix scenarios, and relates it
to proxy-separation uplift (e.g., ΔAUC for the matched-subspace detector
versus MC--SVD).
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np


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

    in_csv = Path(args.in_csv)
    with in_csv.open("r", newline="") as f:
        rows = list(csv.DictReader(f))
    if not rows:
        raise SystemExit(f"Empty CSV: {args.in_csv}")

    req = {"iq_file", "frame_tag", "frames_spec", "bundle_dir_stap", "method_key", "auc_flow_vs_bg"}
    missing = sorted([c for c in req if c not in rows[0].keys()])
    if missing:
        raise SystemExit(f"Missing required columns in {args.in_csv}: {missing}")

    # One row per scenario (clip + window). bundle_dir_stap is shared across method rows.
    scen_keys: set[tuple[str, str]] = set()
    scen_rows: list[dict[str, str | float]] = []
    for row in rows:
        key = (str(row["iq_file"]), str(row["frame_tag"]))
        if key in scen_keys:
            continue
        scen_keys.add(key)
        scen_rows.append(
            {
                "iq_file": str(row["iq_file"]),
                "frame_tag": str(row["frame_tag"]),
                "frames_spec": str(row["frames_spec"]),
                "bundle_dir_stap": str(row["bundle_dir_stap"]),
            }
        )
    shifts_rms: list[float] = []
    shifts_p90: list[float] = []
    failures: list[float] = []
    psr_med: list[float] = []
    for row in scen_rows:
        bdir = Path(str(row["bundle_dir_stap"]))
        meta = _load_meta(bdir)
        bs = meta.get("baseline_stats") or {}
        shifts_rms.append(float(bs.get("reg_shift_rms") or 0.0))
        shifts_p90.append(float(bs.get("reg_shift_p90") or 0.0))
        failures.append(float(bs.get("reg_failed_fraction") or 0.0))
        psr_med.append(float(bs.get("reg_psr_median")) if bs.get("reg_psr_median") is not None else float("nan"))
    for row, shift_rms, shift_p90, failure, psr in zip(
        scen_rows, shifts_rms, shifts_p90, failures, psr_med, strict=True
    ):
        row["reg_shift_rms_px"] = shift_rms
        row["reg_shift_p90_px"] = shift_p90
        row["reg_failed_fraction"] = failure
        row["reg_psr_median"] = psr

    # Proxy-separation uplift: ΔAUC (matched-subspace detector - MC--SVD) per scenario.
    auc_by_scenario: dict[tuple[str, str], dict[str, float]] = {}
    for row in rows:
        method_key = str(row["method_key"])
        if method_key not in {"mc_svd", "stap"}:
            continue
        key = (str(row["iq_file"]), str(row["frame_tag"]))
        auc_by_scenario.setdefault(key, {})
        try:
            auc_by_scenario[key][method_key] = float(row["auc_flow_vs_bg"])
        except (TypeError, ValueError):
            continue

    for row in scen_rows:
        key = (str(row["iq_file"]), str(row["frame_tag"]))
        auc_pair = auc_by_scenario.get(key, {})
        auc_mc = auc_pair.get("mc_svd", float("nan"))
        auc_stap = auc_pair.get("stap", float("nan"))
        row["auc_mc_svd"] = auc_mc
        row["auc_stap"] = auc_stap
        row["delta_auc_stap_minus_mcsvd"] = auc_stap - auc_mc if np.isfinite(auc_mc) and np.isfinite(auc_stap) else float("nan")

    out_csv = Path(args.out_csv)
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(scen_rows[0].keys()) if scen_rows else []
    with out_csv.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(scen_rows)

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

    x_all = np.asarray([float(row["reg_shift_rms_px"]) for row in scen_rows], dtype=np.float64)
    x = x_all[np.isfinite(x_all)]
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

    y = np.asarray([float(row["delta_auc_stap_minus_mcsvd"]) for row in scen_rows], dtype=np.float64)
    m = np.isfinite(x_all) & np.isfinite(y)
    if m.any():
        ax1.scatter(x_all[m], y[m], s=22, color="#1f77b4", alpha=0.85, edgecolor="none")
        ax1.axhline(float(np.median(y[m])), color="#1f77b4", alpha=0.35, linewidth=1.2)
    ax1.set_xlabel("Estimated rigid shift RMS (px)")
    ax1.set_ylabel(r"$\Delta$AUC (matched-subspace detector $-$ MC--SVD)")
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
