from __future__ import annotations

import argparse
import csv
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np


def _to_float(x: Any) -> float:
    if x is None:
        return float("nan")
    try:
        return float(x)
    except Exception:
        return float("nan")


def _quantile(x: list[float], q: float) -> float:
    arr = np.asarray([v for v in x if np.isfinite(v)], dtype=np.float64)
    if arr.size == 0:
        return float("nan")
    return float(np.quantile(arr, q))


def _mean(x: list[float]) -> float:
    arr = np.asarray([v for v in x if np.isfinite(v)], dtype=np.float64)
    if arr.size == 0:
        return float("nan")
    return float(np.mean(arr))


def _median(x: list[float]) -> float:
    return _quantile(x, 0.5)


def _frac_gt(a: list[float], b: list[float]) -> float:
    aa = np.asarray(a, dtype=np.float64)
    bb = np.asarray(b, dtype=np.float64)
    finite = np.isfinite(aa) & np.isfinite(bb)
    if not finite.any():
        return float("nan")
    return float(np.mean(aa[finite] > bb[finite]))


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Aggregate Shin motion sweep CSVs (from scripts/shin_ratbrain_motion_sweep.py) into a curve-ready summary.\n"
            "This script is label-free: it summarizes map stability (corr vs no-motion) and a label-free self-tail recall\n"
            "curve at a fixed background-tail operating point (bg-FPR) across multiple IQData files."
        )
    )
    parser.add_argument(
        "--in-dir",
        type=Path,
        required=True,
        help="Directory containing per-file motion sweep CSVs.",
    )
    parser.add_argument(
        "--glob",
        type=str,
        default="IQData*.csv",
        help="Glob pattern under --in-dir (default: %(default)s).",
    )
    parser.add_argument("--out-csv", type=Path, required=True)
    parser.add_argument("--out-png", type=Path, default=None)
    args = parser.parse_args()

    paths = sorted(args.in_dir.glob(args.glob))
    if not paths:
        raise SystemExit(f"No CSVs found under {args.in_dir} matching {args.glob!r}")

    # Group rows by motion amplitude.
    by_amp: dict[float, list[dict[str, Any]]] = defaultdict(list)
    files_seen: set[str] = set()
    for p in paths:
        with p.open("r", newline="") as f:
            reader = csv.DictReader(f)
            for row in reader:
                amp = _to_float(row.get("amp_px"))
                if not np.isfinite(amp):
                    continue
                by_amp[float(amp)].append(row)
                files_seen.add(str(row.get("iq_file") or p.name))

    amps = sorted(by_amp.keys())
    rows_out: list[dict[str, Any]] = []
    for amp in amps:
        rows = by_amp[amp]
        corr_base = [_to_float(r.get("corr_pd_base")) for r in rows]
        corr_stap = [_to_float(r.get("corr_pd_stap_pre")) for r in rows]
        tpr_base = [_to_float(r.get("tpr_base_self_at_fpr")) for r in rows]
        tpr_stap = [_to_float(r.get("tpr_stap_self_at_fpr")) for r in rows]
        disp_rms = [_to_float(r.get("disp_rms_px")) for r in rows]

        out = {
            "amp_px": float(amp),
            "n_rows": int(len(rows)),
            "corr_base_mean": _mean(corr_base),
            "corr_stap_mean": _mean(corr_stap),
            "corr_base_median": _median(corr_base),
            "corr_stap_median": _median(corr_stap),
            "corr_base_q25": _quantile(corr_base, 0.25),
            "corr_base_q75": _quantile(corr_base, 0.75),
            "corr_stap_q25": _quantile(corr_stap, 0.25),
            "corr_stap_q75": _quantile(corr_stap, 0.75),
            "corr_frac_stap_gt_base": _frac_gt(corr_stap, corr_base),
            "tpr_base_mean": _mean(tpr_base),
            "tpr_stap_mean": _mean(tpr_stap),
            "tpr_base_median": _median(tpr_base),
            "tpr_stap_median": _median(tpr_stap),
            "tpr_base_q25": _quantile(tpr_base, 0.25),
            "tpr_base_q75": _quantile(tpr_base, 0.75),
            "tpr_stap_q25": _quantile(tpr_stap, 0.25),
            "tpr_stap_q75": _quantile(tpr_stap, 0.75),
            "tpr_frac_stap_gt_base": _frac_gt(tpr_stap, tpr_base),
            "disp_rms_px_mean": _mean(disp_rms),
        }
        rows_out.append(out)

    args.out_csv.parent.mkdir(parents=True, exist_ok=True)
    with args.out_csv.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows_out[0].keys()))
        writer.writeheader()
        writer.writerows(rows_out)
    print(f"[shin-motion-agg] wrote {args.out_csv} ({len(rows_out)} amp points, {len(files_seen)} files)")

    if args.out_png is not None:
        try:
            import matplotlib.pyplot as plt  # type: ignore

            args.out_png.parent.mkdir(parents=True, exist_ok=True)
            amps_np = np.array([r["amp_px"] for r in rows_out], dtype=float)

            fig, axes = plt.subplots(1, 2, figsize=(11, 4), constrained_layout=True)

            ax = axes[0]
            base_med = np.array([r["corr_base_median"] for r in rows_out], dtype=float)
            base_q25 = np.array([r["corr_base_q25"] for r in rows_out], dtype=float)
            base_q75 = np.array([r["corr_base_q75"] for r in rows_out], dtype=float)
            stap_med = np.array([r["corr_stap_median"] for r in rows_out], dtype=float)
            stap_q25 = np.array([r["corr_stap_q25"] for r in rows_out], dtype=float)
            stap_q75 = np.array([r["corr_stap_q75"] for r in rows_out], dtype=float)
            ax.plot(amps_np, base_med, "o-", label="baseline PD", color="C0")
            ax.fill_between(amps_np, base_q25, base_q75, color="C0", alpha=0.15, linewidth=0)
            ax.plot(amps_np, stap_med, "o-", label="STAP PD (pre-KA)", color="C1")
            ax.fill_between(amps_np, stap_q25, stap_q75, color="C1", alpha=0.15, linewidth=0)
            ax.set_xlabel("motion amplitude (px)")
            ax.set_ylabel("corr vs no-motion")
            ax.set_title("Map Stability (median ± IQR)")
            ax.grid(True, alpha=0.3)
            ax.legend()

            ax = axes[1]
            base_med = np.array([r["tpr_base_median"] for r in rows_out], dtype=float)
            base_q25 = np.array([r["tpr_base_q25"] for r in rows_out], dtype=float)
            base_q75 = np.array([r["tpr_base_q75"] for r in rows_out], dtype=float)
            stap_med = np.array([r["tpr_stap_median"] for r in rows_out], dtype=float)
            stap_q25 = np.array([r["tpr_stap_q25"] for r in rows_out], dtype=float)
            stap_q75 = np.array([r["tpr_stap_q75"] for r in rows_out], dtype=float)
            ax.plot(amps_np, base_med, "o-", label="baseline", color="C0")
            ax.fill_between(amps_np, base_q25, base_q75, color="C0", alpha=0.15, linewidth=0)
            ax.plot(amps_np, stap_med, "o-", label="STAP", color="C1")
            ax.fill_between(amps_np, stap_q25, stap_q75, color="C1", alpha=0.15, linewidth=0)
            ax.set_xlabel("motion amplitude (px)")
            ax.set_ylabel("self-tail recall @ bg-FPR=1e-3")
            ax.set_title("Self-Tail Recall (median ± IQR)")
            ax.grid(True, alpha=0.3)
            ax.legend()

            fig.suptitle(f"Shin motion sweep aggregate ({len(files_seen)} files)", fontsize=11)
            fig.savefig(args.out_png, dpi=200)
            plt.close(fig)
            print(f"[shin-motion-agg] wrote {args.out_png}")
        except Exception as exc:
            print(f"[shin-motion-agg] failed to write plot: {exc}")


if __name__ == "__main__":
    main()
