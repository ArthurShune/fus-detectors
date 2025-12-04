# figs/fig6_telemetry_rank.py
import argparse
import glob
import json
import os

import matplotlib.pyplot as plt
import numpy as np


def _collect(globpat):
    files = sorted(glob.glob(globpat))
    rows = []
    for f in files:
        try:
            with open(f, "r") as fh:
                j = json.load(fh)
            motion_um = j.get("sim", {}).get("motion_um", None)
            if motion_um is None:
                # try nested structure or amplitude field
                motion_um = j.get("sim", {}).get("motion", {}).get("amp_um", None)
            condR = (
                j.get("stap", {}).get("condR", None)
                or j.get("stap", {}).get("cond_R", None)
                or j.get("condR", None)
            )
            effrank = j.get("stap", {}).get("eff_rank", None)
            rho = j.get("confirm2", {}).get("rho", None)
            rows.append((motion_um, condR, effrank, rho))
        except Exception:
            pass
    rows = [r for r in rows if r[0] is not None and r[1] is not None]
    if not rows:
        return None
    arr = np.array(rows, dtype=float)
    return arr


def _simulate():
    rng = np.random.default_rng(0)
    motion = np.linspace(5, 200, 15)  # microns
    condR = 1e3 + 2e3 * (motion / 200.0) ** 2 + 200 * rng.normal(size=motion.size)
    effr = 30 - 10 * (motion / 200.0) + rng.normal(scale=0.5, size=motion.size)
    rho = 0.2 + 0.5 * (motion / 200.0) + rng.normal(scale=0.03, size=motion.size)
    arr = np.stack([motion, condR, effr, rho], axis=1)
    return arr


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--glob", type=str, default="runs/*.json")
    ap.add_argument("--out", type=str, default="figs/outputs/fig6_telemetry_rank.png")
    args = ap.parse_args()
    os.makedirs(os.path.dirname(args.out), exist_ok=True)

    arr = _collect(args.glob)
    if arr is None:
        arr = _simulate()

    motion = arr[:, 0]
    condR = arr[:, 1]
    effr = arr[:, 2] if arr.shape[1] > 2 and not np.all(np.isnan(arr[:, 2])) else None
    rho = arr[:, 3] if arr.shape[1] > 3 and not np.all(np.isnan(arr[:, 3])) else None

    fig = plt.figure(figsize=(10, 4))
    gs = fig.add_gridspec(1, 3, width_ratios=[1, 1, 1])

    ax0 = fig.add_subplot(gs[0, 0])
    ax0.plot(motion, condR, marker="o")
    ax0.set_xlabel("Motion amplitude (µm)")
    ax0.set_ylabel("cond(#R)")
    ax0.set_title("Covariance condition vs motion")
    ax0.grid(True, ls=":")

    ax1 = fig.add_subplot(gs[0, 1])
    if effr is not None:
        ax1.plot(motion, effr, marker="o")
        ax1.set_ylabel("Effective rank (approx.)")
    else:
        ax1.plot(motion, np.log10(condR), marker="o")
        ax1.set_ylabel("log10 cond(#R)")
    ax1.set_xlabel("Motion amplitude (µm)")
    ax1.set_title("Rank / conditioning vs motion")
    ax1.grid(True, ls=":")

    ax2 = fig.add_subplot(gs[0, 2])
    if rho is not None:
        ax2.hist(rho, bins=10)
        ax2.set_xlabel("ρ (look correlation)")
        ax2.set_ylabel("count")
        ax2.set_title("Confirm‑2 ρ distribution")
        ax2.grid(True, ls=":")
    else:
        ax2.text(0.5, 0.5, "No ρ in telemetry", ha="center", va="center")
        ax2.set_axis_off()

    plt.tight_layout()
    plt.savefig(args.out, dpi=160)
    print(f"Wrote {args.out}")


if __name__ == "__main__":
    main()
