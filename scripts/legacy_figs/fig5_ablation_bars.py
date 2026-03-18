# scripts/legacy_figs/fig5_ablation_bars.py
import argparse
import json
import os

import matplotlib.pyplot as plt
import numpy as np

_DEF = [
    ("Local tiles", 0.020, 1.2),
    ("LW shrinkage", 0.015, 0.8),
    ("KA prior", 0.018, 0.7),
    ("Kronecker accel", 0.005, 0.0),
    ("Angle set opt", 0.012, 0.5),
    ("Confirm-2", 0.010, 0.0),  # boosts effective operating point
    ("EVT + conformal", 0.022, 0.0),
    ("TBD smoothing", 0.008, 0.3),
]


def _load(path):
    if not path:
        return None
    with open(path, "r") as fh:
        return json.load(fh)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--json",
        type=str,
        default=None,
        help="JSON with list of (name, delta_auc_1e5, delta_snr_db)",
    )
    ap.add_argument("--out", type=str, default="figs/outputs/fig5_ablation_bars.png")
    args = ap.parse_args()
    os.makedirs(os.path.dirname(args.out), exist_ok=True)

    data = _load(args.json)
    if not data:
        data = [{"name": n, "delta_auc_1e5": a, "delta_snr_db": s} for (n, a, s) in _DEF]

    names = [d["name"] for d in data]
    auc = np.array([d["delta_auc_1e5"] for d in data], dtype=float)
    snr = np.array([d["delta_snr_db"] for d in data], dtype=float)

    x = np.arange(len(names))
    fig, ax = plt.subplots(1, 2, figsize=(12, 4), constrained_layout=True)
    ax[0].bar(x, auc)
    ax[0].set_xticks(x)
    ax[0].set_xticklabels(names, rotation=35, ha="right")
    ax[0].set_ylabel("ΔAUC @ 1e‑5 (abs)")
    ax[0].set_title("Ablation: incremental AUC gains")
    ax[0].grid(axis="y", ls=":")

    ax[1].bar(x, snr)
    ax[1].set_xticks(x)
    ax[1].set_xticklabels(names, rotation=35, ha="right")
    ax[1].set_ylabel("ΔPD‑SNR (dB)")
    ax[1].set_title("Ablation: incremental PD‑SNR gains")
    ax[1].grid(axis="y", ls=":")

    plt.savefig(args.out, dpi=160)
    print(f"Wrote {args.out}")


if __name__ == "__main__":
    main()
