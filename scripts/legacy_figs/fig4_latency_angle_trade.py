# scripts/legacy_figs/fig4_latency_angle_trade.py
import argparse
import glob
import json
import os
from typing import List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np


def _collect(globpat: str) -> Optional[List[Tuple[int, float, float, Optional[float]]]]:
    files = sorted(glob.glob(globpat))
    rows = []
    for f in files:
        try:
            with open(f, "r") as fh:
                j = json.load(fh)
            K = j.get("sim", {}).get("angles", None) or j.get("sim", {}).get("K", None)
            hz = j.get("perf", {}).get("hz", None)
            snr = j.get("metrics", {}).get("pd_snr_db", None)
            base = j.get("metrics", {}).get("pd_snr_baseline_db", None)
            rows.append((K, hz, snr, base))
        except Exception:
            pass
    rows = [r for r in rows if all(v is not None for v in r[:3])]
    if not rows:
        return None
    arr = np.array(rows, dtype=float)
    # unique by K: take max hz and corresponding snr
    Ks = np.unique(arr[:, 0]).astype(int)
    out = []
    for k in Ks:
        mask = arr[:, 0] == k
        idx = np.argmax(arr[mask, 1])  # best Hz at this K
        sub = arr[mask][idx]
        out.append(
            (
                int(sub[0]),
                float(sub[1]),
                float(sub[2]),
                float(sub[3]) if not np.isnan(sub[3]) else None,
            )
        )
    out.sort(key=lambda r: r[0])
    return out


def _simulate() -> List[Tuple[int, float, float, Optional[float]]]:
    K = np.array([3, 5, 7, 9, 11, 13, 15])
    hz = 30.0 / (K / 3) ** 0.9
    snr = 3.0 + 0.6 * (K - 3)
    base = np.full_like(snr, snr[0] - 3.0)
    return [
        (int(k), float(h), float(s), float(b))
        for k, h, s, b in zip(K, hz, snr, base, strict=False)
    ]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--glob", type=str, default="runs/*.json", help="Telemetry JSON glob")
    ap.add_argument("--out", type=str, default="figs/outputs/fig4_latency_angle_trade.png")
    args = ap.parse_args()
    os.makedirs(os.path.dirname(args.out), exist_ok=True)

    data = _collect(args.glob)
    if data is None or len(data) == 0:
        data = _simulate()

    K = np.array([d[0] for d in data], dtype=int)
    Hz = np.array([d[1] for d in data], dtype=float)
    SNR = np.array([d[2] for d in data], dtype=float)
    BASE = np.array([d[3] if d[3] is not None else np.nan for d in data], dtype=float)

    fig, ax = plt.subplots(1, 2, figsize=(10, 4), constrained_layout=True)
    ax[0].plot(K, Hz, marker="o")
    ax[0].set_xlabel("Angles (K)")
    ax[0].set_ylabel("Update rate (Hz)")
    ax[0].set_title("Latency vs angle count")
    ax[0].grid(True, ls=":")

    if np.all(np.isnan(BASE)):
        ax[1].plot(K, SNR, marker="o")
        ax[1].set_ylabel("PD‑SNR (dB)")
        ax[1].set_title("STAP PD‑SNR vs K")
    else:
        ax[1].plot(K, SNR - BASE, marker="o")
        ax[1].set_ylabel("ΔPD‑SNR vs baseline (dB)")
        ax[1].set_title("ΔSNR vs K")
    ax[1].set_xlabel("Angles (K)")
    ax[1].grid(True, ls=":")

    plt.savefig(args.out, dpi=160)
    print(f"Wrote {args.out}")


if __name__ == "__main__":
    main()
