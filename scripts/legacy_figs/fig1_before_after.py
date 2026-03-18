# scripts/legacy_figs/fig1_before_after.py
import argparse
import os
from pathlib import Path
from typing import Optional

import matplotlib.pyplot as plt
import numpy as np


def _load_array(p: Optional[str]) -> Optional[np.ndarray]:
    if not p:
        return None
    pth = Path(p)
    if not pth.exists():
        raise FileNotFoundError(p)
    if pth.suffix.lower() == ".npz":
        with np.load(pth) as z:
            # try common keys
            for k in ["pd_map", "PD", "arr", "data"]:
                if k in z:
                    return z[k]
            # fallback to first array
            key = list(z.keys())[0]
            return z[key]
    else:
        return np.load(pth)


def _simulate_pd(H=128, W=128, seed=0):
    rng = np.random.default_rng(seed)
    bg = rng.exponential(scale=1.0, size=(H, W)).astype(np.float32)
    mask = np.zeros((H, W), bool)
    yy, xx = np.ogrid[:H, :W]
    cy, cx, ry, rx = H // 2, W // 2, H // 5, W // 6
    mask[((yy - cy) ** 2) / (ry**2) + ((xx - cx - 10) ** 2) / (rx**2) <= 1.0] = True
    svd = bg + 0.7 * mask.astype(np.float32) + 0.1 * rng.standard_normal((H, W))
    stap = bg * 0.7 + 1.4 * mask.astype(np.float32) + 0.05 * rng.standard_normal((H, W))
    svd = np.clip(svd, 0, None).astype(np.float32)
    stap = np.clip(stap, 0, None).astype(np.float32)
    return svd, stap, mask


def _pd_snr_db(P, mask_flow):
    mask_flow = mask_flow.astype(bool)
    mask_bg = ~mask_flow
    mf = float(P[mask_flow].mean()) if mask_flow.any() else 0.0
    vb = float(P[mask_bg].var()) if mask_bg.any() else 0.0
    eps = 1e-12
    return 10.0 * np.log10((mf * mf + eps) / (vb + eps))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--svd", type=str, default=None, help="NPY/NPZ PD map for SVD baseline")
    ap.add_argument("--stap", type=str, default=None, help="NPY/NPZ PD map for STAP")
    ap.add_argument("--mask", type=str, default=None, help="NPY/NPZ vessel mask (bool)")
    ap.add_argument("--out", type=str, default="figs/outputs/fig1_before_after.png")
    args = ap.parse_args()

    os.makedirs(os.path.dirname(args.out), exist_ok=True)

    P_svd = _load_array(args.svd)
    P_stap = _load_array(args.stap)
    mask = _load_array(args.mask)
    if P_svd is None or P_stap is None:
        P_svd, P_stap, mask_sim = _simulate_pd(seed=123)
        if mask is None:
            mask = mask_sim

    H, W = P_svd.shape
    vmin = min(P_svd.min(), P_stap.min())
    vmax = max(P_svd.max(), P_stap.max())

    fig, ax = plt.subplots(1, 3, figsize=(12, 4), constrained_layout=True)
    im0 = ax[0].imshow(P_svd, vmin=vmin, vmax=vmax, origin="upper")
    ax[0].set_title("Baseline: Global SVD PD")
    im1 = ax[1].imshow(P_stap, vmin=vmin, vmax=vmax, origin="upper")
    ax[1].set_title("STAP‑for‑fUS PD")
    im2 = ax[2].imshow(P_stap - P_svd, origin="upper")
    ax[2].set_title("Δ PD (STAP − SVD)")

    if mask is not None:
        try:
            ax[0].contour(mask.astype(float), levels=[0.5], linewidths=0.75)
            ax[1].contour(mask.astype(float), levels=[0.5], linewidths=0.75)
            ax[2].contour(mask.astype(float), levels=[0.5], linewidths=0.75)
        except Exception:
            pass

    fig.colorbar(im0, ax=ax[0], fraction=0.046)
    fig.colorbar(im1, ax=ax[1], fraction=0.046)
    fig.colorbar(im2, ax=ax[2], fraction=0.046)

    if mask is not None:
        snr_svd = _pd_snr_db(P_svd, mask)
        snr_stap = _pd_snr_db(P_stap, mask)
        ax[0].text(0.02, 0.98, f"SNR {snr_svd:.1f} dB", transform=ax[0].transAxes, va="top")
        ax[1].text(0.02, 0.98, f"SNR {snr_stap:.1f} dB", transform=ax[1].transAxes, va="top")
        ax[2].text(
            0.02, 0.98, f"ΔSNR {snr_stap-snr_svd:.1f} dB", transform=ax[2].transAxes, va="top"
        )

    plt.savefig(args.out, dpi=160)
    print(f"Wrote {args.out}")


if __name__ == "__main__":
    main()
