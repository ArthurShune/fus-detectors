#!/usr/bin/env python3
"""
Doppler STAP telemetry on PICMUS carotid PD cube using precomputed masks.

Assumes:
  - PD cube saved at data/picmus/pd_cube.npy (T,H,W).
  - Masks saved at data/picmus/picmus_masks.npz with mask_vessel, mask_bg.

Uses a Doppler STAP configuration tailored to PRF ≈ 100 Hz:
  dt = 0.01 s, L_t = 32, Pf=(5,23) Hz, Pg=(23,33) Hz, Pa=(33,50) Hz.

Reports Pf/Pa peak fractions, alias ratios, and ROC for PD-z and STAP scores.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from eval.metrics import partial_auc, roc_curve, tpr_at_fpr_target
from pipeline.realdata import tile_iter, extract_tile_stack
from pipeline.stap.hemo import HemoStapConfig, hemo_stap_scores_for_tiles


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="PICMUS carotid Doppler telemetry.")
    ap.add_argument(
        "--pd",
        type=Path,
        default=Path("data/picmus/pd_cube.npy"),
        help="Path to PD cube (T,H,W) .npy.",
    )
    ap.add_argument(
        "--masks",
        type=Path,
        default=Path("data/picmus/picmus_masks.npz"),
        help="Path to masks npz with mask_vessel and mask_bg.",
    )
    ap.add_argument(
        "--tile-h",
        type=int,
        default=8,
        help="Tile height.",
    )
    ap.add_argument(
        "--tile-w",
        type=int,
        default=8,
        help="Tile width.",
    )
    ap.add_argument(
        "--stride",
        type=int,
        default=3,
        help="Tile stride.",
    )
    ap.add_argument(
        "--alpha",
        type=float,
        default=0.6,
        help="Minimum fraction of labeled pixels to assign a tile label.",
    )
    ap.add_argument(
        "--pf-band",
        type=float,
        nargs=2,
        default=[5.0, 23.0],
        metavar=("PF_LO", "PF_HI"),
        help="Flow band in Hz (default: 5 23).",
    )
    ap.add_argument(
        "--pg-band",
        type=float,
        nargs=2,
        default=[23.0, 33.0],
        metavar=("PG_LO", "PG_HI"),
        help="Guard band in Hz (default: 23 33).",
    )
    ap.add_argument(
        "--pa-band",
        type=float,
        nargs=2,
        default=[33.0, 50.0],
        metavar=("PA_LO", "PA_HI"),
        help="Alias band in Hz (default: 33 50).",
    )
    ap.add_argument(
        "--lt",
        type=int,
        default=32,
        help="Hankel length L_t (default: 32).",
    )
    ap.add_argument(
        "--dt",
        type=float,
        default=0.01,
        help="Sampling interval dt in seconds (default: 0.01).",
    )
    ap.add_argument(
        "--diff2-hp",
        action="store_true",
        help="Apply a 2nd-order temporal difference high-pass to tile slow-time before STAP/PD-z.",
    )
    return ap.parse_args()


def _pd_z_score(p_t: np.ndarray, baseline_frac: float = 0.2) -> float:
    T = p_t.size
    if T <= 1:
        return 0.0
    n_base = max(1, int(baseline_frac * T))
    n_base = min(n_base, T - 1)
    base = p_t[:n_base]
    mu = float(base.mean())
    sigma = float(base.std())
    sigma_safe = sigma if sigma > 0.0 else 1.0
    z = (p_t - mu) / sigma_safe
    z_det = z[n_base:]
    if z_det.size == 0:
        z_det = z
    return float(z_det.max())


def _diff2_hp(p_t: np.ndarray) -> np.ndarray:
    """
    Second-order temporal difference high-pass.
    Returns an array of length T-2; if T<3, returns zeros of length T.
    """

    T = p_t.size
    if T < 3:
        return np.zeros_like(p_t)
    return p_t[2:] - 2.0 * p_t[1:-1] + p_t[:-2]


def main() -> None:
    args = parse_args()

    pd = np.load(args.pd)  # (T,H,W)
    masks = np.load(args.masks)
    mask_v = masks["mask_vessel"].astype(bool)
    mask_bg = masks["mask_bg"].astype(bool)

    T, H, W = pd.shape
    tile_hw = (args.tile_h, args.tile_w)
    stride = args.stride
    alpha = args.alpha

    labels = []
    p_tiles = []
    pdz_scores = []
    for y0, x0 in tile_iter((H, W), tile_hw, stride):
        tile = extract_tile_stack(pd, y0, x0, tile_hw)  # (T, th, tw)
        p_t = tile.mean(axis=(1, 2))
        if args.diff2_hp:
            p_t = _diff2_hp(p_t)
        p_tiles.append(p_t.astype(np.float32))

        v_cov = mask_v[y0 : y0 + tile_hw[0], x0 : x0 + tile_hw[1]].mean()
        b_cov = mask_bg[y0 : y0 + tile_hw[0], x0 : x0 + tile_hw[1]].mean()
        if v_cov >= alpha:
            labels.append(1)
        elif b_cov >= alpha:
            labels.append(0)
        else:
            labels.append(-1)
        pdz_scores.append(_pd_z_score(p_t))

    p_tiles = np.stack(p_tiles, axis=0)
    labels = np.asarray(labels, dtype=int)
    pdz_scores = np.asarray(pdz_scores, dtype=np.float64)

    pos = labels == 1
    neg = labels == 0
    if not pos.any() or not neg.any():
        raise RuntimeError("No positive or negative tiles; adjust masks/alpha.")

    stap_cfg = HemoStapConfig(
        dt=float(args.dt),
        L_t=int(args.lt),
        pf_band=(float(args.pf_band[0]), float(args.pf_band[1])),
        pg_band=(float(args.pg_band[0]), float(args.pg_band[1])),
        pa_band=(float(args.pa_band[0]), float(args.pa_band[1])),
        bg_beta=0.0,
    )
    scores = hemo_stap_scores_for_tiles(p_tiles, stap_cfg)
    Ef = scores["Ef"]
    Ea = scores["Ea"]
    peak_idx = np.argmax(np.stack([Ef, Ea, scores["Eo"]], axis=1), axis=1)
    alias_ratio = (Ea + 1e-8) / (Ef + 1e-8)

    def frac(arr: np.ndarray, val: int) -> float:
        return float((arr == val).mean()) if arr.size else 0.0

    print("[picmus] band occupancy (all tiles):")
    print(
        f"  Pf peak: {frac(peak_idx,0):.3f}, "
        f"Pa peak: {frac(peak_idx,1):.3f}, "
        f"Po peak: {frac(peak_idx,2):.3f}"
    )
    print(
        f"[picmus] alias log10 median (all): {np.median(np.log10(alias_ratio)):.3f}"
    )

    print("[picmus] band occupancy by class:")
    print(
        f"  Pf peak (vessel/bg): {frac(peak_idx[pos],0):.3f} / {frac(peak_idx[neg],0):.3f}"
    )
    print(
        f"  Pa peak (vessel/bg): {frac(peak_idx[pos],1):.3f} / {frac(peak_idx[neg],1):.3f}"
    )
    print(
        f"  Po peak (vessel/bg): {frac(peak_idx[pos],2):.3f} / {frac(peak_idx[neg],2):.3f}"
    )
    print(
        f"[picmus] alias log10 median (vessel/bg): "
        f"{np.median(np.log10(alias_ratio[pos])):.3f} / {np.median(np.log10(alias_ratio[neg])):.3f}"
    )

    # ROC summaries
    def roc_summary(scores: np.ndarray, name: str) -> None:
        fpr, tpr, _ = roc_curve(scores[pos], scores[neg], num_thresh=4096)
        pauc = partial_auc(fpr, tpr, fpr_max=1e-3)
        fpr_min = 1.0 / float(neg.sum())
        tpr_emp = tpr_at_fpr_target(fpr, tpr, target_fpr=float(np.clip(fpr_min, 1e-8, 1.0)))
        print(
            f"  [{name}] pAUC@1e-3={pauc:.4f}, TPR@FPR_min={tpr_emp:.4f} "
            f"(FPR_min={fpr_min:.3e})"
        )

    print("[picmus] ROC:")
    roc_summary(pdz_scores, "PD-z")
    roc_summary(scores["hemo_br"].astype(np.float64), "Doppler BR")
    roc_summary(scores["hemo_glrt"].astype(np.float64), "Doppler GLRT")


if __name__ == "__main__":
    main()
