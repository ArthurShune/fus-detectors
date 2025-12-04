#!/usr/bin/env python3
"""
Doppler-band telemetry on Wang ULM IQ tiles with optional vessel/background masks.

- Loads one IQDataXXX.dat file (optionally a frame range).
- Forms power time-series per tile.
- Computes Doppler-band Pf/Pa energies using the STAP core (Hankel + covariance +
  Pf/Pa/guard projectors with Doppler bands).
- Outputs label-free stats (Pf/Pa peak fractions, alias ratios) and, if a vessel
  flow map is provided, PD-z ROC and class-specific band stats.

Usage
-----
    PYTHONPATH=. python scripts/wang_doppler_telemetry.py \
        --file-index 1 \
        --frames 0 255 \
        --flow-map path/to/flow_map.npy \
        --vessel-quantile 0.995 \
        --bg-quantile 0.5
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Iterable, Sequence, Tuple

import numpy as np

from eval.metrics import partial_auc, roc_curve, tpr_at_fpr_target
from pipeline.realdata import tile_iter
from pipeline.realdata.wang_ulm import (
    load_wang_iq,
    load_wang_metadata,
    list_wang_files,
)
from pipeline.stap.hemo import HemoStapConfig, hemo_stap_scores_for_tiles


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Doppler-band telemetry on Wang IQ tiles.")
    ap.add_argument(
        "--data-root",
        type=Path,
        default=None,
        help="Root containing IQDataXXX.dat and IQSizeInfo.mat (default: data/wang_ulm).",
    )
    ap.add_argument(
        "--file-index",
        type=int,
        default=1,
        help="1-based index of IQDataXXX.dat to load (default: 1).",
    )
    ap.add_argument(
        "--frames",
        type=int,
        nargs=2,
        default=None,
        metavar=("START", "STOP"),
        help="Optional inclusive frame range; default loads all frames in the file.",
    )
    ap.add_argument(
        "--tile-h",
        type=int,
        default=8,
        help="Tile height (pixels).",
    )
    ap.add_argument(
        "--tile-w",
        type=int,
        default=8,
        help="Tile width (pixels).",
    )
    ap.add_argument(
        "--stride",
        type=int,
        default=3,
        help="Tile stride (pixels).",
    )
    ap.add_argument(
        "--lt",
        type=int,
        default=8,
        help="Hankel length L_t for the Doppler STAP core.",
    )
    ap.add_argument(
        "--flow-map",
        type=Path,
        default=None,
        help="Optional vessel flow map (.npy or .mat) for masks.",
    )
    ap.add_argument(
        "--flow-key",
        type=str,
        default=None,
        help="Optional key when loading a .mat flow map (defaults to the first array).",
    )
    ap.add_argument(
        "--vessel-quantile",
        type=float,
        default=0.995,
        help="Quantile for vessel mask on the flow map (default 0.995).",
    )
    ap.add_argument(
        "--bg-quantile",
        type=float,
        default=0.5,
        help="Quantile for background mask on the flow map (default 0.5).",
    )
    ap.add_argument(
        "--derive-masks",
        action="store_true",
        help="If no flow map is provided, derive vessel/background masks from PD mean using quantiles.",
    )
    ap.add_argument(
        "--pd-vessel-quantile",
        type=float,
        default=0.99,
        help="Quantile on PD mean for derived vessel mask (default 0.99).",
    )
    ap.add_argument(
        "--pd-bg-quantile",
        type=float,
        default=0.2,
        help="Quantile on PD mean for derived background mask (default 0.2).",
    )
    ap.add_argument(
        "--pf-band",
        type=float,
        nargs=2,
        default=[30.0, 200.0],
        metavar=("PF_LO", "PF_HI"),
        help="Flow/Pf band in Hz (default: 30 200).",
    )
    ap.add_argument(
        "--pa-band",
        type=float,
        nargs=2,
        default=[350.0, 500.0],
        metavar=("PA_LO", "PA_HI"),
        help="Alias/Pa band in Hz (default: 350 500).",
    )
    ap.add_argument(
        "--pg-band",
        type=float,
        nargs=2,
        default=[200.0, 350.0],
        metavar=("PG_LO", "PG_HI"),
        help="Guard band in Hz (default: 200 350).",
    )
    ap.add_argument(
        "--dt",
        type=float,
        default=0.001,
        help="Sampling interval (s) for Doppler STAP bands (default: 0.001).",
    )
    return ap.parse_args()


def _frame_indices(frames_arg: Sequence[int] | None) -> np.ndarray | None:
    if frames_arg is None:
        return None
    if len(frames_arg) != 2:
        raise ValueError("frames must be two integers: START STOP")
    start, stop = frames_arg
    if stop < start:
        raise ValueError("frames STOP must be >= START")
    return np.arange(start, stop + 1, dtype=int)


def _load_flow_map(path: Path, key: str | None = None) -> np.ndarray:
    if path.suffix.lower() == ".npy":
        return np.load(path)
    if path.suffix.lower() == ".mat":
        import scipy.io as sio

        mat = sio.loadmat(path)
        if key is not None and key in mat:
            arr = mat[key]
        else:
            # pick the first array-like entry that is not metadata
            arr = None
            for k, v in mat.items():
                if k.startswith("__"):
                    continue
                if isinstance(v, np.ndarray):
                    arr = v
                    break
            if arr is None:
                raise KeyError(f"No array found in {path}, keys={list(mat.keys())}")
        return np.asarray(arr)
    raise ValueError(f"Unsupported flow map extension: {path.suffix}")


def _masks_from_flow(
    flow: np.ndarray, vessel_q: float, bg_q: float
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Return (vessel_mask, bg_mask) as booleans on (H, W).
    """

    if flow.ndim != 2:
        raise ValueError(f"Flow map must be 2-D, got shape {flow.shape}")
    v_thr = np.quantile(flow, vessel_q)
    b_thr = np.quantile(flow, bg_q)
    vessel = flow >= v_thr
    background = flow <= b_thr
    # Avoid overlap
    background = background & (~vessel)
    return vessel.astype(bool), background.astype(bool)


def _pd_z_score(p_t: np.ndarray, baseline_frac: float = 0.2, eps: float = 1e-12) -> float:
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


def main() -> None:
    args = parse_args()

    info = load_wang_metadata(args.data_root)
    files = list_wang_files(args.data_root)
    if not files:
        raise RuntimeError("No IQDataXXX.dat files found.")
    if args.file_index < 1 or args.file_index > len(files):
        raise ValueError(f"file_index {args.file_index} out of range 1..{len(files)}")
    path = files[args.file_index - 1]
    idx = _frame_indices(args.frames)

    print(f"[wang] metadata row={info.row}, col={info.col}, frames={info.frames}")
    print(f"[wang] using file: {path.name}")
    if idx is None:
        print("[wang] loading all frames (one file ~0.6 GB).")
    else:
        print(f"[wang] loading frames {idx[0]}..{idx[-1]} (count={idx.size}).")

    # Load IQ (subset) and form PD cube.
    iq = load_wang_iq(path, info, frames=idx, dtype=np.complex64)  # (T, H, W)
    pd_T_HW = (np.abs(iq) ** 2).astype(np.float32, copy=False)
    T, H, W = pd_T_HW.shape
    print(f"[wang] PD cube shape (T,H,W) = ({T}, {H}, {W})")

    # Optional flow map masks.
    vessel_mask = bg_mask = None
    if args.flow_map is not None:
        flow = _load_flow_map(args.flow_map, key=args.flow_key)
        if flow.shape != (H, W):
            raise ValueError(f"Flow map shape {flow.shape} does not match IQ plane {(H, W)}")
        vessel_mask, bg_mask = _masks_from_flow(flow, args.vessel_quantile, args.bg_quantile)
        print(
            f"[wang] flow map loaded; vessel pixels={vessel_mask.sum()}, "
            f"background pixels={bg_mask.sum()}"
        )
    elif args.derive_masks:
        pd_mean = pd_T_HW.mean(axis=0)
        v_thr = np.quantile(pd_mean, args.pd_vessel_quantile)
        b_thr = np.quantile(pd_mean, args.pd_bg_quantile)
        vessel_mask = pd_mean >= v_thr
        bg_mask = (pd_mean <= b_thr) & (~vessel_mask)
        print(
            f"[wang] derived masks from PD mean; vessel pixels={vessel_mask.sum()}, "
            f"background pixels={bg_mask.sum()}"
        )

    # Tile extraction.
    tile_hw = (int(args.tile_h), int(args.tile_w))
    stride = int(args.stride)
    tile_series = []
    tile_masks = []

    for y0, x0 in tile_iter((H, W), tile_hw, stride):
        tile = pd_T_HW[:, y0 : y0 + tile_hw[0], x0 : x0 + tile_hw[1]]
        p_t = tile.mean(axis=(1, 2))
        tile_series.append(p_t.astype(np.float32))
        if vessel_mask is not None and bg_mask is not None:
            v_cov = vessel_mask[y0 : y0 + tile_hw[0], x0 : x0 + tile_hw[1]].mean()
            b_cov = bg_mask[y0 : y0 + tile_hw[0], x0 : x0 + tile_hw[1]].mean()
            # Label: +1 vessel, 0 background, -1 mixed/other.
            if v_cov >= 0.6:
                tile_masks.append(1)
            elif b_cov >= 0.6:
                tile_masks.append(0)
            else:
                tile_masks.append(-1)
        else:
            tile_masks.append(-1)

    p_tiles = np.stack(tile_series, axis=0)  # (N_tiles, T)
    labels = np.asarray(tile_masks, dtype=int)
    print(f"[wang] collected {p_tiles.shape[0]} tiles, T={T}")

    # Doppler-band STAP config (assume PRF ~ 1 kHz).
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
    hemo_br = scores["hemo_br"]
    hemo_glrt = scores["hemo_glrt"]
    alias_ratio = (Ea + 1e-8) / (Ef + 1e-8)
    peak_idx = np.argmax(np.stack([Ef, Ea, scores["Eg"]], axis=1), axis=1)  # 0=Pf,1=Pa,2=Po

    def frac(arr: np.ndarray, val: int) -> float:
        return float((arr == val).mean()) if arr.size else 0.0

    print("[wang] label-free band occupancy (all tiles):")
    print(f"  Pf peak: {frac(peak_idx,0):.3f}, Pa peak: {frac(peak_idx,1):.3f}, Po peak: {frac(peak_idx,2):.3f}")
    print(
        f"[wang] alias ratio log10 stats (all): mean={np.log10(alias_ratio).mean():.3f}, "
        f"median={np.median(np.log10(alias_ratio)):.3f}"
    )

    # Class-specific telemetry if masks provided.
    if (labels == 1).any() and (labels == 0).any():
        pos = labels == 1
        neg = labels == 0
        print("[wang] band occupancy by class:")
        print(
            f"  Pf peak (vessel/bg): {frac(peak_idx[pos],0):.3f} / {frac(peak_idx[neg],0):.3f}"
        )
        print(
            f"  Pa peak (vessel/bg): {frac(peak_idx[pos],1):.3f} / {frac(peak_idx[neg],1):.3f}"
        )
        print(
            f"  Po peak (vessel/bg): {frac(peak_idx[pos],2):.3f} / {frac(peak_idx[neg],2):.3f}"
        )
        alias_pos = np.log10(alias_ratio[pos])
        alias_neg = np.log10(alias_ratio[neg])
        print(
            f"[wang] alias log10 median (vessel/bg): {np.median(alias_pos):.3f} / {np.median(alias_neg):.3f}"
        )

        # PD-z ROC (baseline detector) on vessel vs background tiles.
        pdz_scores = np.array([_pd_z_score(p) for p in p_tiles])
        fpr, tpr, _ = roc_curve(pdz_scores[pos], pdz_scores[neg], num_thresh=4096)
        pauc = partial_auc(fpr, tpr, fpr_max=1e-3)
        fpr_min = 1.0 / float(neg.sum())
        tpr_emp = tpr_at_fpr_target(fpr, tpr, target_fpr=fpr_min)
        print(
            f"[wang] PD-z ROC: pAUC@1e-3={pauc:.4f}, TPR@FPR_min={tpr_emp:.4f} (FPR_min={fpr_min:.3e})"
        )
    else:
        print("[wang] no vessel/background masks; skipping class-conditional telemetry and ROC.")


if __name__ == "__main__":
    main()
