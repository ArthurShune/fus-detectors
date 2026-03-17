#!/usr/bin/env python3
"""
Physical Doppler "sanity link" telemetry.

Purpose
-------
docs/legacy/simulation_spec.txt calls for a defensible bridge between simulation and real IQ
without in-vivo labels: compare summary slow-time statistics (PSD bands, peak
frequency occupancy, coherence, and simple low-rank structure proxies).

This script computes those summaries for:
  - Phase 0/1 physical Doppler surrogate datasets (runs/sim/*/dataset/icube.npy)
  - Shin RatBrain Fig3 IQ cubes (data/shin_zenodo_10711806/ratbrain_fig3_raw)
  - Twinkling/Gammex RawBCF phantom CFM cubes (data/twinkling_artifact/...)

Outputs are deterministic given inputs and written under reports/.
"""

from __future__ import annotations

import argparse
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Literal

import numpy as np

from pipeline.realdata.shin_ratbrain import load_shin_iq, load_shin_metadata
from pipeline.realdata.twinkling_artifact import (
    RawBCFPar,
    decode_rawbcf_cfm_cube,
    parse_rawbcf_par,
    read_rawbcf_frame,
)
from pipeline.realdata.twinkling_bmode_mask import build_tube_masks_from_rawbcf


def _slugify(text: str) -> str:
    s = re.sub(r"[\s/]+", "_", str(text).strip())
    s = re.sub(r"[^A-Za-z0-9_.-]+", "_", s)
    return s.strip("._-") or "run"


def _parse_indices(spec: str, n_max: int) -> list[int]:
    """
    Parse either:
      - comma list: "0,1,2"
      - slice: "0:10" or "0:10:2"
      - single int: "5"
    """
    spec = (spec or "").strip()
    if not spec:
        return list(range(n_max))
    if "," in spec:
        out: list[int] = []
        for part in spec.split(","):
            part = part.strip()
            if part:
                out.append(int(part))
        return out
    if ":" in spec:
        parts = [p.strip() for p in spec.split(":")]
        if len(parts) not in (2, 3):
            raise ValueError(f"Invalid slice spec: {spec!r}")
        start = int(parts[0]) if parts[0] else 0
        stop = int(parts[1]) if parts[1] else n_max
        step = int(parts[2]) if len(parts) == 3 and parts[2] else 1
        return list(range(start, min(stop, n_max), step))
    return [int(spec)]


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, sort_keys=True)
        f.write("\n")


def _save_np(path: Path, arr: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    np.save(path, np.asarray(arr))


@dataclass(frozen=True)
class BandEdges:
    pf_lo_hz: float = 30.0
    pf_hi_hz: float = 250.0
    pg_lo_hz: float = 250.0
    pg_hi_hz: float = 400.0
    pa_lo_hz: float = 400.0


@dataclass(frozen=True)
class TileSpec:
    h: int = 8
    w: int = 8
    stride: int = 3


def _integral_image2d(frame: np.ndarray) -> np.ndarray:
    H, W = frame.shape
    sat = np.zeros((H + 1, W + 1), dtype=frame.dtype)
    sat[1:, 1:] = frame
    return sat.cumsum(axis=0).cumsum(axis=1)


def _tile_grid(H: int, W: int, tile: TileSpec) -> tuple[np.ndarray, np.ndarray]:
    th, tw = int(tile.h), int(tile.w)
    st = int(tile.stride)
    if th <= 0 or tw <= 0 or st <= 0:
        raise ValueError("tile spec must be positive")
    ys = np.arange(0, H - th + 1, st, dtype=np.int32)
    xs = np.arange(0, W - tw + 1, st, dtype=np.int32)
    return ys, xs


def _tile_sums_from_sat(
    sat: np.ndarray, ys: np.ndarray, xs: np.ndarray, tile: TileSpec
) -> np.ndarray:
    th, tw = int(tile.h), int(tile.w)
    y1 = ys + th
    x1 = xs + tw
    return (
        sat[y1[:, None], x1[None, :]]
        - sat[ys[:, None], x1[None, :]]
        - sat[y1[:, None], xs[None, :]]
        + sat[ys[:, None], xs[None, :]]
    )


def _tile_mean_series(Icube: np.ndarray, tile: TileSpec) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Return (tile_series, ys, xs) where tile_series is complex64 shaped (T, tile_count).
    """
    Icube = np.asarray(Icube, dtype=np.complex64)
    if Icube.ndim != 3:
        raise ValueError(f"Icube must have shape (T,H,W), got {Icube.shape}")
    T, H, W = Icube.shape
    ys, xs = _tile_grid(H, W, tile)
    th, tw = int(tile.h), int(tile.w)
    area = np.complex64(float(th * tw))
    tile_count = int(ys.size) * int(xs.size)
    out = np.empty((T, tile_count), dtype=np.complex64)
    for t in range(T):
        sat = _integral_image2d(Icube[t])
        sums = _tile_sums_from_sat(sat, ys, xs, tile).reshape(-1)
        out[t] = (sums / area).astype(np.complex64, copy=False)
    return out, ys, xs


def _tile_flow_coverage(mask: np.ndarray, tile: TileSpec) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    mask = np.asarray(mask, dtype=np.float32)
    if mask.ndim != 2:
        raise ValueError(f"mask must be 2-D, got {mask.shape}")
    H, W = mask.shape
    ys, xs = _tile_grid(H, W, tile)
    sat = _integral_image2d(mask)
    sums = _tile_sums_from_sat(sat, ys, xs, tile).reshape(-1).astype(np.float32, copy=False)
    area = float(int(tile.h) * int(tile.w))
    cov = (sums / max(area, 1.0)).astype(np.float32, copy=False)
    return cov, ys, xs


def _tile_psd_metrics(
    tile_series: np.ndarray,
    *,
    prf_hz: float,
    bands: BandEdges,
    eps: float = 1e-9,
) -> dict[str, np.ndarray]:
    """
    Compute PSD band energies + non-DC peak frequency for each tile series.

    Returns arrays shaped (tile_count,).
    """
    tile_series = np.asarray(tile_series, dtype=np.complex64)
    if tile_series.ndim != 2:
        raise ValueError(f"tile_series must have shape (T,tiles), got {tile_series.shape}")
    T, tile_count = tile_series.shape
    if T < 8:
        raise ValueError("Need T>=8 for stable PSD summaries")
    if prf_hz <= 0.0:
        raise ValueError("prf_hz must be positive")

    # Demean to stabilize DC handling.
    x = tile_series - tile_series.mean(axis=0, keepdims=True)
    win = np.hanning(T).astype(np.float32, copy=False)
    spec = np.fft.fft(x * win[:, None], axis=0)
    psd_full = (spec.conj() * spec).real.astype(np.float32, copy=False)
    freqs_full = np.fft.fftfreq(T, d=1.0 / float(prf_hz))
    nyq = 0.5 * float(prf_hz)
    pos_mask = (freqs_full >= 0.0) | np.isclose(freqs_full, -nyq, atol=1e-6)
    freqs = np.abs(freqs_full[pos_mask]).astype(np.float32, copy=False)
    psd = psd_full[pos_mask, :]

    pf_lo, pf_hi = float(min(bands.pf_lo_hz, bands.pf_hi_hz)), float(max(bands.pf_lo_hz, bands.pf_hi_hz))
    pg_lo, pg_hi = float(min(bands.pg_lo_hz, bands.pg_hi_hz)), float(max(bands.pg_lo_hz, bands.pg_hi_hz))
    pa_lo = float(bands.pa_lo_hz)
    pa_hi = float(nyq)
    mask_pf = (freqs >= pf_lo) & (freqs <= pf_hi)
    mask_pg = (freqs >= pg_lo) & (freqs <= pg_hi)
    mask_pa = (freqs >= pa_lo) & (freqs <= pa_hi)

    Ef = np.sum(psd[mask_pf, :], axis=0) if bool(np.any(mask_pf)) else np.zeros(tile_count, dtype=np.float32)
    Eg = np.sum(psd[mask_pg, :], axis=0) if bool(np.any(mask_pg)) else np.zeros(tile_count, dtype=np.float32)
    Ea = np.sum(psd[mask_pa, :], axis=0) if bool(np.any(mask_pa)) else np.zeros(tile_count, dtype=np.float32)

    zero_idx = int(np.argmin(np.abs(freqs)))
    Eo = psd[zero_idx, :].astype(np.float32, copy=False) if 0 <= zero_idx < psd.shape[0] else np.zeros(tile_count, dtype=np.float32)

    psd_peak = np.array(psd, copy=True)
    if 0 <= zero_idx < psd_peak.shape[0]:
        psd_peak[zero_idx, :] = -np.inf
    peak_bins = np.argmax(psd_peak, axis=0).astype(np.int32, copy=False)
    fpeak = freqs[peak_bins].astype(np.float32, copy=False)

    malias = np.log((Ea.astype(np.float64) + float(eps)) / (Ef.astype(np.float64) + float(eps))).astype(
        np.float32, copy=False
    )

    return {
        "Ef": Ef.astype(np.float32, copy=False),
        "Eg": Eg.astype(np.float32, copy=False),
        "Ea": Ea.astype(np.float32, copy=False),
        "Eo": Eo.astype(np.float32, copy=False),
        "malias": malias,
        "fpeak_hz": fpeak,
    }


def _tile_lag1_coherence(tile_series: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    tile_series = np.asarray(tile_series, dtype=np.complex64)
    if tile_series.ndim != 2:
        raise ValueError("tile_series must be (T,tiles)")
    x = tile_series - tile_series.mean(axis=0, keepdims=True)
    num = np.sum(x[:-1] * np.conj(x[1:]), axis=0)
    den = np.sum(np.abs(x[:-1]) ** 2, axis=0) + float(eps)
    coh = (np.abs(num) / den).astype(np.float32, copy=False)
    return coh


def _quantiles(x: np.ndarray, qs: Iterable[float]) -> dict[str, float]:
    x = np.asarray(x, dtype=np.float64).ravel()
    x = x[np.isfinite(x)]
    if x.size == 0:
        return {f"q{int(round(100*q))}": float("nan") for q in qs}
    out: dict[str, float] = {}
    for q in qs:
        out[f"q{int(round(100*q))}"] = float(np.quantile(x, float(q)))
    return out


def _select_tiles_by_score(score: np.ndarray, mask: np.ndarray, k: int) -> np.ndarray:
    idx = np.flatnonzero(mask)
    if idx.size == 0:
        return idx
    k = max(1, min(int(k), int(idx.size)))
    vals = score[idx]
    order = np.argsort(vals)
    return idx[order[-k:]]


def _tile_svd_summary(
    Icube: np.ndarray,
    *,
    prf_hz: float,
    tile: TileSpec,
    flow_tile_mask: np.ndarray,
    bg_tile_mask: np.ndarray,
    k_tiles: int = 50,
) -> dict[str, Any]:
    """
    Compute a small, deterministic SVD summary on a subset of tiles.

    We compute SVD of each selected tile's Casorati matrix X (T, th*tw) and
    summarize normalized singular value energy.
    """
    Icube = np.asarray(Icube, dtype=np.complex64)
    T, H, W = Icube.shape
    ys, xs = _tile_grid(H, W, tile)
    th, tw = int(tile.h), int(tile.w)
    tile_count = int(ys.size) * int(xs.size)
    flow_tile_mask = np.asarray(flow_tile_mask, dtype=bool).reshape(-1)
    bg_tile_mask = np.asarray(bg_tile_mask, dtype=bool).reshape(-1)
    if flow_tile_mask.size != tile_count or bg_tile_mask.size != tile_count:
        raise ValueError("tile masks must match tile_count")

    # Deterministic tile "score" for selection: mean PD over the tile.
    pd = np.mean(np.abs(Icube) ** 2, axis=0).astype(np.float32, copy=False)
    sat_pd = _integral_image2d(pd)
    pd_tile = (_tile_sums_from_sat(sat_pd, ys, xs, tile) / float(th * tw)).reshape(-1).astype(np.float32, copy=False)

    idx_flow = _select_tiles_by_score(pd_tile, flow_tile_mask, k_tiles)
    idx_bg = _select_tiles_by_score(-pd_tile, bg_tile_mask, k_tiles)  # lowest PD in bg

    def _tile_matrix(idx_flat: int) -> np.ndarray:
        yi = int(idx_flat // int(xs.size))
        xi = int(idx_flat % int(xs.size))
        y0 = int(ys[yi])
        x0 = int(xs[xi])
        block = Icube[:, y0 : y0 + th, x0 : x0 + tw].reshape(T, th * tw)
        # Remove per-pixel mean to keep clutter/DC from dominating.
        return (block - block.mean(axis=0, keepdims=True)).astype(np.complex64, copy=False)

    def _svd_energy(idx_list: np.ndarray) -> np.ndarray:
        if idx_list.size == 0:
            return np.zeros((0, min(T, th * tw)), dtype=np.float32)
        energies: list[np.ndarray] = []
        for idx in idx_list.tolist():
            X = _tile_matrix(int(idx))
            try:
                s = np.linalg.svd(X, compute_uv=False)
            except np.linalg.LinAlgError:
                continue
            s2 = (np.asarray(s, dtype=np.float64) ** 2)
            denom = float(np.sum(s2) + 1e-12)
            energies.append((s2 / denom).astype(np.float32, copy=False))
        if not energies:
            return np.zeros((0, min(T, th * tw)), dtype=np.float32)
        # Pad to common length for summary; trim to min rank across successful tiles.
        rmin = min(e.size for e in energies)
        stack = np.stack([e[:rmin] for e in energies], axis=0)
        return stack.astype(np.float32, copy=False)

    e_flow = _svd_energy(idx_flow)
    e_bg = _svd_energy(idx_bg)

    def _cum_summary(e: np.ndarray) -> dict[str, Any]:
        if e.size == 0:
            return {"n_tiles": 0}
        cum = np.cumsum(e, axis=1)
        # Report median cumulative energy at a few ranks.
        ranks = [1, 2, 5, 10]
        out: dict[str, Any] = {"n_tiles": int(e.shape[0]), "rank_len": int(e.shape[1])}
        for r in ranks:
            rr = min(int(r), int(cum.shape[1]))
            out[f"cum_r{r}"] = float(np.median(cum[:, rr - 1]))
        # Also include a short median spectrum for plotting (first 16 bins).
        m = np.median(e, axis=0)
        out["median_energy_first16"] = [float(v) for v in m[: min(16, m.size)]]
        return out

    return {
        "prf_hz": float(prf_hz),
        "tile_hw": [int(tile.h), int(tile.w)],
        "tile_stride": int(tile.stride),
        "select_k": int(k_tiles),
        "flow": _cum_summary(e_flow),
        "bg": _cum_summary(e_bg),
    }


def _derive_masks_from_pd(
    Icube: np.ndarray,
    *,
    vessel_q: float,
    bg_q: float,
) -> tuple[np.ndarray, np.ndarray, dict[str, float]]:
    pd = np.mean(np.abs(Icube) ** 2, axis=0).astype(np.float32, copy=False)
    v_thr = float(np.quantile(pd, float(vessel_q)))
    b_thr = float(np.quantile(pd, float(bg_q)))
    mask_flow = pd >= v_thr
    mask_bg = (pd <= b_thr) & (~mask_flow)
    return mask_flow.astype(bool), mask_bg.astype(bool), {"v_thr": v_thr, "b_thr": b_thr}


def summarize_icube(
    *,
    name: str,
    Icube: np.ndarray,
    prf_hz: float,
    tile: TileSpec,
    bands: BandEdges,
    mask_flow: np.ndarray | None,
    mask_bg: np.ndarray | None,
    derive_masks: bool,
    derive_vessel_q: float,
    derive_bg_q: float,
    eps: float = 1e-9,
) -> dict[str, Any]:
    Icube = np.asarray(Icube, dtype=np.complex64)
    T, H, W = Icube.shape

    meta: dict[str, Any] = {
        "name": str(name),
        "shape": [int(T), int(H), int(W)],
        "prf_hz": float(prf_hz),
        "nyquist_hz": float(0.5 * float(prf_hz)),
        "tile_hw": [int(tile.h), int(tile.w)],
        "tile_stride": int(tile.stride),
        "bands_hz": {
            "Pf": [float(bands.pf_lo_hz), float(bands.pf_hi_hz)],
            "Pg": [float(bands.pg_lo_hz), float(bands.pg_hi_hz)],
            "Pa": [float(bands.pa_lo_hz), float(0.5 * float(prf_hz))],
        },
    }

    if mask_flow is None or mask_bg is None:
        if not derive_masks:
            raise ValueError("mask_flow/mask_bg not provided and derive_masks is False")
        mask_flow, mask_bg, thr = _derive_masks_from_pd(Icube, vessel_q=derive_vessel_q, bg_q=derive_bg_q)
        meta["derived_masks"] = {"method": "pd_quantiles", **thr, "vessel_q": float(derive_vessel_q), "bg_q": float(derive_bg_q)}
    else:
        mask_flow = np.asarray(mask_flow, dtype=bool)
        mask_bg = np.asarray(mask_bg, dtype=bool)
        if mask_flow.shape != (H, W) or mask_bg.shape != (H, W):
            raise ValueError("mask shapes must match Icube spatial dims")

    series, _ys, _xs = _tile_mean_series(Icube, tile)
    cov, _ys2, _xs2 = _tile_flow_coverage(mask_flow.astype(np.float32), tile)
    bg_cov, _, _ = _tile_flow_coverage(mask_bg.astype(np.float32), tile)
    # Convert bg mask to "coverage" by marking bg pixels as 1; use the fraction as a bg-coverage proxy.
    bg_cov = bg_cov.astype(np.float32, copy=False)

    flow_tiles = cov >= 0.20
    bg_tiles = bg_cov >= 0.80
    if int(bg_tiles.sum()) < 10:
        # Fall back to low-flow-coverage tiles.
        bg_tiles = cov <= 0.05

    psd = _tile_psd_metrics(series, prf_hz=float(prf_hz), bands=bands, eps=float(eps))
    coh1 = _tile_lag1_coherence(series)

    def _group_summary(mask: np.ndarray) -> dict[str, Any]:
        mask = np.asarray(mask, dtype=bool).reshape(-1)
        out: dict[str, Any] = {"n_tiles": int(mask.sum())}
        if not bool(mask.any()):
            return out
        qs = [0.10, 0.25, 0.50, 0.75, 0.90, 0.95, 0.99]
        for key in ("Ef", "Eg", "Ea", "Eo", "malias", "fpeak_hz"):
            out[key] = _quantiles(psd[key][mask], qs)
        out["coh1"] = _quantiles(coh1[mask], qs)
        out["flow_coverage"] = _quantiles(cov[mask], qs)
        return out

    svd = _tile_svd_summary(
        Icube,
        prf_hz=float(prf_hz),
        tile=tile,
        flow_tile_mask=flow_tiles,
        bg_tile_mask=bg_tiles,
        k_tiles=50,
    )

    return {
        "meta": meta,
        "tile_sets": {
            "flow_tiles": {"method": "flow_coverage>=0.20", "n": int(flow_tiles.sum())},
            "bg_tiles": {"method": "bg_mask>=0.80 (fallback: flow_coverage<=0.05)", "n": int(bg_tiles.sum())},
        },
        "summary": {
            "flow": _group_summary(flow_tiles),
            "bg": _group_summary(bg_tiles),
        },
        "svd": svd,
    }


def _load_sim_run(run_dir: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict[str, Any]]:
    run_dir = Path(run_dir)
    meta_path = run_dir / "dataset" / "meta.json"
    icube_path = run_dir / "dataset" / "icube.npy"
    mask_flow_path = run_dir / "dataset" / "mask_flow.npy"
    mask_bg_path = run_dir / "dataset" / "mask_bg.npy"
    if not meta_path.is_file():
        raise FileNotFoundError(f"Missing {meta_path}")
    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    Icube = np.load(icube_path).astype(np.complex64, copy=False)
    mask_flow = np.load(mask_flow_path).astype(bool, copy=False)
    mask_bg = np.load(mask_bg_path).astype(bool, copy=False)
    return Icube, mask_flow, mask_bg, meta


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Sanity-link telemetry for physical Doppler simulation vs real IQ.")
    ap.add_argument("--out-dir", type=Path, default=Path("reports/physdoppler_sanity_link"), help="Output directory.")
    ap.add_argument("--tag", type=str, default=None, help="Optional output tag (default: derived).")

    ap.add_argument("--pf", type=float, nargs=2, default=(30.0, 250.0), metavar=("PF_LO", "PF_HI"))
    ap.add_argument("--pg", type=float, nargs=2, default=(250.0, 400.0), metavar=("PG_LO", "PG_HI"))
    ap.add_argument("--pa-lo", type=float, default=400.0, help="Pa lower edge in Hz (upper edge is Nyquist).")
    ap.add_argument("--tile-hw", type=int, nargs=2, default=(8, 8), metavar=("H", "W"))
    ap.add_argument("--tile-stride", type=int, default=3)

    ap.add_argument("--derive-masks", action=argparse.BooleanOptionalAction, default=True)
    ap.add_argument("--derive-vessel-q", type=float, default=0.99)
    ap.add_argument("--derive-bg-q", type=float, default=0.20)

    # Inputs
    ap.add_argument("--sim-run", type=Path, default=None, help="Physical simulator run dir (contains dataset/icube.npy).")
    ap.add_argument("--shin-root", type=Path, default=None, help="Root for Shin Fig3 raw (contains SizeInfo.dat).")
    ap.add_argument("--shin-iq-file", type=str, default="IQData001.dat")
    ap.add_argument("--shin-frames", type=str, default="0:128", help="Frame indices spec (default: 0:128).")
    ap.add_argument("--shin-prf-hz", type=float, default=1000.0)

    ap.add_argument("--gammex-seq-root", type=Path, default=None, help="Root containing Gammex along/across dirs.")
    ap.add_argument("--gammex-frames-along", type=str, default="0", help="Frame indices spec for along view.")
    ap.add_argument("--gammex-frames-across", type=str, default="0", help="Frame indices spec for across view.")
    ap.add_argument("--gammex-prf-hz", type=float, default=2500.0)
    ap.add_argument(
        "--gammex-mask-mode",
        type=str,
        default="bmode_tube",
        choices=["none", "bmode_tube"],
        help="If bmode_tube, build structural masks from B-mode ROI.",
    )
    ap.add_argument("--gammex-mask-ref-frames", type=str, default="0:6")
    return ap.parse_args()


def main() -> None:
    args = parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    bands = BandEdges(
        pf_lo_hz=float(args.pf[0]),
        pf_hi_hz=float(args.pf[1]),
        pg_lo_hz=float(args.pg[0]),
        pg_hi_hz=float(args.pg[1]),
        pa_lo_hz=float(args.pa_lo),
    )
    tile = TileSpec(h=int(args.tile_hw[0]), w=int(args.tile_hw[1]), stride=int(args.tile_stride))

    reports: dict[str, Any] = {
        "schema_version": "physdoppler_sanity_link.v1",
        "bands_hz": dataclass_to_dict(bands),
        "tile": dataclass_to_dict(tile),
        "inputs": {},
        "summaries": {},
    }

    tag_parts: list[str] = []
    if args.sim_run is not None:
        sim_run = Path(args.sim_run)
        Icube, mask_flow, mask_bg, meta = _load_sim_run(sim_run)
        prf = float(meta.get("acquisition", {}).get("prf_hz", meta.get("config", {}).get("prf_hz", 0.0)))
        if prf <= 0.0:
            raise ValueError("sim-run meta missing prf_hz")
        name = f"sim_{_slugify(sim_run.name)}"
        reports["inputs"]["sim_run"] = str(sim_run)
        reports["summaries"][name] = summarize_icube(
            name=name,
            Icube=Icube,
            prf_hz=prf,
            tile=tile,
            bands=bands,
            mask_flow=mask_flow,
            mask_bg=mask_bg,
            derive_masks=False,
            derive_vessel_q=float(args.derive_vessel_q),
            derive_bg_q=float(args.derive_bg_q),
        )
        tag_parts.append(name)

    if args.shin_root is not None:
        root = Path(args.shin_root)
        info = load_shin_metadata(root)
        iq_path = root / str(args.shin_iq_file)
        frames = _parse_indices(str(args.shin_frames), int(info.frames))
        Icube = load_shin_iq(iq_path, info, frames=frames)
        name = f"shin_{_slugify(Path(args.shin_iq_file).stem)}_{_slugify(args.shin_frames)}"
        reports["inputs"]["shin_root"] = str(root)
        reports["summaries"][name] = summarize_icube(
            name=name,
            Icube=Icube,
            prf_hz=float(args.shin_prf_hz),
            tile=tile,
            bands=bands,
            mask_flow=None,
            mask_bg=None,
            derive_masks=bool(args.derive_masks),
            derive_vessel_q=float(args.derive_vessel_q),
            derive_bg_q=float(args.derive_bg_q),
        )
        tag_parts.append(name)

    if args.gammex_seq_root is not None:
        seq_root = Path(args.gammex_seq_root)
        along_dir = seq_root / "Flow in Gammex phantom (along - linear probe)"
        across_dir = seq_root / "Flow in Gammex phantom (across - linear probe)"

        def _run_view(
            *,
            view: Literal["along", "across"],
            seq_dir: Path,
            par_path: Path,
            dat_path: Path,
            frames_spec: str,
        ) -> None:
            par = RawBCFPar.from_dict(parse_rawbcf_par(par_path))
            par.validate()
            frame_indices = _parse_indices(frames_spec, int(par.num_frames))
            if not frame_indices:
                return
            mask_flow = None
            mask_bg = None
            if str(args.gammex_mask_mode).strip().lower() == "bmode_tube":
                ref_frames = _parse_indices(str(args.gammex_mask_ref_frames), int(par.num_frames))
                tube = build_tube_masks_from_rawbcf(dat_path, par, ref_frame_indices=ref_frames)
                mask_flow = tube.mask_flow
                mask_bg = tube.mask_bg

            for idx in frame_indices:
                frame = read_rawbcf_frame(dat_path, par, int(idx))
                Icube = decode_rawbcf_cfm_cube(frame, par, order="beam_major")
                name = f"gammex_{view}_frame{int(idx):03d}"
                reports["summaries"][name] = summarize_icube(
                    name=name,
                    Icube=Icube,
                    prf_hz=float(args.gammex_prf_hz),
                    tile=tile,
                    bands=bands,
                    mask_flow=mask_flow,
                    mask_bg=mask_bg,
                    derive_masks=bool(args.derive_masks) if (mask_flow is None or mask_bg is None) else False,
                    derive_vessel_q=float(args.derive_vessel_q),
                    derive_bg_q=float(args.derive_bg_q),
                )
                tag_parts.append(name)

        reports["inputs"]["gammex_seq_root"] = str(seq_root)
        _run_view(
            view="along",
            seq_dir=along_dir,
            par_path=along_dir / "RawBCFCine.par",
            dat_path=along_dir / "RawBCFCine.dat",
            frames_spec=str(args.gammex_frames_along),
        )
        across_par = across_dir / "RawBCFCine_08062017_145434_17.par"
        across_dat = across_dir / "RawBCFCine_08062017_145434_17.dat"
        _run_view(
            view="across",
            seq_dir=across_dir,
            par_path=across_par,
            dat_path=across_dat,
            frames_spec=str(args.gammex_frames_across),
        )

    if not reports["summaries"]:
        raise SystemExit("No inputs specified; pass --sim-run and/or --shin-root and/or --gammex-seq-root.")

    tag = str(args.tag).strip() if args.tag else _slugify("__".join(tag_parts)[:120])
    out_json = out_dir / f"{tag}_summary.json"
    _write_json(out_json, reports)

    # Save a compact per-dataset table as CSV-like JSON for quick diffs.
    rows: list[dict[str, Any]] = []
    for k, v in (reports.get("summaries") or {}).items():
        meta = (v.get("meta") or {}) if isinstance(v, dict) else {}
        flow = ((v.get("summary") or {}).get("flow") or {}) if isinstance(v, dict) else {}
        bg = ((v.get("summary") or {}).get("bg") or {}) if isinstance(v, dict) else {}
        rows.append(
            {
                "key": str(k),
                "T": int(meta.get("shape", [0])[0]) if isinstance(meta.get("shape"), list) else None,
                "H": int(meta.get("shape", [0, 0])[1]) if isinstance(meta.get("shape"), list) else None,
                "W": int(meta.get("shape", [0, 0, 0])[2]) if isinstance(meta.get("shape"), list) else None,
                "prf_hz": meta.get("prf_hz"),
                "flow_tiles": (flow.get("n_tiles") if isinstance(flow, dict) else None),
                "bg_tiles": (bg.get("n_tiles") if isinstance(bg, dict) else None),
                "flow_malias_q50": ((flow.get("malias") or {}).get("q50") if isinstance(flow, dict) else None),
                "bg_malias_q50": ((bg.get("malias") or {}).get("q50") if isinstance(bg, dict) else None),
                "flow_fpeak_q50": ((flow.get("fpeak_hz") or {}).get("q50") if isinstance(flow, dict) else None),
                "bg_fpeak_q50": ((bg.get("fpeak_hz") or {}).get("q50") if isinstance(bg, dict) else None),
                "flow_coh1_q50": ((flow.get("coh1") or {}).get("q50") if isinstance(flow, dict) else None),
                "bg_coh1_q50": ((bg.get("coh1") or {}).get("q50") if isinstance(bg, dict) else None),
            }
        )
    _write_json(out_dir / f"{tag}_table.json", {"rows": rows})

    print(f"[physdoppler-sanity] wrote {out_json}")
    print(f"[physdoppler-sanity] wrote {out_dir / f'{tag}_table.json'}")


def dataclass_to_dict(obj: Any) -> dict[str, Any]:
    if hasattr(obj, "__dict__"):
        return {k: dataclass_to_dict(v) for k, v in vars(obj).items()}
    return obj  # type: ignore[return-value]


if __name__ == "__main__":
    main()
