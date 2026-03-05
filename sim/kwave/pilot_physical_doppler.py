from __future__ import annotations

import argparse
import dataclasses
import hashlib
import json
import platform
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Dict

import numpy as np

from sim.kwave.icube_bundle import write_acceptance_bundle_from_icube
from sim.kwave.physical_doppler import (
    PhysicalDopplerConfig,
    PsfSpec,
    VesselSpec,
    dataset_meta,
    default_brainlike_config,
    generate_icube,
)


def _utc_now_iso() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def _sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _git_info(repo_root: Path) -> dict[str, Any]:
    try:
        commit = (
            subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=str(repo_root))
            .decode("utf-8")
            .strip()
        )
        dirty = subprocess.call(
            ["git", "diff", "--quiet"], cwd=str(repo_root), stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL
        )
        dirty_cached = subprocess.call(
            ["git", "diff", "--quiet", "--cached"],
            cwd=str(repo_root),
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        is_dirty = bool(dirty != 0 or dirty_cached != 0)
        return {"commit": commit, "dirty": is_dirty}
    except Exception:
        return {"commit": None, "dirty": None}


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, sort_keys=True)
        f.write("\n")


def _sanitize_name(s: str) -> str:
    allowed = []
    for ch in str(s):
        if ch.isalnum() or ch in {"_", "-", "."}:
            allowed.append(ch)
        else:
            allowed.append("_")
    out = "".join(allowed).strip("_")
    return out or "run"


def _integral_image2d(frame: np.ndarray) -> np.ndarray:
    """Summed-area table with a 1px zero pad (so window sums are branch-free)."""
    H, W = frame.shape
    sat = np.zeros((H + 1, W + 1), dtype=frame.dtype)
    sat[1:, 1:] = frame
    sat = sat.cumsum(axis=0).cumsum(axis=1)
    return sat


def _tile_grid(H: int, W: int, tile_hw: tuple[int, int], stride: int) -> tuple[np.ndarray, np.ndarray]:
    th, tw = tile_hw
    if th <= 0 or tw <= 0:
        raise ValueError("tile_hw must be positive")
    if stride <= 0:
        raise ValueError("tile_stride must be positive")
    ys = np.arange(0, H - th + 1, stride, dtype=np.int32)
    xs = np.arange(0, W - tw + 1, stride, dtype=np.int32)
    return ys, xs


def _tile_sums_from_sat(
    sat: np.ndarray, ys: np.ndarray, xs: np.ndarray, tile_hw: tuple[int, int]
) -> np.ndarray:
    th, tw = tile_hw
    y1 = ys + th
    x1 = xs + tw
    # Shape: (ny_tiles, nx_tiles)
    return (
        sat[y1[:, None], x1[None, :]]
        - sat[ys[:, None], x1[None, :]]
        - sat[y1[:, None], xs[None, :]]
        + sat[ys[:, None], xs[None, :]]
    )


def _compute_alias_qc_tiles(
    *,
    Icube: np.ndarray,
    mask_flow: np.ndarray,
    prf_hz: float,
    tile_hw: tuple[int, int],
    tile_stride: int,
    flow_band_hz: tuple[float, float] = (30.0, 250.0),
    pa_low_hz: float = 400.0,
    q_bg: float = 0.99,
    bg_cov_max: float = 0.05,
    eps: float = 1e-9,
    require_fpeak_in_pa: bool = True,
) -> tuple[np.ndarray, dict[str, Any], dict[str, np.ndarray]]:
    """
    QC-only observed-alias diagnostic at tile level.

    This is intentionally label-free and defined in terms of simple telemetry:
      - tile-mean PSD band energies in Pf vs Pa
      - non-DC peak frequency fpeak
      - robust threshold from background tiles (low flow coverage)
    """
    Icube = np.asarray(Icube, dtype=np.complex64)
    if Icube.ndim != 3:
        raise ValueError(f"Icube must have shape (T,H,W), got {Icube.shape}")
    T, H, W = Icube.shape
    if T < 8 or prf_hz <= 0.0:
        raise ValueError("Need T>=8 and positive prf_hz for alias QC")

    mask_flow = np.asarray(mask_flow, dtype=bool)
    if mask_flow.shape != (H, W):
        raise ValueError(f"mask_flow must have shape {(H, W)}, got {mask_flow.shape}")

    th, tw = tile_hw
    ys, xs = _tile_grid(H, W, tile_hw, tile_stride)
    ny_tiles, nx_tiles = int(ys.size), int(xs.size)
    tile_count = ny_tiles * nx_tiles
    if tile_count <= 0:
        raise ValueError("No tiles for the requested tile grid")

    area = float(th * tw)
    # Flow coverage per tile (fraction of pixels in flow mask).
    sat_flow = _integral_image2d(mask_flow.astype(np.float32, copy=False))
    cov = (_tile_sums_from_sat(sat_flow, ys, xs, tile_hw) / area).astype(np.float32, copy=False).reshape(-1)

    # Tile-mean slow-time series for each tile (T, tile_count).
    tile_series = np.empty((T, tile_count), dtype=np.complex64)
    for t in range(T):
        sat = _integral_image2d(Icube[t])
        sums = _tile_sums_from_sat(sat, ys, xs, tile_hw).reshape(-1)
        tile_series[t] = (sums / np.complex64(area)).astype(np.complex64, copy=False)

    # Remove tile mean to suppress DC leakage and make the "non-DC peak" stable.
    tile_series = (tile_series - tile_series.mean(axis=0, keepdims=True)).astype(np.complex64, copy=False)
    win = np.hanning(T).astype(np.float32, copy=False)
    spec = np.fft.fft(tile_series * win[:, None], axis=0)
    psd_full = (spec.conj() * spec).real
    freqs_full = np.fft.fftfreq(T, d=1.0 / float(prf_hz))
    nyq = 0.5 * float(prf_hz)
    pos_mask = (freqs_full >= 0.0) | np.isclose(freqs_full, -nyq, atol=1e-6)
    freqs = np.abs(freqs_full[pos_mask]).astype(np.float32, copy=False)
    psd = psd_full[pos_mask, :].astype(np.float32, copy=False)

    # Band energies (no attempt to model guard here; QC only).
    f_lo, f_hi = float(min(flow_band_hz)), float(max(flow_band_hz))
    pa_hi = float(nyq)
    pa_lo = float(pa_low_hz)
    mask_f = (freqs >= f_lo) & (freqs <= f_hi)
    mask_a = (freqs >= pa_lo) & (freqs <= pa_hi)
    Ef = np.sum(psd[mask_f, :], axis=0) if np.any(mask_f) else np.zeros(tile_count, dtype=np.float32)
    Ea = np.sum(psd[mask_a, :], axis=0) if np.any(mask_a) else np.zeros(tile_count, dtype=np.float32)
    malias = np.log((Ea.astype(np.float64) + float(eps)) / (Ef.astype(np.float64) + float(eps))).astype(
        np.float32, copy=False
    )

    # Non-DC peak frequency (absolute Hz).
    psd_peak = np.array(psd, dtype=np.float32, copy=True)
    zero_idx = int(np.argmin(np.abs(freqs)))
    if 0 <= zero_idx < psd_peak.shape[0]:
        psd_peak[zero_idx, :] = -np.inf
    peak_bins = np.argmax(psd_peak, axis=0).astype(np.int32, copy=False)
    fpeak_hz = freqs[peak_bins].astype(np.float32, copy=False)
    fpeak_in_pa = (fpeak_hz >= pa_lo) & (fpeak_hz <= pa_hi)

    # Background tiles from coverage threshold; if too few, fall back to lowest-coverage decile.
    cov_thresh = float(np.clip(bg_cov_max, 0.0, 1.0))
    bg_tiles = cov <= cov_thresh
    bg_method = "coverage<=threshold"
    if int(bg_tiles.sum()) < 10:
        n = max(10, int(round(0.10 * tile_count)))
        idx = np.argsort(cov)[:n]
        bg_tiles = np.zeros(tile_count, dtype=bool)
        bg_tiles[idx] = True
        bg_method = "lowest_coverage_decile"

    q = float(np.clip(q_bg, 0.0, 1.0))
    tau = float(np.quantile(malias[bg_tiles], q)) if bool(bg_tiles.any()) else float(np.quantile(malias, q))

    obs = malias >= tau
    if require_fpeak_in_pa:
        obs &= fpeak_in_pa

    # Flow-like tiles for summary only.
    flow_tiles = cov >= 0.20
    flow_method = "coverage>=0.20"
    if int(flow_tiles.sum()) < 10:
        n = max(10, int(round(0.10 * tile_count)))
        idx = np.argsort(cov)[-n:]
        flow_tiles = np.zeros(tile_count, dtype=bool)
        flow_tiles[idx] = True
        flow_method = "highest_coverage_decile"

    diag: dict[str, Any] = {
        "tile_hw": [int(th), int(tw)],
        "tile_stride": int(tile_stride),
        "shape": [int(H), int(W)],
        "T": int(T),
        "prf_hz": float(prf_hz),
        "bands_hz": {"flow": [float(f_lo), float(f_hi)], "Pa": [float(pa_lo), float(pa_hi)]},
        "bg": {"method": bg_method, "cov_max": float(cov_thresh), "q": float(q), "tau_malias": float(tau)},
        "require_fpeak_in_pa": bool(require_fpeak_in_pa),
        "flow_summary_tiles": {"method": flow_method},
        "summary": {
            "tile_count": int(tile_count),
            "bg_tiles": int(bg_tiles.sum()),
            "flow_tiles": int(flow_tiles.sum()),
            "obs_frac_bg": float(obs[bg_tiles].mean()) if bool(bg_tiles.any()) else None,
            "obs_frac_flow": float(obs[flow_tiles].mean()) if bool(flow_tiles.any()) else None,
            "cov_min": float(cov.min()),
            "cov_median": float(np.median(cov)),
            "cov_max": float(cov.max()),
        },
    }
    debug = {
        "flow_coverage_tiles": cov.astype(np.float32, copy=False),
        "malias_tiles": malias.astype(np.float32, copy=False),
        "fpeak_tiles_hz": fpeak_hz.astype(np.float32, copy=False),
    }
    return obs.astype(bool, copy=False), diag, debug


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Generate physical Doppler surrogate Icube + derived acceptance bundle.")
    ap.add_argument("--out", type=Path, required=True, help="Output run directory (will be created).")
    ap.add_argument(
        "--preset",
        type=str,
        default="microvascular_like",
        choices=["microvascular_like", "alias_stress"],
        help="Simulation preset (geometry + vmax).",
    )
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument(
        "--tier",
        type=str,
        default="smoke",
        choices=["smoke", "paper"],
        help="Runtime tier: small fast config vs Brain-aligned paper config.",
    )
    ap.add_argument("--prf-hz", type=float, default=None, help="Override PRF (Hz).")
    ap.add_argument("--f0-hz", type=float, default=None, help="Override transmit center frequency (Hz).")
    ap.add_argument("--c0", type=float, default=None, help="Override sound speed (m/s).")
    ap.add_argument("--nx", type=int, default=None, help="Override lateral size (pixels).")
    ap.add_argument("--ny", type=int, default=None, help="Override depth size (pixels).")
    ap.add_argument("--dx", type=float, default=None, help="Override dx (m).")
    ap.add_argument("--dy", type=float, default=None, help="Override dy (m).")
    ap.add_argument("--pulses-per-set", type=int, default=None, help="Override pulses per set.")
    ap.add_argument("--ensembles", type=int, default=None, help="Override ensemble count.")
    ap.add_argument("--blood-to-tissue-power-db", type=float, default=-15.0)
    ap.add_argument("--psf-sigma-x0-px", type=float, default=1.25, help="PSF sigma_x at z=0 (pixels).")
    ap.add_argument(
        "--psf-sigma-x-px",
        dest="psf_sigma_x0_px",
        type=float,
        default=argparse.SUPPRESS,
        help="Deprecated alias for --psf-sigma-x0-px.",
    )
    ap.add_argument("--psf-sigma-z0-px", type=float, default=1.75, help="PSF sigma_z at z=0 (pixels).")
    ap.add_argument(
        "--psf-sigma-z-px",
        dest="psf_sigma_z0_px",
        type=float,
        default=argparse.SUPPRESS,
        help="Deprecated alias for --psf-sigma-z0-px.",
    )
    ap.add_argument("--psf-alpha-x-per-m", type=float, default=0.0, help="Depth slope for sigma_x (px/m).")
    ap.add_argument("--psf-alpha-z-per-m", type=float, default=0.0, help="Depth slope for sigma_z (px/m).")
    ap.add_argument("--psf-calib-path", type=Path, default=None, help="Optional JSON calibration file to load.")
    ap.add_argument("--psf-mode", type=str, default="nearest", help="Boundary mode for PSF filtering.")
    ap.add_argument("--noise-snr-db", type=float, default=25.0)
    ap.add_argument(
        "--reinjection-mode",
        type=str,
        default="reservoir",
        choices=["reservoir", "wrap"],
    )
    ap.add_argument("--pad-x-px", type=int, default=None, help="Override reinjection padding in x (pixels).")
    ap.add_argument("--pad-z-px", type=int, default=None, help="Override reinjection padding in z (pixels).")
    ap.add_argument("--reservoir-scale", type=int, default=4)
    ap.add_argument("--reservoir-seed", type=int, default=None, help="Override reservoir RNG seed.")
    ap.add_argument("--reservoir-step-x-px", type=int, default=137, help="Reservoir x offset step (pixels).")
    ap.add_argument("--reservoir-step-z-px", type=int, default=251, help="Reservoir z offset step (pixels).")
    ap.add_argument(
        "--reservoir-step-px",
        dest="reservoir_step_px",
        type=int,
        default=None,
        help="Deprecated: set both reservoir step-x and step-z to the same value.",
    )
    ap.add_argument(
        "--reservoir-offset-init",
        type=int,
        nargs=2,
        default=None,
        metavar=("OX0", "OZ0"),
        help="Optional initial reservoir offsets (pixels).",
    )
    ap.add_argument("--vessel-offset-kx", type=int, default=193, help="Per-vessel decorrelation offset in x.")
    ap.add_argument("--vessel-offset-kz", type=int, default=97, help="Per-vessel decorrelation offset in z.")

    ap.add_argument(
        "--skip-bundle",
        action="store_true",
        help="Only write dataset/ (icube + masks + meta), skip derived acceptance bundle.",
    )
    ap.add_argument(
        "--dataset-name",
        type=str,
        default=None,
        help="Optional acceptance-bundle dataset name (defaults to a sanitized name derived from config).",
    )
    ap.add_argument(
        "--stap-conditional",
        action="store_true",
        help="Enable conditional STAP in derived bundle (default off for clarity).",
    )
    ap.add_argument(
        "--no-alias-qc",
        action="store_true",
        help="Disable QC-only observed-alias diagnostic outputs (alias_observed_tile.npy, alias_diag.json).",
    )
    ap.add_argument("--alias-q-bg", type=float, default=0.99, help="Background quantile for tau_alias.")
    ap.add_argument("--alias-bg-cov-max", type=float, default=0.05, help="Max flow coverage for background tiles.")
    ap.add_argument("--alias-eps", type=float, default=1e-9, help="Stability epsilon for log band-energy ratio.")
    ap.add_argument(
        "--alias-no-fpeak-gate",
        action="store_true",
        help="Do not require non-DC fpeak to fall in Pa for observed-alias tiles (QC only).",
    )
    return ap.parse_args()


def _tier_defaults(tier: str) -> dict[str, Any]:
    if tier == "paper":
        return {
            "Nx": 240,
            "Ny": 240,
            "dx": 90e-6,
            "dy": 90e-6,
            "prf_hz": 1500.0,
            "pulses_per_set": 64,
            "ensembles": 5,
            "f0_hz": 7.5e6,
            "c0": 1540.0,
        }
    # smoke
    return {
        "Nx": 96,
        "Ny": 96,
        "dx": 90e-6,
        "dy": 90e-6,
        "prf_hz": 1500.0,
        "pulses_per_set": 16,
        "ensembles": 2,
        "f0_hz": 7.5e6,
        "c0": 1540.0,
    }


def _build_cfg(args: argparse.Namespace) -> PhysicalDopplerConfig:
    td = _tier_defaults(str(args.tier))
    kw = dict(td)
    for k_src, k_dst in [
        ("nx", "Nx"),
        ("ny", "Ny"),
        ("dx", "dx"),
        ("dy", "dy"),
        ("prf_hz", "prf_hz"),
        ("pulses_per_set", "pulses_per_set"),
        ("ensembles", "ensembles"),
        ("f0_hz", "f0_hz"),
        ("c0", "c0"),
    ]:
        v = getattr(args, k_src)
        if v is not None:
            kw[k_dst] = v

    cfg = default_brainlike_config(
        preset=str(args.preset),
        seed=int(args.seed),
        Nx=int(kw["Nx"]),
        Ny=int(kw["Ny"]),
        dx=float(kw["dx"]),
        dy=float(kw["dy"]),
        prf_hz=float(kw["prf_hz"]),
        pulses_per_set=int(kw["pulses_per_set"]),
        ensembles=int(kw["ensembles"]),
        f0_hz=float(kw["f0_hz"]),
        c0=float(kw["c0"]),
    )
    step_x = int(args.reservoir_step_x_px)
    step_z = int(args.reservoir_step_z_px)
    if getattr(args, "reservoir_step_px", None) is not None:
        step_x = int(args.reservoir_step_px)
        step_z = int(args.reservoir_step_px)
    cfg = dataclasses.replace(
        cfg,
        blood_to_tissue_power_db=float(args.blood_to_tissue_power_db),
        reinjection_mode=str(args.reinjection_mode),
        pad_x_px=int(args.pad_x_px) if args.pad_x_px is not None else cfg.pad_x_px,
        pad_z_px=int(args.pad_z_px) if args.pad_z_px is not None else cfg.pad_z_px,
        reservoir_scale=int(args.reservoir_scale),
        reservoir_seed=int(args.reservoir_seed) if args.reservoir_seed is not None else cfg.reservoir_seed,
        reservoir_step_x_px=int(step_x),
        reservoir_step_z_px=int(step_z),
        reservoir_offset_init=(
            (int(args.reservoir_offset_init[0]), int(args.reservoir_offset_init[1]))
            if args.reservoir_offset_init is not None
            else None
        ),
        vessel_offset_kx=int(args.vessel_offset_kx),
        vessel_offset_kz=int(args.vessel_offset_kz),
        psf=PsfSpec(
            sigma_x0_px=float(args.psf_sigma_x0_px),
            sigma_z0_px=float(args.psf_sigma_z0_px),
            alpha_x_per_m=float(args.psf_alpha_x_per_m),
            alpha_z_per_m=float(args.psf_alpha_z_per_m),
            calib_path=str(args.psf_calib_path) if args.psf_calib_path is not None else None,
            mode=str(args.psf_mode),
        ),
        noise=dataclasses.replace(cfg.noise, snr_db=float(args.noise_snr_db)),
    )
    return cfg


def main() -> None:
    args = parse_args()
    out_root = Path(args.out)
    out_root.mkdir(parents=True, exist_ok=True)
    dataset_dir = out_root / "dataset"
    debug_dir = dataset_dir / "debug"
    bundle_root = out_root / "bundle"
    dataset_dir.mkdir(parents=True, exist_ok=True)
    debug_dir.mkdir(parents=True, exist_ok=True)
    bundle_root.mkdir(parents=True, exist_ok=True)

    cfg = _build_cfg(args)

    result = generate_icube(cfg)
    icube = result["Icube"]
    mask_flow = result["mask_flow"]
    mask_bg = result["mask_bg"]
    mask_alias = result["mask_alias_expected"]
    debug = result["debug"]

    # ---- Write canonical dataset artifacts ----
    paths: Dict[str, Path] = {}

    def _save(dst_dir: Path, name: str, arr: np.ndarray) -> None:
        p = dst_dir / f"{name}.npy"
        np.save(p, arr, allow_pickle=False)
        paths[name] = p

    _save(dataset_dir, "icube", icube)
    _save(dataset_dir, "mask_flow", mask_flow.astype(bool))
    _save(dataset_dir, "mask_bg", mask_bg.astype(bool))
    _save(dataset_dir, "mask_alias_expected", mask_alias.astype(bool))

    _save(debug_dir, "vx_mps", np.asarray(debug.get("vx_mps"), dtype=np.float32))
    _save(debug_dir, "vz_mps", np.asarray(debug.get("vz_mps"), dtype=np.float32))
    _save(debug_dir, "fd_expected_hz", np.asarray(debug.get("fd_expected_hz"), dtype=np.float32))

    alias_qc_diag: dict[str, Any] | None = None
    if not bool(args.no_alias_qc):
        try:
            obs_tile, diag, qc_debug = _compute_alias_qc_tiles(
                Icube=icube,
                mask_flow=mask_flow,
                prf_hz=float(cfg.prf_hz),
                tile_hw=(8, 8),
                tile_stride=3,
                flow_band_hz=(30.0, 250.0),
                pa_low_hz=400.0,
                q_bg=float(args.alias_q_bg),
                bg_cov_max=float(args.alias_bg_cov_max),
                eps=float(args.alias_eps),
                require_fpeak_in_pa=not bool(args.alias_no_fpeak_gate),
            )
            _save(dataset_dir, "alias_observed_tile", obs_tile.astype(bool, copy=False))
            _save(debug_dir, "alias_flow_coverage_tiles", qc_debug["flow_coverage_tiles"])
            _save(debug_dir, "alias_malias_tiles", qc_debug["malias_tiles"])
            _save(debug_dir, "alias_fpeak_tiles_hz", qc_debug["fpeak_tiles_hz"])
            alias_diag_path = debug_dir / "alias_diag.json"
            _write_json(alias_diag_path, diag)
            paths["alias_diag_json"] = alias_diag_path
            alias_qc_diag = diag
        except Exception as e:
            alias_diag_path = debug_dir / "alias_diag.json"
            diag = {"error": str(e)}
            _write_json(alias_diag_path, diag)
            paths["alias_diag_json"] = alias_diag_path
            alias_qc_diag = diag

    config_path = dataset_dir / "config.json"
    _write_json(config_path, dataset_meta(cfg))

    hashes: dict[str, Any] = {}
    for k, p in sorted(paths.items()):
        h = _sha256_file(p)
        info = {"sha256": h, "path": str(p.relative_to(out_root))}
        if k == "icube":
            info.update({"dtype": str(icube.dtype), "shape": list(icube.shape)})
        hashes[k] = info
    hashes["config"] = {"sha256": _sha256_file(config_path), "path": str(config_path.relative_to(out_root))}
    _write_json(dataset_dir / "hashes.json", hashes)

    repo_root = Path(__file__).resolve().parents[2]
    prov: dict[str, Any] = {
        "command": " ".join([_sanitize_name(x) if i == 0 else x for i, x in enumerate(sys.argv)]),
        "cwd": str(Path.cwd()),
        "git": _git_info(repo_root),
        "python": sys.version.splitlines()[0],
        "platform": platform.platform(),
        "numpy": np.__version__,
    }
    # SciPy is used by the Phase 1 generator (map_coordinates, gaussian_filter).
    try:
        import scipy  # type: ignore

        prov["scipy"] = getattr(scipy, "__version__", None)
    except Exception:
        prov["scipy"] = None
    # Torch is used by the derived acceptance-bundle pipeline (baseline + STAP).
    try:
        import torch  # type: ignore

        prov["torch"] = getattr(torch, "__version__", None)
        prov["torch_cuda"] = getattr(getattr(torch, "version", None), "cuda", None)
        prov["cuda_is_available"] = bool(torch.cuda.is_available())
        prov["device0"] = torch.cuda.get_device_name(0) if torch.cuda.is_available() else None
    except Exception:
        prov["torch"] = None
    meta: dict[str, Any] = {
        "schema_version": "physdoppler.v1",
        "created_utc": _utc_now_iso(),
        "provenance": {
            **prov,
        },
        "axes": {"order": ["t", "z", "x"], "units": {"x": "m", "z": "m", "t": "s"}},
        "sim_geom": dataclasses.asdict(cfg.sim_geom),
        "acquisition": {
            "prf_hz": float(cfg.prf_hz),
            "dt_s": float(1.0 / cfg.prf_hz),
            "nyquist_hz": float(0.5 * cfg.prf_hz),
            "f0_hz": float(cfg.sim_geom.f0),
            "c0_mps": float(cfg.sim_geom.c0),
        },
        "slow_time": {"pulses_per_set": int(cfg.pulses_per_set), "ensembles": int(cfg.ensembles), "T": int(cfg.T)},
        "config": dataset_meta(cfg),
        "alias_qc": alias_qc_diag,
        "files": hashes,
    }
    _write_json(dataset_dir / "meta.json", meta)

    if args.skip_bundle:
        print(f"[pilot_physical_doppler] wrote dataset -> {dataset_dir}", flush=True)
        return

    # ---- Derived acceptance bundle ----
    dataset_name_default = (
        f"phys_{args.preset}_pw_{cfg.sim_geom.f0/1e6:.1f}MHz_{cfg.ensembles}ens_{cfg.T}T_seed{cfg.seed}"
    )
    dataset_name = _sanitize_name(args.dataset_name or dataset_name_default)

    # Match the frozen Brain-* profile defaults used elsewhere in the repo (r4c).
    write_acceptance_bundle_from_icube(
        out_root=bundle_root,
        dataset_name=dataset_name,
        Icube=icube,
        prf_hz=float(cfg.prf_hz),
        seed=int(cfg.seed),
        tile_hw=(8, 8),
        tile_stride=3,
        Lt=8,
        diag_load=1e-2,
        cov_estimator="tyler_pca",
        huber_c=5.0,
        mvdr_load_mode="auto",
        mvdr_auto_kappa=30.0,
        constraint_ridge=0.15,
        msd_lambda=0.05,
        msd_ridge=0.06,
        msd_agg_mode="median",
        msd_ratio_rho=0.05,
        motion_half_span_rel=0.25,
        msd_contrast_alpha=0.8,
        baseline_type="mc_svd",
        reg_enable=True,
        reg_method="phasecorr",
        reg_subpixel=4,
        reg_reference="median",
        svd_energy_frac=0.90,
        mask_flow_override=mask_flow.astype(bool),
        mask_bg_override=mask_bg.astype(bool),
        stap_conditional_enable=bool(args.stap_conditional),
        feasibility_mode="updated",
        meta_extra={
            "physical_doppler_dataset": True,
            "preset": str(args.preset),
            "tier": str(args.tier),
            "dataset_rel": str(dataset_dir.relative_to(out_root)),
        },
    )

    print(f"[pilot_physical_doppler] wrote dataset -> {dataset_dir}", flush=True)
    print(f"[pilot_physical_doppler] wrote bundle -> {bundle_root / dataset_name}", flush=True)


if __name__ == "__main__":
    main()
