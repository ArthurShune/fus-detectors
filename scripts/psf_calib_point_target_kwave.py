from __future__ import annotations

import argparse
import dataclasses
import hashlib
import json
import math
import platform
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from typing import Any, Iterable

import numpy as np

from sim.kwave.common import (
    SimGeom,
    _beamform_angle,
    _default_gpu_enabled,
    _demod_iq,
    _precompute_geometry,
    _validate_masks,
    build_grid_and_time,
    build_linear_masks,
    build_medium,
    build_source_p,
)

try:  # pragma: no cover - optional dependency
    from kwave.ksensor import kSensor
    from kwave.ksource import kSource
    from kwave.kspaceFirstOrder2D import kspaceFirstOrder2DC
    from kwave.options.simulation_execution_options import SimulationExecutionOptions
    from kwave.options.simulation_options import SimulationOptions
except Exception:  # pragma: no cover - optional dependency
    kSensor = None  # type: ignore[assignment]
    kSource = None  # type: ignore[assignment]
    kspaceFirstOrder2DC = None  # type: ignore[assignment]
    SimulationExecutionOptions = None  # type: ignore[assignment]
    SimulationOptions = None  # type: ignore[assignment]


FWHM_TO_SIGMA = 1.0 / (2.0 * math.sqrt(2.0 * math.log(2.0)))


def _utc_now_iso() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def _sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


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
            ["git", "diff", "--quiet"],
            cwd=str(repo_root),
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
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


def fwhm_1d(profile: np.ndarray, *, peak_idx: int | None = None, half_rel: float = 0.5) -> float:
    """Return full-width-at-half-maximum in samples.

    Returns NaN if the width is not measurable (e.g. non-positive peak).
    """
    p = np.asarray(profile, dtype=np.float64).ravel()
    if p.size < 3:
        return float("nan")
    if peak_idx is None:
        peak_idx = int(np.argmax(p))
    peak_idx = int(np.clip(peak_idx, 0, p.size - 1))
    peak_val = float(p[peak_idx])
    if not np.isfinite(peak_val) or peak_val <= 0.0:
        return float("nan")
    half = float(half_rel) * peak_val

    # Walk left until we drop below half max.
    left = peak_idx
    while left > 0 and float(p[left]) >= half:
        left -= 1
    if float(p[left]) >= half:
        left_cross = 0.0
    else:
        y0 = float(p[left])
        y1 = float(p[left + 1])
        if y1 == y0:
            left_cross = float(left)
        else:
            t = (half - y0) / (y1 - y0)
            left_cross = float(left) + float(np.clip(t, 0.0, 1.0))

    # Walk right until we drop below half max.
    right = peak_idx
    while right < p.size - 1 and float(p[right]) >= half:
        right += 1
    if float(p[right]) >= half:
        right_cross = float(p.size - 1)
    else:
        y0 = float(p[right - 1])
        y1 = float(p[right])
        if y1 == y0:
            right_cross = float(right)
        else:
            t = (half - y0) / (y1 - y0)
            right_cross = float(right - 1) + float(np.clip(t, 0.0, 1.0))

    width = right_cross - left_cross
    if not np.isfinite(width) or width <= 0.0:
        return float("nan")
    return float(width)


def sigma_from_fwhm_px(fwhm_px: float) -> float:
    if not np.isfinite(fwhm_px) or fwhm_px <= 0.0:
        return float("nan")
    return float(fwhm_px) * float(FWHM_TO_SIGMA)


def fwhm_1d_local(
    profile: np.ndarray,
    *,
    peak_idx: int,
    window_px: int,
    half_rel: float = 0.5,
    baseline_q: float = 0.05,
) -> float:
    """Windowed FWHM with baseline subtraction.

    This avoids pathological widths when the far-field baseline (e.g. residual
    clutter after subtraction) is comparable to the peak amplitude.
    """
    p = np.asarray(profile, dtype=np.float64).ravel()
    if p.size < 3:
        return float("nan")
    peak_idx = int(np.clip(int(peak_idx), 0, p.size - 1))
    w = max(4, int(window_px))
    i0 = max(0, peak_idx - w)
    i1 = min(int(p.size), peak_idx + w + 1)
    win = p[i0:i1]
    if win.size < 3:
        return float("nan")
    q = float(np.clip(baseline_q, 0.0, 0.5))
    base = float(np.quantile(win, q))
    win_adj = win - base
    win_adj[win_adj < 0.0] = 0.0
    return fwhm_1d(win_adj, peak_idx=peak_idx - i0, half_rel=half_rel)


def fit_sigma_vs_depth_linear(z_m: np.ndarray, sigma_px: np.ndarray) -> tuple[float, float, dict[str, Any]]:
    """Fit sigma(z) = sigma0 + alpha*z using least squares.

    Returns (sigma0, alpha, diagnostics).
    """
    z = np.asarray(z_m, dtype=np.float64).ravel()
    s = np.asarray(sigma_px, dtype=np.float64).ravel()
    good = np.isfinite(z) & np.isfinite(s)
    z = z[good]
    s = s[good]
    if z.size < 2:
        raise ValueError("Need at least 2 valid samples to fit sigma(z)")

    # Fit s = alpha*z + sigma0.
    alpha, sigma0 = np.polyfit(z, s, deg=1).tolist()
    pred = alpha * z + sigma0
    resid = s - pred
    ss_res = float(np.sum(resid**2))
    ss_tot = float(np.sum((s - float(np.mean(s))) ** 2) + 1e-12)
    r2 = 1.0 - ss_res / ss_tot
    diag = {
        "n": int(z.size),
        "sigma0_px": float(sigma0),
        "alpha_px_per_m": float(alpha),
        "r2": float(r2),
        "rmse_px": float(np.sqrt(ss_res / float(max(int(z.size), 1)))),
        "z_m_min": float(np.min(z)),
        "z_m_max": float(np.max(z)),
    }
    return float(sigma0), float(alpha), diag


def _parse_angles(values: Iterable[str]) -> list[float]:
    out: list[float] = []
    for v in values:
        for part in str(v).split(","):
            part = part.strip()
            if not part:
                continue
            out.append(float(part))
    if not out:
        raise ValueError("Need at least one angle")
    return out


def _targets_default(g: SimGeom) -> list[tuple[int, int]]:
    # (x_idx, z_idx) in k-Wave medium coordinates: (x,y) == (lateral, depth).
    x0 = int(round(g.Nx / 2))
    z_lo = max(int(g.rx_row_from_top + 10), 20)
    z_hi = min(g.Ny - 20, g.Ny - 1)
    if z_hi <= z_lo:
        z_lo = max(5, int(g.rx_row_from_top + 4))
        z_hi = max(z_lo + 1, g.Ny - 5)
    fracs = [0.20, 0.35, 0.50, 0.65, 0.80]
    # Spread targets laterally so that axial profiles used for FWHM measurement
    # do not contain multiple target peaks.
    x_offsets = [-60, -30, 0, 30, 60]
    targets: list[tuple[int, int]] = []
    for f, ox in zip(fracs, x_offsets, strict=False):
        z_idx = int(round(z_lo + f * (z_hi - z_lo)))
        xi = int(np.clip(x0 + int(ox), 0, g.Nx - 1))
        targets.append((xi, int(np.clip(z_idx, 0, g.Ny - 1))))
    return targets


def _inject_point_targets(
    *,
    g: SimGeom,
    c_map: np.ndarray,
    rho_map: np.ndarray,
    targets_xz_idx: list[tuple[int, int]],
    radius_px: int,
    delta_c_rel: float,
    delta_rho_rel: float,
) -> dict[str, Any]:
    Nx, Ny = int(g.Nx), int(g.Ny)
    rad = max(0, int(radius_px))
    meta_targets: list[dict[str, Any]] = []
    for idx, (xi, zi) in enumerate(targets_xz_idx):
        xi = int(np.clip(int(xi), 0, Nx - 1))
        zi = int(np.clip(int(zi), 0, Ny - 1))
        x0 = max(0, xi - rad)
        x1 = min(Nx - 1, xi + rad)
        z0 = max(0, zi - rad)
        z1 = min(Ny - 1, zi + rad)
        for x in range(x0, x1 + 1):
            for z in range(z0, z1 + 1):
                if (x - xi) * (x - xi) + (z - zi) * (z - zi) > rad * rad:
                    continue
                c_map[x, z] = float(g.c0) * (1.0 + float(delta_c_rel))
                rho_map[x, z] = float(g.rho0) * (1.0 + float(delta_rho_rel))
        meta_targets.append(
            {
                "idx": int(idx),
                "x_idx": int(xi),
                "z_idx": int(zi),
                "x_m": float((xi - g.Nx / 2.0) * g.dx),
                "z_m": float(zi * g.dy),
            }
        )
    return {"targets": meta_targets}


def _run_kwave_once(
    *,
    out_dir: Path,
    angle_deg: float,
    g: SimGeom,
    c_map: np.ndarray,
    rho_map: np.ndarray,
    use_gpu: bool,
) -> tuple[np.ndarray, float]:
    if kspaceFirstOrder2DC is None:
        raise RuntimeError("kwave-python is not available; cannot run PSF calibration.")
    out_dir.mkdir(parents=True, exist_ok=True)

    kgrid = build_grid_and_time(g)
    medium = build_medium(g)
    # Overwrite medium maps with injected targets.
    medium.sound_speed = np.asarray(c_map, dtype=np.float32)
    medium.density = np.asarray(rho_map, dtype=np.float32)

    tx_mask, rx_mask = build_linear_masks(g)
    tx_mask, rx_mask = _validate_masks(kgrid, tx_mask, rx_mask)

    source = kSource()
    source.p_mask = tx_mask
    source.p = build_source_p(g, kgrid, tx_mask, np.deg2rad(angle_deg))
    sensor = kSensor(mask=rx_mask, record=["p"])

    ang_tag = f"{int(round(angle_deg))}"
    input_h5 = f"in_{ang_tag}.h5"
    output_h5 = f"out_{ang_tag}.h5"
    try:
        (out_dir / input_h5).unlink(missing_ok=True)  # type: ignore[arg-type]
    except Exception:
        pass
    try:
        (out_dir / output_h5).unlink(missing_ok=True)  # type: ignore[arg-type]
    except Exception:
        pass

    sim_opts = SimulationOptions(
        pml_inside=False,
        pml_x_size=int(g.pml_size),
        pml_y_size=int(g.pml_size),
        save_to_disk=True,
        data_path=str(out_dir),
        input_filename=input_h5,
        output_filename=output_h5,
    )
    exec_opts = SimulationExecutionOptions(
        is_gpu_simulation=bool(use_gpu),
        show_sim_log=False,
    )

    def _run_with_opts(opts: SimulationExecutionOptions):
        return kspaceFirstOrder2DC(kgrid, source, sensor, medium, sim_opts, opts)

    try:
        sensor_data = _run_with_opts(exec_opts)
    except Exception:
        if use_gpu:
            exec_opts = SimulationExecutionOptions(
                is_gpu_simulation=False,
                show_sim_log=False,
            )
            sensor_data = _run_with_opts(exec_opts)
        else:
            raise
    if isinstance(sensor_data, dict):
        sensor_data = sensor_data.get("p")
    if not isinstance(sensor_data, np.ndarray) or sensor_data.ndim != 2:
        raise RuntimeError("Unexpected sensor data format from k-Wave.")
    return np.asarray(sensor_data, dtype=np.float32), float(kgrid.dt)


def calibrate_psf(
    *,
    g: SimGeom,
    angles_deg: list[float],
    targets_xz_idx: list[tuple[int, int]],
    radius_px: int,
    delta_c_rel: float,
    delta_rho_rel: float,
    seed: int,
    use_gpu: bool,
    work_dir: Path,
    search_radius_px: int = 6,
    subtract_background: bool = True,
    fwhm_window_x_px: int = 80,
    fwhm_window_z_px: int = 80,
    fwhm_baseline_q: float = 0.05,
    fit_mode: str = "constant",
) -> dict[str, Any]:
    rng = np.random.default_rng(int(seed))
    # Start from homogeneous medium and inject deterministic point targets.
    c_map = np.full((g.Nx, g.Ny), float(g.c0), dtype=np.float32)
    rho_map = np.full((g.Nx, g.Ny), float(g.rho0), dtype=np.float32)
    inject_meta = _inject_point_targets(
        g=g,
        c_map=c_map,
        rho_map=rho_map,
        targets_xz_idx=targets_xz_idx,
        radius_px=int(radius_px),
        delta_c_rel=float(delta_c_rel),
        delta_rho_rel=float(delta_rho_rel),
    )

    # Geometry for beamforming (shared across angles).
    XX, ZZ, d_rx = _precompute_geometry(g)

    angle_imgs: list[np.ndarray] = []
    dt: float | None = None
    for a in angles_deg:
        if subtract_background:
            rf_bg, dt_bg = _run_kwave_once(
                out_dir=work_dir / f"angle_{int(round(a))}_bg",
                angle_deg=float(a),
                g=g,
                c_map=np.full((g.Nx, g.Ny), float(g.c0), dtype=np.float32),
                rho_map=np.full((g.Nx, g.Ny), float(g.rho0), dtype=np.float32),
                use_gpu=bool(use_gpu),
            )
            rf_t, dt_i = _run_kwave_once(
                out_dir=work_dir / f"angle_{int(round(a))}_tgt",
                angle_deg=float(a),
                g=g,
                c_map=c_map,
                rho_map=rho_map,
                use_gpu=bool(use_gpu),
            )
            if abs(float(dt_i) - float(dt_bg)) > 1e-12:
                raise RuntimeError("k-Wave dt mismatch between background and target runs.")
            rf = (rf_t - rf_bg).astype(np.float32, copy=False)
        else:
            rf, dt_i = _run_kwave_once(
                out_dir=work_dir / f"angle_{int(round(a))}",
                angle_deg=float(a),
                g=g,
                c_map=c_map,
                rho_map=rho_map,
                use_gpu=bool(use_gpu),
            )
        dt = float(dt_i) if dt is None else dt
        iq = _demod_iq(rf, dt_i, float(g.f0))
        img = _beamform_angle(iq, float(a), dt_i, g, XX, ZZ, d_rx)
        angle_imgs.append(img)
    if dt is None:
        raise RuntimeError("No angles produced RF; cannot calibrate.")

    # Coherent compounding.
    comp = (np.sum(np.stack(angle_imgs, axis=0), axis=0) / float(len(angle_imgs))).astype(np.complex64, copy=False)
    amp = np.abs(comp).astype(np.float32, copy=False)

    meas: list[dict[str, Any]] = []
    sig_x: list[float] = []
    sig_z: list[float] = []
    z_m: list[float] = []

    sr = max(1, int(search_radius_px))
    for t in inject_meta["targets"]:
        x_idx = int(t["x_idx"])
        z_idx = int(t["z_idx"])
        y0 = max(0, z_idx - sr)
        y1 = min(g.Ny - 1, z_idx + sr)
        x0 = max(0, x_idx - sr)
        x1 = min(g.Nx - 1, x_idx + sr)
        patch = amp[y0 : y1 + 1, x0 : x1 + 1]
        if patch.size == 0:
            continue
        loc = int(np.argmax(patch))
        py, px = np.unravel_index(loc, patch.shape)
        y_peak = int(y0 + py)
        x_peak = int(x0 + px)
        peak_val = float(amp[y_peak, x_peak])

        prof_x = amp[y_peak, :]
        prof_z = amp[:, x_peak]
        fwhm_x = fwhm_1d_local(
            prof_x,
            peak_idx=x_peak,
            window_px=int(fwhm_window_x_px),
            baseline_q=float(fwhm_baseline_q),
        )
        fwhm_z = fwhm_1d_local(
            prof_z,
            peak_idx=y_peak,
            window_px=int(fwhm_window_z_px),
            baseline_q=float(fwhm_baseline_q),
        )
        sx = sigma_from_fwhm_px(fwhm_x)
        sz = sigma_from_fwhm_px(fwhm_z)
        zm = float(y_peak) * float(g.dy)

        meas.append(
            {
                "target_idx": int(t["idx"]),
                "expected": {"x_idx": int(x_idx), "z_idx": int(z_idx), "x_m": float(t["x_m"]), "z_m": float(t["z_m"])},
                "peak": {"x_idx": int(x_peak), "z_idx": int(y_peak), "amp": float(peak_val)},
                "widths": {
                    "fwhm_x_px": float(fwhm_x),
                    "fwhm_z_px": float(fwhm_z),
                    "sigma_x_px": float(sx),
                    "sigma_z_px": float(sz),
                },
            }
        )
        if np.isfinite(sx) and np.isfinite(sz):
            sig_x.append(float(sx))
            sig_z.append(float(sz))
            z_m.append(float(zm))

    if len(z_m) < 2:
        raise RuntimeError("Failed to measure PSF widths from point targets (too few valid targets).")

    z_arr = np.asarray(z_m, dtype=np.float64)
    sx_arr = np.asarray(sig_x, dtype=np.float64)
    sz_arr = np.asarray(sig_z, dtype=np.float64)
    fit_mode_clean = str(fit_mode or "constant").strip().lower()
    if fit_mode_clean == "linear":
        sigma_x0, alpha_x, diag_x = fit_sigma_vs_depth_linear(z_arr, sx_arr)
        sigma_z0, alpha_z, diag_z = fit_sigma_vs_depth_linear(z_arr, sz_arr)
    elif fit_mode_clean in {"const", "constant"}:
        sigma_x0 = float(np.median(sx_arr))
        sigma_z0 = float(np.median(sz_arr))
        alpha_x = 0.0
        alpha_z = 0.0
        diag_x = {
            "n": int(z_arr.size),
            "mode": "constant",
            "sigma0_px": float(sigma_x0),
            "alpha_px_per_m": float(alpha_x),
            "r2": None,
            "rmse_px": None,
            "z_m_min": float(np.min(z_arr)),
            "z_m_max": float(np.max(z_arr)),
        }
        diag_z = {
            "n": int(z_arr.size),
            "mode": "constant",
            "sigma0_px": float(sigma_z0),
            "alpha_px_per_m": float(alpha_z),
            "r2": None,
            "rmse_px": None,
            "z_m_min": float(np.min(z_arr)),
            "z_m_max": float(np.max(z_arr)),
        }
    else:
        raise ValueError(f"Unsupported fit_mode: {fit_mode}")

    # Small deterministic perturbation to avoid negative zeros in JSON dumps (cosmetic).
    _ = rng.random() * 0.0

    return {
        "sigma_x0_px": float(sigma_x0),
        "sigma_z0_px": float(sigma_z0),
        "alpha_x_per_m": float(alpha_x),
        "alpha_z_per_m": float(alpha_z),
        "schema_version": "psf_calib.kwavetarget.v1",
        "created_utc": _utc_now_iso(),
        "sim_geom": dataclasses.asdict(g),
        "angles_deg": [float(a) for a in angles_deg],
        "targets": inject_meta["targets"],
        "target_injection": {
            "radius_px": int(radius_px),
            "delta_c_rel": float(delta_c_rel),
            "delta_rho_rel": float(delta_rho_rel),
            "subtract_background": bool(subtract_background),
        },
        "fwhm_measurement": {
            "window_x_px": int(fwhm_window_x_px),
            "window_z_px": int(fwhm_window_z_px),
            "baseline_q": float(fwhm_baseline_q),
        },
        "measurements": meas,
        "fit": {"x": diag_x, "z": diag_z},
    }


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(
        description=(
            "Generate an optional point-target PSF calibration artifact for the physical-doppler surrogate "
            "(fits sigma_x(z), sigma_z(z) for a depth-dependent separable Gaussian blur)."
        )
    )
    ap.add_argument("--out", type=Path, required=True, help="Output JSON path (psf_calib.json).")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument(
        "--angles-deg",
        type=str,
        nargs="+",
        default=["-6,-3,0,3,6"],
        help="Steering angles in degrees (comma-separated or list).",
    )
    ap.add_argument("--nx", type=int, default=240)
    ap.add_argument("--ny", type=int, default=240)
    ap.add_argument("--dx", type=float, default=90e-6)
    ap.add_argument("--dy", type=float, default=90e-6)
    ap.add_argument("--f0-hz", type=float, default=7.5e6)
    ap.add_argument("--c0", type=float, default=1540.0)
    ap.add_argument("--rho0", type=float, default=1000.0)
    ap.add_argument("--pml-size", type=int, default=16)
    ap.add_argument("--cfl", type=float, default=0.3)
    ap.add_argument("--ncycles", type=int, default=3)
    ap.add_argument("--tx-row-from-top", type=int, default=2)
    ap.add_argument("--rx-row-from-top", type=int, default=3)
    ap.add_argument("--alpha-db-mhz-cm", type=float, default=0.5)
    ap.add_argument("--alpha-power", type=float, default=1.5)

    ap.add_argument(
        "--targets-xz-idx",
        type=int,
        nargs=2,
        action="append",
        default=None,
        metavar=("X_IDX", "Z_IDX"),
        help="Point target location in grid indices (x_idx z_idx). May be repeated.",
    )
    ap.add_argument("--target-radius-px", type=int, default=2, help="Disk radius for point targets (pixels).")
    ap.add_argument("--delta-c-rel", type=float, default=0.20, help="Relative sound-speed contrast for targets.")
    ap.add_argument("--delta-rho-rel", type=float, default=0.20, help="Relative density contrast for targets.")
    ap.add_argument("--search-radius-px", type=int, default=6, help="Search radius around expected peak (pixels).")
    ap.add_argument(
        "--no-subtract-background",
        action="store_true",
        help="Disable per-angle homogeneous background subtraction (debug only).",
    )
    ap.add_argument(
        "--fwhm-window-x-px",
        type=int,
        default=20,
        help="Half-window (pixels) used for lateral FWHM measurement around the peak.",
    )
    ap.add_argument(
        "--fwhm-window-z-px",
        type=int,
        default=80,
        help="Half-window (pixels) used for axial FWHM measurement around the peak.",
    )
    ap.add_argument(
        "--fwhm-baseline-q",
        type=float,
        default=0.05,
        help="Baseline quantile subtracted within the FWHM measurement window.",
    )
    ap.add_argument(
        "--fit-mode",
        type=str,
        default="constant",
        choices=["constant", "linear"],
        help="Fit mode for sigma(z): constant median vs linear least-squares slope.",
    )

    ap.add_argument("--force-cpu", action="store_true", help="Force k-Wave execution on CPU.")
    ap.add_argument(
        "--work-dir",
        type=Path,
        default=None,
        help="Optional directory to keep k-Wave H5 artifacts (default: temporary).",
    )
    return ap.parse_args()


def main() -> None:
    args = parse_args()
    angles = _parse_angles(args.angles_deg)

    # Ensure the time array is long enough for echoes from max depth to return.
    t_end = 2.0 * float(args.ny) * float(args.dy) / max(float(args.c0), 1.0) + float(args.ncycles) / max(
        float(args.f0_hz), 1.0
    )
    g = SimGeom(
        Nx=int(args.nx),
        Ny=int(args.ny),
        dx=float(args.dx),
        dy=float(args.dy),
        c0=float(args.c0),
        rho0=float(args.rho0),
        pml_size=int(args.pml_size),
        cfl=float(args.cfl),
        t_end=float(t_end),
        f0=float(args.f0_hz),
        ncycles=int(args.ncycles),
        tx_row_from_top=int(args.tx_row_from_top),
        rx_row_from_top=int(args.rx_row_from_top),
        alpha_db_mhz_cm=float(args.alpha_db_mhz_cm),
        alpha_power=float(args.alpha_power),
    )

    if args.targets_xz_idx is None:
        targets = _targets_default(g)
    else:
        targets = [(int(x), int(z)) for (x, z) in args.targets_xz_idx]

    use_gpu = _default_gpu_enabled()
    if bool(args.force_cpu):
        use_gpu = False

    if args.work_dir is None:
        tmp_ctx = tempfile.TemporaryDirectory(prefix="psf_calib_kwave_")
        work_dir = Path(tmp_ctx.name)
    else:
        tmp_ctx = None
        work_dir = Path(args.work_dir)
        work_dir.mkdir(parents=True, exist_ok=True)

    try:
        payload = calibrate_psf(
            g=g,
            angles_deg=angles,
            targets_xz_idx=targets,
            radius_px=int(args.target_radius_px),
            delta_c_rel=float(args.delta_c_rel),
            delta_rho_rel=float(args.delta_rho_rel),
            seed=int(args.seed),
            use_gpu=bool(use_gpu),
            work_dir=work_dir,
            search_radius_px=int(args.search_radius_px),
            subtract_background=not bool(args.no_subtract_background),
            fwhm_window_x_px=int(args.fwhm_window_x_px),
            fwhm_window_z_px=int(args.fwhm_window_z_px),
            fwhm_baseline_q=float(args.fwhm_baseline_q),
            fit_mode=str(args.fit_mode),
        )
    finally:
        if tmp_ctx is not None:
            tmp_ctx.cleanup()

    repo_root = Path(__file__).resolve().parents[1]
    prov: dict[str, Any] = {
        "command": " ".join(sys.argv),
        "cwd": str(Path.cwd()),
        "git": _git_info(repo_root),
        "python": sys.version.splitlines()[0],
        "platform": platform.platform(),
        "numpy": np.__version__,
        "use_gpu": bool(use_gpu),
    }
    try:  # pragma: no cover - optional dependency
        import scipy  # type: ignore

        prov["scipy"] = getattr(scipy, "__version__", None)
    except Exception:
        prov["scipy"] = None
    try:  # pragma: no cover - optional dependency
        import torch  # type: ignore

        prov["torch"] = getattr(torch, "__version__", None)
        prov["torch_cuda"] = getattr(getattr(torch, "version", None), "cuda", None)
        prov["cuda_is_available"] = bool(torch.cuda.is_available())
        prov["device0"] = torch.cuda.get_device_name(0) if torch.cuda.is_available() else None
    except Exception:
        prov["torch"] = None
    try:  # pragma: no cover - optional dependency
        import kwave  # type: ignore

        prov["kwave"] = getattr(kwave, "__version__", None)
    except Exception:
        prov["kwave"] = None

    payload["provenance"] = prov

    # Include a stable hash of the JSON content (excluding this self-hash).
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    tmp = dict(payload)
    tmp.pop("sha256", None)
    raw = json.dumps(tmp, sort_keys=True).encode("utf-8")
    payload["sha256"] = _sha256_bytes(raw)
    with out_path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, sort_keys=True)
        f.write("\n")
    print(f"[psf_calib_point_target_kwave] wrote {out_path} sha256={_sha256_file(out_path)}", flush=True)


if __name__ == "__main__":
    main()
