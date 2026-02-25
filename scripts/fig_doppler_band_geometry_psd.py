#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

import matplotlib as mpl
import matplotlib.pyplot as plt


def _tile_mean_psd_hz(tile: np.ndarray, *, prf_hz: float) -> tuple[np.ndarray, np.ndarray]:
    """
    Return one-sided PSD for the tile-mean slow-time signal.

    tile: (T, H, W) complex (beamformed IQ, or baseline residual IQ).
    """

    if tile.ndim != 3:
        raise ValueError(f"Expected tile ndim=3 (T,H,W), got shape {tile.shape}")
    if not np.iscomplexobj(tile):
        raise ValueError("Expected complex tile IQ.")

    prf_hz = float(prf_hz)
    if prf_hz <= 0:
        raise ValueError("prf_hz must be > 0")

    x = tile.mean(axis=(1, 2))
    T = int(x.shape[0])
    if T < 8:
        raise ValueError(f"Need T>=8 for a meaningful PSD, got T={T}")

    # Remove mean to suppress the DC spike (we still show Po as a band concept).
    x = x - np.mean(x)
    x = x * np.hanning(T)

    X = np.fft.fft(x)
    f = np.fft.fftfreq(T, d=1.0 / prf_hz)
    m = f >= 0
    f = f[m]
    P = np.abs(X[m]) ** 2
    return f.astype(np.float64, copy=False), P.astype(np.float64, copy=False)


def _summarize_tile_psd(
    tile: np.ndarray,
    *,
    prf_hz: float,
    flow_lo_hz: float,
    flow_hi_hz: float,
    guard_hi_hz: float,
) -> dict[str, float | np.ndarray]:
    f, P = _tile_mean_psd_hz(tile, prf_hz=prf_hz)
    if f.size < 2:
        raise ValueError("PSD frequency grid too small.")

    # Peak excluding DC bin.
    peak_idx = int(np.argmax(P[1:]) + 1)
    f_peak_hz = float(f[peak_idx])

    nyq_hz = 0.5 * float(prf_hz)
    E_flow = float(P[(f >= float(flow_lo_hz)) & (f <= float(flow_hi_hz))].sum())
    E_alias = float(P[(f >= float(guard_hi_hz)) & (f <= nyq_hz + 1e-9)].sum())
    ratio = float(np.log((E_alias + 1e-12) / (E_flow + 1e-12)))
    return {
        "f_hz": f,
        "P": P,
        "f_peak_hz": f_peak_hz,
        "E_flow": E_flow,
        "E_alias": E_alias,
        "log_Ea_over_Ef": ratio,
    }


def _select_tiles_from_debug_dir(
    debug_dir: Path,
    *,
    prf_hz: float,
    flow_lo_hz: float,
    flow_hi_hz: float,
    guard_hi_hz: float,
) -> tuple[Path, Path, dict[str, float | np.ndarray], dict[str, float | np.ndarray]]:
    """
    Select (flow-like tile, alias-dominant tile) from a STAP debug directory.

    The selection rule is deterministic:
      - flow tile: peak in Pf and minimizes log(Ea/Ef) (tie-break: max E_flow)
      - alias tile: peak in Pa and maximizes log(Ea/Ef) (tie-break: max E_alias)
    """

    debug_dir = Path(debug_dir)
    if not debug_dir.is_dir():
        raise FileNotFoundError(debug_dir)

    tile_paths = sorted(debug_dir.glob("tile_*.npz"))
    if not tile_paths:
        raise FileNotFoundError(f"No tile_*.npz found under {debug_dir}")

    nyq_hz = 0.5 * float(prf_hz)

    flow_candidates: list[tuple[Path, dict[str, float | np.ndarray]]] = []
    alias_candidates: list[tuple[Path, dict[str, float | np.ndarray]]] = []
    for p in tile_paths:
        d = np.load(p, allow_pickle=True)
        if "tile" not in d:
            continue
        tile = d["tile"]
        try:
            info = _summarize_tile_psd(
                tile,
                prf_hz=prf_hz,
                flow_lo_hz=flow_lo_hz,
                flow_hi_hz=flow_hi_hz,
                guard_hi_hz=guard_hi_hz,
            )
        except Exception:
            continue

        f_peak = float(info["f_peak_hz"])  # type: ignore[arg-type]
        if float(flow_lo_hz) <= f_peak <= float(flow_hi_hz):
            flow_candidates.append((p, info))
        if float(guard_hi_hz) <= f_peak <= nyq_hz + 1e-9:
            alias_candidates.append((p, info))

    if not flow_candidates:
        raise RuntimeError(f"No flow-like tiles found under {debug_dir} (peak in Pf).")
    if not alias_candidates:
        raise RuntimeError(f"No alias-dominant tiles found under {debug_dir} (peak in Pa).")

    flow_path, flow_info = sorted(
        flow_candidates,
        key=lambda x: (float(x[1]["log_Ea_over_Ef"]), -float(x[1]["E_flow"])),  # type: ignore[arg-type]
    )[0]
    alias_path, alias_info = sorted(
        alias_candidates,
        key=lambda x: (-float(x[1]["log_Ea_over_Ef"]), -float(x[1]["E_alias"])),  # type: ignore[arg-type]
    )[0]
    return flow_path, alias_path, flow_info, alias_info


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Generate a Doppler PSD plot showing Po/Pf/Pg/Pa band geometry and contrasting "
            "flow-like vs alias-dominant tiles (vector PDF).\n"
            "Default mode uses an empirical STAP tile-debug directory so the plotted PSD curves are data-driven."
        )
    )
    parser.add_argument(
        "--out",
        default="figs/paper/doppler_band_geometry_psd.pdf",
        help="Output PDF path (default: %(default)s).",
    )
    parser.add_argument("--prf-hz", type=float, default=1500.0, help="PRF in Hz (default: %(default)s).")
    parser.add_argument("--f0-mhz", type=float, default=7.5, help="Center frequency for velocity axis in MHz.")
    parser.add_argument("--c-ms", type=float, default=1540.0, help="Sound speed in m/s (default: %(default)s).")
    parser.add_argument(
        "--mode",
        choices=["empirical", "conceptual"],
        default="empirical",
        help="Use empirical tile PSDs (recommended) or a conceptual mock-up (default: %(default)s).",
    )
    parser.add_argument(
        "--stap-debug-dir",
        default=(
            "runs/pilot/fair_filter_matrix_smoke_fix_aliascontract/"
            "aliascontract_seed2_mcsvd_full/pw_7.5MHz_5ang_5ens_64T_seed2/stap_debug"
        ),
        help="Empirical mode: directory containing tile_*.npz debug files (default: %(default)s).",
    )
    args = parser.parse_args()

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    prf_hz = float(args.prf_hz)
    nyq_hz = 0.5 * prf_hz

    # Fixed band geometry (brain fUS-like), used consistently in the manuscript text.
    f_flow_lo = 30.0
    f_flow_hi = 250.0
    f_guard_hi = 400.0

    if args.mode == "empirical":
        debug_dir = Path(args.stap_debug_dir)
        flow_path, alias_path, flow_info, alias_info = _select_tiles_from_debug_dir(
            debug_dir,
            prf_hz=prf_hz,
            flow_lo_hz=f_flow_lo,
            flow_hi_hz=f_flow_hi,
            guard_hi_hz=f_guard_hi,
        )

        f = np.asarray(flow_info["f_hz"], dtype=np.float64)
        P_flow = np.asarray(flow_info["P"], dtype=np.float64)
        P_alias = np.asarray(alias_info["P"], dtype=np.float64)

        # Ensure both PSDs share the same frequency grid.
        if P_alias.shape != P_flow.shape:
            raise RuntimeError("Selected tiles produced mismatched PSD shapes.")

        # Normalize to unit peak for a clean dB view.
        psd_flow = P_flow / float(np.max(P_flow))
        psd_alias = P_alias / float(np.max(P_alias))

        label_flow = (
            r"Flow-like tile"
            + rf" ($f_{{\mathrm{{peak}}}}\!\approx\!{float(flow_info['f_peak_hz']):.0f}$ Hz)"
        )
        label_alias = (
            r"Alias-dominant tile"
            + rf" ($f_{{\mathrm{{peak}}}}\!\approx\!{float(alias_info['f_peak_hz']):.0f}$ Hz)"
        )
        note = "empirical tiles (deterministic selection)"
    else:
        # Conceptual fallback: two mock PSDs (power vs Doppler frequency).
        f = np.linspace(0.0, nyq_hz, 2400)
        clutter = 0.06 / (1.0 + (f / 20.0) ** 2) + 0.005

        def gauss(x: np.ndarray, mu: float, sigma: float) -> np.ndarray:
            return np.exp(-0.5 * ((x - mu) / sigma) ** 2)

        psd_flow = (
            clutter
            + 1.00 * gauss(f, 120.0, 35.0)
            + 0.10 * gauss(f, 320.0, 55.0)
            + 0.04 * gauss(f, 650.0, 40.0)
        )
        psd_alias = (
            clutter
            + 0.22 * gauss(f, 140.0, 40.0)
            + 0.14 * gauss(f, 320.0, 60.0)
            + 1.10 * gauss(f, 650.0, 38.0)
        )

        psd_flow = psd_flow / np.max(psd_flow)
        psd_alias = psd_alias / np.max(psd_alias)
        label_flow = r"Flow-like tile (mock)"
        label_alias = r"Alias-dominant tile (mock)"
        note = "conceptual mock PSDs"

    eps = 1e-4
    y_flow_db = 10.0 * np.log10(psd_flow + eps)
    y_alias_db = 10.0 * np.log10(psd_alias + eps)

    mpl.rcParams.update(
        {
            "font.family": "serif",
            "mathtext.fontset": "cm",
            "font.size": 9,
            "axes.linewidth": 0.8,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "legend.frameon": False,
        }
    )

    # IEEE-ish single-column size: ~3.3 inches wide.
    fig, ax = plt.subplots(figsize=(3.45, 2.45), constrained_layout=True)

    # Shaded band regions.
    ax.axvspan(0.0, f_flow_lo, color="#d9d9d9", alpha=0.35, lw=0)  # Po (DC/clutter)
    ax.axvspan(f_flow_lo, f_flow_hi, color="#9ecae1", alpha=0.35, lw=0)  # Pf (flow)
    ax.axvspan(f_flow_hi, f_guard_hi, color="#fde0a2", alpha=0.45, lw=0)  # Pg (guard)
    ax.axvspan(f_guard_hi, nyq_hz, color="#fcbba1", alpha=0.35, lw=0)  # Pa (alias)

    # Boundary lines.
    for x in [f_flow_lo, f_flow_hi, f_guard_hi]:
        ax.axvline(x, color="black", lw=0.8, alpha=0.6)
    ax.axvline(nyq_hz, color="black", lw=0.9, alpha=0.8, linestyle=":")

    # PSD curves.
    ax.plot(f, y_flow_db, color="#1f77b4", lw=1.8, label=label_flow)
    ax.plot(f, y_alias_db, color="#d62728", lw=1.6, linestyle="--", label=label_alias)

    # Axis formatting.
    ax.set_xlim(0.0, nyq_hz)
    ax.set_ylim(-42.0, 2.0)
    ax.set_xlabel(r"Doppler frequency $f$ (Hz)")
    ax.set_ylabel(r"PSD (dB, normalized)")

    ax.grid(True, which="both", axis="y", alpha=0.25, lw=0.6)

    # Band labels near top of plot.
    y_top = 1.2
    ax.text(0.5 * (0.0 + f_flow_lo), y_top, r"$P_o$", ha="center", va="bottom")
    ax.text(0.5 * (f_flow_lo + f_flow_hi), y_top, r"$P_f$", ha="center", va="bottom")
    ax.text(0.5 * (f_flow_hi + f_guard_hi), y_top, r"$P_g$", ha="center", va="bottom")
    ax.text(0.5 * (f_guard_hi + nyq_hz), y_top, r"$P_a$", ha="center", va="bottom")

    # Nyquist annotation (keep inside the plot to avoid crowding the top axis ticks).
    ax.text(nyq_hz, 1.35, "Nyquist", ha="right", va="bottom", fontsize=8)

    # Optional velocity axis (example mapping).
    # v = c f / (2 f0) (assuming cos(theta)=1). Show as |v| in mm/s.
    f0_hz = float(args.f0_mhz) * 1e6
    scale_mm_per_s_per_hz = float(args.c_ms) / (2.0 * f0_hz) * 1e3
    top = ax.twiny()
    top.set_xlim(ax.get_xlim())
    tick_f = np.array([0.0, 250.0, 500.0, nyq_hz])
    tick_f = tick_f[tick_f <= nyq_hz + 1e-9]
    top.set_xticks(tick_f)
    tick_v = scale_mm_per_s_per_hz * tick_f
    top.set_xticklabels([f"{v:.0f}" for v in tick_v])
    top.set_xlabel(rf"Axial speed $|v|$ (mm/s), $f_0={args.f0_mhz:g}$ MHz", fontsize=8)
    top.tick_params(axis="x", labelsize=8)

    ax.legend(loc="lower left", fontsize=8)
    ax.text(
        0.99,
        0.02,
        note,
        transform=ax.transAxes,
        ha="right",
        va="bottom",
        fontsize=7.5,
        color="#333333",
    )

    fig.savefig(out_path, bbox_inches="tight", pad_inches=0.10, format="pdf")
    plt.close(fig)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
