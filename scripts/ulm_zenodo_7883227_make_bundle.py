from __future__ import annotations

import argparse
import json
from pathlib import Path

from pipeline.realdata.ulm_zenodo_7883227 import (
    load_ulm_block_iq,
    load_ulm_zenodo_7883227_params,
)
from sim.kwave.icube_bundle import write_acceptance_bundle_from_icube


def _parse_slice(spec: str) -> slice | None:
    spec = (spec or "").strip()
    if spec in {"", "all", ":", "0:"}:
        return None
    parts = spec.split(":")
    if len(parts) not in {2, 3}:
        raise ValueError(f"Invalid slice spec {spec!r}; expected 'start:stop[:step]' or 'all'.")
    start = int(parts[0]) if parts[0] else None
    stop = int(parts[1]) if parts[1] else None
    step = int(parts[2]) if len(parts) == 3 and parts[2] else None
    if stop is None:
        raise ValueError("Slice spec must include stop (e.g. 0:128).")
    return slice(start, stop, step)


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Build an acceptance bundle from the ULM Zenodo 7883227 IQ blocks.\n"
            "Reads IQ_001_to_025.zip and extracts only the requested block to tmp/.\n"
            "Outputs a k-Wave-compatible bundle so you can run scripts/hab_contract_check.py."
        )
    )
    parser.add_argument(
        "--block-id",
        type=int,
        default=1,
        help="Block id (1..25) inside IQ_001_to_025.zip (default: %(default)s).",
    )
    parser.add_argument(
        "--frames",
        type=str,
        default="0:128",
        help="Frame slice as start:stop[:step] (default: %(default)s).",
    )
    parser.add_argument(
        "--prf-hz",
        type=float,
        default=1000.0,
        help="Slow-time sampling rate in Hz (default: %(default)s).",
    )
    parser.add_argument(
        "--out-root",
        type=Path,
        default=Path("runs/real/ulm7883227_smoke"),
        help="Output root directory for acceptance bundles (default: %(default)s).",
    )
    parser.add_argument(
        "--dataset-name",
        type=str,
        default="",
        help="Optional explicit bundle directory name (default: derived from block-id and frames).",
    )
    parser.add_argument("--tile-h", type=int, default=8)
    parser.add_argument("--tile-w", type=int, default=8)
    parser.add_argument("--tile-stride", type=int, default=3)
    parser.add_argument("--lt", type=int, default=64, help="STAP aperture Lt (default: %(default)s).")
    parser.add_argument(
        "--run-stap",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Whether to run STAP (default: %(default)s). Use --run-stap for full bundles.",
    )
    parser.add_argument(
        "--baseline-type",
        type=str,
        default="mc_svd",
        help="Baseline clutter suppression type (default: %(default)s).",
    )
    parser.add_argument(
        "--svd-energy-frac",
        type=float,
        default=0.95,
        help="MC-SVD baseline energy fraction removed (default: %(default)s).",
    )
    parser.add_argument(
        "--svd-keep-min",
        type=int,
        default=None,
        help="For baseline-type svd_bandpass: 1-based min singular component index to keep.",
    )
    parser.add_argument(
        "--svd-keep-max",
        type=int,
        default=None,
        help="For baseline-type svd_bandpass: 1-based max singular component index to keep.",
    )
    parser.add_argument(
        "--flow-band-hz",
        type=float,
        nargs=2,
        default=(30.0, 250.0),
        metavar=("F_LO", "F_HI"),
        help="Telemetry flow band for Pf-peak (default: %(default)s).",
    )
    parser.add_argument(
        "--alias-band-hz",
        type=float,
        nargs=2,
        default=(500.0, 950.0),
        metavar=("A_LO", "A_HI"),
        help="Telemetry alias band (low, high) in Hz (default: %(default)s).",
    )
    parser.add_argument(
        "--psd-tapers",
        type=int,
        default=3,
        help="Multi-taper PSD tapers for telemetry (default: %(default)s).",
    )
    parser.add_argument(
        "--psd-bandwidth",
        type=float,
        default=2.0,
        help="Multi-taper PSD bandwidth for telemetry (default: %(default)s).",
    )
    args = parser.parse_args()

    frame_slice = _parse_slice(args.frames)
    params = load_ulm_zenodo_7883227_params()
    iq = load_ulm_block_iq(args.block_id, frames=frame_slice)

    dataset_name = (args.dataset_name or "").strip()
    if not dataset_name:
        start = 0 if frame_slice is None or frame_slice.start is None else int(frame_slice.start)
        stop = iq.shape[0] if frame_slice is None else int(frame_slice.stop)  # type: ignore[arg-type]
        dataset_name = f"ulm7883227_block{int(args.block_id):03d}_{start:04d}_{stop:04d}"

    flow_lo, flow_hi = float(args.flow_band_hz[0]), float(args.flow_band_hz[1])
    alias_lo, alias_hi = float(args.alias_band_hz[0]), float(args.alias_band_hz[1])
    alias_center = 0.5 * (alias_lo + alias_hi)
    alias_width = 0.5 * (alias_hi - alias_lo)

    paths = write_acceptance_bundle_from_icube(
        out_root=Path(args.out_root),
        dataset_name=dataset_name,
        Icube=iq,
        prf_hz=float(args.prf_hz),
        tile_hw=(int(args.tile_h), int(args.tile_w)),
        tile_stride=int(args.tile_stride),
        Lt=int(args.lt),
        run_stap=bool(args.run_stap),
        baseline_type=str(args.baseline_type),
        svd_energy_frac=float(args.svd_energy_frac),
        svd_keep_min=int(args.svd_keep_min) if args.svd_keep_min is not None else None,
        svd_keep_max=int(args.svd_keep_max) if args.svd_keep_max is not None else None,
        # Telemetry band design (used for Pf-peak and alias/guard metrics).
        band_ratio_flow_low_hz=flow_lo,
        band_ratio_flow_high_hz=flow_hi,
        band_ratio_alias_center_hz=alias_center,
        band_ratio_alias_width_hz=alias_width,
        psd_tapers=int(args.psd_tapers),
        psd_bandwidth=float(args.psd_bandwidth),
        # For smoke tests, treat conditional execution as a compute heuristic and keep it off.
        stap_conditional_enable=False,
        meta_extra={
            "ulm_zenodo_7883227": {
                "block_id": int(args.block_id),
                "frames_spec": str(args.frames),
                "frame_rate_hz_param_json": float(params.frame_rate_hz),
                "prf_per_emission_hz_param_json": float(params.prf_per_emission_hz),
                "angles_rad_param_json": list(params.angles_rad),
            }
        },
    )

    # Print a small smoke summary to stdout.
    meta_path = Path(paths["meta"])
    meta = json.loads(meta_path.read_text())
    ka = meta.get("ka_contract_v2") or {}
    metrics = ka.get("metrics") if isinstance(ka, dict) else {}
    out = {
        "bundle_dir": str(meta_path.parent),
        "meta": str(meta_path),
        "prf_hz_used": float(meta.get("prf_hz")),
        "frame_rate_hz_param_json": float(params.frame_rate_hz),
        "pf_peak_flow": (metrics or {}).get("pf_peak_flow") if isinstance(metrics, dict) else None,
        "pf_peak_nonbg": (metrics or {}).get("pf_peak_nonbg") if isinstance(metrics, dict) else None,
        "guard_q90": (metrics or {}).get("guard_q90") if isinstance(metrics, dict) else None,
        "iqr_alias_bg": (metrics or {}).get("iqr_alias_bg") if isinstance(metrics, dict) else None,
        "contract_state": ka.get("state") if isinstance(ka, dict) else None,
        "contract_reason": ka.get("reason") if isinstance(ka, dict) else None,
    }
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()

