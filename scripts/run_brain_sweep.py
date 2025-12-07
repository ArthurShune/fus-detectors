#!/usr/bin/env python3
"""
Phase 1.1 robustness sweeps for the clinical STAP PD profile on brain k-Wave regimes.

This script drives a small grid over:
  - Regime: Brain-OpenSkull, Brain-SkullOR
  - PRF: {1200, 1500, 2000} Hz
  - Tile size: {6x6, 8x8, 12x12} with stride {2, 3, 4}
  - Lt: {6, 8, 10}
  - Nuisance knobs: clutter_snr_db scaled by {−3, 0, +3} dB around a nominal value,
    alias_amp_scale in {0.8, 1.0, 1.2}

For each combo we:
  1) Run sim/kwave/pilot_motion.py to generate a k-Wave brain fUS pilot with an
     MC–SVD baseline and PD scoring (KA is disabled).
  2) Use scripts/hab_contract_check.py --score-mode pd on the resulting bundle to
     extract TPR at FPR in {1e-4, 3e-4, 1e-3} for MC–SVD PD vs STAP PD.

Outputs:
  - For each run: a pilot directory under --out-root with a single bundle
    (pw_* seed1) containing pd_base.npy / pd_stap.npy / meta.json.
  - A JSON summary (--summary-json) with one row per sweep point containing
    regime, PRF, tile, stride, Lt, clutter/alias scales, latency, coverage, and
    ΔTPR at the target FPRs.

This is intentionally lightweight and composable: the pilot generator is the
only place that touches k-Wave; downstream aggregation and plotting are handled
by scripts/aggregate_brain_sweep.py.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Tuple


REPO = Path(__file__).resolve().parents[1]


@dataclass
class SweepPoint:
    regime: str          # "open" or "skullor"
    prf_hz: float
    tile: int            # tile height == tile width
    stride: int
    lt: int
    clutter_offset_db: float  # −3, 0, +3 relative to nominal
    alias_scale: float        # 0.8, 1.0, 1.2

    def tag(self) -> str:
        off = int(self.clutter_offset_db)
        return (
            f"{self.regime}_prf{int(self.prf_hz)}_tile{self.tile}"
            f"_s{self.stride}_lt{self.lt}_clut{off:+d}_alias{self.alias_scale:g}"
        )


def _float_list(spec: str) -> List[float]:
    return [float(s) for s in spec.replace(";", ",").split(",") if s.strip()]


def _int_list(spec: str) -> List[int]:
    return [int(float(s)) for s in spec.replace(";", ",").split(",") if s.strip()]


def build_grid(
    regimes: Iterable[str],
    prfs: Iterable[float],
    tiles: Iterable[int],
    lts: Iterable[int],
    clutter_offsets_db: Iterable[float],
    alias_scales: Iterable[float],
) -> List[SweepPoint]:
    # Fixed mapping from tile size to stride, as in the text plan.
    stride_for_tile: Dict[int, int] = {6: 2, 8: 3, 12: 4}
    points: List[SweepPoint] = []
    for regime in regimes:
        r = regime.lower()
        if r not in {"open", "skullor"}:
            raise ValueError(f"Unsupported regime '{regime}' (expected open or skullor)")
        for prf in prfs:
            for tile in tiles:
                if tile not in stride_for_tile:
                    raise ValueError(
                        f"No stride defined for tile={tile}; "
                        "extend stride_for_tile in run_brain_sweep.py."
                    )
                stride = stride_for_tile[tile]
                for lt in lts:
                    for coff in clutter_offsets_db:
                        for a_scale in alias_scales:
                            points.append(
                                SweepPoint(
                                    regime=r,
                                    prf_hz=float(prf),
                                    tile=int(tile),
                                    stride=int(stride),
                                    lt=int(lt),
                                    clutter_offset_db=float(coff),
                                    alias_scale=float(a_scale),
                                )
                            )
    return points


def _nominal_clutter_snr_db_for_regime(regime: str) -> float:
    """
    Nominal clutter SNR (dB) used as the center for ±3 dB offsets.

    For the Brain-* HAB presets the generator uses ~25 dB by default; we reuse
    that here so that offsets are {22, 25, 28} dB.
    """
    # Keep this simple and explicit so we do not depend on Makefile indirection.
    return 25.0 if regime in {"open", "skullor"} else 25.0


def _profile_for_regime(regime: str) -> str:
    if regime == "open":
        return "Brain-OpenSkull"
    if regime == "skullor":
        return "Brain-SkullOR"
    raise ValueError(f"Unsupported regime '{regime}'")


def _pilot_out_dir(root: Path, pt: SweepPoint) -> Path:
    return root / pt.regime / pt.tag()


def _bundle_dir(pilot_dir: Path) -> Path | None:
    # Expect a single pw_* dataset directory in the pilot output.
    cands = sorted(pilot_dir.glob("pw_*"))
    return cands[0] if cands else None


def run_pilot_for_point(
    pt: SweepPoint,
    out_root: Path,
    seed: int,
    nx: int,
    ny: int,
    angles: str,
    ensembles: int,
    pulses: int,
    synthetic: bool,
    dry_run: bool,
) -> Path:
    """
    Run sim/kwave/pilot_motion.py for a single SweepPoint and return the bundle path.
    """
    pilot_dir = _pilot_out_dir(out_root, pt)
    pilot_dir.mkdir(parents=True, exist_ok=True)

    # Skip simulation if a bundle already exists.
    existing = _bundle_dir(pilot_dir)
    if existing is not None and (existing / "meta.json").exists():
        return existing

    profile = _profile_for_regime(pt.regime)
    nominal_clutter = _nominal_clutter_snr_db_for_regime(pt.regime)
    clutter_snr_db = nominal_clutter + pt.clutter_offset_db

    cmd: List[str] = [
        sys.executable,
        str(REPO / "sim" / "kwave" / "pilot_motion.py"),
        "--out",
        str(pilot_dir),
        "--profile",
        profile,
    ]

    # pilot_motion.py declares --angles with nargs="+", and then supports either
    # space-separated or comma-separated specifications. Here we normalize any
    # comma/semicolon-separated spec into a list of separate tokens so that
    # negative angles (e.g. -12) are passed unambiguously as arguments.
    if angles:
        cmd.append("--angles")
        for token in str(angles).replace(";", ",").split(","):
            token_clean = token.strip()
            if token_clean:
                cmd.append(token_clean)

    cmd += [
        "--ensembles",
        str(ensembles),
        "--jitter_um",
        "40",
        "--pulses",
        str(pulses),
        "--prf",
        str(pt.prf_hz),
        "--seed",
        str(seed),
        "--Nx",
        str(nx),
        "--Ny",
        str(ny),
        "--tile-h",
        str(pt.tile),
        "--tile-w",
        str(pt.tile),
        "--tile-stride",
        str(pt.stride),
        "--lt",
        str(pt.lt),
        # Clinical-style STAP PD configuration: conservative covariance and
        # PD scoring only, KA off.
        "--diag-load",
        "0.07",
        "--cov-estimator",
        "tyler_pca",
        "--huber-c",
        "5.0",
        "--fd-span-mode",
        "psd",
        "--fd-span-rel",
        "0.30,1.10",
        "--grid-step-rel",
        "0.20",
        "--max-pts",
        "3",
        "--msd-lambda",
        "0.05",
        "--msd-ridge",
        "0.10",
        "--msd-agg",
        "median",
        "--msd-ratio-rho",
        "0.05",
        "--msd-contrast-alpha",
        "0.6",
        "--mvdr-load-mode",
        "auto",
        "--mvdr-auto-kappa",
        "120",
        "--score-mode",
        "pd",
        "--ka-mode",
        "none",
        "--baseline-type",
        "mc_svd",
        "--stap-device",
        "cuda",
        "--clutter-snr-db",
        f"{clutter_snr_db:.1f}",
        "--alias-amp-scale",
        f"{pt.alias_scale:.3f}",
    ]

    if synthetic:
        cmd.append("--synthetic")

    print("[pilot]", " ".join(cmd))
    if not dry_run:
        subprocess.run(cmd, check=True)

    bundle = _bundle_dir(pilot_dir)
    if bundle is None:
        # In dry-run mode we do not expect outputs; return the pilot directory
        # so callers can still log tags without touching the filesystem.
        if dry_run:
            return pilot_dir
        raise RuntimeError(f"No bundle produced under {pilot_dir}")
    return bundle


def hab_check_pd(bundle: Path, fprs: Iterable[float]) -> Dict[str, float]:
    """
    Run hab_contract_check.py --score-mode pd on a bundle and parse TPRs.
    """
    fpr_args = [f"{fpr:g}" for fpr in fprs]
    cmd = [
        sys.executable,
        str(REPO / "scripts" / "hab_contract_check.py"),
        str(bundle),
        "--score-mode",
        "pd",
        "--fprs",
        *fpr_args,
    ]
    result = subprocess.run(cmd, capture_output=True, text=True, check=True)
    tpr_by_fpr: Dict[str, float] = {}
    for line in result.stdout.splitlines():
        line = line.strip()
        if not line.startswith("fpr="):
            continue
        # Example line:
        #   fpr=0.0001: thr_base=..., tpr_base=0.000, thr_stap=..., tpr_stap=0.209
        try:
            prefix, rest = line.split(":", 1)
            fpr_str = prefix.split("=")[1].strip()
            fpr_val = float(fpr_str)
            tpr_base_str = rest.split("tpr_base=")[1].split(",")[0].strip()
            tpr_stap_str = rest.split("tpr_stap=")[1].split()[0].strip()
            tpr_by_fpr[f"base@{fpr_val:g}"] = float(tpr_base_str)
            tpr_by_fpr[f"stap@{fpr_val:g}"] = float(tpr_stap_str)
        except Exception:
            continue
    return tpr_by_fpr


def summarize_point(
    pt: SweepPoint,
    bundle: Path,
    fprs: Iterable[float],
) -> Dict[str, object]:
    meta_path = bundle / "meta.json"
    meta = json.loads(meta_path.read_text())
    tele = meta.get("stap_fallback_telemetry", {}) or {}

    baseline_ms = float(tele.get("baseline_ms", 0.0))
    stap_ms = float(tele.get("stap_total_ms", 0.0))

    # Coverage fraction for conditional STAP: 1 - skipped_flow0 / total_tiles.
    tile_hw = tuple(int(v) for v in meta.get("tile_hw", [pt.tile, pt.tile]))
    stride = int(meta.get("tile_stride", pt.stride))
    pd_base_path = bundle / "pd_base.npy"
    H = W = None
    if pd_base_path.exists():
        import numpy as np

        pd_base = np.load(pd_base_path)
        H, W = pd_base.shape
    if H is not None and W is not None:
        th, tw = tile_hw
        ny = max(0, (H - th) // stride + 1)
        nx = max(0, (W - tw) // stride + 1)
        tile_count = ny * nx
    else:
        tile_count = None

    skipped = float(tele.get("stap_tiles_skipped_flow0", 0.0))
    coverage = None
    if tile_count and tile_count > 0:
        coverage = 1.0 - skipped / float(tile_count)

    tpr_by_fpr = hab_check_pd(bundle, fprs)
    row: Dict[str, object] = {
        "tag": pt.tag(),
        "regime": pt.regime,
        "prf_hz": pt.prf_hz,
        "tile": pt.tile,
        "stride": pt.stride,
        "lt": pt.lt,
        "clutter_offset_db": pt.clutter_offset_db,
        "alias_scale": pt.alias_scale,
        "baseline_ms": baseline_ms,
        "stap_ms": stap_ms,
        "coverage": coverage,
        "bundle": str(bundle),
    }
    for fpr in fprs:
        key_base = f"base@{fpr:g}"
        key_stap = f"stap@{fpr:g}"
        base_val = float(tpr_by_fpr.get(key_base, float("nan")))
        stap_val = float(tpr_by_fpr.get(key_stap, float("nan")))
        row[f"tpr_base@{fpr:g}"] = base_val
        row[f"tpr_stap@{fpr:g}"] = stap_val
        row[f"delta_tpr@{fpr:g}"] = stap_val - base_val
    return row


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Brain STAP PD robustness sweeps (Phase 1.1).")
    ap.add_argument(
        "--out-root",
        type=Path,
        default=REPO / "runs" / "brain_sweep",
        help="Root directory for pilot runs and bundles.",
    )
    ap.add_argument(
        "--regimes",
        type=str,
        default="open,skullor",
        help="Comma-separated regimes to sweep: open, skullor.",
    )
    ap.add_argument(
        "--prfs",
        type=str,
        default="1200,1500,2000",
        help="Comma-separated PRF values (Hz).",
    )
    ap.add_argument(
        "--tiles",
        type=str,
        default="6,8,12",
        help="Comma-separated tile sizes (pixels; square tiles).",
    )
    ap.add_argument(
        "--lts",
        type=str,
        default="6,8,10",
        help="Comma-separated Lt values (slow-time aperture).",
    )
    ap.add_argument(
        "--clutter-offsets-db",
        type=str,
        nargs="+",
        default=["-3", "0", "3"],
        help=(
            "Clutter SNR offsets (dB) relative to nominal. "
            "Accepts space- or comma-separated specs, e.g. "
            "'--clutter-offsets-db -3 0 3' or '--clutter-offsets-db \"-3,0,3\"'."
        ),
    )
    ap.add_argument(
        "--alias-scales",
        type=str,
        default="0.8,1.0,1.2",
        help="Comma-separated alias_amp_scale values.",
    )
    ap.add_argument(
        "--seed",
        type=int,
        default=1,
        help="Simulation seed to use for all sweep points.",
    )
    ap.add_argument(
        "--nx",
        type=int,
        default=240,
        help="Lateral grid size (Nx) for k-Wave brain runs.",
    )
    ap.add_argument(
        "--ny",
        type=int,
        default=240,
        help="Depth grid size (Ny) for k-Wave brain runs.",
    )
    ap.add_argument(
        "--angles",
        type=str,
        default="-12,-6,0,6,12",
        help="Comma-separated steering angles in degrees.",
    )
    ap.add_argument(
        "--ensembles",
        type=int,
        default=5,
        help="Number of angle ensembles.",
    )
    ap.add_argument(
        "--pulses",
        type=int,
        default=64,
        help="Synthetic slow-time pulses per ensemble.",
    )
    ap.add_argument(
        "--fprs",
        type=str,
        default="1e-4,3e-4,1e-3",
        help="Comma-separated FPR values at which to record TPR.",
    )
    ap.add_argument(
        "--summary-json",
        type=Path,
        default=REPO / "reports" / "brain_sweep_summary.json",
        help="Path for aggregated sweep summary JSON.",
    )
    ap.add_argument(
        "--synthetic",
        action="store_true",
        help="Use synthetic RF (no k-Wave) for quick dry runs.",
    )
    ap.add_argument(
        "--dry-run",
        action="store_true",
        help="Print commands without executing them.",
    )
    return ap.parse_args()


def main() -> None:
    args = parse_args()
    regimes = [r.strip() for r in args.regimes.split(",") if r.strip()]
    prfs = _float_list(args.prfs)
    tiles = _int_list(args.tiles)
    lts = _int_list(args.lts)
    # --clutter-offsets-db may be provided as space-separated tokens
    # (e.g. ``--clutter-offsets-db -3 0 3``) or as one or more
    # comma/semicolon-separated specs (e.g. ``\"-3,0,3\"``). We support
    # both by parsing each token via _float_list and concatenating.
    clutter_offsets: List[float] = []
    for spec in args.clutter_offsets_db:
        clutter_offsets.extend(_float_list(spec))
    alias_scales = _float_list(args.alias_scales)
    fprs = _float_list(args.fprs)

    grid = build_grid(regimes, prfs, tiles, lts, clutter_offsets, alias_scales)
    print(f"[sweep] total points: {len(grid)}")

    args.out_root.mkdir(parents=True, exist_ok=True)
    rows: List[Dict[str, object]] = []
    for pt in grid:
        bundle = run_pilot_for_point(
            pt=pt,
            out_root=args.out_root,
            seed=args.seed,
            nx=args.nx,
            ny=args.ny,
            angles=args.angles,
            ensembles=args.ensembles,
            pulses=args.pulses,
            synthetic=args.synthetic,
            dry_run=args.dry_run,
        )
        if args.dry_run:
            # In dry-run mode we do not have real bundles; just log sweep tags.
            rows.append({"tag": pt.tag(), **asdict(pt)})
            continue
        row = summarize_point(pt, bundle, fprs)
        rows.append(row)

    args.summary_json.parent.mkdir(parents=True, exist_ok=True)
    args.summary_json.write_text(json.dumps(rows, indent=2))
    print(f"[sweep] wrote summary to {args.summary_json}")


if __name__ == "__main__":
    main()
