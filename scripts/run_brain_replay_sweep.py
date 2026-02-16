#!/usr/bin/env python3
"""Brain STAP PD robustness sweeps using the replay pipeline.

This script mirrors ``run_brain_sweep.py`` but, instead of relying on the
integrated PD/STAP path inside ``sim/kwave/pilot_motion.py``, it:

  1) Generates a Brain-* k-Wave pilot (OpenSkull / SkullOR) for each
     (regime, PRF) pair via ``pilot_motion.py``.
  2) For each sweep point (regime, PRF, tile, Lt, clutter/alias knobs),
     replays the STAP PD stage on that pilot using
     ``scripts/replay_stap_from_run.py`` with a clinical-style PD
     configuration, but allowing tile geometry and Lt to vary.
  3) Runs ``scripts/hab_contract_check.py --score-mode pd`` on the replayed
     bundle to extract TPR at the requested FPRs.

The goal is to harden the *replay* clinical STAP PD profile (the one used in
the latency suite) against realistic variation in PRF, tile geometry, Lt,
clutter SNR, and alias amplitude.

Usage (example):

    PYTHONPATH=. python scripts/run_brain_replay_sweep.py \
        --regimes open,skullor \
        --prfs 1200,1500,2000 \
        --tiles 6,8,12 \
        --lts 6,8,10 \
        --clutter-offsets-db -3 0 3 \
        --alias-scales 0.8,1.0,1.2 \
        --summary-json reports/brain_replay_sweep_summary.json

This will write a JSON summary with one row per sweep point containing
regime, PRF, tile/Lt, clutter/alias knobs, latency, coverage, and TPR for
MC–SVD PD vs STAP PD at the chosen FPRs.
"""

from __future__ import annotations

import argparse
import json
import os
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
    """Construct the sweep grid of configuration points.

    We mirror the grid used in ``run_brain_sweep.py``: regimes × PRF × tile ×
    Lt × clutter_offset × alias_scale. Stride is chosen as a simple function
    of tile (6→2, 8→3, 12→4) to keep reasonable overlap.
    """

    stride_for_tile: Dict[int, int] = {6: 2, 8: 3, 12: 4}
    points: List[SweepPoint] = []
    for regime in regimes:
        r = regime.strip()
        if not r:
            continue
        if r not in {"open", "skullor"}:
            raise ValueError(f"Unsupported regime '{regime}' (expected open or skullor)")
        for prf in prfs:
            for tile in tiles:
                if tile not in stride_for_tile:
                    raise ValueError(
                        f"No stride defined for tile={tile}; "
                        "extend stride_for_tile in run_brain_replay_sweep.py."
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
    """Nominal clutter SNR (dB) used as the center for ±3 dB offsets."""

    # Keep this simple and explicit so we do not depend on Makefile indirection.
    return 25.0 if regime in {"open", "skullor"} else 25.0


def _profile_for_regime(regime: str) -> str:
    if regime == "open":
        return "Brain-OpenSkull"
    if regime == "skullor":
        return "Brain-SkullOR"
    raise ValueError(f"Unsupported regime '{regime}'")


def _pilot_root(out_root: Path, regime: str, prf_hz: float) -> Path:
    """Directory under which we keep pilots for a given (regime, PRF).

    All sweep points sharing the same (regime, PRF) reuse the same pilot,
    while STAP/Lt/tile variations are applied only in the replay stage.
    """

    tag = f"{regime}_prf{int(prf_hz)}"
    return out_root / "pilots" / tag


def _bundle_dir_from_pilot(pilot_dir: Path) -> Path | None:
    """Return the single pw_* bundle directory inside a pilot, if present."""

    cands = sorted(pilot_dir.glob("pw_*"))
    return cands[0] if cands else None


def ensure_pilot_for_regime_prf(
    regime: str,
    prf_hz: float,
    out_root: Path,
    seed: int,
    nx: int,
    ny: int,
    angles: str,
    ensembles: int,
    pulses: int,
    clutter_offset_db: float,
    alias_scale: float,
    synthetic: bool,
    dry_run: bool,
) -> Path:
    """Ensure a Brain-* pilot exists for the given (regime, PRF).

    If a pilot already exists (identified by a pw_* bundle directory with
    meta.json), we reuse it. Otherwise we invoke ``sim/kwave/pilot_motion.py``
    with the appropriate Brain-* profile and nuisance knobs.
    """

    pilot_dir = _pilot_root(out_root, regime, prf_hz)
    pilot_dir.mkdir(parents=True, exist_ok=True)

    existing_bundle = _bundle_dir_from_pilot(pilot_dir)
    if existing_bundle is not None and (existing_bundle / "meta.json").exists():
        return existing_bundle

    profile = _profile_for_regime(regime)
    nominal_clutter = _nominal_clutter_snr_db_for_regime(regime)
    clutter_snr_db = nominal_clutter + clutter_offset_db

    cmd: List[str] = [
        sys.executable,
        str(REPO / "sim" / "kwave" / "pilot_motion.py"),
        "--out",
        str(pilot_dir),
        "--profile",
        profile,
    ]

    # pilot_motion.py declares --angles with nargs="+", and then supports
    # either space-separated or comma-separated specifications. Here we
    # normalize any comma/semicolon-separated spec into a list of separate
    # tokens so that negative angles (e.g. -12) are passed unambiguously.
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
        str(prf_hz),
        "--seed",
        str(seed),
        "--Nx",
        str(nx),
        "--Ny",
        str(ny),
        # Tile/Lt settings for the pilot affect only the integrated acceptance
        # path inside pilot_motion; the replay STAP configuration is
        # controlled separately in this script.
        "--tile-h",
        "8",
        "--tile-w",
        "8",
        "--tile-stride",
        "3",
        "--lt",
        "8",
        # Baseline + scoring for the acceptance bundle (we rely on replay for
        # final PD ROC, but pilot_motion still expects these).
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
        f"{alias_scale:.3f}",
    ]

    if synthetic:
        cmd.append("--synthetic")

    print("[pilot]", " ".join(cmd))
    if not dry_run:
        subprocess.run(cmd, check=True)

    bundle = _bundle_dir_from_pilot(pilot_dir)
    if bundle is None:
        if dry_run:
            return pilot_dir
        raise RuntimeError(f"No bundle produced under {pilot_dir}")
    return bundle


def run_replay_for_point(
    pt: SweepPoint,
    pilot_bundle: Path,
    out_root: Path,
    fprs: Iterable[float],
) -> Dict[str, object]:
    """Replay STAP PD on a pilot bundle for a single SweepPoint.

    This calls ``replay_stap_from_run.py`` with a clinical-style PD
    configuration (matching the latency suite) but keeps tile geometry and
    Lt from the sweep point. It then runs ``hab_contract_check.py`` to
    extract ROC metrics and returns a summary row.
    """

    # Out directory for this replay configuration.
    replay_dir = out_root / pt.regime / pt.tag()
    replay_dir.mkdir(parents=True, exist_ok=True)

    # Environment: enable PD-only fast path and snapshot caps as in the
    # clinical STAP preset.
    env = os.environ.copy()
    env.setdefault("STAP_SNAPSHOT_STRIDE", "4")
    env.setdefault("STAP_MAX_SNAPSHOTS", "64")
    env.setdefault("STAP_FAST_PD_ONLY", "1")

    # Reuse the angles/RF from the pilot; replay_stap_from_run.py expects the
    # source directory that contains angle_* subdirectories.
    src_root = pilot_bundle.parent

    cmd: List[str] = [
        sys.executable,
        str(REPO / "scripts" / "replay_stap_from_run.py"),
        "--src",
        str(src_root),
        "--out",
        str(replay_dir),
        "--baseline",
        "mc_svd",
        "--svd-profile",
        "literature",
        "--reg-disable",
        "--stap-device",
        "cuda",
        "--score-mode",
        "pd",
        "--time-window-length",
        "64",
        # Tile/STAP geometry from sweep point.
        "--tile-h",
        str(pt.tile),
        "--tile-w",
        str(pt.tile),
        "--tile-stride",
        str(pt.stride),
        "--lt",
        str(pt.lt),
        # Clinical-style covariance / MSD configuration.
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
        "--fd-min-pts",
        "3",
        "--constraint-mode",
        "exp+deriv",
        "--constraint-ridge",
        "0.18",
        "--mvdr-load-mode",
        "auto",
        "--mvdr-auto-kappa",
        "120.0",
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
        # Whitened band-ratio telemetry parameters.
        "--band-ratio-mode",
        "whitened",
        "--psd-br-flow-low",
        "30.0",
        "--psd-br-flow-high",
        "220.0",
        "--psd-br-alias-center",
        "650.0",
        "--psd-br-alias-width",
        "140.0",
        # Evaluation masks should be simulator truth in Brain-* runs.
        "--flow-mask-mode",
        "default",
        "--flow-mask-pd-quantile",
        "0.995",
        "--flow-mask-depth-min-frac",
        "0.25",
        "--flow-mask-depth-max-frac",
        "0.85",
        "--flow-mask-dilate-iters",
        "2",
    ]

    print("[replay]", " ".join(cmd))
    subprocess.run(cmd, check=True, env=env)

    # The replay creates a new bundle under replay_dir/pw_*.
    replay_bundle = _bundle_dir_from_pilot(replay_dir)
    if replay_bundle is None:
        raise RuntimeError(f"No replay bundle produced under {replay_dir}")

    # Read latency/coverage telemetry from the replay bundle meta.
    meta_path = replay_bundle / "meta.json"
    meta = json.loads(meta_path.read_text())
    tele = meta.get("stap_fallback_telemetry", {}) or {}

    baseline_ms = float(tele.get("baseline_ms", 0.0))
    stap_ms = float(tele.get("stap_total_ms", 0.0))

    # Coverage fraction for conditional STAP: 1 - skipped_flow0 / total_tiles.
    tile_hw = tuple(int(v) for v in meta.get("tile_hw", [pt.tile, pt.tile]))
    stride = int(meta.get("tile_stride", pt.stride))
    pd_base_path = replay_bundle / "pd_base.npy"
    H = W = None
    tile_count = None
    if pd_base_path.exists():
        import numpy as np

        pd_base = np.load(pd_base_path)
        H, W = pd_base.shape
    if H is not None and W is not None:
        th, tw = tile_hw
        ny = max(0, (H - th) // stride + 1)
        nx = max(0, (W - tw) // stride + 1)
        tile_count = ny * nx

    skipped = float(tele.get("stap_tiles_skipped_flow0", 0.0))
    coverage = None
    if tile_count and tile_count > 0:
        coverage = 1.0 - skipped / float(tile_count)

    # Run hab_contract_check.py to extract TPR at the requested FPRs.
    fpr_args = [f"{fpr:g}" for fpr in fprs]
    cmd_hab: List[str] = [
        sys.executable,
        str(REPO / "scripts" / "hab_contract_check.py"),
        str(replay_bundle),
        "--score-mode",
        "pd",
        "--fprs",
        *fpr_args,
    ]
    result = subprocess.run(cmd_hab, capture_output=True, text=True, check=True)
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
        "bundle": str(replay_bundle),
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
    ap = argparse.ArgumentParser(
        description="Brain STAP PD robustness sweeps (replay pipeline)."
    )
    ap.add_argument(
        "--out-root",
        type=Path,
        default=REPO / "runs" / "brain_replay_sweep",
        help="Root directory for pilot runs and replay bundles.",
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
        help="Comma-separated tile sizes (tile height == tile width).",
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
        default=REPO / "reports" / "brain_replay_sweep_summary.json",
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

    # --clutter-offsets-db may be provided as space-separated tokens or as one
    # or more comma/semicolon-separated specs; parse each token via _float_list
    # and concatenate.
    clutter_offsets: List[float] = []
    for spec in args.clutter_offsets_db:
        clutter_offsets.extend(_float_list(spec))
    alias_scales = _float_list(args.alias_scales)
    fprs = _float_list(args.fprs)

    grid = build_grid(regimes, prfs, tiles, lts, clutter_offsets, alias_scales)
    print(f"[replay-sweep] total points: {len(grid)}")

    args.out_root.mkdir(parents=True, exist_ok=True)
    rows: List[Dict[str, object]] = []

    # Cache pilots by (regime, PRF) so we do not resimulate unnecessarily.
    pilot_cache: Dict[Tuple[str, float], Path] = {}

    for pt in grid:
        key = (pt.regime, pt.prf_hz)
        if key in pilot_cache:
            pilot_bundle = pilot_cache[key]
        else:
            pilot_bundle = ensure_pilot_for_regime_prf(
                regime=pt.regime,
                prf_hz=pt.prf_hz,
                out_root=args.out_root,
                seed=args.seed,
                nx=args.nx,
                ny=args.ny,
                angles=args.angles,
                ensembles=args.ensembles,
                pulses=args.pulses,
                clutter_offset_db=pt.clutter_offset_db,
                alias_scale=pt.alias_scale,
                synthetic=args.synthetic,
                dry_run=args.dry_run,
            )
            pilot_cache[key] = pilot_bundle

        if args.dry_run:
            rows.append({"tag": pt.tag(), **asdict(pt)})
            continue

        row = run_replay_for_point(pt, pilot_bundle, args.out_root, fprs)
        rows.append(row)

    args.summary_json.parent.mkdir(parents=True, exist_ok=True)
    args.summary_json.write_text(json.dumps(rows, indent=2))
    print(f"[replay-sweep] wrote summary to {args.summary_json}")


if __name__ == "__main__":
    main()
