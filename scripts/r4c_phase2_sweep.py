#!/usr/bin/env python3
"""
Brain k-Wave phase-2 latency sweep helper.

Runs a small set of STAP replays over k-Wave brain seeds with different
algorithmic settings (tile stride, Doppler grid) and records
latency + PD ratios, without exploding runtime.

Usage (from repo root, inside stap-fus env):

    PYTHONPATH=. python scripts/r4c_phase2_sweep.py \
        --seeds 1 3 7 \
        --tile-strides 2 3 \
        --grid-steps 0.10 0.12 \
        --max-pts-list 5 \
        --max-configs 4

This will:
  - For each sampled (stride, grid_step_rel, max_pts) and each seed,
    run scripts/replay_stap_from_run.py against runs/pilot/r4c_kwave_seed{seed},
    writing bundles under runs/motion/r4c_sweep/...
  - Skip runs whose meta.json already exists (to avoid recomputing).
  - Produce a JSON summary at reports/r4c_phase2_sweep.json and a
    simple TSV at reports/r4c_phase2_sweep.tsv.
"""
from __future__ import annotations

import argparse
import itertools
import json
import math
import random
import subprocess
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import List, Sequence, Tuple


@dataclass
class Config:
    seed: int
    tile_stride: int
    grid_step_rel: float
    max_pts: int


@dataclass
class Result:
    seed: int
    tile_stride: int
    grid_step_rel: float
    max_pts: int
    stap_ms: float | None
    flow_ratio: float | None
    bg_ratio: float | None
    meta_path: str


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Brain k-Wave phase-2 latency sweep helper.")
    ap.add_argument(
        "--seeds",
        nargs="+",
        type=int,
        default=[1, 3, 7],
        help="Brain k-Wave seeds to evaluate (default: 1 3 7).",
    )
    ap.add_argument(
        "--tile-strides",
        nargs="+",
        type=int,
        default=[2],
        help="Tile strides to test (default: 2).",
    )
    ap.add_argument(
        "--grid-steps",
        nargs="+",
        type=float,
        default=[0.10],
        help="grid_step_rel values to test (default: 0.10).",
    )
    ap.add_argument(
        "--max-pts-list",
        nargs="+",
        type=int,
        default=[5],
        help="max_pts values for Doppler grid to test (default: 5).",
    )
    ap.add_argument(
        "--max-configs",
        type=int,
        default=None,
        help=(
            "Optional cap on number of (stride,grid_step,max_pts) "
            "combinations to sample (random subset)."
        ),
    )
    ap.add_argument(
        "--tile-debug-limit",
        type=int,
        default=None,
        help="Optional tile-debug-limit to pass to replay (for quick checks).",
    )
    ap.add_argument(
        "--out-root",
        type=Path,
        default=Path("runs/motion/r4c_sweep"),
        help="Root directory for sweep outputs.",
    )
    ap.add_argument(
        "--reports-root",
        type=Path,
        default=Path("reports"),
        help="Directory for summary reports.",
    )
    ap.add_argument(
        "--dry-run",
        action="store_true",
        help="Print planned commands without executing them.",
    )
    return ap.parse_args()


def iter_configs(
    seeds: Sequence[int],
    tile_strides: Sequence[int],
    grid_steps: Sequence[float],
    max_pts_list: Sequence[int],
    max_configs: int | None,
) -> List[Config]:
    combos: List[Tuple[int, float, int]] = list(
        itertools.product(tile_strides, grid_steps, max_pts_list)
    )
    if max_configs is not None and max_configs < len(combos):
        combos = random.sample(combos, max_configs)
    configs: List[Config] = []
    for seed in seeds:
        for stride, grid_step, max_pts in combos:
            configs.append(
                Config(
                    seed=seed,
                    tile_stride=int(stride),
                    grid_step_rel=float(grid_step),
                    max_pts=int(max_pts),
                )
            )
    return configs


def _format_grid_step_tag(gs: float) -> str:
    # e.g., 0.10 -> "010", 0.08 -> "008"
    return f"{int(round(gs * 1000.0)):03d}"


def replay_for_config(
    cfg: Config, out_root: Path, tile_debug_limit: int | None, dry_run: bool
) -> Result | None:
    seed = cfg.seed
    src_dir = Path(f"runs/pilot/r4c_kwave_seed{seed}")
    if not src_dir.exists():
        print(f"[skip seed{seed}] missing {src_dir}")
        return None

    gs_tag = _format_grid_step_tag(cfg.grid_step_rel)
    cfg_id = f"stride{cfg.tile_stride}_gs{gs_tag}_k{cfg.max_pts}"
    out_dir = out_root / cfg_id
    dataset_name = f"pw_7.5MHz_5ang_5ens_320T_seed{seed}"
    bundle_dir = out_dir / dataset_name
    meta_path = bundle_dir / "meta.json"
    if meta_path.exists():
        # Already computed; just load metrics.
        try:
            meta = json.loads(meta_path.read_text())
            tele = meta.get("stap_fallback_telemetry", {})
            pd_stats = json.loads((bundle_dir / "pd_stats.json").read_text())
            return Result(
                seed=seed,
                tile_stride=cfg.tile_stride,
                grid_step_rel=cfg.grid_step_rel,
                max_pts=cfg.max_pts,
                stap_ms=float(tele.get("stap_ms")) if tele.get("stap_ms") is not None else None,
                flow_ratio=(
                    float(pd_stats.get("flow_ratio"))
                    if pd_stats.get("flow_ratio") is not None
                    else None
                ),
                bg_ratio=(
                    float(pd_stats.get("bg_ratio"))
                    if pd_stats.get("bg_ratio") is not None
                    else None
                ),
                meta_path=str(meta_path),
            )
        except Exception as exc:  # pragma: no cover - defensive
            print(f"[warn seed{seed}] failed to load existing meta {meta_path}: {exc}")
            return None

    out_dir.mkdir(parents=True, exist_ok=True)
    cmd: List[str] = [
        sys.executable,
        "scripts/replay_stap_from_run.py",
        "--src",
        str(src_dir),
        "--out",
        str(out_dir),
        "--baseline",
        "mc_svd",
        "--reg-enable",
        "--reg-method",
        "phasecorr",
        "--reg-subpixel",
        "4",
        "--reg-reference",
        "median",
        "--svd-rank",
        "8",
        "--msd-contrast-alpha",
        "0.8",
        "--tile-h",
        "8",
        "--tile-w",
        "8",
        "--tile-stride",
        str(cfg.tile_stride),
        "--lt",
        "8",
        "--psd-br-flow-low",
        "30",
        "--psd-br-flow-high",
        "250",
        "--psd-br-alias-center",
        "575",
        "--psd-br-alias-width",
        "175",
        "--flow-doppler-min-hz",
        "60",
        "--flow-doppler-max-hz",
        "180",
        "--bg-alias-hz",
        "650",
        "--bg-alias-fraction",
        "0.3",
        "--bg-alias-depth-min-frac",
        "0.30",
        "--bg-alias-depth-max-frac",
        "0.70",
        "--bg-alias-jitter-hz",
        "50.0",
        "--flow-mask-mode",
        "pd_auto",
        "--flow-mask-pd-quantile",
        "0.995",
        "--flow-mask-depth-min-frac",
        "0.25",
        "--flow-mask-depth-max-frac",
        "0.85",
        "--flow-mask-dilate-iters",
        "2",
        "--aperture-phase-std",
        "0.8",
        "--aperture-phase-corr-len",
        "14",
        "--aperture-phase-seed",
        "111",
        "--clutter-beta",
        "1.0",
        "--clutter-snr-db",
        "20.0",
        "--clutter-depth-min-frac",
        "0.20",
        "--clutter-depth-max-frac",
        "0.95",
        "--fd-span-mode",
        "psd",
        "--fd-span-rel",
        "0.30,1.10",
        "--grid-step-rel",
        f"{cfg.grid_step_rel:.4f}",
        "--max-pts",
        str(cfg.max_pts),
        "--fd-min-pts",
        "3",
        "--fd-min-abs-hz",
        "0.0",
        "--stap-device",
        "cuda",
    ]
    if tile_debug_limit is not None:
        cmd.extend(["--tile-debug-limit", str(tile_debug_limit)])

    print(f"[run seed{seed}] {cfg_id} -> {out_dir}")
    if dry_run:
        print("  CMD:", " ".join(cmd))
        return None

    try:
        subprocess.run(cmd, check=True)
    except subprocess.CalledProcessError as exc:
        print(f"[error seed{seed}] replay_stap_from_run failed: {exc}")
        return None

    if not meta_path.exists():
        print(f"[warn seed{seed}] meta.json missing at {meta_path}")
        return None

    meta = json.loads(meta_path.read_text())
    tele = meta.get("stap_fallback_telemetry", {})
    pd_stats = json.loads((bundle_dir / "pd_stats.json").read_text())
    return Result(
        seed=seed,
        tile_stride=cfg.tile_stride,
        grid_step_rel=cfg.grid_step_rel,
        max_pts=cfg.max_pts,
        stap_ms=float(tele.get("stap_ms")) if tele.get("stap_ms") is not None else None,
        flow_ratio=(
            float(pd_stats.get("flow_ratio")) if pd_stats.get("flow_ratio") is not None else None
        ),
        bg_ratio=float(pd_stats.get("bg_ratio")) if pd_stats.get("bg_ratio") is not None else None,
        meta_path=str(meta_path),
    )


def main() -> int:
    args = parse_args()
    args.out_root.mkdir(parents=True, exist_ok=True)
    args.reports_root.mkdir(parents=True, exist_ok=True)

    configs = iter_configs(
        seeds=args.seeds,
        tile_strides=args.tile_strides,
        grid_steps=args.grid_steps,
        max_pts_list=args.max_pts_list,
        max_configs=args.max_configs,
    )
    print(f"[sweep] {len(configs)} seed/config combinations planned.")

    results: List[Result] = []
    for cfg in configs:
        res = replay_for_config(cfg, args.out_root, args.tile_debug_limit, args.dry_run)
        if res is not None:
            results.append(res)

    if args.dry_run:
        print("[sweep] dry-run complete; no results to save.")
        return 0

    summary = {
        "configs": [asdict(r) for r in results],
        "tile_strides": args.tile_strides,
        "grid_steps": args.grid_steps,
        "max_pts_list": args.max_pts_list,
        "seeds": args.seeds,
    }
    json_path = args.reports_root / "r4c_phase2_sweep.json"
    json_path.write_text(json.dumps(summary, indent=2))

    tsv_path = args.reports_root / "r4c_phase2_sweep.tsv"
    with tsv_path.open("w", encoding="utf-8") as f:
        f.write("seed\tstride\tgrid_step\tmax_pts\tstap_ms\tflow_ratio\tbg_ratio\tmeta\n")
        for r in results:
            f.write(
                f"{r.seed}\t{r.tile_stride}\t{r.grid_step_rel:.4f}\t{r.max_pts}\t"
                f"{r.stap_ms if r.stap_ms is not None else math.nan}\t"
                f"{r.flow_ratio if r.flow_ratio is not None else math.nan}\t"
                f"{r.bg_ratio if r.bg_ratio is not None else math.nan}\t"
                f"{r.meta_path}\n"
            )

    print(f"[sweep] saved {len(results)} results to {json_path} and {tsv_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
