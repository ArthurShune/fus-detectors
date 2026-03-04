#!/usr/bin/env python3
"""
Brain-OpenSkull fixed-profile sensitivity sweep (1D).

This script addresses a reviewer concern that a covariance-whitened detector may
be brittle to covariance conditioning and geometry choices by running simple 1D
sweeps over:
  - covariance diagonal loading (--diag-load),
  - tile size (--tile-h/--tile-w; stride chosen to roughly preserve overlap),
  - slow-time aperture (--lt).

For each configuration, we replay the labeled Brain-OpenSkull k-Wave pilot under
the same MC--SVD baseline and STAP score conventions used in the main Brain-*
tables (strict-tail operating points at fixed FPR targets, summarized across
five disjoint 64-frame windows as median and IQR).

Outputs (defaults):
  - reports/brain_openskull_profile_sensitivity.csv
  - reports/brain_openskull_profile_sensitivity.json
  - figs/paper/brain_openskull_profile_sensitivity.pdf
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import shutil
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np


def _parse_csv_ints(s: str) -> list[int]:
    out: list[int] = []
    for tok in (s or "").split(","):
        tok = tok.strip()
        if not tok:
            continue
        out.append(int(tok))
    return out


def _parse_csv_floats(s: str) -> list[float]:
    out: list[float] = []
    for tok in (s or "").split(","):
        tok = tok.strip()
        if not tok:
            continue
        out.append(float(tok))
    return out


def _finite(vals: Iterable[float]) -> np.ndarray:
    v = np.asarray(list(vals), dtype=np.float64)
    return v[np.isfinite(v)]


def _quantile_summary(vals: Iterable[float]) -> tuple[float, float, float]:
    v = _finite(vals)
    if v.size <= 0:
        return float("nan"), float("nan"), float("nan")
    med = float(np.quantile(v, 0.5))
    q25 = float(np.quantile(v, 0.25))
    q75 = float(np.quantile(v, 0.75))
    return med, q25, q75


def _tau_for_fpr(neg: np.ndarray, alpha: float) -> float:
    n = int(neg.size)
    if n <= 0:
        return float("inf")
    neg_sorted = np.sort(neg)
    q = 1.0 - float(alpha)
    k = int(np.ceil(q * n)) - 1
    k = max(0, min(n - 1, k))
    return float(neg_sorted[k])


def _load_score_and_masks(bundle_dir: Path) -> tuple[np.ndarray, np.ndarray]:
    score = np.load(bundle_dir / "score_stap_preka.npy").astype(np.float64, copy=False)
    mf = np.load(bundle_dir / "mask_flow.npy").astype(bool, copy=False)
    mb = np.load(bundle_dir / "mask_bg.npy").astype(bool, copy=False)
    pos = score[mf].ravel()
    neg = score[mb].ravel()
    return pos, neg


def _tpr_at_alpha(bundle_dir: Path, alpha: float) -> float:
    pos, neg = _load_score_and_masks(bundle_dir)
    tau = _tau_for_fpr(neg, alpha)
    if not np.isfinite(tau) or pos.size <= 0:
        return 0.0
    return float(np.mean(pos >= tau))


def _glob_windows(root: Path) -> list[Path]:
    wins = [p for p in root.glob("pw_*_win*_off*") if p.is_dir()]
    wins.sort()
    return wins


def _tag_float(x: float) -> str:
    # Stable-ish tag for directory names (avoid scientific notation in paths).
    s = f"{x:.4f}".rstrip("0").rstrip(".")
    return s.replace("-", "m").replace(".", "p")


def _round_half_up(x: float) -> int:
    return int(math.floor(x + 0.5))


def _stride_for_tile(tile: int, *, target_overlap: float) -> int:
    raw = (1.0 - float(target_overlap)) * float(tile)
    stride = _round_half_up(raw)
    stride = max(1, min(tile - 1, stride))
    return int(stride)


@dataclass(frozen=True)
class Config:
    sweep: str
    value: float
    tile_hw: tuple[int, int]
    tile_stride: int
    Lt: int
    diag_load: float
    out_dir: Path


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--src",
        type=Path,
        default=Path("runs/pilot/r4c_kwave_seed1"),
        help="Brain-OpenSkull source run root.",
    )
    ap.add_argument(
        "--out-root",
        type=Path,
        default=Path("runs/pilot/brain_openskull_profile_sensitivity_r1"),
        help="Root directory to store generated replay bundles.",
    )
    ap.add_argument(
        "--shared-default-dir",
        type=Path,
        default=None,
        help=(
            "Optional existing bundle dir to reuse as the shared default point "
            "(tile=8/stride=3, Lt=8, diag-load=0.07). If provided, the script will not "
            "overwrite or delete it."
        ),
    )
    ap.add_argument(
        "--window-length",
        type=int,
        default=64,
        help="Replay window length.",
    )
    ap.add_argument(
        "--window-offsets",
        type=str,
        default="0,64,128,192,256",
        help="Comma-separated replay window offsets.",
    )
    ap.add_argument(
        "--alphas",
        type=str,
        default="1e-4,3e-4,1e-3",
        help="Comma-separated strict-tail FPR targets.",
    )
    ap.add_argument(
        "--stap-device",
        type=str,
        default="cuda",
        help="STAP device passed to replay_stap_from_run.py.",
    )
    ap.add_argument(
        "--diag-loads",
        type=str,
        default="0.03,0.07,0.12",
        help="Comma-separated diag-load values to sweep.",
    )
    ap.add_argument(
        "--tile-sizes",
        type=str,
        default="6,8,10",
        help="Comma-separated square tile sizes to sweep (tile-h=tile-w).",
    )
    ap.add_argument(
        "--lts",
        type=str,
        default="6,8,10",
        help="Comma-separated Lt values to sweep.",
    )
    ap.add_argument(
        "--skip-replay",
        action="store_true",
        help="Skip replay generation and only summarize existing bundles.",
    )
    ap.add_argument(
        "--autogen-missing",
        action="store_true",
        help="Generate missing bundles under --out-root (default: off).",
    )
    ap.add_argument(
        "--skip-plot",
        action="store_true",
        help="Skip generating the PDF figure.",
    )
    ap.add_argument(
        "--out-csv",
        type=Path,
        default=Path("reports/brain_openskull_profile_sensitivity.csv"),
        help="Output CSV path.",
    )
    ap.add_argument(
        "--out-json",
        type=Path,
        default=Path("reports/brain_openskull_profile_sensitivity.json"),
        help="Output JSON path.",
    )
    ap.add_argument(
        "--out-fig",
        type=Path,
        default=Path("figs/paper/brain_openskull_profile_sensitivity.pdf"),
        help="Output PDF figure path.",
    )
    return ap.parse_args()


def _is_complete(out_dir: Path, *, expected_windows: int) -> bool:
    wins = _glob_windows(out_dir)
    if len(wins) != expected_windows:
        return False
    for w in wins:
        if not (w / "score_stap_preka.npy").is_file():
            return False
        if not (w / "mask_flow.npy").is_file():
            return False
        if not (w / "mask_bg.npy").is_file():
            return False
    return True


def _run_replay(
    *,
    src: Path,
    out_dir: Path,
    window_length: int,
    window_offsets: Sequence[int],
    stap_device: str,
    tile_hw: tuple[int, int],
    tile_stride: int,
    Lt: int,
    diag_load: float,
) -> None:
    # Keep these flags explicit and deterministic (matches the matrix-mode pilot discipline).
    cmd: list[str] = [
        sys.executable,
        "-u",
        "scripts/replay_stap_from_run.py",
        "--src",
        str(src),
        "--out",
        str(out_dir),
        "--stap-profile",
        "clinical",
        "--profile",
        "Brain-OpenSkull",
        "--baseline",
        "mc_svd",
        "--stap-device",
        str(stap_device),
        "--stap-detector-variant",
        "msd_ratio",
        "--stap-debug-samples",
        "0",
        "--score-mode",
        "pd",
        "--flow-mask-mode",
        "default",
        "--time-window-length",
        str(int(window_length)),
        "--synth-amp-jitter",
        "0.0",
        "--synth-phase-jitter",
        "0.0",
        "--synth-noise-level",
        "0.0",
        "--synth-shift-max-px",
        "0",
        "--baseline-support",
        "full",
        "--svd-profile",
        "literature",
        "--svd-energy-frac",
        "0.9",
        "--reg-enable",
        "--reg-method",
        "phasecorr",
        "--reg-subpixel",
        "4",
        "--reg-reference",
        "median",
        "--flow-doppler-min-hz",
        "60.0",
        "--flow-doppler-max-hz",
        "180.0",
        "--bg-alias-hz",
        "650.0",
        "--bg-alias-fraction",
        "0.3",
        "--bg-alias-depth-min-frac",
        "0.3",
        "--bg-alias-depth-max-frac",
        "0.7",
        "--bg-alias-jitter-hz",
        "50.0",
        "--aperture-phase-std",
        "0.8",
        "--aperture-phase-corr-len",
        "14.0",
        "--aperture-phase-seed",
        "111",
        "--clutter-beta",
        "1.0",
        "--clutter-snr-db",
        "20.0",
        "--clutter-mode",
        "lowrank",
        "--clutter-rank",
        "3",
        "--clutter-depth-min-frac",
        "0.2",
        "--clutter-depth-max-frac",
        "0.95",
        "--stap-conditional-disable",
        "--tile-h",
        str(int(tile_hw[0])),
        "--tile-w",
        str(int(tile_hw[1])),
        "--tile-stride",
        str(int(tile_stride)),
        "--lt",
        str(int(Lt)),
        "--diag-load",
        str(float(diag_load)),
    ]
    for off in window_offsets:
        cmd += ["--time-window-offset", str(int(off))]

    env = os.environ.copy()
    repo = str(Path(__file__).resolve().parents[1])
    env["PYTHONPATH"] = repo + (os.pathsep + env["PYTHONPATH"] if env.get("PYTHONPATH") else "")

    out_dir.parent.mkdir(parents=True, exist_ok=True)
    print(f"[profile_sensitivity] replay -> {out_dir}", flush=True)
    subprocess.run(cmd, check=True, env=env)


def _summarize_config(out_dir: Path, alphas: Sequence[float]) -> dict[float, dict[str, float]]:
    wins = _glob_windows(out_dir)
    out: dict[float, dict[str, float]] = {}
    for a in alphas:
        tprs = [_tpr_at_alpha(d, float(a)) for d in wins]
        med, q25, q75 = _quantile_summary(tprs)
        out[float(a)] = {"tpr_med": float(med), "tpr_q25": float(q25), "tpr_q75": float(q75)}
    return out


def _write_csv(out_path: Path, rows: list[dict[str, object]]) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()) if rows else [])
        if rows:
            w.writeheader()
            for r in rows:
                w.writerow(r)


def _plot_pdf(out_path: Path, payload: dict[str, object]) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    alphas: list[float] = [float(a) for a in payload["alphas"]]  # type: ignore[assignment]
    sweeps: dict[str, object] = payload["sweeps"]  # type: ignore[assignment]

    # Use plain-text legend labels (avoid backend-specific mathtext parsing
    # edge cases inside tight_layout/legend bbox computation).
    alpha_labels = {
        1e-4: "1e-4",
        3e-4: "3e-4",
        1e-3: "1e-3",
    }
    alpha_str = lambda a: alpha_labels.get(float(a), f"{float(a):g}")

    fig, axes = plt.subplots(1, 3, figsize=(10.8, 3.2), sharey=True)
    panel_specs = [
        ("diag_load", r"diag-load $\lambda$", axes[0]),
        ("tile_size", r"tile size (px)", axes[1]),
        ("Lt", r"slow-time aperture $L_t$", axes[2]),
    ]

    for key, xlabel, ax in panel_specs:
        spec = sweeps[key]  # type: ignore[index]
        xs: list[float] = [float(v) for v in spec["values"]]  # type: ignore[index]
        for a in alphas:
            meds = [float(spec["stats"][str(v)][str(a)]["tpr_med"]) for v in xs]  # type: ignore[index]
            q25 = [float(spec["stats"][str(v)][str(a)]["tpr_q25"]) for v in xs]  # type: ignore[index]
            q75 = [float(spec["stats"][str(v)][str(a)]["tpr_q75"]) for v in xs]  # type: ignore[index]
            yerr_lo = [m - l for m, l in zip(meds, q25)]
            yerr_hi = [h - m for m, h in zip(meds, q75)]
            ax.errorbar(
                xs,
                meds,
                yerr=[yerr_lo, yerr_hi],
                capsize=2,
                marker="o",
                linewidth=1.2,
                markersize=3.5,
                label=f"α={alpha_str(a)}",
            )
        ax.set_xlabel(xlabel)
        ax.grid(True, alpha=0.3)

    axes[0].set_ylabel("TPR (median/IQR over 5 windows)")
    axes[0].set_ylim(0.0, 1.0)
    axes[0].legend(loc="lower right", fontsize=8, frameon=True)

    fig.tight_layout(pad=0.6, w_pad=0.8)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    args = parse_args()

    window_offsets = _parse_csv_ints(args.window_offsets)
    alphas = _parse_csv_floats(args.alphas)
    expected_windows = len(window_offsets)

    default_tile = 8
    default_stride = 3
    default_overlap = 1.0 - (default_stride / default_tile)
    default_Lt = 8
    default_diag_load = 0.07
    shared_default_dir = (
        Path(args.shared_default_dir)
        if args.shared_default_dir is not None
        else args.out_root / "tile" / f"tile_{default_tile}_s{default_stride}"
    )
    protected_dirs = {shared_default_dir} if args.shared_default_dir is not None else set()

    diag_loads = _parse_csv_floats(args.diag_loads)
    tile_sizes = _parse_csv_ints(args.tile_sizes)
    Lts = _parse_csv_ints(args.lts)

    configs: list[Config] = []

    for v in diag_loads:
        out_dir = args.out_root / "diagload" / f"diagload_{_tag_float(v)}"
        if abs(float(v) - default_diag_load) <= 1e-12:
            out_dir = shared_default_dir
        configs.append(
            Config(
                sweep="diag_load",
                value=float(v),
                tile_hw=(default_tile, default_tile),
                tile_stride=default_stride,
                Lt=default_Lt,
                diag_load=float(v),
                out_dir=out_dir,
            )
        )

    for t in tile_sizes:
        stride = _stride_for_tile(int(t), target_overlap=default_overlap)
        out_dir = args.out_root / "tile" / f"tile_{int(t)}_s{stride}"
        if int(t) == default_tile and int(stride) == default_stride:
            out_dir = shared_default_dir
        configs.append(
            Config(
                sweep="tile_size",
                value=float(t),
                tile_hw=(int(t), int(t)),
                tile_stride=int(stride),
                Lt=default_Lt,
                diag_load=default_diag_load,
                out_dir=out_dir,
            )
        )

    for Lt in Lts:
        out_dir = args.out_root / "lt" / f"lt_{int(Lt)}"
        if int(Lt) == default_Lt:
            out_dir = shared_default_dir
        configs.append(
            Config(
                sweep="Lt",
                value=float(Lt),
                tile_hw=(default_tile, default_tile),
                tile_stride=default_stride,
                Lt=int(Lt),
                diag_load=default_diag_load,
                out_dir=out_dir,
            )
        )

    if not args.skip_replay:
        for cfg in configs:
            if _is_complete(cfg.out_dir, expected_windows=expected_windows):
                continue
            if not args.autogen_missing:
                raise SystemExit(
                    f"Missing or incomplete bundle dir {cfg.out_dir}. "
                    "Re-run with --autogen-missing to generate it."
                )
            if cfg.out_dir in protected_dirs:
                raise SystemExit(
                    f"Incomplete protected bundle dir {cfg.out_dir}; "
                    "either fix it manually or omit --shared-default-dir."
                )
            if cfg.out_dir.exists():
                shutil.rmtree(cfg.out_dir)
            _run_replay(
                src=Path(args.src),
                out_dir=cfg.out_dir,
                window_length=int(args.window_length),
                window_offsets=window_offsets,
                stap_device=str(args.stap_device),
                tile_hw=cfg.tile_hw,
                tile_stride=int(cfg.tile_stride),
                Lt=int(cfg.Lt),
                diag_load=float(cfg.diag_load),
            )

    sweeps: dict[str, dict[str, object]] = {
        "diag_load": {"values": sorted(set(float(v) for v in diag_loads)), "stats": {}},
        "tile_size": {"values": sorted(set(float(v) for v in tile_sizes)), "stats": {}},
        "Lt": {"values": sorted(set(float(v) for v in Lts)), "stats": {}},
    }
    config_by_key: dict[tuple[str, float], Config] = {(c.sweep, float(c.value)): c for c in configs}

    csv_rows: list[dict[str, object]] = []
    for sweep_key, spec in sweeps.items():
        for v in spec["values"]:  # type: ignore[index]
            cfg = config_by_key[(sweep_key, float(v))]
            if not _is_complete(cfg.out_dir, expected_windows=expected_windows):
                raise SystemExit(f"Incomplete bundle dir {cfg.out_dir}")
            summary = _summarize_config(cfg.out_dir, alphas)
            spec["stats"][str(v)] = {str(a): summary[float(a)] for a in alphas}  # type: ignore[index]
            for a in alphas:
                stats = summary[float(a)]
                csv_rows.append(
                    {
                        "sweep": sweep_key,
                        "value": float(v),
                        "alpha": float(a),
                        "tpr_med": float(stats["tpr_med"]),
                        "tpr_q25": float(stats["tpr_q25"]),
                        "tpr_q75": float(stats["tpr_q75"]),
                        "n_windows": int(expected_windows),
                    }
                )

    payload: dict[str, object] = {
        "src": str(Path(args.src)),
        "out_root": str(Path(args.out_root)),
        "window_length": int(args.window_length),
        "window_offsets": window_offsets,
        "alphas": [float(a) for a in alphas],
        "defaults": {
            "tile": default_tile,
            "tile_stride": default_stride,
            "Lt": default_Lt,
            "diag_load": default_diag_load,
        },
        "sweeps": sweeps,
    }

    _write_csv(Path(args.out_csv), csv_rows)
    Path(args.out_json).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out_json).write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")

    if not args.skip_plot:
        _plot_pdf(Path(args.out_fig), payload)

    print(
        f"[profile_sensitivity] wrote {args.out_csv} / {args.out_json} / {args.out_fig}",
        flush=True,
    )


if __name__ == "__main__":
    main()
