#!/usr/bin/env python3
"""KA Contract v2 falsifiability ablation (Phase 4).

This script runs a small, labeled-simulation ablation designed to make the KA
story falsifiable:

  (i)   STAP-only (no score-space KA)
  (ii)  STAP + KA Contract v2 (contract-governed; activates only in C1_SAFETY)
  (iii) STAP + KA Contract v2 (forced; ablation-only, even when contract disables)

We report ΔTPR at fixed FPR targets (and optional pAUC up to a fixed FPR) with a
paired bootstrap CI across non-overlapping time windows.

Example:
  PYTHONPATH=. python scripts/ka_contract_v2_ablation.py \
    --pilot runs/pilot/r4c_kwave_hab_contract_seed2 \
    --profile Brain-AliasContract \
    --out-root runs/ablation/ka_v2_falsifiability \
    --stap-device cpu \
    --window-length 64 \
    --window-offsets 0 64 128 192 256 \
    --summary-csv reports/ka_v2_ablation.csv \
    --summary-json reports/ka_v2_ablation.json
"""

from __future__ import annotations

import argparse
import csv
import json
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Sequence, Tuple

import numpy as np


REPO = Path(__file__).resolve().parents[1]
REPLAY = REPO / "scripts" / "replay_stap_from_run.py"


@dataclass(frozen=True)
class Condition:
    key: str
    label: str
    replay_flags: Tuple[str, ...]


CONDITIONS: Tuple[Condition, ...] = (
    Condition("stap_only", "stap_only", ()),
    Condition("ka_contract", "ka_v2_contract", ("--ka-score-contract-v2",)),
    Condition("ka_forced", "ka_v2_forced", ("--ka-score-contract-v2-force",)),
)


def _float_list(values: Sequence[str]) -> List[float]:
    out: List[float] = []
    for s in values:
        s = str(s).strip()
        if not s:
            continue
        out.append(float(s))
    return out


def _bundle_dir_from_out(out_dir: Path) -> Path:
    bundles = sorted(out_dir.glob("pw_*"))
    if not bundles:
        raise FileNotFoundError(f"No pw_* bundle directory found under {out_dir}")
    if len(bundles) > 1:
        raise RuntimeError(f"Expected a single pw_* bundle under {out_dir}, found {len(bundles)}")
    return bundles[0]


def _existing_bundle_dir(out_dir: Path) -> Path | None:
    """Return the bundle dir if it already exists (for resume), else None."""
    try:
        bundle_dir = _bundle_dir_from_out(out_dir)
    except Exception:
        return None
    if (bundle_dir / "meta.json").exists():
        return bundle_dir
    return None


def _load_score_and_masks(bundle_dir: Path) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    mask_flow = np.load(bundle_dir / "mask_flow.npy").astype(bool)
    mask_bg = np.load(bundle_dir / "mask_bg.npy").astype(bool)
    score_path = bundle_dir / "score_pd_stap.npy"
    if score_path.exists():
        score = np.load(score_path).astype(np.float64, copy=False)
    else:
        pd = np.load(bundle_dir / "pd_stap.npy").astype(np.float64, copy=False)
        score = -pd
    return score, mask_flow, mask_bg


def _tpr_at_fpr(scores_pos: np.ndarray, scores_neg: np.ndarray, fpr: float) -> Tuple[float, float]:
    """Return (threshold, TPR) for right-tail scores at the requested FPR."""
    if not (0.0 < fpr < 1.0):
        raise ValueError(f"fpr must be in (0,1), got {fpr}")
    neg = np.asarray(scores_neg, dtype=np.float64).ravel()
    pos = np.asarray(scores_pos, dtype=np.float64).ravel()
    neg = neg[np.isfinite(neg)]
    pos = pos[np.isfinite(pos)]
    if neg.size == 0 or pos.size == 0:
        return float("nan"), float("nan")
    thr = float(np.quantile(neg, 1.0 - fpr, method="linear"))
    tpr = float(np.mean(pos >= thr))
    return thr, tpr


def _roc_curve(scores_pos: np.ndarray, scores_neg: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Compute ROC (FPR, TPR) for right-tail scores using all thresholds."""
    pos = np.asarray(scores_pos, dtype=np.float64).ravel()
    neg = np.asarray(scores_neg, dtype=np.float64).ravel()
    pos = pos[np.isfinite(pos)]
    neg = neg[np.isfinite(neg)]
    if pos.size == 0 or neg.size == 0:
        return np.array([0.0, 1.0]), np.array([0.0, 1.0])

    scores = np.concatenate([pos, neg], axis=0)
    labels = np.concatenate([np.ones(pos.size, dtype=np.int8), np.zeros(neg.size, dtype=np.int8)], axis=0)
    order = np.argsort(scores)[::-1]
    labels = labels[order]

    tp = np.cumsum(labels == 1)
    fp = np.cumsum(labels == 0)
    tpr = tp / float(pos.size)
    fpr = fp / float(neg.size)

    # Deduplicate by score value (keep last index for each unique score).
    scores_sorted = scores[order]
    change = np.empty(scores_sorted.size, dtype=bool)
    change[0] = True
    change[1:] = scores_sorted[1:] != scores_sorted[:-1]
    idx = np.nonzero(change)[0]
    fpr_u = fpr[idx]
    tpr_u = tpr[idx]
    # Ensure (0,0) prefix.
    if fpr_u.size == 0 or fpr_u[0] != 0.0:
        fpr_u = np.concatenate([[0.0], fpr_u])
        tpr_u = np.concatenate([[0.0], tpr_u])
    return fpr_u, tpr_u


def _pauc(scores_pos: np.ndarray, scores_neg: np.ndarray, fpr_max: float) -> float:
    """Partial AUC up to fpr_max (trapezoid), right-tail scoring."""
    if not (0.0 < fpr_max <= 1.0):
        raise ValueError(f"fpr_max must be in (0,1], got {fpr_max}")
    fpr, tpr = _roc_curve(scores_pos, scores_neg)
    # Clip at fpr_max.
    if fpr_max < 1.0:
        # Interpolate TPR at fpr_max if needed.
        if fpr[-1] < fpr_max:
            fpr = np.concatenate([fpr, [fpr_max]])
            tpr = np.concatenate([tpr, [tpr[-1]]])
        else:
            j = int(np.searchsorted(fpr, fpr_max, side="right"))
            fpr_left = fpr[:j]
            tpr_left = tpr[:j]
            if fpr_left[-1] != fpr_max:
                tpr_at = float(np.interp(fpr_max, fpr, tpr))
                fpr_left = np.concatenate([fpr_left, [fpr_max]])
                tpr_left = np.concatenate([tpr_left, [tpr_at]])
            fpr, tpr = fpr_left, tpr_left
    # NumPy versions in some environments only expose `trapz`.
    return float(np.trapz(tpr, fpr))


def _bootstrap_ci_mean(values: np.ndarray, *, seed: int, n_boot: int = 2000) -> Tuple[float, float]:
    values = np.asarray(values, dtype=np.float64)
    values = values[np.isfinite(values)]
    if values.size == 0:
        return float("nan"), float("nan")
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, values.size, size=(n_boot, values.size))
    means = np.mean(values[idx], axis=1)
    lo = float(np.quantile(means, 0.025))
    hi = float(np.quantile(means, 0.975))
    return lo, hi


def _run_replay(cmd: List[str], *, dry_run: bool) -> None:
    print("[ka_v2] $ " + " ".join(str(c) for c in cmd), flush=True)
    if dry_run:
        return
    subprocess.run(cmd, check=True, cwd=str(REPO))


def _flatten_metrics(metrics: Mapping[str, Any] | None, *, prefix: str) -> Dict[str, Any]:
    out: Dict[str, Any] = {}
    if not metrics:
        return out
    for k, v in metrics.items():
        key = f"{prefix}{k}"
        if isinstance(v, (str, int, float, bool)) or v is None:
            out[key] = v
        else:
            # Keep JSON-serializable summary instead of nested blobs.
            try:
                out[key] = json.dumps(v)
            except Exception:
                out[key] = str(v)
    return out


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="KA Contract v2 falsifiability ablation runner.")
    ap.add_argument(
        "--pilot",
        type=Path,
        required=True,
        help="Pilot run directory (contains meta.json + ens*_angle_* dirs).",
    )
    ap.add_argument(
        "--profile",
        type=str,
        default="Brain-AliasContract",
        choices=["Brain-OpenSkull", "Brain-AliasContract", "Brain-SkullOR", "Brain-Pial128"],
        help="Brain-* operating profile to use for the replay runs.",
    )
    ap.add_argument("--out-root", type=Path, required=True, help="Root directory for ablation outputs.")
    ap.add_argument("--window-length", type=int, default=64, help="Replay window length.")
    ap.add_argument(
        "--window-offsets",
        nargs="+",
        type=int,
        default=[0, 64, 128, 192, 256],
        help="Window offsets to evaluate (non-overlapping recommended).",
    )
    ap.add_argument(
        "--fprs",
        nargs="+",
        type=str,
        default=["1e-4", "3e-4", "1e-3"],
        help="FPR targets (space-separated).",
    )
    ap.add_argument(
        "--pauc-fpr-max",
        type=float,
        default=1e-3,
        help="Compute pAUC up to this FPR (set 0 to disable pAUC).",
    )
    ap.add_argument("--stap-device", type=str, default="cuda", help="STAP device passed to replay.")
    ap.add_argument("--summary-csv", type=Path, required=True, help="Write per-window metrics to CSV.")
    ap.add_argument("--summary-json", type=Path, required=True, help="Write summary JSON.")
    ap.add_argument("--dry-run", action="store_true", help="Print commands without running.")
    ap.add_argument("--bootstrap-seed", type=int, default=1337, help="Seed for bootstrap CIs.")
    return ap.parse_args()


def main() -> None:
    args = parse_args()
    if not REPLAY.exists():
        raise FileNotFoundError(f"Missing {REPLAY}")

    fprs = _float_list(args.fprs)
    for f in fprs:
        if not (0.0 < f < 1.0):
            raise ValueError(f"Invalid FPR {f}; must be in (0,1).")

    pauc_fpr_max = float(args.pauc_fpr_max)
    if pauc_fpr_max < 0.0:
        raise ValueError("--pauc-fpr-max must be >= 0")

    out_root: Path = args.out_root
    out_root.mkdir(parents=True, exist_ok=True)

    pilot_dir = args.pilot.resolve()
    pilot_name = pilot_dir.name

    per_row: List[Dict[str, Any]] = []
    per_cond_tpr: Dict[str, Dict[float, List[float]]] = {
        cond.key: {fpr: [] for fpr in fprs} for cond in CONDITIONS
    }
    per_cond_pauc: Dict[str, List[float]] = {cond.key: [] for cond in CONDITIONS}
    per_cond_states: Dict[str, Dict[str, int]] = {cond.key: {} for cond in CONDITIONS}

    for offset in args.window_offsets:
        mask_flow_ref = None
        mask_bg_ref = None
        for cond in CONDITIONS:
            out_dir = out_root / f"{pilot_name}_off{int(offset)}_{cond.key}"
            bundle_dir = _existing_bundle_dir(out_dir)
            cmd = [
                sys.executable,
                str(REPLAY),
                "--src",
                str(pilot_dir),
                "--out",
                str(out_dir),
                "--profile",
                str(args.profile),
                "--stap-device",
                str(args.stap_device),
                "--time-window-length",
                str(args.window_length),
                "--time-window-offset",
                str(int(offset)),
            ]
            cmd.extend(cond.replay_flags)
            if bundle_dir is None:
                _run_replay(cmd, dry_run=args.dry_run)

            if args.dry_run:
                continue

            if bundle_dir is None:
                bundle_dir = _bundle_dir_from_out(out_dir)
            with open(bundle_dir / "meta.json") as f:
                meta = json.load(f)
            tele = meta.get("stap_fallback_telemetry") or {}

            score, mask_flow, mask_bg = _load_score_and_masks(bundle_dir)
            if mask_flow_ref is None:
                mask_flow_ref = mask_flow
                mask_bg_ref = mask_bg
                eval_mask_mismatch = 0
            else:
                eval_mask_mismatch = int(
                    np.count_nonzero(mask_flow != mask_flow_ref)
                    + np.count_nonzero(mask_bg != mask_bg_ref)
                )
            pos = score[mask_flow]
            neg = score[mask_bg]

            entry: Dict[str, Any] = {
                "pilot": pilot_name,
                "profile": str(args.profile),
                "window_offset": int(offset),
                "window_length": int(args.window_length),
                "condition": cond.key,
                "condition_label": cond.label,
                "ka_contract_v2_state": tele.get("ka_contract_v2_state"),
                "ka_contract_v2_reason": tele.get("ka_contract_v2_reason"),
                "score_ka_v2_state": tele.get("score_ka_v2_state"),
                "score_ka_v2_contract_reason": tele.get("score_ka_v2_contract_reason"),
                "score_ka_v2_disabled_reason": tele.get("score_ka_v2_disabled_reason"),
                "score_ka_v2_applied": tele.get("score_ka_v2_applied"),
                "score_ka_v2_forced": tele.get("score_ka_v2_forced"),
                "score_ka_v2_risk_mode": tele.get("score_ka_v2_risk_mode"),
                "score_ka_v2_scaled_pixel_fraction": tele.get("score_ka_v2_scaled_pixel_fraction"),
                "eval_mask_mismatch_pixels": int(eval_mask_mismatch),
                "neg_count": int(np.sum(mask_bg)),
                "pos_count": int(np.sum(mask_flow)),
            }

            stats = tele.get("score_ka_v2_stats")
            if isinstance(stats, dict):
                entry.update(_flatten_metrics(stats, prefix="score_ka_v2_stats_"))

            state_key = str(entry.get("score_ka_v2_state") or entry.get("ka_contract_v2_state") or "unknown")
            per_cond_states[cond.key][state_key] = per_cond_states[cond.key].get(state_key, 0) + 1

            for fpr in fprs:
                thr, tpr = _tpr_at_fpr(pos, neg, fpr)
                entry[f"tpr@{fpr:g}"] = tpr
                entry[f"thr@{fpr:g}"] = thr
                per_cond_tpr[cond.key][fpr].append(tpr)

            if pauc_fpr_max > 0.0:
                entry[f"pauc@{pauc_fpr_max:g}"] = _pauc(pos, neg, pauc_fpr_max)
                per_cond_pauc[cond.key].append(entry[f"pauc@{pauc_fpr_max:g}"])

            per_row.append(entry)

    if args.dry_run:
        print("[ka_v2] Dry run complete (no outputs written).", flush=True)
        return

    args.summary_csv.parent.mkdir(parents=True, exist_ok=True)
    fieldnames: List[str] = [
        "pilot",
        "profile",
        "window_offset",
        "window_length",
        "condition",
        "condition_label",
        "ka_contract_v2_state",
        "ka_contract_v2_reason",
        "score_ka_v2_state",
        "score_ka_v2_contract_reason",
        "score_ka_v2_disabled_reason",
        "score_ka_v2_applied",
        "score_ka_v2_forced",
        "score_ka_v2_risk_mode",
        "score_ka_v2_scaled_pixel_fraction",
        "eval_mask_mismatch_pixels",
        "neg_count",
        "pos_count",
    ]
    for fpr in fprs:
        fieldnames.extend([f"tpr@{fpr:g}", f"thr@{fpr:g}"])
    if pauc_fpr_max > 0.0:
        fieldnames.append(f"pauc@{pauc_fpr_max:g}")
    # Add any flattened metrics keys (union over rows).
    extra_keys = sorted({k for row in per_row for k in row.keys() if k.startswith("score_ka_v2_stats_")})
    fieldnames.extend(extra_keys)

    with open(args.summary_csv, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for row in per_row:
            w.writerow(row)

    # Summary: paired deltas vs STAP-only across evaluated windows.
    summary: Dict[str, Any] = {
        "pilot": pilot_name,
        "profile": str(args.profile),
        "window_length": int(args.window_length),
        "window_offsets": [int(o) for o in args.window_offsets],
        "fprs": fprs,
        "pauc_fpr_max": pauc_fpr_max,
        "conditions": [c.key for c in CONDITIONS],
        "state_counts": per_cond_states,
        "delta_vs_stap_only": {"tpr": {}, "pauc": {}},
    }

    # Deltas in TPR at fixed FPR.
    for fpr in fprs:
        base = np.array(per_cond_tpr["stap_only"][fpr], dtype=np.float64)
        for cond in ("ka_contract", "ka_forced"):
            vals = np.array(per_cond_tpr[cond][fpr], dtype=np.float64)
            if vals.size != base.size or vals.size == 0:
                continue
            delta = vals - base
            lo, hi = _bootstrap_ci_mean(delta, seed=args.bootstrap_seed + int(1e6 * fpr))
            summary["delta_vs_stap_only"]["tpr"].setdefault(cond, {})[f"{fpr:g}"] = {
                "mean": float(np.mean(delta)),
                "median": float(np.median(delta)),
                "ci95_mean": [lo, hi],
                "n": int(delta.size),
            }

    # Deltas in pAUC.
    if pauc_fpr_max > 0.0:
        base_p = np.array(per_cond_pauc["stap_only"], dtype=np.float64)
        for cond in ("ka_contract", "ka_forced"):
            vals = np.array(per_cond_pauc[cond], dtype=np.float64)
            if vals.size != base_p.size or vals.size == 0:
                continue
            delta = vals - base_p
            lo, hi = _bootstrap_ci_mean(delta, seed=args.bootstrap_seed + int(1e6 * pauc_fpr_max) + 7)
            summary["delta_vs_stap_only"]["pauc"][cond] = {
                "mean": float(np.mean(delta)),
                "median": float(np.median(delta)),
                "ci95_mean": [lo, hi],
                "n": int(delta.size),
            }

    args.summary_json.parent.mkdir(parents=True, exist_ok=True)
    with open(args.summary_json, "w") as f:
        json.dump(summary, f, indent=2, sort_keys=True)

    print(f"[ka_v2] Wrote {args.summary_csv}", flush=True)
    print(f"[ka_v2] Wrote {args.summary_json}", flush=True)


if __name__ == "__main__":
    main()
