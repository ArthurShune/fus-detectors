import argparse
import json
from pathlib import Path

import numpy as np


def load_scores(bundle_dir: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Load STAP scores and BG/flow masks from a bundle.

    Expects files:
      - pd_stap.npy      (score map)
      - mask_bg.npy      (background mask)
      - mask_flow.npy    (flow/positive mask)
    """
    pd_path = bundle_dir / "pd_stap.npy"
    mask_bg_path = bundle_dir / "mask_bg.npy"
    mask_flow_path = bundle_dir / "mask_flow.npy"

    if not pd_path.exists():
        raise FileNotFoundError(f"Missing pd_stap.npy in {bundle_dir}")
    if not mask_bg_path.exists() or not mask_flow_path.exists():
        raise FileNotFoundError(f"Missing mask_bg.npy or mask_flow.npy in {bundle_dir}")

    pd = np.load(pd_path)
    mask_bg = np.load(mask_bg_path).astype(bool)
    mask_flow = np.load(mask_flow_path).astype(bool)

    if pd.shape != mask_bg.shape or pd.shape != mask_flow.shape:
        raise ValueError(
            f"Shape mismatch in {bundle_dir}: "
            f"pd {pd.shape}, mask_bg {mask_bg.shape}, mask_flow {mask_flow.shape}"
        )
    return pd, mask_bg, mask_flow


def tpr_at_fpr(
    scores: np.ndarray,
    mask_bg: np.ndarray,
    mask_flow: np.ndarray,
    alpha: float,
) -> tuple[float, float]:
    """
    Compute TPR at a given FPR alpha for a score map.

    Returns (tpr, threshold) where threshold is chosen on BG tiles.
    """
    neg = scores[mask_bg]
    pos = scores[mask_flow]

    if neg.size == 0 or pos.size == 0:
        return float("nan"), float("nan")

    # Threshold so that P(score > t | H0) ~= alpha.
    k = max(int(np.floor((1.0 - alpha) * neg.size)), 0)
    k = min(k, neg.size - 1)
    t = np.partition(neg, k)[k]

    tpr = float((pos > t).sum() / float(pos.size))
    return tpr, float(t)


def load_gate_masks(ka_bundle: Path, shape: tuple[int, int]) -> tuple[np.ndarray, np.ndarray]:
    """
    Load KA gate masks (flow / bg) from meta.json if present.

    Returns boolean arrays of the given shape; if masks are missing,
    returns all-False arrays.
    """
    meta_path = ka_bundle / "meta.json"
    if not meta_path.exists():
        return np.zeros(shape, dtype=bool), np.zeros(shape, dtype=bool)
    meta = json.loads(meta_path.read_text())
    tele = meta.get("stap_fallback_telemetry") or meta.get("stap_telemetry") or {}
    gate_flow_raw = tele.get("_gate_mask_flow")
    gate_bg_raw = tele.get("_gate_mask_bg")
    gate_flow = np.zeros(shape, dtype=bool)
    gate_bg = np.zeros(shape, dtype=bool)
    if gate_flow_raw is not None:
        gate_flow = np.asarray(gate_flow_raw, dtype=bool)
        if gate_flow.shape != shape:
            raise ValueError(
                f"gate_flow mask shape {gate_flow.shape} does not match scores {shape}"
            )
    if gate_bg_raw is not None:
        gate_bg = np.asarray(gate_bg_raw, dtype=bool)
        if gate_bg.shape != shape:
            raise ValueError(f"gate_bg mask shape {gate_bg.shape} does not match scores {shape}")
    return gate_flow, gate_bg


def summarize_tail_effects(
    stap_scores: np.ndarray,
    ka_scores: np.ndarray,
    mask_bg: np.ndarray,
    mask_flow: np.ndarray,
    gate_flow: np.ndarray,
    gate_bg: np.ndarray,
    thresh: float,
) -> dict:
    """
    Summarize gating coverage and score ratios in the negative tail.
    """
    tail_bg = mask_bg & (stap_scores >= thresh)
    n_tail_bg = int(tail_bg.sum())
    n_bg = int(mask_bg.sum())
    n_flow = int(mask_flow.sum())

    gated_bg = gate_bg & mask_bg
    gated_flow = gate_flow & mask_flow
    gated_tail_bg = gate_bg & tail_bg

    def _safe_frac(num: int, den: int) -> float:
        return float(num) / float(den) if den > 0 else float("nan")

    ratios_tail_bg = []
    if n_tail_bg > 0:
        idx_tail_gated = gated_tail_bg
        base_vals = stap_scores[idx_tail_gated]
        ka_vals = ka_scores[idx_tail_gated]
        if base_vals.size:
            safe_base = np.maximum(base_vals, 1e-9)
            r = ka_vals / safe_base
            r = r[np.isfinite(r)]
            ratios_tail_bg = r.tolist()

    ratios_flow = []
    if gated_flow.any():
        base_vals_f = stap_scores[gated_flow]
        ka_vals_f = ka_scores[gated_flow]
        safe_base_f = np.maximum(base_vals_f, 1e-9)
        r_f = ka_vals_f / safe_base_f
        r_f = r_f[np.isfinite(r_f)]
        ratios_flow = r_f.tolist()

    def _ratio_stats(arr: list[float]) -> dict:
        if not arr:
            return {"count": 0, "median": None, "p10": None, "p90": None}
        v = np.asarray(arr, dtype=float)
        out: dict[str, float | int | None] = {"count": int(v.size)}
        out["median"] = float(np.median(v))
        if v.size >= 2:
            out["p10"] = float(np.quantile(v, 0.10))
            out["p90"] = float(np.quantile(v, 0.90))
        else:
            out["p10"] = float(v[0])
            out["p90"] = float(v[0])
        return out

    return {
        "n_bg": n_bg,
        "n_flow": n_flow,
        "n_tail_bg": n_tail_bg,
        "n_gated_bg": int(gated_bg.sum()),
        "n_gated_flow": int(gated_flow.sum()),
        "n_gated_tail_bg": int(gated_tail_bg.sum()),
        "p_gate_bg": _safe_frac(int(gated_bg.sum()), n_bg),
        "p_gate_flow": _safe_frac(int(gated_flow.sum()), n_flow),
        "p_gate_tail_bg": _safe_frac(int(gated_tail_bg.sum()), n_tail_bg),
        "score_ratios_tail_bg": _ratio_stats(ratios_tail_bg),
        "score_ratios_flow": _ratio_stats(ratios_flow),
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Compare STAP vs STAP+KA on a single bundle at low FPR, "
            "and summarize KA gating coverage and score ratios in the "
            "negative tail using existing pd_stap, masks, and KA gate masks."
        )
    )
    parser.add_argument(
        "--stap-bundle",
        type=Path,
        required=True,
        help="Bundle directory for STAP-only (no KA) replay.",
    )
    parser.add_argument(
        "--ka-bundle",
        type=Path,
        required=True,
        help="Bundle directory for STAP+KA replay.",
    )
    parser.add_argument(
        "--fpr",
        type=float,
        default=1e-4,
        help="Target false positive rate for TPR evaluation (default: 1e-4).",
    )
    args = parser.parse_args()

    stap_dir = args.stap_bundle
    ka_dir = args.ka_bundle

    print(f"[info] STAP bundle:    {stap_dir}")
    print(f"[info] STAP+KA bundle: {ka_dir}")
    print(f"[info] Target FPR:      {args.fpr:g}")

    stap_scores, _, _ = load_scores(stap_dir)
    ka_scores, ka_bg, ka_flow = load_scores(ka_dir)

    # Use the KA bundle's flow / BG masks for both runs so that H0/H1 are
    # defined consistently with the gating telemetry.
    stap_bg = ka_bg
    stap_flow = ka_flow

    gate_flow, gate_bg = load_gate_masks(ka_dir, stap_scores.shape)

    tpr_stap, thresh = tpr_at_fpr(stap_scores, stap_bg, stap_flow, args.fpr)
    pos_ka = ka_scores[stap_flow]
    tpr_ka = float((pos_ka > thresh).sum() / float(pos_ka.size)) if pos_ka.size else float("nan")

    print("\n[results]")
    print(f"  threshold on BG (STAP): {thresh:.6g}")
    print(f"  TPR_STAP at FPR={args.fpr:g}: {tpr_stap:.6g}")
    print(f"  TPR_KA   at same thresh:      {tpr_ka:.6g}")
    print(f"  ΔTPR (KA - STAP):             {tpr_ka - tpr_stap:.6g}")

    tail_stats = summarize_tail_effects(
        stap_scores,
        ka_scores,
        stap_bg,
        stap_flow,
        gate_flow,
        gate_bg,
        thresh,
    )

    print("\n[tail coverage]")
    print(
        f"  BG tiles: total={tail_stats['n_bg']} "
        f"tail={tail_stats['n_tail_bg']} "
        f"gated={tail_stats['n_gated_bg']} "
        f"gated_tail={tail_stats['n_gated_tail_bg']}"
    )
    print(f"  Flow tiles: total={tail_stats['n_flow']} " f"gated={tail_stats['n_gated_flow']}")
    print(
        f"  p_gate_bg_global={tail_stats['p_gate_bg']:.4g}, "
        f"p_gate_flow_global={tail_stats['p_gate_flow']:.4g}, "
        f"p_gate_tail_bg={tail_stats['p_gate_tail_bg']:.4g}"
    )

    r_tail = tail_stats["score_ratios_tail_bg"]
    r_flow = tail_stats["score_ratios_flow"]
    print("\n[score ratios]")
    print(
        "  tail negatives (gated): "
        f"count={r_tail['count']}, "
        f"median={r_tail['median']}, "
        f"p10={r_tail['p10']}, "
        f"p90={r_tail['p90']}"
    )
    print(
        "  positives (gated): "
        f"count={r_flow['count']}, "
        f"median={r_flow['median']}, "
        f"p10={r_flow['p10']}, "
        f"p90={r_flow['p90']}"
    )

    summary = {
        "stap_bundle": str(stap_dir),
        "ka_bundle": str(ka_dir),
        "fpr": args.fpr,
        "threshold": thresh,
        "tpr_stap": tpr_stap,
        "tpr_ka": tpr_ka,
        "delta_tpr": tpr_ka - tpr_stap,
        "tail_stats": tail_stats,
    }
    out_path = ka_dir / "pial_tail_comparison.json"
    out_path.write_text(json.dumps(summary, indent=2))
    print(f"\n[info] Wrote summary to {out_path}")


if __name__ == "__main__":
    main()
