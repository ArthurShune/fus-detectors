#!/usr/bin/env python3
"""
PD-mode sanity checks for acceptance bundles.

This script is meant to make the repository's PD-mode scoring convention explicit:
  - Bundles store `pd_base.npy` / `pd_stap.npy` as PD-mode maps.
  - For ROC we typically use the right-tail score S = -PD (equivalently, lower-tail
    thresholding on the stored PD map).
  - Newer bundles also export `score_pd_base.npy` / `score_pd_stap.npy` explicitly.

For each bundle directory, this script:
  - checks that `score_pd_*.npy == -pd_*.npy` when present
  - prints flow/background summary statistics (medians, ratios)
  - optionally writes a small diagnostic figure under --out-dir

Usage:
  PYTHONPATH=. python scripts/pd_mode_sanity.py runs/.../pw_* --out-dir reports/pd_mode_sanity
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

try:  # optional plotting
    import matplotlib.pyplot as plt
except Exception:  # pragma: no cover
    plt = None


def _load_optional(bundle_dir: Path, name: str) -> np.ndarray | None:
    path = bundle_dir / name
    if not path.exists():
        return None
    return np.load(path)


def _finite(vals: np.ndarray) -> np.ndarray:
    vals = np.asarray(vals, dtype=np.float64).ravel()
    return vals[np.isfinite(vals)]


def _median(arr: np.ndarray) -> float | None:
    v = _finite(arr)
    if v.size == 0:
        return None
    return float(np.median(v))


def _spearmanr(a: np.ndarray, b: np.ndarray) -> float | None:
    a = _finite(a)
    b = _finite(b)
    n = min(a.size, b.size)
    if n < 3:
        return None
    a = a[:n]
    b = b[:n]
    # Rank transform via argsort twice (stable enough for diagnostics).
    ra = np.argsort(np.argsort(a))
    rb = np.argsort(np.argsort(b))
    ra = ra.astype(np.float64)
    rb = rb.astype(np.float64)
    ra -= ra.mean()
    rb -= rb.mean()
    denom = float(np.linalg.norm(ra) * np.linalg.norm(rb))
    if denom <= 0.0:
        return None
    return float((ra @ rb) / denom)


def summarize_bundle(bundle_dir: Path) -> dict:
    pd_base = np.load(bundle_dir / "pd_base.npy")
    pd_stap = np.load(bundle_dir / "pd_stap.npy")
    mask_flow = np.load(bundle_dir / "mask_flow.npy").astype(bool)
    mask_bg = np.load(bundle_dir / "mask_bg.npy").astype(bool)

    score_pd_base = _load_optional(bundle_dir, "score_pd_base.npy")
    score_pd_stap = _load_optional(bundle_dir, "score_pd_stap.npy")

    out: dict[str, object] = {"bundle": bundle_dir.name}

    out["pd_base_flow_median"] = _median(pd_base[mask_flow]) if mask_flow.any() else None
    out["pd_base_bg_median"] = _median(pd_base[mask_bg]) if mask_bg.any() else None
    out["pd_stap_flow_median"] = _median(pd_stap[mask_flow]) if mask_flow.any() else None
    out["pd_stap_bg_median"] = _median(pd_stap[mask_bg]) if mask_bg.any() else None

    # Direction sanity: score is right-tail, PD is left-tail.
    out["spearman(pd_base, -pd_base)"] = _spearmanr(pd_base, -pd_base)
    out["spearman(pd_stap, -pd_stap)"] = _spearmanr(pd_stap, -pd_stap)

    if score_pd_base is not None:
        out["score_pd_base_max_abs_err"] = float(np.nanmax(np.abs(score_pd_base + pd_base)))
    else:
        out["score_pd_base_max_abs_err"] = None
    if score_pd_stap is not None:
        out["score_pd_stap_max_abs_err"] = float(np.nanmax(np.abs(score_pd_stap + pd_stap)))
    else:
        out["score_pd_stap_max_abs_err"] = None

    # Relationship between baseline PD and STAP PD-mode map (not expected to be monotone).
    out["spearman(pd_base, pd_stap)"] = _spearmanr(pd_base, pd_stap)

    return out


def maybe_plot(bundle_dir: Path, out_dir: Path) -> None:
    if plt is None:
        return

    pd_base = np.load(bundle_dir / "pd_base.npy")
    pd_stap = np.load(bundle_dir / "pd_stap.npy")
    mask_flow = np.load(bundle_dir / "mask_flow.npy").astype(bool)
    mask_bg = np.load(bundle_dir / "mask_bg.npy").astype(bool)

    def _vals(arr, m):
        xs = arr[m]
        xs = xs[np.isfinite(xs)]
        if xs.size > 200_000:
            xs = xs[np.random.default_rng(0).choice(xs.size, 200_000, replace=False)]
        return xs

    base_flow = _vals(pd_base, mask_flow)
    base_bg = _vals(pd_base, mask_bg)
    stap_flow = _vals(pd_stap, mask_flow)
    stap_bg = _vals(pd_stap, mask_bg)

    fig, axes = plt.subplots(2, 2, figsize=(10, 8))
    ax = axes[0, 0]
    ax.hist(base_bg, bins=80, alpha=0.6, label="bg", density=True)
    ax.hist(base_flow, bins=80, alpha=0.6, label="flow", density=True)
    ax.set_title("pd_base distribution")
    ax.legend()

    ax = axes[0, 1]
    ax.hist(stap_bg, bins=80, alpha=0.6, label="bg", density=True)
    ax.hist(stap_flow, bins=80, alpha=0.6, label="flow", density=True)
    ax.set_title("pd_stap distribution")
    ax.legend()

    ax = axes[1, 0]
    # Scatter on a subsample for readability.
    rng = np.random.default_rng(0)
    flat = np.isfinite(pd_base) & np.isfinite(pd_stap)
    idx = np.flatnonzero(flat.ravel())
    if idx.size > 40_000:
        idx = rng.choice(idx, 40_000, replace=False)
    b = pd_base.ravel()[idx]
    s = pd_stap.ravel()[idx]
    ax.scatter(b, s, s=2, alpha=0.2)
    ax.set_xlabel("pd_base")
    ax.set_ylabel("pd_stap")
    ax.set_title("pd_stap vs pd_base (sample)")

    ax = axes[1, 1]
    ax.imshow(pd_stap, cmap="viridis")
    ax.set_title("pd_stap map")
    ax.axis("off")

    fig.tight_layout()
    out_dir.mkdir(parents=True, exist_ok=True)
    fig_path = out_dir / f"{bundle_dir.name}_pd_mode_sanity.png"
    fig.savefig(fig_path, dpi=150)
    plt.close(fig)


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("bundles", nargs="+", type=Path, help="Bundle directories (pw_* or similar)")
    ap.add_argument("--out-dir", type=Path, default=None, help="Optional figure output directory")
    return ap.parse_args()


def main() -> None:
    args = parse_args()
    for bundle_dir in args.bundles:
        stats = summarize_bundle(bundle_dir)
        print(f"\n[{stats['bundle']}]")
        for k in sorted(stats.keys()):
            if k == "bundle":
                continue
            print(f"  {k}: {stats[k]}")
        if args.out_dir is not None:
            maybe_plot(bundle_dir, args.out_dir)


if __name__ == "__main__":
    main()

