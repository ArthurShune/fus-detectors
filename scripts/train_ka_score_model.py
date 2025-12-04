#!/usr/bin/env python3
"""
Train a simple logistic risk model for score-space KA on a single HAB bundle.

We fit a logistic regression r(i) = sigmoid(w^T x_i + b) to discriminate
background (H0) vs flow (H1) using per-pixel telemetry features extracted from
an existing STAP PD bundle:

  x_i = [pd_stap(i), m_alias(i), depth_frac(i), geom_flow_mask(i)].

The trained parameters (w, b) and feature normalization (mean/std) are written
to a small JSON file that can be loaded at replay time to drive a score-space
shrink-only KA transform.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def _load_bundle(bundle_dir: Path) -> dict[str, np.ndarray | dict]:
    meta_path = bundle_dir / "meta.json"
    if not meta_path.exists():
        raise FileNotFoundError(f"meta.json not found in {bundle_dir}")
    meta = json.loads(meta_path.read_text())

    def _load(name: str) -> np.ndarray:
        path = bundle_dir / name
        if not path.exists():
            raise FileNotFoundError(path)
        return np.load(path)

    out: dict[str, np.ndarray | dict] = {"meta": meta}
    out["mask_flow"] = _load("mask_flow.npy")
    out["mask_bg"] = _load("mask_bg.npy")
    out["pd_stap"] = _load("pd_stap.npy")
    out["base_band_ratio"] = _load("base_band_ratio_map.npy")
    return out


def train_logistic(
    X: np.ndarray,
    y: np.ndarray,
    lr: float = 0.1,
    epochs: int = 400,
) -> tuple[np.ndarray, float]:
    """Simple logistic regression via batch gradient descent."""
    n, d = X.shape
    w = np.zeros(d, dtype=np.float64)
    b = 0.0
    for _ in range(epochs):
        z = X @ w + b
        z = np.clip(z, -20.0, 20.0)
        p = 1.0 / (1.0 + np.exp(-z))
        diff = p - y
        grad_w = (X.T @ diff) / float(n)
        grad_b = float(diff.mean())
        w -= lr * grad_w
        b -= lr * grad_b
    return w, b


def main() -> None:
    ap = argparse.ArgumentParser(description="Train score-space KA risk model from a HAB bundle.")
    ap.add_argument("bundle", type=Path, help="Path to HAB/STAP bundle (pw_... directory).")
    ap.add_argument(
        "--out-json",
        type=Path,
        required=True,
        help="Output JSON path for trained model parameters.",
    )
    args = ap.parse_args()

    data = _load_bundle(args.bundle)
    pd_stap = np.asarray(data["pd_stap"], dtype=np.float64)
    m_alias = np.asarray(data["base_band_ratio"], dtype=np.float64)
    mask_flow = np.asarray(data["mask_flow"], dtype=bool)
    mask_bg = np.asarray(data["mask_bg"], dtype=bool)
    H, W = pd_stap.shape

    depth = np.linspace(0.0, 1.0, H, endpoint=True)[:, None]
    depth_map = depth.repeat(W, axis=1)

    # Restrict to labeled pixels.
    labelled = mask_flow | mask_bg
    y = mask_bg[labelled].astype(np.float64)  # 1 for background, 0 for flow

    # Features are restricted to telemetry that would be available in a
    # clinical deployment: PD, alias metric, and depth. The model is not
    # allowed to see simulator-only labels such as mask_flow.
    feats = np.stack(
        [
            pd_stap[labelled],
            m_alias[labelled],
            depth_map[labelled],
        ],
        axis=1,
    )

    # Standardize features.
    mean = feats.mean(axis=0)
    std = feats.std(axis=0)
    std[std == 0.0] = 1.0
    X = (feats - mean) / std

    w, b = train_logistic(X, y, lr=0.1, epochs=400)

    model = {
        "weights": w.tolist(),
        "bias": float(b),
        "mean": mean.tolist(),
        "std": std.tolist(),
        "feature_names": ["pd_stap", "m_alias", "depth_frac"],
    }
    args.out_json.parent.mkdir(parents=True, exist_ok=True)
    args.out_json.write_text(json.dumps(model, indent=2))
    print(f"[train_ka_score_model] wrote model to {args.out_json}", flush=True)


if __name__ == "__main__":
    main()
