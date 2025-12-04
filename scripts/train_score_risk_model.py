import argparse
import json
from pathlib import Path

import numpy as np


def _load_npz(path: Path) -> dict:
    return np.load(path)


def _stack_features(paths: list[Path]) -> tuple[np.ndarray, np.ndarray]:
    xs = []
    ys = []
    for p in paths:
        data = _load_npz(p)
        # Use a small feature set for now
        S = data["S_base"].astype(np.float64)
        log_br = data["log_br"].astype(np.float64)
        depth = data["depth_frac"].astype(np.float64)
        neigh = data["neighbor_coh"].astype(np.float64)
        y = data["y"].astype(np.int8)

        # Replace NaNs in log_br with 0 (neutral)
        log_br = np.where(np.isfinite(log_br), log_br, 0.0)

        X = np.stack([S, log_br, depth, neigh], axis=1)
        xs.append(X)
        ys.append(y)

    X_all = np.concatenate(xs, axis=0)
    y_all = np.concatenate(ys, axis=0)
    return X_all, y_all


def _standardize(X: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    mu = X.mean(axis=0)
    sigma = X.std(axis=0)
    sigma[sigma == 0.0] = 1.0
    Xn = (X - mu) / sigma
    return Xn, mu, sigma


def _sigmoid(z: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-z))


def _train_logistic(
    X: np.ndarray,
    y: np.ndarray,
    lr: float = 0.1,
    n_iter: int = 200,
) -> tuple[np.ndarray, float]:
    n, d = X.shape
    w = np.zeros(d, dtype=np.float64)
    b = 0.0
    for _ in range(n_iter):
        z = X @ w + b
        p = _sigmoid(z)
        grad_w = X.T @ (p - y) / n
        grad_b = float((p - y).mean())
        w -= lr * grad_w
        b -= lr * grad_b
    return w, b


def _roc_auc(y: np.ndarray, score: np.ndarray) -> float:
    # Simple AUC implementation (no ties handling optimization)
    order = np.argsort(score)
    y = y[order]
    n_pos = float(y.sum())
    n_neg = float(len(y) - n_pos)
    if n_pos == 0 or n_neg == 0:
        return np.nan
    rank = np.arange(1, len(y) + 1, dtype=np.float64)
    # Sum of ranks of positive samples
    pos_rank_sum = float(rank[y == 1].sum())
    auc = (pos_rank_sum - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg)
    return auc


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Train a simple logistic risk model on STAP features"
    )
    parser.add_argument(
        "--features",
        type=Path,
        nargs="+",
        required=True,
        help=".npz feature files from dump_stap_pixel_features.py",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        required=True,
        help="Directory to write model weights and metadata",
    )
    parser.add_argument("--lr", type=float, default=0.1, help="Learning rate")
    parser.add_argument("--n-iter", type=int, default=200, help="Number of gradient steps")
    args = parser.parse_args()

    X, y = _stack_features(args.features)
    Xn, mu, sigma = _standardize(X)
    w, b = _train_logistic(Xn, y, lr=args.lr, n_iter=args.n_iter)
    z = Xn @ w + b
    p1 = _sigmoid(z)
    auc = _roc_auc(y, p1)

    args.out_dir.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        args.out_dir / "score_risk_model.npz",
        w=w.astype(np.float32),
        b=np.array([b], dtype=np.float32),
        mu=mu.astype(np.float32),
        sigma=sigma.astype(np.float32),
    )
    (args.out_dir / "score_risk_model_meta.json").write_text(
        json.dumps({"auc": float(auc), "n_samples": int(len(y))}, indent=2)
    )
    print(f"[risk] trained logistic model on {len(y)} samples; AUC={auc:.3f}")


if __name__ == "__main__":
    main()
