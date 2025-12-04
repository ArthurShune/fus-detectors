# figs/fig2_roc_calibrated.py
import argparse
import os
from typing import Tuple

import matplotlib.pyplot as plt
import numpy as np

from eval.metrics import roc_curve
from pipeline.calib.evt_pot import (
    fit_pot,
    pick_u_by_mean_excess,
    tail_pvalue,
)
from pipeline.calib.evt_pot import (
    mean_excess as _mean_excess,
)


def _load_scores(p):
    if p is None:
        return None
    if p.endswith(".npz"):
        z = np.load(p)
        for k in ["scores", "arr", "data"]:
            if k in z:
                return z[k].ravel()
        return list(z.values())[0].ravel()
    return np.load(p).ravel()


def _simulate_scores(n_pos=100_000, n_neg=2_000_000, seed=0):
    rng = np.random.default_rng(seed)
    # Baseline: moderate separation
    base_pos = rng.normal(2.0, 1.0, size=n_pos)
    base_neg = np.where(
        rng.random(n_neg) < 0.1, rng.pareto(5.0, n_neg) + 2.0, rng.normal(0.0, 1.0, n_neg)
    )
    # STAP: better separation & lighter tail
    stap_pos = rng.normal(2.6, 1.0, size=n_pos)
    stap_neg = np.where(
        rng.random(n_neg) < 0.05, rng.pareto(6.0, n_neg) + 1.8, rng.normal(0.0, 1.0, n_neg)
    )
    return (base_pos, base_neg, stap_pos, stap_neg)


def _tail_qq(scores_null: np.ndarray, pm) -> Tuple[np.ndarray, np.ndarray]:
    s = np.sort(scores_null)
    # consider top 1% for tail QQ
    s = s[int(0.99 * len(s)) :]
    emp_p = 1.0 - np.linspace(0, 1, len(s), endpoint=False)  # crude plotting positions
    fit_p = np.array([tail_pvalue(pm, float(v)) for v in s])
    return emp_p, fit_p


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--base_pos", type=str, default=None)
    parser.add_argument("--base_neg", type=str, default=None)
    parser.add_argument("--stap_pos", type=str, default=None)
    parser.add_argument("--stap_neg", type=str, default=None)
    parser.add_argument("--out", type=str, default="figs/outputs/fig2_roc_calibrated.png")
    args = parser.parse_args()
    os.makedirs(os.path.dirname(args.out), exist_ok=True)

    fast = os.environ.get("FAST_FIGS", "0") == "1"
    if any(getattr(args, k) is None for k in ["base_pos", "base_neg", "stap_pos", "stap_neg"]):
        npos = 50_000 if fast else 150_000
        nneg = 500_000 if fast else 2_000_000
        base_pos, base_neg, stap_pos, stap_neg = _simulate_scores(n_pos=npos, n_neg=nneg, seed=1)
    else:
        base_pos = _load_scores(args.base_pos)
        base_neg = _load_scores(args.base_neg)
        stap_pos = _load_scores(args.stap_pos)
        stap_neg = _load_scores(args.stap_neg)

    # ROC (dense in tail)
    fpr_b, tpr_b, _ = roc_curve(base_pos, base_neg, num_thresh=2048)
    fpr_s, tpr_s, _ = roc_curve(stap_pos, stap_neg, num_thresh=2048)

    # EVT diagnostics on STAP null
    q0 = 0.98
    u, _ = pick_u_by_mean_excess(stap_neg, q0=q0, grid_size=8, min_exceedances=200)
    pm = fit_pot(stap_neg, u)
    # Mean-excess curve across grid
    grid = np.linspace(np.quantile(stap_neg, q0), np.quantile(stap_neg, 0.999), 8)
    me = _mean_excess(stap_neg, grid)
    # Tail QQ (empirical p vs fitted p in tail)
    emp_p, fit_p = _tail_qq(stap_neg, pm)

    fig = plt.figure(figsize=(11, 8))
    gs = fig.add_gridspec(2, 2, height_ratios=[2, 1])

    # ROC panel
    ax0 = fig.add_subplot(gs[0, :])
    ax0.plot(fpr_b, tpr_b, label="SVD baseline")
    ax0.plot(fpr_s, tpr_s, label="STAP-for-fUS")
    for v in [1e-6, 1e-5, 1e-4]:
        ax0.axvline(v, linestyle=":", linewidth=0.8)
    ax0.set_xscale("log")
    ax0.set_xlabel("False positive rate (FPR)")
    ax0.set_ylabel("True positive rate (TPR)")
    ax0.set_title("ROC at ultra-low FPR")
    ax0.grid(True, which="both", ls=":")
    ax0.legend()

    # Mean-excess plot
    ax1 = fig.add_subplot(gs[1, 0])
    ax1.plot(grid, me, marker="o")
    ax1.set_title("EVT: mean-excess vs threshold u (STAP null)")
    ax1.set_xlabel("u")
    ax1.set_ylabel("mean excess e(u)")
    ax1.grid(True, ls=":")

    # Tail QQ: empirical p vs fitted p (tail)
    ax2 = fig.add_subplot(gs[1, 1])
    ax2.plot(fit_p, emp_p, marker=".", linestyle="None")
    # y=x reference
    m = min(emp_p.min(), fit_p.min())
    M = max(emp_p.max(), fit_p.max())
    ax2.plot([m, M], [m, M], linestyle="--")
    ax2.set_xscale("log")
    ax2.set_yscale("log")
    ax2.set_xlabel("Fitted tail p-value")
    ax2.set_ylabel("Empirical tail p-value")
    ax2.set_title("EVT: Tail QQ (STAP null)")
    ax2.grid(True, which="both", ls=":")

    plt.tight_layout()
    plt.savefig(args.out, dpi=160)
    print(f"Wrote {args.out}")


if __name__ == "__main__":
    main()
