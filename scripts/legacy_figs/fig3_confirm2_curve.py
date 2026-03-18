# scripts/legacy_figs/fig3_confirm2_curve.py
import os
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import norm

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

from pipeline.confirm2.validator import calibrate_confirm2, evaluate_confirm2  # noqa: E402


def _scores_from_bvn(n: int, rho: float, seed: int = 0):
    """
    Generate null look scores S1,S2 from BVN(z1,z2) via s = -log(1 - Phi(z)).
    This yields exact exponential tails (ξ=0), ideal for POT+conformal.
    """
    rng = np.random.default_rng(seed)
    cov = np.array([[1.0, rho], [rho, 1.0]])
    L = np.linalg.cholesky(cov)
    z = rng.standard_normal((n, 2)) @ L.T
    p = 1.0 - norm.cdf(z)  # right-tail p-values
    s = -np.log(p)  # scores with exact exponential tail
    return s[:, 0], s[:, 1]


def main():
    os.makedirs("figs/outputs", exist_ok=True)
    # Rhos and targets to sweep (keep light)
    rhos = [0.0, 0.3, 0.6, 0.85]
    alpha2s = [1e-3, 3e-4, 1e-4]

    # For stable estimates at 1e-4, use ~200k per set (cal+test split inside)
    N = 200_000

    fig, ax = plt.subplots(figsize=(7, 5))
    for rho in rhos:
        s1, s2 = _scores_from_bvn(N, rho, seed=int(1000 * rho) + 7)
        # Split into calibration and test halves
        s1_cal, s1_test = s1[: N // 2], s1[N // 2 :]
        s2_cal, s2_test = s2[: N // 2], s2[N // 2 :]

        xs = []
        ys = []
        yerr_lo = []
        yerr_hi = []
        y_pred = []
        for a2 in alpha2s:
            calib = calibrate_confirm2(s1_cal, s2_cal, alpha2_target=a2, seed=42)
            ev = evaluate_confirm2(calib, s1_test, s2_test)
            xs.append(a2)
            ys.append(ev.empirical_pair_pfa)
            yerr_lo.append(ev.empirical_pair_pfa - ev.pair_ci_lo)
            yerr_hi.append(ev.pair_ci_hi - ev.empirical_pair_pfa)
            y_pred.append(ev.predicted_pair_pfa)

        # Plot empirical with error bars, and analytic as x markers
        ax.errorbar(xs, ys, yerr=[yerr_lo, yerr_hi], fmt="o-", label=f"Empirical (ρ≈{rho:.2f})")
        ax.plot(xs, y_pred, "x--")

    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("Target pair-Pfa (α2)")
    ax.set_ylabel("Pair-Pfa")
    ax.set_title("Confirm-2: Empirical vs Analytic (BVN copula)")
    ax.grid(True, which="both", ls=":")
    ax.legend()
    out = "figs/outputs/fig3_confirm2_curve.png"
    plt.tight_layout()
    plt.savefig(out, dpi=160)
    print(f"Wrote {out}")


if __name__ == "__main__":
    main()
