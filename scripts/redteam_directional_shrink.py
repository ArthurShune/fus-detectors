#!/usr/bin/env python
"""
Red-team sanity check for the directional KA off-band shrink idea.

This script uses only NumPy (no torch) to:
  - Build random Hermitian SPD block matrices R = [[A, B], [B^H, C]].
  - Shrink only the off-band block C -> gamma * C while keeping A and B unchanged.
  - Measure s^H R^{-1} s for s in the passband subspace before/after shrink.

Expectation (when R and the Schur complements remain SPD):
  - As gamma decreases (i.e., stronger off-band shrink), s^H R^{-1} s should not
    decrease (typically increases), implying reduced MVDR variance in passband.

We also guard against non-SPD Schur complements by skipping such cases.
"""
from __future__ import annotations

import argparse

import numpy as np


def _rand_spd(n: int, ridge: float, rng: np.random.Generator) -> np.ndarray:
    A = rng.standard_normal((n, n)) + 1j * rng.standard_normal((n, n))
    A = (A + A.conj().T) * 0.5
    A += ridge * np.eye(n)
    return A


def trial_once(Lt: int, k_passband: int, B_scale: float, rng: np.random.Generator) -> int:
    """Run one random trial. Returns: +1 (better), -1 (worse), 0 (same/skip)."""
    k = k_passband
    A = _rand_spd(k, ridge=2.0, rng=rng)
    C = _rand_spd(Lt - k, ridge=2.0, rng=rng)
    B = (rng.standard_normal((k, Lt - k)) + 1j * rng.standard_normal((k, Lt - k))) * B_scale

    # SPD check via Schur complement S = C - B^H A^{-1} B
    S = C - B.conj().T @ np.linalg.inv(A) @ B
    if np.linalg.eigvalsh(S).min() <= 1e-9:
        return 0  # skip ill-conditioned configuration

    R = np.zeros((Lt, Lt), dtype=np.complex128)
    R[:k, :k] = A
    R[k:, k:] = C
    R[:k, k:] = B
    R[k:, :k] = B.conj().T

    s = np.zeros((Lt,), dtype=np.complex128)
    s[:k] = rng.standard_normal(k) + 1j * rng.standard_normal(k)
    v0 = (s.conj() @ np.linalg.solve(R, s)).real

    # Try multiple gamma values and check the strongest shrink effect
    result = 0
    for gamma in (0.9, 0.7, 0.5):
        Cg = gamma * C
        Sg = Cg - B.conj().T @ np.linalg.inv(A) @ B
        if np.linalg.eigvalsh(Sg).min() <= 1e-9:
            continue  # skip if shrink breaks SPD
        Rg = np.zeros_like(R)
        Rg[:k, :k] = A
        Rg[k:, k:] = Cg
        Rg[:k, k:] = B
        Rg[k:, :k] = B.conj().T
        v = (s.conj() @ np.linalg.solve(Rg, s)).real
        if v > v0 + 1e-9:
            result = 1  # better
        elif v < v0 - 1e-9:
            return -1  # any worsening marks the trial as failing
    return result


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--trials", type=int, default=200, help="number of random trials")
    ap.add_argument("--Lt", type=int, default=6, help="temporal dimension")
    ap.add_argument("--k", type=int, default=2, help="passband subspace dimension")
    ap.add_argument("--Bscale", type=float, default=0.01, help="scale for cross-term magnitude")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    rng = np.random.default_rng(args.seed)
    better = worse = same = 0
    for _ in range(args.trials):
        r = trial_once(args.Lt, args.k, args.Bscale, rng)
        if r > 0:
            better += 1
        elif r < 0:
            worse += 1
        else:
            same += 1

    print(
        f"Directional off-band shrink (C -> gamma*C) results over {args.trials} trials:\n"
        f"  better (s^H R^-1 s increased): {better}\n"
        f"  worse  (decreased):            {worse}\n"
        f"  skipped/neutral:               {same}\n"
    )
    if worse == 0:
        print("PASS: No evidence of degradation under SPD-preserving shrink.")
    else:
        print("WARNING: Some trials worsened — review SPD conditions or cross-term scale.")


if __name__ == "__main__":
    main()
