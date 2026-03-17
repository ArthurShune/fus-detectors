#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

import matplotlib as mpl
import matplotlib.pyplot as plt


def _iter_bundle_dirs(root: Path, pattern: str) -> list[Path]:
    if not root.exists():
        raise FileNotFoundError(root)

    # If root directly contains the needed files, treat it as one bundle.
    if (root / "score_stap_preka.npy").exists() or (root / "score_pd_stap_pre_ka.npy").exists():
        return [root]

    # Otherwise, collect matching subdirectories (e.g., frame000, pw_*, etc.).
    candidates = sorted([p for p in root.glob(pattern) if p.is_dir()])
    if not candidates:
        raise FileNotFoundError(f"No bundle directories found under {root} with pattern {pattern!r}.")
    return candidates


def _load_mask(path: Path) -> np.ndarray:
    m = np.load(path)
    if m.dtype != bool:
        m = m != 0
    return m


def _load_scores(
    bundle_dir: Path,
    pre_name: str,
    post_name: str,
    mask_bg_name: str,
    *,
    scale_name: str | None = None,
    scaled_only: bool = False,
) -> tuple[np.ndarray, np.ndarray]:
    pre_path = bundle_dir / pre_name
    post_path = bundle_dir / post_name
    mask_path = bundle_dir / mask_bg_name
    scale_path = (bundle_dir / scale_name) if (scale_name is not None) else None

    if not pre_path.exists():
        raise FileNotFoundError(pre_path)
    if not post_path.exists():
        raise FileNotFoundError(post_path)
    if not mask_path.exists():
        raise FileNotFoundError(mask_path)
    if scaled_only and (scale_path is None):
        raise ValueError("scaled_only requires scale_name.")
    if scaled_only and scale_path is not None and (not scale_path.exists()):
        raise FileNotFoundError(scale_path)

    s_pre = np.load(pre_path)
    s_post = np.load(post_path)
    m_bg = _load_mask(mask_path)

    if scaled_only and scale_path is not None:
        scale = np.load(scale_path).astype(np.float64, copy=False)
        if scale.shape != m_bg.shape:
            raise ValueError(f"Scale shape mismatch in {bundle_dir}: {scale.shape} vs {m_bg.shape}.")
        m_bg = m_bg & (scale > 1.0)

    if s_pre.shape != s_post.shape:
        raise ValueError(f"Score shape mismatch in {bundle_dir}: {s_pre.shape} vs {s_post.shape}.")
    if m_bg.shape != s_pre.shape:
        raise ValueError(f"Mask shape mismatch in {bundle_dir}: {m_bg.shape} vs {s_pre.shape}.")

    return s_pre[m_bg].astype(np.float64, copy=False), s_post[m_bg].astype(np.float64, copy=False)


def _make_synthetic_log_scores(
    *,
    seed: int,
    n: int,
    tail_frac: float,
    gamma_max: float,
) -> tuple[np.ndarray, np.ndarray, float, float, float]:
    """
    Returns (x_pre, x_post, x_gate, p_scaled, gamma_scaled_median) in log10(score) space.

    The construction is intentionally simple and deterministic:
      - x_pre is a heavy-tailed mixture distribution (a proxy for structured H0 artifacts).
      - a bounded, shrink-only penalty gamma>=1 is applied on the upper tail only.
    """
    rng = np.random.default_rng(int(seed))
    n = int(n)
    tail_frac = float(tail_frac)
    gamma_max = float(gamma_max)
    if n <= 10_000:
        raise ValueError("n must be > 10_000 for a stable PDF.")
    if not (0.0 < tail_frac < 0.5):
        raise ValueError("tail_frac must be in (0, 0.5).")
    if gamma_max <= 1.0:
        raise ValueError("gamma_max must be > 1.")

    # Heavy-tailed distribution in log-score space: mixture of Gaussians.
    # (Values are in log10 domain, so the "right tail" corresponds to larger values.)
    mix = rng.uniform(size=n) < 0.985
    x_pre = np.empty(n, dtype=np.float64)
    x_pre[mix] = rng.normal(loc=-11.35, scale=0.17, size=int(mix.sum()))
    x_pre[~mix] = rng.normal(loc=-10.95, scale=0.25, size=int((~mix).sum()))

    # Apply a bounded penalty only to the upper tail.
    x_gate = float(np.quantile(x_pre, 1.0 - tail_frac))
    gamma = np.ones_like(x_pre)
    m = x_pre > x_gate
    beta = 0.18
    gamma[m] = 1.0 + (gamma_max - 1.0) * (1.0 - np.exp(-(x_pre[m] - x_gate) / beta))
    gamma = np.clip(gamma, 1.0, gamma_max)

    # S_post = S_pre / gamma  =>  log10(S_post) = log10(S_pre) - log10(gamma)
    x_post = x_pre - np.log10(gamma)

    return x_pre, x_post, x_gate, float(np.mean(m)), float(np.median(gamma[m])) if np.any(m) else 1.0


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Generate a shrink-only tail suppression figure (Figure 3): pre/post score PDFs in log-score space. "
            "By default, this loads an empirical example from disk; synthetic mode can generate a toy heavy-tail "
            "example for intuition."
        )
    )
    parser.add_argument(
        "--mode",
        choices=["synthetic", "empirical"],
        default="empirical",
        help="Plot mode (default: %(default)s).",
    )
    parser.add_argument(
        "--root",
        default="runs/real/twinkling_calculi_calcifications_prf500_str4_Lt8_msd_ka_scm_dl015_f050",
        help="Run root (contains bundle dirs or frame* dirs). Default is a local example path.",
    )
    parser.add_argument(
        "--pattern",
        default="**/frame*",
        help="Glob pattern under --root for bundle dirs (default: %(default)s).",
    )
    parser.add_argument(
        "--pre",
        default="score_stap_preka.npy",
        help="Pre-KA score filename (default: %(default)s).",
    )
    parser.add_argument(
        "--post",
        default="score_stap.npy",
        help="Post-KA score filename (default: %(default)s).",
    )
    parser.add_argument(
        "--mask-bg",
        default="mask_bg.npy",
        help="Background mask filename (default: %(default)s).",
    )
    parser.add_argument(
        "--scale",
        default="ka_scale_map.npy",
        help="Empirical mode: KA scale-map filename (default: %(default)s).",
    )
    parser.add_argument(
        "--scaled-only",
        action="store_true",
        help="Empirical mode: restrict to penalized pixels (gamma>1).",
    )
    parser.add_argument(
        "--drop-zeros",
        default=True,
        action=argparse.BooleanOptionalAction,
        help=(
            "Empirical mode: drop zero/negative scores to avoid point-mass spikes in log-score plots "
            "(default: %(default)s)."
        ),
    )
    parser.add_argument(
        "--alpha",
        type=float,
        default=1e-3,
        help="Example FPR target used to define a pre-KA threshold line on H0 (default: %(default)s).",
    )
    parser.add_argument(
        "--n-synth",
        type=int,
        default=450_000,
        help="Synthetic mode: number of samples to draw (default: %(default)s).",
    )
    parser.add_argument(
        "--tail-frac",
        type=float,
        default=0.02,
        help="Synthetic mode: fraction of scores that receive a penalty (default: %(default)s).",
    )
    parser.add_argument(
        "--gamma-max",
        type=float,
        default=3.0,
        help="Synthetic mode: maximum shrink-only penalty gamma (default: %(default)s).",
    )
    parser.add_argument(
        "--max-samples",
        type=int,
        default=2_000_000,
        help="Maximum pooled background pixels to plot (randomly subsamples if exceeded).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=0,
        help="RNG seed for subsampling (default: %(default)s).",
    )
    parser.add_argument(
        "--bins",
        type=int,
        default=260,
        help="Number of histogram bins (default: %(default)s).",
    )
    parser.add_argument(
        "--out",
        default="figs/paper/shrink_only_tail_suppression.pdf",
        help="Output PDF path (default: %(default)s).",
    )
    args = parser.parse_args()

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    alpha = float(args.alpha)
    alpha = max(min(alpha, 0.25), 1e-9)

    if args.mode == "synthetic":
        x_pre, x_post, x_gate, p_scaled, gamma_med = _make_synthetic_log_scores(
            seed=int(args.seed),
            n=int(args.n_synth),
            tail_frac=float(args.tail_frac),
            gamma_max=float(args.gamma_max),
        )
        x_tau = float(np.quantile(x_pre, 1.0 - alpha))
        tail_pre = float(np.mean(x_pre >= x_tau))
        tail_post_at_tau = float(np.mean(x_post >= x_tau))
        extra_note = f"synthetic: p_scaled≈{p_scaled:.3g}, med $\\gamma$≈{gamma_med:.3g}"
    else:
        root = Path(args.root)
        bundle_dirs = _iter_bundle_dirs(root, args.pattern)
        if not bundle_dirs:
            raise RuntimeError("No bundles found.")

        s_pre_all: list[np.ndarray] = []
        s_post_all: list[np.ndarray] = []
        for b in bundle_dirs:
            try:
                s_pre_b, s_post_b = _load_scores(
                    b,
                    args.pre,
                    args.post,
                    args.mask_bg,
                    scale_name=str(args.scale) if args.scaled_only else None,
                    scaled_only=bool(args.scaled_only),
                )
            except FileNotFoundError:
                continue
            if s_pre_b.size == 0:
                continue
            s_pre_all.append(s_pre_b)
            s_post_all.append(s_post_b)

        if not s_pre_all:
            raise RuntimeError(
                f"No bundles under {root} contained {args.pre!r}, {args.post!r}, and {args.mask_bg!r}."
            )

        s_pre = np.concatenate(s_pre_all, axis=0)
        s_post = np.concatenate(s_post_all, axis=0)

        # Optional subsample for speed.
        n = s_pre.size
        if n > args.max_samples:
            rng = np.random.default_rng(args.seed)
            idx = rng.choice(n, size=args.max_samples, replace=False)
            s_pre = s_pre[idx]
            s_post = s_post[idx]

        extra_note_parts = ["empirical"]
        if args.scaled_only:
            extra_note_parts.append("scaled-only")

        # Define the example pre-KA threshold on the full empirical background set.
        # (Optional `--drop-zeros` is a visualization aid for log-score plots and
        # must not change the threshold definition.)
        tau_pre = float(np.quantile(s_pre, 1.0 - alpha))
        tail_pre = float(np.mean(s_pre >= tau_pre))
        tail_post_at_tau = float(np.mean(s_post >= tau_pre))

        if args.drop_zeros:
            keep = (s_pre > 0) & (s_post > 0)
            dropped = 1.0 - float(np.mean(keep))
            extra_note_parts.append(f"dropped≈{dropped:.3g}")
            s_pre = s_pre[keep]
            s_post = s_post[keep]
        if s_pre.size == 0:
            raise RuntimeError("No background samples remain after filtering.")
        extra_note = ", ".join(extra_note_parts)

        eps = 1e-12
        x_pre = np.log10(s_pre + eps)
        x_post = np.log10(s_post + eps)
        x_tau = float(np.log10(tau_pre + eps))

    x_all = np.concatenate([x_pre, x_post], axis=0)
    lo = float(np.quantile(x_all, 1e-3))
    hi = float(np.quantile(x_all, 1 - 1e-4))
    pad = 0.08 * (hi - lo + 1e-6)
    x_min = lo - pad
    x_max = hi + pad

    bins = int(args.bins)
    hist_pre, edges = np.histogram(x_pre, bins=bins, range=(x_min, x_max), density=True)
    hist_post, _ = np.histogram(x_post, bins=bins, range=(x_min, x_max), density=True)
    centers = 0.5 * (edges[:-1] + edges[1:])

    mpl.rcParams.update(
        {
            "font.family": "serif",
            "mathtext.fontset": "cm",
            "font.size": 9,
            "axes.linewidth": 0.8,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "legend.frameon": False,
        }
    )

    fig, axs = plt.subplots(1, 2, figsize=(7.0, 2.55), sharex=True)

    for ax, logy, title in [
        (axs[0], False, "PDF (linear y; body)"),
        (axs[1], True, "PDF (log y; tail)"),
    ]:
        ax.plot(
            centers,
            hist_pre,
            color="#666666",
            lw=1.6,
            linestyle="--",
            label=r"Pre-regularizer $S_{\mathrm{pre}}$",
        )
        ax.plot(
            centers,
            hist_post,
            color="#1f77b4",
            lw=1.7,
            linestyle="-",
            label=r"Post-regularizer $S_{\mathrm{post}}$",
        )

        ax.axvline(x_tau, color="#444444", lw=1.0, alpha=0.8)
        ax.axvspan(x_tau, x_max, color="gray", alpha=0.12, lw=0)

        ax.set_title(title, fontsize=9.5)
        ax.grid(True, axis="y", alpha=0.25, lw=0.6)
        if logy:
            ax.set_yscale("log")
            ax.set_ylim(max(1e-6, 0.1 * np.min(hist_post[hist_post > 0])), 1.2 * np.max(hist_pre))

    axs[0].set_ylabel("Density")
    for ax in axs:
        ax.set_xlim(x_min, x_max)
        ax.set_xlabel(r"$\log_{10}(S+\epsilon)$")

    # Legend + annotation in tail panel.
    axs[0].legend(loc="upper left", fontsize=8)
    axs[1].legend(loc="upper left", fontsize=8)
    axs[1].text(
        0.98,
        0.05,
        (
            f"alpha={alpha:g} (pre threshold)\n"
            f"tail mass pre ≈ {tail_pre:.3g}\n"
            f"tail mass post ≈ {tail_post_at_tau:.3g}\n"
            f"{extra_note}"
        ),
        transform=axs[1].transAxes,
        ha="right",
        va="bottom",
        fontsize=8,
        bbox={"boxstyle": "round,pad=0.25", "facecolor": "white", "edgecolor": "#dddddd", "alpha": 0.95},
    )

    fig.tight_layout()
    fig.savefig(out_path, bbox_inches="tight", format="pdf")
    plt.close(fig)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
