#!/usr/bin/env python3
"""
Plot ROC-style curves (TPR vs FPR) for labeled k-Wave brain stress tests.

This is a "curve credibility" companion to the strict low-FPR operating-point
tables in the main paper. We summarize the ROC shape by aggregating over the
five disjoint 64-frame windows used in the main experiments, reporting median
and IQR TPR at each FPR grid point.

Inputs: precomputed per-window pooled score arrays in each bundle directory:
  - base_pos.npy / base_neg.npy (baseline PD right-tail scores)
  - score_stap_preka.npy (STAP matched-subspace score map; only present in MC--SVD runs)

Default bundle roots match the repository's latest fair-filter matrix output layout.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from eval.metrics import roc_curve


def _glob_windows(root: Path) -> list[Path]:
    # Expect directories like pw_*_win0_off0, ..., each with base_pos/base_neg arrays.
    wins = [p for p in root.glob("pw_*_win*_off*") if p.is_dir()]
    wins.sort()
    return wins


def _load_scores(bundle_dir: Path, kind: str) -> tuple[np.ndarray, np.ndarray]:
    kind = kind.lower().strip()
    if kind == "base":
        pos = np.load(bundle_dir / "base_pos.npy").astype(np.float64, copy=False).ravel()
        neg = np.load(bundle_dir / "base_neg.npy").astype(np.float64, copy=False).ravel()
        return pos, neg
    if kind == "stap":
        # Use the pre-KA matched-subspace STAP score map and the same flow/bg masks
        # used throughout the Brain-* operating-point reporting.
        score = np.load(bundle_dir / "score_stap_preka.npy").astype(np.float64, copy=False)
        mf = np.load(bundle_dir / "mask_flow.npy").astype(bool, copy=False)
        mb = np.load(bundle_dir / "mask_bg.npy").astype(bool, copy=False)
        pos = score[mf].ravel()
        neg = score[mb].ravel()
        return pos, neg
    raise ValueError(f"Unsupported kind: {kind!r}")


def _tpr_on_grid(pos: np.ndarray, neg: np.ndarray, fpr_grid: np.ndarray) -> np.ndarray:
    fpr, tpr, _ = roc_curve(pos, neg, num_thresh=8192)
    order = np.argsort(fpr)
    fpr = fpr[order]
    tpr = tpr[order]
    return np.interp(fpr_grid, fpr, tpr, left=float(tpr[0]), right=float(tpr[-1]))


def _summarize_windows(
    window_dirs: list[Path],
    kind: str,
    fpr_grid: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if not window_dirs:
        raise ValueError("No window dirs provided.")
    tprs: list[np.ndarray] = []
    for d in window_dirs:
        pos, neg = _load_scores(d, kind)
        tprs.append(_tpr_on_grid(pos, neg, fpr_grid))
    T = np.stack(tprs, axis=0)  # (n_win, n_grid)
    med = np.quantile(T, 0.5, axis=0)
    q25 = np.quantile(T, 0.25, axis=0)
    q75 = np.quantile(T, 0.75, axis=0)
    return med, q25, q75


def main() -> None:
    ap = argparse.ArgumentParser(description="Plot k-Wave brain stress-test ROC curves (median+IQR over windows).")
    ap.add_argument(
        "--runs-root",
        type=Path,
        default=Path("runs/pilot/fair_filter_matrix_pd_r3_localbaselines"),
        help="Root containing per-regime run folders (default: %(default)s).",
    )
    ap.add_argument(
        "--out-pdf",
        type=Path,
        default=Path("figs/paper/brain_kwave_roc_curves.pdf"),
        help="Output figure path (PDF).",
    )
    ap.add_argument("--also-png", action="store_true", help="Also write a PNG next to the PDF.")
    args = ap.parse_args()

    runs_root = Path(args.runs_root)
    out_pdf = Path(args.out_pdf)
    out_pdf.parent.mkdir(parents=True, exist_ok=True)

    regimes = [
        (
            "Open-skull surface-artifact\nstress test",
            "open_seed1",
            runs_root / "open_seed1_mcsvd_full",
            runs_root / "open_seed1_svd_similarity_full",
            runs_root / "open_seed1_local_svd_full",
            runs_root / "open_seed1_rpca_full",
            runs_root / "open_seed1_hosvd_full",
        ),
        (
            "Structured-clutter\nleakage stress test",
            "skullor_seed2",
            runs_root / "skullor_seed2_mcsvd_full",
            runs_root / "skullor_seed2_svd_similarity_full",
            runs_root / "skullor_seed2_local_svd_full",
            runs_root / "skullor_seed2_rpca_full",
            runs_root / "skullor_seed2_hosvd_full",
        ),
    ]

    # Common FPR grid for plotting (log-x). The labeled k-Wave windows have n_neg~5e4,
    # so the empirical floor is ~2e-5; we start slightly below to show the quantization.
    fpr_grid = np.logspace(-5, 0, 240, dtype=np.float64)

    try:
        import matplotlib.pyplot as plt
    except Exception as exc:  # pragma: no cover
        raise SystemExit(f"matplotlib required to plot: {exc}")

    plt.rcParams.update(
        {
            "font.family": "serif",
            "mathtext.fontset": "cm",
            "font.size": 10,
            "axes.titlesize": 11,
            "axes.labelsize": 10,
            "legend.fontsize": 9,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )

    colors = {
        "mcsvd": "#666666",
        "svd_similarity": "#9467bd",
        "local_svd": "#8c564b",
        "rpca": "#2ca02c",
        "hosvd": "#ff7f0e",
        "stap": "#1f77b4",
    }

    fig, axes = plt.subplots(1, 2, figsize=(9.2, 4.0), constrained_layout=True)
    if not isinstance(axes, np.ndarray):
        axes = np.asarray([axes])

    for ax, (title, tag, d_mcsvd, d_svd_sim, d_local_svd, d_rpca, d_hosvd) in zip(
        axes, regimes, strict=False
    ):
        # Window directories.
        win_mcsvd = _glob_windows(d_mcsvd)
        win_svd_sim = _glob_windows(d_svd_sim)
        win_local_svd = _glob_windows(d_local_svd)
        win_rpca = _glob_windows(d_rpca)
        win_hosvd = _glob_windows(d_hosvd)
        if not (win_mcsvd and win_svd_sim and win_local_svd and win_rpca and win_hosvd):
            raise SystemExit(
                f"Missing windows for {tag}: mcsvd={len(win_mcsvd)}, svd_sim={len(win_svd_sim)}, "
                f"local_svd={len(win_local_svd)}, rpca={len(win_rpca)}, hosvd={len(win_hosvd)}"
            )

        # Baselines: base curves from their own bundles.
        mc_med, mc_q25, mc_q75 = _summarize_windows(win_mcsvd, "base", fpr_grid)
        ss_med, ss_q25, ss_q75 = _summarize_windows(win_svd_sim, "base", fpr_grid)
        ls_med, ls_q25, ls_q75 = _summarize_windows(win_local_svd, "base", fpr_grid)
        rp_med, rp_q25, rp_q75 = _summarize_windows(win_rpca, "base", fpr_grid)
        ho_med, ho_q25, ho_q75 = _summarize_windows(win_hosvd, "base", fpr_grid)
        # STAP: stap curves from the MC--SVD bundles.
        st_med, st_q25, st_q75 = _summarize_windows(win_mcsvd, "stap", fpr_grid)

        ax.plot(fpr_grid, mc_med, color=colors["mcsvd"], linewidth=1.5, label="MC--SVD")
        ax.fill_between(fpr_grid, mc_q25, mc_q75, color=colors["mcsvd"], alpha=0.14, linewidth=0)

        ax.plot(
            fpr_grid,
            ss_med,
            color=colors["svd_similarity"],
            linewidth=1.3,
            label="Adaptive SVD (SV similarity cutoff)",
        )
        ax.fill_between(fpr_grid, ss_q25, ss_q75, color=colors["svd_similarity"], alpha=0.10, linewidth=0)

        ax.plot(
            fpr_grid,
            ls_med,
            color=colors["local_svd"],
            linewidth=1.3,
            label="Block-wise local SVD (overlap-add)",
        )
        ax.fill_between(fpr_grid, ls_q25, ls_q75, color=colors["local_svd"], alpha=0.10, linewidth=0)

        ax.plot(fpr_grid, rp_med, color=colors["rpca"], linewidth=1.4, label="RPCA")
        ax.fill_between(fpr_grid, rp_q25, rp_q75, color=colors["rpca"], alpha=0.12, linewidth=0)

        ax.plot(fpr_grid, ho_med, color=colors["hosvd"], linewidth=1.4, label="HOSVD")
        ax.fill_between(fpr_grid, ho_q25, ho_q75, color=colors["hosvd"], alpha=0.12, linewidth=0)

        ax.plot(
            fpr_grid,
            st_med,
            color=colors["stap"],
            linewidth=1.8,
            label="STAP on MC--SVD residual",
        )
        ax.fill_between(fpr_grid, st_q25, st_q75, color=colors["stap"], alpha=0.14, linewidth=0)

        ax.set_xscale("log")
        ax.set_xlim(fpr_grid.min(), 1.0)
        ax.set_ylim(-0.02, 1.02)
        ax.grid(True, alpha=0.3)
        ax.set_title(title)
        ax.set_xlabel("FPR")
        if ax is axes[0]:
            ax.set_ylabel("TPR")
        else:
            ax.set_yticklabels([])

        # Indicate the operating points used in the main low-FPR tables.
        for x in (1e-4, 3e-4, 1e-3):
            ax.axvline(x, color="#000000", alpha=0.10, linewidth=0.8)

        # Supported empirical floor under per-window negatives (quantization).
        # Each window has its own n_neg; for these fixed masks it is effectively constant.
        try:
            n_negs = [np.load(p / "base_neg.npy").size for p in win_mcsvd]
            n_neg = int(min(n_negs)) if n_negs else 0
            if n_neg > 0:
                ax.axvline(
                    1.0 / float(n_neg),
                    color="#000000",
                    alpha=0.12,
                    linewidth=0.9,
                    linestyle="--",
                )
        except Exception:
            pass

    axes[0].legend(loc="lower right", frameon=False, ncol=1)
    fig.suptitle("Labeled brain-simulation ROC curves (median and IQR over 5 disjoint windows)")

    fig.savefig(out_pdf, bbox_inches="tight", pad_inches=0.06)
    if bool(args.also_png):
        fig.savefig(out_pdf.with_suffix(".png"), dpi=300, bbox_inches="tight", pad_inches=0.06)
    plt.close(fig)

    print(f"[brain-roc-fig] wrote {out_pdf}")


if __name__ == "__main__":
    main()
