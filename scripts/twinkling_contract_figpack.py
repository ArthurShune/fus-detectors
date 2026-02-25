from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np


def _load_json(path: Path) -> dict[str, Any]:
    obj = json.loads(path.read_text())
    if not isinstance(obj, dict):
        raise ValueError(f"Expected dict JSON at {path}")
    return obj


def _iter_bundle_dirs(root: Path) -> list[Path]:
    out: list[Path] = []
    for meta_path in root.rglob("meta.json"):
        if meta_path.is_file():
            out.append(meta_path.parent)
    out.sort()
    return out


def _load_meta(bundle_dir: Path) -> dict[str, Any] | None:
    try:
        meta = json.loads((bundle_dir / "meta.json").read_text())
    except Exception:
        return None
    return meta if isinstance(meta, dict) else None


def _finite_1d(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=np.float64).ravel()
    return x[np.isfinite(x)]


def _right_tail_threshold(bg_scores: np.ndarray, alpha: float) -> tuple[float | None, float | None]:
    bg = _finite_1d(bg_scores)
    n = int(bg.size)
    if n <= 0:
        return None, None
    a = float(alpha)
    if not np.isfinite(a) or a <= 0.0:
        return float("inf"), 0.0
    if a >= 1.0:
        return float(np.min(bg)), 1.0

    k = int(np.ceil(a * n))
    k = max(1, min(k, n))
    tau = float(np.partition(bg, n - k)[n - k])
    realized = float(np.mean(bg >= tau))
    return tau, realized


def _safe_int_dict(d: object) -> dict[str, int]:
    out: dict[str, int] = {}
    if not isinstance(d, dict):
        return out
    for k, v in d.items():
        if not isinstance(k, str):
            continue
        try:
            out[k] = int(v)  # type: ignore[arg-type]
        except Exception:
            continue
    return out


def _stacked_bar(ax, labels: list[str], series: dict[str, list[int]], *, title: str) -> None:
    import matplotlib.pyplot as plt  # type: ignore

    keys = list(series.keys())
    x = np.arange(len(labels))
    bottom = np.zeros_like(x, dtype=np.float64)
    colors = plt.get_cmap("tab10")
    for i, k in enumerate(keys):
        vals = np.array(series[k], dtype=np.float64)
        ax.bar(x, vals, bottom=bottom, label=k, color=colors(i % 10), alpha=0.9)
        bottom += vals
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=15, ha="right")
    ax.set_title(title)
    ax.grid(True, axis="y", alpha=0.25)
    ax.legend(fontsize=8, frameon=False)


def _select_best_calculi_example(calculi_root: Path, *, fpr_target: float = 1e-3) -> Path | None:
    """
    Pick a bundle where KA applies and where bg tail (pre>=tau) decreases the most after KA
    at a per-frame threshold tau chosen on bg(pre) to hit fpr_target.
    """
    best: tuple[int, Path] | None = None
    for bd in _iter_bundle_dirs(calculi_root):
        meta = _load_meta(bd)
        if meta is None:
            continue
        tele = meta.get("stap_fallback_telemetry") or {}
        if not isinstance(tele, dict) or not bool(tele.get("score_ka_v2_applied", False)):
            continue
        s_pre_path = bd / "score_stap_preka.npy"
        s_post_path = bd / "score_stap.npy"
        bg_path = bd / "mask_bg.npy"
        if not (s_pre_path.is_file() and s_post_path.is_file() and bg_path.is_file()):
            continue
        s_pre = np.load(s_pre_path, allow_pickle=False).astype(np.float64)
        s_post = np.load(s_post_path, allow_pickle=False).astype(np.float64)
        bg = np.load(bg_path, allow_pickle=False).astype(bool)
        tau, _ = _right_tail_threshold(s_pre[bg], float(fpr_target))
        if tau is None:
            continue
        area_pre = int(np.sum(bg & (s_pre >= float(tau))))
        area_post = int(np.sum(bg & (s_post >= float(tau))))
        delta = area_pre - area_post
        if best is None or delta > best[0]:
            best = (delta, bd)
    return best[1] if best is not None else None


def _write_calculi_tail_example_figure(bundle_dir: Path, out_png: Path, *, fpr_target: float = 1e-3) -> dict[str, Any]:
    import matplotlib.pyplot as plt  # type: ignore

    save_dpi = 250

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

    meta = _load_meta(bundle_dir) or {}
    state = (meta.get("ka_contract_v2") or {}).get("state")
    reason = (meta.get("ka_contract_v2") or {}).get("reason")
    tele = meta.get("stap_fallback_telemetry") or {}

    s_pre = np.load(bundle_dir / "score_stap_preka.npy", allow_pickle=False).astype(np.float64)
    s_post = np.load(bundle_dir / "score_stap.npy", allow_pickle=False).astype(np.float64)
    bg = np.load(bundle_dir / "mask_bg.npy", allow_pickle=False).astype(bool)
    flow = np.load(bundle_dir / "mask_flow.npy", allow_pickle=False).astype(bool)
    scale = np.load(bundle_dir / "ka_scale_map.npy", allow_pickle=False).astype(np.float64)
    gate = None
    if (bundle_dir / "ka_gate_map.npy").is_file():
        gate = np.load(bundle_dir / "ka_gate_map.npy", allow_pickle=False).astype(bool)

    tau, fpr_real = _right_tail_threshold(s_pre[bg], float(fpr_target))
    if tau is None or fpr_real is None:
        tau = float("nan")
        fpr_real = float("nan")
    area_pre = int(np.sum(bg & (s_pre >= float(tau)))) if np.isfinite(tau) else 0
    area_post = int(np.sum(bg & (s_post >= float(tau)))) if np.isfinite(tau) else 0

    eps = 1e-14
    pre_log = np.log10(s_pre + eps)
    post_log = np.log10(s_post + eps)
    scale_log = np.log10(scale + 1e-12)

    # Tail histogram on bg proxy.
    bg_pre = pre_log[bg & np.isfinite(pre_log)]
    bg_post = post_log[bg & np.isfinite(post_log)]
    # Focus on the upper tail of the pre distribution for visibility.
    lo = float(np.quantile(bg_pre, 0.98)) if bg_pre.size else float("nan")
    hi = float(np.max(bg_pre)) if bg_pre.size else float("nan")
    if not np.isfinite(lo) or not np.isfinite(hi) or lo >= hi:
        lo, hi = -14.0, -9.0

    fig = plt.figure(figsize=(10.5, 6.2))
    gs = fig.add_gridspec(2, 3, height_ratios=[1.0, 1.0], width_ratios=[1.0, 1.0, 1.1])
    ax0 = fig.add_subplot(gs[0, 0])
    ax1 = fig.add_subplot(gs[0, 1])
    ax2 = fig.add_subplot(gs[0, 2])
    ax3 = fig.add_subplot(gs[1, :])

    im0 = ax0.imshow(pre_log, cmap="magma")
    ax0.set_title(r"$\log_{10}(S_{\mathrm{pre}}+\epsilon)$")
    ax0.axis("off")
    fig.colorbar(im0, ax=ax0, fraction=0.046, pad=0.02)

    im1 = ax1.imshow(post_log, cmap="magma")
    ax1.set_title(r"$\log_{10}(S_{\mathrm{post}}+\epsilon)$")
    ax1.axis("off")
    fig.colorbar(im1, ax=ax1, fraction=0.046, pad=0.02)

    im2 = ax2.imshow(scale, cmap="viridis", vmin=1.0, vmax=float(np.max(scale)))
    ax2.set_title(r"Scale map $\gamma(\phi)\geq 1$")
    ax2.axis("off")
    fig.colorbar(im2, ax=ax2, fraction=0.046, pad=0.02)
    if gate is not None:
        # Overlay gate outline.
        ax2.contour(gate.astype(float), levels=[0.5], colors="white", linewidths=0.8, alpha=0.9)

    bins = np.linspace(lo, hi, 40)
    ax3.hist(bg_pre, bins=bins, alpha=0.55, label=r"$H_0$ proxy (pre)", color="#666666")
    ax3.hist(bg_post, bins=bins, alpha=0.55, label=r"$H_0$ proxy (post)", color="#1f77b4")
    if np.isfinite(tau):
        ax3.axvline(
            np.log10(float(tau) + eps),
            color="k",
            linestyle="--",
            linewidth=1.0,
            label=rf"$\tau_{{\alpha}}$ @ $\alpha={fpr_target:g}$",
        )

    state_map = {
        "C0_OFF": r"$\mathcal{R}_0$ (uninformative)",
        "C1_SAFETY": r"$\mathcal{R}_1$ (guarded)",
        "C2_UPLIFT": r"$\mathcal{R}_2$ (actionable)",
    }
    state_s = state_map.get(str(state), str(state))
    reason_s = str(reason).replace("_", " ")
    ax3.set_title(
        f"Calculi example: regime={state_s}, reason={reason_s}; "
        f"bg-tail area (pre→post)={area_pre}→{area_post} @ FPR≈{fpr_real:.3g}"
    )
    ax3.set_xlabel(r"$\log_{10}(S+\epsilon)$ on $H_0$ proxy (tail-focused)")
    ax3.set_ylabel("count")
    ax3.grid(True, alpha=0.25)
    ax3.legend(fontsize=9, frameon=False)

    fig.tight_layout()
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, dpi=save_dpi, bbox_inches="tight", pad_inches=0.05)
    plt.close(fig)

    max_abs_delta_flow = float(np.max(np.abs(s_pre[flow] - s_post[flow]))) if flow.any() else 0.0
    max_abs_delta_bg = float(np.max(np.abs(s_pre[bg] - s_post[bg]))) if bg.any() else 0.0
    return {
        "bundle_dir": str(bundle_dir),
        "state": state,
        "reason": reason,
        "fpr_target": float(fpr_target),
        "tau": float(tau),
        "fpr_realized": float(fpr_real),
        "bg_tail_area_pre": int(area_pre),
        "bg_tail_area_post": int(area_post),
        "scaled_pixel_fraction": tele.get("score_ka_v2_scaled_pixel_fraction") if isinstance(tele, dict) else None,
        "max_abs_delta_flow": max_abs_delta_flow,
        "max_abs_delta_bg": max_abs_delta_bg,
        "scale_max": float(np.max(scale)),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate Twinkling contract summary figures (states/reasons + calculi tail example).")
    parser.add_argument("--along-summary", type=Path, required=True)
    parser.add_argument("--across-summary", type=Path, required=True)
    parser.add_argument("--calculi-summary", type=Path, required=True)
    parser.add_argument("--out-state-reason-png", type=Path, required=True)
    parser.add_argument("--out-calculi-example-png", type=Path, required=True)
    parser.add_argument("--out-json", type=Path, required=True)
    parser.add_argument("--calculi-root", type=Path, required=True, help="Root containing calculi bundles (to auto-pick an example).")
    parser.add_argument("--calculi-example-bundle", type=Path, default=None, help="Optional explicit bundle dir (overrides auto-pick).")
    parser.add_argument("--example-fpr-target", type=float, default=1e-3, help="FPR target for selecting/annotating the example (default: %(default)s).")
    args = parser.parse_args()

    along = _load_json(args.along_summary)
    across = _load_json(args.across_summary)
    calculi = _load_json(args.calculi_summary)

    labels = ["Gammex along", "Gammex across", "Calculi"]
    sums = [along, across, calculi]

    # Extract counts.
    state_keys = ["C0_OFF", "C1_SAFETY", "C2_UPLIFT"]

    # Keep reason charts readable: plot only top reasons (by total count), collapse the rest into "other".
    max_reasons = 6
    reason_total: dict[str, int] = {}
    for s in sums:
        rc = _safe_int_dict(s.get("ka_reason_counts"))
        for k, v in rc.items():
            reason_total[k] = int(reason_total.get(k, 0)) + int(v)
    reason_keys = [k for k, _ in sorted(reason_total.items(), key=lambda kv: (-kv[1], kv[0]))][:max_reasons]

    state_series: dict[str, list[int]] = {k: [] for k in state_keys}
    reason_series: dict[str, list[int]] = {k: [] for k in reason_keys}
    reason_series["other"] = []
    inv_rows: list[dict[str, Any]] = []
    for lab, s in zip(labels, sums, strict=True):
        sc = _safe_int_dict(s.get("ka_state_counts"))
        rc = _safe_int_dict(s.get("ka_reason_counts"))
        for k in state_keys:
            state_series[k].append(int(sc.get(k, 0)))
        other = 0
        for k, v in rc.items():
            if k in reason_series and k != "other":
                continue
            other += int(v)
        for k in reason_keys:
            reason_series[k].append(int(rc.get(k, 0)))
        reason_series["other"].append(int(other))
        inv_rows.append(
            {
                "label": lab,
                "bundle_count": int(s.get("bundle_count") or 0),
                "ka_applied_bundles": int(s.get("ka_applied_bundles") or 0),
                "max_abs_delta_flow": float(s.get("max_abs_delta_flow") or 0.0),
            }
        )

    # Plot state + reason bars.
    import matplotlib.pyplot as plt  # type: ignore

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

    # Replace internal state keys with paper-facing regime names.
    state_map = {
        "C0_OFF": r"$\mathcal{R}_0$ (uninformative)",
        "C1_SAFETY": r"$\mathcal{R}_1$ (guarded)",
        "C2_UPLIFT": r"$\mathcal{R}_2$ (actionable)",
    }
    reason_map = {k: k.replace("_", " ") for k in reason_series.keys()}
    reason_map["ok"] = "ok"
    reason_map["other"] = "other"

    labels = ["Gammex (along)", "Gammex (across)", "Calculi"]

    state_series_plot = {state_map.get(k, k): v for k, v in state_series.items()}
    reason_series_plot = {reason_map.get(k, k): v for k, v in reason_series.items()}

    fig, axes = plt.subplots(1, 2, figsize=(10.8, 3.8))
    _stacked_bar(axes[0], labels, state_series_plot, title="Prior regimes")
    _stacked_bar(axes[1], labels, reason_series_plot, title="Top reasons (disable/enable)")
    fig.suptitle("KA prior summary (regimes + reasons)")
    fig.tight_layout(rect=[0, 0.0, 1, 0.92])
    args.out_state_reason_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out_state_reason_png, dpi=250, bbox_inches="tight", pad_inches=0.05)
    plt.close(fig)

    # Calculi example.
    if args.calculi_example_bundle is not None:
        example_bundle = args.calculi_example_bundle
    else:
        example_bundle = _select_best_calculi_example(args.calculi_root, fpr_target=float(args.example_fpr_target))
        if example_bundle is None:
            raise SystemExit("Could not auto-select a calculi example bundle (no KA-applied bundles found).")
    example_info = _write_calculi_tail_example_figure(
        example_bundle, args.out_calculi_example_png, fpr_target=float(args.example_fpr_target)
    )

    report = {
        "inputs": {
            "along_summary": str(args.along_summary),
            "across_summary": str(args.across_summary),
            "calculi_summary": str(args.calculi_summary),
            "calculi_root": str(args.calculi_root),
        },
        "outputs": {
            "state_reason_png": str(args.out_state_reason_png),
            "calculi_example_png": str(args.out_calculi_example_png),
        },
        "invariance": inv_rows,
        "calculi_example": example_info,
    }
    args.out_json.parent.mkdir(parents=True, exist_ok=True)
    args.out_json.write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
