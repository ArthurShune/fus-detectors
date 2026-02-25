from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np


def _load_npy(bundle_dir: Path, name: str) -> np.ndarray | None:
    path = bundle_dir / f"{name}.npy"
    if not path.is_file():
        return None
    return np.load(path, allow_pickle=False)


def _load_meta(bundle_dir: Path) -> dict[str, Any]:
    meta_path = bundle_dir / "meta.json"
    if not meta_path.is_file():
        raise FileNotFoundError(f"meta.json not found in {bundle_dir}")
    meta = json.loads(meta_path.read_text())
    if not isinstance(meta, dict):
        raise RuntimeError("meta.json is not a JSON object")
    return meta


def _percentiles(spec: str) -> tuple[float, float]:
    parts = [p.strip() for p in (spec or "").split(",") if p.strip()]
    if len(parts) != 2:
        raise ValueError("Expected 'lo,hi' percentiles (e.g. 1,99).")
    lo = float(parts[0])
    hi = float(parts[1])
    if not (0.0 <= lo < hi <= 100.0):
        raise ValueError("Percentiles must satisfy 0 <= lo < hi <= 100.")
    return lo, hi


def _safe_quantile(x: np.ndarray, q: float, default: float) -> float:
    x = np.asarray(x, dtype=np.float64)
    x = x[np.isfinite(x)]
    if x.size == 0:
        return default
    return float(np.quantile(x, q))


def _fmt_float(x: Any) -> str:
    try:
        if x is None:
            return "NA"
        return f"{float(x):.3g}"
    except Exception:
        return "NA"


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Make a paper-ready multi-panel figure from a Shin RatBrain acceptance bundle.\n"
            "The bundle directory should contain pd_base.npy, pd_stap.npy, mask_flow.npy, "
            "base_m_alias_map.npy, base_peak_freq_map.npy, and optionally ka_scale_map.npy / "
            "ka_gate_map.npy / pd_stap_pre_ka.npy."
        )
    )
    parser.add_argument("--bundle", type=Path, required=True, help="Bundle directory (contains meta.json).")
    parser.add_argument("--out", type=Path, required=True, help="Output image path (.png).")
    parser.add_argument(
        "--title-style",
        type=str,
        default="paper",
        choices=["paper", "debug", "none"],
        help="Figure title style (default: %(default)s).",
    )
    parser.add_argument(
        "--pd-view",
        type=str,
        default="log10",
        choices=["linear", "log10", "sqrt"],
        help="Visualization transform for PD maps (default: %(default)s).",
    )
    parser.add_argument(
        "--pd-clip",
        type=str,
        default="1,99",
        help="PD clip percentiles for display (default: %(default)s).",
    )
    parser.add_argument(
        "--alias-clip",
        type=str,
        default="1,99",
        help="Alias metric clip percentiles (default: %(default)s).",
    )
    parser.add_argument(
        "--scale-max",
        type=float,
        default=2.0,
        help="Max scale shown for ka_scale_map (default: %(default)s).",
    )
    args = parser.parse_args()

    bundle_dir = args.bundle
    meta = _load_meta(bundle_dir)

    pd_base = _load_npy(bundle_dir, "pd_base")
    pd_stap = _load_npy(bundle_dir, "pd_stap")
    if pd_base is None or pd_stap is None:
        raise FileNotFoundError("Bundle must contain pd_base.npy and pd_stap.npy")

    pd_stap_pre = _load_npy(bundle_dir, "pd_stap_pre_ka")
    ka_scale = _load_npy(bundle_dir, "ka_scale_map")
    ka_gate = _load_npy(bundle_dir, "ka_gate_map")
    if pd_stap_pre is None and ka_scale is not None:
        eps = 1e-12
        pd_stap_pre = pd_stap.astype(np.float32, copy=False) / np.maximum(
            ka_scale.astype(np.float32, copy=False), eps
        )
    if pd_stap_pre is None:
        pd_stap_pre = pd_stap

    mask_flow = _load_npy(bundle_dir, "mask_flow")
    mask_bg = _load_npy(bundle_dir, "mask_bg")
    m_alias = _load_npy(bundle_dir, "base_m_alias_map")
    peak_freq = _load_npy(bundle_dir, "base_peak_freq_map")
    guard_frac = _load_npy(bundle_dir, "base_guard_frac_map")

    if mask_flow is None:
        raise FileNotFoundError("mask_flow.npy missing")
    if mask_bg is None:
        mask_bg = ~mask_flow.astype(bool)

    if m_alias is None:
        m_alias = np.zeros_like(pd_base, dtype=np.float32)
    if peak_freq is None:
        peak_freq = np.zeros_like(pd_base, dtype=np.float32)
    if guard_frac is None:
        guard_frac = np.zeros_like(pd_base, dtype=np.float32)

    pd_view = args.pd_view
    eps = 1e-12

    def _pd_xform(x: np.ndarray) -> np.ndarray:
        x = np.asarray(x, dtype=np.float32)
        if pd_view == "linear":
            return x
        if pd_view == "sqrt":
            return np.sqrt(np.maximum(x, 0.0))
        return np.log10(np.maximum(x, 0.0) + eps)

    pd_base_v = _pd_xform(pd_base)
    pd_pre_v = _pd_xform(pd_stap_pre)
    pd_post_v = _pd_xform(pd_stap)

    pd_lo, pd_hi = _percentiles(args.pd_clip)
    vmin_pd = min(
        _safe_quantile(pd_base_v, pd_lo / 100.0, float(np.min(pd_base_v))),
        _safe_quantile(pd_pre_v, pd_lo / 100.0, float(np.min(pd_pre_v))),
        _safe_quantile(pd_post_v, pd_lo / 100.0, float(np.min(pd_post_v))),
    )
    vmax_pd = max(
        _safe_quantile(pd_base_v, pd_hi / 100.0, float(np.max(pd_base_v))),
        _safe_quantile(pd_pre_v, pd_hi / 100.0, float(np.max(pd_pre_v))),
        _safe_quantile(pd_post_v, pd_hi / 100.0, float(np.max(pd_post_v))),
    )

    alias_lo, alias_hi = _percentiles(args.alias_clip)
    vmin_alias = _safe_quantile(m_alias, alias_lo / 100.0, float(np.min(m_alias)))
    vmax_alias = _safe_quantile(m_alias, alias_hi / 100.0, float(np.max(m_alias)))

    prf_hz = meta.get("prf_hz")
    nyq = float(prf_hz) / 2.0 if prf_hz is not None else float(np.max(peak_freq))

    # ---- Build title string ----
    title_style = str(args.title_style).lower().strip()
    title = None
    if title_style != "none":
        if title_style == "debug":
            ka_v2 = meta.get("ka_contract_v2") or {}
            ka_state = ka_v2.get("state") if isinstance(ka_v2, dict) else None
            ka_reason = ka_v2.get("reason") if isinstance(ka_v2, dict) else None
            ka_metrics = (ka_v2.get("metrics") or {}) if isinstance(ka_v2, dict) else {}
            score_tele = meta.get("stap_fallback_telemetry") or {}
            title = (
                f"{bundle_dir.name}\n"
                f"KA v2: {ka_state}/{ka_reason} risk={ka_metrics.get('risk_mode')} "
                f"p_shrink={_fmt_float(ka_metrics.get('p_shrink'))} "
                f"pf_peak_flow={_fmt_float(ka_metrics.get('pf_peak_flow'))}"
                f" (n_flow={ka_metrics.get('n_flow_proxy')}) "
                f"pf_peak_nonbg={_fmt_float(ka_metrics.get('pf_peak_nonbg'))} "
                f"guard_q90={_fmt_float(ka_metrics.get('guard_q90'))}\n"
                f"score_ka_v2_applied={score_tele.get('score_ka_v2_applied')} "
                f"scaled_px_frac={_fmt_float(score_tele.get('score_ka_v2_scaled_pixel_fraction'))} "
                f"scale_p90={_fmt_float(score_tele.get('score_ka_v2_scale_p90'))} "
                f"scale_max={_fmt_float(score_tele.get('score_ka_v2_scale_max'))}"
            )
        else:
            orig = meta.get("orig_data") if isinstance(meta, dict) else None
            clip = None
            frames = None
            if isinstance(orig, dict):
                iq_file = orig.get("iq_file")
                if isinstance(iq_file, str) and iq_file:
                    clip = Path(iq_file).stem
                frames_spec = orig.get("frames_spec")
                if isinstance(frames_spec, str) and frames_spec:
                    frames = frames_spec
            if clip is None:
                ds = meta.get("dataset") if isinstance(meta, dict) else None
                if isinstance(ds, dict):
                    name = ds.get("name")
                    if isinstance(name, str) and name:
                        clip = name
            title = "Shin RatBrain Fig3 example"
            if clip or frames:
                suffix = ", ".join([s for s in [clip, f"frames {frames}" if frames else None] if s])
                title = f"{title} ({suffix})"

    # ---- Plot ----
    import matplotlib.pyplot as plt

    plt.rcParams.update(
        {
            "figure.dpi": 220,
            "font.family": "serif",
            "mathtext.fontset": "cm",
            "font.size": 10,
            "axes.titlesize": 10,
            "axes.labelsize": 10,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )

    fig, axes = plt.subplots(2, 4, figsize=(16, 8), constrained_layout=True)
    if title:
        fig.suptitle(title, fontsize=11)

    def _imshow(ax, img, title_str, cmap="viridis", vmin=None, vmax=None):
        im = ax.imshow(img, cmap=cmap, vmin=vmin, vmax=vmax, origin="upper")
        ax.set_title(title_str, fontsize=9)
        ax.set_xticks([])
        ax.set_yticks([])
        return im

    _imshow(axes[0, 0], pd_base_v, r"Baseline score $S_{\mathrm{base}}$", vmin=vmin_pd, vmax=vmax_pd)
    _imshow(axes[0, 1], pd_pre_v, r"STAP score $S_{\mathrm{pre}}$", vmin=vmin_pd, vmax=vmax_pd)
    _imshow(axes[0, 2], pd_post_v, r"Post-regularization score $S_{\mathrm{post}}$", vmin=vmin_pd, vmax=vmax_pd)

    if ka_scale is None:
        _imshow(
            axes[0, 3],
            np.ones_like(pd_base, dtype=np.float32),
            r"Scale map $\gamma(i)\geq 1$ (missing)",
            vmin=1.0,
            vmax=1.0,
        )
    else:
        scale_show = np.clip(ka_scale.astype(np.float32, copy=False), 1.0, float(args.scale_max))
        _imshow(
            axes[0, 3],
            scale_show,
            rf"Scale map $\gamma(i)\geq 1$ (clip$\leq {args.scale_max:g}$)",
            vmin=1.0,
            vmax=float(args.scale_max),
        )

    # Overlay gate + flow mask on pd_post for interpretability.
    ax = axes[1, 0]
    _imshow(ax, pd_post_v, r"$S_{\mathrm{post}}$ + overlays", vmin=vmin_pd, vmax=vmax_pd)
    flow = mask_flow.astype(bool)
    ax.contour(flow.astype(float), levels=[0.5], colors="lime", linewidths=0.8)
    if ka_gate is not None:
        gate = ka_gate.astype(bool)
        ax.contour(gate.astype(float), levels=[0.5], colors="cyan", linewidths=0.8)

    _imshow(
        axes[1, 1],
        m_alias.astype(np.float32, copy=False),
        r"Alias metric $m_{\mathrm{alias}}$",
        cmap="coolwarm",
        vmin=vmin_alias,
        vmax=vmax_alias,
    )
    _imshow(
        axes[1, 2],
        peak_freq.astype(np.float32, copy=False),
        r"Peak freq. $f_{\mathrm{peak}}$ (Hz)",
        cmap="magma",
        vmin=0.0,
        vmax=nyq,
    )
    _imshow(
        axes[1, 3],
        guard_frac.astype(np.float32, copy=False),
        r"Guard fraction $r_g$",
        cmap="inferno",
        vmin=0.0,
        vmax=1.0,
    )

    out_path = args.out
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=200)
    plt.close(fig)
    print(f"[shin-plot] wrote {out_path}")


if __name__ == "__main__":
    main()
