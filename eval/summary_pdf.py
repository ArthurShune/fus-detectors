"""Utilities to render a one-page acceptance summary PDF."""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any, Dict, Iterable, Optional, Sequence

import matplotlib
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.backends.backend_pdf import PdfPages  # noqa: E402
from matplotlib.image import imread  # noqa: E402

MetricRows = Sequence[tuple[str, str]]


def _maybe_read(path: Path) -> Optional[Any]:
    if not path.exists():
        return None
    try:
        return imread(path)
    except Exception:
        return None


def _fmt_pct(x: float | None, digits: int = 3) -> str:
    if x is None or not math.isfinite(x):
        return "—"
    return f"{100.0 * x:.{digits}f}%"


def _fmt_num(x: float | None, digits: int = 3) -> str:
    if x is None or not math.isfinite(x):
        return "—"
    return f"{x:.{digits}f}"


def _gate_str(ok: bool) -> str:
    return "PASS" if ok else "FAIL"


def _add_table(ax, rows: MetricRows, headers: Iterable[str], font_size: int = 9) -> None:
    ax.axis("off")
    table = ax.table(
        cellText=[[label, value] for label, value in rows],
        colLabels=list(headers),
        cellLoc="left",
        loc="center",
    )
    table.auto_set_font_size(False)
    table.set_fontsize(font_size)
    table.scale(1.0, 1.2)


def save_summary_pdf(
    pdf_path: str,
    acceptance_json: str,
    thumbs_dir: str = "figs/outputs",
    title: str = "Acceptance Summary",
) -> None:
    """Render a landscape PDF summarising the most recent acceptance run."""
    matplotlib.use("Agg", force=True)
    data = json.loads(Path(acceptance_json).read_text())

    perf: Dict[str, Any] = data.get("performance", {})
    gates: Dict[str, Any] = data.get("gates", {})
    calib: Dict[str, Any] = data.get("calibration", {})
    targets: Dict[str, Any] = data.get("targets", {})
    hw = data.get("hw", {})
    env = data.get("env", {})
    confirm2 = data.get("confirm2")

    thumb_paths = {
        "Before / After": Path(thumbs_dir) / "fig1_before_after.png",
        "ROC + EVT": Path(thumbs_dir) / "fig2_roc_calibrated.png",
        "Confirm-2 sweep": Path(thumbs_dir) / "fig3_confirm2_curve.png",
        "Latency trade-off": Path(thumbs_dir) / "fig4_latency_angle_trade.png",
    }
    thumbs = {name: _maybe_read(path) for name, path in thumb_paths.items()}

    with PdfPages(pdf_path) as pdf:
        fig = plt.figure(figsize=(11.69, 8.27))  # A4 landscape
        grid = fig.add_gridspec(
            3,
            4,
            height_ratios=[0.6, 1.3, 1.3],
            width_ratios=[1.1, 1.1, 1.0, 1.0],
        )

        # Header
        ax_header = fig.add_subplot(grid[0, :])
        ax_header.axis("off")
        run_id = data.get("run_id", "(unknown)")
        created = data.get("created_at", "(unknown)")
        header_txt = f"{title} — run {run_id}"
        ax_header.text(0.01, 0.75, header_txt, fontsize=16, fontweight="bold", va="center")
        ax_header.text(0.01, 0.40, f"Created: {created}", fontsize=9)
        ax_header.text(
            0.01,
            0.18,
            (
                f"Targets ▸ ΔSNR ≥ {targets.get('delta_pdsnrdB_min', '–')} dB, "
                f"ΔTPR@FPR={targets.get('fpr_target', '–')} ≥ "
                f"{targets.get('delta_tpr_at_fpr_min', '–')}"
            ),
            fontsize=9,
        )
        overall = bool(data.get("overall_pass", False))
        ax_header.text(
            0.99,
            0.5,
            "PASS" if overall else "FAIL",
            ha="right",
            va="center",
            fontsize=30,
            fontweight="bold",
            color="tab:green" if overall else "tab:red",
        )

        # Performance table
        perf_rows: MetricRows = [
            ("Baseline PD‑SNR (dB)", _fmt_num(perf.get("pd_snr_baseline_db"), 2)),
            ("STAP PD‑SNR (dB)", _fmt_num(perf.get("pd_snr_stap_db"), 2)),
            ("Δ PD‑SNR (dB)", _fmt_num(perf.get("pd_snr_delta_db"), 2)),
            (
                f"STAP TPR @ FPR={targets.get('fpr_target', '–')}",
                _fmt_num(perf.get("tpr_at_fpr_stap"), 4),
            ),
            ("Δ TPR @ FPR", _fmt_num(perf.get("tpr_at_fpr_delta"), 4)),
            ("Partial AUC (≤ target)", _fmt_num(perf.get("pauc_stap"), 6)),
        ]
        ax_perf = fig.add_subplot(grid[1, 0:2])
        _add_table(ax_perf, perf_rows, headers=("Metric", "Value"))

        # Calibration table (per-look conformal)
        cal_rows: MetricRows = [
            ("α target (per-look)", _fmt_pct(calib.get("alpha_target"))),
            ("Empirical Pfa", _fmt_pct(calib.get("empirical_pfa"))),
            (
                "95% CI (lo, hi)",
                f"{_fmt_pct(calib.get('pfa_ci_lo'), 3)}, {_fmt_pct(calib.get('pfa_ci_hi'), 3)}",
            ),
            ("n_null", str(calib.get("n_null", "—"))),
            ("False alarms", str(calib.get("k_false_alarms", "—"))),
        ]
        ax_cal = fig.add_subplot(grid[1, 2])
        _add_table(ax_cal, cal_rows, headers=("Calibration", "Value"))

        # Confirm-2 overlay
        ax_c2 = fig.add_subplot(grid[1, 3])
        ax_c2.axis("off")
        if confirm2:
            pair_ci = confirm2.get("pair_ci") or [None, None]
            rho_ci = confirm2.get("rho_ci") or [None, None]
            c_rows: MetricRows = [
                ("Target α₂", _fmt_pct(confirm2.get("alpha2_target"))),
                (
                    "Empirical pair‑Pfa",
                    _fmt_pct(confirm2.get("empirical_pair_pfa")),
                ),
                (
                    "Predicted pair‑Pfa",
                    _fmt_pct(confirm2.get("predicted_pair_pfa")),
                ),
                (
                    "Pair CI (lo, hi)",
                    f"{_fmt_pct(pair_ci[0])}, {_fmt_pct(pair_ci[1])}",
                ),
                (
                    "ρ̂ (lo, hi)",
                    f"{_fmt_num(confirm2.get('rho_hat'), 3)} "
                    f"({', '.join(_fmt_num(x, 3) for x in rho_ci)})",
                ),
                ("Test pairs", str(confirm2.get("n_pairs_test", "—"))),
            ]
            _add_table(ax_c2, c_rows, headers=("Confirm‑2", "Value"))
        else:
            ax_c2.text(
                0.5,
                0.5,
                "Confirm‑2 overlay not enabled",
                ha="center",
                va="center",
                fontsize=9,
            )

        # Gates summary
        ax_gate = fig.add_subplot(grid[2, 0])
        gate_rows: MetricRows = [
            ("Δ PD‑SNR ≥ target", _gate_str(bool(gates.get("gate_delta_pd_snr")))),
            ("Δ TPR@FPR ≥ target", _gate_str(bool(gates.get("gate_delta_tpr_at_fpr")))),
            ("Calibration CI contains α", _gate_str(bool(gates.get("gate_calibration_ci")))),
        ]
        _add_table(ax_gate, gate_rows, headers=("Gate", "Status"))

        # Environment
        ax_env = fig.add_subplot(grid[2, 1])
        ax_env.axis("off")
        env_lines = [
            f"GPU: {hw.get('gpu_name', '(none)')}",
            f"CUDA: {hw.get('cuda', '–')} · torch: {hw.get('torch', '–')}",
            f"GPU count: {hw.get('gpu_count', '–')}",
            f"Python: {env.get('python', '–')}",
            f"Platform: {env.get('platform', '–')}",
        ]
        ax_env.text(0.0, 1.0, "Environment", fontsize=11, fontweight="bold", va="top")
        ax_env.text(0.0, 0.70, "\n".join(env_lines), fontsize=9, va="top")

        # Thumbnails
        thumb_axes = [
            fig.add_subplot(grid[2, 2]),
            fig.add_subplot(grid[2, 3]),
        ]
        thumb_items = list(thumbs.items())
        for ax, (name, image) in zip(thumb_axes, thumb_items[:2], strict=False):
            ax.set_title(name, fontsize=10)
            ax.axis("off")
            if image is not None:
                ax.imshow(image)
            else:
                ax.text(0.5, 0.5, "Figure not found", ha="center", va="center")

        fig.tight_layout()
        pdf.savefig(fig)
        plt.close(fig)
