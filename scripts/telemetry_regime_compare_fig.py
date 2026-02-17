#!/usr/bin/env python3
"""
Compare contract/telemetry regimes across simulation and real data.

Motivation
----------
"Realism" is not just having a detailed simulator; it's matching measurable
telemetry distributions. This script generates an overlay-histogram figure
comparing a small set of contract-v2 telemetry scalars across:
  - Brain-* k-Wave simulations (sampled replay bundles),
  - Shin RatBrain Fig3 IQ (all-clips telemetry-only sweep bundles),
  - Macé/Urban PD-only (PD-only contract sweep CSV).

We do *not* claim the simulator is "real"; the intended use is to anchor the
simulation to the telemetry regimes we actually see and to be explicit about
mismatches (which strengthens the "contract keeps KA off" story).

Outputs
-------
  - A PNG figure (default: figs/paper/telemetry_regime_compare.png).
  - An optional JSON summary with median/IQR per group and metric.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Tuple

import numpy as np


@dataclass(frozen=True)
class GroupSpec:
    label: str
    kind: str  # "meta_glob" or "csv"
    source: str


def _parse_groups(specs: List[str]) -> List[GroupSpec]:
    groups: List[GroupSpec] = []
    for raw in specs:
        s = str(raw).strip()
        if not s:
            continue
        # Format: "Label=meta:<glob>" or "Label=csv:<path>"
        if "=" not in s or ":" not in s:
            raise ValueError(
                f"Invalid --group '{raw}'. Expected 'Label=meta:<glob>' or 'Label=csv:<path>'."
            )
        label, rest = s.split("=", 1)
        label = label.strip()
        kind, src = rest.split(":", 1)
        kind = kind.strip().lower()
        src = src.strip()
        if kind not in {"meta", "csv"}:
            raise ValueError(f"Invalid group kind '{kind}' in '{raw}' (expected meta or csv).")
        if not label:
            raise ValueError(f"Empty label in group spec '{raw}'.")
        if not src:
            raise ValueError(f"Empty source in group spec '{raw}'.")
        groups.append(GroupSpec(label=label, kind="meta_glob" if kind == "meta" else "csv", source=src))
    if not groups:
        raise ValueError("No groups provided.")
    return groups


def _find_meta_paths(glob_pat: str) -> List[Path]:
    # Accept either an explicit file or a glob.
    p = Path(glob_pat)
    if p.is_file() and p.name == "meta.json":
        return [p]
    # Use pathlib glob relative to repo root (cwd).
    paths = sorted(Path(".").glob(glob_pat))
    # Keep only meta.json files.
    return [Path(x) for x in paths if Path(x).is_file() and Path(x).name == "meta.json"]


def _extract_contract_metrics_from_meta(meta_path: Path) -> Dict[str, float] | None:
    try:
        meta = json.loads(meta_path.read_text())
    except Exception:
        return None
    rep = meta.get("ka_contract_v2")
    if not isinstance(rep, dict):
        return None
    metrics = rep.get("metrics") or {}
    if not isinstance(metrics, dict):
        return None

    def get_float(key: str) -> float | None:
        v = metrics.get(key)
        try:
            fv = float(v)
        except Exception:
            return None
        return fv if np.isfinite(fv) else None

    out: Dict[str, float] = {}
    for key in ("pf_peak_flow", "guard_q90", "iqr_alias_bg"):
        fv = get_float(key)
        if fv is None:
            return None
        out[key] = float(fv)
    return out


def _load_group_from_meta_glob(label: str, glob_pat: str) -> Dict[str, np.ndarray]:
    paths = _find_meta_paths(glob_pat)
    if not paths:
        raise FileNotFoundError(f"[{label}] No meta.json files found for glob: {glob_pat}")

    rows: List[Dict[str, float]] = []
    for mp in paths:
        r = _extract_contract_metrics_from_meta(mp)
        if r is None:
            continue
        rows.append(r)
    if not rows:
        raise RuntimeError(f"[{label}] Found meta.json files but none contained ka_contract_v2 metrics.")

    out: Dict[str, np.ndarray] = {}
    for key in ("pf_peak_flow", "guard_q90", "iqr_alias_bg"):
        out[key] = np.asarray([r[key] for r in rows], dtype=np.float64)
    return out


def _load_group_from_mace_csv(label: str, csv_path: str) -> Dict[str, np.ndarray]:
    try:
        import pandas as pd
    except Exception as exc:  # pragma: no cover
        raise RuntimeError("pandas is required to load Macé CSV inputs") from exc

    path = Path(csv_path)
    if not path.is_file():
        raise FileNotFoundError(f"[{label}] CSV not found: {path}")
    df = pd.read_csv(path)
    if df.empty:
        raise RuntimeError(f"[{label}] Empty CSV: {path}")

    col_map = {
        "pf_peak_flow": "ka_contract_v2_pf_peak_flow",
        "guard_q90": "ka_contract_v2_guard_q90",
        "iqr_alias_bg": "ka_contract_v2_iqr_alias_bg",
    }
    out: Dict[str, np.ndarray] = {}
    for key, col in col_map.items():
        if col not in df.columns:
            raise KeyError(f"[{label}] Missing column '{col}' in {path}")
        s = pd.to_numeric(df[col], errors="coerce").to_numpy(dtype=np.float64, copy=False)
        s = s[np.isfinite(s)]
        if s.size == 0:
            raise RuntimeError(f"[{label}] No finite values for '{col}' in {path}")
        out[key] = s
    return out


def _summarize(x: np.ndarray) -> Dict[str, float]:
    a = np.asarray(x, dtype=np.float64).ravel()
    a = a[np.isfinite(a)]
    if a.size == 0:
        return {"n": 0, "median": float("nan"), "q25": float("nan"), "q75": float("nan")}
    return {
        "n": int(a.size),
        "median": float(np.quantile(a, 0.5)),
        "q25": float(np.quantile(a, 0.25)),
        "q75": float(np.quantile(a, 0.75)),
    }


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Compare telemetry regimes across sim and real datasets.")
    ap.add_argument(
        "--group",
        action="append",
        default=[],
        help=(
            "Add a group. Format: 'Label=meta:<glob>' (reads ka_contract_v2 from meta.json) "
            "or 'Label=csv:<path>' (reads Macé PD-only contract sweep CSV). Can be repeated."
        ),
    )
    ap.add_argument(
        "--out-png",
        type=Path,
        default=Path("figs/paper/telemetry_regime_compare.png"),
        help="Output PNG path.",
    )
    ap.add_argument("--also-pdf", action="store_true", help="Also write a matching PDF next to the PNG.")
    ap.add_argument(
        "--out-json",
        type=Path,
        default=None,
        help="Optional output JSON summary path (median/IQR per group/metric).",
    )
    ap.add_argument(
        "--bins",
        type=int,
        default=30,
        help="Number of histogram bins per metric.",
    )
    return ap.parse_args()


def main() -> None:
    args = parse_args()

    if not args.group:
        # Sensible defaults for the paper repo layout.
        args.group = [
            "Brain-* (sim)=meta:runs/telemetry_regime_compare/brain_*/**/meta.json",
            "Shin Fig3 (IQ)=meta:runs/shin_ratbrain_shinU_e970_telemetry_only_0_128_all80/*/meta.json",
            "Macé (PD-only)=csv:reports/mace_pdonly_contract_v2.csv",
        ]

    groups = _parse_groups(list(args.group))

    values: Dict[str, Dict[str, np.ndarray]] = {}
    for g in groups:
        if g.kind == "meta_glob":
            values[g.label] = _load_group_from_meta_glob(g.label, g.source)
        else:
            values[g.label] = _load_group_from_mace_csv(g.label, g.source)

    # Deferred matplotlib import.
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams.update(
        {
            "figure.dpi": 150,
            "font.size": 9,
            "axes.titlesize": 10,
            "axes.labelsize": 9,
            "legend.fontsize": 8,
        }
    )

    metrics = [
        ("pf_peak_flow", r"$\mathrm{PfPeakFrac}(\mathcal{T}_{flow})$"),
        ("guard_q90", r"$Q_{0.90}(r_g)$"),
        ("iqr_alias_bg", r"$\mathrm{IQR}(m_{alias}\mid\mathcal{T}_{bg})$"),
    ]

    fig, axes = plt.subplots(1, len(metrics), figsize=(13.2, 3.6))
    if len(metrics) == 1:
        axes = [axes]

    # Stable group order, stable colors.
    palette = [
        "#4c78a8",
        "#f58518",
        "#54a24b",
        "#e45756",
        "#72b7b2",
        "#b279a2",
    ]
    labels = [g.label for g in groups]
    color_for = {lab: palette[i % len(palette)] for i, lab in enumerate(labels)}

    summary: Dict[str, Any] = {"groups": {}}
    for key, title in metrics:
        # Compute global finite range for bins.
        all_vals: List[np.ndarray] = []
        for lab in labels:
            all_vals.append(np.asarray(values[lab][key], dtype=np.float64))
        concat = np.concatenate([a[np.isfinite(a)] for a in all_vals], axis=0)
        if concat.size == 0:
            continue
        lo = float(np.quantile(concat, 0.01))
        hi = float(np.quantile(concat, 0.99))
        if not np.isfinite(lo) or not np.isfinite(hi) or lo >= hi:
            lo = float(np.min(concat))
            hi = float(np.max(concat))
        bins = np.linspace(lo, hi, max(5, int(args.bins)))

        ax = axes[metrics.index((key, title))]
        ax.set_title(title)
        # Keep axes reader-friendly; the panel title names the metric.
        ax.set_xlabel("")
        ax.set_ylabel("Density")
        ax.grid(True, alpha=0.20)

        for lab in labels:
            x = np.asarray(values[lab][key], dtype=np.float64)
            x = x[np.isfinite(x)]
            if x.size == 0:
                continue
            ax.hist(
                x,
                bins=bins,
                density=True,
                histtype="step",
                linewidth=1.8,
                alpha=0.95,
                color=color_for[lab],
                label=lab,
            )
            s = _summarize(x)
            summary["groups"].setdefault(lab, {})[key] = s

        if key == "pf_peak_flow":
            ax.set_xlim(0.0, 1.0)

    # Put one shared legend.
    handles, leg_labels = axes[0].get_legend_handles_labels()
    if handles:
        fig.legend(handles, leg_labels, loc="upper center", ncol=min(3, len(leg_labels)), frameon=False)

    fig.tight_layout(rect=[0, 0, 1, 0.92])

    out_png = Path(args.out_png)
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, dpi=200)
    if bool(args.also_pdf):
        fig.savefig(out_png.with_suffix(".pdf"))

    if args.out_json is not None:
        out_json = Path(args.out_json)
        out_json.parent.mkdir(parents=True, exist_ok=True)
        out_json.write_text(json.dumps(summary, indent=2, sort_keys=True))

    print(f"[telemetry-compare] wrote {out_png}")


if __name__ == "__main__":
    main()
