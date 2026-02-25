from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

import numpy as np


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


def _load_npy(bundle_dir: Path, name: str) -> np.ndarray | None:
    path = bundle_dir / f"{name}.npy"
    if not path.is_file():
        return None
    return np.load(path, allow_pickle=False)


def _finite_1d(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=np.float64).ravel()
    return x[np.isfinite(x)]


def _right_tail_threshold(bg_scores: np.ndarray, alpha: float) -> tuple[float | None, float | None]:
    """
    Choose a threshold tau for right-tail scoring (higher = more positive) so that
    mean(bg >= tau) is approximately alpha on the provided sample.

    Returns (tau, realized_fpr). Uses a discrete rank rule (no interpolation).
    """
    bg = _finite_1d(bg_scores)
    n = int(bg.size)
    if n <= 0:
        return None, None
    a = float(alpha)
    if not np.isfinite(a) or a <= 0.0:
        tau = float("inf")
        return tau, 0.0
    if a >= 1.0:
        tau = float(np.min(bg))
        return tau, 1.0

    k = int(np.ceil(a * n))
    k = max(1, min(k, n))
    # tau = k-th largest value (partition on (n-k)-th smallest).
    tau = float(np.partition(bg, n - k)[n - k])
    realized = float(np.mean(bg >= tau))
    return tau, realized


def _parse_indices(spec: str, n_max: int) -> list[int]:
    """
    Parse either:
      - comma list: "0,1,2"
      - slice: "0:10" or "0:10:2"
      - single int: "5"
    """
    spec = (spec or "").strip()
    if not spec:
        return list(range(n_max))
    if "," in spec:
        out: list[int] = []
        for part in spec.split(","):
            part = part.strip()
            if part:
                out.append(int(part))
        return out
    if ":" in spec:
        parts = [p.strip() for p in spec.split(":")]
        if len(parts) not in (2, 3):
            raise ValueError(f"Invalid slice spec: {spec!r}")
        start = int(parts[0]) if parts[0] else 0
        stop = int(parts[1]) if parts[1] else n_max
        step = int(parts[2]) if len(parts) == 3 and parts[2] else 1
        return list(range(start, min(stop, n_max), step))
    return [int(spec)]


def _amp_from_meta(meta: dict[str, Any]) -> float | None:
    wem = meta.get("within_ensemble_motion")
    if not isinstance(wem, dict):
        return None
    amp = wem.get("amp_px")
    try:
        out = float(amp)
    except Exception:
        return None
    return out if np.isfinite(out) else None


def _meta_digest(meta: dict[str, Any]) -> dict[str, Any]:
    """Keep summary JSON small and audit-friendly."""
    out: dict[str, Any] = {}
    for k in ("prf_hz", "Lt", "tile_hw", "tile_stride", "seed", "total_frames"):
        if k in meta:
            out[k] = meta.get(k)
    dataset = meta.get("dataset")
    if isinstance(dataset, dict):
        out["dataset"] = {k: dataset.get(k) for k in ("name", "format") if k in dataset}
    # Selected provenance fields written by our bundle drivers.
    for k in ("twinkling_rawbcf", "twinkling_eval_masks", "within_ensemble_motion"):
        v = meta.get(k)
        if v is not None:
            out[k] = v
    baseline = meta.get("baseline_stats")
    if isinstance(baseline, dict):
        out["baseline_stats"] = {
            k: baseline.get(k)
            for k in ("baseline_type", "svd_keep_min", "svd_keep_max", "svd_rank_kept", "svd_energy_kept_frac")
            if k in baseline
        }
    score_stats = meta.get("score_stats")
    if isinstance(score_stats, dict):
        out["score_stats"] = {"mode": score_stats.get("mode")}
    return out


def _load_motion_amp0_reference(
    *,
    root: Path,
    frame_ids: list[int],
    methods: tuple[str, ...],
    require_amp_px: float | None = None,
) -> dict[str, dict[str, np.ndarray]]:
    """
    Load score arrays from a non-motion Twinkling run and pool them across the
    specified frame indices. Uses B-mode structural masks already present in bundles.
    """
    wanted = set(int(i) for i in frame_ids)
    pooled: dict[str, dict[str, list[np.ndarray]]] = {m: {"bg": [], "flow": []} for m in methods}
    count = 0
    for bundle_dir in _iter_bundle_dirs(root):
        meta = _load_meta(bundle_dir)
        if meta is None:
            continue
        if require_amp_px is not None:
            amp = _amp_from_meta(meta)
            if amp is None or float(amp) != float(require_amp_px):
                continue
        tw = meta.get("twinkling_rawbcf")
        if not isinstance(tw, dict):
            continue
        idx = tw.get("frame_idx")
        try:
            frame_idx = int(idx)
        except Exception:
            continue
        if frame_idx not in wanted:
            continue

        mask_flow = _load_npy(bundle_dir, "mask_flow")
        if mask_flow is None:
            continue
        mask_bg = _load_npy(bundle_dir, "mask_bg")
        if mask_bg is None:
            mask_bg = ~np.asarray(mask_flow, dtype=bool)
        flow = np.asarray(mask_flow, dtype=bool)
        bg = np.asarray(mask_bg, dtype=bool) & (~flow)

        # Required scores.
        score_base = _load_npy(bundle_dir, "score_base")
        score_preka = _load_npy(bundle_dir, "score_stap_preka")
        score_post = _load_npy(bundle_dir, "score_stap")
        if score_base is None or score_preka is None:
            continue
        if score_post is None:
            score_post = score_preka

        score_base_pdlog = _load_npy(bundle_dir, "score_base_pdlog")
        score_base_kasai = _load_npy(bundle_dir, "score_base_kasai")

        score_map: dict[str, np.ndarray] = {
            "base": score_base,
            "stap_preka": score_preka,
            "stap": score_post,
        }
        if score_base_pdlog is not None:
            score_map["base_pdlog"] = score_base_pdlog
        if score_base_kasai is not None:
            score_map["base_kasai"] = score_base_kasai

        for m in methods:
            s = score_map.get(m)
            if s is None:
                continue
            s = np.asarray(s, dtype=np.float64)
            pooled[m]["bg"].append(s[bg])
            pooled[m]["flow"].append(s[flow])
        count += 1

    out: dict[str, dict[str, np.ndarray]] = {}
    for m in methods:
        if not pooled[m]["bg"]:
            continue
        out[m] = {
            "bg": _finite_1d(np.concatenate(pooled[m]["bg"], axis=0)),
            "flow": _finite_1d(np.concatenate(pooled[m]["flow"], axis=0)),
        }
    out["_meta"] = {"count_bundles": np.array([count], dtype=np.int64)}  # type: ignore[assignment]
    return out


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Aggregate Twinkling/Gammex within-ensemble motion ladder results.\n"
            "Groups bundles by meta.within_ensemble_motion.amp_px and reports pooled structural ROC "
            "using B-mode-only masks (mask_flow/mask_bg)."
        )
    )
    parser.add_argument("--root", type=Path, required=True, help="Root directory containing motion-ladder bundles.")
    parser.add_argument("--out-csv", type=Path, required=True)
    parser.add_argument("--out-json", type=Path, required=True)
    parser.add_argument("--out-png", type=Path, default=None, help="Optional plot output (PNG).")
    parser.add_argument(
        "--bootstrap",
        type=int,
        default=0,
        help="Bootstrap replicates across frames for TPR CIs (default: 0, disabled).",
    )
    parser.add_argument(
        "--bootstrap-seed",
        type=int,
        default=0,
        help="Bootstrap RNG seed (default: 0).",
    )
    parser.add_argument(
        "--amp0-ref-root",
        type=Path,
        default=None,
        help=(
            "Optional non-motion bundle root used as an amp=0 equivalence check "
            "(e.g., runs/real/twinkling_gammex_alonglinear17_prf2500_str6_msd_ka)."
        ),
    )
    parser.add_argument(
        "--amp0-ref-frames",
        type=str,
        default="",
        help="Frame indices (slice/list) to select from --amp0-ref-root (default: all).",
    )
    parser.add_argument(
        "--amp0-ref-tol",
        type=float,
        default=5e-3,
        help="Abs tolerance on amp=0 TPR match vs ref (default: 5e-3).",
    )
    parser.add_argument(
        "--fprs",
        type=float,
        nargs="+",
        default=[1e-4, 3e-4, 1e-3],
        help="FPR targets for ROC points (default: %(default)s).",
    )
    args = parser.parse_args()
    bootstrap_n = int(max(0, args.bootstrap))
    bootstrap_seed = int(args.bootstrap_seed)

    bundle_dirs = _iter_bundle_dirs(args.root)
    if not bundle_dirs:
        raise SystemExit(f"No bundles found under: {args.root}")

    # group[amp][method]['bg'/'flow'] -> list[np.ndarray]
    group: dict[float, dict[str, dict[str, list[np.ndarray]]]] = {}
    group_counts: dict[float, int] = {}
    meta_example: dict[float, dict[str, Any]] = {}

    methods = ("base", "base_pdlog", "base_kasai", "stap_preka", "stap")
    for bundle_dir in bundle_dirs:
        meta = _load_meta(bundle_dir)
        if meta is None:
            continue
        amp = _amp_from_meta(meta)
        if amp is None:
            continue

        mask_flow = _load_npy(bundle_dir, "mask_flow")
        if mask_flow is None:
            continue
        mask_bg = _load_npy(bundle_dir, "mask_bg")
        if mask_bg is None:
            mask_bg = ~np.asarray(mask_flow, dtype=bool)

        flow = np.asarray(mask_flow, dtype=bool)
        bg = np.asarray(mask_bg, dtype=bool) & (~flow)

        score_base = _load_npy(bundle_dir, "score_base")
        score_base_pdlog = _load_npy(bundle_dir, "score_base_pdlog")
        score_base_kasai = _load_npy(bundle_dir, "score_base_kasai")
        score_preka = _load_npy(bundle_dir, "score_stap_preka")
        score_post = _load_npy(bundle_dir, "score_stap")
        if score_base is None or score_preka is None:
            continue
        if score_post is None:
            score_post = score_preka

        group.setdefault(amp, {m: {"bg": [], "flow": []} for m in methods})
        group_counts[amp] = group_counts.get(amp, 0) + 1
        meta_example.setdefault(amp, _meta_digest(meta))

        method_scores: list[tuple[str, np.ndarray]] = [("base", score_base)]
        if score_base_pdlog is not None:
            method_scores.append(("base_pdlog", score_base_pdlog))
        if score_base_kasai is not None:
            method_scores.append(("base_kasai", score_base_kasai))
        method_scores.extend([("stap_preka", score_preka), ("stap", score_post)])

        for method, scores in method_scores:
            s = np.asarray(scores, dtype=np.float64)
            group[amp][method]["bg"].append(s[bg])
            group[amp][method]["flow"].append(s[flow])

    amps = sorted(group.keys())
    if not amps:
        raise SystemExit("No bundles contained within_ensemble_motion.amp_px + required arrays.")

    rows: list[dict[str, Any]] = []
    summary: dict[str, Any] = {
        "root": str(args.root),
        "group_key": "within_ensemble_motion.amp_px",
        "amps": amps,
        "fprs": list(map(float, args.fprs)),
        "methods": list(methods),
        "groups": {},
    }

    for amp in amps:
        grp = group[amp]
        grp_out: dict[str, Any] = {
            "amp_px": float(amp),
            "bundle_count": int(group_counts.get(amp, 0)),
            "methods": {},
            "meta_example": meta_example.get(amp, {}),
        }
        for method in methods:
            bg_all = _finite_1d(np.concatenate(grp[method]["bg"], axis=0)) if grp[method]["bg"] else np.array([])
            flow_all = (
                _finite_1d(np.concatenate(grp[method]["flow"], axis=0)) if grp[method]["flow"] else np.array([])
            )
            n_bg = int(bg_all.size)
            n_flow = int(flow_all.size)
            fpr_min = (1.0 / float(n_bg)) if n_bg > 0 else None
            roc_pts: list[dict[str, Any]] = []
            for alpha in args.fprs:
                tau, fpr_real = _right_tail_threshold(bg_all, float(alpha))
                if tau is None or fpr_real is None:
                    continue
                tpr = float(np.mean(flow_all >= float(tau))) if n_flow > 0 else None
                supported = bool(fpr_min is not None and float(alpha) >= float(fpr_min))
                pt: dict[str, Any] = {
                    "fpr_target": float(alpha),
                    "supported": supported,
                    "fpr_min": fpr_min,
                    "thr": float(tau),
                    "fpr_realized": float(fpr_real),
                    "tpr": tpr,
                }
                roc_pts.append(pt)
                rows.append(
                    {
                        "amp_px": float(amp),
                        "method": method,
                        "fpr_target": float(alpha),
                        "supported": supported,
                        "fpr_min": fpr_min,
                        "thr": float(tau),
                        "fpr_realized": float(fpr_real),
                        "tpr": tpr,
                        "n_bg": n_bg,
                        "n_flow": n_flow,
                        "bundle_count": int(group_counts.get(amp, 0)),
                    }
                )
            grp_out["methods"][method] = {"n_bg": n_bg, "n_flow": n_flow, "fpr_min": fpr_min, "roc": roc_pts}

        # Bootstrap CIs (frame-level) for the headline comparison: base vs STAP_preka.
        if bootstrap_n > 0 and grp_out["bundle_count"] >= 2:
            rng = np.random.default_rng(int(bootstrap_seed) + int(round(1000.0 * float(amp))))
            # Per-bundle score vectors.
            base_bg = grp["base"]["bg"]
            base_flow = grp["base"]["flow"]
            stap_bg = grp["stap_preka"]["bg"]
            stap_flow = grp["stap_preka"]["flow"]
            n_bundles = int(grp_out["bundle_count"])

            ci: dict[str, Any] = {"bootstrap": bootstrap_n, "seed": int(bootstrap_seed), "fprs": []}
            for alpha in args.fprs:
                alpha_f = float(alpha)
                tpr_base_bs: list[float] = []
                tpr_stap_bs: list[float] = []
                for _ in range(bootstrap_n):
                    idxs = rng.integers(0, n_bundles, size=n_bundles)
                    bg_b = _finite_1d(np.concatenate([base_bg[i] for i in idxs], axis=0))
                    fl_b = _finite_1d(np.concatenate([base_flow[i] for i in idxs], axis=0))
                    bg_s = _finite_1d(np.concatenate([stap_bg[i] for i in idxs], axis=0))
                    fl_s = _finite_1d(np.concatenate([stap_flow[i] for i in idxs], axis=0))
                    tau_b, _ = _right_tail_threshold(bg_b, alpha_f)
                    tau_s, _ = _right_tail_threshold(bg_s, alpha_f)
                    if tau_b is None or tau_s is None or fl_b.size == 0 or fl_s.size == 0:
                        continue
                    tpr_base_bs.append(float(np.mean(fl_b >= float(tau_b))))
                    tpr_stap_bs.append(float(np.mean(fl_s >= float(tau_s))))
                if not tpr_base_bs or not tpr_stap_bs:
                    continue
                tb = np.asarray(tpr_base_bs, dtype=np.float64)
                ts = np.asarray(tpr_stap_bs, dtype=np.float64)
                deltas = ts - tb
                ci["fprs"].append(
                    {
                        "fpr_target": alpha_f,
                        "base_tpr_median": float(np.median(tb)),
                        "base_tpr_p025": float(np.quantile(tb, 0.025)),
                        "base_tpr_p975": float(np.quantile(tb, 0.975)),
                        "stap_tpr_median": float(np.median(ts)),
                        "stap_tpr_p025": float(np.quantile(ts, 0.025)),
                        "stap_tpr_p975": float(np.quantile(ts, 0.975)),
                        "delta_tpr_median": float(np.median(deltas)),
                        "delta_tpr_p025": float(np.quantile(deltas, 0.025)),
                        "delta_tpr_p975": float(np.quantile(deltas, 0.975)),
                        "bootstrap_effective": int(min(tb.size, ts.size)),
                    }
                )
            grp_out["bootstrap_ci"] = ci

        # Convenience deltas (stap_preka - base) at requested fprs.
        deltas: list[dict[str, Any]] = []
        base_pts = {p["fpr_target"]: p for p in grp_out["methods"]["base"]["roc"]}
        stap_pts = {p["fpr_target"]: p for p in grp_out["methods"]["stap_preka"]["roc"]}
        for alpha in args.fprs:
            a = float(alpha)
            b = base_pts.get(a)
            s = stap_pts.get(a)
            if not b or not s:
                continue
            try:
                db = float(s["tpr"]) - float(b["tpr"])  # type: ignore[arg-type]
            except Exception:
                db = None
            deltas.append({"fpr_target": a, "delta_tpr_stap_preka_minus_base": db})
        grp_out["deltas"] = deltas
        summary["groups"][str(amp)] = grp_out

    # Optional amp=0 equivalence check vs a non-motion reference run.
    if args.amp0_ref_root is not None:
        # Build a subset equivalence check: compare motion amp=0 vs a separate
        # non-motion reference run on the same frame IDs.
        frame_ids: list[int]
        if str(args.amp0_ref_frames).strip():
            frame_ids = _parse_indices(str(args.amp0_ref_frames), 10_000)
        else:
            # Default to all amp=0 frames present in the motion root.
            frame_ids = []
            for bd in _iter_bundle_dirs(args.root):
                meta = _load_meta(bd)
                if meta is None:
                    continue
                amp_v = _amp_from_meta(meta)
                if amp_v is None or float(amp_v) != 0.0:
                    continue
                tw = meta.get("twinkling_rawbcf")
                if isinstance(tw, dict) and "frame_idx" in tw:
                    try:
                        frame_ids.append(int(tw["frame_idx"]))
                    except Exception:
                        pass
            frame_ids = sorted(set(frame_ids))

        ref_methods = ("base", "stap_preka")
        mot = _load_motion_amp0_reference(
            root=args.root, frame_ids=frame_ids, methods=ref_methods, require_amp_px=0.0
        )
        ref = _load_motion_amp0_reference(
            root=args.amp0_ref_root, frame_ids=frame_ids, methods=ref_methods, require_amp_px=None
        )
        mot_count = int(mot.get("_meta", {}).get("count_bundles", np.array([0]))[0])  # type: ignore[index]
        ref_count = int(ref.get("_meta", {}).get("count_bundles", np.array([0]))[0])  # type: ignore[index]

        check: dict[str, Any] = {
            "ref_root": str(args.amp0_ref_root),
            "ref_bundle_count": ref_count,
            "motion_bundle_count": mot_count,
            "frame_ids": frame_ids,
            "tol_abs_tpr": float(args.amp0_ref_tol),
            "comparisons": [],
        }
        for alpha in args.fprs:
            alpha_f = float(alpha)
            out_row: dict[str, Any] = {"fpr_target": alpha_f}
            ok = True
            for m in ref_methods:
                bg_m = mot.get(m, {}).get("bg")
                fl_m = mot.get(m, {}).get("flow")
                bg_r = ref.get(m, {}).get("bg")
                fl_r = ref.get(m, {}).get("flow")
                if bg_m is None or fl_m is None or bg_r is None or fl_r is None:
                    continue
                tau_m, _ = _right_tail_threshold(np.asarray(bg_m, dtype=np.float64), alpha_f)
                tau_r, _ = _right_tail_threshold(np.asarray(bg_r, dtype=np.float64), alpha_f)
                if tau_m is None or tau_r is None:
                    continue
                tpr_m = float(np.mean(np.asarray(fl_m, dtype=np.float64) >= float(tau_m))) if np.asarray(fl_m).size else None
                tpr_r = float(np.mean(np.asarray(fl_r, dtype=np.float64) >= float(tau_r))) if np.asarray(fl_r).size else None
                out_row[f"{m}_tpr_motion_subset"] = tpr_m
                out_row[f"{m}_tpr_ref_subset"] = tpr_r
                if tpr_m is None or tpr_r is None:
                    continue
                diff = float(abs(float(tpr_m) - float(tpr_r)))
                out_row[f"{m}_abs_diff"] = diff
                if diff > float(args.amp0_ref_tol):
                    ok = False
            out_row["pass"] = ok
            check["comparisons"].append(out_row)
        summary["amp0_equivalence_check"] = check

    # Write CSV/JSON
    args.out_csv.parent.mkdir(parents=True, exist_ok=True)
    if rows:
        fieldnames = [
            "amp_px",
            "method",
            "fpr_target",
            "supported",
            "fpr_min",
            "thr",
            "fpr_realized",
            "tpr",
            "n_bg",
            "n_flow",
            "bundle_count",
        ]
        with args.out_csv.open("w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=fieldnames)
            w.writeheader()
            w.writerows(rows)

    args.out_json.parent.mkdir(parents=True, exist_ok=True)
    args.out_json.write_text(json.dumps(summary, indent=2))

    # Optional plot.
    if args.out_png is not None:
        try:
            import matplotlib.pyplot as plt  # type: ignore

            fprs = list(map(float, args.fprs))
            amps_arr = np.array(amps, dtype=np.float64)
            ncols = len(fprs)
            fig, axes = plt.subplots(1, ncols, figsize=(4.0 * ncols, 3.3), sharey=False)
            if ncols == 1:
                axes = [axes]
            for j, alpha in enumerate(fprs):
                ax = axes[j]
                for method, label in (("base", "Baseline"), ("stap_preka", "STAP")):
                    tprs: list[float] = []
                    yerr_lo: list[float] = []
                    yerr_hi: list[float] = []
                    for amp in amps:
                        pts = summary["groups"][str(amp)]["methods"][method]["roc"]
                        pt = next((p for p in pts if float(p["fpr_target"]) == float(alpha)), None)
                        y = float(pt["tpr"]) if pt and pt["tpr"] is not None else float("nan")
                        tprs.append(y)

                        # Optional bootstrap CI (computed only for base vs stap_preka).
                        ci_entry = None
                        ci = summary["groups"][str(amp)].get("bootstrap_ci")
                        if isinstance(ci, dict):
                            for e in ci.get("fprs", []) or []:
                                if float(e.get("fpr_target", float("nan"))) == float(alpha):
                                    ci_entry = e
                                    break
                        if ci_entry is None or not np.isfinite(y):
                            yerr_lo.append(0.0)
                            yerr_hi.append(0.0)
                            continue

                        if method == "base":
                            lo = float(ci_entry.get("base_tpr_p025", y))
                            hi = float(ci_entry.get("base_tpr_p975", y))
                        else:
                            lo = float(ci_entry.get("stap_tpr_p025", y))
                            hi = float(ci_entry.get("stap_tpr_p975", y))
                        yerr_lo.append(max(0.0, y - lo))
                        yerr_hi.append(max(0.0, hi - y))

                    y = np.array(tprs, dtype=np.float64)
                    if bootstrap_n > 0:
                        yerr = np.vstack([np.array(yerr_lo, dtype=np.float64), np.array(yerr_hi, dtype=np.float64)])
                        ax.errorbar(amps_arr, y, yerr=yerr, marker="o", capsize=2, label=label)
                    else:
                        ax.plot(amps_arr, y, marker="o", label=label)
                ax.set_title(f"TPR @ FPR={alpha:g}")
                ax.set_xlabel("Within-ensemble motion amp (px)")
                ax.set_ylim(-0.02, 1.02)
                ax.grid(True, alpha=0.3)
                if j == 0:
                    ax.set_ylabel("TPR (structural mask)")
                ax.legend(loc="lower left")
            fig.tight_layout()
            args.out_png.parent.mkdir(parents=True, exist_ok=True)
            fig.savefig(args.out_png, dpi=160)
            plt.close(fig)
        except Exception as e:
            print(f"[twinkling_eval_motion_ladder] plot failed: {e}")

    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
