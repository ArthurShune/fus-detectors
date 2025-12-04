# eval/acceptance_cli.py
from __future__ import annotations

import argparse
import json
import os
import pathlib
import sys
import time
from dataclasses import asdict, replace
from datetime import datetime, timezone
from typing import Dict, Optional, Tuple

import numpy as np
from scipy.stats import norm

try:
    import torch
except Exception:
    torch = None

ROOT = pathlib.Path(__file__).resolve().parents[1]
# Ensure project root is importable for CLI execution via `python path/to/script.py`
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

# isort: off
from eval.acceptance import (  # noqa: E402
    AcceptanceTargets,
    DetectorDataset,
    Masks,
    acceptance_report,
)
from eval.summary_pdf import save_summary_pdf  # noqa: E402
from pipeline.sim.synthetic import SimConfig, simulate_scores_and_pd  # noqa: E402
from pipeline.utils.telemetry import sample_gpu_stats, system_telemetry  # noqa: E402
from pipeline.confirm2.validator import calibrate_confirm2, evaluate_confirm2  # noqa: E402
from pipeline.confirm2.policy import rho_inflate_policy  # noqa: E402

# isort: on


def _load_arr(path: str | None) -> np.ndarray | None:
    if not path:
        return None
    p = pathlib.Path(path)
    if not p.exists():
        raise FileNotFoundError(path)
    if p.suffix.lower() == ".npz":
        with np.load(p, allow_pickle=False) as z:
            for k in ["arr", "data", "scores", "pd_map", "mask", "S", "P", "M"]:
                if k in z:
                    return z[k]
            key = list(z.keys())[0]
            return z[key]
    else:
        return np.load(p, allow_pickle=False)


def _simulate_scores_and_pd(
    n_pos: int,
    n_neg: int,
    H: int,
    W: int,
    seed: int = 0,
    sim_config: Optional[SimConfig] = None,
):
    cfg = sim_config or SimConfig(seed=seed)
    if sim_config is None:
        cfg = replace(cfg, seed=seed)
    else:
        cfg = replace(cfg, seed=seed)
    return simulate_scores_and_pd(n_pos, n_neg, H, W, cfg)


def _simulate_confirm2_scores(n_pairs: int, rho: float, seed: int = 0):
    rng = np.random.default_rng(seed)
    cov = np.array([[1.0, rho], [rho, 1.0]], dtype=np.float64)
    L = np.linalg.cholesky(cov)
    z = rng.standard_normal((n_pairs, 2)) @ L.T
    p = 1.0 - norm.cdf(z)
    scores = -np.log(p)
    return scores[:, 0].astype(np.float32), scores[:, 1].astype(np.float32)


def _default_outdir(path: str | None) -> str:
    return path or "runs"


def _parse_steer_grid(spec: str | None) -> Tuple[float, ...]:
    if not spec:
        return (0.0,)
    tokens = []
    for part in spec.replace(";", ",").split(","):
        part = part.strip()
        if part:
            tokens.append(float(part))
    return tuple(tokens) if tokens else (0.0,)


def _summary_stats(arr: Optional[np.ndarray]) -> Optional[Dict[str, float]]:
    if arr is None:
        return None
    flat = np.asarray(arr).astype(np.float64).ravel()
    if flat.size == 0:
        return {"count": 0}
    q5, q50, q95 = np.percentile(flat, [5, 50, 95])
    return {
        "count": int(flat.size),
        "min": float(np.min(flat)),
        "max": float(np.max(flat)),
        "mean": float(np.mean(flat)),
        "std": float(np.std(flat)),
        "p05": float(q5),
        "p50": float(q50),
        "p95": float(q95),
    }


def _mask_stats(mask: Optional[np.ndarray]) -> Optional[Dict[str, float]]:
    if mask is None:
        return None
    arr = np.asarray(mask).astype(bool)
    return {
        "count": int(arr.size),
        "true_count": int(arr.sum()),
        "false_count": int(arr.size - arr.sum()),
        "true_fraction": float(arr.mean()),
    }


def _apply_bundle_defaults(args: argparse.Namespace) -> tuple[Optional[dict], str]:
    """Populate CLI inputs from a bundle directory when --bundle is used."""

    bundle_path = getattr(args, "bundle", None)
    desired_mode = (getattr(args, "score_mode", "auto") or "auto").lower()
    if not bundle_path:
        resolved = "msd" if desired_mode == "auto" else desired_mode
        args.score_mode = resolved
        return None, resolved

    bundle_dir = pathlib.Path(bundle_path)
    meta_path = bundle_dir / "meta.json"
    if not meta_path.exists():
        raise FileNotFoundError(f"{bundle_dir} is missing meta.json; is this a STAP bundle?")
    meta = json.loads(meta_path.read_text())
    default_mode = meta.get("score_pool_default", "msd")
    resolved_mode = default_mode if desired_mode == "auto" else desired_mode
    pool_files = meta.get("score_pool_files") or {}
    if resolved_mode not in pool_files:
        raise ValueError(
            f"Bundle does not provide score pool mode '{resolved_mode}'. "
            f"Available: {sorted(pool_files.keys())}"
        )

    mapping = pool_files[resolved_mode]

    def _assign(attr: str, filename: Optional[str]) -> None:
        if getattr(args, attr) is None:
            if not filename:
                raise ValueError(f"Bundle missing file for {attr} in mode {resolved_mode}")
            setattr(args, attr, str(bundle_dir / filename))

    for attr in ("base_pos", "base_neg", "stap_pos", "stap_neg"):
        _assign(attr, mapping.get(attr))

    bundle_files = meta.get("bundle_files") or {}
    _assign("base_pd", bundle_files.get("pd_base", "pd_base.npy"))
    _assign("stap_pd", bundle_files.get("pd_stap", "pd_stap.npy"))
    _assign("mask_flow", bundle_files.get("mask_flow", "mask_flow.npy"))
    _assign("mask_bg", bundle_files.get("mask_bg", "mask_bg.npy"))

    args.bundle_meta = meta
    args.score_mode = resolved_mode
    return meta, resolved_mode


def run(args: argparse.Namespace) -> tuple[str, str]:
    run_start = time.perf_counter()
    # Resolve target device (optional)
    device = args.device
    if device is not None:
        if device.lower() == "none":
            device = None
        elif torch is None:
            raise RuntimeError("--device specified but torch is not available")
        elif device.startswith("cuda") and not torch.cuda.is_available():
            raise RuntimeError("CUDA requested but not available")

    # Load or simulate inputs
    sim_cfg: Optional[SimConfig] = None
    scenario_motion_um: Optional[float] = None
    scenario_dropout: Optional[float] = None

    bundle_meta = None
    bundle_meta, score_mode_selected = _apply_bundle_defaults(args)

    if args.simulate:
        # small/fast mode for CI if FAST_ACCEPT=1
        fast = os.environ.get("FAST_ACCEPT", "0") == "1"
        npos = 30_000 if fast else 120_000
        nneg = 300_000 if fast else 1_000_000
        steer_cfg = _parse_steer_grid(args.steer_grid)
        sim_cfg = SimConfig(
            motion_amp_um=30.0,
            motion_freq_hz=0.5,
            prf_hz=3000.0,
            prf_jitter_pct=0.0,
            heterogeneity="medium",
            skull_attenuation_db=0.0,
            scatter_density=1.0,
            sensor_dropout=0.05,
            angle_count=9,
            tile=(8, 8),
            tile_stride=4,
            seed=args.seed,
            enable_motion_comp=args.motion_comp,
            doppler_grid=steer_cfg,
            steer_fuse=args.steer_fuse,
            angle_grouping=args.angle_grouping,
            steer_mode=args.steer_mode,
            tbd_enable=args.tbd,
        )
        scenario_motion_um = sim_cfg.motion_amp_um
        scenario_dropout = sim_cfg.sensor_dropout
        (
            base_pos,
            base_neg,
            stap_pos,
            stap_neg,
            pd_b,
            pd_s,
            m_flow,
            m_bg,
        ) = _simulate_scores_and_pd(
            n_pos=npos,
            n_neg=nneg,
            H=args.height,
            W=args.width,
            seed=args.seed,
            sim_config=sim_cfg,
        )
        if args.confirm2:
            pairs = 60_000 if fast else 200_000
            s1_null, s2_null = _simulate_confirm2_scores(
                pairs,
                rho=args.confirm2_rho,
                seed=args.seed + 101,
            )
    else:
        base_pos = _load_arr(args.base_pos)
        base_neg = _load_arr(args.base_neg)
        stap_pos = _load_arr(args.stap_pos)
        stap_neg = _load_arr(args.stap_neg)
        pd_b = _load_arr(args.base_pd)
        pd_s = _load_arr(args.stap_pd)
        m_flow = _load_arr(args.mask_flow)
        m_bg = _load_arr(args.mask_bg)
        if m_bg is None and m_flow is not None:
            m_bg = ~m_flow.astype(bool)
        if args.confirm2:
            s1_null = _load_arr(args.confirm2_scores1)
            s2_null = _load_arr(args.confirm2_scores2)
            if s1_null is None or s2_null is None:
                raise ValueError(
                    "Confirm-2 requires --confirm2-scores1 and "
                    "--confirm2-scores2 when not simulating"
                )

    timings: Dict[str, float] = {}
    tele = system_telemetry()

    timings_key = "data_generation_sec" if args.simulate else "data_loading_sec"
    timings[timings_key] = time.perf_counter() - run_start

    pd_stats_json = None
    if args.base_pd:
        stats_candidate = pathlib.Path(args.base_pd).parent / "pd_stats.json"
        if stats_candidate.exists():
            with open(stats_candidate, "r") as f:
                pd_stats_json = json.load(f)

    base_stats = pd_stats_json.get("baseline") if pd_stats_json else None
    stap_stats = pd_stats_json.get("stap") if pd_stats_json else None

    base = DetectorDataset(
        scores_pos=base_pos, scores_null=base_neg, pd_map=pd_b, pd_stats=base_stats
    )
    stap = DetectorDataset(
        scores_pos=stap_pos, scores_null=stap_neg, pd_map=pd_s, pd_stats=stap_stats
    )
    roc_summary = None
    if stap_pos is not None and stap_neg is not None:
        n_pos = int(len(stap_pos))
        n_neg = int(len(stap_neg))
        if n_neg > 0:
            fpr_min = 1.0 / float(n_neg)
            roc_summary = {
                "npos": n_pos,
                "nneg": n_neg,
                "fpr_min": fpr_min,
                "fpr_target": float(args.fpr_target),
                "target_resolvable": bool(args.fpr_target >= fpr_min),
            }
            try:
                null_quantiles = {
                    "q99": float(np.quantile(stap_neg, 0.99)),
                    "q999": float(np.quantile(stap_neg, 0.999)),
                }
                if n_neg >= 10000:
                    null_quantiles["q9999"] = float(np.quantile(stap_neg, 0.9999))
                roc_summary["null_quantiles"] = null_quantiles
            except Exception:
                pass
            try:
                pos_quantiles = {
                    "q90": float(np.quantile(stap_pos, 0.90)),
                    "q99": float(np.quantile(stap_pos, 0.99)),
                }
                roc_summary["pos_quantiles"] = pos_quantiles
            except Exception:
                pass

    masks = None
    if (pd_b is not None) and (pd_s is not None) and (m_flow is not None) and (m_bg is not None):
        masks = Masks(mask_flow=m_flow.astype(bool), mask_bg=m_bg.astype(bool))

    targets = AcceptanceTargets(
        delta_pdsnrdB_min=args.delta_snr_min,
        delta_tpr_at_fpr_min=args.delta_tpr_min,
        fpr_target=args.fpr_target,
        alpha_for_calibration=args.alpha,
    )

    t0 = time.perf_counter()
    report = acceptance_report(
        base,
        stap,
        masks,
        targets,
        seed=args.seed,
        evd_mode=args.evd_mode,
        evd_endpoint=args.evd_endpoint,
        min_exceedances_weibull=args.evd_min_exceed,
    )
    t1 = time.perf_counter()
    timings["acceptance_eval_sec"] = t1 - t0

    rho_inflate_used = args.rho_inflate

    # Build telemetry JSON
    payload = {
        "run_id": tele.get("run_id"),
        "created_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "seed": args.seed,
        "sim": {
            "simulate": bool(args.simulate),
            "H": args.height,
            "W": args.width,
            "motion_comp": bool(args.motion_comp),
            "steer_grid": list(_parse_steer_grid(args.steer_grid)),
            "steer_fuse": args.steer_fuse,
            "angle_grouping": args.angle_grouping,
        },
        "device": device or "cpu",
        "targets": targets.__dict__,
        "evd": {
            "mode": args.evd_mode,
            "endpoint_hint": args.evd_endpoint,
            "min_exceed_weibull": args.evd_min_exceed,
        },
        "performance": report["performance"],
        "calibration": report["calibration"],
        "evt_diagnostics": report["evt_diagnostics"],
        "gates": report["gates"],
        "overall_pass": report["overall_pass"],
        "timings": timings,
        "timing_sec": timings.get("acceptance_eval_sec", t1 - t0),
        "conformal": report.get("conformal"),
        "hw": tele.get("hw", {}),
        "env": tele.get("env", {}),
        "git": tele.get("git", {}),
        "nvidia_smi_snapshot": tele.get("nvidia_smi"),
        "telemetry": tele,
        "seeds": {
            "main": args.seed,
            "confirm2": args.seed if args.confirm2 else None,
        },
        "stap_options": {
            "motion_comp": bool(args.motion_comp),
            "steer_mode": args.steer_mode,
            "steer_grid": list(_parse_steer_grid(args.steer_grid)),
            "steer_fuse": args.steer_fuse,
            "angle_grouping": args.angle_grouping,
            "robust_cov": args.robust_cov,
            "huber_c": float(args.huber_c),
            "lt_auto": bool(args.lt_auto),
            "fd_auto": bool(args.fd_auto),
            "tbd": bool(args.tbd),
            "rho_inflate": float(rho_inflate_used),
            "rho_inflate_policy": args.rho_inflate_policy,
            "score_mode": score_mode_selected,
            "bundle": args.bundle,
        },
        "data_summary": {
            "counts": {
                "npos": int(len(base_pos)),
                "nneg": int(len(base_neg)),
                "pd_shape": tuple(pd_b.shape) if isinstance(pd_b, np.ndarray) else None,
            },
            "baseline": {
                "scores_pos": _summary_stats(base_pos),
                "scores_null": _summary_stats(base_neg),
                "pd_map": _summary_stats(pd_b),
            },
            "stap": {
                "scores_pos": _summary_stats(stap_pos),
                "scores_null": _summary_stats(stap_neg),
                "pd_map": _summary_stats(pd_s),
            },
            "masks": {
                "flow": _mask_stats(m_flow),
                "background": _mask_stats(m_bg),
            },
        },
    }
    if roc_summary is not None:
        payload["roc_summary"] = roc_summary

    confirm2_payload = None
    policy_decision = None
    if args.confirm2:
        confirm2_start = time.perf_counter()
        n_pairs = min(len(s1_null), len(s2_null))
        if n_pairs < 2000:
            raise ValueError("Confirm-2 calibration requires at least 2k null pairs")
        half = n_pairs // 2
        if args.rho_inflate_policy == "auto":
            prelim = calibrate_confirm2(
                s1_null[:half],
                s2_null[:half],
                alpha2_target=args.confirm2_alpha2,
                seed=args.seed,
                rho_inflate=0.0,
                device=device,
            )
            policy_decision = rho_inflate_policy(
                rho_hat=prelim.rho_hat,
                rho_groups=None,
                motion_um=scenario_motion_um,
                dropout=scenario_dropout,
            )
            rho_inflate_used = policy_decision.delta
        calib = calibrate_confirm2(
            s1_null[:half],
            s2_null[:half],
            alpha2_target=args.confirm2_alpha2,
            seed=args.seed,
            rho_inflate=rho_inflate_used,
            device=device,
        )
        ev = evaluate_confirm2(
            calib,
            s1_null[half:n_pairs],
            s2_null[half:n_pairs],
            alpha=args.confirm2_ci_alpha,
            device=device,
        )
        timings["confirm2_total_sec"] = time.perf_counter() - confirm2_start
        confirm2_payload = {
            "alpha2_target": args.confirm2_alpha2,
            "alpha1_per_look": calib.alpha1,
            "rho_hat": calib.rho_hat,
            "rho_ci": [calib.rho_lo, calib.rho_hi],
            "rho_eff": calib.rho_eff,
            "rho_inflate": rho_inflate_used,
            "copula_mode": calib.copula_mode,
            "lambda_u_emp": calib.lambda_u_emp,
            "lambda_u_gauss": calib.lambda_u_gauss,
            "df_t": calib.df_t,
            "empirical_pair_pfa": ev.empirical_pair_pfa,
            "pair_ci": [ev.pair_ci_lo, ev.pair_ci_hi],
            "predicted_pair_pfa": ev.predicted_pair_pfa,
            "n_pairs_test": ev.n_null_pairs,
            "k_joint": ev.k_joint,
            "device": device or "cpu",
            "pot": {
                "look1": calib.pm1.as_dict(),
                "look2": calib.pm2.as_dict(),
            },
            "thresholds": {
                "tau": [calib.tau1, calib.tau2],
                "gamma": [calib.gamma1, calib.gamma2],
                "k": [calib.k1, calib.k2],
                "n_cal": [calib.n_cal1, calib.n_cal2],
            },
            "pairs_calibration": half,
            "timing_sec": timings["confirm2_total_sec"],
        }
        if policy_decision is not None:
            confirm2_payload["rho_policy"] = asdict(policy_decision)
        payload["confirm2"] = confirm2_payload

    payload["timings"]["total_sec"] = time.perf_counter() - run_start
    payload["timing_sec"] = payload["timings"]["total_sec"]
    payload["compute"] = {
        "gpu": sample_gpu_stats(include_nvidia_smi=False),
    }

    # Save JSON
    out_dir = _default_outdir(args.out_dir)
    os.makedirs(out_dir, exist_ok=True)
    ts = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    json_path = os.path.join(out_dir, f"acceptance_{ts}.json")
    with open(json_path, "w") as f:
        json.dump(payload, f, indent=2)
    print(f"[acceptance] wrote {json_path}")

    if args.csv:
        row = {
            "json_path": json_path,
            "overall_pass": payload["overall_pass"],
            "pd_snr_delta_db": payload["performance"].get("pd_snr_delta_db"),
            "tpr_at_fpr_delta": payload["performance"].get("tpr_at_fpr_delta"),
            "pauc_delta": payload["performance"].get("pauc_delta"),
            "alpha": args.alpha,
            "fpr_target": args.fpr_target,
            "delta_snr_min": args.delta_snr_min,
            "delta_tpr_min": args.delta_tpr_min,
            "seed": args.seed,
            "timing_sec": payload["timing_sec"],
        }
        if confirm2_payload:
            row.update(
                {
                    "confirm2_alpha2": confirm2_payload["alpha2_target"],
                    "confirm2_rho_hat": confirm2_payload["rho_hat"],
                    "confirm2_emp_pair": confirm2_payload["empirical_pair_pfa"],
                    "confirm2_pred_pair": confirm2_payload["predicted_pair_pfa"],
                }
            )
        csv_path = pathlib.Path(args.csv)
        csv_path.parent.mkdir(parents=True, exist_ok=True)
        write_header = not csv_path.exists()
        import csv

        with csv_path.open("a", newline="") as fcsv:
            writer = csv.DictWriter(fcsv, fieldnames=list(row.keys()))
            if write_header:
                writer.writeheader()
            writer.writerow(row)

    # Save PDF summary
    pdf_dir = "reports" if args.report_dir is None else args.report_dir
    os.makedirs(pdf_dir, exist_ok=True)
    pdf_path = os.path.join(pdf_dir, f"acceptance_summary_{ts}.pdf")
    thumbs_dir = args.fig_dir or "figs/outputs"
    save_summary_pdf(
        pdf_path=pdf_path,
        acceptance_json=json_path,
        thumbs_dir=thumbs_dir,
        title="STAP-for-fUS — Acceptance Summary",
    )
    print(f"[report] wrote {pdf_path}")
    return json_path, pdf_path


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(
        description="Run acceptance metrics and produce a one-page PDF summary."
    )
    # From files / bundles
    ap.add_argument(
        "--bundle",
        type=str,
        default=None,
        help="Acceptance bundle directory (pw_* folder) to auto-load inputs",
    )
    ap.add_argument(
        "--score-mode",
        type=str,
        default="auto",
        choices=["auto", "msd", "pd", "band_ratio"],
        help="Score pool to use when --bundle is provided (default: bundle's recommendation)",
    )
    # Manual overrides
    ap.add_argument("--base_pos", type=str, default=None, help="NPY/NPZ baseline positive scores")
    ap.add_argument("--base_neg", type=str, default=None, help="NPY/NPZ baseline null scores")
    ap.add_argument("--stap_pos", type=str, default=None, help="NPY/NPZ STAP positive scores")
    ap.add_argument("--stap_neg", type=str, default=None, help="NPY/NPZ STAP null scores")
    ap.add_argument("--base_pd", type=str, default=None, help="NPY/NPZ baseline PD map")
    ap.add_argument("--stap_pd", type=str, default=None, help="NPY/NPZ STAP PD map")
    ap.add_argument("--mask_flow", type=str, default=None, help="NPY/NPZ vessel mask (bool)")
    ap.add_argument("--mask_bg", type=str, default=None, help="NPY/NPZ background mask (bool)")
    # Simulation
    ap.add_argument("--simulate", action="store_true", help="Use synthetic data")
    default_device = None
    if torch is not None and torch.cuda.is_available():
        default_device = "cuda"

    ap.add_argument("--height", type=int, default=128)
    ap.add_argument("--width", type=int, default=128)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument(
        "--device",
        type=str,
        default=default_device,
        help="Device for optional GPU acceleration (e.g., cuda, cpu, none)",
    )
    ap.add_argument(
        "--steer-mode",
        type=str,
        default="bank",
        choices=["bank", "lcmv"],
        help="Steering frontend to emulate (bank or lcmv)",
    )
    # Confirm-2 options
    ap.add_argument("--confirm2", action="store_true", help="Calibrate/evaluate Confirm-2 overlay")
    ap.add_argument(
        "--confirm2-scores1",
        dest="confirm2_scores1",
        type=str,
        default=None,
        help="NPY/NPZ null scores (look 1) when using --confirm2 without --simulate",
    )
    ap.add_argument(
        "--confirm2-scores2",
        dest="confirm2_scores2",
        type=str,
        default=None,
        help="NPY/NPZ null scores (look 2) when using --confirm2 without --simulate",
    )
    ap.add_argument("--confirm2-alpha2", dest="confirm2_alpha2", type=float, default=1e-3)
    ap.add_argument(
        "--confirm2-ci-alpha",
        dest="confirm2_ci_alpha",
        type=float,
        default=0.05,
        help="Two-sided CI alpha for Confirm-2 pair-Pfa (default 0.05)",
    )
    ap.add_argument(
        "--confirm2-rho",
        dest="confirm2_rho",
        type=float,
        default=0.5,
        help="Correlation used when simulating Confirm-2 pairs",
    )
    ap.add_argument(
        "--rho-inflate",
        type=float,
        default=0.0,
        help="Additive rho inflation before mapping alpha2 to alpha1",
    )
    ap.add_argument(
        "--robust-cov",
        type=str,
        default="none",
        choices=["none", "scm", "huber", "tyler", "tyler_pca"],
        help="Robust covariance setting (metadata only).",
    )
    ap.add_argument(
        "--huber-c", type=float, default=5.0, help="Huber c parameter if robust-cov=huber"
    )
    ap.add_argument(
        "--lt-auto", action="store_true", help="Enable Lt auto-selection telemetry flag"
    )
    ap.add_argument(
        "--fd-auto", action="store_true", help="Enable Doppler bank auto-sizing telemetry flag"
    )
    ap.add_argument(
        "--rho-inflate-policy",
        type=str,
        default="none",
        choices=["none", "auto"],
        help="Strategy for Confirm-2 rho inflation (none|auto).",
    )
    ap.add_argument(
        "--motion-comp",
        action="store_true",
        help="Enable simulated tile motion compensation effects",
    )
    ap.add_argument(
        "--steer-grid",
        type=str,
        default=None,
        help="Comma-separated Doppler offsets (Hz) for steering bank",
    )
    ap.add_argument(
        "--steer-fuse",
        type=str,
        default="max",
        help="Fusion rule for steering bank (max|sos|lse)",
    )
    ap.add_argument(
        "--angle-grouping",
        type=str,
        default="none",
        help="Grouping strategy for Confirm-2 looks (spectral|greedy|none)",
    )
    ap.add_argument("--tbd", action="store_true", help="Apply temporal HMM smoothing (TBD)")
    # Targets
    ap.add_argument(
        "--alpha",
        type=float,
        default=1e-5,
        help="Per-look calibration target for EVT+conformal",
    )
    ap.add_argument(
        "--fpr_target",
        type=float,
        default=1e-5,
        help="FPR operating point for TPR@FPR",
    )
    ap.add_argument("--delta_snr_min", type=float, default=3.0)
    ap.add_argument("--delta_tpr_min", type=float, default=0.05)
    ap.add_argument(
        "--evd-mode",
        type=str,
        default="weibull",
        choices=["weibull", "gpd"],
        help="Tail model for EVT (weibull for bounded ratios, gpd for generic).",
    )
    ap.add_argument(
        "--evd-endpoint",
        type=float,
        default=None,
        help="Optional upper endpoint hint for bounded scores (e.g., 1/rho for MSD ratio).",
    )
    ap.add_argument(
        "--evd-min-exceed",
        type=int,
        default=500,
        help="Minimum exceedances required for Weibull POT fit (default 500).",
    )
    # Output
    ap.add_argument("--out_dir", type=str, default="runs")
    ap.add_argument("--report_dir", type=str, default="reports")
    ap.add_argument(
        "--fig_dir",
        type=str,
        default="figs/outputs",
        help="Where thumbnails are looked up",
    )
    ap.add_argument("--csv", type=str, default=None, help="Optional CSV for summary metrics")
    return ap


def main(argv: list[str] | None = None):
    ap = build_parser()
    args = ap.parse_args(argv)
    run(args)


if __name__ == "__main__":
    main()
