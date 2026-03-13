from __future__ import annotations

import os
import json
import time
from pathlib import Path
from typing import Any, Sequence

import numpy as np

from sim.kwave import common as kw


def _default_masks_generic(H: int, W: int) -> tuple[np.ndarray, np.ndarray]:
    cy = min(int(0.6 * H), H - 1)
    cx = W // 2
    rr = max(4, int(0.12 * min(H, W)))
    yy, xx = np.ogrid[:H, :W]
    mask_flow = (yy - cy) ** 2 + (xx - cx) ** 2 <= rr**2
    mask_bg = np.ones((H, W), dtype=bool)
    mask_bg[: max(1, H // 8), :] = False
    mask_bg &= ~mask_flow
    if mask_bg.sum() < 64:
        mask_bg = ~mask_flow
    return mask_flow.astype(bool), mask_bg.astype(bool)


def write_acceptance_bundle_from_icube(
    *,
    out_root: Path,
    dataset_name: str,
    Icube: np.ndarray,
    prf_hz: float,
    seed: int = 0,
    tile_hw: tuple[int, int] = (8, 8),
    tile_stride: int = 3,
    local_svd_hann: bool = True,
    Lt: int = 8,
    diag_load: float = 0.07,
    stap_cov_train_trim_q: float = 0.0,
    cov_estimator: str = "tyler_pca",
    huber_c: float = 5.0,
    mvdr_load_mode: str = "auto",
    mvdr_auto_kappa: float = 120.0,
    constraint_ridge: float = 0.18,
    fd_span_mode: str = "psd",
    fd_span_rel: tuple[float, float] = (0.30, 1.10),
    fd_fixed_span_hz: float | None = None,
    constraint_mode: str = "exp+deriv",
    grid_step_rel: float = 0.20,
    fd_min_pts: int = 3,
    fd_max_pts: int = 11,
    fd_min_abs_hz: float = 0.0,
    msd_lambda: float | None = 0.05,
    stap_detector_variant: str = "msd_ratio",
    stap_whiten_gamma: float = 1.0,
    hybrid_rescue_rule: str = "guard_frac_v1",
    msd_ridge: float = 0.10,
    msd_agg_mode: str = "median",
    msd_ratio_rho: float = 0.05,
    motion_half_span_rel: float | None = None,
    msd_contrast_alpha: float | None = 0.6,
    alias_psd_select_enable: bool = False,
    alias_psd_select_ratio_thresh: float = 1.2,
    alias_psd_select_bins: int = 1,
    score_mode: str = "pd",
    stap_device: str | None = None,
    baseline_type: str = "mc_svd",
    reg_enable: bool = False,
    reg_method: str = "phasecorr",
    reg_subpixel: int = 4,
    reg_reference: str = "median",
    svd_rank: int | None = None,
    svd_energy_frac: float | None = 0.95,
    svd_keep_min: int | None = None,
    svd_keep_max: int | None = None,
    svd_sim_smooth: int = 7,
    svd_sim_kappa: float = 2.5,
    svd_sim_r_min: int = 1,
    # Baseline RPCA parameters (used only when baseline_type=rpca).
    rpca_lambda: float | None = None,
    rpca_max_iters: int = 250,
    rpca_spatial_downsample: int | None = None,
    rpca_t_sub: int | None = None,
    rpca_tol: float = 1e-4,
    rpca_rank_k_max: int = 8,
    # Baseline HOSVD parameters (used only when baseline_type=hosvd).
    hosvd_ranks: tuple[int, int, int] | None = None,
    hosvd_energy_fracs: tuple[float, float, float] | None = (0.99, 0.99, 0.99),
    hosvd_spatial_downsample: int = 2,
    hosvd_t_sub: int | None = None,
    hosvd_max_iters: int = 1,
    flow_mask_mode: str = "pd_auto",
    flow_mask_pd_quantile: float = 0.995,
    flow_mask_depth_min_frac: float = 0.25,
    flow_mask_depth_max_frac: float = 0.85,
    flow_mask_erode_iters: int = 0,
    flow_mask_dilate_iters: int = 2,
    flow_mask_min_pixels: int = 64,
    flow_mask_min_coverage_fraction: float = 0.0,
    flow_mask_union_default: bool = True,
    psd_tapers: int = 3,
    psd_bandwidth: float = 2.0,
    # Optional external evaluation masks (e.g., B-mode structural tube masks for phantoms).
    mask_flow_override: np.ndarray | None = None,
    mask_bg_override: np.ndarray | None = None,
    # Optional conditional STAP (compute heuristic). Disable for structural-mask evaluations.
    stap_conditional_enable: bool = True,
    # Doppler band design for band-ratio telemetry.
    band_ratio_mode: str = "whitened",
    band_ratio_flow_low_hz: float = 30.0,
    band_ratio_flow_high_hz: float = 220.0,
    band_ratio_alias_center_hz: float = 400.0,
    band_ratio_alias_width_hz: float = 120.0,
    # Optional score-space KA v2 (shrink-only veto). This is disabled by default
    # for safety; enable explicitly for profile studies.
    score_ka_v2_enable: bool = False,
    score_ka_v2_mode: str = "safety",  # safety|uplift|auto
    run_stap: bool = True,
    feasibility_mode: kw.FeasibilityMode = "legacy",
    meta_extra: dict[str, Any] | None = None,
) -> dict[str, str]:
    """
    Convert a pre-beamformed slow-time IQ cube into an "acceptance bundle".

    The bundle format matches what `scripts/hab_contract_check.py` expects:
      - meta.json
      - mask_flow.npy / mask_bg.npy
      - base_band_ratio_map.npy (+ base_m_alias_map.npy)
      - base_score_map.npy / stap_score_map.npy / stap_score_pool_map.npy
      - pd_base.npy / pd_stap.npy
      - score_pd_base.npy / score_pd_stap.npy   (explicit right-tail PD scores; current convention score_pd=pd)
      - score_base_pdlog.npy / score_base_kasai.npy (optional baseline score maps; right-tail)
    """
    if Icube.ndim != 3:
        raise ValueError(f"Icube must have shape (T,H,W), got {Icube.shape}")
    if not np.iscomplexobj(Icube):
        raise ValueError("Icube must be complex-valued IQ")
    Icube = np.asarray(Icube, dtype=np.complex64)

    bundle_dir = out_root / dataset_name
    bundle_dir.mkdir(parents=True, exist_ok=True)
    paths: dict[str, str] = {}

    def _save(name: str, arr: np.ndarray) -> None:
        path = bundle_dir / f"{name}.npy"
        np.save(path, arr, allow_pickle=False)
        paths[name] = str(path)

    def _save_text(key: str, filename: str, text: str) -> None:
        path = bundle_dir / filename
        with open(path, "w", encoding="utf-8") as f:
            f.write(text)
        paths[key] = str(path)

    # ---- Band-ratio spec ----
    feas_mode = kw._normalize_feasibility_mode(feasibility_mode)
    band_ratio_mode_norm = (band_ratio_mode or "legacy").strip().lower()
    if band_ratio_mode_norm not in {"legacy", "whitened"}:
        raise ValueError("band_ratio_mode must be 'legacy' or 'whitened'")
    use_whitened_ratio = band_ratio_mode_norm == "whitened"
    if not use_whitened_ratio:
        raise ValueError("This bundle writer requires band_ratio_mode='whitened'.")
    band_ratio_spec = {
        "flow_low_hz": float(band_ratio_flow_low_hz),
        "flow_high_hz": float(band_ratio_flow_high_hz),
        "alias_center_hz": float(band_ratio_alias_center_hz),
        "alias_width_hz": float(band_ratio_alias_width_hz),
    }
    variant_requested_norm = str(stap_detector_variant or "msd_ratio").strip().lower()
    stap_detector_variant_norm = {
        "msd": "msd_ratio",
        "ratio": "msd_ratio",
        "default": "msd_ratio",
        "stap": "msd_ratio",
        "whitened_pd": "whitened_power",
        "power": "whitened_power",
        "raw": "unwhitened_ratio",
        "unwhitened": "unwhitened_ratio",
        "raw_ratio": "unwhitened_ratio",
        "no_whiten": "unwhitened_ratio",
        "msd_unwhitened": "unwhitened_ratio",
        "hybrid": "hybrid_rescue",
        "hybrid_rescue": "hybrid_rescue",
        "adaptive_guard": "hybrid_rescue",
        "adaptive_guard_v1": "hybrid_rescue",
        "guard_promote": "hybrid_rescue",
    }.get(variant_requested_norm, variant_requested_norm)
    if stap_detector_variant_norm not in {"msd_ratio", "whitened_power", "unwhitened_ratio", "hybrid_rescue"}:
        raise ValueError(
            f"Unsupported stap_detector_variant={stap_detector_variant!r}. Expected 'msd_ratio', "
            "'whitened_power', 'unwhitened_ratio', 'hybrid_rescue', or 'adaptive_guard'."
        )
    hybrid_rule_cfg = None
    hybrid_variant_label: str | None = None
    if stap_detector_variant_norm == "hybrid_rescue":
        if variant_requested_norm in {"adaptive_guard", "adaptive_guard_v1", "guard_promote"}:
            hybrid_rule_cfg = kw._normalize_hybrid_rescue_rule("guard_promote_v1")
            hybrid_variant_label = "adaptive_guard"
        else:
            hybrid_rule_cfg = kw._normalize_hybrid_rescue_rule(hybrid_rescue_rule)
            hybrid_variant_label = "hybrid_rescue"
    if getattr(kw, "_normalize_whiten_gamma", None) is not None:
        stap_whiten_gamma = kw._normalize_whiten_gamma(  # type: ignore[attr-defined]
            stap_whiten_gamma,
            detector_variant=("msd_ratio" if stap_detector_variant_norm == "hybrid_rescue" else stap_detector_variant_norm),
        )
    else:
        stap_whiten_gamma = float(min(1.0, max(0.0, float(stap_whiten_gamma))))
    br_spec = kw.BandRatioSpec(
        flow_low_hz=float(band_ratio_spec["flow_low_hz"]),
        flow_high_hz=float(band_ratio_spec["flow_high_hz"]),
        alias_center_hz=float(band_ratio_spec["alias_center_hz"]),
        alias_width_hz=float(band_ratio_spec["alias_width_hz"]),
    )

    # ---- Baseline PD ----
    baseline_type_norm = (baseline_type or "").strip().lower()
    baseline_device = "cuda" if (kw._resolve_stap_device(stap_device).startswith("cuda")) else "cpu"
    if baseline_type_norm == "mc_svd":
        pd_base, baseline_telemetry, baseline_filtered_cube = kw._baseline_pd_mcsvd(
            Icube,
            reg_enable=reg_enable,
            reg_method=reg_method,
            reg_subpixel=max(1, int(reg_subpixel)),
            reg_reference=reg_reference,
            svd_rank=svd_rank,
            svd_energy_frac=svd_energy_frac,
            device=baseline_device,
            return_filtered_cube=True,
        )
    elif baseline_type_norm in {"svd_similarity", "svd_sim"}:
        if reg_method != "phasecorr":
            raise ValueError("Only phasecorr registration is supported for SVD baselines.")
        reg_cube, tele_reg = kw._register_stack_phasecorr(
            Icube,
            reg_enable=reg_enable,
            upsample=max(1, int(reg_subpixel)),
            ref_strategy=reg_reference,
        )
        pd_base, baseline_telemetry, baseline_filtered_cube = kw._baseline_pd_svd_similarity(
            reg_cube,
            svd_sim_smooth=int(svd_sim_smooth),
            svd_sim_kappa=float(svd_sim_kappa),
            svd_sim_r_min=int(svd_sim_r_min),
            svd_sim_r_max=svd_rank,
            device=baseline_device,
            return_filtered_cube=True,
        )
        try:
            baseline_ms = float((baseline_telemetry or {}).get("baseline_ms", 0.0) or 0.0)
            reg_ms = float((tele_reg or {}).get("reg_ms", 0.0) or 0.0)
            baseline_telemetry = dict(baseline_telemetry or {})
            baseline_telemetry["baseline_ms"] = baseline_ms + reg_ms
        except Exception:
            baseline_telemetry = dict(baseline_telemetry or {})
        baseline_telemetry = {**dict(tele_reg or {}), **dict(baseline_telemetry or {})}
    elif baseline_type_norm in {"local_svd", "svd_local"}:
        if reg_method != "phasecorr":
            raise ValueError("Only phasecorr registration is supported for SVD baselines.")
        reg_cube, tele_reg = kw._register_stack_phasecorr(
            Icube,
            reg_enable=reg_enable,
            upsample=max(1, int(reg_subpixel)),
            ref_strategy=reg_reference,
        )
        pd_base, baseline_telemetry, baseline_filtered_cube = kw._baseline_pd_local_svd(
            reg_cube,
            tile_hw=tile_hw,
            stride=int(tile_stride),
            svd_energy_frac=float(svd_energy_frac) if svd_energy_frac is not None else 0.90,
            hann=bool(local_svd_hann),
            device=baseline_device,
            return_filtered_cube=True,
        )
        try:
            baseline_ms = float((baseline_telemetry or {}).get("baseline_ms", 0.0) or 0.0)
            reg_ms = float((tele_reg or {}).get("reg_ms", 0.0) or 0.0)
            baseline_telemetry = dict(baseline_telemetry or {})
            baseline_telemetry["baseline_ms"] = baseline_ms + reg_ms
        except Exception:
            baseline_telemetry = dict(baseline_telemetry or {})
        baseline_telemetry = {**dict(tele_reg or {}), **dict(baseline_telemetry or {})}
    elif baseline_type_norm in {"adaptive_local_svd", "local_svd_similarity", "adaptive_local"}:
        if reg_method != "phasecorr":
            raise ValueError("Only phasecorr registration is supported for SVD baselines.")
        reg_cube, tele_reg = kw._register_stack_phasecorr(
            Icube,
            reg_enable=reg_enable,
            upsample=max(1, int(reg_subpixel)),
            ref_strategy=reg_reference,
        )
        pd_base, baseline_telemetry, baseline_filtered_cube = kw._baseline_pd_adaptive_local_svd(
            reg_cube,
            tile_hw=tile_hw,
            stride=int(tile_stride),
            svd_sim_smooth=int(svd_sim_smooth),
            svd_sim_kappa=float(svd_sim_kappa),
            svd_sim_r_min=int(svd_sim_r_min),
            svd_sim_r_max=svd_rank,
            hann=bool(local_svd_hann),
            device=baseline_device,
            return_filtered_cube=True,
        )
        try:
            baseline_ms = float((baseline_telemetry or {}).get("baseline_ms", 0.0) or 0.0)
            reg_ms = float((tele_reg or {}).get("reg_ms", 0.0) or 0.0)
            baseline_telemetry = dict(baseline_telemetry or {})
            baseline_telemetry["baseline_ms"] = baseline_ms + reg_ms
        except Exception:
            baseline_telemetry = dict(baseline_telemetry or {})
        baseline_telemetry = {**dict(tele_reg or {}), **dict(baseline_telemetry or {})}
    elif baseline_type_norm in {"raw", "none", "identity"}:
        # STAP-only / no baseline clutter suppression: keep the (optionally registered) IQ cube
        # and use PD as a simple magnitude-squared average.
        if reg_method != "phasecorr":
            raise ValueError("Only phasecorr registration is supported for baseline_type=raw/none.")
        t0_raw = time.perf_counter()
        reg_cube, tele_reg = kw._register_stack_phasecorr(
            Icube,
            reg_enable=reg_enable,
            upsample=max(1, int(reg_subpixel)),
            ref_strategy=reg_reference,
        )
        pd_base = (np.mean(np.abs(reg_cube) ** 2, axis=0)).astype(np.float32, copy=False)
        baseline_filtered_cube = reg_cube
        baseline_telemetry = dict(tele_reg or {})
        baseline_telemetry["baseline_type"] = "raw"
        baseline_telemetry["baseline_ms"] = float(1000.0 * (time.perf_counter() - t0_raw))
    elif baseline_type_norm in {"svd_bandpass", "svd_range", "ulm_svd"}:
        if svd_keep_min is None:
            raise ValueError("svd_keep_min must be provided for baseline_type=svd_bandpass")
        pd_base, baseline_telemetry, baseline_filtered_cube = kw._baseline_pd_svd_bandpass(
            Icube,
            reg_enable=reg_enable,
            reg_method=reg_method,
            reg_subpixel=max(1, int(reg_subpixel)),
            reg_reference=reg_reference,
            svd_keep_min=int(svd_keep_min),
            svd_keep_max=int(svd_keep_max) if svd_keep_max is not None else None,
            device=baseline_device,
            return_filtered_cube=True,
        )
    elif baseline_type_norm in {"rpca", "hosvd"}:
        if reg_method != "phasecorr":
            raise ValueError("Only phasecorr registration is supported for RPCA/HOSVD baselines.")
        reg_cube, tele_reg = kw._register_stack_phasecorr(
            Icube,
            reg_enable=reg_enable,
            upsample=max(1, int(reg_subpixel)),
            ref_strategy=reg_reference,
        )
        if baseline_type_norm == "rpca":
            pd_base, baseline_telemetry, baseline_filtered_cube = kw._baseline_pd_rpca(
                reg_cube,
                lambda_=rpca_lambda,
                max_iters=int(rpca_max_iters),
                spatial_downsample=rpca_spatial_downsample,
                t_sub=rpca_t_sub,
                tol=float(rpca_tol),
                rank_k_max=int(rpca_rank_k_max),
                return_filtered_cube=True,
            )
        else:
            pd_base, baseline_telemetry, baseline_filtered_cube = kw._baseline_pd_hosvd(
                reg_cube,
                ranks=hosvd_ranks,
                energy_fracs=hosvd_energy_fracs,
                max_iters=int(hosvd_max_iters),
                spatial_downsample=max(1, int(hosvd_spatial_downsample)),
                t_sub=hosvd_t_sub,
                return_filtered_cube=True,
            )
        try:
            baseline_ms = float((baseline_telemetry or {}).get("baseline_ms", 0.0) or 0.0)
            reg_ms = float((tele_reg or {}).get("reg_ms", 0.0) or 0.0)
            baseline_telemetry = dict(baseline_telemetry or {})
            baseline_telemetry["baseline_ms"] = baseline_ms + reg_ms
        except Exception:
            baseline_telemetry = dict(baseline_telemetry or {})
        baseline_telemetry = {**dict(tele_reg or {}), **dict(baseline_telemetry or {})}
    else:
        raise ValueError(
            f"Unsupported baseline_type={baseline_type_norm!r} for IQ bundle writer. "
            "Use mc_svd, svd_similarity, local_svd, adaptive_local_svd, raw/none, svd_bandpass, rpca, or hosvd."
        )

    H, W = pd_base.shape
    tile_count = kw._tile_count((H, W), tile_hw, tile_stride)

    # ---- Additional baseline score maps (standard Doppler short-ensemble scores) ----
    # These are derived *only* from the baseline-filtered cube and are intended to
    # strengthen baseline fairness in short-ensemble real-data evaluations.
    #
    # score_base_pdlog: log power Doppler (monotone transform of pd_base; right-tail).
    # score_base_kasai: log |lag-1 autocorrelation| ("Kasai power"; right-tail).
    score_base_pdlog: np.ndarray | None = None
    score_base_kasai: np.ndarray | None = None
    try:
        eps = 1e-12
        score_base_pdlog = np.log(pd_base.astype(np.float64, copy=False) + eps).astype(
            np.float32, copy=False
        )
        if baseline_filtered_cube is not None and baseline_filtered_cube.shape[0] >= 2:
            y = baseline_filtered_cube.astype(np.complex64, copy=False)
            r1 = np.sum(y[1:] * np.conj(y[:-1]), axis=0).astype(np.complex64, copy=False)
            score_base_kasai = np.log(np.abs(r1).astype(np.float64, copy=False) + eps).astype(
                np.float32, copy=False
            )
    except Exception:
        score_base_pdlog = None
        score_base_kasai = None

    # ---- Masks ----
    mask_flow_default, mask_bg_default = _default_masks_generic(H, W)
    if mask_flow_override is not None or mask_bg_override is not None:
        if mask_flow_override is None or mask_bg_override is None:
            raise ValueError("mask_flow_override and mask_bg_override must be provided together.")
        mask_flow = np.asarray(mask_flow_override, dtype=bool)
        mask_bg = np.asarray(mask_bg_override, dtype=bool)
        if mask_flow.shape != (H, W) or mask_bg.shape != (H, W):
            raise ValueError(
                f"Override mask shape mismatch: got flow={mask_flow.shape}, bg={mask_bg.shape}, "
                f"expected {(H, W)}"
            )
        # Enforce disjointness.
        mask_bg = mask_bg & (~mask_flow)
        flow_mask_stats = {
            "mode": "override",
            "coverage_default": float(mask_flow_default.mean()),
            "coverage_pre_union": float(mask_flow.mean()),
            "coverage_post_union": float(mask_flow.mean()),
            "union_applied": 0.0,
            "pd_auto_used": 0.0,
            "override_flow_coverage": float(mask_flow.mean()),
            "override_bg_coverage": float(mask_bg.mean()),
        }
    else:
        mask_flow, mask_bg, flow_mask_stats = kw._resolve_flow_mask(
            pd_base,
            mask_flow_default,
            mask_bg_default,
            mode=(flow_mask_mode or "pd_auto").strip().lower(),
            pd_quantile=flow_mask_pd_quantile,
            depth_min_frac=flow_mask_depth_min_frac,
            depth_max_frac=flow_mask_depth_max_frac,
            erode_iters=flow_mask_erode_iters,
            dilate_iters=flow_mask_dilate_iters,
            min_pixels=flow_mask_min_pixels,
            min_coverage_frac=flow_mask_min_coverage_fraction,
            union_with_default=flow_mask_union_default,
        )

    # ---- Whitened band-ratio telemetry (baseline + STAP) ----
    base_br_recorder = kw.BandRatioRecorder(
        prf_hz,
        tile_count,
        br_spec,
        tapers=psd_tapers,
        bandwidth=psd_bandwidth,
    )
    kw._collect_band_ratio_from_cube(
        baseline_filtered_cube,
        base_br_recorder,
        mask_bg,
        tile_hw,
        tile_stride,
    )
    base_tile_scores, base_br_stats = base_br_recorder.finalize()
    baseline_telemetry = dict(baseline_telemetry or {})
    baseline_telemetry.setdefault("band_ratio_stats", base_br_stats or {"count": 0})
    # IMPORTANT: the clinical STAP pipeline operates on the baseline-filtered
    # slow-time cube (e.g. MC-SVD residual), not the raw IQ. Keep this cube so
    # that the STAP pass is applied on the same residual used to form `pd_base`
    # and the telemetry/masks.
    cube_for_stap = baseline_filtered_cube

    if base_tile_scores.size == tile_count:
        base_band_ratio_map = kw._tile_scores_to_map(base_tile_scores, (H, W), tile_hw, tile_stride)
    else:
        base_band_ratio_map = np.ones((H, W), dtype=np.float32)
    base_m_alias_map = (-base_band_ratio_map).astype(np.float32, copy=False)
    base_guard_frac_map = kw._tile_scores_to_map(
        np.asarray(base_br_recorder.rg_raw, dtype=np.float32),
        (H, W),
        tile_hw,
        tile_stride,
    )
    base_peak_freq_map = kw._tile_scores_to_map(
        np.asarray(base_br_recorder.peak_freqs, dtype=np.float32),
        (H, W),
        tile_hw,
        tile_stride,
    )
    base_peak_bin_map = kw._tile_scores_to_map(
        np.asarray(base_br_recorder.peak_bins, dtype=np.int32),
        (H, W),
        tile_hw,
        tile_stride,
    ).astype(np.int16, copy=False)
    stap_cond_loaded_map = np.full((H, W), np.nan, dtype=np.float32)
    stap_cov_rank_proxy_map = np.full((H, W), np.nan, dtype=np.float32)
    stap_flow_mu_ratio_map = np.full((H, W), np.nan, dtype=np.float32)
    stap_bg_var_inflation_map = np.full((H, W), np.nan, dtype=np.float32)
    stap_gamma_flow_map = np.full((H, W), np.nan, dtype=np.float32)
    stap_gamma_perp_map = np.full((H, W), np.nan, dtype=np.float32)

    # ---- STAP PD + optional STAP band-ratio recorder ----
    hybrid_rescue_mask: np.ndarray | None = None
    if run_stap:
        def _run_stap_once(
            *,
            detector_variant: str,
            whiten_gamma: float,
            recorder: kw.BandRatioRecorder | None,
            conditional_mask_flow: np.ndarray | None = None,
            conditional_mask_bg: np.ndarray | None = None,
            conditional_enable_override: bool | None = None,
        ) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
            return kw._stap_pd(
                cube_for_stap,
                tile_hw=tile_hw,
                stride=tile_stride,
                Lt=Lt,
                prf_hz=prf_hz,
                diag_load=diag_load,
                cov_train_trim_q=stap_cov_train_trim_q,
                estimator=cov_estimator,
                huber_c=huber_c,
                mvdr_load_mode=mvdr_load_mode,
                mvdr_auto_kappa=mvdr_auto_kappa,
                constraint_ridge=constraint_ridge,
                fd_span_mode=fd_span_mode,
                fd_span_rel=fd_span_rel,
                fd_fixed_span_hz=fd_fixed_span_hz,
                constraint_mode=constraint_mode,
                grid_step_rel=grid_step_rel,
                min_pts=fd_min_pts,
                max_pts=fd_max_pts,
                fd_min_abs_hz=fd_min_abs_hz,
                msd_lambda=msd_lambda,
                detector_variant=detector_variant,
                whiten_gamma=whiten_gamma,
                msd_ridge=msd_ridge,
                msd_agg_mode=msd_agg_mode,
                msd_ratio_rho=msd_ratio_rho,
                motion_half_span_rel=motion_half_span_rel,
                msd_contrast_alpha=msd_contrast_alpha,
                stap_device=stap_device_resolved,
                tile_batch=tile_batch_eff,
                pd_base_full=pd_base,
                mask_flow=conditional_mask_flow if conditional_mask_flow is not None else mask_flow,
                mask_bg=conditional_mask_bg if conditional_mask_bg is not None else mask_bg,
                conditional_enable=(
                    bool(conditional_enable_override)
                    if conditional_enable_override is not None
                    else bool(stap_conditional_enable)
                ),
                ka_mode="none",
                ka_prior_library=None,
                ka_opts=None,
                alias_psd_select_enable=alias_psd_select_enable,
                alias_psd_select_ratio_thresh=alias_psd_select_ratio_thresh,
                alias_psd_select_bins=alias_psd_select_bins,
                psd_telemetry=False,
                psd_tapers=psd_tapers,
                psd_bandwidth=psd_bandwidth,
                band_ratio_recorder=recorder,
                feasibility_mode=feas_mode,
                band_ratio_spec=dict(band_ratio_spec),
            )

        def _merge_branch_info(
            primary_info: dict[str, Any],
            secondary_info: dict[str, Any] | None,
        ) -> dict[str, Any]:
            merged = dict(primary_info or {})
            if not secondary_info:
                return merged
            for key in ("stap_extract_ms", "stap_batch_proc_ms", "stap_post_ms", "stap_total_ms"):
                v0 = float(merged.get(key, 0.0) or 0.0)
                v1 = float(secondary_info.get(key, 0.0) or 0.0)
                if v0 or v1:
                    merged[key] = v0 + v1
            merged["stap_fast_path_used"] = bool(
                merged.get("stap_fast_path_used", False)
                or secondary_info.get("stap_fast_path_used", False)
            )
            return merged

        stap_br_recorder = kw.BandRatioRecorder(
            prf_hz,
            tile_count,
            br_spec,
            tapers=psd_tapers,
            bandwidth=psd_bandwidth,
        )
        t_stap_start = time.time()
        stap_device_resolved = kw._resolve_stap_device(stap_device)
        env_fast = os.getenv("STAP_FAST_PATH", "").lower() in {"1", "true", "yes", "on"}
        tile_batch_eff = 192 if stap_device_resolved.startswith("cuda") else (64 if env_fast else 1)
        env_tile_batch = os.getenv("STAP_TILE_BATCH", "").strip()
        if env_tile_batch:
            try:
                tb = int(env_tile_batch)
                if tb > 0:
                    tile_batch_eff = tb
            except Exception:
                pass
        core_variant = "msd_ratio" if stap_detector_variant_norm == "hybrid_rescue" else stap_detector_variant_norm
        core_gamma = float(stap_whiten_gamma)
        if stap_detector_variant_norm == "hybrid_rescue" and hybrid_variant_label == "adaptive_guard":
            pd_stap, stap_scores, rescue_info = _run_stap_once(
                detector_variant="unwhitened_ratio",
                whiten_gamma=0.0,
                recorder=None,
            )
            choose_advanced = kw._hybrid_choose_advanced_mask(
                base_guard_frac_map,
                direction=str((hybrid_rule_cfg or {}).get("direction", ">=")),
                threshold=float((hybrid_rule_cfg or {}).get("threshold", 0.0)),
                prefer_advanced_on_invalid=True,
            )
            rescue_scores = stap_scores.copy()
            rescue_pd = pd_stap.copy()
            rescue_mask = (~choose_advanced).astype(np.bool_, copy=False)
            hybrid_stats = {
                "feature": str((hybrid_rule_cfg or {}).get("feature", "base_guard_frac_map")),
                "direction": str((hybrid_rule_cfg or {}).get("direction", ">=")),
                "threshold": float((hybrid_rule_cfg or {}).get("threshold", 0.0)),
                "prefer_advanced_on_invalid": True,
                "advanced_fraction": float(np.mean(choose_advanced)) if choose_advanced.size else None,
                "rescue_fraction": float(np.mean(rescue_mask)) if rescue_mask.size else None,
                "rescue_pixels": int(np.count_nonzero(rescue_mask)),
                "advanced_pixels": int(np.count_nonzero(choose_advanced)),
            }
            promote_fraction = float(hybrid_stats["advanced_fraction"] or 0.0)
            if bool(np.any(choose_advanced)):
                pd_promote, score_promote, promote_info = _run_stap_once(
                    detector_variant="msd_ratio",
                    whiten_gamma=float(core_gamma),
                    recorder=stap_br_recorder,
                    conditional_mask_flow=choose_advanced,
                    conditional_mask_bg=~choose_advanced,
                    conditional_enable_override=True,
                )
                pd_stap = np.where(choose_advanced, pd_promote, rescue_pd).astype(
                    np.float32, copy=False
                )
                stap_scores = np.where(choose_advanced, score_promote, rescue_scores).astype(
                    np.float32, copy=False
                )
                stap_info = _merge_branch_info(rescue_info, promote_info)
            else:
                stap_info = dict(rescue_info)
            hybrid_rescue_mask = rescue_mask
            stap_info["hybrid_rescue"] = {
                "enabled": True,
                "rule": dict(hybrid_rule_cfg or {}),
                "stats": hybrid_stats,
                "advanced_detector_variant": "msd_ratio",
                "advanced_whiten_gamma": float(stap_whiten_gamma),
                "rescue_detector_variant": "unwhitened_ratio",
                "rescue_whiten_gamma": 0.0,
                "rescue_condR_median": rescue_info.get("median_condR"),
                "rescue_cond_loaded_median": rescue_info.get("median_cond_loaded"),
                "promote_fraction": promote_fraction,
                "promote_active": bool(np.any(choose_advanced)),
            }
            stap_info["detector_variant_effective"] = str(hybrid_variant_label or "hybrid_rescue")
            stap_info["hybrid_rescue_mask_fraction"] = hybrid_stats.get("rescue_fraction")
            stap_info["adaptive_guard_promote_fraction"] = promote_fraction
        else:
            pd_stap, stap_scores, stap_info = _run_stap_once(
                detector_variant=str(core_variant),
                whiten_gamma=float(core_gamma),
                recorder=stap_br_recorder,
            )
            if stap_detector_variant_norm == "hybrid_rescue":
                _pd_rescue, rescue_scores, rescue_info = _run_stap_once(
                    detector_variant="unwhitened_ratio",
                    whiten_gamma=0.0,
                    recorder=None,
                )
                feature_name = str(hybrid_rule_cfg["feature"]) if hybrid_rule_cfg is not None else "base_guard_frac_map"
                feature_map = {
                    "base_guard_frac_map": base_guard_frac_map,
                    "base_m_alias_map": base_m_alias_map,
                    "base_band_ratio_map": base_band_ratio_map,
                    "base_peak_freq_map": base_peak_freq_map,
                }.get(feature_name)
                if feature_map is None:
                    raise ValueError(f"Unsupported hybrid rescue feature map {feature_name!r}")
                stap_scores, hybrid_rescue_mask, hybrid_stats = kw._apply_hybrid_rescue_score_map(
                    stap_scores,
                    rescue_scores,
                    feature_name=feature_name,
                    feature_map=feature_map,
                    direction=str(hybrid_rule_cfg["direction"] if hybrid_rule_cfg is not None else ">="),
                    threshold=float(hybrid_rule_cfg["threshold"] if hybrid_rule_cfg is not None else 0.0),
                )
                stap_info["hybrid_rescue"] = {
                    "enabled": True,
                    "rule": dict(hybrid_rule_cfg or {}),
                    "stats": hybrid_stats,
                    "advanced_detector_variant": "msd_ratio",
                    "advanced_whiten_gamma": float(stap_whiten_gamma),
                    "rescue_detector_variant": "unwhitened_ratio",
                    "rescue_whiten_gamma": 0.0,
                    "rescue_condR_median": rescue_info.get("median_condR"),
                    "rescue_cond_loaded_median": rescue_info.get("median_cond_loaded"),
                }
                stap_info["detector_variant_effective"] = str(hybrid_variant_label or "hybrid_rescue")
                stap_info["hybrid_rescue_mask_fraction"] = hybrid_stats.get("rescue_fraction")
        tile_metric_maps = (
            stap_info.pop("_tile_metric_maps", {}) if isinstance(stap_info, dict) else {}
        )
        tile_metric_map_errors = (
            stap_info.pop("_tile_metric_map_errors", {}) if isinstance(stap_info, dict) else {}
        )
        if isinstance(stap_info, dict):
            stap_info["tile_metric_map_keys"] = (
                sorted(tile_metric_maps.keys()) if isinstance(tile_metric_maps, dict) else []
            )
            if isinstance(tile_metric_map_errors, dict) and tile_metric_map_errors:
                stap_info["tile_metric_map_errors"] = dict(tile_metric_map_errors)
            if isinstance(tile_metric_maps, dict):
                for _k, _v in tile_metric_maps.items():
                    try:
                        stap_info[f"tile_metric_{_k}_finite"] = int(
                            np.isfinite(np.asarray(_v, dtype=np.float32)).sum()
                        )
                    except Exception:
                        stap_info[f"tile_metric_{_k}_finite"] = None
        if isinstance(tile_metric_maps, dict) and tile_metric_maps:
            try:
                stap_cond_loaded_map = np.asarray(tile_metric_maps.get("cond_loaded"), dtype=np.float32)
                stap_cov_rank_proxy_map = np.asarray(
                    tile_metric_maps.get("cov_rank_proxy"), dtype=np.float32
                )
                stap_flow_mu_ratio_map = np.asarray(
                    tile_metric_maps.get("flow_mu_ratio"), dtype=np.float32
                )
                stap_bg_var_inflation_map = np.asarray(
                    tile_metric_maps.get("bg_var_inflation"), dtype=np.float32
                )
                stap_gamma_flow_map = np.asarray(
                    tile_metric_maps.get("gamma_flow"), dtype=np.float32
                )
                stap_gamma_perp_map = np.asarray(
                    tile_metric_maps.get("gamma_perp"), dtype=np.float32
                )
            except Exception:
                stap_cond_loaded_map = np.full((H, W), np.nan, dtype=np.float32)
                stap_cov_rank_proxy_map = np.full((H, W), np.nan, dtype=np.float32)
                stap_flow_mu_ratio_map = np.full((H, W), np.nan, dtype=np.float32)
                stap_bg_var_inflation_map = np.full((H, W), np.nan, dtype=np.float32)
                stap_gamma_flow_map = np.full((H, W), np.nan, dtype=np.float32)
                stap_gamma_perp_map = np.full((H, W), np.nan, dtype=np.float32)
        # _stap_pd may return ndarray-valued debug/gate payloads that are saved
        # separately in the main k-Wave acceptance harness. Drop them here to keep
        # meta.json JSON-serializable.
        stap_info.pop("_gate_mask_flow", None)
        stap_info.pop("_gate_mask_bg", None)
        stap_info["band_ratio_bands_hz"] = dict(band_ratio_spec)
        stap_info["detector_variant"] = str(stap_detector_variant_norm)
        stap_info["whiten_gamma"] = float(stap_whiten_gamma)
        stap_info["band_ratio_mode_requested"] = band_ratio_mode_norm
        stap_info["band_ratio_mode_effective"] = band_ratio_mode_norm
        stap_info["band_ratio_flavor"] = "whitened_mt_logratio" if use_whitened_ratio else "legacy"
        stap_info["stap_input_cube"] = "baseline_filtered"
        mt_meta = {"tapers": int(psd_tapers), "bandwidth": float(psd_bandwidth)}
        stap_info["band_ratio_mt_params"] = mt_meta
        baseline_telemetry.setdefault("band_ratio_mt_params", mt_meta)
        stap_info["stap_ms"] = float(1000.0 * (time.time() - t_stap_start))
        stap_info["flow_mask_stats"] = flow_mask_stats
        stap_info["stap_conditional_enable"] = bool(stap_conditional_enable)
        stap_tile_scores, stap_br_stats = stap_br_recorder.finalize()
        stap_info["band_ratio_br_stats"] = stap_br_stats or {"count": 0}
        if stap_tile_scores.size == tile_count:
            stap_band_ratio_map = kw._tile_scores_to_map(
                stap_tile_scores, (H, W), tile_hw, tile_stride
            )
        else:
            stap_band_ratio_map = np.ones((H, W), dtype=np.float32)
    else:
        pd_stap = pd_base.astype(np.float32, copy=False)
        stap_scores = np.zeros_like(pd_base, dtype=np.float32)
        stap_band_ratio_map = np.ones((H, W), dtype=np.float32)
        stap_info = {
            "stap_ms": 0.0,
            "flow_mask_stats": flow_mask_stats,
            "stap_conditional_enable": bool(stap_conditional_enable),
            "detector_variant": str(stap_detector_variant_norm),
            "whiten_gamma": float(stap_whiten_gamma),
            "band_ratio_bands_hz": dict(band_ratio_spec),
            "band_ratio_mode_requested": band_ratio_mode_norm,
            "band_ratio_mode_effective": band_ratio_mode_norm,
            "band_ratio_flavor": "whitened_mt_logratio",
            "band_ratio_br_stats": {"count": 0},
            "stap_skipped": True,
            "stap_input_cube": "baseline_filtered",
        }

    score_mode_norm = str(score_mode or "pd").strip().lower()
    if score_mode_norm not in {"pd", "msd", "band_ratio"}:
        raise ValueError("score_mode must be 'pd', 'msd', or 'band_ratio'")
    if score_mode_norm == "pd":
        stap_score_pool_map = pd_stap
    elif score_mode_norm == "band_ratio":
        stap_score_pool_map = stap_band_ratio_map
    else:
        stap_score_pool_map = stap_scores

    # ---- Contract v2 telemetry (label-free) ----
    ka_contract_v2_report: dict[str, Any] | None = None
    ka_contract_v2_inputs: dict[str, Any] | None = None
    if kw.evaluate_ka_contract_v2 is not None:
        tile_cov_flow, tile_coords = kw._tile_coverages(mask_flow, tile_hw, tile_stride)
        th, tw = tile_hw
        if score_mode_norm == "pd":
            score_map_for_contract = pd_stap  # right-tail (higher = more flow evidence)
        elif score_mode_norm == "band_ratio":
            score_map_for_contract = stap_band_ratio_map
        else:
            score_map_for_contract = stap_scores
        s_base_tiles = np.zeros(tile_cov_flow.shape[0], dtype=np.float32)
        for idx, (y0, x0) in enumerate(tile_coords):
            s_base_tiles[idx] = float(np.mean(score_map_for_contract[y0 : y0 + th, x0 : x0 + tw]))
        m_alias_tiles = (-np.asarray(base_tile_scores, dtype=np.float32)).astype(np.float32, copy=False)
        r_guard_tiles = np.asarray(base_br_recorder.rg_raw, dtype=np.float32)
        valid_tiles = np.asarray(base_br_recorder.tile_filled, dtype=bool)
        peak_freq_tiles = np.asarray(base_br_recorder.peak_freqs, dtype=np.float32)
        pf_peak_tiles = (peak_freq_tiles >= float(br_spec.flow_low_hz)) & (
            peak_freq_tiles <= float(br_spec.flow_high_hz)
        )
        ka_contract_v2_inputs = {
            "tile_coords": tile_coords,
            "tile_cov_flow": tile_cov_flow,
            "s_base_tiles": s_base_tiles,
            "m_alias_tiles": m_alias_tiles,
            "r_guard_tiles": r_guard_tiles,
            "pf_peak_tiles": pf_peak_tiles.astype(bool),
            "valid_tiles": valid_tiles,
        }
        ka_contract_v2_report = kw.evaluate_ka_contract_v2(
            s_base=s_base_tiles,
            m_alias=m_alias_tiles,
            r_guard=r_guard_tiles,
            pf_peak=pf_peak_tiles,
            c_flow=tile_cov_flow,
            valid_mask=valid_tiles,
        )
        # Band-sanity sentinel for short slow-time sequences: refuse to declare
        # C1/C2 eligibility when the requested bands collapse to DC-only bins or
        # overlap after discrete binning.
        try:
            br_stats = baseline_telemetry.get("band_ratio_stats") or {}
            reasons: list[str] = []
            if isinstance(br_stats, dict):
                if br_stats.get("br_bins_overlap"):
                    reasons.append("bands_overlap")
                if int(br_stats.get("br_flow_bins_nodc") or 0) <= 0:
                    reasons.append("flow_band_empty_nodc")
                if int(br_stats.get("br_alias_bins_nodc") or 0) <= 0:
                    reasons.append("alias_band_empty_nodc")
            if reasons:
                ka_contract_v2_report = {
                    "state": "C0_OFF",
                    "reason": "band_sanity_invalid",
                    "metrics": {"band_sanity_reasons": reasons},
                }
        except Exception:
            pass
        try:
            stap_info["ka_contract_v2_state"] = str(ka_contract_v2_report.get("state"))
            stap_info["ka_contract_v2_reason"] = str(ka_contract_v2_report.get("reason"))
            ka_metrics = ka_contract_v2_report.get("metrics", {}) or {}
            for key in (
                "iqr_alias_bg",
                "guard_q90",
                "delta_bg_flow_median",
                "delta_tail",
                "p_shrink",
                "uplift_eligible",
            ):
                stap_info[f"ka_contract_v2_{key}"] = ka_metrics.get(key)
        except Exception:
            pass

    # ---- Optional score-space KA v2 (shrink-only, contract-driven) ----
    # This modifies pd_stap only (PD mode) by shrinking PD on high-risk tiles so
    # that the PD score S=PD is shrunk. High-confidence flow pixels are explicitly
    # protected.
    ka_gate_map: np.ndarray | None = None
    ka_scale_map: np.ndarray | None = None
    pd_stap_pre_ka: np.ndarray | None = None
    score_stap_preka_map = stap_scores.astype(np.float32, copy=False)
    score_stap_map = score_stap_preka_map
    if score_ka_v2_enable and kw.derive_score_shrink_v2_tile_scales is not None:
        stap_info["score_ka_v2_requested"] = True
        if ka_contract_v2_report is None or ka_contract_v2_inputs is None:
            stap_info["score_ka_v2_disabled_reason"] = "missing_contract_v2"
        elif score_mode_norm not in {"pd", "msd"}:
            stap_info["score_ka_v2_disabled_reason"] = "score_mode_not_supported"
        else:
            state = str(ka_contract_v2_report.get("state") or "C0_OFF")
            reason = str(ka_contract_v2_report.get("reason") or "")
            stap_info["score_ka_v2_state"] = state
            stap_info["score_ka_v2_contract_reason"] = reason
            if reason != "ok":
                stap_info["score_ka_v2_disabled_reason"] = f"contract_{state}:{reason}"
            else:
                mode_norm = str(score_ka_v2_mode or "safety").strip().lower()
                if mode_norm not in {"safety", "uplift", "auto"}:
                    stap_info["score_ka_v2_disabled_reason"] = "invalid_score_ka_v2_mode"
                else:
                    apply_mode: str | None = None
                    if state == "C1_SAFETY" and mode_norm in {"safety", "auto"}:
                        apply_mode = "safety"
                    elif state == "C2_UPLIFT" and mode_norm in {"uplift", "auto"}:
                        apply_mode = "uplift"
                    else:
                        stap_info["score_ka_v2_disabled_reason"] = f"mode_mismatch_{mode_norm}:{state}"

                    if apply_mode is not None:
                        ka_metrics = ka_contract_v2_report.get("metrics", {}) or {}
                        risk_mode = str(ka_metrics.get("risk_mode") or "alias").strip().lower()
                        stap_info["score_ka_v2_risk_mode"] = risk_mode
                        if risk_mode == "guard":
                            risk_tiles = ka_contract_v2_inputs["r_guard_tiles"]
                        else:
                            risk_tiles = ka_contract_v2_inputs["m_alias_tiles"]
                        shrink = kw.derive_score_shrink_v2_tile_scales(
                            report=ka_contract_v2_report,
                            s_base=ka_contract_v2_inputs["s_base_tiles"],
                            m_alias=risk_tiles,
                            c_flow=ka_contract_v2_inputs["tile_cov_flow"],
                            valid_mask=ka_contract_v2_inputs["valid_tiles"],
                            mode=apply_mode,
                        )
                        if not shrink.get("apply"):
                            stap_info["score_ka_v2_disabled_reason"] = str(
                                shrink.get("reason") or "unknown"
                            )
                        else:
                            scale_tiles = np.asarray(shrink["scale_tiles"], dtype=np.float32)
                            gated_tiles = np.asarray(shrink["gated_tiles"], dtype=bool)
                            tile_coords = ka_contract_v2_inputs["tile_coords"]
                            th, tw = tile_hw

                            # Build pixel maps by averaging overlaps, then restrict
                            # the action to the union of gated tiles.
                            scale_map = kw._tile_scores_to_map(scale_tiles, (H, W), tile_hw, tile_stride)
                            gate_union = np.zeros((H, W), dtype=bool)
                            for idx, (y0, x0) in enumerate(tile_coords):
                                if idx < gated_tiles.size and gated_tiles[idx]:
                                    gate_union[y0 : y0 + th, x0 : x0 + tw] = True
                            scale_final = np.ones_like(scale_map, dtype=np.float32)
                            scale_final[gate_union] = scale_map[gate_union].astype(np.float32, copy=False)

                            # Protected pixels: never modify flow mask and, optionally,
                            # extreme-score pixels (configurable in KaContractV2Config).
                            if score_mode_norm == "pd":
                                score_pre = pd_stap.astype(np.float32, copy=False)
                            else:
                                score_pre = score_stap_preka_map
                            s_base_pix = score_pre
                            # IMPORTANT: never mutate `mask_flow` (evaluation mask).
                            prot_pix = np.asarray(mask_flow, dtype=bool).copy()
                            cfg = ka_contract_v2_report.get("config") or {}
                            protect_hi_by_score = bool(cfg.get("protect_hi_by_score", True))
                            if protect_hi_by_score:
                                q_hi = float(cfg.get("q_hi_protect", 0.99999))
                                finite = np.isfinite(s_base_pix)
                                if finite.any():
                                    thr_hi = float(np.quantile(s_base_pix[finite], q_hi))
                                    prot_pix |= s_base_pix >= thr_hi
                            scale_final[prot_pix] = 1.0

                            score_post = (score_pre / np.maximum(scale_final, 1e-12)).astype(
                                np.float32, copy=False
                            )
                            if score_mode_norm == "pd":
                                pd_stap_pre_ka = score_pre
                                pd_stap = score_post
                                stap_score_pool_map = pd_stap
                            else:
                                score_stap_map = score_post
                                stap_score_pool_map = score_stap_map
                            ka_scale_map = scale_final
                            ka_gate_map = gate_union

                            stap_info["score_ka_v2_applied"] = True
                            stap_info["score_ka_v2_mode_applied"] = apply_mode
                            stap_info["score_ka_v2_stats"] = shrink.get("stats", {})
                            scaled = scale_final > (1.0 + 1e-6)
                            stap_info["score_ka_v2_scaled_pixel_fraction"] = float(np.mean(scaled))
                            if scaled.any():
                                vals = scale_final[scaled].astype(np.float64, copy=False)
                                stap_info["score_ka_v2_scale_p50"] = float(np.median(vals))
                                stap_info["score_ka_v2_scale_p90"] = float(np.quantile(vals, 0.90))
                                stap_info["score_ka_v2_scale_max"] = float(np.max(vals))

    # ---- Save maps ----
    _save("pd_base", pd_base.astype(np.float32, copy=False))
    _save("pd_stap", pd_stap.astype(np.float32, copy=False))
    _save("score_pd_base", (pd_base).astype(np.float32, copy=False))
    _save("score_pd_stap", (pd_stap).astype(np.float32, copy=False))
    # Score vNext: primary detector scores (right tail).
    _save("score_base", pd_base.astype(np.float32, copy=False))
    if score_base_pdlog is not None:
        _save("score_base_pdlog", score_base_pdlog.astype(np.float32, copy=False))
    if score_base_kasai is not None:
        _save("score_base_kasai", score_base_kasai.astype(np.float32, copy=False))
    _save("score_stap_preka", score_stap_preka_map.astype(np.float32, copy=False))
    _save("score_stap", score_stap_map.astype(np.float32, copy=False))
    if hybrid_rescue_mask is not None:
        _save("hybrid_rescue_mask", hybrid_rescue_mask.astype(np.bool_, copy=False))
    if stap_detector_variant_norm == "msd_ratio":
        if stap_whiten_gamma <= 1e-8:
            score_name_text = (
                "stap_score_v1: unwhitened flow matched-subspace ratio (gamma=0; right tail)\n"
            )
            score_stap_definition_vnext = (
                "score_stap_preka is the matched-subspace ratio computed without covariance "
                "whitening (fractional whitening exponent gamma=0), using the same fixed Pf "
                "subspace projection; score_stap equals score_stap_preka unless a score-space "
                "KA veto is applied."
            )
        elif stap_whiten_gamma >= 1.0 - 1e-8:
            score_name_text = "stap_score_v1: whitened flow matched-subspace ratio (right tail)\n"
            score_stap_definition_vnext = (
                "score_stap_preka is the STAP matched-subspace ratio computed from the "
                "baseline-filtered IQ cube via local covariance whitening and fixed Pf "
                "subspace projection; score_stap equals score_stap_preka unless a "
                "score-space KA veto is applied."
            )
        else:
            score_name_text = (
                f"stap_score_v1: fractionally whitened flow matched-subspace ratio (gamma={stap_whiten_gamma:.2f}; right tail)\n"
            )
            score_stap_definition_vnext = (
                "score_stap_preka is a matched-subspace ratio computed with fractional "
                f"covariance whitening exponent gamma={stap_whiten_gamma:.3f}, interpolating "
                "between the unwhitened detector (gamma=0) and the full STAP score (gamma=1); "
                "score_stap equals score_stap_preka unless a score-space KA veto is applied."
            )
    elif stap_detector_variant_norm == "whitened_power":
        if stap_whiten_gamma >= 1.0 - 1e-8:
            score_name_text = (
                "stap_score_v1: whitened total power (no band partition; right tail)\n"
            )
            score_stap_definition_vnext = (
                "score_stap_preka is a detector ablation: total whitened slow-time power "
                "computed by whitening snapshots with local R^{-1/2} and averaging ||y||^2 "
                "(no Doppler band partition or Pf projection); score_stap equals score_stap_preka "
                "unless a score-space KA veto is applied."
            )
        else:
            score_name_text = (
                f"stap_score_v1: fractionally whitened total power (gamma={stap_whiten_gamma:.2f}; right tail)\n"
            )
            score_stap_definition_vnext = (
                "score_stap_preka is a detector ablation: total slow-time power after fractional "
                f"covariance whitening exponent gamma={stap_whiten_gamma:.3f} (no Doppler band "
                "partition or Pf projection); score_stap equals score_stap_preka unless a "
                "score-space KA veto is applied."
            )
    elif stap_detector_variant_norm == "unwhitened_ratio":
        score_name_text = (
            "stap_score_v1: unwhitened flow matched-subspace ratio (no whitening; right tail)\n"
        )
        score_stap_definition_vnext = (
            "score_stap_preka is a detector ablation: flow-band matched-subspace ratio "
            "computed without covariance whitening (R=I), using the same fixed Pf "
            "subspace projection; score_stap equals score_stap_preka unless a "
            "score-space KA veto is applied."
        )
    else:
        rule_name = str((hybrid_rule_cfg or {}).get("name", "guard_frac_v1"))
        if hybrid_variant_label == "adaptive_guard":
            score_name_text = (
                f"stap_score_v1: adaptive guard-promoted matched-subspace score ({rule_name}; right tail)\n"
            )
            score_stap_definition_vnext = (
                "score_stap_preka is an adaptive detector-family score: the unwhitened flow-band "
                "matched-subspace ratio is used by default, with promotion onto the advanced "
                "Huber-whitened matched-subspace score on pixels selected by a fixed baseline "
                "guard-energy rule. score_stap equals score_stap_preka unless a score-space "
                "KA veto is applied."
            )
        else:
            score_name_text = (
                f"stap_score_v1: hybrid rescue matched-subspace score ({rule_name}; right tail)\n"
            )
            score_stap_definition_vnext = (
                "score_stap_preka is a hybrid detector-family score: the advanced Huber-whitened "
                "matched-subspace score is used by default, with an unwhitened matched-subspace "
                "rescue branch on pixels selected by a fixed baseline-feature rule. "
                "score_stap equals score_stap_preka unless a score-space KA veto is applied."
            )
    _save_text("score_name", "score_name.txt", score_name_text)
    if pd_stap_pre_ka is not None:
        _save("pd_stap_pre_ka", pd_stap_pre_ka.astype(np.float32, copy=False))
        _save("score_pd_stap_pre_ka", (pd_stap_pre_ka).astype(np.float32, copy=False))
    _save("mask_flow", mask_flow.astype(np.bool_, copy=False))
    _save("mask_bg", mask_bg.astype(np.bool_, copy=False))
    _save("base_band_ratio_map", base_band_ratio_map.astype(np.float32, copy=False))
    _save("base_m_alias_map", base_m_alias_map.astype(np.float32, copy=False))
    _save("base_guard_frac_map", base_guard_frac_map.astype(np.float32, copy=False))
    _save("base_peak_freq_map", base_peak_freq_map.astype(np.float32, copy=False))
    _save("base_peak_bin_map", base_peak_bin_map.astype(np.int16, copy=False))
    _save("stap_band_ratio_map", stap_band_ratio_map.astype(np.float32, copy=False))
    _save("stap_cond_loaded_map", stap_cond_loaded_map.astype(np.float32, copy=False))
    _save("stap_cov_rank_proxy_map", stap_cov_rank_proxy_map.astype(np.float32, copy=False))
    _save("stap_flow_mu_ratio_map", stap_flow_mu_ratio_map.astype(np.float32, copy=False))
    _save("stap_bg_var_inflation_map", stap_bg_var_inflation_map.astype(np.float32, copy=False))
    _save("stap_gamma_flow_map", stap_gamma_flow_map.astype(np.float32, copy=False))
    _save("stap_gamma_perp_map", stap_gamma_perp_map.astype(np.float32, copy=False))
    _save("base_score_map", pd_base.astype(np.float32, copy=False))
    _save("stap_score_map", stap_scores.astype(np.float32, copy=False))
    _save("stap_score_pool_map", stap_score_pool_map.astype(np.float32, copy=False))
    if ka_scale_map is not None:
        _save("ka_scale_map", ka_scale_map.astype(np.float32, copy=False))
    if ka_gate_map is not None:
        _save("ka_gate_map", ka_gate_map.astype(np.bool_, copy=False))

    telemetry_combined = dict(baseline_telemetry or {})
    telemetry_combined.update(stap_info)
    telemetry_combined["stap_detector_variant"] = str(stap_detector_variant_norm)
    telemetry_combined["stap_whiten_gamma"] = float(stap_whiten_gamma)
    telemetry_combined["stap_cov_train_trim_q"] = float(stap_cov_train_trim_q)
    if hybrid_rule_cfg is not None:
        telemetry_combined["hybrid_rescue_rule"] = dict(hybrid_rule_cfg)

    def _score_proxy_tail_stats(arr: np.ndarray) -> dict[str, float | int | None]:
        arr = np.asarray(arr, dtype=np.float64)
        flow_vals = arr[np.asarray(mask_flow, dtype=bool)]
        bg_vals = arr[np.asarray(mask_bg, dtype=bool)]
        flow_vals = flow_vals[np.isfinite(flow_vals)]
        bg_vals = bg_vals[np.isfinite(bg_vals)]
        out: dict[str, float | int | None] = {
            "n_flow": int(flow_vals.size),
            "n_bg": int(bg_vals.size),
            "flow_q50": float(np.quantile(flow_vals, 0.50)) if flow_vals.size else None,
            "flow_q90": float(np.quantile(flow_vals, 0.90)) if flow_vals.size else None,
            "bg_q99": float(np.quantile(bg_vals, 0.99)) if bg_vals.size else None,
            "bg_q999": float(np.quantile(bg_vals, 0.999)) if bg_vals.size else None,
        }
        if flow_vals.size and bg_vals.size:
            neg_sorted = np.sort(bg_vals)
            less = np.searchsorted(neg_sorted, flow_vals, side="left")
            right = np.searchsorted(neg_sorted, flow_vals, side="right")
            equal = right - less
            out["auc_flow_bg"] = float(
                (float(np.sum(less)) + 0.5 * float(np.sum(equal))) / float(flow_vals.size * bg_vals.size)
            )
            thr = float(np.quantile(bg_vals, 0.999))
            out["thr_fpr1e3"] = thr
            out["flow_hit_fpr1e3"] = float(np.mean(flow_vals >= thr))
            out["flow_margin_q50_fpr1e3"] = (
                None if out["flow_q50"] is None else float(out["flow_q50"]) - thr
            )
            out["flow_margin_q90_fpr1e3"] = (
                None if out["flow_q90"] is None else float(out["flow_q90"]) - thr
            )
        else:
            out["auc_flow_bg"] = None
            out["thr_fpr1e3"] = None
            out["flow_hit_fpr1e3"] = None
            out["flow_margin_q50_fpr1e3"] = None
            out["flow_margin_q90_fpr1e3"] = None
        return out

    proxy_score_tail_stats: dict[str, dict[str, float | int | None]] = {
        "base_pd": _score_proxy_tail_stats(pd_base.astype(np.float32, copy=False)),
        "stap_preka": _score_proxy_tail_stats(score_stap_preka_map.astype(np.float32, copy=False)),
        "stap_post": _score_proxy_tail_stats(score_stap_map.astype(np.float32, copy=False)),
    }
    if score_base_pdlog is not None:
        proxy_score_tail_stats["base_pdlog"] = _score_proxy_tail_stats(
            score_base_pdlog.astype(np.float32, copy=False)
        )
    if score_base_kasai is not None:
        proxy_score_tail_stats["base_kasai"] = _score_proxy_tail_stats(
            score_base_kasai.astype(np.float32, copy=False)
        )
    telemetry_combined["proxy_score_tail_stats"] = proxy_score_tail_stats

    # ---- Meta ----
    meta: dict[str, Any] = {
        "dataset": {
            "name": dataset_name,
            "format": "icube",
        },
        "seed": int(seed),
        "prf_hz": float(prf_hz),
        "total_frames": int(Icube.shape[0]),
        "Lt": int(Lt),
        "run_stap": bool(run_stap),
        "tile_hw": [int(tile_hw[0]), int(tile_hw[1])],
        "tile_stride": int(tile_stride),
        "score_stats": {
            "mode": str((score_mode or "pd").strip().lower()),
            "stap_detector_variant": str(stap_detector_variant_norm),
            "stap_whiten_gamma": float(stap_whiten_gamma),
            "hybrid_rescue_rule": (dict(hybrid_rule_cfg) if hybrid_rule_cfg is not None else None),
        },
        "stap_conditional_enable": bool(stap_conditional_enable),
        "score_vnext": {
            "score_files": {
                "base": "score_base.npy",
                "base_pdlog": "score_base_pdlog.npy" if score_base_pdlog is not None else None,
                "base_kasai": "score_base_kasai.npy" if score_base_kasai is not None else None,
                "stap_pre_ka": "score_stap_preka.npy",
                "stap": "score_stap.npy",
            },
            "score_name_file": "score_name.txt",
            "score_convention": "right_tail (higher = more flow evidence)",
            "score_base_definition": (
                "score_base is the baseline PD score map (right-tail); optional "
                "baseline extras may include score_base_pdlog (log power Doppler) "
                "and score_base_kasai (log |lag-1 autocorr|, Kasai power)."
            ),
            "score_stap_definition": score_stap_definition_vnext,
            "stap_detector_variant": str(stap_detector_variant_norm),
            "stap_whiten_gamma": float(stap_whiten_gamma),
            "hybrid_rescue_rule": (dict(hybrid_rule_cfg) if hybrid_rule_cfg is not None else None),
        },
        "pd_mode": {
            "pd_files": {
                "base": "pd_base.npy",
                "stap": "pd_stap.npy",
                "stap_pre_ka": "pd_stap_pre_ka.npy" if pd_stap_pre_ka is not None else None,
            },
            "score_files": {
                "base": "score_pd_base.npy",
                "stap": "score_pd_stap.npy",
                "stap_pre_ka": (
                    "score_pd_stap_pre_ka.npy" if pd_stap_pre_ka is not None else None
                ),
            },
            "roc_convention": "right_tail_on_pd (equivalently score_pd=pd)",
        },
        "baseline_stats": baseline_telemetry,
        "stap_fallback_telemetry": telemetry_combined,
        "ka_contract_v2": ka_contract_v2_report,
    }
    if meta_extra:
        meta.update(meta_extra)

    meta_path = bundle_dir / "meta.json"
    meta_path.write_text(json.dumps(meta, indent=2))
    paths["meta"] = str(meta_path)
    return paths
