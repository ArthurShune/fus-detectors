from __future__ import annotations

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
    Lt: int = 8,
    diag_load: float = 0.07,
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
      - score_pd_base.npy / score_pd_stap.npy   (explicit right-tail scores S=-pd)
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

    # ---- Band-ratio spec ----
    feas_mode = kw._normalize_feasibility_mode(feasibility_mode)
    band_ratio_mode_norm = (band_ratio_mode or "legacy").strip().lower()
    if band_ratio_mode_norm not in {"legacy", "whitened"}:
        raise ValueError("band_ratio_mode must be 'legacy' or 'whitened'")
    use_whitened_ratio = band_ratio_mode_norm == "whitened"
    if not use_whitened_ratio:
        raise ValueError("This entrypoint is intended for whitened telemetry (mc_svd residual).")
    band_ratio_spec = {
        "flow_low_hz": float(band_ratio_flow_low_hz),
        "flow_high_hz": float(band_ratio_flow_high_hz),
        "alias_center_hz": float(band_ratio_alias_center_hz),
        "alias_width_hz": float(band_ratio_alias_width_hz),
    }
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
    else:
        raise ValueError(
            f"Unsupported baseline_type={baseline_type_norm!r} for IQ bundle writer. "
            "Use mc_svd or svd_bandpass."
        )

    H, W = pd_base.shape
    tile_count = kw._tile_count((H, W), tile_hw, tile_stride)

    # ---- Masks ----
    mask_flow_default, mask_bg_default = _default_masks_generic(H, W)
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

    # ---- STAP PD + optional STAP band-ratio recorder ----
    if run_stap:
        stap_br_recorder = kw.BandRatioRecorder(
            prf_hz,
            tile_count,
            br_spec,
            tapers=psd_tapers,
            bandwidth=psd_bandwidth,
        )
        t_stap_start = time.time()
        pd_stap, stap_scores, stap_info = kw._stap_pd(
            cube_for_stap,
            tile_hw=tile_hw,
            stride=tile_stride,
            Lt=Lt,
            prf_hz=prf_hz,
            diag_load=diag_load,
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
            msd_ridge=msd_ridge,
            msd_agg_mode=msd_agg_mode,
            msd_ratio_rho=msd_ratio_rho,
            motion_half_span_rel=motion_half_span_rel,
            msd_contrast_alpha=msd_contrast_alpha,
            stap_device=kw._resolve_stap_device(stap_device),
            tile_batch=192 if kw._resolve_stap_device(stap_device).startswith("cuda") else 1,
            pd_base_full=pd_base,
            mask_flow=mask_flow,
            mask_bg=mask_bg,
            ka_mode="none",
            ka_prior_library=None,
            ka_opts=None,
            alias_psd_select_enable=alias_psd_select_enable,
            alias_psd_select_ratio_thresh=alias_psd_select_ratio_thresh,
            alias_psd_select_bins=alias_psd_select_bins,
            psd_telemetry=False,
            psd_tapers=psd_tapers,
            psd_bandwidth=psd_bandwidth,
            band_ratio_recorder=stap_br_recorder,
            feasibility_mode=feas_mode,
            band_ratio_spec=dict(band_ratio_spec),
        )
        # _stap_pd may return ndarray-valued debug/gate payloads that are saved
        # separately in the main k-Wave acceptance harness. Drop them here to keep
        # meta.json JSON-serializable.
        stap_info.pop("_gate_mask_flow", None)
        stap_info.pop("_gate_mask_bg", None)
        stap_info["band_ratio_bands_hz"] = dict(band_ratio_spec)
        stap_info["band_ratio_mode_requested"] = band_ratio_mode_norm
        stap_info["band_ratio_mode_effective"] = band_ratio_mode_norm
        stap_info["band_ratio_flavor"] = "whitened_mt_logratio" if use_whitened_ratio else "legacy"
        stap_info["stap_input_cube"] = "baseline_filtered"
        mt_meta = {"tapers": int(psd_tapers), "bandwidth": float(psd_bandwidth)}
        stap_info["band_ratio_mt_params"] = mt_meta
        baseline_telemetry.setdefault("band_ratio_mt_params", mt_meta)
        stap_info["stap_ms"] = float(1000.0 * (time.time() - t_stap_start))
        stap_info["flow_mask_stats"] = flow_mask_stats
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
            "band_ratio_bands_hz": dict(band_ratio_spec),
            "band_ratio_mode_requested": band_ratio_mode_norm,
            "band_ratio_mode_effective": band_ratio_mode_norm,
            "band_ratio_flavor": "whitened_mt_logratio",
            "band_ratio_br_stats": {"count": 0},
            "stap_skipped": True,
            "stap_input_cube": "baseline_filtered",
        }

    # ---- Contract v2 telemetry (label-free) ----
    ka_contract_v2_report: dict[str, Any] | None = None
    ka_contract_v2_inputs: dict[str, Any] | None = None
    if kw.evaluate_ka_contract_v2 is not None:
        tile_cov_flow, tile_coords = kw._tile_coverages(mask_flow, tile_hw, tile_stride)
        th, tw = tile_hw
        score_map_for_contract = -pd_stap  # higher = more flow evidence (PD mode)
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
    # This modifies pd_stap only (PD mode) by inflating PD on high-risk tiles so
    # that a score defined as S=-PD is shrunk. High-confidence flow pixels are
    # explicitly protected.
    ka_gate_map: np.ndarray | None = None
    ka_scale_map: np.ndarray | None = None
    pd_stap_pre_ka: np.ndarray | None = None
    if score_ka_v2_enable and kw.derive_score_shrink_v2_tile_scales is not None:
        stap_info["score_ka_v2_requested"] = True
        if ka_contract_v2_report is None or ka_contract_v2_inputs is None:
            stap_info["score_ka_v2_disabled_reason"] = "missing_contract_v2"
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

                            # Protected pixels: never modify flow mask or extreme-score pixels.
                            pd_stap_orig = pd_stap.astype(np.float32, copy=False)
                            s_base_pix = -pd_stap_orig
                            # IMPORTANT: never mutate `mask_flow` (evaluation mask).
                            prot_pix = np.asarray(mask_flow, dtype=bool).copy()
                            cfg = ka_contract_v2_report.get("config") or {}
                            q_hi = float(cfg.get("q_hi_protect", 0.995))
                            finite = np.isfinite(s_base_pix)
                            if finite.any():
                                thr_hi = float(np.quantile(s_base_pix[finite], q_hi))
                                prot_pix |= s_base_pix >= thr_hi
                            scale_final[prot_pix] = 1.0

                            pd_stap_pre_ka = pd_stap_orig
                            pd_stap = (pd_stap_orig * scale_final).astype(np.float32, copy=False)
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
    _save("score_pd_base", (-pd_base).astype(np.float32, copy=False))
    _save("score_pd_stap", (-pd_stap).astype(np.float32, copy=False))
    if pd_stap_pre_ka is not None:
        _save("pd_stap_pre_ka", pd_stap_pre_ka.astype(np.float32, copy=False))
        _save("score_pd_stap_pre_ka", (-pd_stap_pre_ka).astype(np.float32, copy=False))
    _save("mask_flow", mask_flow.astype(np.bool_, copy=False))
    _save("mask_bg", mask_bg.astype(np.bool_, copy=False))
    _save("base_band_ratio_map", base_band_ratio_map.astype(np.float32, copy=False))
    _save("base_m_alias_map", base_m_alias_map.astype(np.float32, copy=False))
    _save("base_guard_frac_map", base_guard_frac_map.astype(np.float32, copy=False))
    _save("base_peak_freq_map", base_peak_freq_map.astype(np.float32, copy=False))
    _save("stap_band_ratio_map", stap_band_ratio_map.astype(np.float32, copy=False))
    _save("base_score_map", pd_base.astype(np.float32, copy=False))
    _save("stap_score_map", stap_scores.astype(np.float32, copy=False))
    _save("stap_score_pool_map", pd_stap.astype(np.float32, copy=False))
    if ka_scale_map is not None:
        _save("ka_scale_map", ka_scale_map.astype(np.float32, copy=False))
    if ka_gate_map is not None:
        _save("ka_gate_map", ka_gate_map.astype(np.bool_, copy=False))

    telemetry_combined = dict(baseline_telemetry or {})
    telemetry_combined.update(stap_info)

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
        "score_stats": {"mode": str((score_mode or "pd").strip().lower())},
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
            "roc_convention": "lower_tail_on_pd (equivalently right_tail_on_score=-pd)",
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
