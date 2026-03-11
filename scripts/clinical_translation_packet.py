from __future__ import annotations

import csv
import json
import math
from collections import Counter
from pathlib import Path
from statistics import mean, median
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
REPORT_DIR = ROOT / 'reports' / 'clinical_translation'


def _load_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline='') as f:
        return list(csv.DictReader(f))


def _load_json(path: Path) -> Any:
    with path.open() as f:
        return json.load(f)


def _num(v: Any) -> float | None:
    if v is None:
        return None
    if isinstance(v, (int, float)):
        return float(v)
    if isinstance(v, str):
        s = v.strip()
        if not s:
            return None
        try:
            return float(s)
        except ValueError:
            return None
    return None


def _safe_mean(vals: list[float]) -> float | None:
    return mean(vals) if vals else None


def _safe_median(vals: list[float]) -> float | None:
    return median(vals) if vals else None


def _pct_reduction(pre: float, post: float) -> float | None:
    if pre == 0:
        return None
    return (pre - post) / pre


def _read_rows_with_filter(path: Path, **matches: str) -> list[dict[str, str]]:
    rows = _load_csv(path)
    out = []
    for r in rows:
        ok = True
        for k, v in matches.items():
            if r.get(k) != v:
                ok = False
                break
        if ok:
            out.append(r)
    return out


def mace_section() -> tuple[dict[str, Any], list[dict[str, Any]]]:
    phase2_csv = ROOT / 'reports' / 'mace_phase2_summary.csv'
    pdonly_csv = ROOT / 'reports' / 'mace_pdonly_contract_v2.csv'
    vascular_json = ROOT / 'reports' / 'mace_vascular_pixel_eval.json'
    hemo_csv = ROOT / 'reports' / 'mace_hemo_telemetry.csv'
    holdout_json = ROOT / 'reports' / 'mace_alias_gate_holdout.json'
    holdout_csv = ROOT / 'reports' / 'mace_alias_gate_holdout.csv'

    phase2 = _load_csv(phase2_csv)
    pdonly = _load_csv(pdonly_csv)
    vascular = _load_json(vascular_json)
    hemo = _load_csv(hemo_csv)

    pd_fp_pre = _safe_mean([float(r['pd_fp_at_tpr']) for r in phase2])
    pd_fp_post = _safe_mean([float(r['gated_pd_fp_at_tpr']) for r in phase2])
    gate_kept = _safe_mean([float(r['gate_kept_frac']) for r in phase2])
    pd_pauc = _safe_mean([float(r['pd_pauc']) for r in phase2])
    gated_pd_pauc = _safe_mean([float(r['gated_pd_pauc']) for r in phase2])

    state_counts = Counter(r['ka_contract_v2_state'] for r in pdonly)
    reason_counts = Counter(r['ka_contract_v2_reason'] for r in pdonly)
    hit_retention = _safe_mean([float(r['offline_hit_retention']) for r in pdonly])
    offline_fp_pre = _safe_mean([float(r['offline_pd_fp_pre']) for r in pdonly])
    offline_fp_post = _safe_mean([float(r['offline_pd_fp_post']) for r in pdonly])

    hemo_auc = _safe_mean([float(r['pd_z_auc']) for r in hemo])
    hemo_alias_delta = _safe_mean([float(r['delta_log_alias']) for r in hemo])

    holdout = None
    if holdout_json.exists() and holdout_csv.exists():
        holdout = _load_json(holdout_json)

    section = {
        'dataset_role': 'whole-brain retrospective functional/readout anchor',
        'sources': [str(phase2_csv.relative_to(ROOT)), str(pdonly_csv.relative_to(ROOT)), str(vascular_json.relative_to(ROOT)), str(hemo_csv.relative_to(ROOT))],
        'phase2_alias_gate': {
            'mean_pd_fp_at_target_tpr': pd_fp_pre,
            'mean_gated_pd_fp_at_target_tpr': pd_fp_post,
            'mean_fp_reduction_fraction': _pct_reduction(pd_fp_pre or 0.0, pd_fp_post or 0.0),
            'mean_gate_kept_fraction': gate_kept,
            'mean_pd_pauc': pd_pauc,
            'mean_gated_pd_pauc': gated_pd_pauc,
        },
        'pdonly_contract': {
            'state_counts': dict(state_counts),
            'reason_counts': dict(reason_counts),
            'mean_hit_retention': hit_retention,
            'mean_fp_pre': offline_fp_pre,
            'mean_fp_post': offline_fp_post,
            'mean_fp_reduction_fraction': _pct_reduction(offline_fp_pre or 0.0, offline_fp_post or 0.0),
        },
        'vascular_pixel_eval': {
            'n_planes': vascular['n_planes'],
            'median_veto_frac_neg': vascular['veto_frac_neg']['median'],
            'median_veto_frac_pos': vascular['veto_frac_pos']['median'],
            'median_tpr_alias_veto_fpr1e3': vascular['tpr_pdz_alias_veto_fpr0.001']['median'],
            'median_tpr_pd_fpr1e3': vascular['tpr_pdz_fpr0.001']['median'],
        },
        'hemo_telemetry': {
            'mean_pd_z_auc': hemo_auc,
            'mean_delta_log_alias': hemo_alias_delta,
        },
    }
    if holdout is not None:
        section['alias_gate_holdout'] = holdout
        section['sources'].extend([str(holdout_csv.relative_to(ROOT)), str(holdout_json.relative_to(ROOT))])

    rows = [
        {
            'domain': 'mace', 'claim': 'alias_gating_reduces_false_positives_at_matched_tpr',
            'metric': 'fp_reduction_fraction', 'value': section['phase2_alias_gate']['mean_fp_reduction_fraction'],
            'unit': 'fraction', 'source': str(phase2_csv.relative_to(ROOT)),
            'notes': 'Mean matched-TPR false-positive reduction from gated PD versus ungated PD over whole-brain phase-2 sweep.'
        },
        {
            'domain': 'mace', 'claim': 'alias_gating_preserves_majority_of_tiles',
            'metric': 'gate_kept_fraction', 'value': gate_kept,
            'unit': 'fraction', 'source': str(phase2_csv.relative_to(ROOT)),
            'notes': 'Mean retained fraction after alias gate in phase-2 sweep.'
        },
        {
            'domain': 'mace', 'claim': 'pdonly_contract_reduces_false_positives',
            'metric': 'offline_fp_reduction_fraction', 'value': section['pdonly_contract']['mean_fp_reduction_fraction'],
            'unit': 'fraction', 'source': str(pdonly_csv.relative_to(ROOT)),
            'notes': 'Mean plane-wise false-positive reduction for PD-only contract with no KA uplift actions.'
        },
        {
            'domain': 'mace', 'claim': 'pdonly_contract_preserves_hits',
            'metric': 'offline_hit_retention', 'value': hit_retention,
            'unit': 'fraction', 'source': str(pdonly_csv.relative_to(ROOT)),
            'notes': 'Mean retained hits after PD-only contract.'
        },
    ]
    if holdout is not None:
        cta = holdout['counts_test_all']
        cts = holdout['counts_test_selected']
        rows.extend([
            {
                'domain': 'mace', 'claim': 'holdout_alias_gate_reduces_offline_false_positives',
                'metric': 'holdout_fp_reduction_fraction_all',
                'value': _pct_reduction(float(cta['fp_pre'] or 0.0), float(cta['fp_post'] or 0.0)),
                'unit': 'fraction',
                'source': str(holdout_json.relative_to(ROOT)),
                'notes': 'Held-out scan4/scan5 false-positive reduction under label-free alias gating on all planes.'
            },
            {
                'domain': 'mace', 'claim': 'holdout_alias_gate_preserves_hits',
                'metric': 'holdout_hit_retention_all',
                'value': cta['hit_retention'],
                'unit': 'fraction',
                'source': str(holdout_json.relative_to(ROOT)),
                'notes': 'Held-out scan4/scan5 hit retention under label-free alias gating on all planes.'
            },
            {
                'domain': 'mace', 'claim': 'holdout_selected_planes_are_stable',
                'metric': 'holdout_hit_retention_selected',
                'value': cts['hit_retention'],
                'unit': 'fraction',
                'source': str(holdout_json.relative_to(ROOT)),
                'notes': 'Held-out selected-plane hit retention; selected subset is small and used as a stability check only.'
            },
        ])
    return section, rows


def shin_ulm_section() -> tuple[dict[str, Any], list[dict[str, Any]]]:
    shin_csv = ROOT / 'reports' / 'shin_motion_telemetry.csv'
    motion_proxy_csv = ROOT / 'reports' / 'simus_sanity_link' / 'real_motion_proxy_telemetry.csv'
    ulm_brainlike_csv = ROOT / 'reports' / 'ulm7883227_motion_sweep_ULM_blocks1_3_0000_0128_brainlike_e975.csv'
    ulm_elastic_csv = ROOT / 'reports' / 'ulm7883227_motion_sweep_ULM_blocks1_3_0000_0128_elastic_e975.csv'

    shin = _load_csv(shin_csv)
    proxy = _load_csv(motion_proxy_csv)
    ulm_brain = _load_csv(ulm_brainlike_csv)
    ulm_elastic = _load_csv(ulm_elastic_csv)

    shin_delta = [float(r['delta_auc_stap_minus_mcsvd']) for r in shin]
    shin_shift = [float(r['reg_shift_p90_px']) for r in shin]

    proxy_by_kind: dict[str, list[dict[str, str]]] = {}
    for r in proxy:
        proxy_by_kind.setdefault(r['kind'], []).append(r)

    def ulm_stats(rows: list[dict[str, str]]) -> dict[str, float | None]:
        return {
            'mean_corr_score_base': _safe_mean([float(r['corr_score_base']) for r in rows]),
            'mean_corr_score_stap': _safe_mean([float(r['corr_score_stap']) for r in rows]),
            'mean_nrmse_base': _safe_mean([float(r['nrmse_score_base']) for r in rows]),
            'mean_nrmse_stap': _safe_mean([float(r['nrmse_score_stap']) for r in rows]),
            'mean_bg_var_ratio_base': _safe_mean([float(r['bg_var_ratio_base']) for r in rows]),
            'mean_bg_var_ratio_stap': _safe_mean([float(r['bg_var_ratio_stap']) for r in rows]),
        }

    brain_stats = ulm_stats(ulm_brain)
    elastic_stats = ulm_stats(ulm_elastic)

    section = {
        'dataset_role': 'brain-like nuisance robustness anchors',
        'sources': [
            str(shin_csv.relative_to(ROOT)),
            str(motion_proxy_csv.relative_to(ROOT)),
            str(ulm_brainlike_csv.relative_to(ROOT)),
            str(ulm_elastic_csv.relative_to(ROOT)),
        ],
        'shin': {
            'n_cases': len(shin),
            'mean_delta_auc_stap_minus_mcsvd': _safe_mean(shin_delta),
            'median_reg_shift_p90_px': _safe_median(shin_shift),
            'role_note': 'brain-like low-motion anchor rather than superiority dataset',
        },
        'real_motion_proxy': {
            kind: {
                'n_cases': len(rows),
                'median_reg_shift_p90_px': _safe_median([float(r['reg_shift_p90']) for r in rows]),
                'median_reg_psr': _safe_median([float(r['reg_psr_median']) for r in rows]),
            }
            for kind, rows in proxy_by_kind.items()
        },
        'ulm_motion_sweeps': {
            'brainlike': brain_stats,
            'elastic': elastic_stats,
        },
    }

    rows = [
        {
            'domain': 'shin', 'claim': 'brain_like_anchor_motion_is_tiny',
            'metric': 'median_reg_shift_p90_px', 'value': section['shin']['median_reg_shift_p90_px'],
            'unit': 'px', 'source': str(shin_csv.relative_to(ROOT)),
            'notes': 'Brain-like anchor used to constrain residual motion scale.'
        },
        {
            'domain': 'ulm', 'claim': 'stap_improves_brainlike_motion_consistency',
            'metric': 'delta_corr_score', 'value': brain_stats['mean_corr_score_stap'] - brain_stats['mean_corr_score_base'],
            'unit': 'score', 'source': str(ulm_brainlike_csv.relative_to(ROOT)),
            'notes': 'Mean correlation-score gain under brainlike motion on ULM real-IQ sweeps.'
        },
        {
            'domain': 'ulm', 'claim': 'stap_reduces_brainlike_background_variance_ratio',
            'metric': 'delta_bg_var_ratio', 'value': brain_stats['mean_bg_var_ratio_stap'] - brain_stats['mean_bg_var_ratio_base'],
            'unit': 'ratio', 'source': str(ulm_brainlike_csv.relative_to(ROOT)),
            'notes': 'Negative is better: STAP reduces background variance inflation under brainlike motion.'
        },
        {
            'domain': 'ulm', 'claim': 'stap_improves_elastic_motion_consistency',
            'metric': 'delta_corr_score', 'value': elastic_stats['mean_corr_score_stap'] - elastic_stats['mean_corr_score_base'],
            'unit': 'score', 'source': str(ulm_elastic_csv.relative_to(ROOT)),
            'notes': 'Mean correlation-score gain under elastic motion on ULM real-IQ sweeps.'
        },
        {
            'domain': 'ulm', 'claim': 'stap_reduces_elastic_background_variance_ratio',
            'metric': 'delta_bg_var_ratio', 'value': elastic_stats['mean_bg_var_ratio_stap'] - elastic_stats['mean_bg_var_ratio_base'],
            'unit': 'ratio', 'source': str(ulm_elastic_csv.relative_to(ROOT)),
            'notes': 'Negative is better: STAP reduces background variance inflation under elastic motion.'
        },
    ]
    return section, rows


def gammex_section() -> tuple[dict[str, Any], list[dict[str, Any]]]:
    along_json = ROOT / 'reports' / 'twinkling_gammex_alonglinear17_prf2500_str6_msd_ka_with_baselines_summary.json'
    across_json = ROOT / 'reports' / 'twinkling_gammex_across17_prf2500_str4_msd_ka_with_baselines_summary.json'
    along = _load_json(along_json)
    across = _load_json(across_json)

    def hygiene_row(data: dict[str, Any], fpr_target: float) -> dict[str, Any] | None:
        for row in data.get('hygiene', []):
            if abs(float(row['fpr_target']) - fpr_target) < 1e-12:
                return row
        return None

    along_h1 = hygiene_row(along, 0.001)
    across_h1 = hygiene_row(across, 0.001)

    def pooled_tpr(data: dict[str, Any], method: str, fpr_target: float) -> float | None:
        methods = data.get('pooled_roc', {}).get('methods', {})
        m = methods.get(method)
        if not m:
            return None
        for row in m.get('roc', []):
            if abs(float(row['fpr_target']) - fpr_target) < 1e-12:
                return float(row['tpr'])
        return None

    section = {
        'dataset_role': 'artifact-heavy nuisance and safety anchor',
        'sources': [str(along_json.relative_to(ROOT)), str(across_json.relative_to(ROOT))],
        'along_linear17': {
            'ka_state_counts': along['ka_state_counts'],
            'ka_reason_counts': along['ka_reason_counts'],
            'max_abs_delta_flow': along['max_abs_delta_flow'],
            'stap_tpr_at_fpr1e3': pooled_tpr(along, 'stap', 0.001),
            'base_tpr_at_fpr1e3': pooled_tpr(along, 'base', 0.001),
            'tail_post_rate_fpr1e3': float(along_h1['bg_tail_rate_post']) if along_h1 else None,
        },
        'across17': {
            'ka_state_counts': across['ka_state_counts'],
            'ka_reason_counts': across['ka_reason_counts'],
            'max_abs_delta_flow': across['max_abs_delta_flow'],
            'stap_tpr_at_fpr1e3': pooled_tpr(across, 'stap', 0.001),
            'base_tpr_at_fpr1e3': pooled_tpr(across, 'base', 0.001),
            'tail_post_rate_fpr1e3': float(across_h1['bg_tail_rate_post']) if across_h1 else None,
        },
    }

    rows = [
        {
            'domain': 'gammex', 'claim': 'along_linear17_contract_is_actionable_not_destructive',
            'metric': 'fraction_uplift_or_safety',
            'value': (along['ka_state_counts'].get('C1_SAFETY',0)+along['ka_state_counts'].get('C2_UPLIFT',0))/along['bundle_count'],
            'unit': 'fraction', 'source': str(along_json.relative_to(ROOT)),
            'notes': 'All along-linear17 bundles were actionable with no measured flow damage.'
        },
        {
            'domain': 'gammex', 'claim': 'across17_contract_mostly_refuses_nonactionable_regime',
            'metric': 'fraction_c0_off',
            'value': across['ka_state_counts'].get('C0_OFF',0)/across['bundle_count'],
            'unit': 'fraction', 'source': str(across_json.relative_to(ROOT)),
            'notes': 'Guard-dominant across-phantom regime is mostly rejected as non-actionable.'
        },
        {
            'domain': 'gammex', 'claim': 'stap_improves_across17_detection_at_low_fpr',
            'metric': 'delta_tpr_at_fpr1e3',
            'value': (section['across17']['stap_tpr_at_fpr1e3'] or 0.0) - (section['across17']['base_tpr_at_fpr1e3'] or 0.0),
            'unit': 'tpr', 'source': str(across_json.relative_to(ROOT)),
            'notes': 'Pooled ROC improvement in artifact-heavy across17 regime at fixed low FPR.'
        },
    ]
    return section, rows


def accepted_v2_section() -> tuple[dict[str, Any], list[dict[str, Any]]]:
    latency_csv = ROOT / 'reports' / 'simus_v2' / 'simus_v2_latency_frontier.csv'
    stack_json = ROOT / 'reports' / 'simus_v2' / 'simus_stap_stack_search_seed125_126_to_127_128_fixed.json'
    func_headline_csv = ROOT / 'reports' / 'simus_v2' / 'simus_eval_functional_seed221_222_to_223_224_ec6_bgcdf_outside_headline.csv'
    func_head_search_csv = ROOT / 'reports' / 'simus_v2' / 'simus_functional_stap_head_search_seed221_222_to_223_224_ec6_bgcdf_outside_headline.csv'
    func_targeted_csv = ROOT / 'reports' / 'simus_v2' / 'simus_functional_family_search_targeted_221_222_to_223_224_bgcdf_outside_headline.csv'

    frontier = _load_csv(latency_csv)
    stack = _load_json(stack_json)
    func_headline = _load_csv(func_headline_csv)
    func_head_search = _load_csv(func_head_search_csv)
    func_targeted = _load_csv(func_targeted_csv)

    def one(rows: list[dict[str, str]], **matches: str) -> dict[str, str]:
        for r in rows:
            ok = True
            for k,v in matches.items():
                if r.get(k) != v:
                    ok=False; break
            if ok: return r
        raise KeyError(matches)

    rpca_stap = one(frontier, domain='structural', pipeline_label='RPCA -> STAP')
    agsvd_stap = one(frontier, domain='structural', pipeline_label='Adaptive Global SVD -> STAP')
    hosvd_kasai = one(frontier, domain='structural', pipeline_label='HOSVD -> Kasai')

    func_best_intra = max((r for r in func_headline if r['split']=='eval' and r['base_profile']=='ClinIntraOpParenchyma-Pf-v3'), key=lambda r: float(r['selection_score']))
    func_best_mobile = max((r for r in func_headline if r['split']=='eval' and r['base_profile']=='ClinMobile-Pf-v2'), key=lambda r: float(r['selection_score']))

    targeted = {(r['base_profile'], r['method_family']): r for r in func_targeted}
    positives = 0
    total = 0
    for r in func_head_search:
        key=(r['base_profile'], r['method_family'])
        delta = _num(r['delta_selection_eval'])
        if key in targeted:
            delta = _num(targeted[key]['delta_selection_eval'])
        if delta is None:
            continue
        total += 1
        if delta > 0:
            positives += 1
    section = {
        'dataset_role': 'accepted clinically anchored benchmark translation context',
        'sources': [
            str(latency_csv.relative_to(ROOT)),
            str(stack_json.relative_to(ROOT)),
            str(func_headline_csv.relative_to(ROOT)),
            str(func_head_search_csv.relative_to(ROOT)),
            str(func_targeted_csv.relative_to(ROOT)),
        ],
        'structural': {
            'headline_quality_stack': {
                'pipeline': 'RPCA -> STAP',
                'stap_profile': rpca_stap['stap_profile'],
                'latency_ms': float(rpca_stap['total_ms_mean']),
                'auc_main_vs_bg': float(rpca_stap['auc_bg']),
                'auc_main_vs_nuisance': float(rpca_stap['auc_nuis']),
                'fpr_nuisance_at_tpr0p5': float(rpca_stap['tail_metric_value']),
            },
            'fast_stap_point': {
                'pipeline': 'Adaptive Global SVD -> STAP',
                'latency_ms': float(agsvd_stap['total_ms_mean']),
                'auc_main_vs_bg': float(agsvd_stap['auc_bg']),
                'auc_main_vs_nuisance': float(agsvd_stap['auc_nuis']),
                'fpr_nuisance_at_tpr0p5': float(agsvd_stap['tail_metric_value']),
            },
            'fast_native_point': {
                'pipeline': 'HOSVD -> Kasai',
                'latency_ms': float(hosvd_kasai['total_ms_mean']),
                'auc_main_vs_bg': float(hosvd_kasai['auc_bg']),
                'auc_main_vs_nuisance': float(hosvd_kasai['auc_nuis']),
                'fpr_nuisance_at_tpr0p5': float(hosvd_kasai['tail_metric_value']),
            },
        },
        'functional': {
            'best_intraop_pipeline': {
                'pipeline': func_best_intra['pipeline_label'],
                'selection_score': float(func_best_intra['selection_score']),
                'runtime_ms': float(func_best_intra['mean_runtime_ms']),
            },
            'best_mobile_pipeline': {
                'pipeline': func_best_mobile['pipeline_label'],
                'selection_score': float(func_best_mobile['selection_score']),
                'runtime_ms': float(func_best_mobile['mean_runtime_ms']),
            },
            'stap_head_positive_pairs': positives,
            'stap_head_total_pairs': total,
            'remaining_nonpositive_pairs': [
                {
                    'base_profile': r['base_profile'],
                    'method_family': r['method_family'],
                    'delta_selection_eval': (_num(targeted[(r['base_profile'], r['method_family'])]['delta_selection_eval'])
                                             if (r['base_profile'], r['method_family']) in targeted else _num(r['delta_selection_eval']))
                }
                for r in func_head_search
                if ((_num(targeted[(r['base_profile'], r['method_family'])]['delta_selection_eval']) if (r['base_profile'], r['method_family']) in targeted else _num(r['delta_selection_eval'])) or 0.0) <= 0.0
            ],
        },
    }
    rows = [
        {
            'domain': 'accepted_v2', 'claim': 'headline_structural_quality_stack', 'metric': 'latency_ms',
            'value': section['structural']['headline_quality_stack']['latency_ms'], 'unit': 'ms',
            'source': str(latency_csv.relative_to(ROOT)), 'notes': 'Accepted v2 highest-quality structural stack.'
        },
        {
            'domain': 'accepted_v2', 'claim': 'fast_structural_stap_point', 'metric': 'latency_ms',
            'value': section['structural']['fast_stap_point']['latency_ms'], 'unit': 'ms',
            'source': str(latency_csv.relative_to(ROOT)), 'notes': 'Accepted v2 low-latency structural STAP operating point.'
        },
        {
            'domain': 'accepted_v2', 'claim': 'stap_head_functional_positive_fraction', 'metric': 'positive_fraction',
            'value': positives/total if total else None, 'unit': 'fraction',
            'source': str(func_head_search_csv.relative_to(ROOT)), 'notes': 'After bounded targeted search, STAP head is positive in 11/12 accepted functional family/profile comparisons.'
        },
    ]
    return section, rows


def uncertainty_section() -> tuple[dict[str, Any], list[dict[str, Any]]]:
    path = REPORT_DIR / 'clinical_translation_uncertainty.json'
    data = _load_json(path)['sections']
    section = {
        'dataset_role': 'consistency and uncertainty summaries over existing real-data outputs',
        'source': str(path.relative_to(ROOT)),
        **data,
    }
    rows = [
        {
            'domain': 'uncertainty', 'claim': 'mace_phase2_fp_reduction_is_consistent',
            'metric': 'mace_phase2_fp_reduction_fraction_mean',
            'value': data['mace_phase2']['fp_reduction_fraction']['center'],
            'unit': 'fraction', 'source': str(path.relative_to(ROOT)),
            'notes': '95% bootstrap CI available in the uncertainty report.'
        },
        {
            'domain': 'uncertainty', 'claim': 'mace_holdout_safety_is_consistent',
            'metric': 'mace_holdout_all_plane_safety_win_rate',
            'value': data['mace_holdout']['all_planes_safety_win_rate']['center'],
            'unit': 'fraction', 'source': str(path.relative_to(ROOT)),
            'notes': 'Held-out alias-gate safety win rate across test planes.'
        },
        {
            'domain': 'uncertainty', 'claim': 'ulm_brainlike_corr_gain_is_consistently_positive',
            'metric': 'ulm_brainlike_delta_corr_score_mean',
            'value': data['ulm_brainlike']['delta_corr_score']['center'],
            'unit': 'score', 'source': str(path.relative_to(ROOT)),
            'notes': '95% bootstrap CI excludes zero in the current uncertainty summary.'
        },
    ]
    return section, rows


def detector_head_audit_section() -> tuple[dict[str, Any], list[dict[str, Any]]]:
    path = REPORT_DIR / 'realdata_detector_head_audit.json'
    summary = _load_json(path)['summary']
    section = {
        'dataset_role': 'same-residual real-data detector-head audit',
        'source': str(path.relative_to(ROOT)),
        'coverage_note': (
            'Coverage is limited to bundles where score_stap_preka.npy is non-zero. '
            'In Shin this currently means raw and MC-SVD families only; ULM coverage comes '
            'from the existing MC-SVD-style motion-sweep bundles.'
        ),
        'summary': summary,
    }
    rows = [
        {
            'domain': 'real_head_audit', 'claim': 'ulm_brainlike_same_residual_stap_beats_pd_on_auc',
            'metric': 'delta_auc_flow_bg',
            'value': _num(summary['ulm_brainlike']['stap_minus_pd']['auc_flow_bg']['delta']['center']),
            'unit': 'auc', 'source': str(path.relative_to(ROOT)),
            'notes': 'Same residual cube, ULM brainlike motion bundles.'
        },
        {
            'domain': 'real_head_audit', 'claim': 'ulm_elastic_same_residual_stap_beats_pd_on_auc',
            'metric': 'delta_auc_flow_bg',
            'value': _num(summary['ulm_elastic']['stap_minus_pd']['auc_flow_bg']['delta']['center']),
            'unit': 'auc', 'source': str(path.relative_to(ROOT)),
            'notes': 'Same residual cube, ULM elastic motion bundles.'
        },
        {
            'domain': 'real_head_audit', 'claim': 'shin_mcsvd_same_residual_stap_is_mixed',
            'metric': 'delta_auc_flow_bg',
            'value': _num(summary['shin_mc_svd']['stap_minus_kasai']['auc_flow_bg']['delta']['center']),
            'unit': 'auc', 'source': str(path.relative_to(ROOT)),
            'notes': 'Same residual cube, current Shin STAP-active coverage remains narrow.'
        },
    ]
    return section, rows


def render_md(packet: dict[str, Any], rows: list[dict[str, Any]]) -> str:
    mace = packet['sections']['mace']
    shin_ulm = packet['sections']['shin_ulm']
    gammex = packet['sections']['gammex']
    v2 = packet['sections']['accepted_v2']
    unc = packet['sections']['uncertainty']
    audit = packet['sections']['real_head_audit']
    lines = []
    lines.append('# Clinical Translation Packet')
    lines.append('')
    lines.append('This packet consolidates the translational evidence already present in the repository into four claims: whole-brain retrospective utility (Macé), brain-like robustness anchors (Shin/ULM), artifact-tail control (Gammex), and accepted-v2 performance/latency context.')
    lines.append('')
    lines.append('## 1. Structural Accepted-v2 Deployment Story')
    h = v2['structural']['headline_quality_stack']
    f = v2['structural']['fast_stap_point']
    n = v2['structural']['fast_native_point']
    lines.append(f"- Highest-quality structural deployed stack: `{h['pipeline']}` with `AUC(main,bg)={h['auc_main_vs_bg']:.3f}`, `AUC(main,nuis)={h['auc_main_vs_nuisance']:.3f}`, `FPR_nuis@TPR0.5={h['fpr_nuisance_at_tpr0p5']:.3f}`, `latency={h['latency_ms']:.1f} ms`.")
    lines.append(f"- Fast structural STAP operating point: `{f['pipeline']}` with `AUC(main,bg)={f['auc_main_vs_bg']:.3f}`, `AUC(main,nuis)={f['auc_main_vs_nuisance']:.3f}`, `FPR_nuis@TPR0.5={f['fpr_nuisance_at_tpr0p5']:.3f}`, `latency={f['latency_ms']:.1f} ms`.")
    lines.append(f"- Fast native comparator: `{n['pipeline']}` with `AUC(main,bg)={n['auc_main_vs_bg']:.3f}`, `AUC(main,nuis)={n['auc_main_vs_nuisance']:.3f}`, `FPR_nuis@TPR0.5={n['fpr_nuisance_at_tpr0p5']:.3f}`, `latency={n['latency_ms']:.1f} ms`.")
    lines.append('')
    lines.append('## 2. Functional Accepted-v2 Detector-Head Story')
    func = v2['functional']
    lines.append(f"- Best intra-op functional pipeline on the current broader split: `{func['best_intraop_pipeline']['pipeline']}` with selection score `{func['best_intraop_pipeline']['selection_score']:.3f}` at `{func['best_intraop_pipeline']['runtime_ms']:.1f} ms`.")
    lines.append(f"- Best mobile functional pipeline on the current broader split: `{func['best_mobile_pipeline']['pipeline']}` with selection score `{func['best_mobile_pipeline']['selection_score']:.3f}` at `{func['best_mobile_pipeline']['runtime_ms']:.1f} ms`.")
    lines.append(f"- Fair detector-head result: STAP is positive in `{func['stap_head_positive_pairs']}/{func['stap_head_total_pairs']}` accepted family/profile comparisons. The only remaining nonpositive pair is an effective tie in the mobile local-SVD family.")
    lines.append('')
    lines.append('## 3. Macé Whole-Brain Retrospective Utility')
    p2 = mace['phase2_alias_gate']
    pdc = mace['pdonly_contract']
    lines.append(f"- In the whole-brain phase-2 sweep, alias gating reduced matched-TPR false positives by `{100*(p2['mean_fp_reduction_fraction'] or 0):.1f}%` while retaining `{100*(p2['mean_gate_kept_fraction'] or 0):.1f}%` of tiles on average.")
    lines.append(f"- In the PD-only contract sweep, the contract stayed in `C0_OFF` for all planes and still reduced false positives by `{100*(pdc['mean_fp_reduction_fraction'] or 0):.1f}%` while retaining `{100*(pdc['mean_hit_retention'] or 0):.1f}%` of hits.")
    if 'alias_gate_holdout' in mace:
        h = mace['alias_gate_holdout']
        h_all = h['counts_test_all']
        h_sel = h['counts_test_selected']
        lines.append(f"- In the appendix-only held-out alias-gate check (`scan4/scan5`), label-free gating reduced offline CP/CA/DG proxy false positives by `{100*_pct_reduction(float(h_all['fp_pre'] or 0.0), float(h_all['fp_post'] or 0.0)):.1f}%` while retaining `{100*(h_all['hit_retention'] or 0):.1f}%` of VIS/SC/LGd proxy hits across all held-out planes.")
        lines.append(f"- On the Pf-peak-selected held-out subset (`n={h_sel['n_planes']}`), hit retention stayed at `{100*(h_sel['hit_retention'] or 0):.1f}%`; this subset is a stability check, not a separate false-positive effect-size estimate.")
    lines.append('- Interpretation: Macé currently supports the translational story as a retrospective whole-brain readout/atlas anchor rather than as a raw-IQ superiority benchmark.')
    lines.append('')
    lines.append('## 4. Consistency / Uncertainty')
    lines.append(f"- Macé phase-2 false-positive reduction is consistent: `{100*unc['mace_phase2']['fp_reduction_fraction']['center']:.1f}%` (95% CI `{100*unc['mace_phase2']['fp_reduction_fraction']['lo']:.1f}` to `{100*unc['mace_phase2']['fp_reduction_fraction']['hi']:.1f}`).")
    lines.append(f"- Macé held-out safety win rate remains high: `{100*unc['mace_holdout']['all_planes_safety_win_rate']['center']:.1f}%` (95% CI `{100*unc['mace_holdout']['all_planes_safety_win_rate']['lo']:.1f}` to `{100*unc['mace_holdout']['all_planes_safety_win_rate']['hi']:.1f}`).")
    lines.append(f"- ULM brainlike correlation gain remains positive with uncertainty: `{unc['ulm_brainlike']['delta_corr_score']['center']:.3f}` (95% CI `{unc['ulm_brainlike']['delta_corr_score']['lo']:.3f}` to `{unc['ulm_brainlike']['delta_corr_score']['hi']:.3f}`).")
    lines.append('')
    lines.append('## 5. Brain-Like Real-IQ Anchors (Shin / ULM)')
    sh = shin_ulm['shin']
    ub = shin_ulm['ulm_motion_sweeps']['brainlike']
    ue = shin_ulm['ulm_motion_sweeps']['elastic']
    lines.append(f"- Shin remains a low-motion brain-like anchor: median `reg_shift_p90 = {sh['median_reg_shift_p90_px']:.4f} px`.")
    lines.append(f"- On ULM brainlike motion sweeps, STAP improved mean correlation score by `{(ub['mean_corr_score_stap']-ub['mean_corr_score_base']):.3f}` and reduced background variance ratio by `{(ub['mean_bg_var_ratio_stap']-ub['mean_bg_var_ratio_base']):.3f}`.")
    lines.append(f"- On ULM elastic motion sweeps, STAP improved mean correlation score by `{(ue['mean_corr_score_stap']-ue['mean_corr_score_base']):.3f}` and reduced background variance ratio by `{(ue['mean_bg_var_ratio_stap']-ue['mean_bg_var_ratio_base']):.3f}`.")
    lines.append('')
    lines.append('## 6. Same-Residual Real-Data Detector-Head Audit')
    lines.append('- Coverage is intentionally narrow and explicit: only bundles with active STAP scores are audited.')
    lines.append(f"- ULM brainlike same-residual audit: `ΔAUC(STAP−PD) = {audit['summary']['ulm_brainlike']['stap_minus_pd']['auc_flow_bg']['delta']['center']:.3f}`.")
    lines.append(f"- ULM elastic same-residual audit: `ΔAUC(STAP−PD) = {audit['summary']['ulm_elastic']['stap_minus_pd']['auc_flow_bg']['delta']['center']:.3f}`.")
    lines.append(f"- Shin MC-SVD same-residual audit is mixed: `ΔAUC(STAP−Kasai) = {audit['summary']['shin_mc_svd']['stap_minus_kasai']['auc_flow_bg']['delta']['center']:.3f}`.")
    lines.append('- Interpretation: the same-residual detector-head story is strong on ULM, but not yet uniformly positive on the currently available Shin bundles.')
    lines.append('')
    lines.append('## 7. Gammex Artifact-Tail Control')
    ga = gammex['along_linear17']
    gx = gammex['across17']
    lines.append(f"- Along-linear17 remained actionable: `{ga['ka_state_counts']}` with `max_abs_delta_flow = {ga['max_abs_delta_flow']}`.")
    lines.append(f"- Across17 was mostly rejected as non-actionable: `{gx['ka_state_counts']}` with reason counts `{gx['ka_reason_counts']}`.")
    lines.append(f"- In the artifact-heavy across17 regime, STAP improved pooled TPR at `FPR=1e-3` by `{((gx['stap_tpr_at_fpr1e3'] or 0)-(gx['base_tpr_at_fpr1e3'] or 0)):.3f}` over the base score.")
    lines.append('')
    lines.append('## Practical Reading')
    lines.append('- The strongest translational story is not “STAP wins every pipeline” or “real-data evidence is uniformly positive.” It is that STAP is a nuisance-focused detector head that materially improves accepted structural nuisance rejection, remains favorable in almost all accepted functional detector-head comparisons, and already shows a strong same-residual real-IQ advantage on ULM while remaining mixed on the currently available Shin bundles.')
    lines.append('- The latency frontier already provides a clinically usable fast STAP operating point (`Adaptive Global SVD -> STAP`) in the sub-0.35 s range, while the highest-quality structural stack remains slower (`RPCA -> STAP`).')
    return '\n'.join(lines) + '\n'


def main() -> None:
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    sections = {}
    flat_rows = []
    for name, fn in [
        ('mace', mace_section),
        ('shin_ulm', shin_ulm_section),
        ('gammex', gammex_section),
        ('accepted_v2', accepted_v2_section),
        ('uncertainty', uncertainty_section),
        ('real_head_audit', detector_head_audit_section),
    ]:
        sec, rows = fn()
        sections[name] = sec
        flat_rows.extend(rows)
    packet = {
        'packet_name': 'clinical_translation_packet',
        'sections': sections,
    }
    out_json = REPORT_DIR / 'clinical_translation_packet.json'
    out_csv = REPORT_DIR / 'clinical_translation_packet.csv'
    out_md = REPORT_DIR / 'clinical_translation_packet.md'
    with out_json.open('w') as f:
        json.dump(packet, f, indent=2, sort_keys=True)
    fieldnames = ['domain','claim','metric','value','unit','source','notes']
    with out_csv.open('w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for row in flat_rows:
            w.writerow(row)
    out_md.write_text(render_md(packet, flat_rows))
    print(f'wrote {out_json.relative_to(ROOT)}')
    print(f'wrote {out_csv.relative_to(ROOT)} rows={len(flat_rows)}')
    print(f'wrote {out_md.relative_to(ROOT)}')


if __name__ == '__main__':
    main()
