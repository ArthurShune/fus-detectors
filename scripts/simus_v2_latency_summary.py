#!/usr/bin/env python3
from __future__ import annotations

import csv
import json
from pathlib import Path
from statistics import mean

ROOT = Path('/home/arthu/stap-for-fus')
REPORTS = ROOT / 'reports' / 'simus_v2'
RUNS = ROOT / 'runs' / 'sim_eval'


def read_csv(path: Path):
    with open(path, newline='') as fh:
        return list(csv.DictReader(fh))


def read_json(path: Path):
    with open(path) as fh:
        return json.load(fh)


def load_meta_times(paths):
    baseline = []
    stap = []
    for path in paths:
        obj = read_json(path)
        tel = obj.get('stap_fallback_telemetry', {})
        if tel.get('baseline_ms') is not None:
            baseline.append(float(tel['baseline_ms']))
        if tel.get('stap_total_ms') is not None:
            stap.append(float(tel['stap_total_ms']))
    return {
        'baseline_ms_mean': mean(baseline) if baseline else None,
        'stap_ms_mean': mean(stap) if stap else None,
        'total_ms_mean': (mean(baseline) + mean(stap)) if baseline and stap else (mean(baseline) if baseline else None),
        'n_cases': len(baseline),
    }


def structural_rows():
    rows = []

    stack = read_json(REPORTS / 'simus_stap_stack_search_seed125_126_to_127_128_fixed.json')
    sel = stack['eval_summary'][0]
    meta_paths = sorted((RUNS / 'simus_v2_stap_stack_search_seed125_126_to_127_128_fixed' / 'eval').glob('*/*/meta.json'))
    times = load_meta_times(meta_paths)
    rows.append({
        'domain': 'structural',
        'comparison_type': 'headline_pipeline',
        'profile_scope': 'accepted_v2_eval_seed127_128',
        'label': 'Headline STAP stack',
        'pipeline': 'RPCA -> STAP',
        'readout_or_head': 'STAP',
        'config': f"lam1_it250_ds2_t32_r4 + {sel['stap_profile']}",
        'latency_protocol': 'held-out bundle runtime on RTX 4080 SUPER',
        'baseline_ms_mean': times['baseline_ms_mean'],
        'stap_ms_mean': times['stap_ms_mean'],
        'total_ms_mean': times['total_ms_mean'],
        'selection_score': sel['selection_score'],
        'auc_bg': sel['mean_auc_main_vs_bg'],
        'auc_nuis': sel['mean_auc_main_vs_nuisance'],
        'fpr_nuis_at_tpr05': sel['mean_fpr_nuisance_match@0p5'],
        'notes': 'Best frozen deployed structural stack on accepted v2',
    })

    sym = read_json(REPORTS / 'simus_symmetric_pipeline_compare_seed125_126_to_127_128_headline.json')
    simple_rows = sym['eval_best_simple']
    best_simple = max(simple_rows, key=lambda r: float(r['selection_score']))
    sym_meta_all = sorted((RUNS / 'simus_v2_symmetric_pipeline_compare_seed125_126_to_127_128' / 'eval').rglob('meta.json'))
    simple_family = best_simple['baseline_type']
    simple_meta_paths = [p for p in sym_meta_all if simple_family in p.parent.name]
    simple_times = load_meta_times(simple_meta_paths)
    rows.append({
        'domain': 'structural',
        'comparison_type': 'headline_native_simple',
        'profile_scope': 'accepted_v2_eval_seed127_128',
        'label': 'Best native simple stack',
        'pipeline': best_simple['pipeline_label'],
        'readout_or_head': best_simple['detector_label'],
        'config': best_simple['config_name'],
        'latency_protocol': 'held-out bundle runtime on RTX 4080 SUPER',
        'baseline_ms_mean': simple_times['baseline_ms_mean'],
        'stap_ms_mean': 0.0,
        'total_ms_mean': simple_times['baseline_ms_mean'],
        'selection_score': best_simple['selection_score'],
        'auc_bg': best_simple['mean_auc_main_vs_bg'],
        'auc_nuis': best_simple['mean_auc_main_vs_nuisance'],
        'fpr_nuis_at_tpr05': best_simple['mean_fpr_nuisance_match@0p5'],
        'notes': 'Fastest strong native simple structural stack on accepted v2',
    })

    # Same-residual RPCA head comparison for interpretability.
    rpca_simple = next(r for r in simple_rows if r['baseline_type'] == 'rpca')
    rpca_sym_meta = [p for p in sym_meta_all if 'rpca' in p.parent.name]
    rpca_times = load_meta_times(rpca_sym_meta)
    rows.append({
        'domain': 'structural',
        'comparison_type': 'same_residual_native_head',
        'profile_scope': 'accepted_v2_eval_seed127_128',
        'label': 'Same residual native head',
        'pipeline': rpca_simple['pipeline_label'],
        'readout_or_head': rpca_simple['detector_label'],
        'config': rpca_simple['config_name'],
        'latency_protocol': 'held-out bundle runtime on RTX 4080 SUPER',
        'baseline_ms_mean': rpca_times['baseline_ms_mean'],
        'stap_ms_mean': 0.0,
        'total_ms_mean': rpca_times['baseline_ms_mean'],
        'selection_score': rpca_simple['selection_score'],
        'auc_bg': rpca_simple['mean_auc_main_vs_bg'],
        'auc_nuis': rpca_simple['mean_auc_main_vs_nuisance'],
        'fpr_nuis_at_tpr05': rpca_simple['mean_fpr_nuisance_match@0p5'],
        'notes': 'Kasai on the same RPCA residual used by the structural headline stack',
    })

    return rows


def functional_rows():
    rows = []
    func_headline = read_csv(REPORTS / 'simus_eval_functional_seed221_222_to_223_224_ec6_bgcdf_outside_headline.csv')
    eval_rows = [r for r in func_headline if r['split'] == 'eval']
    for profile in ['ClinIntraOpParenchyma-Pf-v3', 'ClinMobile-Pf-v2']:
        best = max((r for r in eval_rows if r['base_profile'] == profile), key=lambda r: float(r['selection_score']))
        rows.append({
            'domain': 'functional',
            'comparison_type': 'headline_pipeline',
            'profile_scope': profile,
            'label': f'{profile} best pipeline',
            'pipeline': best['pipeline_label'],
            'readout_or_head': best['detector_label'],
            'config': best['config_name'],
            'latency_protocol': 'held-out functional case runtime on RTX 4080 SUPER',
            'baseline_ms_mean': None,
            'stap_ms_mean': None,
            'total_ms_mean': float(best['mean_runtime_ms']),
            'selection_score': float(best['selection_score']),
            'auc_bg': float(best['mean_auc_activation_vs_bg']),
            'auc_nuis': float(best['mean_auc_activation_vs_nuisance']),
            'fpr_nuis_at_tpr05': float(best['mean_outside_frac_task@1e-03']),
            'notes': 'Common readout bgcdf_outside_glm',
        })

    head = read_csv(REPORTS / 'simus_functional_stap_head_search_seed221_222_to_223_224_ec6_bgcdf_outside_headline.csv')
    targeted = read_csv(REPORTS / 'simus_functional_family_search_targeted_221_222_to_223_224_bgcdf_outside_headline.csv')
    overrides = {(r['base_profile'], r['method_family']): r for r in targeted}
    final_rows = []
    for r in head:
        key = (r['base_profile'], r['method_family'])
        final_rows.append(overrides.get(key, r))
    pos = [r for r in final_rows if float(r['delta_selection_eval']) > 0]
    rows.append({
        'domain': 'functional',
        'comparison_type': 'detector_head_aggregate',
        'profile_scope': 'accepted_v2_eval_seed223_224',
        'label': 'STAP head aggregate',
        'pipeline': 'native head vs STAP head',
        'readout_or_head': 'STAP',
        'config': 'bgcdf_outside_glm',
        'latency_protocol': 'held-out functional case runtime on RTX 4080 SUPER',
        'baseline_ms_mean': mean(float(r['native_runtime_ms_eval']) for r in final_rows),
        'stap_ms_mean': mean(float(r['stap_runtime_ms_eval']) - float(r['native_runtime_ms_eval']) for r in final_rows),
        'total_ms_mean': mean(float(r['stap_runtime_ms_eval']) for r in final_rows),
        'selection_score': mean(float(r['delta_selection_eval']) for r in final_rows),
        'auc_bg': None,
        'auc_nuis': None,
        'fpr_nuis_at_tpr05': None,
        'notes': f"STAP positive in {len(pos)}/{len(final_rows)} family/profile comparisons; remaining miss is an effective tie",
    })

    for r in targeted:
        rows.append({
            'domain': 'functional',
            'comparison_type': 'remaining_targeted_pair',
            'profile_scope': r['base_profile'],
            'label': f"{r['base_profile']} {r['method_family']}",
            'pipeline': f"native vs STAP on {r['method_family']}",
            'readout_or_head': 'STAP',
            'config': r['stap_config_name_dev'],
            'latency_protocol': 'held-out functional case runtime on RTX 4080 SUPER',
            'baseline_ms_mean': float(r['native_runtime_ms_eval']),
            'stap_ms_mean': float(r['stap_runtime_ms_eval']) - float(r['native_runtime_ms_eval']),
            'total_ms_mean': float(r['stap_runtime_ms_eval']),
            'selection_score': float(r['delta_selection_eval']),
            'auc_bg': float(r['stap_auc_activation_vs_bg_eval']),
            'auc_nuis': float(r['stap_auc_activation_vs_nuisance_eval']),
            'fpr_nuis_at_tpr05': None,
            'notes': 'Targeted broader-split family check',
        })
    return rows


def main():
    rows = structural_rows() + functional_rows()
    out_csv = REPORTS / 'simus_v2_latency_summary.csv'
    out_json = REPORTS / 'simus_v2_latency_summary.json'
    out_json.parent.mkdir(parents=True, exist_ok=True)
    fields = [
        'domain', 'comparison_type', 'profile_scope', 'label', 'pipeline', 'readout_or_head', 'config',
        'latency_protocol', 'baseline_ms_mean', 'stap_ms_mean', 'total_ms_mean', 'selection_score',
        'auc_bg', 'auc_nuis', 'fpr_nuis_at_tpr05', 'notes'
    ]
    with open(out_csv, 'w', newline='') as fh:
        w = csv.DictWriter(fh, fieldnames=fields)
        w.writeheader()
        for row in rows:
            w.writerow(row)
    payload = {'schema_version': 'simus_v2_latency_summary.v1', 'rows': rows}
    with open(out_json, 'w') as fh:
        json.dump(payload, fh, indent=2)
    print(out_csv)
    print(out_json)

if __name__ == '__main__':
    main()
