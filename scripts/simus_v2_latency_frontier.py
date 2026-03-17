#!/usr/bin/env python3
from __future__ import annotations

import csv
import json
from collections import defaultdict
from pathlib import Path
from statistics import mean
from typing import Any

ROOT = Path('/home/arthu/fus-detectors')
REPORTS = ROOT / 'reports' / 'simus_v2'


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline='', encoding='utf-8') as fh:
        return list(csv.DictReader(fh))


def read_json(path: Path) -> dict[str, Any]:
    with path.open(encoding='utf-8') as fh:
        return json.load(fh)


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        raise ValueError('no rows')
    fieldnames: list[str] = []
    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('w', newline='', encoding='utf-8') as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('w', encoding='utf-8') as fh:
        json.dump(payload, fh, indent=2, sort_keys=True)
        fh.write('\n')


def _structural_rows() -> list[dict[str, Any]]:
    payload = read_json(REPORTS / 'simus_symmetric_pipeline_compare_seed125_126_to_127_128.json')
    rows = [r for r in payload['rows'] if str(r.get('split')) == 'eval']
    grouped: dict[tuple[str, str, str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[(
            str(row['method_family']),
            str(row['config_name']),
            str(row['detector_head']),
            str(row['pipeline_label']),
        )].append(row)

    out: list[dict[str, Any]] = []
    for (method_family, config_name, detector_head, pipeline_label), items in sorted(grouped.items()):
        total_vals: list[float] = []
        base_vals: list[float] = []
        stap_vals: list[float] = []
        sel_vals: list[float] = []
        auc_bg_vals: list[float] = []
        auc_n_vals: list[float] = []
        fpr_vals: list[float] = []
        baseline_label = str(items[0]['baseline_label'])
        baseline_type = str(items[0]['baseline_type'])
        stap_profile = items[0].get('stap_profile') or ''
        for item in items:
            meta = read_json(ROOT / item['bundle_dir'] / 'meta.json')
            tel = meta.get('stap_fallback_telemetry', {})
            base = float(tel.get('baseline_ms') or 0.0)
            stap = float(tel.get('stap_total_ms') or 0.0) if detector_head == 'stap' else 0.0
            total_vals.append(base + stap)
            base_vals.append(base)
            stap_vals.append(stap)
            sel_vals.append(float(item['selection_score']))
            auc_bg_vals.append(float(item['auc_main_vs_bg']))
            auc_n_vals.append(float(item['auc_main_vs_nuisance']))
            fpr_vals.append(float(item['fpr_nuisance_match@0p5']))
        out.append({
            'domain': 'structural',
            'profile_scope': 'accepted_v2_eval_seed127_128',
            'source': 'symmetric_pipeline_compare',
            'method_family': method_family,
            'pipeline_label': pipeline_label,
            'detector_head': detector_head,
            'baseline_label': baseline_label,
            'baseline_type': baseline_type,
            'config_name': config_name,
            'stap_profile': stap_profile,
            'baseline_ms_mean': mean(base_vals),
            'stap_ms_mean': mean(stap_vals),
            'total_ms_mean': mean(total_vals),
            'selection_score': mean(sel_vals),
            'auc_bg': mean(auc_bg_vals),
            'auc_nuis': mean(auc_n_vals),
            'tail_metric_label': 'fpr_nuisance@TPR0.5',
            'tail_metric_value': mean(fpr_vals),
            'notes': 'Accepted structural v2 held-out eval, frozen family configs',
        })

    # Add the selected joint stack search result explicitly; this is the structural headline.
    stack = read_json(REPORTS / 'simus_stap_stack_search_seed125_126_to_127_128_fixed.json')
    eval_rows = [r for r in stack['rows'] if str(r.get('split')) == 'eval']
    total_vals = []
    base_vals = []
    stap_vals = []
    auc_bg_vals = []
    auc_n_vals = []
    fpr_vals = []
    for item in eval_rows:
        meta = read_json(ROOT / item['bundle_dir'] / 'meta.json')
        tel = meta.get('stap_fallback_telemetry', {})
        base = float(tel.get('baseline_ms') or 0.0)
        stap = float(tel.get('stap_total_ms') or 0.0)
        total_vals.append(base + stap)
        base_vals.append(base)
        stap_vals.append(stap)
        auc_bg_vals.append(float(item['auc_main_vs_bg']))
        auc_n_vals.append(float(item['auc_main_vs_nuisance']))
        fpr_vals.append(float(item['fpr_nuisance_match@0p5']))
    selected = stack['selected_stack']
    out.append({
        'domain': 'structural',
        'profile_scope': 'accepted_v2_eval_seed127_128',
        'source': 'joint_stack_search',
        'method_family': str(selected['residual_family']),
        'pipeline_label': 'RPCA -> STAP (headline stack)',
        'detector_head': 'stap',
        'baseline_label': 'RPCA',
        'baseline_type': 'rpca',
        'config_name': str(selected['config_name']),
        'stap_profile': str(selected['stap_profile']),
        'baseline_ms_mean': mean(base_vals),
        'stap_ms_mean': mean(stap_vals),
        'total_ms_mean': mean(total_vals),
        'selection_score': float(selected['selection_score']),
        'auc_bg': mean(auc_bg_vals),
        'auc_nuis': mean(auc_n_vals),
        'tail_metric_label': 'fpr_nuisance@TPR0.5',
        'tail_metric_value': mean(fpr_vals),
        'notes': 'Best accepted structural deployed stack after joint residualizer+STAP search',
    })
    return out


def _functional_rows() -> list[dict[str, Any]]:
    rows = read_csv(REPORTS / 'simus_eval_functional_seed221_222_to_223_224_ec6_bgcdf_outside_headline.csv')
    eval_rows = [r for r in rows if r['split'] == 'eval']
    out: list[dict[str, Any]] = []
    for row in eval_rows:
        out.append({
            'domain': 'functional',
            'profile_scope': row['base_profile'],
            'source': 'functional_eval',
            'method_family': row['method_family'],
            'pipeline_label': row['pipeline_label'],
            'detector_head': row['detector_head'],
            'baseline_label': row['baseline_label'],
            'baseline_type': row['baseline_type'],
            'config_name': row['config_name'],
            'stap_profile': row.get('stap_profile') or '',
            'baseline_ms_mean': None,
            'stap_ms_mean': None,
            'total_ms_mean': float(row['mean_runtime_ms']),
            'selection_score': float(row['selection_score']),
            'auc_bg': float(row['mean_auc_activation_vs_bg']),
            'auc_nuis': float(row['mean_auc_activation_vs_nuisance']),
            'tail_metric_label': 'outside_frac_task@1e-03',
            'tail_metric_value': float(row['mean_outside_frac_task@1e-03']),
            'notes': 'Accepted functional v2 held-out eval, common readout bgcdf_outside_glm',
        })

    head_rows = read_csv(REPORTS / 'simus_functional_stap_head_search_seed221_222_to_223_224_ec6_bgcdf_outside_headline.csv')
    targeted = {
        (r['base_profile'], r['method_family']): r
        for r in read_csv(REPORTS / 'simus_functional_family_search_targeted_221_222_to_223_224_bgcdf_outside_headline.csv')
    }
    final = []
    for row in head_rows:
        final.append(targeted.get((row['base_profile'], row['method_family']), row))
    native_mean = mean(float(r['native_runtime_ms_eval']) for r in final)
    stap_mean = mean(float(r['stap_runtime_ms_eval']) for r in final)
    delta_mean = mean(float(r['delta_selection_eval']) for r in final)
    pos = sum(float(r['delta_selection_eval']) > 0 for r in final)
    out.append({
        'domain': 'functional',
        'profile_scope': 'accepted_v2_eval_seed223_224',
        'source': 'functional_head_audit',
        'method_family': 'aggregate',
        'pipeline_label': 'native simple head vs STAP head',
        'detector_head': 'stap',
        'baseline_label': 'aggregate',
        'baseline_type': 'aggregate',
        'config_name': 'bgcdf_outside_glm',
        'stap_profile': 'mixed_frozen_search',
        'baseline_ms_mean': native_mean,
        'stap_ms_mean': stap_mean - native_mean,
        'total_ms_mean': stap_mean,
        'selection_score': delta_mean,
        'auc_bg': None,
        'auc_nuis': None,
        'tail_metric_label': 'delta_selection_eval',
        'tail_metric_value': delta_mean,
        'notes': f'STAP positive in {pos}/{len(final)} family/profile comparisons; remaining miss is an effective tie',
    })
    return out


def _apply_pareto(rows: list[dict[str, Any]], group_keys: tuple[str, ...]) -> list[dict[str, Any]]:
    groups: dict[tuple[Any, ...], list[int]] = defaultdict(list)
    for idx, row in enumerate(rows):
        groups[tuple(row.get(k) for k in group_keys)].append(idx)
    for indices in groups.values():
        for i in indices:
            a = rows[i]
            dominated = False
            for j in indices:
                if i == j:
                    continue
                b = rows[j]
                if b['selection_score'] is None or a['selection_score'] is None:
                    continue
                if (
                    float(b['total_ms_mean']) <= float(a['total_ms_mean'])
                    and float(b['selection_score']) >= float(a['selection_score'])
                    and (
                        float(b['total_ms_mean']) < float(a['total_ms_mean'])
                        or float(b['selection_score']) > float(a['selection_score'])
                    )
                ):
                    dominated = True
                    break
            a['pareto_selection_runtime'] = not dominated
    return rows


def main() -> None:
    structural = _apply_pareto(_structural_rows(), ('domain', 'profile_scope'))
    functional = _apply_pareto(_functional_rows(), ('domain', 'profile_scope'))
    rows = structural + functional
    out_csv = REPORTS / 'simus_v2_latency_frontier.csv'
    out_json = REPORTS / 'simus_v2_latency_frontier.json'
    pareto_csv = REPORTS / 'simus_v2_latency_frontier_pareto.csv'
    pareto_json = REPORTS / 'simus_v2_latency_frontier_pareto.json'
    write_csv(out_csv, rows)
    write_json(out_json, {'schema_version': 'simus_v2_latency_frontier.v1', 'rows': rows})
    pareto_rows = [r for r in rows if r.get('pareto_selection_runtime')]
    write_csv(pareto_csv, pareto_rows)
    write_json(pareto_json, {'schema_version': 'simus_v2_latency_frontier_pareto.v1', 'rows': pareto_rows})
    print(out_csv)
    print(out_json)
    print(pareto_csv)
    print(pareto_json)


if __name__ == '__main__':
    main()
