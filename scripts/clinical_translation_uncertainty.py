#!/usr/bin/env python3
from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = ROOT / 'reports' / 'clinical_translation'


def _load_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline='') as f:
        return list(csv.DictReader(f))


def _load_json(path: Path) -> dict[str, Any]:
    with path.open() as f:
        return json.load(f)


def _finite(xs: list[float]) -> np.ndarray:
    arr = np.asarray(xs, dtype=np.float64)
    return arr[np.isfinite(arr)]


def _bootstrap_ci(xs: list[float], *, stat: str = 'mean', n_boot: int = 4000, seed: int = 0) -> dict[str, float | None]:
    arr = _finite(xs)
    if arr.size == 0:
        return {'center': None, 'lo': None, 'hi': None, 'n': 0}
    if stat == 'median':
        center = float(np.median(arr))
        fn = np.median
    else:
        center = float(np.mean(arr))
        fn = np.mean
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, arr.size, size=(n_boot, arr.size))
    boots = fn(arr[idx], axis=1)
    lo, hi = np.quantile(boots, [0.025, 0.975])
    return {'center': center, 'lo': float(lo), 'hi': float(hi), 'n': int(arr.size)}


def _win_rate(xs: list[float], *, better: str = 'positive') -> dict[str, float | int]:
    arr = _finite(xs)
    if arr.size == 0:
        return {'wins': 0, 'ties': 0, 'losses': 0, 'win_rate': float('nan'), 'n': 0}
    if better == 'negative':
        wins = int(np.sum(arr < 0))
        losses = int(np.sum(arr > 0))
    else:
        wins = int(np.sum(arr > 0))
        losses = int(np.sum(arr < 0))
    ties = int(np.sum(arr == 0))
    return {'wins': wins, 'ties': ties, 'losses': losses, 'win_rate': float(wins / arr.size), 'n': int(arr.size)}


def mace_phase2() -> tuple[dict[str, Any], list[dict[str, Any]]]:
    path = ROOT / 'reports' / 'mace_phase2_summary.csv'
    rows = _load_csv(path)
    fp_red = []
    gate_kept = []
    for r in rows:
        pre = float(r['pd_fp_at_tpr'])
        post = float(r['gated_pd_fp_at_tpr'])
        if pre > 0:
            fp_red.append((pre - post) / pre)
        gate_kept.append(float(r['gate_kept_frac']))
    sec = {
        'source': str(path.relative_to(ROOT)),
        'fp_reduction_fraction': _bootstrap_ci(fp_red),
        'gate_kept_fraction': _bootstrap_ci(gate_kept),
        'fp_reduction_win_rate': _win_rate(fp_red),
    }
    flat = [
        {'domain': 'mace_phase2', 'metric': 'fp_reduction_fraction_mean', **sec['fp_reduction_fraction']},
        {'domain': 'mace_phase2', 'metric': 'gate_kept_fraction_mean', **sec['gate_kept_fraction']},
        {'domain': 'mace_phase2', 'metric': 'fp_reduction_win_rate', **sec['fp_reduction_win_rate']},
    ]
    return sec, flat


def mace_holdout() -> tuple[dict[str, Any], list[dict[str, Any]]]:
    path = ROOT / 'reports' / 'mace_alias_gate_holdout.csv'
    rows = [r for r in _load_csv(path) if r['split'] == 'test']
    all_fp_red = []
    all_hit_ret = []
    safe_plane = []
    sel_fp_red = []
    sel_hit_ret = []
    for r in rows:
        fp_pre = float(r['fp_pre'])
        fp_post = float(r['fp_post'])
        hits_pre = float(r['hits_pre'])
        hits_post = float(r['hits_post'])
        selected = str(r['selected']).lower() in {'1', 'true', 'yes'}
        if fp_pre > 0:
            val = (fp_pre - fp_post) / fp_pre
            all_fp_red.append(val)
            if selected:
                sel_fp_red.append(val)
        if hits_pre > 0:
            ret = hits_post / hits_pre
            all_hit_ret.append(ret)
            if selected:
                sel_hit_ret.append(ret)
        safe_plane.append(float((fp_post <= fp_pre) and (hits_post >= hits_pre)))
    sec = {
        'source': str(path.relative_to(ROOT)),
        'all_planes_fp_reduction_fraction': _bootstrap_ci(all_fp_red),
        'all_planes_hit_retention': _bootstrap_ci(all_hit_ret),
        'all_planes_safety_win_rate': _bootstrap_ci(safe_plane),
        'selected_planes_fp_reduction_fraction': _bootstrap_ci(sel_fp_red),
        'selected_planes_hit_retention': _bootstrap_ci(sel_hit_ret),
    }
    flat = [
        {'domain': 'mace_holdout', 'metric': 'all_planes_fp_reduction_fraction_mean', **sec['all_planes_fp_reduction_fraction']},
        {'domain': 'mace_holdout', 'metric': 'all_planes_hit_retention_mean', **sec['all_planes_hit_retention']},
        {'domain': 'mace_holdout', 'metric': 'all_planes_safety_win_rate_mean', **sec['all_planes_safety_win_rate']},
        {'domain': 'mace_holdout', 'metric': 'selected_planes_hit_retention_mean', **sec['selected_planes_hit_retention']},
    ]
    return sec, flat


def shin_motion() -> tuple[dict[str, Any], list[dict[str, Any]]]:
    path = ROOT / 'reports' / 'shin_motion_telemetry.csv'
    rows = _load_csv(path)
    deltas = [float(r['delta_auc_stap_minus_mcsvd']) for r in rows]
    shifts = [float(r['reg_shift_p90_px']) for r in rows]
    sec = {
        'source': str(path.relative_to(ROOT)),
        'delta_auc_stap_minus_mcsvd': _bootstrap_ci(deltas),
        'delta_auc_win_rate': _win_rate(deltas),
        'reg_shift_p90_px_median': _bootstrap_ci(shifts, stat='median'),
    }
    flat = [
        {'domain': 'shin', 'metric': 'delta_auc_stap_minus_mcsvd_mean', **sec['delta_auc_stap_minus_mcsvd']},
        {'domain': 'shin', 'metric': 'delta_auc_win_rate', **sec['delta_auc_win_rate']},
        {'domain': 'shin', 'metric': 'reg_shift_p90_px_median', **sec['reg_shift_p90_px_median']},
    ]
    return sec, flat


def ulm_motion(tag: str, path: Path) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    rows = _load_csv(path)
    dcorr = [float(r['corr_score_stap']) - float(r['corr_score_base']) for r in rows]
    dnrmse = [float(r['nrmse_score_stap']) - float(r['nrmse_score_base']) for r in rows]
    dbg = [float(r['bg_var_ratio_stap']) - float(r['bg_var_ratio_base']) for r in rows]
    sec = {
        'source': str(path.relative_to(ROOT)),
        'delta_corr_score': _bootstrap_ci(dcorr),
        'delta_corr_win_rate': _win_rate(dcorr),
        'delta_nrmse': _bootstrap_ci(dnrmse),
        'delta_nrmse_win_rate': _win_rate(dnrmse, better='negative'),
        'delta_bg_var_ratio': _bootstrap_ci(dbg),
        'delta_bg_var_win_rate': _win_rate(dbg, better='negative'),
    }
    flat = [
        {'domain': tag, 'metric': 'delta_corr_score_mean', **sec['delta_corr_score']},
        {'domain': tag, 'metric': 'delta_corr_win_rate', **sec['delta_corr_win_rate']},
        {'domain': tag, 'metric': 'delta_nrmse_mean', **sec['delta_nrmse']},
        {'domain': tag, 'metric': 'delta_nrmse_win_rate', **sec['delta_nrmse_win_rate']},
        {'domain': tag, 'metric': 'delta_bg_var_ratio_mean', **sec['delta_bg_var_ratio']},
        {'domain': tag, 'metric': 'delta_bg_var_win_rate', **sec['delta_bg_var_win_rate']},
    ]
    return sec, flat


def gammex_consistency() -> tuple[dict[str, Any], list[dict[str, Any]]]:
    along = _load_json(ROOT / 'reports' / 'twinkling_gammex_alonglinear17_prf2500_str6_msd_ka_with_baselines_summary.json')
    across = _load_json(ROOT / 'reports' / 'twinkling_gammex_across17_prf2500_str4_msd_ka_with_baselines_summary.json')
    def _state_rate(obj: dict[str, Any], state: str) -> float:
        counts = obj['ka_state_counts']
        total = float(sum(counts.values()))
        return float(counts.get(state, 0) / total) if total > 0 else float('nan')
    sec = {
        'along_summary_source': 'reports/twinkling_gammex_alonglinear17_prf2500_str6_msd_ka_with_baselines_summary.json',
        'across_summary_source': 'reports/twinkling_gammex_across17_prf2500_str4_msd_ka_with_baselines_summary.json',
        'along_actionable_rate': _state_rate(along, 'C1_SAFETY') + _state_rate(along, 'C2_UPLIFT'),
        'across_actionable_rate': _state_rate(across, 'C1_SAFETY') + _state_rate(across, 'C2_UPLIFT'),
        'across_tpr_gain_at_fpr1e3': float((across.get('stap_tpr_at_fpr1e3') or 0.0) - (across.get('base_tpr_at_fpr1e3') or 0.0)),
    }
    flat = [
        {'domain': 'gammex', 'metric': 'along_actionable_rate', 'center': sec['along_actionable_rate'], 'lo': None, 'hi': None, 'n': None},
        {'domain': 'gammex', 'metric': 'across_actionable_rate', 'center': sec['across_actionable_rate'], 'lo': None, 'hi': None, 'n': None},
        {'domain': 'gammex', 'metric': 'across_tpr_gain_at_fpr1e3', 'center': sec['across_tpr_gain_at_fpr1e3'], 'lo': None, 'hi': None, 'n': None},
    ]
    return sec, flat


def _md(sections: dict[str, Any]) -> str:
    out = ['# Clinical Translation Uncertainty Summary', '']
    m2 = sections['mace_phase2']
    out += [
        '## Macé phase-2',
        f"- FP reduction fraction: {100*m2['fp_reduction_fraction']['center']:.1f}% (95% CI {100*m2['fp_reduction_fraction']['lo']:.1f} to {100*m2['fp_reduction_fraction']['hi']:.1f}; n={m2['fp_reduction_fraction']['n']})",
        f"- Gate-kept fraction: {100*m2['gate_kept_fraction']['center']:.1f}% (95% CI {100*m2['gate_kept_fraction']['lo']:.1f} to {100*m2['gate_kept_fraction']['hi']:.1f})",
        f"- Plane-wise win rate for FP reduction: {100*m2['fp_reduction_win_rate']['win_rate']:.1f}% ({m2['fp_reduction_win_rate']['wins']}/{m2['fp_reduction_win_rate']['n']})",
        '',
    ]
    mh = sections['mace_holdout']
    out += [
        '## Macé held-out alias gate',
        f"- All-plane FP reduction fraction: {100*mh['all_planes_fp_reduction_fraction']['center']:.1f}% (95% CI {100*mh['all_planes_fp_reduction_fraction']['lo']:.1f} to {100*mh['all_planes_fp_reduction_fraction']['hi']:.1f})",
        f"- All-plane hit retention: {100*mh['all_planes_hit_retention']['center']:.1f}% (95% CI {100*mh['all_planes_hit_retention']['lo']:.1f} to {100*mh['all_planes_hit_retention']['hi']:.1f})",
        f"- All-plane safety win rate: {100*mh['all_planes_safety_win_rate']['center']:.1f}% (95% CI {100*mh['all_planes_safety_win_rate']['lo']:.1f} to {100*mh['all_planes_safety_win_rate']['hi']:.1f})",
        '',
    ]
    sh = sections['shin']
    out += [
        '## Shin',
        f"- ΔAUC(STAP−MC-SVD): {sh['delta_auc_stap_minus_mcsvd']['center']:.3f} (95% CI {sh['delta_auc_stap_minus_mcsvd']['lo']:.3f} to {sh['delta_auc_stap_minus_mcsvd']['hi']:.3f})",
        f"- Win rate for positive ΔAUC: {100*sh['delta_auc_win_rate']['win_rate']:.1f}% ({sh['delta_auc_win_rate']['wins']}/{sh['delta_auc_win_rate']['n']})",
        f"- Median reg_shift_p90: {sh['reg_shift_p90_px_median']['center']:.4f} px (95% CI {sh['reg_shift_p90_px_median']['lo']:.4f} to {sh['reg_shift_p90_px_median']['hi']:.4f})",
        '',
    ]
    for key, title in [('ulm_brainlike', 'ULM brainlike'), ('ulm_elastic', 'ULM elastic')]:
        s = sections[key]
        out += [
            f'## {title}',
            f"- Δcorr(STAP−base): {s['delta_corr_score']['center']:.3f} (95% CI {s['delta_corr_score']['lo']:.3f} to {s['delta_corr_score']['hi']:.3f})",
            f"- ΔnRMSE(STAP−base): {s['delta_nrmse']['center']:.3f} (95% CI {s['delta_nrmse']['lo']:.3f} to {s['delta_nrmse']['hi']:.3f})",
            f"- Δbg-var-ratio(STAP−base): {s['delta_bg_var_ratio']['center']:.3f} (95% CI {s['delta_bg_var_ratio']['lo']:.3f} to {s['delta_bg_var_ratio']['hi']:.3f})",
            f"- Win rates: corr {100*s['delta_corr_win_rate']['win_rate']:.1f}%, nRMSE {100*s['delta_nrmse_win_rate']['win_rate']:.1f}%, bg-var {100*s['delta_bg_var_win_rate']['win_rate']:.1f}%",
            '',
        ]
    g = sections['gammex']
    out += [
        '## Gammex',
        f"- Along actionable rate: {100*g['along_actionable_rate']:.1f}%",
        f"- Across actionable rate: {100*g['across_actionable_rate']:.1f}%",
        f"- Across pooled TPR gain at FPR=1e-3: {g['across_tpr_gain_at_fpr1e3']:.3f}",
        '',
    ]
    return '\n'.join(out)


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    sections = {}
    flat: list[dict[str, Any]] = []
    for key, fn in [
        ('mace_phase2', mace_phase2),
        ('mace_holdout', mace_holdout),
        ('shin', shin_motion),
        ('ulm_brainlike', lambda: ulm_motion('ulm_brainlike', ROOT / 'reports' / 'ulm7883227_motion_sweep_ULM_blocks1_3_0000_0128_brainlike_e975.csv')),
        ('ulm_elastic', lambda: ulm_motion('ulm_elastic', ROOT / 'reports' / 'ulm7883227_motion_sweep_ULM_blocks1_3_0000_0128_elastic_e975.csv')),
        ('gammex', gammex_consistency),
    ]:
        sec, rows = fn()
        sections[key] = sec
        flat.extend(rows)
    (OUT_DIR / 'clinical_translation_uncertainty.json').write_text(json.dumps({'sections': sections}, indent=2, sort_keys=True, allow_nan=True))
    with (OUT_DIR / 'clinical_translation_uncertainty.csv').open('w', newline='') as f:
        cols = ['domain','metric','center','lo','hi','n','wins','ties','losses','win_rate']
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        for row in flat:
            w.writerow({k: row.get(k) for k in cols})
    (OUT_DIR / 'clinical_translation_uncertainty.md').write_text(_md(sections))
    print('wrote', OUT_DIR / 'clinical_translation_uncertainty.json')
    print('wrote', OUT_DIR / 'clinical_translation_uncertainty.csv')
    print('wrote', OUT_DIR / 'clinical_translation_uncertainty.md')


if __name__ == '__main__':
    main()
