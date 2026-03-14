#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import dataclasses
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

import numpy as np

from scripts.simus_eval_structural import evaluate_structural_metrics
from scripts.simus_structural_candidate_compare import DEFAULT_CANDIDATES
from sim.simus.bundle import derive_bundle_from_run
from sim.simus.config import (
    BackgroundCompartmentSpec,
    MotionSpec,
    PhaseScreenSpec,
    SimusConfig,
    StructuredClutterSpec,
    default_profile_config,
)
from sim.simus.pilot_pymust_simus import write_simus_run


PROFILE_LABELS = {
    "ClinMobile-Pf-v2": "Mobile",
    "ClinIntraOpParenchyma-Pf-v3": "Intra-operative parenchymal",
}

STRESS_CITATIONS = {
    "bulk_motion": r"\cite{DemeneClutterSVD2015,Soloukey2020FUSAwake,MobileFUS2025}",
    "nuisance_reflectivity": r"\cite{DemeneClutterSVD2015,BarangerAdaptiveSVD2018}",
    "cardiac_pulsation": r"\cite{BarangerAdaptiveSVD2018}",
    "short_ensemble": r"\cite{Imbault2017,Soloukey2020FUSAwake,MobileFUS2025}",
}


@dataclass(frozen=True)
class StressLevel:
    level: str
    label: str
    apply: Callable[[SimusConfig], SimusConfig]
    value_text: Callable[[SimusConfig], str]


@dataclass(frozen=True)
class StressAxis:
    key: str
    label: str
    description: str
    provenance_text: str
    levels: tuple[StressLevel, ...]


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        raise ValueError("no rows")
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames: list[str] = []
    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _fmt(value: float | None, digits: int = 3) -> str:
    if value is None or not np.isfinite(value):
        return "--"
    return f"{float(value):.{digits}f}"


def _score_path(bundle_dir: Path, *, role: str) -> Path:
    if role == "baseline":
        candidates = [bundle_dir / "score_pd_base.npy", bundle_dir / "score_base.npy"]
    else:
        candidates = [bundle_dir / "score_stap_preka.npy", bundle_dir / "score_stap.npy"]
    return next((p for p in candidates if p.is_file()), candidates[0])


def _candidate_overrides() -> dict[str, Any]:
    cand = next(c for c in DEFAULT_CANDIDATES if c.name == "unwhitened_ref")
    return {
        "stap_detector_variant": cand.detector_variant,
        "stap_whiten_gamma": float(cand.whiten_gamma),
        "diag_load": float(cand.diag_load),
        "cov_estimator": str(cand.cov_estimator),
        "huber_c": float(cand.huber_c),
        "stap_cov_train_trim_q": float(cand.stap_cov_train_trim_q),
        "mvdr_auto_kappa": float(cand.mvdr_auto_kappa),
        "constraint_ridge": float(cand.constraint_ridge),
        "stap_conditional_enable": False,
    }


def _scale_noncardiac_motion(spec: MotionSpec, scale: float) -> MotionSpec:
    scale = float(scale)
    if not spec.enabled:
        return spec
    return MotionSpec(
        enabled=True,
        breathing_hz=float(spec.breathing_hz),
        breathing_amp_x_px=float(spec.breathing_amp_x_px) * scale,
        breathing_amp_z_px=float(spec.breathing_amp_z_px) * scale,
        cardiac_hz=float(spec.cardiac_hz),
        cardiac_amp_x_px=float(spec.cardiac_amp_x_px),
        cardiac_amp_z_px=float(spec.cardiac_amp_z_px),
        random_walk_sigma_px=float(spec.random_walk_sigma_px) * scale,
        pulse_jitter_sigma_px=float(spec.pulse_jitter_sigma_px) * scale,
        drift_x_px=float(spec.drift_x_px) * scale,
        drift_z_px=float(spec.drift_z_px) * scale,
        elastic_amp_px=float(spec.elastic_amp_px) * scale,
        elastic_sigma_px=float(spec.elastic_sigma_px),
        elastic_depth_decay_frac=float(spec.elastic_depth_decay_frac),
        elastic_temporal_rho=float(spec.elastic_temporal_rho),
        elastic_mode_count=int(spec.elastic_mode_count),
        elastic_lateral_scale=float(spec.elastic_lateral_scale),
        elastic_axial_scale=float(spec.elastic_axial_scale),
    )


def _scale_cardiac_only(spec: MotionSpec, scale: float) -> MotionSpec:
    scale = float(scale)
    if not spec.enabled:
        return spec
    return MotionSpec(
        enabled=True,
        breathing_hz=float(spec.breathing_hz),
        breathing_amp_x_px=float(spec.breathing_amp_x_px),
        breathing_amp_z_px=float(spec.breathing_amp_z_px),
        cardiac_hz=float(spec.cardiac_hz),
        cardiac_amp_x_px=float(spec.cardiac_amp_x_px) * scale,
        cardiac_amp_z_px=float(spec.cardiac_amp_z_px) * scale,
        random_walk_sigma_px=float(spec.random_walk_sigma_px),
        pulse_jitter_sigma_px=float(spec.pulse_jitter_sigma_px),
        drift_x_px=float(spec.drift_x_px),
        drift_z_px=float(spec.drift_z_px),
        elastic_amp_px=float(spec.elastic_amp_px),
        elastic_sigma_px=float(spec.elastic_sigma_px),
        elastic_depth_decay_frac=float(spec.elastic_depth_decay_frac),
        elastic_temporal_rho=float(spec.elastic_temporal_rho),
        elastic_mode_count=int(spec.elastic_mode_count),
        elastic_lateral_scale=float(spec.elastic_lateral_scale),
        elastic_axial_scale=float(spec.elastic_axial_scale),
    )


def _scale_phase_screen(spec: PhaseScreenSpec, scale: float) -> PhaseScreenSpec:
    scale = float(scale)
    if not spec.enabled:
        return spec
    return PhaseScreenSpec(
        enabled=True,
        std_rad=float(spec.std_rad) * scale,
        corr_len_elem=float(spec.corr_len_elem),
        drift_rho=float(spec.drift_rho),
        drift_sigma_rad=float(spec.drift_sigma_rad) * scale,
    )


def _scale_background_rc(cfg: SimusConfig, scale: float) -> SimusConfig:
    scale = float(scale)
    return dataclasses.replace(
        cfg,
        tissue_rc_scale=float(cfg.tissue_rc_scale) * scale,
        structured_clutter=tuple(
            dataclasses.replace(item, rc_scale=float(item.rc_scale) * scale) for item in tuple(cfg.structured_clutter)
        ),
        background_compartments=tuple(
            dataclasses.replace(item, rc_scale=float(item.rc_scale) * scale)
            for item in tuple(cfg.background_compartments)
        ),
    )


def _nominal_reflectivity_shift_db(scale: float) -> str:
    return f"{20.0 * np.log10(float(scale)):+.1f} dB"


def _motion_value_text(scale: float) -> Callable[[SimusConfig], str]:
    def _inner(cfg: SimusConfig) -> str:
        return f"{scale:.2f}x bulk motion / phase nuisance"

    return _inner


def _cardiac_value_text(scale: float) -> Callable[[SimusConfig], str]:
    def _inner(cfg: SimusConfig) -> str:
        hz = float(cfg.motion.cardiac_hz)
        return f"{scale:.1f}x cardiac amplitude at {hz:.2f} Hz"

    return _inner


def _reflectivity_value_text(scale: float) -> Callable[[SimusConfig], str]:
    def _inner(cfg: SimusConfig) -> str:
        return f"{_nominal_reflectivity_shift_db(scale)} nuisance reflectivity"

    return _inner


def _ensemble_value_text(T: int) -> Callable[[SimusConfig], str]:
    def _inner(cfg: SimusConfig) -> str:
        duration_ms = 1e3 * float(T) / max(float(cfg.prf_hz), 1e-9)
        return f"{int(T)} frames ({duration_ms:.1f} ms)"

    return _inner


def _stress_axes() -> tuple[StressAxis, ...]:
    return (
        StressAxis(
            key="bulk_motion",
            label="Bulk tissue motion",
            description="Profile-based rigid and elastic tissue motion scaled around the held-out setting.",
            provenance_text=(
                "Stress axis motivated by reported fUS sensitivity to mm/s-scale tissue motion in bedside and "
                "intra-operative settings."
            ),
            levels=(
                StressLevel(
                    level="reference",
                    label="Reference",
                    apply=lambda cfg: dataclasses.replace(
                        cfg,
                        profile=f"{cfg.profile}+bulkmotion_ref",
                    ),
                    value_text=_motion_value_text(1.0),
                ),
                StressLevel(
                    level="moderate",
                    label="Moderate",
                    apply=lambda cfg: dataclasses.replace(
                        cfg,
                        profile=f"{cfg.profile}+bulkmotion_mod",
                        motion=_scale_noncardiac_motion(cfg.motion, 1.75),
                        phase_screen=_scale_phase_screen(cfg.phase_screen, 1.50),
                    ),
                    value_text=_motion_value_text(1.75),
                ),
                StressLevel(
                    level="hard",
                    label="Hard",
                    apply=lambda cfg: dataclasses.replace(
                        cfg,
                        profile=f"{cfg.profile}+bulkmotion_hard",
                        motion=_scale_noncardiac_motion(cfg.motion, 2.75),
                        phase_screen=_scale_phase_screen(cfg.phase_screen, 2.00),
                    ),
                    value_text=_motion_value_text(2.75),
                ),
            ),
        ),
        StressAxis(
            key="nuisance_reflectivity",
            label="Nuisance reflectivity",
            description="Structured-clutter and background reflectivity proxy increased relative to microvascular flow.",
            provenance_text=(
                "Stress axis motivated by the large tissue-to-blood amplitude disparity reported in ultrafast "
                "Doppler / fUS clutter-filtering studies."
            ),
            levels=(
                StressLevel(
                    level="reference",
                    label="Reference",
                    apply=lambda cfg: dataclasses.replace(cfg, profile=f"{cfg.profile}+reflect_ref"),
                    value_text=_reflectivity_value_text(1.0),
                ),
                StressLevel(
                    level="moderate",
                    label="Moderate",
                    apply=lambda cfg: dataclasses.replace(
                        _scale_background_rc(cfg, 2.0),
                        profile=f"{cfg.profile}+reflect_mod",
                    ),
                    value_text=_reflectivity_value_text(2.0),
                ),
                StressLevel(
                    level="hard",
                    label="Hard",
                    apply=lambda cfg: dataclasses.replace(
                        _scale_background_rc(cfg, 4.0),
                        profile=f"{cfg.profile}+reflect_hard",
                    ),
                    value_text=_reflectivity_value_text(4.0),
                ),
            ),
        ),
        StressAxis(
            key="cardiac_pulsation",
            label="Cardiac pulsation",
            description="Cardiac-like tissue-motion component amplified while the rest of the motion field is held fixed.",
            provenance_text=(
                "Stress axis motivated by neonatal and open-skull reports of cardiac-like tissue pulsation broadening "
                "the low-frequency tissue spectrum."
            ),
            levels=(
                StressLevel(
                    level="reference",
                    label="Reference",
                    apply=lambda cfg: dataclasses.replace(cfg, profile=f"{cfg.profile}+cardiac_ref"),
                    value_text=_cardiac_value_text(1.0),
                ),
                StressLevel(
                    level="moderate",
                    label="Moderate",
                    apply=lambda cfg: dataclasses.replace(
                        cfg,
                        profile=f"{cfg.profile}+cardiac_mod",
                        motion=_scale_cardiac_only(cfg.motion, 2.0),
                    ),
                    value_text=_cardiac_value_text(2.0),
                ),
                StressLevel(
                    level="hard",
                    label="Hard",
                    apply=lambda cfg: dataclasses.replace(
                        cfg,
                        profile=f"{cfg.profile}+cardiac_hard",
                        motion=_scale_cardiac_only(cfg.motion, 3.0),
                    ),
                    value_text=_cardiac_value_text(3.0),
                ),
            ),
        ),
        StressAxis(
            key="short_ensemble",
            label="Short ensemble",
            description="Slow-time support reduced around the 64-frame held-out setting.",
            provenance_text=(
                "Stress axis motivated by clinically constrained acquisitions in which shorter ensembles are used "
                "to limit motion corruption or workflow burden."
            ),
            levels=(
                StressLevel(
                    level="reference",
                    label="Reference",
                    apply=lambda cfg: dataclasses.replace(cfg, profile=f"{cfg.profile}+ens64", T=64),
                    value_text=_ensemble_value_text(64),
                ),
                StressLevel(
                    level="moderate",
                    label="Moderate",
                    apply=lambda cfg: dataclasses.replace(cfg, profile=f"{cfg.profile}+ens48", T=48),
                    value_text=_ensemble_value_text(48),
                ),
                StressLevel(
                    level="hard",
                    label="Hard",
                    apply=lambda cfg: dataclasses.replace(cfg, profile=f"{cfg.profile}+ens32", T=32),
                    value_text=_ensemble_value_text(32),
                ),
            ),
        ),
    )


def _motion_speed_mm_s(run_dir: Path) -> float | None:
    meta = json.loads((Path(run_dir) / "dataset" / "meta.json").read_text(encoding="utf-8"))
    debug_dir = Path(run_dir) / "dataset" / "debug"
    dx_path = debug_dir / "motion_rigid_dx_px.npy"
    dz_path = debug_dir / "motion_rigid_dz_px.npy"
    if not dx_path.is_file() or not dz_path.is_file():
        return None
    dx = np.load(dx_path).astype(np.float64, copy=False)
    dz = np.load(dz_path).astype(np.float64, copy=False)
    if dx.size < 2 or dz.size < 2:
        return None
    grid = meta.get("grid", {})
    prf_hz = float(meta.get("acquisition", {}).get("prf_hz", 0.0))
    dx_m = float(grid.get("dx_m", 0.0))
    dz_m = float(grid.get("dz_m", 0.0))
    d_disp_m = np.sqrt((np.diff(dx) * dx_m) ** 2 + (np.diff(dz) * dz_m) ** 2)
    return float(np.quantile(d_disp_m * prf_hz * 1e3, 0.90)) if d_disp_m.size else None


def _bundle_name(run_dir: Path, profile_key: str, axis_key: str, level: str, role: str) -> str:
    return f"{Path(run_dir).name}_{profile_key}_{axis_key}_{level}_{role}"


def _build_tables(
    summary_rows: list[dict[str, Any]],
    out_dir: Path,
    *,
    axes: tuple[StressAxis, ...],
    seeds: list[int],
    tier: str,
) -> None:
    grouped: dict[str, list[dict[str, Any]]] = {}
    for row in summary_rows:
        grouped.setdefault(str(row["setting_label"]), []).append(row)
    level_order = {"reference": 0, "moderate": 1, "hard": 2}
    axis_citation_map = {axis.key: STRESS_CITATIONS[axis.key] for axis in axes}

    for setting_label, items in grouped.items():
        items = sorted(
            items,
            key=lambda row: (
                str(row["axis_label"]),
                level_order.get(str(row["level"]), 99),
            ),
        )
        lines = [
            r"\begin{table}[t]",
            r"\centering",
            r"\scriptsize",
            r"\setlength{\tabcolsep}{4pt}",
            r"\resizebox{\linewidth}{!}{%",
            r"\begin{tabular}{@{}P{2.6cm} P{1.4cm} P{3.4cm} C{1.35cm} C{1.35cm} C{1.35cm} C{1.35cm}@{}}",
            r"\hline",
            r"Stress axis & Level & Applied perturbation & \shortstack{PD\\AUC$_{\mathrm{main/nuis}}$} & \shortstack{Default\\AUC$_{\mathrm{main/nuis}}$} & \shortstack{PD\\FPR$_{\mathrm{nuis}}$@$0.5$} & \shortstack{Default\\FPR$_{\mathrm{nuis}}$@$0.5$} \\",
            r"\hline",
        ]
        for row in items:
            axis_with_citation = r"\shortstack[l]{%s\\%s}" % (
                str(row["axis_label"]),
                axis_citation_map[str(row["axis_key"])],
            )
            lines.append(
                " & ".join(
                    [
                        axis_with_citation,
                        str(row["level_label"]),
                        str(row["value_text"]),
                        _fmt(row.get("baseline_auc_main_vs_nuisance")),
                        _fmt(row.get("default_auc_main_vs_nuisance")),
                        _fmt(row.get("baseline_fpr_nuisance_match_0p5")),
                        _fmt(row.get("default_fpr_nuisance_match_0p5")),
                    ]
                )
                + r" \\"
            )
        lines.extend(
            [
                r"\hline",
                r"\end{tabular}",
                r"}",
                (
                    r"\caption{Clinically grounded reduced-grid SIMUS stress audit on the %s scene family "
                    r"(two held-out realizations). Each row perturbs one axis at a time around the "
                    r"prespecified mobile setting, keeps the MC--SVD residualizer fixed, and compares the "
                    r"same-residual PD score to the fixed band-limited matched-subspace default. The purpose "
                    r"of this supplementary audit is not to create a second headline benchmark, but to show "
                    r"that the task becomes visibly harder under literature-grounded nuisance perturbations.}"
                )
                % (setting_label.lower(),),
                r"\label{tab:simus_stress_" + setting_label.lower().replace(" ", "_").replace("-", "") + r"}",
                r"\end{table}",
                "",
            ]
        )
        (out_dir / f"simus_clinical_stress_{setting_label.lower().replace(' ', '_').replace('-', '')}_table.tex").write_text(
            "\n".join(lines) + "\n",
            encoding="utf-8",
        )

    provenance_lines = [
        r"\begin{table}[t]",
        r"\centering",
        r"\scriptsize",
        r"\setlength{\tabcolsep}{4pt}",
        r"\begin{tabular}{@{}P{2.6cm} P{4.8cm} P{5.2cm}@{}}",
        r"\hline",
        r"Stress axis & Supplementary perturbation used here & Literature grounding \\",
        r"\hline",
    ]
    for axis in axes:
        level_text = "; ".join(f"{lvl.label}: {lvl.value_text(default_profile_config(profile='ClinMobile-Pf-v2', tier='paper', seed=127))}" for lvl in axis.levels)
        provenance_lines.append(
            " & ".join(
                [
                    axis.label,
                    level_text,
                    axis.provenance_text + " " + STRESS_CITATIONS[axis.key],
                ]
            )
            + r" \\"
        )
    provenance_lines.extend(
        [
            r"\hline",
            r"\end{tabular}",
            r"\caption{Clinically grounded SIMUS stress axes used in the supplementary reduced-grid audit. These are literature-motivated perturbations around the prespecified mobile scene family rather than a claim of exact one-to-one replay of any single in vivo operating condition.}",
            r"\label{tab:simus_clinical_stress_provenance}",
            r"\end{table}",
            "",
        ]
    )
    (out_dir / "simus_clinical_stress_provenance_table.tex").write_text("\n".join(provenance_lines) + "\n", encoding="utf-8")


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Held-out clinically grounded SIMUS stress sweep.")
    ap.add_argument(
        "--profiles",
        type=str,
        default="ClinMobile-Pf-v2,ClinIntraOpParenchyma-Pf-v3",
    )
    ap.add_argument(
        "--axes",
        type=str,
        default="bulk_motion,nuisance_reflectivity,cardiac_pulsation,short_ensemble",
    )
    ap.add_argument("--seeds", type=str, default="127,128")
    ap.add_argument("--tier", type=str, default="paper", choices=["smoke", "paper"])
    ap.add_argument("--stap-profile", type=str, default="Brain-SIMUS-Clin-MotionRobust-v0")
    ap.add_argument("--stap-device", type=str, default="cuda")
    ap.add_argument("--out-root", type=Path, default=Path("runs/simus_clinical_stress"))
    ap.add_argument("--out-json", type=Path, default=Path("reports/simus_clinical_stress.json"))
    ap.add_argument("--out-csv", type=Path, default=Path("reports/simus_clinical_stress.csv"))
    ap.add_argument(
        "--out-tex-dir",
        type=Path,
        default=Path("reports"),
    )
    ap.add_argument("--reuse-existing", action=argparse.BooleanOptionalAction, default=True)
    return ap.parse_args()


def main() -> None:
    args = parse_args()
    profiles = [x.strip() for x in str(args.profiles).split(",") if x.strip()]
    requested_axes = {x.strip() for x in str(args.axes).split(",") if x.strip()}
    seeds = [int(x.strip()) for x in str(args.seeds).split(",") if x.strip()]
    candidate_overrides = _candidate_overrides()
    out_root = Path(args.out_root)
    out_root.mkdir(parents=True, exist_ok=True)
    axes = tuple(axis for axis in _stress_axes() if axis.key in requested_axes)
    if not axes:
        raise ValueError(f"no supported axes selected from {sorted(requested_axes)}")

    detail_rows: list[dict[str, Any]] = []

    for profile in profiles:
        if profile not in PROFILE_LABELS:
            raise ValueError(f"unsupported profile {profile!r}")
        setting_label = PROFILE_LABELS[profile]
        for axis in axes:
            for level in axis.levels:
                for seed in seeds:
                    cfg = default_profile_config(profile=profile, tier=str(args.tier), seed=int(seed))  # type: ignore[arg-type]
                    cfg = level.apply(cfg)
                    run_dir = out_root / f"{profile}_seed{seed}_{axis.key}_{level.level}"
                    dataset_meta_path = run_dir / "dataset" / "meta.json"
                    if not bool(args.reuse_existing) or not dataset_meta_path.is_file():
                        write_simus_run(out_root=run_dir, cfg=cfg, skip_bundle=True)
                    motion_speed_mm_s = _motion_speed_mm_s(run_dir)
                    baseline_bundle_dir = derive_bundle_from_run(
                        run_dir=run_dir,
                        out_root=run_dir / "bundles",
                        dataset_name=_bundle_name(run_dir, profile, axis.key, level.level, "baseline"),
                        stap_profile=str(args.stap_profile),
                        baseline_type="mc_svd",
                        run_stap=False,
                        stap_device=str(args.stap_device),
                        bundle_overrides={},
                        meta_extra={
                            "stress_axis": axis.key,
                            "stress_level": level.level,
                            "stress_value_text": level.value_text(cfg),
                        },
                    )
                    default_bundle_dir = derive_bundle_from_run(
                        run_dir=run_dir,
                        out_root=run_dir / "bundles",
                        dataset_name=_bundle_name(run_dir, profile, axis.key, level.level, "default"),
                        stap_profile=str(args.stap_profile),
                        baseline_type="mc_svd",
                        run_stap=True,
                        stap_device=str(args.stap_device),
                        bundle_overrides=candidate_overrides,
                        meta_extra={
                            "stress_axis": axis.key,
                            "stress_level": level.level,
                            "stress_value_text": level.value_text(cfg),
                        },
                    )
                    _, masks, _ = derive_masks = (None, None, None)  # type: ignore[assignment]
                    from sim.simus.bundle import load_canonical_run

                    _, masks, _ = load_canonical_run(run_dir)
                    mask_h1_pf_main = masks["mask_h1_pf_main"]
                    mask_h0_bg = masks["mask_h0_bg"]
                    mask_h0_nuisance_pa = masks.get("mask_h0_nuisance_pa")
                    mask_h1_alias_qc = masks.get("mask_h1_alias_qc")

                    baseline_score = np.load(_score_path(baseline_bundle_dir, role="baseline")).astype(np.float32, copy=False)
                    default_score = np.load(_score_path(default_bundle_dir, role="default")).astype(np.float32, copy=False)
                    baseline_metrics = evaluate_structural_metrics(
                        score=baseline_score,
                        mask_h1_pf_main=mask_h1_pf_main,
                        mask_h0_bg=mask_h0_bg,
                        mask_h0_nuisance_pa=mask_h0_nuisance_pa,
                        mask_h1_alias_qc=mask_h1_alias_qc,
                        fprs=[1e-3],
                        match_tprs=[0.5],
                    )
                    default_metrics = evaluate_structural_metrics(
                        score=default_score,
                        mask_h1_pf_main=mask_h1_pf_main,
                        mask_h0_bg=mask_h0_bg,
                        mask_h0_nuisance_pa=mask_h0_nuisance_pa,
                        mask_h1_alias_qc=mask_h1_alias_qc,
                        fprs=[1e-3],
                        match_tprs=[0.5],
                    )
                    detail_rows.append(
                        {
                            "profile": profile,
                            "setting_label": setting_label,
                            "seed": int(seed),
                            "axis_key": axis.key,
                            "axis_label": axis.label,
                            "level": level.level,
                            "level_label": level.label,
                            "value_text": level.value_text(cfg),
                            "motion_speed_p90_mm_s": motion_speed_mm_s,
                            "baseline_auc_main_vs_bg": baseline_metrics.get("auc_main_vs_bg"),
                            "baseline_auc_main_vs_nuisance": baseline_metrics.get("auc_main_vs_nuisance"),
                            "baseline_fpr_nuisance_match_0p5": baseline_metrics.get("fpr_nuisance_match@0p5"),
                            "baseline_tpr_main_1e-03": baseline_metrics.get("tpr_main@1e-03"),
                            "default_auc_main_vs_bg": default_metrics.get("auc_main_vs_bg"),
                            "default_auc_main_vs_nuisance": default_metrics.get("auc_main_vs_nuisance"),
                            "default_fpr_nuisance_match_0p5": default_metrics.get("fpr_nuisance_match@0p5"),
                            "default_tpr_main_1e-03": default_metrics.get("tpr_main@1e-03"),
                        }
                    )

    summary_rows: list[dict[str, Any]] = []
    summary_keys = ("profile", "setting_label", "axis_key", "axis_label", "level", "level_label", "value_text")
    grouped: dict[tuple[Any, ...], list[dict[str, Any]]] = {}
    for row in detail_rows:
        key = tuple(row[k] for k in summary_keys)
        grouped.setdefault(key, []).append(row)

    metric_keys = [
        "motion_speed_p90_mm_s",
        "baseline_auc_main_vs_bg",
        "baseline_auc_main_vs_nuisance",
        "baseline_fpr_nuisance_match_0p5",
        "baseline_tpr_main_1e-03",
        "default_auc_main_vs_bg",
        "default_auc_main_vs_nuisance",
        "default_fpr_nuisance_match_0p5",
        "default_tpr_main_1e-03",
    ]
    for key, items in sorted(grouped.items()):
        rec = {name: value for name, value in zip(summary_keys, key)}
        rec["count"] = int(len(items))
        for metric_key in metric_keys:
            vals = np.asarray(
                [float(item[metric_key]) for item in items if item.get(metric_key) is not None and np.isfinite(item.get(metric_key))],
                dtype=np.float64,
            )
            rec[metric_key] = float(np.mean(vals)) if vals.size else None
        rec["delta_auc_main_vs_nuisance"] = (
            (float(rec["default_auc_main_vs_nuisance"]) if rec["default_auc_main_vs_nuisance"] is not None else np.nan)
            - (float(rec["baseline_auc_main_vs_nuisance"]) if rec["baseline_auc_main_vs_nuisance"] is not None else np.nan)
        )
        rec["delta_fpr_nuisance_match_0p5"] = (
            (float(rec["default_fpr_nuisance_match_0p5"]) if rec["default_fpr_nuisance_match_0p5"] is not None else np.nan)
            - (float(rec["baseline_fpr_nuisance_match_0p5"]) if rec["baseline_fpr_nuisance_match_0p5"] is not None else np.nan)
        )
        summary_rows.append(rec)

    _write_csv(Path(args.out_csv), detail_rows)
    _write_json(
        Path(args.out_json),
        {
            "schema_version": "simus_clinical_stress.v1",
            "profiles": profiles,
            "axes": sorted(requested_axes),
            "seeds": seeds,
            "stap_profile": str(args.stap_profile),
            "rows": detail_rows,
            "summary_rows": summary_rows,
        },
    )
    _build_tables(
        summary_rows,
        Path(args.out_tex_dir),
        axes=axes,
        seeds=seeds,
        tier=str(args.tier),
    )


if __name__ == "__main__":
    main()
