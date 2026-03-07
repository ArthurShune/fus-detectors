#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

import numpy as np

from scripts.simus_eval_structural import (
    _baseline_label,
    _headline_label,
    _pipeline_label,
    _score_label,
    _score_semantics,
    evaluate_structural_metrics,
)
from sim.simus.bundle import (
    SUPPORTED_SIMUS_STAP_PROFILES,
    derive_bundle_from_run,
    load_canonical_run,
    slugify,
)


def _parse_case(spec: str) -> tuple[str, Path]:
    parts = str(spec).split("::", 1)
    if len(parts) != 2:
        raise ValueError(f"expected name::run_dir, got {spec!r}")
    return parts[0], Path(parts[1])


def _split_csv_list(spec: str) -> list[str]:
    return [s.strip() for s in str(spec or "").split(",") if s.strip()]


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


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, sort_keys=True)
        f.write("\n")


@dataclass(frozen=True)
class MethodSpec:
    key: str
    baseline_type: str
    run_stap: bool
    role: str


@dataclass(frozen=True)
class CandidateSpec:
    method_family: str
    config_name: str
    method: MethodSpec
    stap_profile: str
    override_builder: Callable[[tuple[int, int, int]], dict[str, Any]]


def _identity_overrides(_: tuple[int, int, int]) -> dict[str, Any]:
    return {}


def _rpca_default_lambda(shape: tuple[int, int, int]) -> float:
    T, H, W = (int(v) for v in shape)
    return float(1.0 / np.sqrt(float(max(T, H * W))))


def _candidate_specs() -> list[CandidateSpec]:
    out: list[CandidateSpec] = []

    def add(
        *,
        method_family: str,
        config_name: str,
        baseline_type: str,
        run_stap: bool,
        role: str,
        stap_profile: str = "Brain-SIMUS-Clin",
        override_builder: Callable[[tuple[int, int, int]], dict[str, Any]] = _identity_overrides,
    ) -> None:
        out.append(
            CandidateSpec(
                method_family=method_family,
                config_name=config_name,
                method=MethodSpec(
                    key=f"{method_family}:{config_name}",
                    baseline_type=str(baseline_type),
                    run_stap=bool(run_stap),
                    role=str(role),
                ),
                stap_profile=str(stap_profile),
                override_builder=override_builder,
            )
        )

    for profile in SUPPORTED_SIMUS_STAP_PROFILES:
        add(
            method_family="stap",
            config_name=profile,
            baseline_type="mc_svd",
            run_stap=True,
            role="stap",
            stap_profile=profile,
        )

    for energy in (0.80, 0.85, 0.90, 0.95):
        add(
            method_family="mc_svd",
            config_name=f"ef{int(round(100 * energy))}",
            baseline_type="mc_svd",
            run_stap=False,
            role="baseline",
            override_builder=lambda shape, energy=energy: {"svd_energy_frac": float(energy)},
        )
    for rank in (2, 4, 6):
        add(
            method_family="mc_svd",
            config_name=f"rank{rank}",
            baseline_type="mc_svd",
            run_stap=False,
            role="baseline",
            override_builder=lambda shape, rank=rank: {"svd_rank": int(rank), "svd_energy_frac": None},
        )

    add(
        method_family="svd_similarity",
        config_name="default",
        baseline_type="svd_similarity",
        run_stap=False,
        role="baseline",
    )
    add(
        method_family="svd_similarity",
        config_name="sensitive_r6",
        baseline_type="svd_similarity",
        run_stap=False,
        role="baseline",
        override_builder=lambda shape: {
            "svd_sim_smooth": 5,
            "svd_sim_kappa": 2.0,
            "svd_sim_r_min": 1,
            "svd_rank": 6,
        },
    )
    add(
        method_family="svd_similarity",
        config_name="sensitive_r8",
        baseline_type="svd_similarity",
        run_stap=False,
        role="baseline",
        override_builder=lambda shape: {
            "svd_sim_smooth": 5,
            "svd_sim_kappa": 2.0,
            "svd_sim_r_min": 1,
            "svd_rank": 8,
        },
    )
    add(
        method_family="svd_similarity",
        config_name="conservative_r8",
        baseline_type="svd_similarity",
        run_stap=False,
        role="baseline",
        override_builder=lambda shape: {
            "svd_sim_smooth": 9,
            "svd_sim_kappa": 3.0,
            "svd_sim_r_min": 1,
            "svd_rank": 8,
        },
    )
    add(
        method_family="svd_similarity",
        config_name="conservative_r10",
        baseline_type="svd_similarity",
        run_stap=False,
        role="baseline",
        override_builder=lambda shape: {
            "svd_sim_smooth": 9,
            "svd_sim_kappa": 3.0,
            "svd_sim_r_min": 1,
            "svd_rank": 10,
        },
    )
    add(
        method_family="svd_similarity",
        config_name="balanced_r8",
        baseline_type="svd_similarity",
        run_stap=False,
        role="baseline",
        override_builder=lambda shape: {
            "svd_sim_smooth": 7,
            "svd_sim_kappa": 2.5,
            "svd_sim_r_min": 1,
            "svd_rank": 8,
        },
    )

    add(
        method_family="local_svd",
        config_name="tile8_s3_ef90",
        baseline_type="local_svd",
        run_stap=False,
        role="baseline",
        override_builder=lambda shape: {
            "tile_hw": (8, 8),
            "tile_stride": 3,
            "svd_energy_frac": 0.90,
        },
    )
    add(
        method_family="local_svd",
        config_name="tile6_s2_ef90",
        baseline_type="local_svd",
        run_stap=False,
        role="baseline",
        override_builder=lambda shape: {
            "tile_hw": (6, 6),
            "tile_stride": 2,
            "svd_energy_frac": 0.90,
        },
    )
    add(
        method_family="local_svd",
        config_name="tile8_s2_ef90",
        baseline_type="local_svd",
        run_stap=False,
        role="baseline",
        override_builder=lambda shape: {
            "tile_hw": (8, 8),
            "tile_stride": 2,
            "svd_energy_frac": 0.90,
        },
    )
    add(
        method_family="local_svd",
        config_name="tile8_s3_ef95",
        baseline_type="local_svd",
        run_stap=False,
        role="baseline",
        override_builder=lambda shape: {
            "tile_hw": (8, 8),
            "tile_stride": 3,
            "svd_energy_frac": 0.95,
        },
    )
    add(
        method_family="local_svd",
        config_name="tile12_s4_ef95",
        baseline_type="local_svd",
        run_stap=False,
        role="baseline",
        override_builder=lambda shape: {
            "tile_hw": (12, 12),
            "tile_stride": 4,
            "svd_energy_frac": 0.95,
        },
    )
    add(
        method_family="local_svd",
        config_name="tile12_s3_ef95",
        baseline_type="local_svd",
        run_stap=False,
        role="baseline",
        override_builder=lambda shape: {
            "tile_hw": (12, 12),
            "tile_stride": 3,
            "svd_energy_frac": 0.95,
        },
    )
    add(
        method_family="local_svd",
        config_name="tile16_s4_ef95",
        baseline_type="local_svd",
        run_stap=False,
        role="baseline",
        override_builder=lambda shape: {
            "tile_hw": (16, 16),
            "tile_stride": 4,
            "svd_energy_frac": 0.95,
        },
    )
    add(
        method_family="local_svd",
        config_name="tile8_s3_ef90_rect",
        baseline_type="local_svd",
        run_stap=False,
        role="baseline",
        override_builder=lambda shape: {
            "tile_hw": (8, 8),
            "tile_stride": 3,
            "svd_energy_frac": 0.90,
            "local_svd_hann": False,
        },
    )
    add(
        method_family="local_svd",
        config_name="tile12_s4_ef95_rect",
        baseline_type="local_svd",
        run_stap=False,
        role="baseline",
        override_builder=lambda shape: {
            "tile_hw": (12, 12),
            "tile_stride": 4,
            "svd_energy_frac": 0.95,
            "local_svd_hann": False,
        },
    )
    add(
        method_family="local_svd",
        config_name="tile16_s4_ef95_rect",
        baseline_type="local_svd",
        run_stap=False,
        role="baseline",
        override_builder=lambda shape: {
            "tile_hw": (16, 16),
            "tile_stride": 4,
            "svd_energy_frac": 0.95,
            "local_svd_hann": False,
        },
    )

    add(
        method_family="adaptive_local_svd",
        config_name="tile8_s3_bal_r8",
        baseline_type="adaptive_local_svd",
        run_stap=False,
        role="baseline",
        override_builder=lambda shape: {
            "tile_hw": (8, 8),
            "tile_stride": 3,
            "svd_sim_smooth": 7,
            "svd_sim_kappa": 2.5,
            "svd_sim_r_min": 1,
            "svd_rank": 8,
        },
    )
    add(
        method_family="adaptive_local_svd",
        config_name="tile8_s2_sens_r8",
        baseline_type="adaptive_local_svd",
        run_stap=False,
        role="baseline",
        override_builder=lambda shape: {
            "tile_hw": (8, 8),
            "tile_stride": 2,
            "svd_sim_smooth": 5,
            "svd_sim_kappa": 2.0,
            "svd_sim_r_min": 1,
            "svd_rank": 8,
        },
    )
    add(
        method_family="adaptive_local_svd",
        config_name="tile12_s4_bal_r8",
        baseline_type="adaptive_local_svd",
        run_stap=False,
        role="baseline",
        override_builder=lambda shape: {
            "tile_hw": (12, 12),
            "tile_stride": 4,
            "svd_sim_smooth": 7,
            "svd_sim_kappa": 2.5,
            "svd_sim_r_min": 1,
            "svd_rank": 8,
        },
    )
    add(
        method_family="adaptive_local_svd",
        config_name="tile12_s3_cons_r10",
        baseline_type="adaptive_local_svd",
        run_stap=False,
        role="baseline",
        override_builder=lambda shape: {
            "tile_hw": (12, 12),
            "tile_stride": 3,
            "svd_sim_smooth": 9,
            "svd_sim_kappa": 3.0,
            "svd_sim_r_min": 1,
            "svd_rank": 10,
        },
    )
    add(
        method_family="adaptive_local_svd",
        config_name="tile16_s4_cons_r10",
        baseline_type="adaptive_local_svd",
        run_stap=False,
        role="baseline",
        override_builder=lambda shape: {
            "tile_hw": (16, 16),
            "tile_stride": 4,
            "svd_sim_smooth": 9,
            "svd_sim_kappa": 3.0,
            "svd_sim_r_min": 1,
            "svd_rank": 10,
        },
    )
    add(
        method_family="adaptive_local_svd",
        config_name="tile16_s4_bal_r8",
        baseline_type="adaptive_local_svd",
        run_stap=False,
        role="baseline",
        override_builder=lambda shape: {
            "tile_hw": (16, 16),
            "tile_stride": 4,
            "svd_sim_smooth": 7,
            "svd_sim_kappa": 2.5,
            "svd_sim_r_min": 1,
            "svd_rank": 8,
        },
    )
    add(
        method_family="adaptive_local_svd",
        config_name="tile8_s2_sens_r8_rect",
        baseline_type="adaptive_local_svd",
        run_stap=False,
        role="baseline",
        override_builder=lambda shape: {
            "tile_hw": (8, 8),
            "tile_stride": 2,
            "svd_sim_smooth": 5,
            "svd_sim_kappa": 2.0,
            "svd_sim_r_min": 1,
            "svd_rank": 8,
            "local_svd_hann": False,
        },
    )
    add(
        method_family="adaptive_local_svd",
        config_name="tile12_s4_bal_r8_rect",
        baseline_type="adaptive_local_svd",
        run_stap=False,
        role="baseline",
        override_builder=lambda shape: {
            "tile_hw": (12, 12),
            "tile_stride": 4,
            "svd_sim_smooth": 7,
            "svd_sim_kappa": 2.5,
            "svd_sim_r_min": 1,
            "svd_rank": 8,
            "local_svd_hann": False,
        },
    )
    add(
        method_family="adaptive_local_svd",
        config_name="tile16_s4_cons_r10_rect",
        baseline_type="adaptive_local_svd",
        run_stap=False,
        role="baseline",
        override_builder=lambda shape: {
            "tile_hw": (16, 16),
            "tile_stride": 4,
            "svd_sim_smooth": 9,
            "svd_sim_kappa": 3.0,
            "svd_sim_r_min": 1,
            "svd_rank": 10,
            "local_svd_hann": False,
        },
    )

    add(
        method_family="rpca",
        config_name="lam1_it250",
        baseline_type="rpca",
        run_stap=False,
        role="baseline",
        override_builder=lambda shape: {
            "rpca_lambda": _rpca_default_lambda(shape),
            "rpca_max_iters": 250,
        },
    )
    add(
        method_family="rpca",
        config_name="lam0p25_it250",
        baseline_type="rpca",
        run_stap=False,
        role="baseline",
        override_builder=lambda shape: {
            "rpca_lambda": 0.25 * _rpca_default_lambda(shape),
            "rpca_max_iters": 250,
        },
    )
    add(
        method_family="rpca",
        config_name="lam0p5_it250",
        baseline_type="rpca",
        run_stap=False,
        role="baseline",
        override_builder=lambda shape: {
            "rpca_lambda": 0.5 * _rpca_default_lambda(shape),
            "rpca_max_iters": 250,
        },
    )
    add(
        method_family="rpca",
        config_name="lam2_it250",
        baseline_type="rpca",
        run_stap=False,
        role="baseline",
        override_builder=lambda shape: {
            "rpca_lambda": 2.0 * _rpca_default_lambda(shape),
            "rpca_max_iters": 250,
        },
    )
    add(
        method_family="rpca",
        config_name="lam0p5_it250_ds1_t32_r4",
        baseline_type="rpca",
        run_stap=False,
        role="baseline",
        override_builder=lambda shape: {
            "rpca_lambda": 0.5 * _rpca_default_lambda(shape),
            "rpca_max_iters": 250,
            "rpca_spatial_downsample": 1,
            "rpca_t_sub": min(32, int(shape[0])),
            "rpca_rank_k_max": 4,
            "rpca_tol": 1e-4,
        },
    )
    add(
        method_family="rpca",
        config_name="lam0p5_it250_ds1_t64_r8",
        baseline_type="rpca",
        run_stap=False,
        role="baseline",
        override_builder=lambda shape: {
            "rpca_lambda": 0.5 * _rpca_default_lambda(shape),
            "rpca_max_iters": 250,
            "rpca_spatial_downsample": 1,
            "rpca_t_sub": min(64, int(shape[0])),
            "rpca_rank_k_max": 8,
            "rpca_tol": 1e-4,
        },
    )
    add(
        method_family="rpca",
        config_name="lam0p5_it250_ds1_full_r12",
        baseline_type="rpca",
        run_stap=False,
        role="baseline",
        override_builder=lambda shape: {
            "rpca_lambda": 0.5 * _rpca_default_lambda(shape),
            "rpca_max_iters": 250,
            "rpca_spatial_downsample": 1,
            "rpca_t_sub": int(shape[0]),
            "rpca_rank_k_max": 12,
            "rpca_tol": 1e-4,
        },
    )
    add(
        method_family="rpca",
        config_name="lam1_it250_ds1_t64_r8_tol1e3",
        baseline_type="rpca",
        run_stap=False,
        role="baseline",
        override_builder=lambda shape: {
            "rpca_lambda": _rpca_default_lambda(shape),
            "rpca_max_iters": 250,
            "rpca_spatial_downsample": 1,
            "rpca_t_sub": min(64, int(shape[0])),
            "rpca_rank_k_max": 8,
            "rpca_tol": 1e-3,
        },
    )
    add(
        method_family="rpca",
        config_name="lam1_it250_ds2_t32_r4",
        baseline_type="rpca",
        run_stap=False,
        role="baseline",
        override_builder=lambda shape: {
            "rpca_lambda": _rpca_default_lambda(shape),
            "rpca_max_iters": 250,
            "rpca_spatial_downsample": 2,
            "rpca_t_sub": min(32, int(shape[0])),
            "rpca_rank_k_max": 4,
            "rpca_tol": 1e-4,
        },
    )

    add(
        method_family="hosvd",
        config_name="ef99_ds2_full",
        baseline_type="hosvd",
        run_stap=False,
        role="baseline",
        override_builder=lambda shape: {
            "hosvd_energy_fracs": (0.99, 0.99, 0.99),
            "hosvd_spatial_downsample": 2,
            "hosvd_t_sub": None,
        },
    )
    add(
        method_family="hosvd",
        config_name="ef95_ds1_full",
        baseline_type="hosvd",
        run_stap=False,
        role="baseline",
        override_builder=lambda shape: {
            "hosvd_energy_fracs": (0.95, 0.95, 0.95),
            "hosvd_spatial_downsample": 1,
            "hosvd_t_sub": None,
        },
    )
    add(
        method_family="hosvd",
        config_name="ef99_ds2_t32",
        baseline_type="hosvd",
        run_stap=False,
        role="baseline",
        override_builder=lambda shape: {
            "hosvd_energy_fracs": (0.99, 0.99, 0.99),
            "hosvd_spatial_downsample": 2,
            "hosvd_t_sub": min(32, int(shape[0])),
        },
    )
    add(
        method_family="hosvd",
        config_name="ef99_ds2_t64",
        baseline_type="hosvd",
        run_stap=False,
        role="baseline",
        override_builder=lambda shape: {
            "hosvd_energy_fracs": (0.99, 0.99, 0.99),
            "hosvd_spatial_downsample": 2,
            "hosvd_t_sub": min(64, int(shape[0])),
        },
    )
    add(
        method_family="hosvd",
        config_name="ef99_ds1_full",
        baseline_type="hosvd",
        run_stap=False,
        role="baseline",
        override_builder=lambda shape: {
            "hosvd_energy_fracs": (0.99, 0.99, 0.99),
            "hosvd_spatial_downsample": 1,
            "hosvd_t_sub": None,
        },
    )
    add(
        method_family="hosvd",
        config_name="ef95_ds2_t32",
        baseline_type="hosvd",
        run_stap=False,
        role="baseline",
        override_builder=lambda shape: {
            "hosvd_energy_fracs": (0.95, 0.95, 0.95),
            "hosvd_spatial_downsample": 2,
            "hosvd_t_sub": min(32, int(shape[0])),
        },
    )
    add(
        method_family="hosvd",
        config_name="ef90_ds1_full",
        baseline_type="hosvd",
        run_stap=False,
        role="baseline",
        override_builder=lambda shape: {
            "hosvd_energy_fracs": (0.90, 0.90, 0.90),
            "hosvd_spatial_downsample": 1,
            "hosvd_t_sub": None,
        },
    )
    add(
        method_family="hosvd",
        config_name="rank8_16_16_ds1_full",
        baseline_type="hosvd",
        run_stap=False,
        role="baseline",
        override_builder=lambda shape: {
            "hosvd_ranks": (min(8, int(shape[0])), 16, 16),
            "hosvd_energy_fracs": None,
            "hosvd_spatial_downsample": 1,
            "hosvd_t_sub": None,
        },
    )
    add(
        method_family="hosvd",
        config_name="rank12_24_24_ds1_full",
        baseline_type="hosvd",
        run_stap=False,
        role="baseline",
        override_builder=lambda shape: {
            "hosvd_ranks": (min(12, int(shape[0])), 24, 24),
            "hosvd_energy_fracs": None,
            "hosvd_spatial_downsample": 1,
            "hosvd_t_sub": None,
        },
    )
    add(
        method_family="hosvd",
        config_name="rank8_16_16_ds2_t32",
        baseline_type="hosvd",
        run_stap=False,
        role="baseline",
        override_builder=lambda shape: {
            "hosvd_ranks": (min(8, int(shape[0])), 16, 16),
            "hosvd_energy_fracs": None,
            "hosvd_spatial_downsample": 2,
            "hosvd_t_sub": min(32, int(shape[0])),
        },
    )
    return out


def _score_path(bundle_dir: Path, *, run_stap: bool) -> Path:
    return bundle_dir / ("score_stap_preka.npy" if run_stap else "score_base.npy")


def _selection_tuple(summary: dict[str, Any]) -> tuple[float, float, float, float]:
    return (
        float(summary["selection_score"]),
        float(summary["mean_auc_main_vs_bg"]),
        float(summary["mean_auc_main_vs_nuisance"]),
        -float(summary["mean_fpr_nuisance_match@0p5"]),
    )


def _summarize_config_rows(rows: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    grouped: dict[tuple[str, str], list[dict[str, Any]]] = {}
    for row in rows:
        grouped.setdefault((str(row["method_family"]), str(row["config_name"])), []).append(row)
    out: dict[str, dict[str, Any]] = {}
    for (method_family, config_name), items in grouped.items():
        auc_bg = np.asarray([float(r["auc_main_vs_bg"]) for r in items], dtype=np.float64)
        auc_n = np.asarray([float(r["auc_main_vs_nuisance"]) for r in items], dtype=np.float64)
        fpr = np.asarray([float(r["fpr_nuisance_match@0p5"]) for r in items], dtype=np.float64)
        summary = {
            "method_family": method_family,
            "config_name": config_name,
            "count": int(len(items)),
            "mean_auc_main_vs_bg": float(np.mean(auc_bg)),
            "mean_auc_main_vs_nuisance": float(np.mean(auc_n)),
            "mean_fpr_nuisance_match@0p5": float(np.mean(fpr)),
        }
        summary["selection_score"] = (
            float(summary["mean_auc_main_vs_bg"])
            + float(summary["mean_auc_main_vs_nuisance"])
            - float(summary["mean_fpr_nuisance_match@0p5"])
        )
        out[f"{method_family}::{config_name}"] = summary
    return out


def _select_best_configs(summaries: dict[str, dict[str, Any]]) -> dict[str, dict[str, Any]]:
    by_method: dict[str, list[dict[str, Any]]] = {}
    for summary in summaries.values():
        by_method.setdefault(str(summary["method_family"]), []).append(summary)
    out: dict[str, dict[str, Any]] = {}
    for method_family, items in by_method.items():
        items_sorted = sorted(items, key=_selection_tuple, reverse=True)
        out[method_family] = items_sorted[0]
    return out


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Fair frozen-profile search for SIMUS baselines and STAP.")
    ap.add_argument("--dev-case", type=str, action="append", required=True, help="Development case name::run_dir")
    ap.add_argument("--eval-case", type=str, action="append", required=True, help="Held-out case name::run_dir")
    ap.add_argument("--out-root", type=Path, default=Path("runs/sim_eval/simus_fair_profile_search"))
    ap.add_argument(
        "--out-csv",
        type=Path,
        default=Path("reports/simus_motion/simus_fair_profile_search.csv"),
    )
    ap.add_argument(
        "--out-json",
        type=Path,
        default=Path("reports/simus_motion/simus_fair_profile_search.json"),
    )
    ap.add_argument("--fprs", type=str, default="1e-4,1e-3")
    ap.add_argument("--match-tprs", type=str, default="0.25,0.5,0.75")
    ap.add_argument("--stap-device", type=str, default="cuda")
    ap.add_argument("--reuse-bundles", action=argparse.BooleanOptionalAction, default=True)
    return ap.parse_args()


def main() -> None:
    args = parse_args()
    fprs = [float(x) for x in _split_csv_list(str(args.fprs))]
    match_tprs = [float(x) for x in _split_csv_list(str(args.match_tprs))]
    dev_cases = [_parse_case(spec) for spec in args.dev_case]
    eval_cases = [_parse_case(spec) for spec in args.eval_case]
    out_root = Path(args.out_root)
    dev_root = out_root / "dev"
    eval_root = out_root / "eval"
    dev_root.mkdir(parents=True, exist_ok=True)
    eval_root.mkdir(parents=True, exist_ok=True)

    candidates = _candidate_specs()
    rows: list[dict[str, Any]] = []
    details: dict[str, Any] = {
        "schema_version": "simus_fair_profile_search.v1",
        "dev_cases": [{"name": name, "run_dir": str(run_dir)} for name, run_dir in dev_cases],
        "eval_cases": [{"name": name, "run_dir": str(run_dir)} for name, run_dir in eval_cases],
        "candidate_count": int(len(candidates)),
        "candidates": [],
    }
    for candidate in candidates:
        details["candidates"].append(
            {
                "method_family": candidate.method_family,
                "config_name": candidate.config_name,
                "baseline_type": candidate.method.baseline_type,
                "run_stap": bool(candidate.method.run_stap),
                "stap_profile": candidate.stap_profile,
            }
        )

    def eval_caseset(split: str, cases: list[tuple[str, Path]], selected_only: dict[str, dict[str, Any]] | None) -> None:
        root = dev_root if split == "dev" else eval_root
        for case_name, run_dir in cases:
            icube, masks, run_meta = load_canonical_run(run_dir)
            mask_h1_pf_main = masks["mask_h1_pf_main"]
            mask_h0_bg = masks["mask_h0_bg"]
            mask_h0_nuisance_pa = masks.get("mask_h0_nuisance_pa")
            mask_h1_alias_qc = masks.get("mask_h1_alias_qc")
            for candidate in candidates:
                if selected_only is not None:
                    chosen = selected_only.get(candidate.method_family)
                    if not chosen or chosen["config_name"] != candidate.config_name:
                        continue
                bundle_root = root / case_name
                dataset_name = f"{case_name}_{slugify(candidate.method_family)}_{slugify(candidate.config_name)}"
                bundle_dir = bundle_root / slugify(dataset_name)
                bundle_overrides = candidate.override_builder(tuple(int(v) for v in icube.shape))
                if not bool(args.reuse_bundles) or not (bundle_dir / "meta.json").is_file():
                    bundle_dir = derive_bundle_from_run(
                        run_dir=run_dir,
                        out_root=bundle_root,
                        dataset_name=dataset_name,
                        stap_profile=candidate.stap_profile,
                        baseline_type=str(candidate.method.baseline_type),
                        run_stap=bool(candidate.method.run_stap),
                        stap_device=str(args.stap_device),
                        bundle_overrides=bundle_overrides,
                        meta_extra={
                            "simus_fair_profile_search": True,
                            "search_split": split,
                            "case_name": case_name,
                            "method_family": candidate.method_family,
                            "config_name": candidate.config_name,
                        },
                    )
                score_path = _score_path(bundle_dir, run_stap=bool(candidate.method.run_stap))
                score = np.load(score_path).astype(np.float32, copy=False)
                metrics = evaluate_structural_metrics(
                    score=score,
                    mask_h1_pf_main=mask_h1_pf_main,
                    mask_h0_bg=mask_h0_bg,
                    mask_h0_nuisance_pa=mask_h0_nuisance_pa,
                    mask_h1_alias_qc=mask_h1_alias_qc,
                    fprs=fprs,
                    match_tprs=match_tprs,
                )
                row = {
                    "split": split,
                    "case_name": case_name,
                    "run_dir": str(run_dir),
                    "method_family": candidate.method_family,
                    "config_name": candidate.config_name,
                    "method_label": _pipeline_label(candidate.method),
                    "headline_label": _headline_label("vnext", candidate.method),
                    "baseline_label": _baseline_label(candidate.method.baseline_type),
                    "baseline_type": candidate.method.baseline_type,
                    "role": candidate.method.role,
                    "run_stap": int(candidate.method.run_stap),
                    "eval_score": "vnext",
                    "score_label": _score_label("vnext", candidate.method),
                    "score_semantics": _score_semantics("vnext", candidate.method),
                    "stap_profile": candidate.stap_profile if candidate.method.run_stap else None,
                    "bundle_dir": str(bundle_dir),
                    "score_file": score_path.name,
                    "T": int(icube.shape[0]),
                    "H": int(icube.shape[1]),
                    "W": int(icube.shape[2]),
                    "prf_hz": run_meta.get("acquisition", {}).get("prf_hz"),
                    "motion_disp_rms_px": run_meta.get("motion", {}).get("telemetry", {}).get("disp_rms_px"),
                    "phase_rms_rad": run_meta.get("phase_screen", {}).get("telemetry", {}).get("phase_rms_rad"),
                    "bundle_overrides_json": json.dumps(bundle_overrides, sort_keys=True),
                }
                row.update(metrics)
                rows.append(row)

    eval_caseset("dev", dev_cases, selected_only=None)
    dev_rows = [row for row in rows if row["split"] == "dev"]
    dev_summary = _summarize_config_rows(dev_rows)
    selected = _select_best_configs(dev_summary)
    eval_caseset("eval", eval_cases, selected_only=selected)
    eval_rows = [row for row in rows if row["split"] == "eval"]
    eval_summary = _summarize_config_rows(eval_rows)

    details["dev_summary"] = dev_summary
    details["selected_configs"] = selected
    details["eval_summary"] = eval_summary
    _write_csv(Path(args.out_csv), rows)
    _write_json(Path(args.out_json), {"rows": rows, "details": details})
    print(f"[simus-fair-profile-search] wrote {args.out_csv}")
    print(f"[simus-fair-profile-search] wrote {args.out_json}")


if __name__ == "__main__":
    main()
