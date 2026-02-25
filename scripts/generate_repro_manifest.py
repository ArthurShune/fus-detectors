#!/usr/bin/env python3
"""
Phase 7: Reproducibility manifest generator (JSON + LaTeX appendix).

This script is intentionally *lightweight*: it does not rerun experiments.
Instead it produces a pinned manifest capturing:
  - git commit hash + dirty status
  - core runtime environment versions
  - dataset locations / DOIs used in the paper
  - the exact commands used to regenerate each figure/table artifact

Outputs (tracked):
  - repro_manifest.json
  - appendix_repro_manifest.tex

Usage:
  PYTHONPATH=. python scripts/generate_repro_manifest.py
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import subprocess
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable


REPO = Path(__file__).resolve().parents[1]


@dataclass(frozen=True)
class DatasetInfo:
    name: str
    purpose: str
    local_path: str
    doi_or_url: str


@dataclass(frozen=True)
class ArtifactInfo:
    name: str
    paper_refs: list[str]
    outputs: list[str]
    commands: list[str]
    notes: str | None = None


def _run(cmd: list[str]) -> str:
    p = subprocess.run(cmd, cwd=str(REPO), check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    return p.stdout.strip()


def _git_info() -> dict[str, Any]:
    try:
        commit = _run(["git", "rev-parse", "HEAD"])
        short = _run(["git", "rev-parse", "--short", "HEAD"])
        branch = _run(["git", "rev-parse", "--abbrev-ref", "HEAD"])
        status = _run(["git", "status", "--porcelain=v1"])
    except Exception as exc:
        return {"error": str(exc)}

    dirty_lines = [ln for ln in status.splitlines() if ln.strip()]
    return {
        "commit": commit,
        "commit_short": short,
        "branch": branch,
        "dirty": bool(dirty_lines),
        "dirty_paths": [ln.strip() for ln in dirty_lines[:200]],
    }


def _env_info() -> dict[str, Any]:
    versions: dict[str, str | None] = {}
    for pkg, mod_name in (
        ("numpy", "numpy"),
        ("scipy", "scipy"),
        ("torch", "torch"),
        ("matplotlib", "matplotlib"),
    ):
        try:
            mod = __import__(mod_name)
            versions[pkg] = getattr(mod, "__version__", None)
        except Exception:
            versions[pkg] = None

    return {
        "python": sys.version.splitlines()[0],
        "platform": platform.platform(),
        "machine": platform.machine(),
        "processor": platform.processor(),
        "versions": versions,
        "cwd": str(REPO),
    }


def _tex_escape(s: str) -> str:
    # Keep this minimal; we mostly use \path / verbatim for commands.
    return (
        s.replace("\\", "\\textbackslash{}")
        .replace("_", "\\_")
        .replace("%", "\\%")
        .replace("&", "\\&")
        .replace("#", "\\#")
    )


def _render_verbatim_block(lines: Iterable[str]) -> str:
    body = "\n".join(lines).rstrip() + "\n"
    # Keep command blocks readable while avoiding overfull boxes in narrow margins.
    return "{\\footnotesize\n\\begin{verbatim}\n" + body + "\\end{verbatim}\n}\n"


def _render_breakable_path(p: str) -> str:
    """
    Render a filesystem path using \\path (nolinkurl) but also permit a hard
    line break after common directory prefixes. This avoids overfull boxes in
    narrow list environments while preserving copy/paste fidelity.
    """
    def _chunk_underscores(s: str, max_len: int) -> list[str]:
        if len(s) <= max_len or "_" not in s:
            return [s]
        parts = s.split("_")
        segs: list[str] = []
        cur = ""
        for i, part in enumerate(parts):
            piece = part + ("_" if i < len(parts) - 1 else "")
            if cur and len(cur) + len(piece) > max_len:
                segs.append(cur)
                cur = piece
            else:
                cur += piece
        if cur:
            segs.append(cur)
        return segs

    p = str(p)
    # Keep segments short enough to fit within nested list environments.
    max_seg = 44
    for prefix in ("runs/real/", "runs/pilot/", "reports/", "figs/paper/"):
        if p.startswith(prefix) and len(p) > len(prefix):
            rest = p[len(prefix) :]
            chunks = _chunk_underscores(rest, max_seg)
            out = f"\\path{{{prefix}}}"
            for ch in chunks:
                out += f"\\allowbreak\\path{{{ch}}}"
            return out

    chunks = _chunk_underscores(p, max_seg)
    out = f"\\path{{{chunks[0]}}}"
    for ch in chunks[1:]:
        out += f"\\allowbreak\\path{{{ch}}}"
    return out


def _render_appendix_tex(manifest: dict[str, Any], *, out_path: Path) -> None:
    git = manifest.get("git") or {}
    env = manifest.get("env") or {}
    datasets = manifest.get("datasets") or []
    selections = manifest.get("selections") or {}
    artifacts = manifest.get("artifacts") or []

    def _sentence(s: str) -> str:
        s = str(s or "").strip()
        if not s:
            return ""
        return s if s.endswith(".") else s + "."

    lines: list[str] = []
    lines.append("% AUTO-GENERATED by scripts/generate_repro_manifest.py; DO NOT EDIT BY HAND.")
    lines.append("\\section{Reproducibility Manifest}")
    lines.append("\\label{app:repro}")
    lines.append("")
    lines.append(
        "This appendix records a minimal reproduction recipe: exact commands, data locations, and a pinned repository state."
    )
    lines.append(
        "The manifest lists the minimal set of commands/outputs needed to reproduce the main claims; additional exploratory reports referenced in the text are not required unless explicitly listed here."
    )
    lines.append("Run \\path{PYTHONPATH=. python scripts/generate_repro_manifest.py} to refresh this manifest.")
    lines.append("")

    lines.append("\\paragraph{Repository state.}")
    if "error" in git:
        lines.append(f"Git metadata unavailable: {_tex_escape(str(git.get('error')))}.")
    else:
        lines.append(
            "Commit: "
            f"\\texttt{{{_tex_escape(str(git.get('commit_short') or ''))}}} "
            f"(full: \\texttt{{{_tex_escape(str(git.get('commit') or ''))}}}); "
            f"branch: \\texttt{{{_tex_escape(str(git.get('branch') or ''))}}}; "
            f"dirty: \\texttt{{{str(bool(git.get('dirty'))).lower()}}}."
        )
    lines.append("")

    lines.append("\\paragraph{Environment.}")
    versions = (env.get("versions") or {}) if isinstance(env, dict) else {}
    # Use \path (nolinkurl) for long identifiers to permit line breaks.
    lines.append(f"Python: \\path{{{str(env.get('python') or '')}}}.")
    lines.append(f"Platform: \\path{{{str(env.get('platform') or '')}}}.")
    lines.append(
        "Packages: "
        + ", ".join(
            f"\\texttt{{{_tex_escape(k)}}}={{{_tex_escape(str(v))}}}"
            for k, v in versions.items()
            if v is not None
        )
        + "."
    )
    lines.append("")

    lines.append("\\paragraph{Datasets.}")
    lines.append(
        "Paths below are the expected on-disk locations used by the scripts in this repository. "
        "Large outputs under \\path{runs/} and \\path{reports/} are not tracked."
    )
    lines.append("\\begin{itemize}[nosep]")
    for ds in datasets:
        if not isinstance(ds, dict):
            continue
        name = _tex_escape(str(ds.get("name") or ""))
        purpose = _tex_escape(str(ds.get("purpose") or ""))
        local_path = str(ds.get("local_path") or "").strip()
        source = str(ds.get("doi_or_url") or "").strip()

        def _render_source(source: str) -> str:
            # URLs/DOIs render nicely as \path (nolinkurl); free-form provenance is
            # left as plain text to permit normal line breaking.
            if source.startswith(("http://", "https://")):
                return f"\\path{{{source}}}"
            return _tex_escape(source)

        local_lines: list[str] = []
        if local_path:
            if " (extracts to " in local_path and local_path.endswith(")"):
                before, after = local_path.split(" (extracts to ", 1)
                after = after[:-1]  # strip trailing ')'
                local_lines.append(f"Local archive: \\path{{{before}}}.")
                local_lines.append(f"Extracts to: \\path{{{after}}}.")
            else:
                local_lines.append(f"Local: \\path{{{local_path}}}.")

        source_text = _render_source(source) if source else ""

        lines.append(f"  \\item \\textbf{{{name}}}: {purpose}")
        for ln in local_lines:
            lines.append(f"    \\\\ {ln}")
        if source_text:
            lines.append(f"    \\\\ Source: {source_text}.")
    lines.append("\\end{itemize}")
    lines.append("")

    lines.append("\\paragraph{Pre-registered selections used in this manuscript.}")
    lines.append("\\begin{itemize}[nosep]")
    for k, v in selections.items():
        lines.append(f"  \\item \\textbf{{{_tex_escape(str(k))}}}: {_tex_escape(_sentence(str(v)))}")
    lines.append("\\end{itemize}")
    lines.append("")

    lines.append("\\paragraph{Artifact commands.}")
    lines.append(
        "Each entry below lists the command(s) used to regenerate the referenced paper artifact(s). "
        "Most scripts write a JSON summary containing the full parameterization and are safe to rerun."
    )
    lines.append("\\begin{enumerate}[label=(R\\arabic*),nosep]")
    for art in artifacts:
        if not isinstance(art, dict):
            continue
        name = str(art.get("name") or "")
        refs = art.get("paper_refs") or []
        outs = art.get("outputs") or []
        cmds = art.get("commands") or []
        notes = art.get("notes")

        lines.append(f"  \\item \\textbf{{{_tex_escape(name)}}}")
        def _render_ref_item(r: str) -> str:
            r = str(r)
            if ": " in r:
                head, tail = r.split(": ", 1)
                tail_s = tail.strip()
                if any(
                    tail_s.endswith(ext)
                    for ext in (
                        ".png",
                        ".pdf",
                        ".csv",
                        ".json",
                        ".tex",
                        ".npy",
                    )
                ):
                    return f"{_tex_escape(head)}: \\path{{{tail_s}}}"
            return _tex_escape(r)

        if refs:
            if len(refs) == 1:
                lines.append(f"    (paper refs: {_render_ref_item(refs[0])}).")
            else:
                lines.append("    Paper refs:")
                lines.append("    \\begin{itemize}[nosep,leftmargin=0em,label={}]")
                for r in refs:
                    lines.append(f"      \\item {_render_ref_item(r)}")
                lines.append("    \\end{itemize}")
        else:
            lines.append(".")
        if outs:
            lines.append("    Outputs:")
            # Paths are long and appear in nested lists; use raggedright to
            # avoid TeX choosing overfull lines over short-but-legal breaks.
            lines.append("    \\begingroup\\raggedright")
            lines.append("    \\begin{itemize}[nosep,leftmargin=0em,label={}]")
            for o in outs:
                lines.append(f"      \\item {_render_breakable_path(str(o))}")
            lines.append("    \\end{itemize}")
            lines.append("    \\endgroup")
        if notes:
            lines.append("    Notes: " + _tex_escape(_sentence(str(notes))))
        if cmds:
            lines.append(_render_verbatim_block([str(c) for c in cmds]))
    lines.append("\\end{enumerate}")
    lines.append("")

    out_path.write_text("\n".join(lines) + "\n")


def _default_datasets() -> list[DatasetInfo]:
    return [
        DatasetInfo(
            name="Brain-* k-Wave pilots (simulated IQ)",
            purpose="Labeled low-FPR ROC + ablations.",
            local_path="runs/pilot/",
            doi_or_url="generated locally via sim/kwave/pilot_motion.py (Makefile targets r4c-*)",
        ),
        DatasetInfo(
            name="Mace/Urban whole-brain mouse fUS (PD-only)",
            purpose="PD-only telemetry + hold-out alias-gate evaluation (no Doppler IQ).",
            local_path="data/whole-brain-fUS",
            doi_or_url="https://doi.org/10.5281/zenodo.4905862",
        ),
        DatasetInfo(
            name="Shin RatBrain Fig3 (LOCA-ULM) beamformed IQ",
            purpose="Real-IQ ingestion, telemetry, contract + label-free robustness sweeps.",
            local_path="data/shin_zenodo_10711806/RatBrain_Fig3.zip (extracts to data/shin_zenodo_10711806/ratbrain_fig3_raw)",
            doi_or_url="https://doi.org/10.5281/zenodo.10711806",
        ),
        DatasetInfo(
            name="Twinkling artifact dataset (RawBCF phantoms)",
            purpose="Structurally labeled phantom ROC (Gammex) + within-ensemble motion ladders + KA hygiene (calculi).",
            local_path="data/twinkling_artifact/",
            doi_or_url="https://doi.org/10.17816/DD76511",
        ),
        DatasetInfo(
            name="ULM Zenodo 7883227 (rat brain kHz IQ)",
            purpose="Label-free motion robustness + one-time baseline calibration sweep.",
            local_path="data/ulm_zenodo_7883227/IQ_001_to_025.zip (extracts to tmp/ulm_zenodo_7883227/)",
            doi_or_url="https://doi.org/10.5281/zenodo.7883227",
        ),
    ]


def _default_selections() -> dict[str, str]:
    return {
        "Brain seed sweep (AliasContract pilot)": "seed2-seed12 (n=11).",
        "Shin all-clips telemetry": "IQData001-IQData080 (n=80), window 0:128.",
        "Shin motion subset": "IQData001-005, IQData010, IQData020, IQData040, IQData060, IQData080 (n=10).",
        "Mace holdout split (deduplicated)": "train: scan1/2 + scan3/6; test: scan4 + scan5.",
        "Gammex flow phantom structural ROC (along-linear17)": "frames 0:85 (n=85), PRF=2500 Hz, N=17 shots.",
        "Gammex flow phantom structural ROC (across-linear17)": "frames 0:200 (n=200), PRF=2500 Hz, N=17 shots.",
        "Gammex flow phantom within-ensemble motion ladder": "along-linear17, frames 0:85, amps 0..2 px, kinds {rw,step}.",
        "Twinkling calculi KA hygiene": "calcifications sequence, frames 0:50 (n=50), PRF approx 500 Hz, N=9 shots.",
        "ULM 7883227 baseline sweep": "blocks 1-3, frames 0:128, MC-SVD energy-frac sweep (label-free).",
        "ULM 7883227 motion sweeps": "blocks 1-3, frames 0:128, frozen baseline e=0.975, motion kinds {brainlike,elastic}.",
    }


def _default_artifacts() -> list[ArtifactInfo]:
    return [
        ArtifactInfo(
            name="PD-mode sanity checks (PD-mode map and score S=PD)",
            paper_refs=["Notation (PD convention)"],
            outputs=["reports/pd_mode_sanity/ (optional figs)"],
            commands=[
                "PYTHONPATH=. python scripts/pd_mode_sanity.py \\",
                "  runs/<bundle_root>/pw_* \\",
                "  --out-dir reports/pd_mode_sanity",
            ],
            notes="Run on any generated acceptance bundle root containing pw_* directories.",
        ),
        ArtifactInfo(
            name="Conditional STAP leakage ablation (full vs conditional masks)",
            paper_refs=["Results (Brain-k-Wave): conditional STAP ablation"],
            outputs=["reports/condstap_leakage.csv", "reports/condstap_leakage.json"],
            commands=[
                "PYTHONPATH=. python scripts/conditional_stap_leakage_ablation.py \\",
                "  --pilots runs/pilot/r4_kwave_seed1 runs/pilot/r4_kwave_seed2 \\",
                "  --profile Brain-OpenSkull \\",
                "  --out-root runs/ablation/condstap_leakage \\",
                "  --window-length 64 --window-offset 0 --disjoint-offset 64 \\",
                "  --summary-csv reports/condstap_leakage.csv \\",
                "  --summary-json reports/condstap_leakage.json",
            ],
        ),
        ArtifactInfo(
            name="Brain-* baseline fairness: MC--SVD(e) tune-once sweep",
            paper_refs=["Baseline fairness (Brain-*)"],
            outputs=["reports/brain_mcsvd_energy_sweep_seed1.csv", "reports/brain_mcsvd_energy_sweep_seed1.json"],
            commands=[
                "PYTHONPATH=. python scripts/brain_mcsvd_energy_sweep.py \\",
                "  --pilot runs/pilot/r4_kwave_seed1 \\",
                "  --profile Brain-OpenSkull \\",
                "  --out-root runs/sweep/mcsvd_energy_brain_seed1 \\",
                "  --window-length 64 --window-offset 0 \\",
                "  --energy-fracs 0.90,0.95,0.97,0.975,0.98,0.99 \\",
                "  --fprs 1e-4,3e-4,1e-3 \\",
                "  --summary-csv reports/brain_mcsvd_energy_sweep_seed1.csv \\",
                "  --summary-json reports/brain_mcsvd_energy_sweep_seed1.json",
            ],
        ),
        ArtifactInfo(
            name="Brain-* ROC curve figure (median+IQR over disjoint windows)",
            paper_refs=["Figure: brain_kwave_roc_curves.pdf"],
            outputs=["figs/paper/brain_kwave_roc_curves.pdf"],
            commands=[
                "PYTHONPATH=. python scripts/fig_brain_kwave_roc_curves.py \\",
                "  --runs-root runs/pilot/fair_filter_matrix_full_clinical_cpu_v2 \\",
                "  --out-pdf figs/paper/brain_kwave_roc_curves.pdf",
            ],
            notes="Reads precomputed per-window score/mask arrays under runs/pilot/fair_filter_matrix_full_clinical_cpu_v2.",
        ),
        ArtifactInfo(
            name="Brain-* cross-window threshold-transfer audit (STAP)",
            paper_refs=["Appendix: cross-window threshold calibration audit (Brain-*)"],
            outputs=[
                "reports/brain_crosswindow_calibration.csv",
                "reports/brain_crosswindow_calibration_summary.json",
                "reports/brain_crosswindow_calibration_table.tex",
            ],
            commands=[
                "PYTHONPATH=. python scripts/brain_crosswindow_calibration.py \\",
                "  --runs-root runs/pilot/fair_filter_matrix_full_clinical_cpu_v2 \\",
                "  --alphas 1e-4,3e-4,1e-3 \\",
                "  --out-csv reports/brain_crosswindow_calibration.csv \\",
                "  --out-json reports/brain_crosswindow_calibration_summary.json",
            ],
            notes="Calibrates thresholds on one 64-frame window's negatives and evaluates on disjoint windows (ordered pairs).",
        ),
        ArtifactInfo(
            name="Knowledge-aided prior falsifiability ablation (STAP-only vs contract vs forced)",
            paper_refs=["KA evaluation discipline"],
            outputs=["reports/ka_v2_ablation.csv", "reports/ka_v2_ablation.json"],
            commands=[
                "PYTHONPATH=. python scripts/ka_contract_v2_ablation.py \\",
                "  --pilot runs/pilot/r4c_kwave_hab_contract_seed2 \\",
                "  --profile Brain-AliasContract \\",
                "  --out-root runs/ablation/ka_v2_falsifiability \\",
                "  --stap-device cpu \\",
                "  --window-length 64 \\",
                "  --window-offsets 0 64 128 192 256 \\",
                "  --summary-csv reports/ka_v2_ablation.csv \\",
                "  --summary-json reports/ka_v2_ablation.json",
            ],
        ),
        ArtifactInfo(
            name="Knowledge-aided prior positive-control ablation (OpenSkull shallow-alias + high-rank ghosts)",
            paper_refs=["KA evaluation discipline (positive control)"],
            outputs=[
                "reports/ka_v2_ablation_openskull_shallowalias_e50.csv",
                "reports/ka_v2_ablation_openskull_shallowalias_e50.json",
            ],
            commands=[
                "PYTHONPATH=. python scripts/ka_contract_v2_ablation.py \\",
                "  --pilot runs/pilot/r4_kwave_seed1 \\",
                "  --profile Brain-OpenSkull \\",
                "  --out-root runs/ablation/openskull_shallowalias_e50 \\",
                "  --stap-device cpu \\",
                "  --window-length 64 \\",
                "  --window-offsets 0 64 128 192 256 \\",
                "  --summary-csv reports/ka_v2_ablation_openskull_shallowalias_e50.csv \\",
                "  --summary-json reports/ka_v2_ablation_openskull_shallowalias_e50.json \\",
                "  --replay-extra \\",
                "    --svd-energy-frac 0.50 \\",
                "    --flow-doppler-min-hz 60 --flow-doppler-max-hz 180 \\",
                "    --flow-doppler-noise-amp 60 --flow-doppler-noise-mode fft_band \\",
                "    --bg-alias-hz 650 --bg-alias-fraction 0.7 \\",
                "    --bg-alias-depth-min-frac 0.12 --bg-alias-depth-max-frac 0.28 \\",
                "    --bg-alias-jitter-hz 35 \\",
                "    --flow-mask-suppress-alias-depth \\",
                "    --clutter-beta 1.0 --clutter-snr-db 20 \\",
                "    --clutter-depth-min-frac 0.20 --clutter-depth-max-frac 0.95 \\",
                "    --aperture-phase-std 0.8 --aperture-phase-corr-len 14 \\",
                "    --bg-alias-highrank-mode gw_reverb_add \\",
                "    --bg-alias-highrank-coverage 0.3 --bg-alias-highrank-amp 0.3 \\",
                "    --bg-alias-highrank-margin-px 3 \\",
                "    --bg-alias-highrank-freq-jitter-hz 25 \\",
                "    --bg-alias-highrank-drift-step-hz 12 \\",
                "    --bg-alias-highrank-pf-leak-eta 0.0 \\",
                "    --ka-score-contract-v2-mode auto",
            ],
            notes="Synthetic, replay-only KA-positive regime used as a contract-mechanics positive control; not a realism benchmark.",
        ),
        ArtifactInfo(
            name="Mace PD-only holdout alias-gate evaluation (label-free thresholds)",
            paper_refs=["Table: Mace hold-out alias gate"],
            outputs=["reports/mace_alias_gate_holdout.csv", "reports/mace_alias_gate_holdout.json"],
            commands=[
                "PYTHONPATH=. python scripts/mace_alias_gate_holdout.py \\",
                "  --out-csv reports/mace_alias_gate_holdout.csv \\",
                "  --out-json reports/mace_alias_gate_holdout.json",
            ],
        ),
        ArtifactInfo(
            name="Mace pixel-level vascular-atlas check (independent structural labels)",
            paper_refs=["Mace section: vascular atlas pixel-level check"],
            outputs=["reports/mace_vascular_pixel_eval.csv", "reports/mace_vascular_pixel_eval.json"],
            commands=[
                "PYTHONPATH=. python scripts/mace_vascular_pixel_eval.py \\",
                "  --out-csv reports/mace_vascular_pixel_eval.csv \\",
                "  --out-json reports/mace_vascular_pixel_eval.json",
            ],
            notes="Maps atlas.Vascular into each scan plane using Transformation.mat; evaluates vascular vs non-vascular pixels (structural, not clinical efficacy).",
        ),
        ArtifactInfo(
            name="Mace atlas-alignment sanity overlay figure",
            paper_refs=["Appendix: Mace atlas alignment sanity check"],
            outputs=["figs/paper/mace_atlas_overlay.png"],
            commands=[
                "PYTHONPATH=. python scripts/mace_atlas_overlay_fig.py \\",
                "  --scan-name scan1 \\",
                "  --plane-indices 5 10 15 \\",
                "  --out-png figs/paper/mace_atlas_overlay.png",
            ],
            notes="Overlays atlas ROIs and atlas.Vascular contours on mean PD for representative planes (sanity check of Transformation.mat alignment).",
        ),
        ArtifactInfo(
            name="Telemetry regime comparison (sim vs real; contract-v2 scalars)",
            paper_refs=["Figure: telemetry_regime_compare.png"],
            outputs=["figs/paper/telemetry_regime_compare.png"],
            commands=[
                "# Sample Brain-* replay bundles (write meta.json with ka_contract_v2 telemetry):",
                "PYTHONPATH=. python scripts/replay_stap_from_run.py \\",
                "  --src runs/pilot/r4_kwave_seed1 \\",
                "  --out runs/telemetry_regime_compare/brain_open_seed1 \\",
                "  --profile Brain-OpenSkull \\",
                "  --time-window-length 64 \\",
                "  --time-window-offset 0 --time-window-offset 64 --time-window-offset 128 \\",
                "  --time-window-offset 192 --time-window-offset 256",
                "PYTHONPATH=. python scripts/replay_stap_from_run.py \\",
                "  --src runs/pilot/r4c_kwave_hab_contract_seed2_v2 \\",
                "  --out runs/telemetry_regime_compare/brain_aliascontract_seed2 \\",
                "  --profile Brain-AliasContract \\",
                "  --time-window-length 64 \\",
                "  --time-window-offset 0 --time-window-offset 64 --time-window-offset 128 \\",
                "  --time-window-offset 192 --time-window-offset 256",
                "PYTHONPATH=. python scripts/replay_stap_from_run.py \\",
                "  --src runs/pilot/r4c_kwave_hab_v3_skull_seed2_v2 \\",
                "  --out runs/telemetry_regime_compare/brain_skullor_seed2 \\",
                "  --profile Brain-SkullOR \\",
                "  --time-window-length 64 \\",
                "  --time-window-offset 0 --time-window-offset 64 --time-window-offset 128 \\",
                "  --time-window-offset 192 --time-window-offset 256",
                "",
                "# Generate the overlay histogram figure:",
                "# (reads Shin bundles and Mace CSV from other artifacts)",
                "PYTHONPATH=. python scripts/telemetry_regime_compare_fig.py \\",
                "  --out-png figs/paper/telemetry_regime_compare.png",
            ],
            notes="Overlays PfPeakFrac(flow), guard_q90, and iqr_alias_bg across Brain-* simulations, Shin real IQ, and Mace PD-only planes; intended as telemetry anchoring (not a claim that sims match real data).",
        ),
        ArtifactInfo(
            name="Mace PD-only prior dashboard (plane sweep + paper figure)",
            paper_refs=["Figure: mace_pdonly_contract_v2_dashboard.png"],
            outputs=["reports/mace_pdonly_contract_v2.csv", "figs/paper/mace_pdonly_contract_v2_dashboard.png"],
            commands=[
                "PYTHONPATH=. python scripts/mace_pdonly_contract_v2_sweep.py \\",
                "  --out-csv reports/mace_pdonly_contract_v2.csv",
                "",
                "PYTHONPATH=. python scripts/mace_pdonly_contract_v2_dashboard_fig.py \\",
                "  --in-csv reports/mace_pdonly_contract_v2.csv \\",
                "  --out-png figs/paper/mace_pdonly_contract_v2_dashboard.png",
            ],
        ),
        ArtifactInfo(
            name="Shin all-clips prior-regime telemetry (n=80; telemetry-only)",
            paper_refs=["Shin section: all-clips telemetry audit"],
            outputs=[
                "reports/shin_ratbrain_shinU_e970_telemetry_only_0_128_all80.csv",
                "reports/shin_ratbrain_shinU_e970_telemetry_only_0_128_all80_summary.json",
            ],
            commands=[
                "PYTHONPATH=. python -m scripts.shin_ratbrain_allclips_contract_hygiene \\",
                "  --extract-from-zip --no-run-stap --no-score-ka-v2 \\",
                "  --profile U --svd-energy-frac 0.97 --frames-list 0:128 \\",
                "  --out-root runs/shin_ratbrain_shinU_e970_telemetry_only_0_128_all80 \\",
                "  --out-csv reports/shin_ratbrain_shinU_e970_telemetry_only_0_128_all80.csv \\",
                "  --out-json reports/shin_ratbrain_shinU_e970_telemetry_only_0_128_all80_summary.json",
            ],
        ),
        ArtifactInfo(
            name="Shin brainlike motion sweep (10 clips) + aggregate plots",
            paper_refs=[
                "Table: Shin motion endpoints",
                "Figure: shin_motion_brainlike_batch_U_agg.png",
                "Figure: shin_motion_brainlike_compare_reg1.png",
                "Figure: shin_motion_brainlike_compare.png",
            ],
            outputs=[
                "reports/shin_motion_brainlike_batch_U/*.csv",
                "reports/shin_motion_brainlike_batch_U_agg.csv",
                "figs/paper/shin_motion_brainlike_batch_U_agg.png",
                "figs/paper/shin_motion_brainlike_compare_reg1.png",
                "figs/paper/shin_motion_brainlike_compare.png",
            ],
            commands=[
                "# Per-clip sweeps (canonical window 0:128; KA disabled):",
                "for f in IQData{001,002,003,004,005,010,020,040,060,080}.dat; do",
                "  PYTHONPATH=. python -m scripts.shin_ratbrain_motion_sweep \\",
                "    --iq-file \"$f\" --frames 0:128 --profile U --svd-energy-frac 0.97 \\",
                "    --motion-kind brainlike --amp-px-list 0,0.5,1,2 \\",
                "    --no-score-ka-v2 --out-root runs/shin_motion_brainlike_batch_U \\",
                "    --out-csv \"reports/shin_motion_brainlike_batch_U/${f%.dat}.csv\" \\",
                "    --out-png \"reports/shin_motion_brainlike_batch_U/${f%.dat}.png\";",
                "done",
                "",
                "# Aggregate curve (writes CSV + plot):",
                "PYTHONPATH=. python -m scripts.shin_ratbrain_motion_aggregate \\",
                "  --in-dir reports/shin_motion_brainlike_batch_U \\",
                "  --out-csv reports/shin_motion_brainlike_batch_U_agg.csv \\",
                "  --out-png reports/shin_motion_brainlike_batch_U_agg.png",
                "",
                "# Compare across windows / reg ablations (writes comparison plots):",
                "PYTHONPATH=. python -m scripts.shin_ratbrain_motion_compare \\",
                "  --curve \"0:128=reports/shin_motion_brainlike_batch_U_agg.csv\" \\",
                "  --curve \"64:192=reports/shin_motion_brainlike_batch_U_f64_192_reg1_agg.csv\" \\",
                "  --curve \"122:250=reports/shin_motion_brainlike_batch_U_f122_250_reg1_agg.csv\" \\",
                "  --out-png reports/shin_motion_brainlike_compare_reg1.png",
                "cp -f \\",
                "  reports/shin_motion_brainlike_compare_reg1.png \\",
                "  figs/paper/shin_motion_brainlike_compare_reg1.png",
            ],
            notes="Window-specific batch runs (64:192, 122:250) follow the same pattern.",
        ),
        ArtifactInfo(
            name="Shin motion crop/alignment sensitivity (post-hoc)",
            paper_refs=["Shin motion section: crop sensitivity note"],
            outputs=["reports/shin_motion_brainlike_batch_U_crop_sensitivity.csv"],
            commands=[
                "PYTHONPATH=. python -m scripts.shin_ratbrain_motion_sensitivity_posthoc \\",
                "  --run-root runs/shin_motion_brainlike_batch_U \\",
                "  --amps 0,0.5,1,2 \\",
                "  --crop-margins 0,4,8,12 \\",
                "  --align-maps-list 0,1 \\",
                "  --out-csv reports/shin_motion_brainlike_batch_U_crop_sensitivity.csv",
            ],
        ),
        ArtifactInfo(
            name="Twinkling decode sanity (RawBCF -> B-mode/CFM)",
            paper_refs=["Twinkling section: RawBCF decode sanity"],
            outputs=["reports/twinkling_decode_sanity/*/decode_report.json"],
            commands=[
                "SEQ_DIR=\"data/twinkling_artifact/Flow in Gammex phantom\"",
                "SEQ_DIR=\"$SEQ_DIR/Flow in Gammex phantom (along - linear probe)\"",
                "PYTHONPATH=. python scripts/twinkling_decode_sanity.py \\",
                "  --seq-dir \"$SEQ_DIR\" \\",
                "  --frame-idx 0 \\",
                "  --out-dir reports/twinkling_decode_sanity",
            ],
        ),
        ArtifactInfo(
            name="Gammex flow phantom structural ROC (along-linear17, PRF=2500)",
            paper_refs=["Twinkling section: Table (structural ROC)", "Figure: twinkling_gammex_along_mask_overlay.png"],
            outputs=[
                "runs/real/twinkling_gammex_alonglinear17_prf2500_str6_msd_ka/",
                "reports/twinkling_gammex_alonglinear17_prf2500_str6_msd_ka_with_baselines.csv",
                "reports/twinkling_gammex_alonglinear17_prf2500_str6_msd_ka_with_baselines_summary.json",
                "figs/paper/twinkling_gammex_along_mask_overlay.png",
            ],
            commands=[
                "SEQ_DIR=\"data/twinkling_artifact/Flow in Gammex phantom\"",
                "SEQ_DIR=\"$SEQ_DIR/Flow in Gammex phantom (along - linear probe)\"",
                "OUT_ROOT=\"runs/real/twinkling_gammex_alonglinear17_prf2500_str6_msd_ka\"",
                "MASK_GLOB=$OUT_ROOT/*__mask_debug",
                "",
                "PYTHONPATH=. python scripts/twinkling_make_bundles.py \\",
                "  --seq-dir \"$SEQ_DIR\" \\",
                "  --out-root \"$OUT_ROOT\" \\",
                "  --frames 0:85 \\",
                "  --prf-hz 2500 --Lt 16 \\",
                "  --tile-hw 8 8 --tile-stride 6 \\",
                "  --cov-estimator tyler_pca --diag-load 0.07 \\",
                "  --baseline-type svd_bandpass --svd-keep-min 2 --svd-keep-max 17 \\",
                "  --score-mode msd --score-ka-v2-enable --score-ka-v2-mode auto",
                "",
                "PYTHONPATH=. python scripts/twinkling_eval_structural.py \\",
                "  --root \"$OUT_ROOT\" \\",
                "  --out-csv reports/twinkling_gammex_alonglinear17_prf2500_str6_msd_ka_with_baselines.csv \\",
                "  --out-summary-json \\",
                "    reports/twinkling_gammex_alonglinear17_prf2500_str6_msd_ka_with_baselines_summary.json",
                "",
                "PYTHONPATH=. python scripts/fig_twinkling_mask_overlay.py \\",
                "  --bmode $MASK_GLOB/bmode_roi_ref_norm.png \\",
                "  --mask-flow $MASK_GLOB/mask_flow.png \\",
                "  --mask-bg $MASK_GLOB/mask_bg.png \\",
                "  --out figs/paper/twinkling_gammex_along_mask_overlay.png \\",
                "  --min-width 1200 --dpi 300",
            ],
            notes="Along-linear uses stride 6; this matches the fixed geometry-only stride policy (largest stride with >=500 tiles).",
        ),
        ArtifactInfo(
            name="Gammex flow phantom structural ROC (across-linear17, PRF=2500)",
            paper_refs=["Twinkling section: Table (structural ROC)", "Figure: twinkling_gammex_across_mask_overlay.png"],
            outputs=[
                "runs/real/twinkling_gammex_across17_prf2500_str4_msd_ka/",
                "reports/twinkling_gammex_across17_prf2500_str4_msd_ka_with_baselines.csv",
                "reports/twinkling_gammex_across17_prf2500_str4_msd_ka_with_baselines_summary.json",
                "figs/paper/twinkling_gammex_across_mask_overlay.png",
            ],
            commands=[
                "SEQ_DIR=\"data/twinkling_artifact/Flow in Gammex phantom\"",
                "SEQ_DIR=\"$SEQ_DIR/Flow in Gammex phantom (across - linear probe)\"",
                "DAT_PATH=\"$SEQ_DIR/RawBCFCine_08062017_145434_17.dat\"",
                "PAR_PATH=\"$SEQ_DIR/RawBCFCine_08062017_145434_17.par\"",
                "OUT_ROOT=\"runs/real/twinkling_gammex_across17_prf2500_str4_msd_ka\"",
                "MASK_GLOB=$OUT_ROOT/*__mask_debug",
                "",
                "PYTHONPATH=. python scripts/twinkling_make_bundles.py \\",
                "  --seq-dir \"$SEQ_DIR\" \\",
                "  --dat-path \"$DAT_PATH\" \\",
                "  --par-path \"$PAR_PATH\" \\",
                "  --out-root \"$OUT_ROOT\" \\",
                "  --frames 0:200 \\",
                "  --prf-hz 2500 --Lt 16 \\",
                "  --tile-hw 8 8 --tile-stride 4 \\",
                "  --cov-estimator tyler_pca --diag-load 0.07 \\",
                "  --baseline-type svd_bandpass --svd-keep-min 2 --svd-keep-max 17 \\",
                "  --score-mode msd --score-ka-v2-enable --score-ka-v2-mode auto",
                "",
                "PYTHONPATH=. python scripts/twinkling_eval_structural.py \\",
                "  --root \"$OUT_ROOT\" \\",
                "  --out-csv reports/twinkling_gammex_across17_prf2500_str4_msd_ka_with_baselines.csv \\",
                "  --out-summary-json \\",
                "    reports/twinkling_gammex_across17_prf2500_str4_msd_ka_with_baselines_summary.json",
                "",
                "PYTHONPATH=. python scripts/fig_twinkling_mask_overlay.py \\",
                "  --bmode $MASK_GLOB/bmode_roi_ref_norm.png \\",
                "  --mask-flow $MASK_GLOB/mask_flow.png \\",
                "  --mask-bg $MASK_GLOB/mask_bg.png \\",
                "  --out figs/paper/twinkling_gammex_across_mask_overlay.png \\",
                "  --min-width 780 --dpi 300",
            ],
            notes="Across-linear uses stride 4; this matches the fixed geometry-only stride policy (largest stride with >=500 tiles).",
        ),
        ArtifactInfo(
            name="Gammex within-ensemble motion ladder (along-linear17; PRF=2500)",
            paper_refs=[
                "Twinkling section: Figure (within-ensemble motion ladder)",
                "Figure: twinkling_gammex_alonglinear17_prf2500_within_ensemble_motion_ladder_rw_ci_v2.png",
                "Figure: twinkling_gammex_alonglinear17_prf2500_within_ensemble_motion_ladder_step_ci_v2.png",
            ],
            outputs=[
                "runs/real/twinkling_gammex_alonglinear17_prf2500_within_ensemble_motion_rw_v2/",
                "runs/real/twinkling_gammex_alonglinear17_prf2500_within_ensemble_motion_step_v2/",
                "reports/twinkling_gammex_alonglinear17_prf2500_within_ensemble_motion_ladder_rw_ci_v2.json",
                "reports/twinkling_gammex_alonglinear17_prf2500_within_ensemble_motion_ladder_step_ci_v2.json",
                "figs/paper/twinkling_gammex_alonglinear17_prf2500_within_ensemble_motion_ladder_rw_ci_v2.png",
                "figs/paper/twinkling_gammex_alonglinear17_prf2500_within_ensemble_motion_ladder_step_ci_v2.png",
            ],
            commands=[
                "SEQ_DIR=\"data/twinkling_artifact/Flow in Gammex phantom\"",
                "SEQ_DIR=\"$SEQ_DIR/Flow in Gammex phantom (along - linear probe)\"",
                "REF_ROOT=\"runs/real/twinkling_gammex_alonglinear17_prf2500_str6_msd_nomotion_ref_fast_f020\"",
                "OUT_RW=\"runs/real/twinkling_gammex_alonglinear17_prf2500_within_ensemble_motion_rw_v2\"",
                "OUT_STEP=\"runs/real/twinkling_gammex_alonglinear17_prf2500_within_ensemble_motion_step_v2\"",
                "REP_DIR=\"reports\"",
                "FIG_DIR=\"figs/paper\"",
                "BASE=\"twinkling_gammex_alonglinear17_prf2500\"",
                "SUF_RW=\"within_ensemble_motion_ladder_rw_ci_v2\"",
                "SUF_STEP=\"within_ensemble_motion_ladder_step_ci_v2\"",
                "RW_PREFIX=\"$REP_DIR/${BASE}_${SUF_RW}\"",
                "STEP_PREFIX=\"$REP_DIR/${BASE}_${SUF_STEP}\"",
                "RW_FIG=\"$FIG_DIR/${BASE}_${SUF_RW}.png\"",
                "STEP_FIG=\"$FIG_DIR/${BASE}_${SUF_STEP}.png\"",
                "",
                "# Amp=0 reference bundles for equivalence checks (20 frames):",
                "PYTHONPATH=. python scripts/twinkling_make_bundles.py \\",
                "  --seq-dir \"$SEQ_DIR\" \\",
                "  --out-root \"$REF_ROOT\" \\",
                "  --frames 0:20 \\",
                "  --prf-hz 2500 --Lt 16 \\",
                "  --tile-hw 8 8 --tile-stride 6 \\",
                "  --cov-estimator tyler_pca --diag-load 0.07 \\",
                "  --baseline-type svd_bandpass --svd-keep-min 2 --svd-keep-max 17 \\",
                "  --score-mode msd",
                "",
                "# Within-ensemble motion sweeps (85 frames; rw and step):",
                "PYTHONPATH=. python scripts/twinkling_gammex_within_ensemble_motion_sweep.py \\",
                "  --seq-dir \"$SEQ_DIR\" \\",
                "  --out-root \"$OUT_RW\" \\",
                "  --frames 0:85 --mask-ref-frames 0:20 \\",
                "  --prf-hz 2500 --Lt 16 \\",
                "  --tile-hw 8 8 --tile-stride 6 \\",
                "  --cov-estimator tyler_pca --diag-load 0.07 \\",
                "  --baseline-type svd_bandpass --svd-keep-min 2 --svd-keep-max 17 \\",
                "  --score-mode msd \\",
                "  --motion-kind rw --amp-px-list 0,0.25,0.5,1,1.5,2",
                "PYTHONPATH=. python scripts/twinkling_gammex_within_ensemble_motion_sweep.py \\",
                "  --seq-dir \"$SEQ_DIR\" \\",
                "  --out-root \"$OUT_STEP\" \\",
                "  --frames 0:85 --mask-ref-frames 0:20 \\",
                "  --prf-hz 2500 --Lt 16 \\",
                "  --tile-hw 8 8 --tile-stride 6 \\",
                "  --cov-estimator tyler_pca --diag-load 0.07 \\",
                "  --baseline-type svd_bandpass --svd-keep-min 2 --svd-keep-max 17 \\",
                "  --score-mode msd \\",
                "  --motion-kind step --amp-px-list 0,0.25,0.5,1,1.5,2",
                "",
                "# Aggregate + bootstrap CIs + plots:",
                "PYTHONPATH=. python scripts/twinkling_eval_motion_ladder.py \\",
                "  --root \"$OUT_RW\" \\",
                "  --out-csv \"${RW_PREFIX}.csv\" \\",
                "  --out-json \"${RW_PREFIX}.json\" \\",
                "  --out-png \"$RW_FIG\" \\",
                "  --bootstrap 1000 --bootstrap-seed 0 \\",
                "  --amp0-ref-root \"$REF_ROOT\" \\",
                "  --amp0-ref-frames 0:20 --amp0-ref-tol 5e-3",
                "PYTHONPATH=. python scripts/twinkling_eval_motion_ladder.py \\",
                "  --root \"$OUT_STEP\" \\",
                "  --out-csv \"${STEP_PREFIX}.csv\" \\",
                "  --out-json \"${STEP_PREFIX}.json\" \\",
                "  --out-png \"$STEP_FIG\" \\",
                "  --bootstrap 1000 --bootstrap-seed 0 \\",
                "  --amp0-ref-root \"$REF_ROOT\" \\",
                "  --amp0-ref-frames 0:20 --amp0-ref-tol 5e-3",
            ],
        ),
        ArtifactInfo(
            name="Twinkling calculi KA hygiene + contract telemetry (calcifications; PRF approx 500)",
            paper_refs=["Twinkling section: KA hygiene on calculi", "Figure: twinkling_calculi_tail_example.png"],
            outputs=[
                "runs/real/twinkling_calculi_calcifications_prf500_str4_Lt8_msd_ka_scm_dl015_f050/",
                "reports/twinkling_calculi_calcifications_prf500_str4_Lt8_msd_ka_scm_dl015_f050_roc.csv",
                "reports/twinkling_calculi_calcifications_prf500_str4_Lt8_msd_ka_scm_dl015_f050_summary.json",
            ],
            commands=[
                "BASE_DIR=\"data/twinkling_artifact/Twinkling artifact on calculi\"",
                "SEQ_DIR=\"$BASE_DIR/Twinkling and Flash artifacts on artificial calculi\"",
                "RUN_TAG=\"twinkling_calculi_calcifications_prf500_str4_Lt8_msd_ka_scm_dl015_f050\"",
                "OUT_ROOT=\"runs/real/${RUN_TAG}\"",
                "REP_PREFIX=\"reports/${RUN_TAG}\"",
                "",
                "PYTHONPATH=. python scripts/twinkling_make_bundles.py \\",
                "  --seq-dir \"$SEQ_DIR\" \\",
                "  --out-root \"$OUT_ROOT\" \\",
                "  --frames 0:50 \\",
                "  --prf-hz 500 --Lt 8 \\",
                "  --tile-hw 8 8 --tile-stride 4 \\",
                "  --cov-estimator scm --diag-load 0.15 \\",
                "  --baseline-type svd_bandpass --svd-keep-min 2 --svd-keep-max 9 \\",
                "  --score-mode msd --score-ka-v2-enable --score-ka-v2-mode auto \\",
                "  --mask-mode none",
                "",
                "PYTHONPATH=. python scripts/twinkling_eval_structural.py \\",
                "  --root \"$OUT_ROOT\" \\",
                "  --out-csv \"${REP_PREFIX}_roc.csv\" \\",
                "  --out-summary-json \"${REP_PREFIX}_summary.json\"",
            ],
            notes="Calculi runs use proxy masks (not tube structural masks). Used for contract/tail-hygiene instrumentation (not clinical detection claims).",
        ),
        ArtifactInfo(
            name="Twinkling contract figure pack (states/reasons + calculi tail example)",
            paper_refs=[
                "Twinkling section: Figure (twinkling_contract_states_reasons.png)",
                "Twinkling section: Figure (twinkling_calculi_tail_example.png)",
            ],
            outputs=[
                "figs/paper/twinkling_contract_states_reasons.png",
                "figs/paper/twinkling_calculi_tail_example.png",
                "reports/twinkling_contract_figpack.json",
            ],
            commands=[
                "ALONG_BASE=\"twinkling_gammex_alonglinear17_prf2500_str6_msd_ka_with_baselines\"",
                "ACROSS_BASE=\"twinkling_gammex_across17_prf2500_str4_msd_ka_with_baselines\"",
                "ALONG_SUM=\"reports/${ALONG_BASE}_summary.json\"",
                "ACROSS_SUM=\"reports/${ACROSS_BASE}_summary.json\"",
                "CALC_TAG=\"twinkling_calculi_calcifications_prf500_str4_Lt8_msd_ka_scm_dl015_f050\"",
                "CALC_SUM=\"reports/${CALC_TAG}_summary.json\"",
                "CALC_ROOT=\"runs/real/${CALC_TAG}\"",
                "OUT_STATE=\"figs/paper/twinkling_contract_states_reasons.png\"",
                "OUT_EX=\"figs/paper/twinkling_calculi_tail_example.png\"",
                "OUT_JSON=\"reports/twinkling_contract_figpack.json\"",
                "",
                "PYTHONPATH=. python scripts/twinkling_contract_figpack.py \\",
                "  --along-summary \"$ALONG_SUM\" \\",
                "  --across-summary \"$ACROSS_SUM\" \\",
                "  --calculi-summary \"$CALC_SUM\" \\",
                "  --calculi-root \"$CALC_ROOT\" \\",
                "  --out-state-reason-png \"$OUT_STATE\" \\",
                "  --out-calculi-example-png \"$OUT_EX\" \\",
                "  --out-json \"$OUT_JSON\"",
            ],
        ),
        ArtifactInfo(
            name="ULM 7883227 one-time baseline sweep (MC-SVD energy fraction)",
            paper_refs=["ULM section: baseline calibration (frozen svd_energy_frac=0.975)"],
            outputs=[
                "runs/real/ulm7883227_baseline_mcsvd_energy_sweep_ULM_0000_0128/",
                "reports/ulm7883227_baseline_mcsvd_energy_sweep_ULM_0000_0128.csv",
                "reports/ulm7883227_baseline_mcsvd_energy_sweep_ULM_0000_0128.json",
            ],
            commands=[
                "PYTHONPATH=. python scripts/ulm_zenodo_7883227_baseline_sweep.py \\",
                "  --block-ids 1,2,3 --frames 0:128 \\",
                "  --svd-energy-frac-list 0.90,0.95,0.97,0.975,0.98,0.99 \\",
                "  --profile ULM --prf-hz 1000 --Lt 64 \\",
                "  --tile-h 8 --tile-w 8 --tile-stride 3 \\",
                "  --reg-enable --reg-subpixel 4 \\",
                "  --no-run-stap --bg-tail-fpr 0.001 --stability-split \\",
                "  --cache-dir tmp/ulm_zenodo_7883227 \\",
                "  --out-root runs/real/ulm7883227_baseline_mcsvd_energy_sweep_ULM_0000_0128 \\",
                "  --out-csv reports/ulm7883227_baseline_mcsvd_energy_sweep_ULM_0000_0128.csv \\",
                "  --out-json reports/ulm7883227_baseline_mcsvd_energy_sweep_ULM_0000_0128.json",
            ],
        ),
        ArtifactInfo(
            name="ULM 7883227 motion sweeps (brainlike + elastic; fairness alignment + degeneracy)",
            paper_refs=[
                "ULM section: Figure (ulm7883227_motion_sweep_*_brainlike_e975.png)",
                "ULM section: Figure (ulm7883227_motion_sweep_*_elastic_e975.png)",
            ],
            outputs=[
                "reports/ulm7883227_motion_sweep_ULM_blocks1_3_0000_0128_brainlike_e975.csv",
                "reports/ulm7883227_motion_sweep_ULM_blocks1_3_0000_0128_brainlike_e975.json",
                "figs/paper/ulm7883227_motion_sweep_ULM_blocks1_3_0000_0128_brainlike_e975.png",
                "reports/ulm7883227_motion_sweep_ULM_blocks1_3_0000_0128_elastic_e975.csv",
                "reports/ulm7883227_motion_sweep_ULM_blocks1_3_0000_0128_elastic_e975.json",
                "figs/paper/ulm7883227_motion_sweep_ULM_blocks1_3_0000_0128_elastic_e975.png",
            ],
            commands=[
                "PYTHONPATH=. python scripts/ulm_zenodo_7883227_motion_sweep.py \\",
                "  --block-ids 1,2,3 --frames 0:128 \\",
                "  --profile ULM --prf-hz 1000 --Lt 64 \\",
                "  --tile-h 8 --tile-w 8 --tile-stride 3 \\",
                "  --baseline-type mc_svd --svd-energy-frac 0.975 \\",
                "  --reg-enable --reg-subpixel 4 \\",
                "  --cov-estimator scm --diag-load 0.07 \\",
                "  --motion-kind brainlike --amp-px-list 0,0.5,1,2,3 \\",
                "  --out-root runs/real/ulm7883227_motion_sweep_ULM_brainlike_e975 \\",
                "  --out-csv reports/ulm7883227_motion_sweep_ULM_blocks1_3_0000_0128_brainlike_e975.csv \\",
                "  --out-json reports/ulm7883227_motion_sweep_ULM_blocks1_3_0000_0128_brainlike_e975.json \\",
                "  --out-png figs/paper/ulm7883227_motion_sweep_ULM_blocks1_3_0000_0128_brainlike_e975.png",
                "PYTHONPATH=. python scripts/ulm_zenodo_7883227_motion_sweep.py \\",
                "  --block-ids 1,2,3 --frames 0:128 \\",
                "  --profile ULM --prf-hz 1000 --Lt 64 \\",
                "  --tile-h 8 --tile-w 8 --tile-stride 3 \\",
                "  --baseline-type mc_svd --svd-energy-frac 0.975 \\",
                "  --reg-enable --reg-subpixel 4 \\",
                "  --cov-estimator scm --diag-load 0.07 \\",
                "  --motion-kind elastic --amp-px-list 0,0.5,1,2,3 \\",
                "  --out-root runs/real/ulm7883227_motion_sweep_ULM_elastic_e975 \\",
                "  --out-csv reports/ulm7883227_motion_sweep_ULM_blocks1_3_0000_0128_elastic_e975.csv \\",
                "  --out-json reports/ulm7883227_motion_sweep_ULM_blocks1_3_0000_0128_elastic_e975.json \\",
                "  --out-png figs/paper/ulm7883227_motion_sweep_ULM_blocks1_3_0000_0128_elastic_e975.png",
            ],
        ),
        ArtifactInfo(
            name="Paper build modes (paper vs supplement vs full)",
            paper_refs=["Build packaging (Phase 7)"],
            outputs=["stap_fus_paper.pdf", "stap_fus_supplement.pdf", "stap_fus_methodology.pdf"],
            commands=[
                "pdflatex stap_fus_methodology.tex   # full (default)",
                "pdflatex stap_fus_paper.tex         # paper-only (no appendices)",
                "pdflatex stap_fus_supplement.tex    # supplement-only (appendices only)",
            ],
        ),
    ]


def build_manifest() -> dict[str, Any]:
    manifest: dict[str, Any] = {
        "git": _git_info(),
        "env": _env_info(),
        "datasets": [asdict(d) for d in _default_datasets()],
        "selections": _default_selections(),
        "artifacts": [asdict(a) for a in _default_artifacts()],
    }
    return manifest


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Generate reproducibility manifest (JSON + LaTeX appendix).")
    ap.add_argument("--out-json", type=Path, default=Path("repro_manifest.json"))
    ap.add_argument("--out-tex", type=Path, default=Path("appendix_repro_manifest.tex"))
    return ap.parse_args()


def main() -> None:
    args = parse_args()

    manifest = build_manifest()
    args.out_json.write_text(json.dumps(manifest, indent=2))
    _render_appendix_tex(manifest, out_path=args.out_tex)

    print(f"[repro-manifest] wrote {args.out_json}")
    print(f"[repro-manifest] wrote {args.out_tex}")


if __name__ == "__main__":
    main()
