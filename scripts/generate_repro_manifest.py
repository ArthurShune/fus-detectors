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
  # Preferred (captures the same CUDA-enabled conda environment used in experiments)
  PYTHONPATH=. conda run -n stap-fus python scripts/generate_repro_manifest.py

  # Fallback (may record a different interpreter than the one used for CUDA runs)
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

    # Ignore common build products so compiling the paper doesn't mark the repo as "dirty".
    # (We still record ignored paths in JSON for transparency.)
    ignore_suffixes = (
        ".aux",
        ".fdb_latexmk",
        ".fls",
        ".log",
        ".out",
        ".synctex.gz",
        ".pdf",
    )
    ignore_exact_paths = {
        "repro_manifest.json",
        "appendix_repro_manifest.tex",
    }

    def _status_paths(line: str) -> list[str]:
        # Porcelain v1 is 2 status chars + space + path (or "old -> new" for renames).
        path = line[3:].strip()
        if " -> " in path:
            a, b = path.split(" -> ", 1)
            return [a.strip(), b.strip()]
        return [path]

    raw_lines = [ln for ln in status.splitlines() if ln.strip()]
    ignored: list[str] = []
    kept: list[str] = []
    for ln in raw_lines:
        paths = _status_paths(ln)
        if all((p.endswith(ignore_suffixes) or p in ignore_exact_paths) for p in paths):
            ignored.append(ln.strip())
        else:
            kept.append(ln.strip())
    return {
        "commit": commit,
        "commit_short": short,
        "branch": branch,
        "dirty": bool(kept),
        "dirty_paths_count": len(kept),
        "dirty_ignored_paths_count": len(ignored),
        "dirty_paths": kept[:200],
        "dirty_ignored_paths": ignored[:200],
    }


def _env_info_current() -> dict[str, Any]:
    versions: dict[str, str | None] = {}
    for pkg, mod_name in (
        ("numpy", "numpy"),
        ("scipy", "scipy"),
        ("torch", "torch"),
        ("matplotlib", "matplotlib"),
        ("cupy", "cupy"),
    ):
        try:
            mod = __import__(mod_name)
            versions[pkg] = getattr(mod, "__version__", None)
        except Exception:
            versions[pkg] = None

    torch_cuda: dict[str, Any] = {}
    try:
        import torch

        torch_cuda = {
            "cuda_is_available": bool(torch.cuda.is_available()),
            "cuda_version": getattr(torch.version, "cuda", None),
            "device_count": int(torch.cuda.device_count()) if torch.cuda.is_available() else 0,
            "device0_name": (
                str(torch.cuda.get_device_name(0))
                if torch.cuda.is_available() and torch.cuda.device_count() > 0
                else None
            ),
        }
    except Exception:
        torch_cuda = {}

    return {
        "python": sys.version.splitlines()[0],
        "platform": platform.platform(),
        "machine": platform.machine(),
        "processor": platform.processor(),
        "versions": versions,
        "torch_cuda": torch_cuda,
        "cwd": str(REPO),
    }


def _env_info_conda(env_name: str) -> dict[str, Any]:
    """
    Query a named conda env via `conda run` so the manifest reflects the
    CUDA-enabled runtime used for experiments even if this script is invoked
    from a different interpreter.
    """
    code = r"""
import json, platform, sys

def ver(mod):
    try:
        m = __import__(mod)
        return getattr(m, "__version__", None)
    except Exception:
        return None

out = {
    "python": sys.version.splitlines()[0],
    "platform": platform.platform(),
    "machine": platform.machine(),
    "processor": platform.processor(),
    "versions": {
        "numpy": ver("numpy"),
        "scipy": ver("scipy"),
        "torch": ver("torch"),
        "matplotlib": ver("matplotlib"),
        "cupy": ver("cupy"),
    },
    "torch_cuda": {},
}

try:
    import torch
    out["torch_cuda"] = {
        "cuda_is_available": bool(torch.cuda.is_available()),
        "cuda_version": getattr(torch.version, "cuda", None),
        "device_count": int(torch.cuda.device_count()) if torch.cuda.is_available() else 0,
        "device0_name": (
            str(torch.cuda.get_device_name(0))
            if torch.cuda.is_available() and torch.cuda.device_count() > 0
            else None
        ),
    }
except Exception:
    out["torch_cuda"] = {}

print(json.dumps(out))
"""
    env = os.environ.copy()
    env["CONDA_NO_PLUGINS"] = "true"
    try:
        p = subprocess.run(
            ["conda", "run", "-n", env_name, "python", "-c", code],
            cwd=str(REPO),
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            env=env,
        )
        lines = [ln.strip() for ln in p.stdout.splitlines() if ln.strip()]
        payload = lines[-1] if lines else ""
        return json.loads(payload) if payload else {"error": "empty conda-run output"}
    except Exception as exc:
        return {"error": str(exc)}


def _repo_meta() -> dict[str, Any]:
    def _infer_git_remote_url() -> str | None:
        try:
            out = _run(["git", "remote", "-v"])
        except Exception:
            return None
        out = str(out or "").strip()
        if not out:
            return None
        first_fetch: str | None = None
        for ln in out.splitlines():
            ln = ln.strip()
            if not ln:
                continue
            parts = ln.split()
            if len(parts) < 3:
                continue
            name, url, kind = parts[0], parts[1], parts[2]
            if kind != "(fetch)":
                continue
            if first_fetch is None:
                first_fetch = url
            if name == "origin":
                return url
        return first_fetch

    def _infer_release_tag() -> str | None:
        try:
            tag = _run(["git", "describe", "--tags", "--exact-match"])
        except Exception:
            return None
        tag = str(tag or "").strip()
        return tag or None

    url = (os.environ.get("STAP_FUS_PUBLIC_REPO_URL") or "").strip() or None
    tag = (os.environ.get("STAP_FUS_RELEASE_TAG") or "").strip() or None
    url = url or _infer_git_remote_url()
    tag = tag or _infer_release_tag()
    return {"public_url": url, "release_tag": tag}


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


def _render_breakable_digest(s: str, *, chunk: int = 16) -> str:
    """
    Render a long hex digest (e.g. git SHA / sha256) so it can line-break without
    introducing spaces in the underlying text.
    """
    s = str(s or "").strip()
    if not s:
        return "\\texttt{}"
    parts = [s[i : i + int(chunk)] for i in range(0, len(s), int(chunk))]
    out = f"\\texttt{{{_tex_escape(parts[0])}}}"
    for p in parts[1:]:
        out += f"\\allowbreak\\texttt{{{_tex_escape(p)}}}"
    return out


def _render_appendix_tex(manifest: dict[str, Any], *, out_path: Path) -> None:
    repo = manifest.get("repo") or {}
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
    preferred_env = None
    if isinstance(env, dict):
        preferred_env = str(env.get("preferred_conda_env") or "").strip() or None
    preferred_env = preferred_env or "stap-fus"
    lines.append(
        "Run "
        f"\\path{{PYTHONPATH=. conda run -n {preferred_env} python scripts/generate_repro_manifest.py}} "
        "to refresh this manifest."
    )
    lines.append("")

    lines.append("\\paragraph{Repository state.}")
    if "error" in git:
        lines.append(f"Git metadata unavailable: {_tex_escape(str(git.get('error')))}.")
    else:
        public_url = str(repo.get("public_url") or "").strip()
        release_tag = str(repo.get("release_tag") or "").strip()
        if public_url:
            lines.append(f"Public repository: \\path{{{public_url}}}.")
        else:
            lines.append(
                "Public repository: (not recorded; set \\texttt{STAP\\_FUS\\_PUBLIC\\_REPO\\_URL} or add a git remote such as \\texttt{origin})."
            )
        if release_tag:
            lines.append(f"Release tag: \\texttt{{{_tex_escape(release_tag)}}}.")
        else:
            lines.append("Release tag: (not recorded; set \\texttt{STAP\\_FUS\\_RELEASE\\_TAG} or create a git tag).")
        commit_short = str(git.get("commit_short") or "").strip()
        commit_full = str(git.get("commit") or "").strip()
        commit_full_tex = _render_breakable_digest(commit_full, chunk=10)
        lines.append(
            "Commit: "
            f"\\texttt{{{_tex_escape(commit_short)}}} "
            f"(full: {commit_full_tex}); "
            f"branch: \\texttt{{{_tex_escape(str(git.get('branch') or ''))}}}; "
            f"dirty: \\texttt{{{str(bool(git.get('dirty'))).lower()}}}."
        )
        dirty_paths_count = git.get("dirty_paths_count")
        dirty_ignored_count = git.get("dirty_ignored_paths_count")
        if dirty_paths_count is not None or dirty_ignored_count is not None:
            lines.append(
                "Dirty details: "
                f"non-ignored paths={_tex_escape(str(dirty_paths_count))}, "
                f"ignored build products={_tex_escape(str(dirty_ignored_count))}."
            )
        lines.append(
            "Note: dirty status ignores common LaTeX build products (e.g., "
            "\\texttt{*.aux, *.log, *.out, *.fls, *.fdb\\_latexmk, *.synctex.gz, *.pdf})."
        )
        lines.append(
            "For archival reproducibility, generate this manifest from a clean, tagged release commit "
            "(\\texttt{dirty=false}) and record the public repository URL and tag."
        )
    lines.append("")

    lines.append("\\paragraph{Environment.}")
    env_current = env.get("current") if isinstance(env, dict) else {}
    env_conda = env.get("conda") if isinstance(env, dict) else {}
    env_primary = env_conda if isinstance(env_conda, dict) and "error" not in env_conda else env_current
    env_primary_label = f"conda env \\texttt{{{_tex_escape(preferred_env)}}}" if env_primary is env_conda else "current interpreter"

    env_yml = env.get("environment_yml") if isinstance(env, dict) else {}
    docker_spec = env.get("dockerfile") if isinstance(env, dict) else {}
    if isinstance(env_yml, dict) and env_yml.get("path"):
        sha = env_yml.get("sha256")
        if sha:
            lines.append(
                f"Conda spec: \\path{{{str(env_yml.get('path'))}}} (sha256: {_render_breakable_digest(str(sha), chunk=16)})."
            )
        else:
            lines.append(f"Conda spec: \\path{{{str(env_yml.get('path'))}}}.")
        lines.append("Create env: \\path{conda env create -f environment.yml}.")
    if isinstance(docker_spec, dict) and docker_spec.get("path"):
        sha = docker_spec.get("sha256")
        if sha:
            lines.append(
                f"Docker: \\path{{{str(docker_spec.get('path'))}}} (sha256: {_render_breakable_digest(str(sha), chunk=16)})."
            )
        else:
            lines.append(f"Docker: \\path{{{str(docker_spec.get('path'))}}}.")
        lines.append("Build container: \\path{docker build -t stap-fus -f Dockerfile .}.")

    if isinstance(env_conda, dict) and "error" in env_conda:
        lines.append(f"Conda query failed for env \\texttt{{{_tex_escape(preferred_env)}}}: {_tex_escape(str(env_conda.get('error')))}.")

    versions = (env_primary.get("versions") or {}) if isinstance(env_primary, dict) else {}
    torch_cuda = (env_primary.get("torch_cuda") or {}) if isinstance(env_primary, dict) else {}
    # Use \path (nolinkurl) for long identifiers to permit line breaks.
    lines.append(f"Runtime ({env_primary_label}):")
    lines.append(f"Python: \\path{{{str(env_primary.get('python') or '')}}}.")
    lines.append(f"Platform: \\path{{{str(env_primary.get('platform') or '')}}}.")
    pkg_items = [
        f"\\texttt{{{_tex_escape(k)}}}={{{_tex_escape(str(v))}}}"
        for k, v in versions.items()
        if v is not None
    ]
    if pkg_items:
        lines.append("Packages: " + ", ".join(pkg_items) + ".")
    if isinstance(torch_cuda, dict) and torch_cuda:
        lines.append(
            "Torch CUDA: "
            f"available=\\texttt{{{str(bool(torch_cuda.get('cuda_is_available'))).lower()}}}, "
            f"cuda=\\texttt{{{_tex_escape(str(torch_cuda.get('cuda_version') or ''))}}}, "
            f"device0=\\path{{{str(torch_cuda.get('device0_name') or '')}}}."
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
        "Brain low-FPR matrix (fixed-profile pilots)": (
            "OpenSkull seed1; AliasContract seed2; SkullOR seed2; "
            "64-frame windows at offsets {0,64,128,192,256}."
        ),
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
            name="RTX 4080 SUPER audit logs (nvidia-smi + stap-fus conda env versions)",
            paper_refs=["Table: latency_summary_4080super"],
            outputs=[
                "reports/hw/4080super_nvidia_smi.txt",
                "reports/hw/4080super_env.txt",
            ],
            commands=[
                "nvidia-smi > reports/hw/4080super_nvidia_smi.txt",
                "",
                "conda run -n stap-fus python -c \"import platform,sys; import torch; "
                "print('platform:', platform.platform()); "
                "print('python:', sys.version.splitlines()[0]); "
                "print('torch:', torch.__version__); "
                "print('torch.version.cuda:', torch.version.cuda); "
                "print('cuda_is_available:', torch.cuda.is_available()); "
                "print('device0_name:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else None);\" "
                "> reports/hw/4080super_env.txt",
            ],
            notes="These logs are hardware-dependent and are captured once per machine/driver stack.",
        ),
        ArtifactInfo(
            name="RTX 4080 SUPER refactor gates (quick/phase/full)",
            paper_refs=["Reproducibility gates (Phase 7)"],
            outputs=[
                "reports/refactor/4080super_refactor_quick.log",
                "reports/refactor/4080super_refactor_phase.log",
                "reports/refactor/4080super_refactor_full.log",
            ],
            commands=[
                "conda run -n stap-fus make refactor-quick 2>&1 | tee reports/refactor/4080super_refactor_quick.log",
                "conda run -n stap-fus make refactor-phase 2>&1 | tee reports/refactor/4080super_refactor_phase.log",
                "conda run -n stap-fus make refactor-full  2>&1 | tee reports/refactor/4080super_refactor_full.log",
            ],
            notes="The 'full' gate is optional on constrained machines; see Makefile targets.",
        ),
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
                "  --pilots runs/pilot/r4c_kwave_seed1 \\",
                "  --profile Brain-OpenSkull \\",
                "  --out-root runs/ablation/condstap_leakage \\",
                "  --window-length 64 --window-offset 0 --disjoint-offset 64 \\",
                "  --summary-csv reports/condstap_leakage.csv \\",
                "  --summary-json reports/condstap_leakage.json",
            ],
        ),
        ArtifactInfo(
            name="CUDA latency replay (Brain-* k-Wave; steady-state windows 2..5)",
            paper_refs=["Table: latency_summary"],
            outputs=[
                "runs/latency_s12_publish_offsets/ (Brain-OpenSkull; full)",
                "runs/latency_s13_publish_offsets_cond/ (Brain-OpenSkull; conditional)",
                "runs/latency_s14_aliascontract_full/ (Brain-AliasContract; full)",
                "runs/latency_s14_aliascontract_cond/ (Brain-AliasContract; conditional)",
                "runs/latency_s15_skullor_full/ (Brain-SkullOR; full)",
                "runs/latency_s15_skullor_cond/ (Brain-SkullOR; conditional)",
            ],
            commands=[
                # OpenSkull: full + conditional
                "PYTHONPATH=. STAP_FAST_CUDA_GRAPH=1 conda run -n stap-fus \\",
                "  python scripts/latency_rerun_check.py \\",
                "  --src runs/latency_pilot_open \\",
                "  --out-root runs/latency_s12_publish_offsets \\",
                "  --profile Brain-OpenSkull \\",
                "  --window-length 64 --window-offset 0 \\",
                "  --window-offsets 0,64,128,192,256 \\",
                "  --stap-device cuda --stap-debug-samples 0 \\",
                "  --tile-batch 192 --cuda-warmup-heavy",
                "",
                "PYTHONPATH=. STAP_FAST_CUDA_GRAPH=1 conda run -n stap-fus \\",
                "  python scripts/latency_rerun_check.py \\",
                "  --src runs/latency_pilot_open \\",
                "  --out-root runs/latency_s13_publish_offsets_cond \\",
                "  --profile Brain-OpenSkull \\",
                "  --window-length 64 --window-offset 0 \\",
                "  --window-offsets 0,64,128,192,256 \\",
                "  --stap-device cuda --stap-debug-samples 0 \\",
                "  --tile-batch 192 --cuda-warmup-heavy \\",
                "  --stap-conditional on",
                "",
                # AliasContract: full + conditional
                "PYTHONPATH=. STAP_FAST_CUDA_GRAPH=1 conda run -n stap-fus \\",
                "  python scripts/latency_rerun_check.py \\",
                "  --src runs/latency_pilot_aliascontract \\",
                "  --out-root runs/latency_s14_aliascontract_full \\",
                "  --profile Brain-AliasContract \\",
                "  --window-length 64 --window-offset 0 \\",
                "  --window-offsets 0,64,128,192,256 \\",
                "  --stap-device cuda --stap-debug-samples 0 \\",
                "  --tile-batch 192 --cuda-warmup-heavy",
                "",
                "PYTHONPATH=. STAP_FAST_CUDA_GRAPH=1 conda run -n stap-fus \\",
                "  python scripts/latency_rerun_check.py \\",
                "  --src runs/latency_pilot_aliascontract \\",
                "  --out-root runs/latency_s14_aliascontract_cond \\",
                "  --profile Brain-AliasContract \\",
                "  --window-length 64 --window-offset 0 \\",
                "  --window-offsets 0,64,128,192,256 \\",
                "  --stap-device cuda --stap-debug-samples 0 \\",
                "  --tile-batch 192 --cuda-warmup-heavy \\",
                "  --stap-conditional on",
                "",
                # SkullOR: full + conditional
                "PYTHONPATH=. STAP_FAST_CUDA_GRAPH=1 conda run -n stap-fus \\",
                "  python scripts/latency_rerun_check.py \\",
                "  --src runs/latency_pilot_skullor \\",
                "  --out-root runs/latency_s15_skullor_full \\",
                "  --profile Brain-SkullOR \\",
                "  --window-length 64 --window-offset 0 \\",
                "  --window-offsets 0,64,128,192,256 \\",
                "  --stap-device cuda --stap-debug-samples 0 \\",
                "  --tile-batch 192 --cuda-warmup-heavy",
                "",
                "PYTHONPATH=. STAP_FAST_CUDA_GRAPH=1 conda run -n stap-fus \\",
                "  python scripts/latency_rerun_check.py \\",
                "  --src runs/latency_pilot_skullor \\",
                "  --out-root runs/latency_s15_skullor_cond \\",
                "  --profile Brain-SkullOR \\",
                "  --window-length 64 --window-offset 0 \\",
                "  --window-offsets 0,64,128,192,256 \\",
                "  --stap-device cuda --stap-debug-samples 0 \\",
                "  --tile-batch 192 --cuda-warmup-heavy \\",
                "  --stap-conditional on",
            ],
            notes=(
                "GPU latency is reported as the steady-state mean over windows 2..5 (window1 is cold: CUDA init, "
                "Triton JIT, and CUDA-graph capture). Use the script's 'optimized steady(avg win2..5)' line."
            ),
        ),
        ArtifactInfo(
            name="CUDA latency replay (real data: Shin + Gammex; steady-state frames 2..N)",
            paper_refs=["Table: latency_summary"],
            outputs=["runs/latency_s17_realdata_cuda_unfold/ (Shin + Gammex CUDA latency runs)"],
            commands=[
                # Shin RatBrain Fig3
                "PYTHONPATH=. \\",
                "STAP_TILING_UNFOLD=1 STAP_FAST_CUDA_GRAPH=1 \\",
                "STAP_SNAPSHOT_STRIDE=4 STAP_TYLER_MAX_ITER=1 STAP_TYLER_EARLY_STOP=0 \\",
                "STAP_TYLER_TRITON_CAPTURE=1 \\",
                "conda run -n stap-fus python scripts/latency_realdata_rerun_check.py \\",
                "  --stap-device cuda --tile-batch 192 --clean \\",
                "  --out-root runs/latency_s17_realdata_cuda_unfold \\",
                "  shin --iq-file IQData001.dat \\",
                "  --windows 0:128,64:192,122:250 \\",
                "  --Lt 64 --svd-energy-frac 0.97",
                "",
                # Gammex flow phantom
                "PYTHONPATH=. \\",
                "STAP_TILING_UNFOLD=1 STAP_FAST_CUDA_GRAPH=1 \\",
                "STAP_SNAPSHOT_STRIDE=4 STAP_TYLER_MAX_ITER=1 STAP_TYLER_EARLY_STOP=0 \\",
                "STAP_TYLER_TRITON_CAPTURE=1 \\",
                "conda run -n stap-fus python scripts/latency_realdata_rerun_check.py \\",
                "  --stap-device cuda --tile-batch 512 --clean \\",
                "  --out-root runs/latency_s17_realdata_cuda_unfold \\",
                "  gammex --frames-along 0:6 --frames-across 0:6",
            ],
            notes=(
                "The script prints cold(win1) and steady(avg win2..N); the latency table uses the steady-state mean "
                "and excludes window/frame 1 (graph capture + Triton JIT)."
            ),
        ),
        ArtifactInfo(
            name="RTX 4080 SUPER latency reruns + consolidated summaries (legacy vs optimized; steady-state windows/frames 2..N)",
            paper_refs=["Table: latency_summary_4080super"],
            outputs=[
                "reports/hw/4080super_nvidia_smi.txt",
                "reports/hw/4080super_env.txt",
                "runs/latency_4080super_brain_openskull_w64_tb192_off0_64_128_192_256/",
                "runs/latency_4080super_brain_openskull_w64_tb192_off0_64_128_192_256_cond/",
                "runs/latency_4080super_brain_aliascontract_w64_tb192_off0_64_128_192_256/",
                "runs/latency_4080super_brain_aliascontract_w64_tb192_off0_64_128_192_256_cond/",
                "runs/latency_4080super_brain_skullor_w64_tb192_off0_64_128_192_256/",
                "runs/latency_4080super_brain_skullor_w64_tb192_off0_64_128_192_256_cond/",
                "runs/latency_4080super_shin_fig3_w128_tb192/",
                "runs/latency_4080super_gammex_linear17_f0_6_tb512/",
                "reports/latency/4080super_latency_summary.csv",
                "reports/latency/4080super_latency_summary.json",
            ],
            commands=[
                "# Hardware audit (record driver/GPU; record stap-fus env versions):",
                "nvidia-smi > reports/hw/4080super_nvidia_smi.txt",
                "conda run -n stap-fus python -c 'import platform, sys; import torch; "
                "print(\"python\", sys.version.splitlines()[0]); "
                "print(\"platform\", platform.platform()); "
                "print(\"torch\", torch.__version__); "
                "print(\"torch_cuda\", getattr(torch.version, \"cuda\", None)); "
                "print(\"cuda_is_available\", torch.cuda.is_available()); "
                "print(\"device0\", torch.cuda.get_device_name(0) if torch.cuda.is_available() else None)' "
                "> reports/hw/4080super_env.txt",
                "",
                "# Brain k-Wave latency reruns (offsets 0/64/128/192/256; tile_batch=192):",
                "PYTHONPATH=. conda run -n stap-fus python scripts/latency_rerun_check.py \\",
                "  --src runs/pilot/r4c_kwave_seed1 \\",
                "  --out-root runs/latency_4080super_brain_openskull_w64_tb192_off0_64_128_192_256 \\",
                "  --profile Brain-OpenSkull \\",
                "  --window-length 64 --window-offsets 0,64,128,192,256 \\",
                "  --stap-device cuda --tile-batch 192 \\",
                "  --cuda-warmup-heavy --stap-debug-samples 0",
                "",
                "PYTHONPATH=. conda run -n stap-fus python scripts/latency_rerun_check.py \\",
                "  --src runs/pilot/r4c_kwave_seed1 \\",
                "  --out-root runs/latency_4080super_brain_openskull_w64_tb192_off0_64_128_192_256_cond \\",
                "  --profile Brain-OpenSkull \\",
                "  --window-length 64 --window-offsets 0,64,128,192,256 \\",
                "  --stap-device cuda --tile-batch 192 \\",
                "  --cuda-warmup-heavy --stap-debug-samples 0 \\",
                "  --stap-conditional on",
                "",
                "PYTHONPATH=. conda run -n stap-fus python scripts/latency_rerun_check.py \\",
                "  --src runs/pilot/r4c_kwave_hab_seed2 \\",
                "  --out-root runs/latency_4080super_brain_aliascontract_w64_tb192_off0_64_128_192_256 \\",
                "  --profile Brain-AliasContract \\",
                "  --window-length 64 --window-offsets 0,64,128,192,256 \\",
                "  --stap-device cuda --tile-batch 192 \\",
                "  --cuda-warmup-heavy --stap-debug-samples 0",
                "",
                "PYTHONPATH=. conda run -n stap-fus python scripts/latency_rerun_check.py \\",
                "  --src runs/pilot/r4c_kwave_hab_seed2 \\",
                "  --out-root runs/latency_4080super_brain_aliascontract_w64_tb192_off0_64_128_192_256_cond \\",
                "  --profile Brain-AliasContract \\",
                "  --window-length 64 --window-offsets 0,64,128,192,256 \\",
                "  --stap-device cuda --tile-batch 192 \\",
                "  --cuda-warmup-heavy --stap-debug-samples 0 \\",
                "  --stap-conditional on",
                "",
                "PYTHONPATH=. conda run -n stap-fus python scripts/latency_rerun_check.py \\",
                "  --src runs/pilot/r4c_kwave_hab_v3_skull_seed2 \\",
                "  --out-root runs/latency_4080super_brain_skullor_w64_tb192_off0_64_128_192_256 \\",
                "  --profile Brain-SkullOR \\",
                "  --window-length 64 --window-offsets 0,64,128,192,256 \\",
                "  --stap-device cuda --tile-batch 192 \\",
                "  --cuda-warmup-heavy --stap-debug-samples 0",
                "",
                "PYTHONPATH=. conda run -n stap-fus python scripts/latency_rerun_check.py \\",
                "  --src runs/pilot/r4c_kwave_hab_v3_skull_seed2 \\",
                "  --out-root runs/latency_4080super_brain_skullor_w64_tb192_off0_64_128_192_256_cond \\",
                "  --profile Brain-SkullOR \\",
                "  --window-length 64 --window-offsets 0,64,128,192,256 \\",
                "  --stap-device cuda --tile-batch 192 \\",
                "  --cuda-warmup-heavy --stap-debug-samples 0 \\",
                "  --stap-conditional on",
                "",
                "# Shin RatBrain Fig3 latency (requires dataset; see docs/data_download.md):",
                "PYTHONPATH=. STAP_FAST_PATH=1 STAP_TILING_UNFOLD=0 STAP_FAST_CUDA_GRAPH=0 \\",
                "STAP_SNAPSHOT_STRIDE=4 STAP_TYLER_MAX_ITER=1 STAP_TYLER_EARLY_STOP=0 STAP_TYLER_TRITON_CAPTURE=1 \\",
                "conda run -n stap-fus python scripts/latency_realdata_rerun_check.py \\",
                "  --stap-device cuda --tile-batch 192 --clean \\",
                "  --out-root runs/latency_4080super_shin_fig3_w128_tb192/legacy \\",
                "  shin --data-root data/shin_zenodo_10711806/ratbrain_fig3_raw \\",
                "  --iq-file IQData001.dat \\",
                "  --windows 0:128,64:192,122:250 \\",
                "  --prf-hz 1000 --Lt 64 --svd-energy-frac 0.97",
                "",
                "PYTHONPATH=. STAP_FAST_PATH=1 STAP_TILING_UNFOLD=1 STAP_FAST_CUDA_GRAPH=1 \\",
                "STAP_SNAPSHOT_STRIDE=4 STAP_TYLER_MAX_ITER=1 STAP_TYLER_EARLY_STOP=0 STAP_TYLER_TRITON_CAPTURE=1 \\",
                "conda run -n stap-fus python scripts/latency_realdata_rerun_check.py \\",
                "  --stap-device cuda --tile-batch 192 --clean \\",
                "  --out-root runs/latency_4080super_shin_fig3_w128_tb192/optimized \\",
                "  shin --data-root data/shin_zenodo_10711806/ratbrain_fig3_raw \\",
                "  --iq-file IQData001.dat \\",
                "  --windows 0:128,64:192,122:250 \\",
                "  --prf-hz 1000 --Lt 64 --svd-energy-frac 0.97",
                "",
                "# Gammex linear17 latency (along/across frames 0..5):",
                "PYTHONPATH=. STAP_FAST_PATH=1 STAP_TILING_UNFOLD=0 STAP_FAST_CUDA_GRAPH=0 \\",
                "STAP_SNAPSHOT_STRIDE=4 STAP_TYLER_MAX_ITER=1 STAP_TYLER_EARLY_STOP=0 STAP_TYLER_TRITON_CAPTURE=1 \\",
                "conda run -n stap-fus python scripts/latency_realdata_rerun_check.py \\",
                "  --stap-device cuda --tile-batch 512 --clean \\",
                "  --out-root runs/latency_4080super_gammex_linear17_f0_6_tb512/legacy \\",
                "  gammex --frames-along 0:6 --frames-across 0:6",
                "",
                "PYTHONPATH=. STAP_FAST_PATH=1 STAP_TILING_UNFOLD=1 STAP_FAST_CUDA_GRAPH=1 \\",
                "STAP_SNAPSHOT_STRIDE=4 STAP_TYLER_MAX_ITER=1 STAP_TYLER_EARLY_STOP=0 STAP_TYLER_TRITON_CAPTURE=1 \\",
                "conda run -n stap-fus python scripts/latency_realdata_rerun_check.py \\",
                "  --stap-device cuda --tile-batch 512 --clean \\",
                "  --out-root runs/latency_4080super_gammex_linear17_f0_6_tb512/optimized \\",
                "  gammex --frames-along 0:6 --frames-across 0:6",
                "",
                "# Collect consolidated CSV/JSON (steady windows/frames 2..N; includes parity checks):",
                "PYTHONPATH=. conda run -n stap-fus python scripts/collect_latency_summary.py",
            ],
            notes=(
                "The replay scripts print cold(win1) and steady(avg win2..N); the paper tables report steady-state means. "
                "For Brain k-Wave, scripts/latency_rerun_check.py forces legacy vs optimized CUDA-graph settings internally "
                "(legacy: STAP_FAST_CUDA_GRAPH=0; optimized: STAP_FAST_CUDA_GRAPH=1)."
            ),
        ),
        ArtifactInfo(
            name="RTX 4080 SUPER CUDA latency replay (Brain-* k-Wave; legacy vs optimized; steady-state windows 2..5)",
            paper_refs=["Table: latency_summary_4080super"],
            outputs=[
                "runs/latency_4080super_brain_openskull_w64_tb192_off0_64_128_192_256/",
                "runs/latency_4080super_brain_openskull_w64_tb192_off0_64_128_192_256_cond/",
                "runs/latency_4080super_brain_aliascontract_w64_tb192_off0_64_128_192_256/",
                "runs/latency_4080super_brain_aliascontract_w64_tb192_off0_64_128_192_256_cond/",
                "runs/latency_4080super_brain_skullor_w64_tb192_off0_64_128_192_256/",
                "runs/latency_4080super_brain_skullor_w64_tb192_off0_64_128_192_256_cond/",
            ],
            commands=[
                # OpenSkull seed1
                "PYTHONPATH=. conda run -n stap-fus python scripts/latency_rerun_check.py \\",
                "  --src runs/pilot/r4c_kwave_seed1 \\",
                "  --out-root runs/latency_4080super_brain_openskull_w64_tb192_off0_64_128_192_256 \\",
                "  --profile Brain-OpenSkull \\",
                "  --window-length 64 --window-offsets 0,64,128,192,256 \\",
                "  --stap-device cuda --stap-debug-samples 0 \\",
                "  --tile-batch 192 --cuda-warmup-heavy \\",
                "  --stap-conditional off",
                "",
                "PYTHONPATH=. conda run -n stap-fus python scripts/latency_rerun_check.py \\",
                "  --src runs/pilot/r4c_kwave_seed1 \\",
                "  --out-root runs/latency_4080super_brain_openskull_w64_tb192_off0_64_128_192_256_cond \\",
                "  --profile Brain-OpenSkull \\",
                "  --window-length 64 --window-offsets 0,64,128,192,256 \\",
                "  --stap-device cuda --stap-debug-samples 0 \\",
                "  --tile-batch 192 --cuda-warmup-heavy \\",
                "  --stap-conditional on",
                "",
                # AliasContract seed2
                "PYTHONPATH=. conda run -n stap-fus python scripts/latency_rerun_check.py \\",
                "  --src runs/pilot/r4c_kwave_hab_seed2 \\",
                "  --out-root runs/latency_4080super_brain_aliascontract_w64_tb192_off0_64_128_192_256 \\",
                "  --profile Brain-AliasContract \\",
                "  --window-length 64 --window-offsets 0,64,128,192,256 \\",
                "  --stap-device cuda --stap-debug-samples 0 \\",
                "  --tile-batch 192 --cuda-warmup-heavy \\",
                "  --stap-conditional off",
                "",
                "PYTHONPATH=. conda run -n stap-fus python scripts/latency_rerun_check.py \\",
                "  --src runs/pilot/r4c_kwave_hab_seed2 \\",
                "  --out-root runs/latency_4080super_brain_aliascontract_w64_tb192_off0_64_128_192_256_cond \\",
                "  --profile Brain-AliasContract \\",
                "  --window-length 64 --window-offsets 0,64,128,192,256 \\",
                "  --stap-device cuda --stap-debug-samples 0 \\",
                "  --tile-batch 192 --cuda-warmup-heavy \\",
                "  --stap-conditional on",
                "",
                # SkullOR seed2
                "PYTHONPATH=. conda run -n stap-fus python scripts/latency_rerun_check.py \\",
                "  --src runs/pilot/r4c_kwave_hab_v3_skull_seed2 \\",
                "  --out-root runs/latency_4080super_brain_skullor_w64_tb192_off0_64_128_192_256 \\",
                "  --profile Brain-SkullOR \\",
                "  --window-length 64 --window-offsets 0,64,128,192,256 \\",
                "  --stap-device cuda --stap-debug-samples 0 \\",
                "  --tile-batch 192 --cuda-warmup-heavy \\",
                "  --stap-conditional off",
                "",
                "PYTHONPATH=. conda run -n stap-fus python scripts/latency_rerun_check.py \\",
                "  --src runs/pilot/r4c_kwave_hab_v3_skull_seed2 \\",
                "  --out-root runs/latency_4080super_brain_skullor_w64_tb192_off0_64_128_192_256_cond \\",
                "  --profile Brain-SkullOR \\",
                "  --window-length 64 --window-offsets 0,64,128,192,256 \\",
                "  --stap-device cuda --stap-debug-samples 0 \\",
                "  --tile-batch 192 --cuda-warmup-heavy \\",
                "  --stap-conditional on",
            ],
            notes=(
                "The script writes both legacy/ and optimized/ subtrees per out-root and reports "
                "cold(win1) and steady(avg win2..N). Publishable latency uses windows 2..N."
            ),
        ),
        ArtifactInfo(
            name="RTX 4080 SUPER CUDA latency replay (real data: Shin + Gammex; legacy vs optimized; steady-state frames 2..N)",
            paper_refs=["Table: latency_summary_4080super"],
            outputs=[
                "runs/latency_4080super_shin_fig3_w128_tb192/legacy/",
                "runs/latency_4080super_shin_fig3_w128_tb192/optimized/",
                "runs/latency_4080super_gammex_linear17_f0_6_tb512/legacy/",
                "runs/latency_4080super_gammex_linear17_f0_6_tb512/optimized/",
            ],
            commands=[
                # Legacy (graphs disabled)
                "PYTHONPATH=. \\",
                "STAP_TILING_UNFOLD=0 STAP_FAST_CUDA_GRAPH=0 \\",
                "STAP_SNAPSHOT_STRIDE=4 STAP_TYLER_MAX_ITER=1 STAP_TYLER_EARLY_STOP=0 \\",
                "STAP_TYLER_TRITON_CAPTURE=1 \\",
                "conda run -n stap-fus python scripts/latency_realdata_rerun_check.py \\",
                "  --stap-device cuda --tile-batch 192 --clean \\",
                "  --out-root runs/latency_4080super_shin_fig3_w128_tb192/legacy \\",
                "  shin --iq-file IQData001.dat \\",
                "  --windows 0:128,64:192,122:250 \\",
                "  --Lt 64 --svd-energy-frac 0.97",
                "",
                # Gammex legacy
                "PYTHONPATH=. \\",
                "STAP_FAST_PATH=1 STAP_TILING_UNFOLD=0 STAP_FAST_CUDA_GRAPH=0 \\",
                "STAP_SNAPSHOT_STRIDE=4 STAP_TYLER_MAX_ITER=1 STAP_TYLER_EARLY_STOP=0 \\",
                "STAP_TYLER_TRITON_CAPTURE=1 \\",
                "conda run -n stap-fus python scripts/latency_realdata_rerun_check.py \\",
                "  --stap-device cuda --tile-batch 512 --clean \\",
                "  --out-root runs/latency_4080super_gammex_linear17_f0_6_tb512/legacy \\",
                "  gammex --frames-along 0:6 --frames-across 0:6",
                "",
                # Optimized (graphs enabled when supported)
                "PYTHONPATH=. \\",
                "STAP_TILING_UNFOLD=1 STAP_FAST_CUDA_GRAPH=1 \\",
                "STAP_SNAPSHOT_STRIDE=4 STAP_TYLER_MAX_ITER=1 STAP_TYLER_EARLY_STOP=0 \\",
                "STAP_TYLER_TRITON_CAPTURE=1 \\",
                "conda run -n stap-fus python scripts/latency_realdata_rerun_check.py \\",
                "  --stap-device cuda --tile-batch 192 --clean \\",
                "  --out-root runs/latency_4080super_shin_fig3_w128_tb192/optimized \\",
                "  shin --iq-file IQData001.dat \\",
                "  --windows 0:128,64:192,122:250 \\",
                "  --Lt 64 --svd-energy-frac 0.97",
                "",
                # Gammex optimized
                "PYTHONPATH=. \\",
                "STAP_FAST_PATH=1 STAP_TILING_UNFOLD=1 STAP_FAST_CUDA_GRAPH=1 \\",
                "STAP_SNAPSHOT_STRIDE=4 STAP_TYLER_MAX_ITER=1 STAP_TYLER_EARLY_STOP=0 \\",
                "STAP_TYLER_TRITON_CAPTURE=1 \\",
                "conda run -n stap-fus python scripts/latency_realdata_rerun_check.py \\",
                "  --stap-device cuda --tile-batch 512 --clean \\",
                "  --out-root runs/latency_4080super_gammex_linear17_f0_6_tb512/optimized \\",
                "  gammex --frames-along 0:6 --frames-across 0:6",
            ],
            notes=(
                "These are 'latency-only' replay settings (e.g., STAP_TYLER_MAX_ITER=1, STAP_SNAPSHOT_STRIDE=4) "
                "to isolate compute throughput; see per-bundle meta.json for exact telemetry."
            ),
        ),
        ArtifactInfo(
            name="RTX 4080 SUPER latency summary aggregation (CSV/JSON)",
            paper_refs=["Table: latency_summary_4080super"],
            outputs=[
                "reports/latency/4080super_latency_summary.csv",
                "reports/latency/4080super_latency_summary.json",
            ],
            commands=[
                "PYTHONPATH=. conda run -n stap-fus python scripts/collect_latency_summary.py",
            ],
            notes="Reads existing runs/latency_4080super_* folders and writes steady-state windows 2..N summaries.",
        ),
        ArtifactInfo(
            name="Brain-* baseline fairness: MC--SVD(e) tune-once sweep",
            paper_refs=["Baseline fairness (Brain-*)"],
            outputs=["reports/brain_mcsvd_energy_sweep_seed1.csv", "reports/brain_mcsvd_energy_sweep_seed1.json"],
            commands=[
                "PYTHONPATH=. python scripts/brain_mcsvd_energy_sweep.py \\",
                "  --pilot runs/pilot/r4c_kwave_seed1 \\",
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
            name="Brain-* low-FPR baseline matrix (vnext; includes adaptive/local SVD baselines)",
            paper_refs=["Table: brain_kwave_vnext_baselines", "Figure: brain_kwave_roc_curves.pdf"],
            outputs=[
                "runs/pilot/fair_filter_matrix_pd_r3_localbaselines/ (generated bundles)",
                "reports/fair_matrix_vnext_r3_localbaselines.csv",
                "reports/fair_matrix_vnext_r3_localbaselines.json",
                "reports/brain_kwave_vnext_baselines_table.tex",
            ],
            commands=[
                "bash scripts/reproduce_table5_brain_kwave.sh",
                "",
                "# Manual breakdown (equivalent to the script above):",
                "PYTHONPATH=. conda run -n stap-fus python scripts/fair_filter_comparison.py \\",
                "  --mode matrix --eval-score vnext \\",
                "  --matrix-regimes open,aliascontract,skullor \\",
                "  --matrix-seeds-open 1 --matrix-seeds-aliascontract 2 --matrix-seeds-skullor 2 \\",
                "  --window-length 64 --window-offsets 0,64,128,192,256 \\",
                "  --matrix-use-profile \\",
                "  --matrix-mcsvd-energy-frac 0.90 --matrix-mcsvd-baseline-support window \\",
                "  --methods mcsvd,svd_similarity,local_svd,rpca,hosvd,stap_full \\",
                "  --generated-root runs/pilot/fair_filter_matrix_pd_r3_localbaselines \\",
                "  --autogen-missing --stap-device cuda \\",
                "  --out-csv reports/fair_matrix_vnext_r3_localbaselines.csv \\",
                "  --out-json reports/fair_matrix_vnext_r3_localbaselines.json",
                "",
                "PYTHONPATH=. conda run -n stap-fus python scripts/brain_kwave_vnext_baselines_table.py \\",
                "  --fair-matrix-json reports/fair_matrix_vnext_r3_localbaselines.json \\",
                "  --out-tex reports/brain_kwave_vnext_baselines_table.tex",
            ],
            notes=(
                "Generates per-window acceptance bundles and writes a vnext-style strict-tail report, then renders the "
                "LaTeX table used in the main Brain-* baseline table; this run root is also consumed by the ROC and "
                "cross-window calibration scripts."
            ),
        ),
        ArtifactInfo(
            name="Brain-* ROC curve figure (median+IQR over disjoint windows)",
            paper_refs=["Figure: brain_kwave_roc_curves.pdf"],
            outputs=["figs/paper/brain_kwave_roc_curves.pdf"],
            commands=[
                "PYTHONPATH=. python scripts/fig_brain_kwave_roc_curves.py \\",
                "  --runs-root runs/pilot/fair_filter_matrix_pd_r3_localbaselines \\",
                "  --out-pdf figs/paper/brain_kwave_roc_curves.pdf",
            ],
            notes="Reads per-window score/mask arrays under the Brain-* pilot run root.",
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
                "  --runs-root runs/pilot/fair_filter_matrix_pd_r3_localbaselines \\",
                "  --alphas 1e-4,3e-4,1e-3 \\",
                "  --out-csv reports/brain_crosswindow_calibration.csv \\",
                "  --out-json reports/brain_crosswindow_calibration_summary.json",
            ],
            notes="Calibrates thresholds on one 64-frame window's negatives and evaluates on disjoint windows (ordered pairs).",
        ),
        ArtifactInfo(
            name="Brain-* baseline sanity table (strict vs relaxed operating points)",
            paper_refs=["Appendix: baseline sanity checks at relaxed operating points (Brain-*)"],
            outputs=[
                "reports/brain_baseline_sanity_relaxed.csv",
                "reports/brain_baseline_sanity_relaxed.json",
                "reports/brain_baseline_sanity_relaxed_table.tex",
            ],
            commands=[
                "PYTHONPATH=. python scripts/brain_baseline_sanity_table.py \\",
                "  --fair-matrix-json reports/fair_matrix_vnext_r3_localbaselines.json \\",
                "  --alphas 0.001,0.01,0.1 \\",
                "  --out-csv reports/brain_baseline_sanity_relaxed.csv \\",
                "  --out-json reports/brain_baseline_sanity_relaxed.json \\",
                "  --out-tex reports/brain_baseline_sanity_relaxed_table.tex",
            ],
            notes="Reads per-window score/mask arrays under the Brain-* pilot run root.",
        ),
        ArtifactInfo(
            name="Brain-* detector-component ablation (whitening vs band selectivity)",
            paper_refs=["Table: brain_detector_ablation (detector-component ablation)"],
            outputs=[
                "runs/pilot/fair_filter_matrix_pd_r3_localbaselines/*_mcsvd_det_whitened_power/ (generated bundles)",
                "runs/pilot/fair_filter_matrix_pd_r3_localbaselines/*_mcsvd_det_unwhitened_ratio/ (generated bundles)",
                "reports/fair_matrix_detector_ablations.csv",
                "reports/fair_matrix_detector_ablations.json",
                "reports/brain_detector_ablation.csv",
                "reports/brain_detector_ablation.json",
                "reports/brain_detector_ablation_table.tex",
            ],
            commands=[
                "PYTHONPATH=. conda run -n stap-fus python scripts/fair_filter_comparison.py \\",
                "  --mode matrix --eval-score vnext \\",
                "  --matrix-regimes open,aliascontract,skullor \\",
                "  --matrix-seeds-open 1 --matrix-seeds-aliascontract 2 --matrix-seeds-skullor 2 \\",
                "  --window-length 64 --window-offsets 0,64,128,192,256 \\",
                "  --matrix-use-profile \\",
                "  --matrix-mcsvd-energy-frac 0.90 --matrix-mcsvd-baseline-support full \\",
                "  --methods mcsvd,stap_det_whitened_power,stap_det_unwhitened_ratio,stap_full \\",
                "  --generated-root runs/pilot/fair_filter_matrix_pd_r3_localbaselines \\",
                "  --autogen-missing --stap-device cuda \\",
                "  --out-csv reports/fair_matrix_detector_ablations.csv \\",
                "  --out-json reports/fair_matrix_detector_ablations.json",
                "PYTHONPATH=. python scripts/brain_detector_ablation_table.py \\",
                "  --fair-matrix-json reports/fair_matrix_detector_ablations.json \\",
                "  --out-csv reports/brain_detector_ablation.csv \\",
                "  --out-json reports/brain_detector_ablation.json \\",
                "  --out-tex reports/brain_detector_ablation_table.tex",
            ],
            notes=(
                "Ablates detector components on the identical MC--SVD residual: "
                "whitened total power (no band partition) and unwhitened matched-subspace ratio (R=I)."
            ),
        ),
        ArtifactInfo(
            name="Brain-* detector-swap fairness check (matched-subspace scoring on competing residuals)",
            paper_refs=["Table: brain_detector_swap (paired residual detector swap)"],
            outputs=[
                "runs/pilot/fair_filter_matrix_pd_r3_localbaselines/*_hosvd_stap_full/ (generated bundles)",
                "runs/pilot/fair_filter_matrix_pd_r3_localbaselines/*_rpca_stap_full/ (generated bundles)",
                "reports/fair_matrix_detector_swap.csv",
                "reports/fair_matrix_detector_swap.json",
                "reports/brain_detector_swap.csv",
                "reports/brain_detector_swap.json",
                "reports/brain_detector_swap_table.tex",
            ],
            commands=[
                "PYTHONPATH=. conda run -n stap-fus python scripts/fair_filter_comparison.py \\",
                "  --mode matrix --eval-score vnext \\",
                "  --matrix-regimes open,aliascontract,skullor \\",
                "  --matrix-seeds-open 1 --matrix-seeds-aliascontract 2 --matrix-seeds-skullor 2 \\",
                "  --window-length 64 --window-offsets 0,64,128,192,256 \\",
                "  --matrix-use-profile \\",
                "  --methods rpca_pair,rpca_stap,hosvd_pair,hosvd_stap \\",
                "  --generated-root runs/pilot/fair_filter_matrix_pd_r3_localbaselines \\",
                "  --regen-runs --autogen-missing --stap-device cuda \\",
                "  --out-csv reports/fair_matrix_detector_swap.csv \\",
                "  --out-json reports/fair_matrix_detector_swap.json",
                "PYTHONPATH=. python scripts/brain_detector_swap_table.py \\",
                "  --fair-matrix-json reports/fair_matrix_detector_swap.json \\",
                "  --out-csv reports/brain_detector_swap.csv \\",
                "  --out-json reports/brain_detector_swap.json \\",
                "  --out-tex reports/brain_detector_swap_table.tex",
            ],
            notes=(
                "Paired-residual comparison: each run generates both PD (score_base.npy) and STAP "
                "(score_stap_preka.npy) scores on the same RPCA/HOSVD residual for each window."
            ),
        ),
        ArtifactInfo(
            name="Brain-* covariance-training trim ablation (self-training sensitivity)",
            paper_refs=["Results: covariance-training contamination ablation (Brain-*)"],
            outputs=[
                "runs/pilot/fair_filter_matrix_pd_r3_localbaselines/*_mcsvd_covtrim_q05/ (generated bundles)",
                "reports/fair_matrix_covtrim.csv",
                "reports/fair_matrix_covtrim.json",
                "reports/brain_cov_train_ablation.csv",
                "reports/brain_cov_train_ablation.json",
                "reports/brain_cov_train_ablation_table.tex",
            ],
            commands=[
                "PYTHONPATH=. conda run -n stap-fus python scripts/fair_filter_comparison.py \\",
                "  --mode matrix --eval-score vnext \\",
                "  --matrix-regimes open,aliascontract,skullor \\",
                "  --matrix-seeds-open 1 --matrix-seeds-aliascontract 2 --matrix-seeds-skullor 2 \\",
                "  --window-length 64 --window-offsets 0,64,128,192,256 \\",
                "  --matrix-use-profile \\",
                "  --matrix-mcsvd-energy-frac 0.90 --matrix-mcsvd-baseline-support full \\",
                "  --methods stap_full,stap_covtrim_q05 \\",
                "  --generated-root runs/pilot/fair_filter_matrix_pd_r3_localbaselines \\",
                "  --autogen-missing --stap-device cuda \\",
                "  --out-csv reports/fair_matrix_covtrim.csv \\",
                "  --out-json reports/fair_matrix_covtrim.json",
                "PYTHONPATH=. python scripts/brain_cov_train_ablation_table.py \\",
                "  --runs-root runs/pilot/fair_filter_matrix_pd_r3_localbaselines \\",
                "  --alphas 1e-4,3e-4,1e-3 \\",
                "  --out-csv reports/brain_cov_train_ablation.csv \\",
                "  --out-json reports/brain_cov_train_ablation.json \\",
                "  --out-tex reports/brain_cov_train_ablation_table.tex",
            ],
            notes=(
                "Enables the ablation via replay_stap_from_run.py --stap-cov-trim-q (plumbed through fair_filter_comparison "
                "method key stap_covtrim_q05). This excludes the top 5% highest-energy Hankel snapshots per tile when estimating "
                "R_i (guard/trim style training contamination check)."
            ),
        ),
        ArtifactInfo(
            name="Brain-OpenSkull fixed-profile parameter sensitivity sweep (diag-load, tile size, Lt)",
            paper_refs=[
                "Appendix: fixed-profile parameter sensitivity (Brain-OpenSkull)",
                "Results: covariance-training contamination ablation (Brain-*)",
            ],
            outputs=[
                "runs/pilot/brain_openskull_profile_sensitivity_r1/ (generated bundles)",
                "reports/brain_openskull_profile_sensitivity.csv",
                "reports/brain_openskull_profile_sensitivity.json",
                "figs/paper/brain_openskull_profile_sensitivity.pdf",
            ],
            commands=[
                "PYTHONPATH=. conda run -n stap-fus python scripts/brain_openskull_profile_sensitivity.py \\",
                "  --autogen-missing --stap-device cuda",
            ],
            notes=(
                "Runs 1D sweeps (vary one parameter at a time) around the frozen Brain-OpenSkull operating profile, "
                "reporting within-window strict-tail TPR medians/IQR over five disjoint windows. "
                "Sweep ranges are configurable via --diag-loads/--tile-sizes/--lts."
            ),
        ),
        ArtifactInfo(
            name="Brain-* strict-tail collapse visual (representative window; baseline vs STAP)",
            paper_refs=["Appendix: baseline sanity checks at relaxed operating points (Brain-*)"],
            outputs=["figs/paper/brain_tail_collapse_visual.pdf"],
            commands=[
                "ROOT=\"runs/pilot/fair_filter_matrix_pd_r3_localbaselines/open_seed1_mcsvd_full\"",
                "BUNDLE_DIR=\"$ROOT/pw_7.5MHz_5ang_5ens_64T_seed1_win0_off0\"",
                "PYTHONPATH=. python scripts/fig_brain_tail_collapse_visual.py \\",
                "  --bundle-dir \"$BUNDLE_DIR\" \\",
                "  --alphas 0.1,0.01,1e-3 \\",
                "  --out figs/paper/brain_tail_collapse_visual.pdf",
            ],
            notes="Uses one representative Brain-OpenSkull window to visualize background-tail dominance (baseline PD) and tail reshaping (STAP).",
        ),
        ArtifactInfo(
            name="Knowledge-aided prior falsifiability ablation (STAP-only vs contract vs forced)",
            paper_refs=["KA evaluation discipline"],
            outputs=["reports/ka_v2_ablation.csv", "reports/ka_v2_ablation.json"],
            commands=[
                "PYTHONPATH=. python scripts/ka_contract_v2_ablation.py \\",
                "  --pilot runs/pilot/r4c_kwave_hab_contract_seed2_v2 \\",
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
                "  --pilot runs/pilot/r4c_kwave_seed1 \\",
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
                "  --scan-name scan_anatomy \\",
                "  --plane-indices 5 10 15 \\",
                "  --out-png figs/paper/mace_atlas_overlay.png",
            ],
            notes="Overlays atlas ROIs and atlas.Vascular contours on the anatomical reference volume for representative planes (sanity check of Transformation.mat alignment).",
        ),
        ArtifactInfo(
            name="Telemetry regime comparison (sim vs real; contract-v2 scalars)",
            paper_refs=["Figure: telemetry_regime_compare.png"],
            outputs=["figs/paper/telemetry_regime_compare.png"],
            commands=[
                "# Sample Brain-* replay bundles (write meta.json with ka_contract_v2 telemetry):",
                "PYTHONPATH=. python scripts/replay_stap_from_run.py \\",
                "  --src runs/pilot/r4c_kwave_seed1 \\",
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
            name="Leading structural-fidelity figure (Gammex phantom; matched-FPR decision differences)",
            paper_refs=["Figure 1: leading_structural_fidelity_gammex.pdf"],
            outputs=["figs/paper/leading_structural_fidelity_gammex.pdf"],
            commands=[
                "SEQ_DIR=\"data/twinkling_artifact/Flow in Gammex phantom\"",
                "SEQ_DIR=\"$SEQ_DIR/Flow in Gammex phantom (along - linear probe)\"",
                "PYTHONPATH=. conda run -n stap-fus python scripts/fig_leading_structural_fidelity.py \\",
                "  --seq-dir \"$SEQ_DIR\" \\",
                "  --frame-idx 0 --mask-ref-frames 0:10 \\",
                "  --prf-hz 2500 --Lt 16 \\",
                "  --tile-hw 8 8 --tile-stride 6 \\",
                "  --cov-estimator tyler_pca --diag-load 0.07 \\",
                "  --fd-span-mode psd --feasibility-mode legacy \\",
                "  --svd-keep-min 2 --svd-keep-max 17 \\",
                "  --flow-band-hz 150 450 --alias-band-hz 700 1200 \\",
                "  --fpr 1e-2 \\",
                "  --stap-device auto \\",
                "  --out figs/paper/leading_structural_fidelity_gammex.pdf",
            ],
            notes=(
                "Generates a single-frame qualitative panel using the structurally labeled tube mask; "
                "the script writes a temporary one-frame acceptance bundle under runs/_tmp_leading_structural_fidelity/."
            ),
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
            name="SIMUS/PyMUST moving-scatterer sanity link + bundle contract (micro + alias; non-performance claims)",
            paper_refs=[
                "Simulation spec: moving-scatterer physical Doppler regime (SIMUS)",
                "Appendix (QC): physical-doppler sanity-link telemetry",
            ],
            outputs=[
                # Sanity-link telemetry vs real IQ (tracked).
                "reports/physdoppler_sanity_link/simus_micro_paper_seed0_vs_shin_gammex_summary.json",
                "reports/physdoppler_sanity_link/simus_micro_paper_seed0_vs_shin_gammex_table.json",
                "reports/physdoppler_sanity_link/simus_alias_paper_seed0_vs_shin_gammex_summary.json",
                "reports/physdoppler_sanity_link/simus_alias_paper_seed0_vs_shin_gammex_table.json",
                # Baseline-vs-STAP comparison (tracked).
                "reports/simus_baseline_compare/simus_pymust_paper_seed0_compare.csv",
                "reports/simus_baseline_compare/simus_pymust_paper_seed0_compare.json",
                # Contract check logs (tracked).
                "reports/simus_contract/simus_pymust_paper_micro_seed0_hab_contract.txt",
                "reports/simus_contract/simus_pymust_paper_alias_seed0_hab_contract.txt",
                # Derived acceptance bundles (not tracked; runs/ is ignored).
                "runs/sim/simus_pymust_paper_micro_seed0/bundle/simus_pymust_paper_micro_seed0/",
                "runs/sim/simus_pymust_paper_alias_seed0/bundle/simus_pymust_paper_alias_seed0/",
                # Bundle-sweep dirs for baseline comparisons (not tracked; runs/ is ignored).
                "runs/sim_eval/simus_baseline_compare_r1/",
            ],
            commands=[
                # Canonical SIMUS datasets (not tracked; deterministic given seed).
                "PYTHONPATH=. conda run -n stap-fus python sim/simus/pilot_pymust_simus.py \\",
                "  --out runs/sim/simus_pymust_paper_micro_seed0 --preset microvascular_like --tier paper --seed 0 --skip-bundle",
                "PYTHONPATH=. conda run -n stap-fus python sim/simus/pilot_pymust_simus.py \\",
                "  --out runs/sim/simus_pymust_paper_alias_seed0 --preset alias_stress --tier paper --seed 0 --skip-bundle",
                "",
                # Bundle derivation from canonical dataset/ (no re-sim).
                "PYTHONPATH=. conda run -n stap-fus python scripts/icube_make_bundle.py \\",
                "  --run runs/sim/simus_pymust_paper_micro_seed0 --stap-device cpu",
                "PYTHONPATH=. conda run -n stap-fus python scripts/icube_make_bundle.py \\",
                "  --run runs/sim/simus_pymust_paper_alias_seed0 --stap-device cpu",
                "",
                # HAB contract check logs.
                "PYTHONPATH=. conda run -n stap-fus python scripts/hab_contract_check.py \\",
                "  runs/sim/simus_pymust_paper_micro_seed0/bundle/simus_pymust_paper_micro_seed0 \\",
                "  2>&1 | tee reports/simus_contract/simus_pymust_paper_micro_seed0_hab_contract.txt",
                "PYTHONPATH=. conda run -n stap-fus python scripts/hab_contract_check.py \\",
                "  runs/sim/simus_pymust_paper_alias_seed0/bundle/simus_pymust_paper_alias_seed0 \\",
                "  2>&1 | tee reports/simus_contract/simus_pymust_paper_alias_seed0_hab_contract.txt",
                "",
                # Sanity-link telemetry vs real IQ (Shin Fig3 + Gammex phantom).
                "PYTHONPATH=. conda run -n stap-fus python scripts/physical_doppler_sanity_link.py \\",
                "  --sim-run runs/sim/simus_pymust_paper_micro_seed0 \\",
                "  --shin-root data/shin_zenodo_10711806/ratbrain_fig3_raw --shin-iq-file IQData001.dat \\",
                "  --shin-frames 0:128 --shin-prf-hz 1000 \\",
                "  --gammex-seq-root \"data/twinkling_artifact/Flow in Gammex phantom\" \\",
                "  --gammex-frames-along 0 --gammex-frames-across 0 --gammex-prf-hz 2500 \\",
                "  --gammex-mask-mode bmode_tube --gammex-mask-ref-frames 0:6 \\",
                "  --pf 30 250 --pg 250 400 --pa-lo 400 --tile-hw 8 8 --tile-stride 3 \\",
                "  --tag simus_micro_paper_seed0_vs_shin_gammex",
                "PYTHONPATH=. conda run -n stap-fus python scripts/physical_doppler_sanity_link.py \\",
                "  --sim-run runs/sim/simus_pymust_paper_alias_seed0 \\",
                "  --shin-root data/shin_zenodo_10711806/ratbrain_fig3_raw --shin-iq-file IQData001.dat \\",
                "  --shin-frames 0:128 --shin-prf-hz 1000 \\",
                "  --gammex-seq-root \"data/twinkling_artifact/Flow in Gammex phantom\" \\",
                "  --gammex-frames-along 0 --gammex-frames-across 0 --gammex-prf-hz 2500 \\",
                "  --gammex-mask-mode bmode_tube --gammex-mask-ref-frames 0:6 \\",
                "  --pf 30 250 --pg 250 400 --pa-lo 400 --tile-hw 8 8 --tile-stride 3 \\",
                "  --tag simus_alias_paper_seed0_vs_shin_gammex",
                "",
                # Baseline vs STAP comparison (paper-style baselines; uses sim ground-truth masks).
                "PYTHONPATH=. conda run -n stap-fus python scripts/icube_baseline_compare.py \\",
                "  --run runs/sim/simus_pymust_paper_micro_seed0 \\",
                "  --run runs/sim/simus_pymust_paper_alias_seed0 \\",
                "  --out-root runs/sim_eval/simus_baseline_compare_r1 \\",
                "  --out-csv reports/simus_baseline_compare/simus_pymust_paper_seed0_compare.csv \\",
                "  --out-json reports/simus_baseline_compare/simus_pymust_paper_seed0_compare.json \\",
                "  --tag clinical_like --stap-device cpu",
            ],
            notes=(
                "SIMUS/PyMUST runs are used as a moving-scatterer credibility anchor. "
                "Sanity-link summaries compare PSD-band / coherence / low-rank proxies against open real IQ; "
                "contract checks verify bundle integrity and KA-friendly regime telemetry. "
                "No performance claims are made from these comparisons."
            ),
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


def _sha256(path: Path) -> str:
    import hashlib

    h = hashlib.sha256()
    h.update(path.read_bytes())
    return h.hexdigest()


def build_manifest(*, conda_env: str | None, public_repo_url: str | None, release_tag: str | None) -> dict[str, Any]:
    env_current = _env_info_current()
    env_conda = _env_info_conda(conda_env) if conda_env else {"error": "conda env not set"}

    env_spec_path = REPO / "environment.yml"
    dockerfile_path = REPO / "Dockerfile"
    env_spec = {
        "path": str(env_spec_path.relative_to(REPO)) if env_spec_path.exists() else "environment.yml",
        "sha256": _sha256(env_spec_path) if env_spec_path.exists() else None,
    }
    docker_spec = {
        "path": str(dockerfile_path.relative_to(REPO)) if dockerfile_path.exists() else "Dockerfile",
        "sha256": _sha256(dockerfile_path) if dockerfile_path.exists() else None,
    }

    repo_meta = _repo_meta()
    if public_repo_url:
        repo_meta["public_url"] = public_repo_url
    if release_tag:
        repo_meta["release_tag"] = release_tag

    manifest: dict[str, Any] = {
        "repo": repo_meta,
        "git": _git_info(),
        "env": {
            "preferred_conda_env": conda_env,
            "conda": env_conda,
            "current": env_current,
            "environment_yml": env_spec,
            "dockerfile": docker_spec,
        },
        "datasets": [asdict(d) for d in _default_datasets()],
        "selections": _default_selections(),
        "artifacts": [asdict(a) for a in _default_artifacts()],
    }
    return manifest


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Generate reproducibility manifest (JSON + LaTeX appendix).")
    ap.add_argument("--out-json", type=Path, default=Path("repro_manifest.json"))
    ap.add_argument("--out-tex", type=Path, default=Path("appendix_repro_manifest.tex"))
    ap.add_argument(
        "--conda-env",
        type=str,
        default=os.environ.get("STAP_FUS_CONDA_ENV", "stap-fus"),
        help="Conda env name to query for package/CUDA versions (preferred).",
    )
    ap.add_argument(
        "--public-repo-url",
        type=str,
        default=os.environ.get("STAP_FUS_PUBLIC_REPO_URL", ""),
        help="Optional public repository URL to record in the manifest.",
    )
    ap.add_argument(
        "--release-tag",
        type=str,
        default=os.environ.get("STAP_FUS_RELEASE_TAG", ""),
        help="Optional release tag (e.g., v0.1.0) to record in the manifest.",
    )
    return ap.parse_args()


def main() -> None:
    args = parse_args()

    manifest = build_manifest(
        conda_env=str(args.conda_env).strip() or None,
        public_repo_url=str(args.public_repo_url).strip() or None,
        release_tag=str(args.release_tag).strip() or None,
    )
    args.out_json.write_text(json.dumps(manifest, indent=2))
    _render_appendix_tex(manifest, out_path=args.out_tex)

    print(f"[repro-manifest] wrote {args.out_json}")
    print(f"[repro-manifest] wrote {args.out_tex}")


if __name__ == "__main__":
    main()
