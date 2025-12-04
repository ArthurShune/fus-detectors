# pipeline/utils/telemetry.py
from __future__ import annotations

import os
import platform
import subprocess
import uuid
from pathlib import Path
from typing import Any, Dict, List, Optional


def _try_imports() -> Dict[str, Any]:
    info: Dict[str, Any] = {
        "gpu": None,
        "gpu_count": 0,
        "cuda": None,
        "torch": None,
        "cupy": None,
        "kwave": None,
    }
    try:
        import torch

        info["torch"] = torch.__version__
        info["gpu_count"] = torch.cuda.device_count()
        if torch.cuda.is_available():
            props = torch.cuda.get_device_properties(torch.cuda.current_device())
            info["cuda"] = torch.version.cuda
            info["gpu"] = {
                "name": props.name,
                "total_mem_MB": props.total_memory / 1e6,
                "multi_processor_count": props.multi_processor_count,
                "compute_capability": f"{props.major}.{props.minor}",
            }
        else:
            info["gpu"] = None
    except Exception:
        pass
    try:
        import cupy as cp  # type: ignore

        info["cupy"] = cp.__version__
    except Exception:
        info["cupy"] = None
    try:
        import kwave  # type: ignore

        info["kwave"] = getattr(kwave, "__version__", "unknown")
    except Exception:
        info["kwave"] = None
    return info


def _run_cmd(cmd: List[str], cwd: Optional[Path] = None) -> Optional[str]:
    try:
        res = subprocess.run(
            cmd,
            cwd=cwd,
            capture_output=True,
            text=True,
            check=False,
            timeout=2.0,
        )
        if res.returncode == 0:
            return res.stdout.strip()
    except Exception:
        return None
    return None


def _git_info() -> Dict[str, Any]:
    root = Path(__file__).resolve().parents[2]
    if not (root / ".git").exists():
        return {"present": False}
    info: Dict[str, Any] = {"present": True}
    commit = _run_cmd(["git", "rev-parse", "HEAD"], cwd=root)
    short = _run_cmd(["git", "rev-parse", "--short", "HEAD"], cwd=root)
    branch = _run_cmd(["git", "rev-parse", "--abbrev-ref", "HEAD"], cwd=root)
    describe = _run_cmd(["git", "describe", "--tags", "--dirty", "--always"], cwd=root)
    status = _run_cmd(["git", "status", "--porcelain"], cwd=root)
    info.update(
        {
            "commit": commit,
            "commit_short": short,
            "branch": branch,
            "describe": describe,
            "dirty": bool(status),
        }
    )
    return info


def _nvidia_smi_snapshot() -> Optional[List[Dict[str, Any]]]:
    cmd = [
        "nvidia-smi",
        "--query-gpu=index,name,utilization.gpu,utilization.memory,memory.used,memory.total,temperature.gpu,clocks.sm",
        "--format=csv,noheader,nounits",
    ]
    try:
        res = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            check=False,
            timeout=2.0,
        )
    except Exception:
        return None
    if res.returncode != 0 or not res.stdout:
        return None
    snapshot: List[Dict[str, Any]] = []
    for line in res.stdout.strip().splitlines():
        parts = [p.strip() for p in line.split(",")]
        if len(parts) != 8:
            continue
        try:
            snapshot.append(
                {
                    "index": int(parts[0]),
                    "name": parts[1],
                    "util_gpu_pct": float(parts[2]),
                    "util_mem_pct": float(parts[3]),
                    "mem_used_MB": float(parts[4]),
                    "mem_total_MB": float(parts[5]),
                    "temp_C": float(parts[6]),
                    "sm_clock_MHz": float(parts[7]),
                }
            )
        except ValueError:
            continue
    return snapshot or None


def sample_gpu_stats(include_nvidia_smi: bool = False) -> Dict[str, Any]:
    stats: Dict[str, Any] = {}
    try:
        import torch

        if torch.cuda.is_available():
            idx = torch.cuda.current_device()
            stats.update(
                {
                    "device_index": idx,
                    "current_mem_MB": torch.cuda.memory_allocated(idx) / 1e6,
                    "max_mem_MB": torch.cuda.max_memory_allocated(idx) / 1e6,
                    "reserved_mem_MB": torch.cuda.memory_reserved(idx) / 1e6,
                    "max_reserved_mem_MB": torch.cuda.max_memory_reserved(idx) / 1e6,
                }
            )
    except Exception:
        pass
    if include_nvidia_smi:
        stats["nvidia_smi"] = _nvidia_smi_snapshot()
    return stats


def system_telemetry(include_nvidia_smi: bool = True) -> Dict[str, Any]:
    hw = _try_imports()
    env = {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "pid": os.getpid(),
        "conda_env": os.environ.get("CONDA_DEFAULT_ENV"),
    }
    telemetry = {
        "run_id": str(uuid.uuid4()),
        "hw": hw,
        "env": env,
        "git": _git_info(),
    }
    if include_nvidia_smi:
        telemetry["nvidia_smi"] = _nvidia_smi_snapshot()
    return telemetry
