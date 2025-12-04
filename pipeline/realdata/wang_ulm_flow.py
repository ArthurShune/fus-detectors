from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Sequence, Tuple

import numpy as np

from pipeline.realdata import WangIQInfo, wang_data_root
from pipeline.realdata.wang_ulm import (
    load_wang_iq,
    load_wang_metadata,
    list_wang_files,
)


@dataclass
class ULMFlowConfig:
    """
    Minimal configuration for a Wang ULM-like flow map.

    The defaults are chosen to keep the computation light while providing a
    sparse vessel/flow map suitable for telemetry masks.
    """

    file_index: int = 1  # 1-based IQDataXXX index
    frames: Optional[Tuple[int, int]] = (0, 255)  # inclusive [start, stop]
    quantile: float = 0.999  # global power threshold for bubble detection
    highpass: str = "diff"  # 'diff' (2nd-order difference) or 'svd'
    svd_k: int = 1  # number of leading singular modes to remove if highpass='svd'


def _frame_indices(
    frames: Optional[Tuple[int, int]], max_frames: int
) -> Optional[np.ndarray]:
    """
    Convert an inclusive (start, stop) range to an index array, or None.
    """

    if frames is None:
        return None
    start, stop = int(frames[0]), int(frames[1])
    if stop < start:
        raise ValueError("frames STOP must be >= START")
    if start < 0 or stop >= max_frames:
        raise ValueError(
            f"frames must lie within [0, {max_frames - 1}], got ({start}, {stop})"
        )
    return np.arange(start, stop + 1, dtype=int)


def highpass_iq_temporal(iq: np.ndarray) -> np.ndarray:
    """
    Simple temporal clutter suppression on complex IQ data.

    Applies per-pixel mean subtraction followed by a 2nd-order difference
    along the time axis:

        iq_hp[t] = iq[t + 2] - 2 * iq[t + 1] + iq[t]

    Parameters
    ----------
    iq : np.ndarray
        Complex IQ array shaped (T, H, W).

    Returns
    -------
    np.ndarray
        High-pass IQ array with shape (T-2, H, W). If T < 3, returns an
        empty array with shape (0, H, W).
    """

    if iq.ndim != 3:
        raise ValueError(f"iq must be 3-D (T,H,W), got shape {iq.shape}")

    T = iq.shape[0]
    if T < 3:
        H, W = iq.shape[1], iq.shape[2]
        return np.zeros((0, H, W), dtype=iq.dtype)

    iq = iq.astype(np.complex64, copy=False)
    iq = iq - iq.mean(axis=0, keepdims=True)
    hp = iq[2:] - 2.0 * iq[1:-1] + iq[:-2]
    return hp


def highpass_iq_svd(iq: np.ndarray, k: int = 1) -> np.ndarray:
    """
    Temporal clutter suppression via low-rank removal on the (T, H*W) matrix.

    Removes the first k left singular modes from the time-by-pixel matrix and
    returns the residual reshaped to (T, H, W).
    """

    if iq.ndim != 3:
        raise ValueError(f"iq must be 3-D (T,H,W), got shape {iq.shape}")
    T, H, W = iq.shape
    if T == 0:
        return iq
    if k < 0:
        raise ValueError(f"k must be >=0, got {k}")
    X = iq.reshape(T, -1).astype(np.complex64, copy=False)
    X = X - X.mean(axis=0, keepdims=True)
    # Full SVD on (T x HW) with T small (~256) and HW moderate (~50k).
    U, S, Vh = np.linalg.svd(X, full_matrices=False)
    k_eff = min(k, S.shape[0])
    if k_eff > 0:
        U_k = U[:, :k_eff]
        S_k = S[:k_eff]
        Vh_k = Vh[:k_eff, :]
        X -= (U_k * S_k) @ Vh_k
    return X.reshape(T, H, W)


def detect_bubbles_threshold(
    iq_hp: np.ndarray,
    quantile: float = 0.999,
) -> np.ndarray:
    """
    Simple magnitude-based bubble detection on high-pass IQ.

    Parameters
    ----------
    iq_hp : np.ndarray
        High-pass IQ array shaped (T, H, W).
    quantile : float
        Global power quantile used as a threshold; pixels above this value
        are considered bubble detections.

    Returns
    -------
    np.ndarray
        Boolean array shaped (T, H, W) indicating detected bubbles.
    """

    if iq_hp.ndim != 3:
        raise ValueError(f"iq_hp must be 3-D (T,H,W), got shape {iq_hp.shape}")
    if iq_hp.shape[0] == 0:
        H, W = iq_hp.shape[1], iq_hp.shape[2]
        return np.zeros((0, H, W), dtype=bool)

    pd = (np.abs(iq_hp) ** 2).astype(np.float32, copy=False)
    flat = pd.ravel()
    if flat.size == 0:
        H, W = iq_hp.shape[1], iq_hp.shape[2]
        return np.zeros((pd.shape[0], H, W), dtype=bool)

    q = float(quantile)
    if not (0.0 < q < 1.0):
        raise ValueError(f"quantile must be in (0,1), got {q}")

    thr = np.quantile(flat, q)
    mask = pd > thr
    return mask


def accumulate_flow_map(mask: np.ndarray) -> np.ndarray:
    """
    Accumulate per-frame detections into a vessel/flow map.

    Parameters
    ----------
    mask : np.ndarray
        Boolean detection mask shaped (T, H, W).

    Returns
    -------
    np.ndarray
        Float32 vessel/flow map shaped (H, W), normalized to [0, 1].
    """

    if mask.ndim != 3:
        raise ValueError(f"mask must be 3-D (T,H,W), got shape {mask.shape}")
    if mask.shape[0] == 0:
        H, W = mask.shape[1], mask.shape[2]
        return np.zeros((H, W), dtype=np.float32)

    counts = mask.sum(axis=0).astype(np.float32)
    max_val = float(counts.max())
    if max_val > 0.0:
        counts /= max_val
    return counts


def compute_flow_map_for_wang_file(
    cfg: ULMFlowConfig,
    data_root: Path | None = None,
) -> np.ndarray:
    """
    Convenience wrapper: load a Wang IQData file, apply a minimal ULM-like
    pipeline, and return a (H, W) flow map.
    """

    info = load_wang_metadata(data_root)
    files = list_wang_files(data_root)
    if not files:
        raise RuntimeError("No IQDataXXX.dat files found.")
    if cfg.file_index < 1 or cfg.file_index > len(files):
        raise ValueError(f"file_index {cfg.file_index} out of range 1..{len(files)}")
    path = files[cfg.file_index - 1]

    idx = _frame_indices(cfg.frames, info.frames)
    iq = load_wang_iq(path, info, frames=idx, dtype=np.complex64)  # (T, H, W)

    if cfg.highpass == "diff":
        iq_hp = highpass_iq_temporal(iq)
    elif cfg.highpass == "svd":
        iq_hp = highpass_iq_svd(iq, k=int(cfg.svd_k))
    else:
        raise ValueError(f"Unknown highpass mode '{cfg.highpass}', expected 'diff' or 'svd'")
    mask = detect_bubbles_threshold(iq_hp, quantile=cfg.quantile)
    flow = accumulate_flow_map(mask)
    return flow


def compute_and_save_flow_map(
    cfg: ULMFlowConfig,
    out_path: Path | None = None,
    data_root: Path | None = None,
) -> np.ndarray:
    """
    Compute a flow map for a Wang IQData file and save it to disk.

    Parameters
    ----------
    cfg : ULMFlowConfig
        Configuration specifying file index, frame range, and detection quantile.
    out_path : Path, optional
        Output .npy path. If None, defaults to data/wang_ulm/flow_map.npy.
    data_root : Path, optional
        Root containing IQDataXXX.dat and IQSizeInfo.mat. If None, uses
        pipeline.realdata.wang_data_root().
    """

    flow = compute_flow_map_for_wang_file(cfg, data_root=data_root)

    if out_path is None:
        base = wang_data_root() if data_root is None else Path(data_root)
        out_path = base / "flow_map.npy"

    out_path.parent.mkdir(parents=True, exist_ok=True)
    np.save(out_path, flow.astype(np.float32, copy=False))
    return flow


__all__ = [
    "ULMFlowConfig",
    "highpass_iq_temporal",
    "detect_bubbles_threshold",
    "accumulate_flow_map",
    "compute_flow_map_for_wang_file",
    "compute_and_save_flow_map",
]
