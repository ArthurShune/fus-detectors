from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Optional, Sequence, Tuple

import numpy as np
import scipy.io as sio

from pipeline.realdata import WangIQInfo, wang_data_root


def load_wang_metadata(root: Path | None = None) -> WangIQInfo:
    """
    Load IQSizeInfo.mat and return basic dimensions (row, col, frames).
    """

    base = wang_data_root() if root is None else Path(root)
    info_path = base / "IQSizeInfo.mat"
    if not info_path.is_file():
        raise FileNotFoundError(f"IQSizeInfo.mat not found under {base}")
    mat = sio.loadmat(info_path)
    if "IQSizeInfo" not in mat:
        raise KeyError(f"Expected IQSizeInfo in {info_path}, found keys={list(mat.keys())}")
    arr = np.asarray(mat["IQSizeInfo"]).ravel()
    if arr.size != 3:
        raise ValueError(f"Unexpected IQSizeInfo shape {arr.shape}")
    row, col, frames = (int(arr[0]), int(arr[1]), int(arr[2]))
    return WangIQInfo(row=row, col=col, frames=frames)


def list_wang_files(root: Path | None = None) -> List[Path]:
    """
    List IQDataXXX.dat files under the Wang ULM dataset root, sorted.
    """

    base = wang_data_root() if root is None else Path(root)
    files = sorted(base.glob("IQData*.dat"))
    return [p for p in files if p.is_file()]


def _reshape_polarity(
    buf: np.ndarray, info: WangIQInfo, frames: Optional[Sequence[int]] = None
) -> np.ndarray:
    """
    Reshape a polarity buffer (1-D view) into (frames, row, col) using Fortran order.
    Optionally select a subset of frames to avoid materializing the full volume.
    """

    row, col, total_frames = info.row, info.col, info.frames
    arr = np.reshape(buf, (row, col, total_frames), order="F")
    if frames is None:
        return np.moveaxis(arr, 2, 0)  # -> (frames, row, col)
    # Frame selection
    idx = np.asarray(frames, dtype=int)
    arr_sel = arr[:, :, idx]
    return np.moveaxis(arr_sel, 2, 0)


def load_wang_iq(
    path: Path,
    info: WangIQInfo,
    frames: Optional[Sequence[int]] = None,
    dtype: np.dtype = np.complex64,
) -> np.ndarray:
    """
    Load a single IQDataXXX.dat file and return IQ as a complex array.

    Parameters
    ----------
    path : Path
        Path to the .dat file.
    info : WangIQInfo
        Metadata (row, col, frames).
    frames : optional sequence of ints
        If provided, select a subset of frames; otherwise load all frames.
    dtype : numpy dtype
        Complex dtype to cast to (default: complex64).
    """

    total = 2 * info.n_samples_per_polarity
    raw = np.memmap(path, dtype="<f8", mode="r", shape=(total,))

    n = info.n_samples_per_polarity
    I_raw = raw[:n]
    Q_raw = raw[n:]

    I = _reshape_polarity(I_raw, info, frames=frames)
    Q = _reshape_polarity(Q_raw, info, frames=frames)

    iq = I.astype(np.float32, copy=False) + 1j * Q.astype(np.float32, copy=False)
    return iq.astype(dtype, copy=False)


def iter_wang_frames(
    path: Path,
    info: WangIQInfo,
    chunk_size: int = 32,
) -> Iterable[np.ndarray]:
    """
    Stream IQ frames in chunks to avoid holding the full volume in memory.

    Yields complex arrays shaped (chunk, row, col). The final chunk may be shorter.
    """

    total_frames = info.frames
    for start in range(0, total_frames, chunk_size):
        stop = min(start + chunk_size, total_frames)
        idx = np.arange(start, stop, dtype=int)
        yield load_wang_iq(path, info, frames=idx)


__all__ = [
    "load_wang_metadata",
    "list_wang_files",
    "load_wang_iq",
    "iter_wang_frames",
]
