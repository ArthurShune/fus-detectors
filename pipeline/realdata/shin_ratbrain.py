from __future__ import annotations

from pathlib import Path
from typing import Iterable, List, Optional, Sequence

import numpy as np

from pipeline.realdata import ShinIQInfo, shin_ratbrain_data_root


def load_shin_metadata(root: Path | None = None) -> ShinIQInfo:
    """
    Load SizeInfo.dat and return basic dimensions (row, col, frames).

    The Shin RatBrain Fig3 dataset encodes SizeInfo.dat as 3 little-endian
    int32 values: (row, col, frames).
    """

    base = shin_ratbrain_data_root() if root is None else Path(root)
    info_path = base / "SizeInfo.dat"
    if not info_path.is_file():
        raise FileNotFoundError(f"SizeInfo.dat not found under {base}")
    dims = np.fromfile(info_path, dtype="<i4", count=3)
    if dims.size != 3:
        raise ValueError(f"Expected 3 int32 in {info_path}, got {dims.size}")
    row, col, frames = (int(dims[0]), int(dims[1]), int(dims[2]))
    return ShinIQInfo(row=row, col=col, frames=frames)


def list_shin_files(root: Path | None = None) -> List[Path]:
    """
    List IQData*.dat files under the Shin RatBrain dataset root, sorted.
    """

    base = shin_ratbrain_data_root() if root is None else Path(root)
    files = sorted(base.glob("IQData*.dat"))
    return [p for p in files if p.is_file()]


def _reshape_polarity(
    buf: np.ndarray, info: ShinIQInfo, frames: Optional[Sequence[int]] = None
) -> np.ndarray:
    """
    Reshape a polarity buffer (1-D view) into (frames, row, col) using Fortran order.
    Optionally select a subset of frames to avoid materializing the full volume.
    """

    row, col, total_frames = info.row, info.col, info.frames
    arr = np.reshape(buf, (row, col, total_frames), order="F")
    if frames is None:
        return np.moveaxis(arr, 2, 0)  # -> (frames, row, col)
    idx = np.asarray(frames, dtype=int)
    arr_sel = arr[:, :, idx]
    return np.moveaxis(arr_sel, 2, 0)


def load_shin_iq(
    path: Path,
    info: ShinIQInfo,
    frames: Optional[Sequence[int]] = None,
    dtype: np.dtype = np.complex64,
    *,
    validate_size: bool = True,
) -> np.ndarray:
    """
    Load a single IQData*.dat file and return IQ as a complex array.

    File format matches LoadIQ.m:
      - raw = fread(f, 'double')  -> float64
      - first half is I, second half is Q
      - both are laid out in MATLAB column-major order (row, col, frames)

    Parameters
    ----------
    path : Path
        Path to IQData*.dat.
    info : ShinIQInfo
        Metadata (row, col, frames).
    frames : optional sequence of ints
        If provided, select a subset of frames; otherwise load all frames.
    dtype : numpy dtype
        Complex dtype to cast to (default: complex64).
    validate_size : bool
        If true, validate file size matches the expected float64 length.
    """

    total = 2 * info.n_samples_per_polarity
    if validate_size:
        expected_bytes = int(total * 8)
        actual_bytes = int(path.stat().st_size)
        if actual_bytes != expected_bytes:
            raise ValueError(
                f"Unexpected file size for {path.name}: {actual_bytes} bytes, "
                f"expected {expected_bytes} bytes for (row,col,frames)=({info.row},{info.col},{info.frames})."
            )

    raw = np.memmap(path, dtype="<f8", mode="r", shape=(total,))
    n = info.n_samples_per_polarity
    I_raw = raw[:n]
    Q_raw = raw[n:]

    I = _reshape_polarity(I_raw, info, frames=frames)
    Q = _reshape_polarity(Q_raw, info, frames=frames)

    iq = I.astype(np.float32, copy=False) + 1j * Q.astype(np.float32, copy=False)
    return iq.astype(dtype, copy=False)


def iter_shin_frames(
    path: Path,
    info: ShinIQInfo,
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
        yield load_shin_iq(path, info, frames=idx)


__all__ = [
    "load_shin_metadata",
    "list_shin_files",
    "load_shin_iq",
    "iter_shin_frames",
]

