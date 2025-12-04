from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Tuple

import numpy as np


@dataclass
class MaceScan:
    """
    Whole-brain visually evoked mouse fUS scan (Macé/Urban dataset).

    The PD data are exposed as a 4-D array with shape (T, H, W, Z),
    where T is the number of PD volumes (time), H and W are the in-plane
    spatial dimensions, and Z is the number of coronal planes.
    """

    name: str
    pd: np.ndarray  # (T, H, W, Z), float32
    dt: float  # seconds between PD volumes (fast-100 ms mode -> 0.1)
    voxel_size_um: Tuple[float, float, float]  # (dy, dx, dz) in micrometers
    planes_mm: np.ndarray  # (Z,) plane positions in mm (as provided)

    @property
    def shape(self) -> Tuple[int, int, int, int]:
        return tuple(self.pd.shape)  # type: ignore[return-value]

    @property
    def n_time(self) -> int:
        return int(self.pd.shape[0])

    @property
    def height(self) -> int:
        return int(self.pd.shape[1])

    @property
    def width(self) -> int:
        return int(self.pd.shape[2])

    @property
    def n_planes(self) -> int:
        return int(self.pd.shape[3])


@dataclass
class WangIQInfo:
    """
    Metadata for a Wang ULM IQ volume (single .dat file).

    The raw file stores 2*row*col*frames doubles laid out as [IData(:); QData(:)]
    in MATLAB column-major order, where IQ(t, y, x) has shape (row, col, frames).
    """

    row: int
    col: int
    frames: int

    @property
    def n_samples_per_polarity(self) -> int:
        return int(self.row * self.col * self.frames)


def iter_pd_slices(scan: MaceScan) -> Iterable[Tuple[int, np.ndarray]]:
    """
    Yield per-plane PD volumes shaped (T, H, W) for a given scan.

    Returns (z_index, pd_T_HW) for each coronal plane.
    """

    T, H, W, Z = scan.pd.shape
    for z in range(Z):
        yield z, scan.pd[:, :, :, z]


def tile_iter(
    shape: Tuple[int, int], tile_hw: Tuple[int, int], stride: int
) -> Iterable[Tuple[int, int]]:
    """
    Simple 2D tile iterator (H, W) -> (y0, x0) with given tile size and stride.
    """

    H, W = shape
    th, tw = tile_hw
    for y in range(0, H - th + 1, stride):
        for x in range(0, W - tw + 1, stride):
            yield y, x


def tile_coords(
    shape: Tuple[int, int], tile_hw: Tuple[int, int], stride: int
) -> List[Tuple[int, int]]:
    """
    Convenience wrapper returning all (y0, x0) tile origins.
    """

    return list(tile_iter(shape, tile_hw, stride))


def extract_tile_stack(
    pd_T_HW: np.ndarray,
    y0: int,
    x0: int,
    tile_hw: Tuple[int, int],
) -> np.ndarray:
    """
    Extract a (T, th, tw) PD tile stack from a (T, H, W) PD volume.
    """

    th, tw = tile_hw
    return pd_T_HW[:, y0 : y0 + th, x0 : x0 + tw]


def mace_data_root(base: Path | str = Path("data")) -> Path:
    """
    Return the expected root directory for the Macé/Urban dataset.
    """

    base_path = Path(base)
    return base_path / "whole-brain-fUS"


def wang_data_root(base: Path | str = Path("data")) -> Path:
    """
    Return the expected root directory for the Wang ULM IQ dataset.
    """

    base_path = Path(base)
    return base_path / "wang_ulm"


__all__ = [
    "MaceScan",
    "WangIQInfo",
    "iter_pd_slices",
    "tile_iter",
    "tile_coords",
    "extract_tile_stack",
    "mace_data_root",
    "wang_data_root",
]
