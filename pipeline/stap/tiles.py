# pipeline/stap/tiles.py
from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass

import numpy as np

Array = np.ndarray


def hann1d(L: int, dtype=np.float32) -> Array:
    """
    1D Hann window of length L in [0,1], peak at 1.
    """
    if L <= 1:
        return np.ones((L,), dtype=dtype)
    w = np.hanning(L).astype(dtype)
    # np.hanning peaks at 1 but touches 0 at the ends; enforce small positive floor
    w /= w.max() + 1e-12
    min_weight = np.array(1e-3, dtype=dtype)
    w = np.maximum(w, min_weight)
    return w


def hann2d(h: int, w: int, dtype=np.float32) -> Array:
    """
    Separable 2D Hann window (outer product), max 1.
    """
    wy = hann1d(h, dtype=dtype)
    wx = hann1d(w, dtype=dtype)
    return (wy[:, None] * wx[None, :]).astype(dtype)


def _grid_1d(L: int, win: int, stride: int) -> list[int]:
    """
    Start indices for tiling along one dimension, ensuring last tile hits the end.
    """
    if win > L:
        raise ValueError(f"Window {win} > length {L}")
    starts = list(range(0, max(1, L - win + 1), max(1, stride)))
    if starts[-1] != L - win:
        starts.append(L - win)
    return starts


@dataclass(frozen=True)
class TileSpec:
    y0: int
    y1: int
    x0: int
    x1: int
    weight: Array  # (tile_h, tile_w), Hann weights


def make_tile_specs(
    H: int, W: int, tile_h: int, tile_w: int, stride: int, dtype=np.float32
) -> list[TileSpec]:
    """
    Prepare tile coordinates + Hann weights for overlap-add stitching.
    """
    ys = _grid_1d(H, tile_h, stride)
    xs = _grid_1d(W, tile_w, stride)
    w2d = hann2d(tile_h, tile_w, dtype=dtype)
    specs: list[TileSpec] = []
    for y0 in ys:
        y1 = y0 + tile_h
        for x0 in xs:
            x1 = x0 + tile_w
            specs.append(TileSpec(y0=y0, y1=y1, x0=x0, x1=x1, weight=w2d.copy()))
    return specs


def extract_tiles_2d(img_hw: Array, specs: list[TileSpec]) -> list[Array]:
    """
    Extract 2D tiles (no copy guarantees not enforced).
    """
    H, W = img_hw.shape
    tiles: list[Array] = []
    for sp in specs:
        assert 0 <= sp.y0 < sp.y1 <= H and 0 <= sp.x0 < sp.x1 <= W
        tiles.append(img_hw[sp.y0 : sp.y1, sp.x0 : sp.x1])
    return tiles


def extract_tiles_3d(arr_t_hw: Array, specs: list[TileSpec]) -> list[Array]:
    """
    Extract 3D tiles from (T,H,W) arrays: returns a list of (T, tile_h, tile_w).
    """
    T, H, W = arr_t_hw.shape
    tiles: list[Array] = []
    for sp in specs:
        tiles.append(arr_t_hw[:, sp.y0 : sp.y1, sp.x0 : sp.x1])
    return tiles


def overlap_add(
    tiles: Iterable[Array], specs: list[TileSpec], H: int, W: int, dtype=np.float32
) -> Array:
    """
    Stitch per-tile 2D arrays using Hann-weighted overlap-add, normalized by
    the sum of weights. If weights vanish anywhere (shouldn't), falls back to 1.

    Parameters
    ----------
    tiles : iterable of (tile_h, tile_w) arrays
        The per-tile maps to be stitched (e.g., PD snippets).
    specs : list of TileSpec
        Must align 1:1 with `tiles`.
    H, W : int
        Output image size.
    dtype : numpy dtype
        Accumulator dtype.

    Returns
    -------
    out : (H, W) array
    """
    acc = np.zeros((H, W), dtype=dtype)
    wsum = np.zeros((H, W), dtype=dtype)
    for tile, sp in zip(tiles, specs, strict=False):
        assert tile.shape == sp.weight.shape, f"Tile {tile.shape} != weight {sp.weight.shape}"
        acc[sp.y0 : sp.y1, sp.x0 : sp.x1] += tile.astype(dtype) * sp.weight
        wsum[sp.y0 : sp.y1, sp.x0 : sp.x1] += sp.weight
    wsum_safe = np.where(wsum > 0, wsum, 1.0)
    return (acc / wsum_safe).astype(dtype)
