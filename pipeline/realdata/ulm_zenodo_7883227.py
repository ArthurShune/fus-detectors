from __future__ import annotations

import json
import re
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import h5py
import numpy as np


@dataclass(frozen=True)
class Ulm7883227Params:
    frame_rate_hz: float
    prf_per_emission_hz: float
    angles_rad: list[float]


def ulm_zenodo_7883227_root(base: Path | str = Path("data")) -> Path:
    return Path(base) / "ulm_zenodo_7883227"


def ulm_zenodo_7883227_iq_zip_path(root: Path | None = None) -> Path:
    base = ulm_zenodo_7883227_root() if root is None else Path(root)
    return base / "IQ_001_to_025.zip"


def ulm_zenodo_7883227_param_json_path(root: Path | None = None) -> Path:
    base = ulm_zenodo_7883227_root() if root is None else Path(root)
    return base / "param.json"


def load_ulm_zenodo_7883227_params(root: Path | None = None) -> Ulm7883227Params:
    path = ulm_zenodo_7883227_param_json_path(root)
    if not path.is_file():
        raise FileNotFoundError(f"param.json not found at {path}")
    meta = json.loads(path.read_text())
    frame_rate_hz = float(meta["param"]["FrameRate"])
    prf_per_emission_hz = float(meta["param"]["PRF"])
    angles = meta["param"].get("Angles") or []
    angles_rad = [float(a) for a in angles]
    return Ulm7883227Params(
        frame_rate_hz=frame_rate_hz,
        prf_per_emission_hz=prf_per_emission_hz,
        angles_rad=angles_rad,
    )


def list_ulm_blocks(root: Path | None = None) -> list[int]:
    zip_path = ulm_zenodo_7883227_iq_zip_path(root)
    if not zip_path.is_file():
        raise FileNotFoundError(f"IQ zip not found at {zip_path}")
    blocks: list[int] = []
    pat = re.compile(r"^IQ/IQ_(\d{3})\.hdf5$")
    with zipfile.ZipFile(zip_path, "r") as zf:
        for name in zf.namelist():
            m = pat.match(name)
            if not m:
                continue
            blocks.append(int(m.group(1)))
    blocks.sort()
    return blocks


def _block_member_name(block_id: int) -> str:
    if block_id < 1:
        raise ValueError("block_id must be >= 1")
    return f"IQ/IQ_{int(block_id):03d}.hdf5"


def ensure_ulm_block_extracted(
    block_id: int,
    *,
    root: Path | None = None,
    cache_dir: Path | None = None,
) -> Path:
    """
    Ensure IQ_XXX.hdf5 is extracted under cache_dir and return its path.

    Extraction is performed from IQ_001_to_025.zip on demand to avoid unpacking
    the full archive for small experiments.
    """
    cache = Path("tmp/ulm_zenodo_7883227") if cache_dir is None else Path(cache_dir)
    cache.mkdir(parents=True, exist_ok=True)

    out_path = cache / f"IQ_{int(block_id):03d}.hdf5"
    if out_path.is_file():
        return out_path

    zip_path = ulm_zenodo_7883227_iq_zip_path(root)
    if not zip_path.is_file():
        raise FileNotFoundError(f"IQ zip not found at {zip_path}")

    member = _block_member_name(block_id)
    with zipfile.ZipFile(zip_path, "r") as zf:
        names = set(zf.namelist())
        if member not in names:
            raise FileNotFoundError(f"Missing member {member!r} in {zip_path}")
        zf.extract(member, path=cache)

    extracted = cache / member  # cache/IQ/IQ_XXX.hdf5
    if not extracted.is_file():
        raise FileNotFoundError(f"Extraction failed; expected {extracted}")
    extracted.replace(out_path)
    try:
        (cache / "IQ").rmdir()
    except OSError:
        pass
    return out_path


def load_ulm_block_iq(
    block_id: int,
    *,
    frames: slice | Sequence[int] | None = None,
    root: Path | None = None,
    cache_dir: Path | None = None,
    dtype: np.dtype = np.complex64,
) -> np.ndarray:
    """
    Load one ULM Zenodo 7883227 block and return IQ as (T, H, W) complex.

    The stored HDF5 dataset `iq` has shape (2, H, W, T) float32 where the first
    dimension is (I, Q). This converts to complex and transposes to (T, H, W).
    """
    h5_path = ensure_ulm_block_extracted(block_id, root=root, cache_dir=cache_dir)
    with h5py.File(h5_path, "r") as f:
        if "iq" not in f:
            raise KeyError(f"Expected dataset 'iq' in {h5_path}")
        d = f["iq"]
        if d.ndim != 4 or int(d.shape[0]) != 2:
            raise ValueError(f"Unexpected iq shape {tuple(d.shape)} in {h5_path}")
        frame_sel = slice(None) if frames is None else frames
        I = np.asarray(d[0, :, :, frame_sel], dtype=np.float32)
        Q = np.asarray(d[1, :, :, frame_sel], dtype=np.float32)
    iq_hw_t = I + 1j * Q
    iq_t_hw = np.transpose(iq_hw_t, (2, 0, 1))
    return iq_t_hw.astype(dtype, copy=False)


__all__ = [
    "Ulm7883227Params",
    "ulm_zenodo_7883227_root",
    "ulm_zenodo_7883227_iq_zip_path",
    "ulm_zenodo_7883227_param_json_path",
    "load_ulm_zenodo_7883227_params",
    "list_ulm_blocks",
    "ensure_ulm_block_extracted",
    "load_ulm_block_iq",
]

