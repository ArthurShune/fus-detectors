from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Tuple, Dict, Any

import h5py
import numpy as np
from scipy.signal import hilbert


# ----------------------------------------------------------------------
# Paths / helpers
# ----------------------------------------------------------------------


def picmus_data_root() -> Path:
    """
    Root directory for PICMUS data.
    Assumes you put PICMUS_carotid_cross.uff in data/picmus.
    """
    return Path("data") / "picmus"


@dataclass
class PicmusCarotidMeta:
    fs: float
    prf: float
    c: float
    pitch: float
    geometry_x: np.ndarray  # (n_el,)
    n_samples: int
    n_tx: int
    n_el: int


# ----------------------------------------------------------------------
# Dataclasses
# ----------------------------------------------------------------------


@dataclass
class PicmusCarotidRF:
    """
    Minimal RF / IQ cube for PICMUS carotid-cross dataset.

    rf: complex or real RF data shaped (T, S, C):
        T: slow-time index (emissions / frames)
        S: fast-time samples
        C: channels / elements
    fs: sampling frequency [Hz] (fast-time)
    prf: pulse repetition frequency [Hz] (slow-time), if available, else None
    center_freq: center frequency [Hz], if available, else None
    meta: raw metadata dict for debugging / extension
    """
    name: str
    rf: np.ndarray
    fs: Optional[float]
    prf: Optional[float]
    center_freq: Optional[float]
    meta: Dict[str, Any]


# ----------------------------------------------------------------------
# Internal utilities
# ----------------------------------------------------------------------


def _find_channel_group(f: h5py.File) -> h5py.Group:
    """
    Try to locate the uff.channel_data group in the UFF file.

    Strategy:
      1. Prefer a group with attribute 'class' ending in 'channel_data'.
      2. Fallback to a top-level group literally named 'channel_data'.
      3. Fallback to the only group if there is exactly one.
    """
    # 1. Look for attr 'class' ~ 'uff.channel_data'
    for name, obj in f.items():
        if isinstance(obj, h5py.Group):
            cl = obj.attrs.get("class", None)
            if isinstance(cl, bytes):
                cl = cl.decode("utf-8", errors="ignore")
            if isinstance(cl, str) and cl.endswith("channel_data"):
                return obj

    # 2. Fallback: group literally called 'channel_data'
    if "channel_data" in f and isinstance(f["channel_data"], h5py.Group):
        return f["channel_data"]

    # 3. Fallback: single group in file
    groups = [obj for obj in f.values() if isinstance(obj, h5py.Group)]
    if len(groups) == 1:
        return groups[0]

    raise RuntimeError("Could not locate a 'channel_data' group in PICMUS UFF file.")


def _find_rf_dataset(g: h5py.Group) -> h5py.Dataset:
    """
    Inside the channel_data group, pick the largest numeric dataset
    as the RF / IQ array.

    This is heuristic, but works for typical UFF layouts.
    """
    candidates = []
    for name, obj in g.items():
        if isinstance(obj, h5py.Dataset):
            if obj.dtype.kind.lower() in ("f", "c", "i"):  # float / complex / int
                candidates.append((name, obj))

    if not candidates:
        raise RuntimeError("No numeric datasets found in channel_data group.")

    # Choose the dataset with the largest number of elements
    name, ds = max(candidates, key=lambda kv: kv[1].size)
    return ds


def _extract_metadata_from_group(g: h5py.Group) -> Dict[str, Any]:
    """
    Pull out a few likely metadata fields from the channel_data group.

    UFF / USTB usually stores properties as attributes or small datasets.
    We keep everything in a dict so you can inspect later.
    """
    meta: Dict[str, Any] = {}

    # 1. Group attributes
    for k, v in g.attrs.items():
        meta[f"attr:{k}"] = v

    # 2. Try to capture some known-ish fields if present
    for key in ("sampling_frequency", "fs", "center_frequency", "fc", "prf", "PRF"):
        if key in g:
            ds = g[key]
            if isinstance(ds, h5py.Dataset) and ds.size == 1:
                meta[key] = float(ds[()])

    # 3. Also look under 'probe' and 'sequence' if they exist
    for sub in ("probe", "sequence"):
        if sub in g and isinstance(g[sub], h5py.Group):
            sg = g[sub]
            for k, v in sg.attrs.items():
                meta[f"{sub}.attr:{k}"] = v
            for name, ds in sg.items():
                if isinstance(ds, h5py.Dataset) and ds.size == 1:
                    meta[f"{sub}.{name}"] = float(ds[()])

    return meta


def _get_scalar(meta: Dict[str, Any], keys: Tuple[str, ...]) -> Optional[float]:
    """
    Try a list of keys in the meta dict, return the first scalar value found.
    """
    for k in keys:
        if k in meta:
            try:
                return float(meta[k])
            except Exception:
                continue
    return None


# ----------------------------------------------------------------------
# Public loader
# ----------------------------------------------------------------------


def load_picmus_carotid_uff(
    path: Optional[str | Path] = None,
    name: str = "PICMUS_carotid_cross",
) -> PicmusCarotidRF:
    """
    Load the PICMUS carotid-cross dataset from a UFF (HDF5) file.

    By default, expects PICMUS_carotid_cross.uff under data/picmus.

    Returns:
        PicmusCarotidRF with rf shaped (T, S, C).
    """
    if path is None:
        path = picmus_data_root() / "PICMUS_carotid_cross.uff"
    path = Path(path)

    if not path.exists():
        raise FileNotFoundError(f"UFF file not found at {path}")

    with h5py.File(path, "r") as f:
        ch_grp = _find_channel_group(f)
        rf_ds = _find_rf_dataset(ch_grp)

        meta = _extract_metadata_from_group(ch_grp)

        raw = rf_ds[()]  # load into memory, NumPy array

        # Heuristic interpretation: [samples, channels, frames]
        # This is the conventional USTB channel_data layout.
        if raw.ndim != 3:
            raise RuntimeError(
                f"Expected RF dataset with ndims=3, got shape {raw.shape} "
                f"in dataset {rf_ds.name}"
            )

        n_samp, n_chan, n_frames = raw.shape

        # Reorder to (T, S, C) = (frames, samples, channels)
        rf = np.moveaxis(raw, 2, 0)

        # Cast to complex64 if it looks complex, else float32
        if np.iscomplexobj(rf):
            rf = rf.astype(np.complex64, copy=False)
        else:
            rf = rf.astype(np.float32, copy=False)

    fs = _get_scalar(meta, ("sampling_frequency", "fs"))
    prf = _get_scalar(meta, ("prf", "PRF", "sequence.PRF"))
    fc = _get_scalar(meta, ("center_frequency", "fc", "probe.center_frequency"))

    return PicmusCarotidRF(
        name=name,
        rf=rf,
        fs=fs,
        prf=prf,
        center_freq=fc,
        meta=meta,
    )


# ----------------------------------------------------------------------
# DAS beamforming helpers
# ----------------------------------------------------------------------


def load_picmus_rf(
    path: Optional[str | Path] = None,
) -> Tuple[np.ndarray, PicmusCarotidMeta]:
    """
    Load RF from PICMUS_carotid_cross.uff and return rf with shape (n_samples, n_tx, n_el).
    """
    if path is None:
        path = picmus_data_root() / "PICMUS_carotid_cross.uff"
    path = Path(path)
    with h5py.File(path, "r") as f:
        g = _find_channel_group(f)
        data = g["data"][()]  # expected shape (samples, elements, tx)
        if data.ndim != 3:
            raise RuntimeError(f"Expected RF data ndim=3, got {data.shape}")
        rf = np.asarray(data, dtype=np.float32)
        # If axes are (samples, elements, tx), reorder to (samples, tx, elements)
        if rf.shape[1] == 128 and rf.shape[2] == 1536:
            rf = np.transpose(rf, (0, 2, 1))
        n_samples, n_tx, n_el = rf.shape

        fs = float(g["sampling_frequency"][()])
        prf = float(g["PRF"][()]) if "PRF" in g else 100.0
        # Speed of sound not stored; assume soft tissue
        c = 1540.0
        pitch = float(g["probe/pitch"][()]) if "probe/pitch" in g else 3.0e-4
        geometry_x = None
        if "probe" in g and "geometry" in g["probe"]:
            geom = g["probe/geometry"][()]
            if geom.shape[0] >= 2:
                geometry_x = np.asarray(geom[0], dtype=np.float32)
        if geometry_x is None:
            geometry_x = (np.arange(n_el, dtype=np.float32) - 0.5 * (n_el - 1)) * pitch

    meta = PicmusCarotidMeta(
        fs=fs,
        prf=prf,
        c=c,
        pitch=pitch,
        geometry_x=geometry_x,
        n_samples=n_samples,
        n_tx=n_tx,
        n_el=n_el,
    )
    return rf, meta


def make_picmus_grid(
    meta: PicmusCarotidMeta,
    z_min_m: float = 1e-3,
    z_max_m: float = 6e-3,
    n_z: int = 128,
    n_x: int = 64,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Build a simple Cartesian imaging grid (z, x) in meters.
    The depth range is small because the RF has only ~75 samples.
    """
    z_grid = np.linspace(z_min_m, z_max_m, n_z, dtype=np.float32)
    x_span = (meta.geometry_x.max() - meta.geometry_x.min())
    x_grid = np.linspace(-0.5 * x_span, 0.5 * x_span, n_x, dtype=np.float32)
    return z_grid, x_grid


def precompute_das_delays(
    meta: PicmusCarotidMeta,
    z_grid: np.ndarray,
    x_grid: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Precompute fractional sample indices and valid mask for DAS with normal-incidence TX.
    """
    c = meta.c
    fs = meta.fs
    n_samples = meta.n_samples
    x_el = meta.geometry_x.astype(np.float32)
    n_el = meta.n_el

    z = z_grid[:, None, None]
    x = x_grid[None, :, None]
    x_e = x_el[None, None, :]

    d_tx = z
    d_rx = np.sqrt((x - x_e) ** 2 + z**2)
    tof = (d_tx + d_rx) / c
    sample_idx = tof * fs
    i0 = np.floor(sample_idx).astype(np.int32)
    valid = (i0 >= 0) & (i0 < (n_samples - 1))
    return sample_idx.astype(np.float32), valid


def das_beamform_picmus(
    rf: np.ndarray,
    meta: PicmusCarotidMeta,
    z_grid: np.ndarray,
    x_grid: np.ndarray,
    sample_idx: np.ndarray,
    valid_mask: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Simple delay-and-sum beamformer for PICMUS carotid RF.
    Returns IQ cube (n_tx, n_z, n_x) and PD cube.
    """
    n_samples, n_tx, n_el = rf.shape
    n_z = z_grid.shape[0]
    n_x = x_grid.shape[0]

    idx_f = sample_idx  # (n_z, n_x, n_el)
    idx0 = np.floor(idx_f).astype(np.int32)
    w = idx_f - idx0
    i0_clipped = np.clip(idx0, 0, n_samples - 2)
    i1_clipped = i0_clipped + 1
    valid = valid_mask

    img_rf = np.zeros((n_tx, n_z, n_x), dtype=np.float32)

    for itx in range(n_tx):
        rf_slice = rf[:, itx, :]  # (n_samples, n_el)
        # Gather s0/s1
        s0 = rf_slice[i0_clipped, np.arange(n_el)]  # (n_z, n_x, n_el)
        s1 = rf_slice[i1_clipped, np.arange(n_el)]
        contrib = (1.0 - w) * s0 + w * s1
        contrib = np.where(valid, contrib, 0.0)
        img_rf[itx] = contrib.sum(axis=2)

    iq_cube = hilbert(img_rf, axis=1).astype(np.complex64, copy=False)
    pd_cube = (np.abs(iq_cube) ** 2).astype(np.float32, copy=False)
    return iq_cube, pd_cube
# ----------------------------------------------------------------------
# Convenience helpers
# ----------------------------------------------------------------------


def rf_to_pd_cube(rf: np.ndarray) -> np.ndarray:
    """
    Form a simple per-emission power cube PD(T, H, W) from RF/ IQ data.

    This treats the RF matrix for each emission (samples x channels) as a
    2-D frame. No beamforming is applied; power is computed as |rf|^2.

    Parameters
    ----------
    rf : np.ndarray
        RF data shaped (T, S, C) or (T, H, W).

    Returns
    -------
    pd : np.ndarray
        Power cube shaped (T, H, W), float32.
    """

    if rf.ndim != 3:
        raise ValueError(f"rf must be 3-D (T, H, W), got shape {rf.shape}")
    pd = (np.abs(rf) ** 2).astype(np.float32, copy=False)
    return pd


def rf_to_hilbert_iq(rf: np.ndarray) -> np.ndarray:
    """
    Form an analytic (Hilbert) IQ cube from real RF.

    Applies the Hilbert transform along the fast-time (samples) axis for each
    emission and channel, yielding a complex IQ cube shaped (T, H, W) where
    H corresponds to samples/depth and W to channels/lateral.
    """

    if rf.ndim != 3:
        raise ValueError(f"rf must be 3-D (T, H, W), got shape {rf.shape}")
    if np.iscomplexobj(rf):
        return rf.astype(np.complex64, copy=False)
    try:
        from scipy.signal import hilbert
    except Exception as e:
        raise RuntimeError("scipy is required for Hilbert IQ conversion") from e

    # Apply Hilbert along samples axis (axis=1): shape (T, H, W)
    iq = hilbert(rf, axis=1)
    return iq.astype(np.complex64, copy=False)
