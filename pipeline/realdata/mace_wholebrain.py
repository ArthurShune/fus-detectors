from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Mapping, Tuple

import numpy as np
import scipy.io as sio

from pipeline.realdata import MaceScan, mace_data_root


@dataclass
class MaceAtlas:
    """
    Allen CCFv3 atlas bundle for the Macé/Urban dataset.
    """

    histology: np.ndarray  # (Ha, Wa, Za), uint8
    regions: np.ndarray  # (Ha, Wa, Za), int16
    vascular: np.ndarray  # (Ha, Wa, Za), float32
    voxel_size_um: Tuple[float, float, float]
    direction: str
    raw_struct: object


@dataclass
class MaceTransform:
    """
    Affine transform mapping scan volumes into atlas coordinates.
    """

    M: np.ndarray  # (4, 4)
    size: Tuple[int, int, int]
    voxel_size_um: Tuple[float, float, float]
    scale: Tuple[float, float, float]
    raw_struct: object


@dataclass
class MaceRegionInfo:
    """
    Mapping between Allen atlas region acronyms and indices.
    """

    acronyms: List[str]
    names: List[str]
    volumes_mm3: np.ndarray
    label_for_acr: Mapping[str, int]


def _load_scan_struct(path: Path) -> Dict[str, object]:
    mat = sio.loadmat(path, squeeze_me=True, struct_as_record=False)
    if "scanfus" not in mat:
        raise KeyError(f"Expected 'scanfus' in {path.name}, found keys={list(mat.keys())}")
    scanfus = mat["scanfus"]
    data = np.asarray(scanfus.Data)
    if data.ndim != 4:
        raise ValueError(f"Expected 4-D Data in {path.name}, got shape {data.shape}")
    voxel_size = np.asarray(scanfus.VoxelSize, dtype=np.float64)
    planes = np.asarray(scanfus.Planes, dtype=np.float64)
    direction = str(scanfus.Direction)
    return {
        "data_raw": data,
        "voxel_size": voxel_size,
        "planes": planes,
        "direction": direction,
    }


def _reorder_to_T_H_W_Z(data_raw: np.ndarray) -> np.ndarray:
    """
    Reorder (DV, RL, PA, TM) -> (TM, DV, RL, PA).
    """

    if data_raw.ndim != 4:
        raise ValueError(f"Expected 4-D array, got shape {data_raw.shape}")
    dv, rl, pa, tm = data_raw.shape
    if pa <= 1 or tm <= 1:
        # The dataset should have 20 planes and 70 time frames per the Zenodo
        # description; we keep this guard mostly to catch accidental misuse.
        raise ValueError(f"Unexpected Macé volume shape {data_raw.shape}")
    # Move time axis to front: (DV, RL, PA, TM) -> (TM, DV, RL, PA)
    data_T_H_W_Z = np.moveaxis(data_raw, 3, 0)
    return data_T_H_W_Z.astype(np.float32, copy=False)


def load_mace_scan(scan_path: Path) -> MaceScan:
    """
    Load a single `scan*.mat` file into a `MaceScan`.
    """

    info = _load_scan_struct(scan_path)
    pd_T_H_W_Z = _reorder_to_T_H_W_Z(info["data_raw"])  # type: ignore[arg-type]
    voxel_size = np.asarray(info["voxel_size"], dtype=np.float64)
    if voxel_size.shape != (3,):
        raise ValueError(f"Unexpected voxel_size shape {voxel_size.shape} in {scan_path.name}")
    dy, dx, dz = voxel_size.tolist()
    planes = np.asarray(info["planes"], dtype=np.float64)
    dt = 0.1  # fast-100 ms mode -> 10 Hz PD volumes (per dataset description)
    return MaceScan(
        name=scan_path.stem,
        pd=pd_T_H_W_Z,
        dt=float(dt),
        voxel_size_um=(float(dy), float(dx), float(dz)),
        planes_mm=planes,
    )


def find_mace_scans(root: Path | None = None) -> List[Path]:
    """
    Enumerate all `scan*.mat` files under the Macé dataset root.
    """

    base = mace_data_root() if root is None else root
    if not base.is_dir():
        raise FileNotFoundError(f"Macé dataset root not found: {base}")
    scans: List[Path] = []
    for p in base.glob("scan*.mat"):
        if not p.is_file() or p.name.endswith(".Zone.Identifier"):
            continue
        stem = p.stem
        # Keep only numbered functional scans (scan1.mat, ..., scan6.mat).
        if stem.startswith("scan") and stem[4:].isdigit():
            scans.append(p)
    scans = sorted(scans)
    if not scans:
        raise FileNotFoundError(f"No scan*.mat files found under {base}")
    return scans


def load_all_mace_scans(root: Path | None = None) -> List[MaceScan]:
    """
    Load all Macé whole-brain scans under the dataset root.
    """

    scans: List[MaceScan] = []
    for path in find_mace_scans(root):
        scans.append(load_mace_scan(path))
    return scans


def load_mace_atlas(root: Path | None = None) -> MaceAtlas:
    """
    Load the Allen CCFv3 atlas bundle associated with the Macé dataset.
    """

    base = mace_data_root() if root is None else root
    mat = sio.loadmat(base / "allen_brain_atlas.mat", squeeze_me=True, struct_as_record=False)
    if "atlas" not in mat:
        raise KeyError(
            f"'atlas' struct not found in allen_brain_atlas.mat (keys={list(mat.keys())})"
        )
    atlas_struct = mat["atlas"]
    histology = np.asarray(atlas_struct.Histology)
    regions = np.asarray(atlas_struct.Regions)
    vascular = np.asarray(atlas_struct.Vascular)
    voxel_size = np.asarray(atlas_struct.VoxelSize, dtype=np.float64)
    direction = str(atlas_struct.Direction)
    if voxel_size.shape != (3,):
        raise ValueError(f"Unexpected atlas VoxelSize shape {voxel_size.shape}")
    vy, vx, vz = voxel_size.tolist()
    return MaceAtlas(
        histology=histology,
        regions=regions,
        vascular=vascular,
        voxel_size_um=(float(vy), float(vx), float(vz)),
        direction=direction,
        raw_struct=atlas_struct,
    )


def load_mace_region_info(root: Path | None = None) -> MaceRegionInfo:
    """
    Load region metadata (acronym -> index mapping) from the Allen atlas.

    The atlas `Regions` volume uses integer labels 1..509 to refer to entries
    in `infoRegions.acr` / `infoRegions.name`. Index 0 is background.
    """

    base = mace_data_root() if root is None else root
    mat = sio.loadmat(base / "allen_brain_atlas.mat", squeeze_me=True, struct_as_record=False)
    if "atlas" not in mat:
        raise KeyError(
            f"'atlas' struct not found in allen_brain_atlas.mat (keys={list(mat.keys())})"
        )
    atlas_struct = mat["atlas"]
    info = atlas_struct.infoRegions
    acr = np.asarray(info.acr, dtype=object)
    name = np.asarray(info.name, dtype=object)
    vol = np.asarray(info.vol, dtype=np.float64)
    if acr.shape[0] != vol.shape[0]:
        raise ValueError(
            f"infoRegions.acr and vol length mismatch: {acr.shape[0]} vs {vol.shape[0]}"
        )
    acr_list = [str(a) for a in acr]
    name_list = [str(n) for n in name]
    # Labels in Regions are 1-based; index 0 is reserved for background.
    label_for_acr: Dict[str, int] = {}
    for idx, a in enumerate(acr_list):
        label_for_acr[a] = idx  # idx corresponds to the value stored in Regions
    return MaceRegionInfo(
        acronyms=acr_list,
        names=name_list,
        volumes_mm3=vol,
        label_for_acr=label_for_acr,
    )


def load_mace_transform(root: Path | None = None) -> MaceTransform:
    """
    Load the affine transform mapping scan volumes into the atlas space.
    """

    base = mace_data_root() if root is None else root
    mat = sio.loadmat(base / "Transformation.mat", squeeze_me=True, struct_as_record=False)
    if "Transf" not in mat:
        raise KeyError(
            f"'Transf' struct not found in Transformation.mat (keys={list(mat.keys())})"
        )
    transf_struct = mat["Transf"]
    M = np.asarray(transf_struct.M, dtype=np.float64)
    size = np.asarray(transf_struct.size, dtype=np.int32)
    voxel_size = np.asarray(transf_struct.VoxelSize, dtype=np.float64)
    scale = np.asarray(transf_struct.scale, dtype=np.float64)
    if M.shape != (4, 4):
        raise ValueError(f"Unexpected transform matrix shape {M.shape}")
    if size.shape != (3,):
        raise ValueError(f"Unexpected transform size shape {size.shape}")
    if voxel_size.shape != (3,):
        raise ValueError(f"Unexpected transform voxel size shape {voxel_size.shape}")
    if scale.shape != (3,):
        raise ValueError(f"Unexpected transform scale shape {scale.shape}")
    return MaceTransform(
        M=M,
        size=(int(size[0]), int(size[1]), int(size[2])),
        voxel_size_um=(float(voxel_size[0]), float(voxel_size[1]), float(voxel_size[2])),
        scale=(float(scale[0]), float(scale[1]), float(scale[2])),
        raw_struct=transf_struct,
    )


def build_mace_transform_matrix(transf: MaceTransform) -> Tuple[np.ndarray, np.ndarray]:
    """
    Return (A, t) such that atlas_xyz = A @ scan_xyz + t.

    We follow the convention that scan_xyz = [pa_idx, dv_idx, rl_idx]^T, which
    matches the empirical in-bounds mapping between scan and atlas indices.
    """

    M = np.asarray(transf.M, dtype=np.float64)
    M_T = M.T
    A = M_T[:3, :3]
    t = M_T[:3, 3]
    return A, t


def scan_plane_to_atlas_indices(
    H: int,
    W: int,
    plane_idx: int,
    A: np.ndarray,
    t: np.ndarray,
    atlas_shape: Tuple[int, int, int],
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Map a single scan plane (H, W, plane_idx) into atlas indices.

    Parameters
    ----------
    H, W : int
        Scan plane height and width (DV, RL).
    plane_idx : int
        Plane index along the PA axis (0-based).
    A, t : np.ndarray
        Affine transform such that atlas_xyz = A @ scan_xyz + t.
    atlas_shape : (Ha, Wa, Za)
        Shape of the atlas Regions volume (DV, AP, LR).

    Returns
    -------
    i_dv, i_ap, i_lr : np.ndarray
        Integer atlas indices for each (y, x) position, flattened.
    inside : np.ndarray (bool)
        Mask indicating which pixels map inside the atlas bounds.
    """

    Ha, Wa, Za = atlas_shape
    ys, xs = np.meshgrid(
        np.arange(H, dtype=np.float64), np.arange(W, dtype=np.float64), indexing="ij"
    )
    dv = ys.ravel()
    rl = xs.ravel()
    pa = np.full_like(dv, float(plane_idx))

    scan_xyz = np.stack([pa, dv, rl], axis=1)
    atlas_xyz = scan_xyz @ A.T + t

    dv_a = atlas_xyz[:, 0]
    ap_a = atlas_xyz[:, 1]
    lr_a = atlas_xyz[:, 2]

    i_dv = np.rint(dv_a).astype(np.int64)
    i_ap = np.rint(ap_a).astype(np.int64)
    i_lr = np.rint(lr_a).astype(np.int64)

    inside = (i_dv >= 0) & (i_dv < Ha) & (i_ap >= 0) & (i_ap < Wa) & (i_lr >= 0) & (i_lr < Za)
    return i_dv, i_ap, i_lr, inside


__all__ = [
    "MaceAtlas",
    "MaceTransform",
    "MaceRegionInfo",
    "load_mace_scan",
    "find_mace_scans",
    "load_all_mace_scans",
    "load_mace_atlas",
    "load_mace_region_info",
    "load_mace_transform",
    "build_mace_transform_matrix",
    "scan_plane_to_atlas_indices",
]
