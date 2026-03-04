"""Replay source-loading and window-spec helpers for bundle generation."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import numpy as np

from sim.kwave.common import AngleData, SimGeom


@dataclass(frozen=True)
class WindowSpec:
    offset: int | None
    length: int | None
    suffix: str | None

    @property
    def label(self) -> str:
        return self.suffix or (
            f"offset{self.offset}_len{self.length}" if self.length is not None else "full"
        )


def _load_dir(path: Path, ang: float) -> AngleData:
    if not path.exists():
        raise FileNotFoundError(f"Expected {path} with rf.npy/dt.npy")
    rf = np.load(path / "rf.npy")
    dt = float(np.load(path / "dt.npy"))
    return AngleData(angle_deg=float(ang), rf=rf, dt=dt)


def _load_angle_data(src_root: Path, angles: Sequence[float]) -> list[list[AngleData]]:
    """Load one or more ensembles of angle data from the source directory."""
    angle_data: list[AngleData] = []
    simple_ok = True
    for ang in angles:
        name = f"angle_{int(round(ang))}"
        d = src_root / name
        if not d.exists():
            simple_ok = False
            break
        angle_data.append(_load_dir(d, ang))
    if simple_ok:
        return [angle_data]

    ensemble_dirs = sorted(
        {
            p.name.split("_")[0]
            for p in src_root.iterdir()
            if p.is_dir() and p.name.startswith("ens")
        }
    )
    if not ensemble_dirs:
        missing = ", ".join(f"angle_{int(round(a))}" for a in angles)
        raise FileNotFoundError(
            f"Expected angle directories {missing} under {src_root}, none found."
        )

    angle_sets: list[list[AngleData]] = []
    for ens in ensemble_dirs:
        ens_set: list[AngleData] = []
        for ang in angles:
            name = f"{ens}_angle_{int(round(ang))}"
            d = src_root / name
            ens_set.append(_load_dir(d, ang))
        angle_sets.append(ens_set)
    return angle_sets


def _load_meta(src_root: Path) -> dict:
    meta_path = src_root / "meta.json"
    if not meta_path.exists():
        raise FileNotFoundError(f"missing required metadata file: {meta_path}")
    return json.loads(meta_path.read_text())


def _build_geom(meta: dict) -> SimGeom:
    geom_meta = meta.get("geometry") or meta.get("sim_geom")
    if geom_meta is None:
        raise KeyError("meta.json missing 'geometry' or 'sim_geom'")
    return SimGeom(
        Nx=int(geom_meta["Nx"]),
        Ny=int(geom_meta["Ny"]),
        dx=float(geom_meta["dx"]),
        dy=float(geom_meta["dy"]),
        c0=float(geom_meta["c0"]),
        rho0=float(geom_meta["rho0"]),
        pml_size=int(geom_meta.get("pml", 20)),
        cfl=float(geom_meta.get("cfl", 0.3)),
        f0=float(meta.get("f0_hz", 7.5e6)),
        ncycles=int(meta.get("ncycles", 3)),
    )


def _infer_angles(meta: dict) -> list[float]:
    if "angles_deg" in meta:
        return [float(a) for a in meta["angles_deg"]]
    if "angles_deg_sets" in meta:
        angle_sets_meta = meta["angles_deg_sets"]
        if not angle_sets_meta:
            raise ValueError("meta['angles_deg_sets'] is empty")
        return [float(a) for a in angle_sets_meta[0]]
    if "base_angles_deg" in meta:
        return [float(a) for a in meta["base_angles_deg"]]
    if "angles_used_deg" in meta:
        used = meta["angles_used_deg"]
        if not used:
            raise ValueError("meta['angles_used_deg'] is empty")
        return [float(a) for a in used[0]]
    raise KeyError(
        "meta.json missing 'angles_deg'/'angles_deg_sets'/'base_angles_deg'/'angles_used_deg'"
    )


def load_replay_source(src_root: Path) -> tuple[dict, SimGeom, list[list[AngleData]]]:
    meta = _load_meta(src_root)
    geom = _build_geom(meta)
    angles = _infer_angles(meta)
    angle_sets = _load_angle_data(src_root, angles)
    return meta, geom, angle_sets


def build_window_specs(
    *,
    window_length: int | None,
    window_offsets: Sequence[int],
    total_frames_full: int,
) -> list[WindowSpec]:
    specs: list[WindowSpec] = []
    if window_length is None:
        if window_offsets:
            raise ValueError("--time-window-offset requires --time-window-length")
        specs.append(WindowSpec(offset=None, length=None, suffix=None))
        return specs

    if window_length <= 0:
        raise ValueError("--time-window-length must be positive")
    offsets = list(window_offsets) if window_offsets else [0]
    for idx, offset in enumerate(offsets):
        if offset < 0:
            raise ValueError("time-window offsets must be non-negative")
        if offset + window_length > total_frames_full:
            raise ValueError(
                "time window offset "
                f"{offset} + length {window_length} exceeds total frames {total_frames_full}"
            )
        suffix = None if len(offsets) == 1 else f"win{idx}_off{offset}"
        specs.append(WindowSpec(offset=int(offset), length=int(window_length), suffix=suffix))
    return specs
