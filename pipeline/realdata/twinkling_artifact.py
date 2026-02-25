from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Iterator, Literal

import numpy as np


def twinkling_artifact_data_root(base: Path | str = Path("data")) -> Path:
    base_path = Path(base)
    return base_path / "twinkling_artifact"


def _parse_par_value(raw: str) -> object:
    raw = str(raw).strip()
    if not raw:
        return ""
    if "//" in raw:
        parts = [p.strip() for p in raw.split("//") if p.strip()]
        ints: list[int] = []
        for part in parts:
            try:
                ints.append(int(part))
            except Exception:
                return raw
        return tuple(ints) if len(ints) > 1 else ints[0]
    try:
        return int(raw)
    except Exception:
        pass
    try:
        return float(raw)
    except Exception:
        return raw


def parse_rawbcf_par(path: Path) -> dict[str, object]:
    out: dict[str, object] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or "=" not in line:
            continue
        key, value = line.split("=", 1)
        out[key.strip()] = _parse_par_value(value)
    return out


@dataclass
class RawBCFPar:
    num_frames: int
    raw_frame_size: int
    header_size: int
    num_b_beams: int
    b_beam_samples: int
    num_cfm_shots: int
    num_cfm_beams: int
    cfm_beam_samples: int

    num_sweeps: int | None = None
    beams_in_sweep: int | None = None
    first_scan_cfm_beam: int | None = None
    cfm_density: int | None = None
    num_first_cfm_sample: tuple[int, int] | int | None = None
    cfm_filter_order: int | None = None

    raw: dict[str, object] | None = None

    @staticmethod
    def from_dict(d: dict[str, object]) -> "RawBCFPar":
        def _get_int(key: str) -> int:
            value = d.get(key)
            if value is None:
                raise KeyError(key)
            if isinstance(value, bool):
                raise TypeError(f"{key} must be int, got bool")
            return int(value)  # type: ignore[arg-type]

        def _get_opt_int(key: str) -> int | None:
            value = d.get(key)
            if value is None:
                return None
            if isinstance(value, bool):
                return None
            try:
                return int(value)  # type: ignore[arg-type]
            except Exception:
                return None

        raw_first = d.get("NumOfFirstCFMSample")
        num_first: tuple[int, int] | int | None
        if isinstance(raw_first, tuple) and len(raw_first) == 2:
            a, b = raw_first
            num_first = (int(a), int(b))
        elif raw_first is None:
            num_first = None
        else:
            try:
                num_first = int(raw_first)  # type: ignore[arg-type]
            except Exception:
                num_first = None

        return RawBCFPar(
            num_frames=_get_int("NumOfFrames"),
            raw_frame_size=_get_int("RawFrameSize"),
            header_size=_get_int("HeaderSize"),
            num_b_beams=_get_int("NumOfBBeams"),
            b_beam_samples=_get_int("SizeofBBeamAtSamples"),
            num_cfm_shots=_get_int("NumOfCFShots"),
            num_cfm_beams=_get_int("NumOfCFMBeams"),
            cfm_beam_samples=_get_int("SizeofCFMBeamAtSamples"),
            num_sweeps=_get_opt_int("NumOfSweeps"),
            beams_in_sweep=_get_opt_int("BeamsInSweep"),
            first_scan_cfm_beam=_get_opt_int("FirstScanCFMBeam"),
            cfm_density=_get_opt_int("CFMDensity"),
            num_first_cfm_sample=num_first,
            cfm_filter_order=_get_opt_int("CFMFilterOrder"),
            raw=dict(d),
        )

    @property
    def n_lines_per_frame(self) -> int:
        return int(self.num_b_beams + self.num_cfm_shots * self.num_cfm_beams)

    @property
    def sample_bytes(self) -> int:
        header_total = self.header_size * self.n_lines_per_frame
        payload_bytes = self.raw_frame_size - header_total
        n_samples = self.num_b_beams * self.b_beam_samples + (
            self.num_cfm_shots * self.num_cfm_beams * self.cfm_beam_samples
        )
        if n_samples <= 0:
            raise ValueError("Invalid sample counts in par.")
        val = payload_bytes / float(n_samples)
        val_round = int(round(val))
        if abs(val - val_round) > 1e-6:
            raise ValueError(f"Non-integer bytes/sample inferred: {val}")
        return val_round

    @property
    def b_line_bytes(self) -> int:
        return int(self.header_size + self.b_beam_samples * self.sample_bytes)

    @property
    def cfm_line_bytes(self) -> int:
        return int(self.header_size + self.cfm_beam_samples * self.sample_bytes)

    def validate(self) -> None:
        expected = self.num_b_beams * self.b_line_bytes + (
            self.num_cfm_shots * self.num_cfm_beams * self.cfm_line_bytes
        )
        if expected != self.raw_frame_size:
            raise ValueError(
                "raw_frame_size mismatch: "
                f"expected {expected} from par, got {self.raw_frame_size}"
            )
        if self.sample_bytes != 8:
            raise ValueError(f"Unexpected sample_bytes={self.sample_bytes}; expected 8 (2x int32).")


def read_rawbcf_frame(dat_path: Path, par: RawBCFPar, frame_idx: int) -> bytes:
    if frame_idx < 0 or frame_idx >= par.num_frames:
        raise IndexError(f"frame_idx out of range: {frame_idx} not in [0,{par.num_frames})")
    with dat_path.open("rb") as f:
        f.seek(int(frame_idx) * par.raw_frame_size)
        frame = f.read(par.raw_frame_size)
    if len(frame) != par.raw_frame_size:
        raise IOError(f"Short read: expected {par.raw_frame_size} bytes, got {len(frame)}")
    return frame


def iter_rawbcf_frames(
    dat_path: Path, par: RawBCFPar, frame_indices: Iterable[int] | None = None
) -> Iterator[tuple[int, bytes]]:
    indices = list(range(par.num_frames)) if frame_indices is None else list(frame_indices)
    with dat_path.open("rb") as f:
        for idx in indices:
            if idx < 0 or idx >= par.num_frames:
                raise IndexError(f"frame_idx out of range: {idx} not in [0,{par.num_frames})")
            f.seek(int(idx) * par.raw_frame_size)
            frame = f.read(par.raw_frame_size)
            if len(frame) != par.raw_frame_size:
                raise IOError(f"Short read at frame {idx}: expected {par.raw_frame_size} bytes")
            yield idx, frame


def decode_rawbcf_bmode_iq(frame: bytes, par: RawBCFPar) -> np.ndarray:
    par.validate()
    if len(frame) != par.raw_frame_size:
        raise ValueError(f"frame length mismatch: expected {par.raw_frame_size}, got {len(frame)}")
    if par.sample_bytes != 8:
        raise ValueError("Only 8-byte complex samples are supported.")
    # Complex IQ samples are stored as (I,Q) int32 little-endian pairs.
    out = np.empty((par.num_b_beams, par.b_beam_samples), dtype=np.complex64)
    offset = 0
    for b in range(par.num_b_beams):
        offset += par.header_size
        data = np.frombuffer(
            frame, dtype="<i4", count=par.b_beam_samples * 2, offset=offset
        ).reshape(-1, 2)
        out[b] = data[:, 0].astype(np.float32) + 1j * data[:, 1].astype(np.float32)
        offset += par.b_beam_samples * 2 * 4
    return out


CFMOrder = Literal["beam_major", "shot_major"]


def decode_rawbcf_cfm_cube(frame: bytes, par: RawBCFPar, *, order: CFMOrder = "beam_major") -> np.ndarray:
    """
    Decode the CFM (color flow) block into an IQ cube with shape (T, H, W):
      - T = NumOfCFShots (slow-time ensemble length)
      - H = SizeofCFMBeamAtSamples (axial samples per beam)
      - W = NumOfCFMBeams (lateral beams)

    The file layout is assumed to be:
      - First, all B-mode beams
      - Then, CFM beams as (beam-major, shot-minor) by default

    The `beam_major` assumption is validated empirically on the Gammex phantom:
    it yields substantially higher temporal coherence than `shot_major`.
    """
    par.validate()
    if len(frame) != par.raw_frame_size:
        raise ValueError(f"frame length mismatch: expected {par.raw_frame_size}, got {len(frame)}")
    if par.sample_bytes != 8:
        raise ValueError("Only 8-byte complex samples are supported.")

    cube = np.empty(
        (par.num_cfm_shots, par.cfm_beam_samples, par.num_cfm_beams), dtype=np.complex64
    )
    offset = par.num_b_beams * par.b_line_bytes

    if order == "beam_major":
        for beam in range(par.num_cfm_beams):
            for shot in range(par.num_cfm_shots):
                offset += par.header_size
                data = np.frombuffer(
                    frame, dtype="<i4", count=par.cfm_beam_samples * 2, offset=offset
                ).reshape(-1, 2)
                cube[shot, :, beam] = data[:, 0].astype(np.float32) + 1j * data[:, 1].astype(
                    np.float32
                )
                offset += par.cfm_beam_samples * 2 * 4
    elif order == "shot_major":
        for shot in range(par.num_cfm_shots):
            for beam in range(par.num_cfm_beams):
                offset += par.header_size
                data = np.frombuffer(
                    frame, dtype="<i4", count=par.cfm_beam_samples * 2, offset=offset
                ).reshape(-1, 2)
                cube[shot, :, beam] = data[:, 0].astype(np.float32) + 1j * data[:, 1].astype(
                    np.float32
                )
                offset += par.cfm_beam_samples * 2 * 4
    else:
        raise ValueError(f"Unknown CFM order: {order!r}")

    if offset != par.raw_frame_size:
        raise ValueError(f"Decode ended at offset={offset}, expected {par.raw_frame_size}")
    return cube

