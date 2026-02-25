from __future__ import annotations

import argparse
import json
import math
import os
import re
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import asdict
from pathlib import Path
from typing import Any

import numpy as np
import scipy.ndimage as ndi

from pipeline.realdata.twinkling_artifact import (
    RawBCFPar,
    decode_rawbcf_cfm_cube,
    parse_rawbcf_par,
    read_rawbcf_frame,
)
from pipeline.realdata.twinkling_bmode_mask import build_tube_masks_from_rawbcf
from sim.kwave.icube_bundle import write_acceptance_bundle_from_icube


def _slugify(text: str) -> str:
    s = re.sub(r"[\s/]+", "_", text.strip())
    s = re.sub(r"[^A-Za-z0-9_.-]+", "_", s)
    return s.strip("._-") or "seq"


def _parse_indices(spec: str, n_max: int) -> list[int]:
    """
    Parse either:
      - comma list: "0,1,2"
      - slice: "0:10" or "0:10:2"
      - single int: "5"
    """
    spec = (spec or "").strip()
    if not spec:
        return list(range(n_max))
    if "," in spec:
        out: list[int] = []
        for part in spec.split(","):
            part = part.strip()
            if part:
                out.append(int(part))
        return out
    if ":" in spec:
        parts = [p.strip() for p in spec.split(":")]
        if len(parts) not in (2, 3):
            raise ValueError(f"Invalid slice spec: {spec!r}")
        start = int(parts[0]) if parts[0] else 0
        stop = int(parts[1]) if parts[1] else n_max
        step = int(parts[2]) if len(parts) == 3 and parts[2] else 1
        return list(range(start, min(stop, n_max), step))
    return [int(spec)]


def _parse_float_list(spec: str) -> list[float]:
    spec = (spec or "").strip()
    if not spec:
        raise ValueError("Expected a non-empty float list (e.g. '0,0.5,1,2').")
    out: list[float] = []
    for part in spec.replace(" ", "").split(","):
        if not part:
            continue
        out.append(float(part))
    if not out:
        raise ValueError("Expected a non-empty float list (e.g. '0,0.5,1,2').")
    return out


def _amp_tag(amp_px: float) -> str:
    # Stable filesystem-friendly tag (avoid scientific notation).
    s = f"{float(amp_px):.3f}".rstrip("0").rstrip(".")
    return ("amp" + s).replace(".", "p").replace("-", "m")


def _motion_shifts(kind: str, *, T: int, amp_px: float, seed: int) -> tuple[np.ndarray, np.ndarray]:
    """
    Return (dy[T], dx[T]) in pixels for per-shot rigid translation.

    We keep the mean shift ~0 to avoid turning this into a pure label-misalignment test.
    """
    kind = (kind or "sine").strip().lower()
    T = int(T)
    if T <= 0:
        raise ValueError("T must be positive.")
    amp = float(max(0.0, amp_px))
    if amp <= 0.0:
        return np.zeros((T,), dtype=np.float32), np.zeros((T,), dtype=np.float32)

    if kind in {"sine", "sin"}:
        tt = np.arange(T, dtype=np.float32)
        dx = amp * np.sin(2.0 * math.pi * tt / float(T)).astype(np.float32)
        dy = np.zeros_like(dx)
    elif kind in {"step", "burst"}:
        dx = np.zeros((T,), dtype=np.float32)
        dx[T // 2 :] = amp
        dx -= float(np.mean(dx))
        dy = np.zeros_like(dx)
    elif kind in {"rw", "randomwalk"}:
        rng = np.random.default_rng(int(seed))
        steps = rng.normal(scale=1.0, size=T).astype(np.float32)
        walk = np.cumsum(steps).astype(np.float32)
        walk -= float(np.mean(walk))
        rms = float(np.sqrt(np.mean(walk * walk))) + 1e-12
        dx = (amp * (walk / rms)).astype(np.float32, copy=False)
        dy = np.zeros_like(dx)
    else:
        raise ValueError(f"Unsupported motion kind: {kind!r} (use sine|step|rw).")
    return dy.astype(np.float32, copy=False), dx.astype(np.float32, copy=False)


def _apply_translation_per_shot(iq: np.ndarray, dy: np.ndarray, dx: np.ndarray) -> np.ndarray:
    iq = np.asarray(iq, dtype=np.complex64)
    T, H, W = iq.shape
    if dy.shape != (T,) or dx.shape != (T,):
        raise ValueError(f"dy/dx must have shape (T,), got {dy.shape} / {dx.shape} for T={T}")
    out = np.empty_like(iq)
    for t in range(T):
        shift = (float(dy[t]), float(dx[t]))
        re = ndi.shift(iq[t].real, shift=shift, order=1, mode="nearest", prefilter=False)
        im = ndi.shift(iq[t].imag, shift=shift, order=1, mode="nearest", prefilter=False)
        out[t] = re.astype(np.float32, copy=False) + 1j * im.astype(np.float32, copy=False)
    return out


def _process_frame_worker(
    *,
    frame_idx: int,
    dat_path: str,
    par_dict: dict[str, Any],
    seq_dir: str,
    par_path: str,
    seq_slug: str,
    out_root: str,
    motion_kind: str,
    amp_list: list[float],
    amp_shifts: dict[str, dict[str, list[float]]],
    prf_hz: float,
    tile_hw: tuple[int, int],
    tile_stride: int,
    Lt: int,
    diag_load: float,
    cov_estimator: str,
    score_mode: str,
    baseline_type: str,
    svd_keep_min: int,
    svd_keep_max: int | None,
    flow_lo: float,
    flow_hi: float,
    alias_center: float,
    alias_halfwidth: float,
    score_ka_v2_enable: bool,
    score_ka_v2_mode: str,
    mask_flow_override: np.ndarray,
    mask_bg_override: np.ndarray,
    mask_meta: dict[str, Any],
    seed: int,
    limit_threads: bool,
) -> int:
    os.environ.setdefault("STAP_FAST_PATH", "1")
    if limit_threads:
        os.environ.setdefault("OMP_NUM_THREADS", "1")
        os.environ.setdefault("MKL_NUM_THREADS", "1")
        os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
        os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")
        try:
            import torch

            torch.set_num_threads(1)
        except Exception:
            pass

    par = RawBCFPar.from_dict(par_dict)
    par.validate()

    frame = read_rawbcf_frame(Path(dat_path), par, int(frame_idx))
    Icube = decode_rawbcf_cfm_cube(frame, par, order="beam_major")
    count = 0
    for amp_px in amp_list:
        shifts = amp_shifts.get(str(float(amp_px)))
        if shifts is None:
            raise RuntimeError(f"Missing precomputed shifts for amp={amp_px}")
        dy = np.asarray(shifts["dy"], dtype=np.float32)
        dx = np.asarray(shifts["dx"], dtype=np.float32)
        Icube_m = _apply_translation_per_shot(Icube, dy=dy, dx=dx) if float(amp_px) > 0.0 else Icube

        dataset_name = f"{seq_slug}/{motion_kind}/{_amp_tag(float(amp_px))}/frame{int(frame_idx):03d}"
        meta_extra: dict[str, Any] = {
            "twinkling_rawbcf": {
                "seq_dir": str(seq_dir),
                "par_path": str(par_path),
                "dat_path": str(dat_path),
                "frame_idx": int(frame_idx),
                "decode_cfm_order": "beam_major",
                "dtype": "int32_iq_pairs_le",
                "par_keys": {k: str(v) for k, v in (par.raw or {}).items()},
            },
            "twinkling_eval_masks": mask_meta,
            "within_ensemble_motion": {
                "kind": str(motion_kind),
                "amp_px": float(amp_px),
                "seed": int(seed),
                "dy_px": [float(x) for x in dy.tolist()],
                "dx_px": [float(x) for x in dx.tolist()],
                "note": (
                    "Per-shot rigid translation applied to CFM IQ within a single cine frame "
                    "prior to baseline/STAP. Structural masks are B-mode-only and fixed."
                ),
            },
        }
        write_acceptance_bundle_from_icube(
            out_root=Path(out_root),
            dataset_name=dataset_name,
            Icube=Icube_m,
            prf_hz=float(prf_hz),
            tile_hw=tile_hw,
            tile_stride=int(tile_stride),
            Lt=int(Lt),
            diag_load=float(diag_load),
            cov_estimator=str(cov_estimator),
            score_mode=str(score_mode),
            baseline_type=str(baseline_type),
            svd_keep_min=int(svd_keep_min),
            svd_keep_max=int(svd_keep_max) if svd_keep_max is not None else None,
            band_ratio_flow_low_hz=float(flow_lo),
            band_ratio_flow_high_hz=float(flow_hi),
            band_ratio_alias_center_hz=float(alias_center),
            # NOTE: `band_ratio_alias_width_hz` is a *half-width* in Hz
            # (see `sim/kwave/common.py:_freq_bins_from_hz`). Do not double it.
            band_ratio_alias_width_hz=float(alias_halfwidth),
            score_ka_v2_enable=bool(score_ka_v2_enable),
            score_ka_v2_mode=str(score_ka_v2_mode),
            mask_flow_override=mask_flow_override,
            mask_bg_override=mask_bg_override,
            stap_conditional_enable=False,
            meta_extra=meta_extra,
        )
        count += 1
    return count


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Within-ensemble motion ladder on Twinkling/Gammex: inject per-shot rigid motion into CFM IQ "
            "within each cine frame, write acceptance bundles, and record motion metadata in meta.json.\n\n"
            "Evaluation masks are B-mode structural tube masks (mask_flow/mask_bg) and are kept fixed."
        )
    )
    parser.add_argument("--seq-dir", type=Path, required=True)
    parser.add_argument("--par-path", type=Path, default=None)
    parser.add_argument("--dat-path", type=Path, default=None)
    parser.add_argument("--out-root", type=Path, default=Path("runs/real/twinkling_gammex_within_ensemble_motion"))
    parser.add_argument("--frames", type=str, default="0:20")
    parser.add_argument("--mask-ref-frames", type=str, default="0:20")

    parser.add_argument("--prf-hz", type=float, required=True, help="CFM PRF in Hz (must be provided explicitly).")
    parser.add_argument("--Lt", type=int, default=16)
    parser.add_argument("--tile-hw", type=int, nargs=2, default=(8, 8))
    parser.add_argument("--tile-stride", type=int, default=6)
    parser.add_argument("--diag-load", type=float, default=0.07)
    parser.add_argument(
        "--cov-estimator",
        type=str,
        default="tyler_pca",
        choices=["scm", "huber", "tyler", "tyler_pca"],
    )
    parser.add_argument(
        "--baseline-type",
        type=str,
        default="svd_bandpass",
        choices=["svd_bandpass", "mc_svd"],
    )
    parser.add_argument("--svd-keep-min", type=int, default=2)
    parser.add_argument("--svd-keep-max", type=int, default=None)
    parser.add_argument(
        "--score-mode",
        type=str,
        default="msd",
        choices=["pd", "msd", "band_ratio"],
        help="Primary score family (default: msd; evaluates score_stap_preka).",
    )
    parser.add_argument("--score-ka-v2-enable", action="store_true", help="Enable score-space KA v2 (default off).")
    parser.add_argument("--score-ka-v2-mode", type=str, default="auto", choices=["safety", "uplift", "auto"])
    parser.add_argument("--flow-band-hz", type=float, nargs=2, default=(150.0, 450.0))
    parser.add_argument("--alias-band-center-hz", type=float, default=950.0)
    parser.add_argument("--alias-band-halfwidth-hz", type=float, default=250.0)

    parser.add_argument("--amp-px-list", type=str, required=True, help="Comma-separated amps in pixels, e.g. '0,0.5,1,2'.")
    parser.add_argument("--motion-kind", type=str, default="sine", choices=["sine", "step", "rw"])
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--workers",
        type=int,
        default=0,
        help=(
            "Process frames in parallel using multiple worker processes (spawn). "
            "Each worker processes all amps for one frame. "
            "When set, we default OMP/MKL/OPENBLAS threads to 1 to avoid oversubscription. "
            "Default: 0 (sequential)."
        ),
    )

    args = parser.parse_args()

    # Ensure STAP batched fast-path is on (still CPU if no CUDA).
    os.environ.setdefault("STAP_FAST_PATH", "1")

    seq_dir = Path(args.seq_dir)
    if args.par_path is not None:
        par_path = Path(args.par_path)
    else:
        cand = seq_dir / "RawBCFCine.par"
        if cand.exists():
            par_path = cand
        else:
            pars = sorted(seq_dir.glob("*.par"))
            if len(pars) != 1:
                raise FileNotFoundError(f"Could not auto-detect a single .par in {seq_dir}: {pars}")
            par_path = pars[0]

    if args.dat_path is not None:
        dat_path = Path(args.dat_path)
    else:
        cand = seq_dir / "RawBCFCine.dat"
        if cand.exists():
            dat_path = cand
        else:
            dats = sorted(seq_dir.glob("*.dat"))
            if len(dats) != 1:
                raise FileNotFoundError(f"Could not auto-detect a single .dat in {seq_dir}: {dats}")
            dat_path = dats[0]

    par_dict = parse_rawbcf_par(par_path)
    par = RawBCFPar.from_dict(par_dict)
    par.validate()

    frame_indices = _parse_indices(args.frames, par.num_frames)
    mask_ref_frames = _parse_indices(args.mask_ref_frames, par.num_frames)

    # Structural tube masks from B-mode only (fixed across all motion amps).
    tube = build_tube_masks_from_rawbcf(dat_path, par, ref_frame_indices=mask_ref_frames)
    mask_flow_override = tube.mask_flow
    mask_bg_override = tube.mask_bg
    mask_meta = {
        "mask_source": "bmode_structural_tube",
        "b0_bbeam_0based": int(tube.mapping.b0),
        "s0_sample_0based": int(tube.mapping.s0),
        "cfm_density_Q": int(tube.mapping.Q),
        "valid_beam_fraction": float(np.mean(tube.mapping.valid_beams)),
        "ref_frames": mask_ref_frames,
        "params": tube.params,
        "qc": tube.qc,
    }

    # Frozen per-sequence identifier.
    rel = str(seq_dir)
    try:
        if seq_dir.is_relative_to(Path.cwd()):
            rel = str(seq_dir.relative_to(Path.cwd()))
    except Exception:
        pass
    seq_slug = _slugify(f"{rel}__{par_path.stem}")

    T = int(par.num_cfm_shots)
    Lt_req = int(args.Lt)
    Lt = max(2, min(Lt_req, T - 1))
    if Lt != Lt_req:
        print(f"[twinkling_motion] Clamped Lt from {Lt_req} to {Lt} (NumOfCFShots={T}).")

    amp_list = sorted(set(float(a) for a in _parse_float_list(args.amp_px_list)))
    if 0.0 not in amp_list:
        amp_list = [0.0] + amp_list

    tile_hw = (int(args.tile_hw[0]), int(args.tile_hw[1]))
    tile_stride = int(args.tile_stride)

    flow_lo, flow_hi = float(args.flow_band_hz[0]), float(args.flow_band_hz[1])
    alias_center = float(args.alias_band_center_hz)
    alias_halfwidth = float(args.alias_band_halfwidth_hz)

    written = 0
    out_root = Path(args.out_root)
    amp_shifts: dict[float, tuple[np.ndarray, np.ndarray]] = {}
    for amp_px in amp_list:
        amp_shifts[float(amp_px)] = _motion_shifts(str(args.motion_kind), T=T, amp_px=float(amp_px), seed=int(args.seed))

    amp_shifts_list: dict[str, dict[str, list[float]]] = {}
    for amp_px, (dy, dx) in amp_shifts.items():
        amp_shifts_list[str(float(amp_px))] = {"dy": [float(x) for x in dy.tolist()], "dx": [float(x) for x in dx.tolist()]}

    out_root.mkdir(parents=True, exist_ok=True)
    workers = int(max(0, args.workers))
    if workers <= 1:
        for frame_idx in frame_indices:
            written += _process_frame_worker(
                frame_idx=int(frame_idx),
                dat_path=str(dat_path),
                par_dict=par_dict,
                seq_dir=str(seq_dir),
                par_path=str(par_path),
                seq_slug=seq_slug,
                out_root=str(out_root),
                motion_kind=str(args.motion_kind),
                amp_list=amp_list,
                amp_shifts=amp_shifts_list,
                prf_hz=float(args.prf_hz),
                tile_hw=tile_hw,
                tile_stride=tile_stride,
                Lt=int(Lt),
                diag_load=float(args.diag_load),
                cov_estimator=str(args.cov_estimator),
                score_mode=str(args.score_mode),
                baseline_type=str(args.baseline_type),
                svd_keep_min=int(args.svd_keep_min),
                svd_keep_max=int(args.svd_keep_max) if args.svd_keep_max is not None else None,
                flow_lo=float(flow_lo),
                flow_hi=float(flow_hi),
                alias_center=float(alias_center),
                alias_halfwidth=float(alias_halfwidth),
                score_ka_v2_enable=bool(args.score_ka_v2_enable),
                score_ka_v2_mode=str(args.score_ka_v2_mode),
                mask_flow_override=mask_flow_override,
                mask_bg_override=mask_bg_override,
                mask_meta=mask_meta,
                seed=int(args.seed),
                limit_threads=False,
            )
            print(f"[twinkling_motion] wrote frame {int(frame_idx):03d} ({written} bundles so far)")
    else:
        # Spawn is safer with torch + OpenMP stacks.
        import multiprocessing as mp

        os.environ.setdefault("OMP_NUM_THREADS", "1")
        os.environ.setdefault("MKL_NUM_THREADS", "1")
        os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
        os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")
        ctx = mp.get_context("spawn")
        with ProcessPoolExecutor(max_workers=workers, mp_context=ctx) as ex:
            futs = {
                ex.submit(
                    _process_frame_worker,
                    frame_idx=int(fi),
                    dat_path=str(dat_path),
                    par_dict=par_dict,
                    seq_dir=str(seq_dir),
                    par_path=str(par_path),
                    seq_slug=seq_slug,
                    out_root=str(out_root),
                    motion_kind=str(args.motion_kind),
                    amp_list=amp_list,
                    amp_shifts=amp_shifts_list,
                    prf_hz=float(args.prf_hz),
                    tile_hw=tile_hw,
                    tile_stride=tile_stride,
                    Lt=int(Lt),
                    diag_load=float(args.diag_load),
                    cov_estimator=str(args.cov_estimator),
                    score_mode=str(args.score_mode),
                    baseline_type=str(args.baseline_type),
                    svd_keep_min=int(args.svd_keep_min),
                    svd_keep_max=int(args.svd_keep_max) if args.svd_keep_max is not None else None,
                    flow_lo=float(flow_lo),
                    flow_hi=float(flow_hi),
                    alias_center=float(alias_center),
                    alias_halfwidth=float(alias_halfwidth),
                    score_ka_v2_enable=bool(args.score_ka_v2_enable),
                    score_ka_v2_mode=str(args.score_ka_v2_mode),
                    mask_flow_override=mask_flow_override,
                    mask_bg_override=mask_bg_override,
                    mask_meta=mask_meta,
                    seed=int(args.seed),
                    limit_threads=True,
                ): int(fi)
                for fi in frame_indices
            }
            for fut in as_completed(futs):
                fi = futs[fut]
                n = fut.result()
                written += int(n)
                print(f"[twinkling_motion] wrote frame {fi:03d} ({n} bundles; total={written})")

    report = {
        "seq_dir": str(seq_dir),
        "par_path": str(par_path),
        "dat_path": str(dat_path),
        "out_root": str(out_root),
        "seq_slug": seq_slug,
        "frames": frame_indices,
        "mask_ref_frames": mask_ref_frames,
        "motion_kind": str(args.motion_kind),
        "amp_px_list": amp_list,
        "seed": int(args.seed),
        "prf_hz": float(args.prf_hz),
        "T_num_cfm_shots": int(T),
        "Lt": int(Lt),
        "tile_hw": list(tile_hw),
        "tile_stride": int(tile_stride),
        "diag_load": float(args.diag_load),
        "cov_estimator": str(args.cov_estimator),
        "baseline_type": str(args.baseline_type),
        "svd_keep_min": int(args.svd_keep_min),
        "svd_keep_max": int(args.svd_keep_max) if args.svd_keep_max is not None else None,
        "score_mode": str(args.score_mode),
        "score_ka_v2_enable": bool(args.score_ka_v2_enable),
        "score_ka_v2_mode": str(args.score_ka_v2_mode),
        "band_ratio_flow_hz": [float(flow_lo), float(flow_hi)],
        "band_ratio_alias_center_hz": float(alias_center),
        "band_ratio_alias_halfwidth_hz": float(alias_halfwidth),
        "written_bundle_count": int(written),
        "mask_qc": tube.qc,
        "mask_params": tube.params,
        "rawbcf_par": asdict(par),
    }
    out_root.mkdir(parents=True, exist_ok=True)
    report_path = out_root / f"{seq_slug}__within_ensemble_motion_report.json"
    report_path.write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
