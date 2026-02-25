from __future__ import annotations

import argparse
import json
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
from sim.kwave import common as kw


def _iter_bundle_dirs(root: Path) -> list[Path]:
    out: list[Path] = []
    for meta_path in root.rglob("meta.json"):
        if meta_path.is_file():
            out.append(meta_path.parent)
    out.sort()
    return out


def _load_meta(bundle_dir: Path) -> dict[str, Any] | None:
    try:
        meta = json.loads((bundle_dir / "meta.json").read_text())
    except Exception:
        return None
    return meta if isinstance(meta, dict) else None


def _apply_translation_per_shot(iq: np.ndarray, dy: np.ndarray, dx: np.ndarray) -> np.ndarray:
    iq = np.asarray(iq, dtype=np.complex64)
    T, _, _ = iq.shape
    if dy.shape != (T,) or dx.shape != (T,):
        raise ValueError(f"dy/dx must have shape (T,), got {dy.shape} / {dx.shape} for T={T}")
    out = np.empty_like(iq)
    for t in range(T):
        shift = (float(dy[t]), float(dx[t]))
        re = ndi.shift(iq[t].real, shift=shift, order=1, mode="nearest", prefilter=False)
        im = ndi.shift(iq[t].imag, shift=shift, order=1, mode="nearest", prefilter=False)
        out[t] = re.astype(np.float32, copy=False) + 1j * im.astype(np.float32, copy=False)
    return out


def _compute_baseline_scores_from_filtered_cube(filtered: np.ndarray, pd_base: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    eps = 1e-12
    score_pdlog = np.log(pd_base.astype(np.float64, copy=False) + eps).astype(np.float32, copy=False)
    if filtered.shape[0] < 2:
        raise ValueError("Need T>=2 for Kasai.")
    y = filtered.astype(np.complex64, copy=False)
    r1 = np.sum(y[1:] * np.conj(y[:-1]), axis=0).astype(np.complex64, copy=False)
    score_kasai = np.log(np.abs(r1).astype(np.float64, copy=False) + eps).astype(np.float32, copy=False)
    return score_pdlog, score_kasai


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Backfill baseline score maps for existing Twinkling bundles.\n"
            "Writes:\n"
            "  - score_base_pdlog.npy  (log power Doppler; derived from pd_base)\n"
            "  - score_base_kasai.npy  (log |lag-1 autocorr| on baseline-filtered cube)\n\n"
            "This avoids re-running STAP; it re-decodes the RawBCF cine frame and re-applies the baseline "
            "clutter filter using settings inferred from meta.json (svd_bandpass only)."
        )
    )
    parser.add_argument("--root", type=Path, required=True, help="Root containing bundles (directories with meta.json).")
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing score_base_pdlog.npy / score_base_kasai.npy if present.",
    )
    args = parser.parse_args()

    bundle_dirs = _iter_bundle_dirs(args.root)
    if not bundle_dirs:
        raise SystemExit(f"No bundles found under: {args.root}")

    n_done = 0
    n_skipped = 0
    n_failed = 0
    failures: list[dict[str, str]] = []

    for bundle_dir in bundle_dirs:
        out_pdlog = bundle_dir / "score_base_pdlog.npy"
        out_kasai = bundle_dir / "score_base_kasai.npy"
        if (out_pdlog.exists() and out_kasai.exists()) and not bool(args.overwrite):
            n_skipped += 1
            continue

        meta = _load_meta(bundle_dir)
        if meta is None:
            n_failed += 1
            failures.append({"bundle": str(bundle_dir), "error": "missing_meta"})
            continue

        tw = meta.get("twinkling_rawbcf")
        if not isinstance(tw, dict):
            n_failed += 1
            failures.append({"bundle": str(bundle_dir), "error": "missing_twinkling_rawbcf"})
            continue
        try:
            dat_path = Path(str(tw["dat_path"]))
            par_path = Path(str(tw["par_path"]))
            frame_idx = int(tw["frame_idx"])
        except Exception as e:
            n_failed += 1
            failures.append({"bundle": str(bundle_dir), "error": f"bad_twinkling_rawbcf:{e}"})
            continue

        baseline = meta.get("baseline_stats") or {}
        baseline_type = str(baseline.get("baseline_type") or "").strip().lower()
        if baseline_type not in {"svd_bandpass", "svd_range", "ulm_svd"}:
            n_failed += 1
            failures.append({"bundle": str(bundle_dir), "error": f"unsupported_baseline_type:{baseline_type}"})
            continue
        try:
            keep_min = int(baseline.get("svd_keep_min") or 1)
            keep_max = baseline.get("svd_keep_max")
            keep_max_i = int(keep_max) if keep_max is not None else None
            reg_enable = bool(baseline.get("reg_enable", False))
            reg_method = str(baseline.get("reg_method") or "phasecorr")
            reg_subpixel = int(baseline.get("reg_subpixel") or 4)
            reg_reference = str(baseline.get("reg_reference") or "median")
        except Exception as e:
            n_failed += 1
            failures.append({"bundle": str(bundle_dir), "error": f"bad_baseline_settings:{e}"})
            continue

        # Decode cine frame -> CFM IQ cube (T,H,W).
        try:
            par_dict = parse_rawbcf_par(par_path)
            par = RawBCFPar.from_dict(par_dict)
            par.validate()
            frame = read_rawbcf_frame(dat_path, par, frame_idx)
            Icube = decode_rawbcf_cfm_cube(frame, par, order=str(tw.get("decode_cfm_order") or "beam_major"))
        except Exception as e:
            n_failed += 1
            failures.append({"bundle": str(bundle_dir), "error": f"decode_failed:{e}"})
            continue

        # If the bundle used within-ensemble motion injection, re-apply it before baseline filtering.
        wem = meta.get("within_ensemble_motion")
        if isinstance(wem, dict):
            try:
                amp = float(wem.get("amp_px") or 0.0)
                if amp > 0.0:
                    dy = np.asarray(wem.get("dy_px") or [], dtype=np.float32)
                    dx = np.asarray(wem.get("dx_px") or [], dtype=np.float32)
                    if dy.size and dx.size:
                        Icube = _apply_translation_per_shot(Icube, dy=dy, dx=dx)
            except Exception:
                # Best-effort: if motion metadata is malformed, proceed without it.
                pass

        # Apply baseline filter and compute score maps.
        try:
            pd_base, _, filtered = kw._baseline_pd_svd_bandpass(
                Icube,
                reg_enable=reg_enable,
                reg_method=reg_method,
                reg_subpixel=reg_subpixel,
                reg_reference=reg_reference,
                svd_keep_min=keep_min,
                svd_keep_max=keep_max_i,
                device="cpu",
                return_filtered_cube=True,
            )
            score_pdlog, score_kasai = _compute_baseline_scores_from_filtered_cube(filtered, pd_base)
            np.save(out_pdlog, score_pdlog, allow_pickle=False)
            np.save(out_kasai, score_kasai, allow_pickle=False)
            n_done += 1
        except Exception as e:
            n_failed += 1
            failures.append({"bundle": str(bundle_dir), "error": f"baseline_compute_failed:{e}"})
            continue

    report = {
        "root": str(args.root),
        "overwrite": bool(args.overwrite),
        "bundles_total": int(len(bundle_dirs)),
        "bundles_done": int(n_done),
        "bundles_skipped": int(n_skipped),
        "bundles_failed": int(n_failed),
        "failures_head": failures[:20],
    }
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()

