#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import math
import sys
import tempfile
import zlib
from pathlib import Path
from typing import Any

import numpy as np
import scipy.ndimage as ndi
import scipy.io as sio
from skimage.filters import threshold_otsu

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.ulm7883227_structural_roc import (
    _derive_structural_masks,
    _evaluate_window_score,
    _load_bundle_scores,
)
from sim.kwave.icube_bundle import write_acceptance_bundle_from_icube


MI_INT8 = 1
MI_UINT8 = 2
MI_INT16 = 3
MI_UINT16 = 4
MI_INT32 = 5
MI_UINT32 = 6
MI_SINGLE = 7
MI_DOUBLE = 9
MI_INT64 = 12
MI_UINT64 = 13
MI_MATRIX = 14
MI_COMPRESSED = 15

MX_CELL_CLASS = 1
MX_STRUCT_CLASS = 2
MX_OBJECT_CLASS = 3
MX_CHAR_CLASS = 4
MX_SPARSE_CLASS = 5
MX_DOUBLE_CLASS = 6
MX_SINGLE_CLASS = 7
MX_INT8_CLASS = 8
MX_UINT8_CLASS = 9
MX_INT16_CLASS = 10
MX_UINT16_CLASS = 11
MX_INT32_CLASS = 12
MX_UINT32_CLASS = 13
MX_INT64_CLASS = 14
MX_UINT64_CLASS = 15

MI_TO_DTYPE = {
    MI_INT8: np.dtype("<i1"),
    MI_UINT8: np.dtype("<u1"),
    MI_INT16: np.dtype("<i2"),
    MI_UINT16: np.dtype("<u2"),
    MI_INT32: np.dtype("<i4"),
    MI_UINT32: np.dtype("<u4"),
    MI_SINGLE: np.dtype("<f4"),
    MI_DOUBLE: np.dtype("<f8"),
    MI_INT64: np.dtype("<i8"),
    MI_UINT64: np.dtype("<u8"),
}

MX_CLASS_DEFAULT_MI = {
    MX_DOUBLE_CLASS: MI_DOUBLE,
    MX_SINGLE_CLASS: MI_SINGLE,
    MX_INT8_CLASS: MI_INT8,
    MX_UINT8_CLASS: MI_UINT8,
    MX_INT16_CLASS: MI_INT16,
    MX_UINT16_CLASS: MI_UINT16,
    MX_INT32_CLASS: MI_INT32,
    MX_UINT32_CLASS: MI_UINT32,
    MX_INT64_CLASS: MI_INT64,
    MX_UINT64_CLASS: MI_UINT64,
}


def _align8(n: int) -> int:
    return (int(n) + 7) & ~7


def _iter_mat_elements(buf: bytes):
    off = 0
    size = len(buf)
    while off + 8 <= size:
        tag0 = int.from_bytes(buf[off : off + 4], "little", signed=False)
        tag1 = int.from_bytes(buf[off + 4 : off + 8], "little", signed=False)
        if tag0 == 0 and tag1 == 0:
            break
        if (tag0 >> 16) != 0:
            dtype = int(tag0 & 0xFFFF)
            nbytes = int((tag0 >> 16) & 0xFFFF)
            payload = bytes(buf[off + 4 : off + 4 + nbytes])
            yield dtype, payload
            off += 8
            continue
        dtype = int(tag0)
        nbytes = int(tag1)
        data_off = off + 8
        data_end = data_off + nbytes
        if data_end > size:
            raise ValueError(f"MAT element overruns buffer: dtype={dtype} nbytes={nbytes}")
        payload = bytes(buf[data_off:data_end])
        yield dtype, payload
        off = data_off + _align8(nbytes)


def _parse_numeric_array(payload: bytes, *, mi_type: int, dims: tuple[int, ...]) -> np.ndarray:
    dtype = MI_TO_DTYPE.get(int(mi_type))
    if dtype is None:
        raise ValueError(f"unsupported MAT numeric type {mi_type}")
    arr = np.frombuffer(payload, dtype=dtype)
    expected = int(np.prod(dims, dtype=np.int64))
    if arr.size != expected:
        raise ValueError(f"MAT array size mismatch: expected {expected}, got {arr.size}")
    return np.array(arr.reshape(dims, order="F"), copy=True)


def _parse_matrix(payload: bytes) -> tuple[str, np.ndarray]:
    array_flags: np.ndarray | None = None
    dims: tuple[int, ...] | None = None
    name = ""
    real_part: np.ndarray | None = None
    imag_part: np.ndarray | None = None
    data_mi_type: int | None = None
    is_complex = False
    mx_class = None
    for dtype, subpayload in _iter_mat_elements(payload):
        if array_flags is None:
            if dtype != MI_UINT32:
                raise ValueError("expected array flags as first miMATRIX subelement")
            array_flags = np.frombuffer(subpayload, dtype=np.dtype("<u4")).copy()
            flags_word = int(array_flags[0]) if array_flags.size else 0
            mx_class = int(flags_word & 0xFF)
            is_complex = bool(flags_word & 0x0800)
            continue
        if dims is None:
            dims_arr = np.frombuffer(subpayload, dtype=np.dtype("<i4")).copy()
            dims = tuple(int(v) for v in dims_arr.tolist())
            if not dims:
                dims = (1, 1)
            continue
        if name == "":
            name = bytes(subpayload).decode("utf-8", errors="ignore").rstrip("\x00")
            continue
        if real_part is None:
            data_mi_type = int(dtype)
            real_part = _parse_numeric_array(subpayload, mi_type=data_mi_type, dims=dims)
            continue
        if is_complex and imag_part is None:
            imag_part = _parse_numeric_array(subpayload, mi_type=int(dtype), dims=dims)
            continue
    if mx_class not in MX_CLASS_DEFAULT_MI:
        raise ValueError(f"unsupported MAT mxClass {mx_class}")
    if real_part is None or dims is None:
        raise ValueError("missing real payload in miMATRIX")
    if data_mi_type is None:
        data_mi_type = int(MX_CLASS_DEFAULT_MI[int(mx_class)])
    arr = real_part
    if is_complex:
        if imag_part is None:
            raise ValueError("complex MAT variable missing imaginary payload")
        arr = arr.astype(np.complex64) + 1j * imag_part.astype(np.complex64)
    return name, np.asarray(arr)


def load_mat_v5(path: Path) -> dict[str, np.ndarray]:
    try:
        mats = sio.loadmat(path)
        return {
            str(k): np.asarray(v)
            for k, v in mats.items()
            if not str(k).startswith("__")
        }
    except Exception:
        pass

    raw = path.read_bytes()
    if len(raw) < 128:
        raise ValueError(f"{path} is too small to be a MAT-v5 file")
    endian = raw[126:128]
    if endian != b"IM":
        raise ValueError(f"{path} is not little-endian MAT-v5 (endian tag={endian!r})")
    vars_out: dict[str, np.ndarray] = {}

    def _walk(buf: bytes) -> None:
        for dtype, payload in _iter_mat_elements(buf):
            if dtype == MI_COMPRESSED:
                _walk(zlib.decompress(payload))
            elif dtype == MI_MATRIX:
                name, arr = _parse_matrix(payload)
                if name:
                    vars_out[name] = arr

    _walk(raw[128:])
    return vars_out


def _normalize_for_display(x: np.ndarray, *, qlo: float = 0.02, qhi: float = 0.995) -> np.ndarray:
    arr = np.asarray(x, dtype=np.float32)
    vals = arr[np.isfinite(arr)]
    if vals.size == 0:
        return np.zeros_like(arr, dtype=np.float32)
    lo = float(np.quantile(vals, qlo))
    hi = float(np.quantile(vals, qhi))
    if not np.isfinite(lo) or not np.isfinite(hi) or hi <= lo:
        hi = lo + 1.0
    return np.clip((arr - lo) / (hi - lo), 0.0, 1.0).astype(np.float32, copy=False)


def _fpr_tag(fpr: float) -> str:
    return f"{float(fpr):.0e}"


def _crop_bbox(*masks: np.ndarray, pad: int = 6) -> tuple[int, int, int, int]:
    union = np.zeros_like(np.asarray(masks[0], dtype=bool))
    for mask in masks:
        union |= np.asarray(mask, dtype=bool)
    yy, xx = np.where(union)
    if yy.size == 0 or xx.size == 0:
        h, w = union.shape
        return 0, h, 0, w
    y0 = max(0, int(yy.min()) - int(pad))
    y1 = min(int(union.shape[0]), int(yy.max()) + 1 + int(pad))
    x0 = max(0, int(xx.min()) - int(pad))
    x1 = min(int(union.shape[1]), int(xx.max()) + 1 + int(pad))
    return y0, y1, x0, x1


def _safe_corr(a: np.ndarray, b: np.ndarray) -> float | None:
    aa = np.asarray(a, dtype=np.float64).ravel()
    bb = np.asarray(b, dtype=np.float64).ravel()
    keep = np.isfinite(aa) & np.isfinite(bb)
    if int(np.count_nonzero(keep)) < 8:
        return None
    aa = aa[keep]
    bb = bb[keep]
    if float(np.std(aa)) == 0.0 or float(np.std(bb)) == 0.0:
        return None
    return float(np.corrcoef(aa, bb)[0, 1])


def _derive_deep_fus_masks(
    reference_pd: np.ndarray,
    *,
    support_quantile: float,
    vessel_quantile: float,
    background_quantile: float,
    edge_margin: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict[str, Any]]:
    ref = np.asarray(reference_pd, dtype=np.float64)
    finite = np.isfinite(ref)
    if not finite.any():
        raise ValueError("Deep-FUS reference map has no finite values")
    positive = ref[finite & (ref > 0.0)]
    support = finite.copy()
    support_threshold = None
    if positive.size >= 512:
        raw_thr = float(np.quantile(positive, float(np.clip(support_quantile, 0.0, 0.95))))
        try:
            otsu_thr = float(threshold_otsu(np.log10(np.maximum(positive, np.finfo(np.float32).tiny))))
            otsu_thr = float(10.0**otsu_thr)
            support_threshold = max(raw_thr, otsu_thr)
        except Exception:
            support_threshold = raw_thr
        support = finite & (ref >= float(support_threshold))
        support = ndi.binary_opening(support, iterations=1)
        support = ndi.binary_closing(support, iterations=2)
        support = ndi.binary_fill_holes(support)
        if edge_margin > 0:
            support[:edge_margin, :] = False
            support[-edge_margin:, :] = False
            support[:, :edge_margin] = False
            support[:, -edge_margin:] = False
        if int(support.sum()) < 512:
            support = finite.copy()

    mask_flow, mask_bg, qc = _derive_structural_masks(
        ref,
        support_mask=support,
        vessel_quantile=float(vessel_quantile),
        bg_quantile=float(background_quantile),
        erode_iters=1,
        guard_dilate_iters=2,
        edge_margin=int(edge_margin),
        vessel_mask_mode="area",
        peak_size=3,
        peak_dilate_iters=1,
        background_mask_mode="global_low",
        shell_inner_dilate_iters=4,
        shell_outer_dilate_iters=10,
    )
    qc = dict(qc)
    qc["support_threshold"] = None if support_threshold is None else float(support_threshold)
    qc["support_pixels"] = int(support.sum())
    return support.astype(bool), mask_flow.astype(bool), mask_bg.astype(bool), qc


def _plot_representative_case(
    *,
    ref_pd: np.ndarray,
    pd_map: np.ndarray,
    fixed_map: np.ndarray,
    support_mask: np.ndarray,
    mask_flow: np.ndarray,
    mask_bg: np.ndarray,
    pd_metrics: dict[str, Any],
    fixed_metrics: dict[str, Any],
    variant_label: str,
    out_path: Path,
    title_prefix: str,
) -> None:
    try:
        import matplotlib.pyplot as plt
    except Exception as exc:  # pragma: no cover
        raise RuntimeError(f"matplotlib required for Deep-FUS figure: {exc}") from exc

    def _det(score: np.ndarray, metrics: dict[str, Any]) -> np.ndarray:
        thr = metrics.get(f"thr@{_fpr_tag(1e-2)}")
        if thr is None or not np.isfinite(float(thr)):
            return np.zeros_like(score, dtype=bool)
        return np.asarray(score >= float(thr), dtype=bool)

    pd_det = _det(pd_map, pd_metrics)
    fixed_det = _det(fixed_map, fixed_metrics)
    y0, y1, x0, x1 = _crop_bbox(support_mask, mask_flow, mask_bg, pd_det, fixed_det, pad=8)

    bg_ref = _normalize_for_display(np.log10(np.maximum(ref_pd, np.finfo(np.float32).tiny)))
    bg_pd = _normalize_for_display(np.log10(np.maximum(pd_map, np.finfo(np.float32).tiny)))
    bg_fixed = _normalize_for_display(np.asarray(fixed_map, dtype=np.float32))

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(1, 3, figsize=(11.0, 3.8), constrained_layout=True)
    panels = [
        (bg_ref, np.zeros_like(mask_flow, dtype=bool), "Deep-FUS reference"),
        (bg_pd, pd_det, "Baseline power Doppler"),
        (bg_fixed, fixed_det, variant_label),
    ]
    for ax, (bg_img, det_mask, title) in zip(axes, panels):
        ax.imshow(bg_img[y0:y1, x0:x1], cmap="magma", interpolation="nearest")
        ax.contour(
            support_mask[y0:y1, x0:x1].astype(np.uint8),
            levels=[0.5],
            colors=["#f8fafc"],
            linewidths=0.8,
        )
        ax.contour(
            mask_flow[y0:y1, x0:x1].astype(np.uint8),
            levels=[0.5],
            colors=["#00d5ff"],
            linewidths=1.2,
        )
        ax.contour(
            mask_bg[y0:y1, x0:x1].astype(np.uint8),
            levels=[0.5],
            colors=["#ffb000"],
            linewidths=1.0,
            linestyles="--",
        )
        if np.any(det_mask):
            overlay = np.zeros((*det_mask[y0:y1, x0:x1].shape, 4), dtype=np.float32)
            overlay[..., 0] = 1.0
            overlay[..., 1] = 0.15
            overlay[..., 2] = 0.15
            overlay[..., 3] = 0.55 * det_mask[y0:y1, x0:x1].astype(np.float32)
            ax.imshow(overlay, interpolation="nearest")
        ax.set_title(title, fontsize=10.5, fontweight="bold")
        ax.set_xticks([])
        ax.set_yticks([])

    axes[1].text(
        0.5,
        -0.14,
        f"AUC {float(pd_metrics['auc']):.3f} | TPR@1e-2 {float(pd_metrics[f'tpr@{_fpr_tag(1e-2)}']):.3f}",
        transform=axes[1].transAxes,
        ha="center",
        va="top",
        fontsize=8.8,
    )
    axes[2].text(
        0.5,
        -0.14,
        f"AUC {float(fixed_metrics['auc']):.3f} | TPR@1e-2 {float(fixed_metrics[f'tpr@{_fpr_tag(1e-2)}']):.3f}",
        transform=axes[2].transAxes,
        ha="center",
        va="top",
        fontsize=8.8,
    )
    fig.suptitle(title_prefix, fontsize=11.5, fontweight="bold")
    fig.savefig(out_path, bbox_inches="tight", pad_inches=0.04)
    if out_path.suffix.lower() == ".pdf":
        fig.savefig(out_path.with_suffix(".png"), dpi=250, bbox_inches="tight", pad_inches=0.04)
    plt.close(fig)


def _score_summary(rows: list[dict[str, Any]], key: str) -> dict[str, Any]:
    vals = np.asarray(
        [float(r[key]) for r in rows if r.get(key) is not None and np.isfinite(r.get(key))],
        dtype=np.float64,
    )
    if vals.size == 0:
        return {"mean": None, "min": None, "max": None, "n": 0}
    return {
        "mean": float(np.mean(vals)),
        "min": float(np.min(vals)),
        "max": float(np.max(vals)),
        "n": int(vals.size),
    }


def _write_rows_csv(csv_path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def _build_summary(
    *,
    rows: list[dict[str, Any]],
    data_dir: Path,
    baseline_type: str,
    variant: str,
    gamma: float,
    mask_qc: dict[str, Any],
    complete: bool,
) -> dict[str, Any]:
    return {
        "data_dir": str(data_dir),
        "n_files": int(len(rows)),
        "baseline_type": str(baseline_type),
        "stap_detector_variant": str(variant),
        "stap_whiten_gamma": float(gamma),
        "complete": bool(complete),
        "notes": [
            "Deep-FUS x is sparse compounded beamformed data, not the paper's clutter-suppressed complex IQ format.",
            "This is a bounded same-cube structural comparison against the released reference power-Doppler images y.",
            "prf_hz is a placeholder visualization/evaluation parameter in this exploratory run, not recovered from release metadata.",
        ],
        "aggregate": {
            "auc_pd": _score_summary(rows, "auc_pd"),
            "auc_fixed": _score_summary(rows, "auc_fixed"),
            "delta_auc": _score_summary(rows, "delta_auc"),
            "tpr1e2_pd": _score_summary(rows, "tpr1e2_pd"),
            "tpr1e2_fixed": _score_summary(rows, "tpr1e2_fixed"),
            "corr_pd_ref": _score_summary(rows, "corr_pd_ref"),
            "corr_fixed_ref": _score_summary(rows, "corr_fixed_ref"),
            "improved_auc_files": int(
                sum(
                    1
                    for r in rows
                    if r.get("delta_auc") is not None and np.isfinite(r["delta_auc"]) and float(r["delta_auc"]) > 0.0
                )
            ),
        },
        "mask_qc_last_file": mask_qc,
    }


def _variant_label(variant: str) -> str:
    raw = str(variant).strip().lower()
    if raw == "unwhitened_ratio":
        return "Fixed matched-subspace head"
    if raw == "whitened_power":
        return "Whitened-power variant"
    if raw == "msd_ratio":
        return "Whitened matched-subspace variant"
    if raw == "adaptive_guard":
        return "Adaptive guard variant"
    return raw.replace("_", " ").title()


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(
        description="Bounded fixed-head exploration on the public Deep-FUS sample test set."
    )
    ap.add_argument(
        "--data-dir",
        type=Path,
        default=ROOT / "external" / "deep-fus" / "data" / "unpacked" / "test",
    )
    ap.add_argument("--pattern", type=str, default="fr*.mat")
    ap.add_argument("--max-files", type=int, default=8)
    ap.add_argument("--lt", type=int, default=64)
    ap.add_argument("--tile-h", type=int, default=8)
    ap.add_argument("--tile-w", type=int, default=8)
    ap.add_argument("--tile-stride", type=int, default=3)
    ap.add_argument("--prf-hz", type=float, default=1000.0)
    ap.add_argument("--baseline-type", type=str, default="mc_svd")
    ap.add_argument("--svd-energy-frac", type=float, default=0.975)
    ap.add_argument("--stap-detector-variant", type=str, default="unwhitened_ratio")
    ap.add_argument("--stap-whiten-gamma", type=float, default=None)
    ap.add_argument("--reg-enable", action="store_true")
    ap.add_argument("--reg-subpixel", type=int, default=4)
    ap.add_argument("--stap-device", type=str, default="cuda")
    ap.add_argument("--support-quantile", type=float, default=0.05)
    ap.add_argument("--vessel-quantile", type=float, default=0.92)
    ap.add_argument("--background-quantile", type=float, default=0.50)
    ap.add_argument("--edge-margin", type=int, default=2)
    ap.add_argument(
        "--out-root",
        type=Path,
        default=ROOT / "tmp" / "deep_fus_fixed_head_explore",
    )
    return ap.parse_args()


def main() -> int:
    args = parse_args()
    variant = str(args.stap_detector_variant).strip().lower()
    gamma = (
        float(args.stap_whiten_gamma)
        if args.stap_whiten_gamma is not None
        else (0.0 if variant == "unwhitened_ratio" else 1.0)
    )
    data_dir = Path(args.data_dir)
    files = sorted(data_dir.glob(str(args.pattern)))
    if not files:
        raise FileNotFoundError(f"no Deep-FUS .mat files found under {data_dir}")
    if int(args.max_files) > 0:
        files = files[: int(args.max_files)]

    out_root = Path(args.out_root)
    out_root.mkdir(parents=True, exist_ok=True)
    csv_path = out_root / "deep_fus_fixed_head_summary.csv"
    json_path = out_root / "deep_fus_fixed_head_summary.json"
    rows: list[dict[str, Any]] = []
    rep_payload: dict[str, Any] | None = None

    for idx, path in enumerate(files, start=1):
        print(f"[{idx}/{len(files)}] loading {path.name}", flush=True)
        mats = load_mat_v5(path)
        if "x" not in mats or "y" not in mats:
            raise KeyError(f"{path} is missing x/y arrays")
        x = np.asarray(mats["x"], dtype=np.float32)
        y = np.asarray(mats["y"], dtype=np.float32)
        if x.ndim != 3 or y.ndim != 2:
            raise ValueError(f"unexpected Deep-FUS shapes for {path.name}: x={x.shape}, y={y.shape}")
        if x.shape[:2] != y.shape:
            raise ValueError(f"spatial shape mismatch for {path.name}: x={x.shape}, y={y.shape}")

        support_mask, mask_flow, mask_bg, mask_qc = _derive_deep_fus_masks(
            y,
            support_quantile=float(args.support_quantile),
            vessel_quantile=float(args.vessel_quantile),
            background_quantile=float(args.background_quantile),
            edge_margin=int(args.edge_margin),
        )

        cube = np.asarray(np.moveaxis(x, -1, 0), dtype=np.complex64)
        with tempfile.TemporaryDirectory(dir=out_root) as td:
            print(f"[{idx}/{len(files)}] bundling {path.name} with {variant}", flush=True)
            bundle = write_acceptance_bundle_from_icube(
                out_root=Path(td),
                dataset_name=path.stem,
                Icube=cube,
                prf_hz=float(args.prf_hz),
                tile_hw=(int(args.tile_h), int(args.tile_w)),
                tile_stride=int(args.tile_stride),
                Lt=min(int(args.lt), int(cube.shape[0]) - 1),
                diag_load=0.07,
                cov_estimator="scm",
                stap_device=str(args.stap_device),
                baseline_type=str(args.baseline_type),
                reg_enable=bool(args.reg_enable),
                reg_subpixel=max(1, int(args.reg_subpixel)),
                svd_energy_frac=float(args.svd_energy_frac),
                run_stap=True,
                stap_detector_variant=variant,
                stap_whiten_gamma=float(gamma),
                score_ka_v2_enable=False,
                stap_conditional_enable=False,
                flow_mask_mode="pd_auto",
                mask_flow_override=mask_flow,
                mask_bg_override=mask_bg,
                band_ratio_flow_low_hz=10.0,
                band_ratio_flow_high_hz=150.0,
                band_ratio_alias_center_hz=350.0,
                band_ratio_alias_width_hz=150.0,
            )
            scores = _load_bundle_scores(
                Path(bundle["meta"]).parent,
                stap_variant=variant,
                include_postka=False,
            )

        pd_map = np.asarray(scores["pd"], dtype=np.float32)
        fixed_map = np.asarray(scores["matched_subspace"], dtype=np.float32)
        pd_metrics = _evaluate_window_score(pd_map, mask_flow, mask_bg, fprs=[1e-2, 1e-3], tpr_targets=[0.5])
        fixed_metrics = _evaluate_window_score(
            fixed_map,
            mask_flow,
            mask_bg,
            fprs=[1e-2, 1e-3],
            tpr_targets=[0.5],
        )
        row = {
            "file": path.name,
            "n_frames": int(cube.shape[0]),
            "auc_pd": pd_metrics["auc"],
            "auc_fixed": fixed_metrics["auc"],
            "delta_auc": (
                None
                if pd_metrics["auc"] is None or fixed_metrics["auc"] is None
                else float(fixed_metrics["auc"]) - float(pd_metrics["auc"])
            ),
            "tpr1e2_pd": pd_metrics[f"tpr@{_fpr_tag(1e-2)}"],
            "tpr1e2_fixed": fixed_metrics[f"tpr@{_fpr_tag(1e-2)}"],
            "tpr1e3_pd": pd_metrics[f"tpr@{_fpr_tag(1e-3)}"],
            "tpr1e3_fixed": fixed_metrics[f"tpr@{_fpr_tag(1e-3)}"],
            "fpr50_pd": pd_metrics["fpr_at_tpr50"],
            "fpr50_fixed": fixed_metrics["fpr_at_tpr50"],
            "corr_pd_ref": _safe_corr(pd_map[support_mask], y[support_mask]),
            "corr_fixed_ref": _safe_corr(fixed_map[support_mask], y[support_mask]),
            "support_pixels": int(support_mask.sum()),
            "flow_pixels": int(mask_flow.sum()),
            "bg_pixels": int(mask_bg.sum()),
            }
        rows.append(row)
        _write_rows_csv(csv_path, rows)
        json_path.write_text(
            json.dumps(
                _build_summary(
                    rows=rows,
                    data_dir=data_dir,
                    baseline_type=str(args.baseline_type),
                    variant=variant,
                    gamma=float(gamma),
                    mask_qc=mask_qc,
                    complete=False,
                ),
                indent=2,
                sort_keys=True,
            )
        )

        if rep_payload is None:
            rep_payload = {
                "file": path.name,
                "ref_pd": y,
                "pd_map": pd_map,
                "fixed_map": fixed_map,
                "support_mask": support_mask,
                "mask_flow": mask_flow,
                "mask_bg": mask_bg,
                "pd_metrics": pd_metrics,
                "fixed_metrics": fixed_metrics,
            }
        print(
            f"[{idx}/{len(files)}] {path.name}: "
            f"AUC pd={row['auc_pd']:.3f} fixed={row['auc_fixed']:.3f} "
            f"delta={row['delta_auc']:+.3f}",
            flush=True,
        )

    deltas = np.asarray(
        [float(r["delta_auc"]) for r in rows if r.get("delta_auc") is not None and np.isfinite(r["delta_auc"])],
        dtype=np.float64,
    )
    target_delta = float(np.median(deltas)) if deltas.size else None
    if target_delta is not None and rep_payload is not None and len(rows) == 1:
        _plot_representative_case(
            ref_pd=np.asarray(rep_payload["ref_pd"], dtype=np.float32),
            pd_map=np.asarray(rep_payload["pd_map"], dtype=np.float32),
            fixed_map=np.asarray(rep_payload["fixed_map"], dtype=np.float32),
            support_mask=np.asarray(rep_payload["support_mask"], dtype=bool),
            mask_flow=np.asarray(rep_payload["mask_flow"], dtype=bool),
            mask_bg=np.asarray(rep_payload["mask_bg"], dtype=bool),
            pd_metrics=dict(rep_payload["pd_metrics"]),
            fixed_metrics=dict(rep_payload["fixed_metrics"]),
            variant_label=_variant_label(variant),
            out_path=out_root / "deep_fus_fixed_head_maps.pdf",
            title_prefix=f"Deep-FUS sample set: representative file {rows[0]['file']}",
        )
    elif target_delta is not None and rep_payload is not None:
        best_idx = 0
        best_gap = math.inf
        for ii, row in enumerate(rows):
            val = row.get("delta_auc")
            if val is None or not np.isfinite(val):
                continue
            gap = abs(float(val) - target_delta)
            if gap < best_gap:
                best_gap = gap
                best_idx = ii
        chosen = rows[best_idx]["file"]
        chosen_mats = load_mat_v5(data_dir / chosen)
        chosen_x = np.asarray(chosen_mats["x"], dtype=np.float32)
        chosen_y = np.asarray(chosen_mats["y"], dtype=np.float32)
        support_mask, mask_flow, mask_bg, _ = _derive_deep_fus_masks(
            chosen_y,
            support_quantile=float(args.support_quantile),
            vessel_quantile=float(args.vessel_quantile),
            background_quantile=float(args.background_quantile),
            edge_margin=int(args.edge_margin),
        )
        cube = np.asarray(np.moveaxis(chosen_x, -1, 0), dtype=np.complex64)
        with tempfile.TemporaryDirectory(dir=out_root) as td:
            bundle = write_acceptance_bundle_from_icube(
                out_root=Path(td),
                dataset_name=Path(chosen).stem,
                Icube=cube,
                prf_hz=float(args.prf_hz),
                tile_hw=(int(args.tile_h), int(args.tile_w)),
                tile_stride=int(args.tile_stride),
                Lt=min(int(args.lt), int(cube.shape[0]) - 1),
                diag_load=0.07,
                cov_estimator="scm",
                stap_device=str(args.stap_device),
                baseline_type=str(args.baseline_type),
                reg_enable=bool(args.reg_enable),
                reg_subpixel=max(1, int(args.reg_subpixel)),
                svd_energy_frac=float(args.svd_energy_frac),
                run_stap=True,
                stap_detector_variant=variant,
                stap_whiten_gamma=float(gamma),
                score_ka_v2_enable=False,
                stap_conditional_enable=False,
                flow_mask_mode="pd_auto",
                mask_flow_override=mask_flow,
                mask_bg_override=mask_bg,
                band_ratio_flow_low_hz=10.0,
                band_ratio_flow_high_hz=150.0,
                band_ratio_alias_center_hz=350.0,
                band_ratio_alias_width_hz=150.0,
            )
            scores = _load_bundle_scores(
                Path(bundle["meta"]).parent,
                stap_variant=variant,
                include_postka=False,
            )
        pd_map = np.asarray(scores["pd"], dtype=np.float32)
        fixed_map = np.asarray(scores["matched_subspace"], dtype=np.float32)
        pd_metrics = _evaluate_window_score(pd_map, mask_flow, mask_bg, fprs=[1e-2, 1e-3], tpr_targets=[0.5])
        fixed_metrics = _evaluate_window_score(fixed_map, mask_flow, mask_bg, fprs=[1e-2, 1e-3], tpr_targets=[0.5])
        _plot_representative_case(
            ref_pd=chosen_y,
            pd_map=pd_map,
            fixed_map=fixed_map,
            support_mask=support_mask,
            mask_flow=mask_flow,
            mask_bg=mask_bg,
            pd_metrics=pd_metrics,
            fixed_metrics=fixed_metrics,
            variant_label=_variant_label(variant),
            out_path=out_root / "deep_fus_fixed_head_maps.pdf",
            title_prefix=f"Deep-FUS sample set: representative file {chosen}",
        )

    _write_rows_csv(csv_path, rows)
    json_path.write_text(
        json.dumps(
            _build_summary(
                rows=rows,
                data_dir=data_dir,
                baseline_type=str(args.baseline_type),
                variant=variant,
                gamma=float(gamma),
                mask_qc=mask_qc,
                complete=True,
            ),
            indent=2,
            sort_keys=True,
        )
    )
    print(f"Wrote {csv_path}")
    print(f"Wrote {json_path}")
    figure_path = out_root / "deep_fus_fixed_head_maps.pdf"
    if figure_path.exists():
        print(f"Wrote {figure_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
