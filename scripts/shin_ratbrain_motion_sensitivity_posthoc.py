from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import scipy.ndimage as ndi

from sim.kwave.common import _phasecorr_shift


def _to_float(x: Any) -> float:
    try:
        return float(x)
    except Exception:
        return float("nan")


def _parse_float_list(spec: str) -> list[float]:
    out: list[float] = []
    for part in (spec or "").split(","):
        part = part.strip()
        if not part:
            continue
        out.append(float(part))
    if not out:
        raise ValueError("Expected a non-empty comma-separated float list.")
    return out


def _parse_int_list(spec: str) -> list[int]:
    out: list[int] = []
    for part in (spec or "").split(","):
        part = part.strip()
        if not part:
            continue
        out.append(int(part))
    if not out:
        raise ValueError("Expected a non-empty comma-separated int list.")
    return out


def _parse_amp_from_bundle_name(name: str) -> float | None:
    # Naming convention: ..._a{amp_tag}, with amp_tag like 0p00, 0p50, 2p00, m1p00.
    if "_a" not in name:
        return None
    tag = name.split("_a")[-1]
    tag = tag.replace("m", "-").replace("p", ".")
    try:
        return float(tag)
    except Exception:
        return None


def _pearson_corr(a: np.ndarray, b: np.ndarray) -> float:
    a = np.asarray(a, dtype=np.float64).ravel()
    b = np.asarray(b, dtype=np.float64).ravel()
    finite = np.isfinite(a) & np.isfinite(b)
    a = a[finite]
    b = b[finite]
    if a.size == 0:
        return float("nan")
    a0 = a - float(np.mean(a))
    b0 = b - float(np.mean(b))
    denom = float(np.sqrt(np.sum(a0 * a0) * np.sum(b0 * b0))) + 1e-12
    return float(np.sum(a0 * b0) / denom)


def _summarize(vals: list[float]) -> dict[str, float]:
    arr = np.asarray([v for v in vals if np.isfinite(v)], dtype=np.float64)
    if arr.size == 0:
        return {"median": float("nan"), "q25": float("nan"), "q75": float("nan")}
    return {
        "median": float(np.quantile(arr, 0.5)),
        "q25": float(np.quantile(arr, 0.25)),
        "q75": float(np.quantile(arr, 0.75)),
    }


def _load_map(bundle_dir: Path, name: str) -> np.ndarray | None:
    path = bundle_dir / f"{name}.npy"
    if not path.is_file():
        return None
    return np.load(path, allow_pickle=False)


def _iter_bundles(run_root: Path) -> list[Path]:
    out: list[Path] = []
    for meta_path in run_root.glob("*/meta.json"):
        if meta_path.is_file():
            out.append(meta_path.parent)
    out.sort()
    return out


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Post-hoc sensitivity analysis for Shin motion sweeps.\n"
            "Recomputes corr vs no-motion from existing bundles under different crop margins\n"
            "and map-alignment settings (no pipeline reruns)."
        )
    )
    parser.add_argument("--run-root", type=Path, required=True, help="Directory containing motion sweep bundles (meta.json).")
    parser.add_argument("--amps", type=str, default="0,0.5,1,2", help="Amplitudes to include (default: %(default)s).")
    parser.add_argument(
        "--crop-margins",
        type=str,
        default="0,4,8,12",
        help="Comma-separated crop margins in pixels (default: %(default)s).",
    )
    parser.add_argument(
        "--align-maps-list",
        type=str,
        default="0,1",
        help="Comma-separated align flags (0/1) indicating whether to phase-correlate-align maps (default: %(default)s).",
    )
    parser.add_argument("--out-csv", type=Path, required=True, help="Output CSV summary path.")
    args = parser.parse_args()

    amps = sorted(set(float(a) for a in _parse_float_list(args.amps)))
    crop_margins = sorted(set(int(m) for m in _parse_int_list(args.crop_margins)))
    align_flags = sorted(set(int(v) for v in _parse_int_list(args.align_maps_list)))
    align_flags = [0 if v <= 0 else 1 for v in align_flags]

    bundles = _iter_bundles(args.run_root)
    if not bundles:
        raise SystemExit(f"No bundles found under {args.run_root}")

    # Group by IQData file and amplitude.
    by_file: dict[str, dict[float, Path]] = defaultdict(dict)
    for bundle_dir in bundles:
        meta_path = bundle_dir / "meta.json"
        meta = json.loads(meta_path.read_text())
        orig = meta.get("orig_data") or {}
        iq_file = str(orig.get("iq_file") or "")
        iq_name = Path(iq_file).name if iq_file else bundle_dir.name
        amp = _parse_amp_from_bundle_name(bundle_dir.name)
        if amp is None:
            continue
        by_file[iq_name][float(amp)] = bundle_dir

    rows_out: list[dict[str, Any]] = []
    for amp in amps:
        for margin in crop_margins:
            for align_flag in align_flags:
                corr_base_list: list[float] = []
                corr_stap_list: list[float] = []
                for iq_name, by_amp in by_file.items():
                    if 0.0 not in by_amp or float(amp) not in by_amp:
                        continue
                    ref_dir = by_amp[0.0]
                    cur_dir = by_amp[float(amp)]
                    ref_base = _load_map(ref_dir, "pd_base")
                    ref_pre = _load_map(ref_dir, "pd_stap_pre_ka") or _load_map(ref_dir, "pd_stap")
                    cur_base = _load_map(cur_dir, "pd_base")
                    cur_pre = _load_map(cur_dir, "pd_stap_pre_ka") or _load_map(cur_dir, "pd_stap")
                    if ref_base is None or ref_pre is None or cur_base is None or cur_pre is None:
                        continue

                    H, W = ref_base.shape
                    if 2 * margin >= H or 2 * margin >= W:
                        sl = (slice(0, H), slice(0, W))
                    else:
                        sl = (slice(margin, H - margin), slice(margin, W - margin))

                    base_map = np.asarray(cur_base, dtype=np.float32)
                    pre_map = np.asarray(cur_pre, dtype=np.float32)
                    ref_base_map = np.asarray(ref_base, dtype=np.float32)
                    ref_pre_map = np.asarray(ref_pre, dtype=np.float32)

                    if align_flag:
                        dyb, dxb, _ = _phasecorr_shift(ref_base_map, base_map, upsample=4)
                        dys, dxs, _ = _phasecorr_shift(ref_pre_map, pre_map, upsample=4)
                        base_map = ndi.shift(base_map, shift=(dyb, dxb), order=1, mode="nearest", prefilter=False)
                        pre_map = ndi.shift(pre_map, shift=(dys, dxs), order=1, mode="nearest", prefilter=False)

                    corr_base_list.append(_pearson_corr(base_map[sl], ref_base_map[sl]))
                    corr_stap_list.append(_pearson_corr(pre_map[sl], ref_pre_map[sl]))

                summ_base = _summarize(corr_base_list)
                summ_stap = _summarize(corr_stap_list)
                frac_gt = float(
                    np.mean(
                        np.asarray(corr_stap_list, dtype=np.float64)
                        > np.asarray(corr_base_list, dtype=np.float64)
                    )
                ) if corr_base_list and corr_stap_list and len(corr_base_list) == len(corr_stap_list) else float("nan")

                rows_out.append(
                    {
                        "amp_px": float(amp),
                        "crop_margin_px": int(margin),
                        "align_maps": bool(align_flag),
                        "n_files": int(min(len(corr_base_list), len(corr_stap_list))),
                        "corr_base_median": float(summ_base["median"]),
                        "corr_base_q25": float(summ_base["q25"]),
                        "corr_base_q75": float(summ_base["q75"]),
                        "corr_stap_median": float(summ_stap["median"]),
                        "corr_stap_q25": float(summ_stap["q25"]),
                        "corr_stap_q75": float(summ_stap["q75"]),
                        "corr_frac_stap_gt_base": frac_gt,
                    }
                )

    if not rows_out:
        raise SystemExit("No summary rows produced; check amps/crops and available bundles.")

    args.out_csv.parent.mkdir(parents=True, exist_ok=True)
    with args.out_csv.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows_out[0].keys()))
        writer.writeheader()
        writer.writerows(rows_out)
    print(f"[shin-motion-sens] wrote {args.out_csv} ({len(rows_out)} rows)")


if __name__ == "__main__":
    main()

