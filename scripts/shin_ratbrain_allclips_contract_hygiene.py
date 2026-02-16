from __future__ import annotations

import argparse
import csv
import json
import subprocess
from dataclasses import asdict
from pathlib import Path
from typing import Any

import numpy as np

from pipeline.realdata.shin_ratbrain import list_shin_files, load_shin_iq, load_shin_metadata
from sim.kwave.icube_bundle import write_acceptance_bundle_from_icube


def _parse_slice(spec: str) -> tuple[list[int] | None, str]:
    spec = (spec or "").strip()
    if spec in {"", "all", ":", "0:"}:
        return None, "all"
    parts = spec.split(":")
    if len(parts) not in {2, 3}:
        raise ValueError(f"Invalid slice spec {spec!r}; expected 'start:stop[:step]' or 'all'.")
    start = int(parts[0]) if parts[0] else 0
    stop = int(parts[1]) if parts[1] else None
    step = int(parts[2]) if len(parts) == 3 and parts[2] else 1
    if stop is None:
        raise ValueError("Slice spec must include stop (e.g. 0:128).")
    frames = list(range(start, stop, step))
    tag = f"f{start}_{stop}"
    return frames, tag


def _parse_frames_list(spec: str) -> list[str]:
    items = [s.strip() for s in (spec or "").split(",") if s.strip()]
    if not items:
        return ["0:128"]
    return items


def _safe_quantile(x: np.ndarray, q: float) -> float | None:
    x = np.asarray(x, dtype=np.float64)
    x = x[np.isfinite(x)]
    if x.size == 0:
        return None
    return float(np.quantile(x, q))


def _connected_components(binary: np.ndarray, connectivity: int = 4) -> int:
    binary = np.asarray(binary, dtype=bool)
    if binary.size == 0:
        return 0
    try:
        import scipy.ndimage as ndi  # type: ignore

        if connectivity == 8:
            structure = np.ones((3, 3), dtype=int)
        else:
            structure = np.array([[0, 1, 0], [1, 1, 1], [0, 1, 0]], dtype=int)
        _, n = ndi.label(binary, structure=structure)
        return int(n)
    except Exception:
        H, W = binary.shape
        visited = np.zeros_like(binary, dtype=bool)
        n = 0
        neigh = [(-1, 0), (1, 0), (0, -1), (0, 1)]
        if connectivity == 8:
            neigh += [(-1, -1), (-1, 1), (1, -1), (1, 1)]
        for y in range(H):
            for x in range(W):
                if not binary[y, x] or visited[y, x]:
                    continue
                n += 1
                stack = [(y, x)]
                visited[y, x] = True
                while stack:
                    cy, cx = stack.pop()
                    for dy, dx in neigh:
                        ny, nx = cy + dy, cx + dx
                        if 0 <= ny < H and 0 <= nx < W and binary[ny, nx] and not visited[ny, nx]:
                            visited[ny, nx] = True
                            stack.append((ny, nx))
        return int(n)


def _profile_to_bands(profile: str) -> tuple[float, float, float, float]:
    """Return (flow_low, flow_high, alias_center, alias_half_width)."""
    p = (profile or "").strip().lower()
    if p in {"u", "shin_u", "ulm"}:
        return 60.0, 250.0, 400.0, 100.0
    if p in {"s", "shin_s", "strict"}:
        return 20.0, 200.0, 380.0, 120.0
    if p in {"l", "shin_l", "low"}:
        return 10.0, 120.0, 330.0, 170.0
    raise ValueError("Profile must be one of: U, S, L.")


def _format_energy_frac(e: float) -> str:
    return f"{float(e):.3f}"


def _unzip_all_iqdata(*, zip_path: Path, data_root: Path) -> None:
    if not zip_path.is_file():
        raise FileNotFoundError(f"zip not found: {zip_path}")
    data_root.mkdir(parents=True, exist_ok=True)
    cmd = [
        "unzip",
        "-o",
        "-j",
        str(zip_path),
        "RatBrain_Fig3/IQData*.dat",
        "RatBrain_Fig3/SizeInfo.dat",
        "-d",
        str(data_root),
    ]
    subprocess.run(cmd, check=True)


def _iter_iq_files(data_root: Path, *, iq_files: str) -> list[Path]:
    spec = (iq_files or "").strip()
    if spec in {"", "all"}:
        return list_shin_files(data_root)
    names = [s.strip() for s in spec.split(",") if s.strip()]
    out: list[Path] = []
    for name in names:
        path = data_root / name
        if not path.is_file():
            raise FileNotFoundError(f"IQ file not found: {path}")
        out.append(path)
    return out


def _load_npy(bundle_dir: Path, name: str) -> np.ndarray | None:
    path = bundle_dir / f"{name}.npy"
    if not path.is_file():
        return None
    return np.load(path, allow_pickle=False)


def _compute_hygiene_row(
    *,
    bundle_dir: Path,
    meta: dict[str, Any],
    bg_tail_quantile: float,
    connectivity: int,
) -> dict[str, Any] | None:
    pd_post = _load_npy(bundle_dir, "pd_stap")
    pd_base = _load_npy(bundle_dir, "pd_base")
    mask_flow = _load_npy(bundle_dir, "mask_flow")
    mask_bg = _load_npy(bundle_dir, "mask_bg")
    if pd_post is None or mask_flow is None:
        return None
    if mask_bg is None:
        mask_bg = ~mask_flow.astype(bool)

    ka_scale = _load_npy(bundle_dir, "ka_scale_map")
    pd_pre = _load_npy(bundle_dir, "pd_stap_pre_ka")
    if pd_pre is None:
        # Best-effort: undo scaling if available; otherwise pre==post.
        if ka_scale is not None:
            scale = np.asarray(ka_scale, dtype=np.float32)
            pd_pre = (pd_post.astype(np.float32) / np.maximum(scale, 1e-12)).astype(np.float32, copy=False)
        else:
            pd_pre = pd_post.astype(np.float32, copy=False)

    flow = np.asarray(mask_flow, dtype=bool)
    bg = np.asarray(mask_bg, dtype=bool)
    s_pre = -np.asarray(pd_pre, dtype=np.float64)
    s_post = -np.asarray(pd_post, dtype=np.float64)

    q99_pre = _safe_quantile(s_pre[bg], 0.99)
    q99_post = _safe_quantile(s_post[bg], 0.99)
    q999_pre = _safe_quantile(s_pre[bg], 0.999)
    q999_post = _safe_quantile(s_post[bg], 0.999)

    thr = _safe_quantile(s_pre[bg], float(bg_tail_quantile))
    if thr is None:
        return None

    hit_pre = bg & (s_pre >= float(thr))
    hit_post = bg & (s_post >= float(thr))
    area_pre = int(np.sum(hit_pre))
    area_post = int(np.sum(hit_post))
    clust_pre = _connected_components(hit_pre, connectivity=int(connectivity))
    clust_post = _connected_components(hit_post, connectivity=int(connectivity))

    q_hi_protect = None
    ka_v2 = meta.get("ka_contract_v2") or {}
    if isinstance(ka_v2, dict):
        cfg = ka_v2.get("config") or {}
        if isinstance(cfg, dict):
            try:
                q_hi_protect = float(cfg.get("q_hi_protect", 0.995))
            except Exception:
                q_hi_protect = None
    prot = flow.copy()
    if q_hi_protect is not None:
        finite = np.isfinite(s_pre)
        if finite.any():
            thr_hi = float(np.quantile(s_pre[finite], q_hi_protect))
            prot |= s_pre >= thr_hi
    max_abs_delta_prot = float(np.max(np.abs(pd_post[prot] - pd_pre[prot]))) if prot.any() else 0.0

    scaled_frac = None
    scaled_frac_on_flow = None
    scaled_frac_on_bg = None
    scaled_frac_on_prot = None
    if ka_scale is not None:
        ka_scale = np.asarray(ka_scale, dtype=np.float32)
        scaled = ka_scale > (1.0 + 1e-6)
        scaled_frac = float(np.mean(scaled))
        scaled_frac_on_flow = float(np.mean(scaled[flow])) if flow.any() else 0.0
        scaled_frac_on_bg = float(np.mean(scaled[bg])) if bg.any() else 0.0
        scaled_frac_on_prot = float(np.mean(scaled[prot])) if prot.any() else 0.0

    tele = meta.get("stap_fallback_telemetry") or {}
    ka_metrics = (ka_v2.get("metrics") or {}) if isinstance(ka_v2, dict) else {}

    row = {
        "bundle": str(bundle_dir),
        "bundle_name": bundle_dir.name,
        "prf_hz": meta.get("prf_hz"),
        "Lt": meta.get("Lt"),
        "frames": meta.get("total_frames"),
        "pd_base_mean": float(np.mean(pd_base)) if pd_base is not None else None,
        "pd_pre_mean": float(np.mean(pd_pre)),
        "pd_post_mean": float(np.mean(pd_post)),
        "bg_tail_quantile": float(bg_tail_quantile),
        "bg_score_q99_pre": q99_pre,
        "bg_score_q99_post": q99_post,
        "bg_score_q999_pre": q999_pre,
        "bg_score_q999_post": q999_post,
        "bg_tail_thr": float(thr),
        "bg_tail_area_pre": area_pre,
        "bg_tail_area_post": area_post,
        "bg_tail_clusters_pre": clust_pre,
        "bg_tail_clusters_post": clust_post,
        "max_abs_pd_delta_prot": max_abs_delta_prot,
        "ka_state": ka_v2.get("state") if isinstance(ka_v2, dict) else None,
        "ka_reason": ka_v2.get("reason") if isinstance(ka_v2, dict) else None,
        "ka_risk_mode": ka_metrics.get("risk_mode") if isinstance(ka_metrics, dict) else None,
        "ka_p_shrink": ka_metrics.get("p_shrink") if isinstance(ka_metrics, dict) else None,
        "ka_pf_peak_nonbg": ka_metrics.get("pf_peak_nonbg") if isinstance(ka_metrics, dict) else None,
        "ka_pf_peak_flow": ka_metrics.get("pf_peak_flow") if isinstance(ka_metrics, dict) else None,
        "ka_n_flow_proxy": ka_metrics.get("n_flow_proxy") if isinstance(ka_metrics, dict) else None,
        "ka_uplift_veto_pf_peak_reason": ka_metrics.get("uplift_veto_pf_peak_reason")
        if isinstance(ka_metrics, dict)
        else None,
        "ka_guard_q90": ka_metrics.get("guard_q90") if isinstance(ka_metrics, dict) else None,
        "score_ka_v2_applied": tele.get("score_ka_v2_applied") if isinstance(tele, dict) else None,
        "scaled_px_frac": tele.get("score_ka_v2_scaled_pixel_fraction") if isinstance(tele, dict) else None,
        "scaled_frac_total": scaled_frac,
        "scaled_frac_on_flow": scaled_frac_on_flow,
        "scaled_frac_on_bg": scaled_frac_on_bg,
        "scaled_frac_on_prot": scaled_frac_on_prot,
    }
    return row


def _summarize_rows(rows: list[dict[str, Any]]) -> dict[str, Any]:
    def _counts(key: str) -> dict[str, int]:
        out: dict[str, int] = {}
        for r in rows:
            v = r.get(key)
            if v is None:
                continue
            out[str(v)] = out.get(str(v), 0) + 1
        return dict(sorted(out.items(), key=lambda kv: (-kv[1], kv[0])))

    def _median(key: str) -> float | None:
        vals = [r.get(key) for r in rows]
        x = np.array([v for v in vals if isinstance(v, (int, float)) and np.isfinite(v)], dtype=float)
        if x.size == 0:
            return None
        return float(np.median(x))

    def _mean(key: str) -> float | None:
        vals = [r.get(key) for r in rows]
        x = np.array([v for v in vals if isinstance(v, (int, float)) and np.isfinite(v)], dtype=float)
        if x.size == 0:
            return None
        return float(np.mean(x))

    return {
        "n_bundles": int(len(rows)),
        "ka_state_counts": _counts("ka_state"),
        "ka_reason_counts": _counts("ka_reason"),
        "ka_risk_mode_counts": _counts("ka_risk_mode"),
        "median_pf_peak_flow": _median("ka_pf_peak_flow"),
        "median_guard_q90": _median("ka_guard_q90"),
        "median_p_shrink": _median("ka_p_shrink"),
        "median_scaled_px_frac": _median("scaled_px_frac"),
        "mean_bg_tail_clusters_pre": _mean("bg_tail_clusters_pre"),
        "mean_bg_tail_clusters_post": _mean("bg_tail_clusters_post"),
        "mean_bg_tail_area_pre": _mean("bg_tail_area_pre"),
        "mean_bg_tail_area_post": _mean("bg_tail_area_post"),
        "max_max_abs_pd_delta_prot": float(
            np.max([r.get("max_abs_pd_delta_prot", 0.0) for r in rows if r.get("max_abs_pd_delta_prot") is not None])
        )
        if rows
        else 0.0,
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Run Shin RatBrain Fig3 bundles over many IQData clips and compute label-free\n"
            "contract + hygiene metrics (C0/C1/C2 counts, background-tail clusters/area,\n"
            "and protected-set invariance). Intended to scale beyond 'n=10 representative clips'."
        )
    )
    parser.add_argument(
        "--data-root",
        type=Path,
        default=Path("data/shin_zenodo_10711806/ratbrain_fig3_raw"),
        help="Directory containing SizeInfo.dat and IQData*.dat (default: %(default)s).",
    )
    parser.add_argument(
        "--zip-path",
        type=Path,
        default=Path("data/shin_zenodo_10711806/RatBrain_Fig3.zip"),
        help="Optional zip path for extracting missing IQData files (default: %(default)s).",
    )
    parser.add_argument(
        "--extract-from-zip",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Extract IQData*.dat and SizeInfo.dat from --zip-path into --data-root (default: %(default)s).",
    )
    parser.add_argument(
        "--expected-clips",
        type=int,
        default=80,
        help="Expected number of IQData clips in the dataset (default: %(default)s).",
    )
    parser.add_argument(
        "--iq-files",
        type=str,
        default="all",
        help="Comma-separated IQData*.dat file names or 'all' (default: %(default)s).",
    )
    parser.add_argument(
        "--frames-list",
        type=str,
        default="0:128",
        help="Comma-separated frame slices start:stop[:step] (default: %(default)s).",
    )
    parser.add_argument("--prf-hz", type=float, default=1000.0)
    parser.add_argument("--profile", type=str, default="U", help="Frozen Shin profile: U, S, or L (default: %(default)s).")
    parser.add_argument("--Lt", type=int, default=64)
    parser.add_argument("--tile-h", type=int, default=8)
    parser.add_argument("--tile-w", type=int, default=8)
    parser.add_argument("--tile-stride", type=int, default=3)

    parser.add_argument("--baseline-type", type=str, default="mc_svd", choices=["mc_svd", "svd_bandpass"])
    parser.add_argument("--svd-rank", type=str, default="none", help="For mc_svd: int or 'none' (energy-frac).")
    parser.add_argument("--svd-energy-frac", type=float, default=0.97)
    parser.add_argument("--svd-keep-min", type=int, default=3)
    parser.add_argument("--svd-keep-max", type=int, default=40)
    parser.add_argument(
        "--reg-enable",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Enable rigid phase-correlation registration in baselines (default: %(default)s).",
    )
    parser.add_argument("--reg-subpixel", type=int, default=4)

    parser.add_argument("--flow-mask-mode", type=str, default="pd_auto")
    parser.add_argument("--flow-mask-pd-quantile", type=float, default=0.99)
    parser.add_argument("--flow-mask-min-pixels", type=int, default=64)
    parser.add_argument("--flow-mask-union-default", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--flow-mask-depth-min-frac", type=float, default=0.0)
    parser.add_argument("--flow-mask-depth-max-frac", type=float, default=1.0)
    parser.add_argument("--flow-mask-erode-iters", type=int, default=0)
    parser.add_argument("--flow-mask-dilate-iters", type=int, default=1)

    parser.add_argument(
        "--score-ka-v2",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Enable score-space KA v2 (shrink-only) when contract permits (default: %(default)s).",
    )
    parser.add_argument(
        "--score-ka-v2-mode",
        type=str,
        default="auto",
        help="KA v2 application mode: safety|uplift|auto (default: %(default)s).",
    )
    parser.add_argument(
        "--run-stap",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Whether to run STAP when building bundles (default: %(default)s).",
    )

    parser.add_argument("--bg-tail-quantile", type=float, default=0.999)
    parser.add_argument("--connectivity", type=int, default=4, choices=[4, 8])
    parser.add_argument(
        "--out-root",
        type=Path,
        default=None,
        help="Output root for bundles (default: derived from profile/frames).",
    )
    parser.add_argument(
        "--out-csv",
        type=Path,
        default=None,
        help="Output CSV for per-bundle hygiene metrics (default: derived from profile/frames).",
    )
    parser.add_argument(
        "--out-json",
        type=Path,
        default=None,
        help="Output JSON summary path (default: derived from profile/frames).",
    )
    parser.add_argument(
        "--resume",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Skip bundles already present under out_root (default: %(default)s).",
    )
    parser.add_argument("--max-files", type=int, default=None, help="Optional limit for debugging.")
    args = parser.parse_args()

    data_root = Path(args.data_root)
    if args.extract_from_zip:
        _unzip_all_iqdata(zip_path=Path(args.zip_path), data_root=data_root)

    info = load_shin_metadata(data_root)
    iq_paths = _iter_iq_files(data_root, iq_files=str(args.iq_files))
    if args.expected_clips and len(iq_paths) < int(args.expected_clips):
        msg = (
            f"[shin-allclips] found {len(iq_paths)} IQData*.dat under {data_root}, expected {int(args.expected_clips)}.\n"
            "If you have the full archive, run with --extract-from-zip (or manually unzip RatBrain_Fig3.zip)."
        )
        print(msg)

    if args.max_files is not None:
        iq_paths = iq_paths[: int(args.max_files)]

    frames_specs = _parse_frames_list(str(args.frames_list))
    profile_tag = str(args.profile).strip().upper()
    flow_low_hz, flow_high_hz, alias_center_hz, alias_hw_hz = _profile_to_bands(profile_tag)
    baseline_type = str(args.baseline_type).strip().lower()

    svd_rank: int | None
    if baseline_type == "mc_svd":
        if str(args.svd_rank).strip().lower() in {"none", "auto", "energy", "ef"}:
            svd_rank = None
        else:
            svd_rank = int(args.svd_rank)
        svd_tag = f"e{_format_energy_frac(float(args.svd_energy_frac))}" if svd_rank is None else f"r{int(svd_rank)}"
    else:
        svd_rank = None
        svd_tag = f"k{int(args.svd_keep_min)}_{int(args.svd_keep_max)}"

    out_root = args.out_root
    out_csv = args.out_csv
    out_json = args.out_json
    if out_root is None or out_csv is None or out_json is None:
        # Derive stable default paths from key settings.
        frame_tag = frames_specs[0].replace(":", "_").replace(",", "_")
        run_tag = f"shin_ratbrain_{profile_tag}_{svd_tag}_Lt{int(args.Lt)}_ka_{str(args.score_ka_v2_mode)}_{frame_tag}_allclips"
        out_root = out_root or Path("runs") / run_tag
        out_csv = out_csv or Path("reports") / f"{run_tag}_hygiene.csv"
        out_json = out_json or Path("reports") / f"{run_tag}_summary.json"

    out_root = Path(out_root)
    out_csv = Path(out_csv)
    out_json = Path(out_json)
    out_root.mkdir(parents=True, exist_ok=True)
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    out_json.parent.mkdir(parents=True, exist_ok=True)

    rows: list[dict[str, Any]] = []
    wrote_header = False
    with out_csv.open("w", newline="") as f_csv:
        writer: csv.DictWriter | None = None

        for frame_spec in frames_specs:
            frames, frame_tag = _parse_slice(frame_spec)
            for iq_path in iq_paths:
                dataset_name = f"shin_{iq_path.stem}_{frame_tag}_p{profile_tag}_Lt{int(args.Lt)}_{baseline_type}_{svd_tag}"
                bundle_dir = out_root / dataset_name
                meta_path = bundle_dir / "meta.json"
                if bool(args.resume) and meta_path.is_file():
                    meta = json.loads(meta_path.read_text())
                else:
                    Icube = load_shin_iq(iq_path, info, frames=frames)
                    meta_extra = {
                        "orig_data": {
                            "dataset": "ShinRatBrain_Fig3",
                            "iq_file": str(iq_path),
                            "sizeinfo": asdict(info),
                            "frames_spec": frame_spec,
                        },
                        "shin_profile": {
                            "name": profile_tag,
                            "flow_low_hz": flow_low_hz,
                            "flow_high_hz": flow_high_hz,
                            "alias_center_hz": alias_center_hz,
                            "alias_half_width_hz": alias_hw_hz,
                        },
                    }
                    paths = write_acceptance_bundle_from_icube(
                        out_root=out_root,
                        dataset_name=dataset_name,
                        Icube=Icube,
                        prf_hz=float(args.prf_hz),
                        tile_hw=(int(args.tile_h), int(args.tile_w)),
                        tile_stride=int(args.tile_stride),
                        Lt=int(args.Lt),
                        baseline_type=baseline_type,
                        svd_rank=svd_rank,
                        svd_energy_frac=float(args.svd_energy_frac),
                        svd_keep_min=int(args.svd_keep_min),
                        svd_keep_max=int(args.svd_keep_max),
                        flow_mask_mode=str(args.flow_mask_mode),
                        flow_mask_pd_quantile=float(args.flow_mask_pd_quantile),
                        flow_mask_depth_min_frac=float(args.flow_mask_depth_min_frac),
                        flow_mask_depth_max_frac=float(args.flow_mask_depth_max_frac),
                        flow_mask_erode_iters=int(args.flow_mask_erode_iters),
                        flow_mask_dilate_iters=int(args.flow_mask_dilate_iters),
                        flow_mask_min_pixels=int(args.flow_mask_min_pixels),
                        flow_mask_union_default=bool(args.flow_mask_union_default),
                        band_ratio_flow_low_hz=float(flow_low_hz),
                        band_ratio_flow_high_hz=float(flow_high_hz),
                        band_ratio_alias_center_hz=float(alias_center_hz),
                        band_ratio_alias_width_hz=float(alias_hw_hz),
                        run_stap=bool(args.run_stap),
                        reg_enable=bool(args.reg_enable),
                        reg_subpixel=int(args.reg_subpixel),
                        score_mode="pd",
                        cov_estimator="tyler_pca",
                        score_ka_v2_enable=bool(args.score_ka_v2),
                        score_ka_v2_mode=str(args.score_ka_v2_mode),
                        meta_extra=meta_extra,
                    )
                    meta = json.loads(Path(paths["meta"]).read_text())

                row = _compute_hygiene_row(
                    bundle_dir=bundle_dir,
                    meta=meta,
                    bg_tail_quantile=float(args.bg_tail_quantile),
                    connectivity=int(args.connectivity),
                )
                if row is None:
                    continue
                row["iq_file"] = iq_path.name
                row["frames_spec"] = frame_spec
                row["profile"] = profile_tag
                row["baseline_type"] = baseline_type
                row["svd_tag"] = svd_tag
                row["reg_enable"] = bool(args.reg_enable)
                row["score_ka_v2_mode"] = str(args.score_ka_v2_mode)

                if writer is None:
                    writer = csv.DictWriter(f_csv, fieldnames=list(row.keys()))
                    writer.writeheader()
                    wrote_header = True
                writer.writerow(row)
                rows.append(row)

                if len(rows) % 10 == 0:
                    print(f"[shin-allclips] processed bundles={len(rows)} (latest: {bundle_dir.name})")

    if not wrote_header:
        print("[shin-allclips] no rows written; check data-root and settings")
        return

    summary = _summarize_rows(rows)
    out_json.write_text(json.dumps(summary, indent=2))
    print(f"[shin-allclips] wrote {out_csv}")
    print(f"[shin-allclips] wrote {out_json}")
    print(f"[shin-allclips] summary: n={summary.get('n_bundles')} state_counts={summary.get('ka_state_counts')}")


if __name__ == "__main__":
    main()
