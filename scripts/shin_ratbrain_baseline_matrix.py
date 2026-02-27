from __future__ import annotations

import argparse
import csv
import json
import os
from dataclasses import asdict
from pathlib import Path
from typing import Any

import numpy as np

from pipeline.realdata.shin_ratbrain import load_shin_iq, load_shin_metadata
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


def _parse_list(spec: str) -> list[str]:
    return [s.strip() for s in (spec or "").split(",") if s.strip()]


def _default_shin_subset10() -> list[str]:
    return [
        "IQData001.dat",
        "IQData002.dat",
        "IQData003.dat",
        "IQData004.dat",
        "IQData005.dat",
        "IQData010.dat",
        "IQData020.dat",
        "IQData040.dat",
        "IQData060.dat",
        "IQData080.dat",
    ]


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
        # Fallback: simple flood-fill (OK at these map sizes).
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


def _safe_quantile(x: np.ndarray, q: float) -> float | None:
    x = np.asarray(x, dtype=np.float64)
    x = x[np.isfinite(x)]
    if x.size == 0:
        return None
    return float(np.quantile(x, q))


def _auc_pos_vs_neg(pos: np.ndarray, neg: np.ndarray) -> float | None:
    """
    AUC = P{S_pos > S_neg} + 0.5 P{S_pos == S_neg}.

    Computes the probability-of-superiority (Mann-Whitney) without external deps.
    """
    pos = np.asarray(pos, dtype=np.float64)
    neg = np.asarray(neg, dtype=np.float64)
    pos = pos[np.isfinite(pos)]
    neg = neg[np.isfinite(neg)]
    if pos.size == 0 or neg.size == 0:
        return None
    neg_sorted = np.sort(neg)
    less = np.searchsorted(neg_sorted, pos, side="left")
    right = np.searchsorted(neg_sorted, pos, side="right")
    equal = right - less
    auc = (float(np.sum(less)) + 0.5 * float(np.sum(equal))) / float(pos.size * neg.size)
    return float(auc)


def _tail_metrics(
    *,
    score: np.ndarray,
    mask_flow: np.ndarray,
    mask_bg: np.ndarray,
    alphas: list[float],
    connectivity: int,
) -> dict[str, float | int | None]:
    score = np.asarray(score, dtype=np.float64)
    flow = np.asarray(mask_flow, dtype=bool)
    bg = np.asarray(mask_bg, dtype=bool)

    out: dict[str, float | int | None] = {}
    n_flow = int(flow.sum())
    n_bg = int(bg.sum())
    out["n_flow"] = n_flow
    out["n_bg"] = n_bg
    out["fpr_min"] = (1.0 / float(n_bg)) if n_bg > 0 else None
    out["bg_q99"] = _safe_quantile(score[bg], 0.99) if n_bg > 0 else None
    out["bg_q999"] = _safe_quantile(score[bg], 0.999) if n_bg > 0 else None

    if n_bg <= 0:
        for a in alphas:
            tag = f"{a:.0e}"
            out[f"thr_{tag}"] = None
            out[f"fpr_{tag}"] = None
            out[f"hit_flow_{tag}"] = None
            out[f"bg_area_{tag}"] = None
            out[f"bg_clusters_{tag}"] = None
        return out

    bg_scores = score[bg]
    bg_scores = bg_scores[np.isfinite(bg_scores)]
    if bg_scores.size == 0:
        return out

    for alpha in alphas:
        alpha = float(alpha)
        tag = f"{alpha:.0e}"
        q = float(np.clip(1.0 - alpha, 0.0, 1.0))
        thr = float(np.quantile(bg_scores, q))
        hit_bg = bg & (score >= thr)
        hit_flow = flow & (score >= thr)
        fpr = float(np.mean(hit_bg[bg])) if bg.any() else None
        hit_flow_rate = float(np.mean(hit_flow[flow])) if flow.any() else None
        out[f"thr_{tag}"] = float(thr)
        out[f"fpr_{tag}"] = fpr
        out[f"hit_flow_{tag}"] = hit_flow_rate
        out[f"bg_area_{tag}"] = int(np.sum(hit_bg))
        out[f"bg_clusters_{tag}"] = _connected_components(hit_bg, connectivity=connectivity)
    return out


def _load_npy(path: str | Path) -> np.ndarray:
    return np.load(str(path), allow_pickle=False)


def main() -> None:
    ap = argparse.ArgumentParser(
        description=(
            "Run a baseline+STAP scenario matrix on Shin RatBrain Fig3 and summarize label-free tail metrics.\n"
            "Outputs are written as acceptance bundles under --out-root (same bundle schema as other runs)."
        )
    )
    ap.add_argument(
        "--data-root",
        type=Path,
        default=Path("data/shin_zenodo_10711806/ratbrain_fig3_raw"),
        help="Directory containing SizeInfo.dat and IQData*.dat (default: %(default)s).",
    )
    ap.add_argument(
        "--iq-files",
        type=str,
        default="subset10",
        help=(
            "Comma-separated IQData file names, or 'subset10' (default), or 'all'. "
            "Example: IQData001.dat,IQData010.dat"
        ),
    )
    ap.add_argument(
        "--frames-list",
        type=str,
        default="0:128,64:192,122:250",
        help="Comma-separated frame slices (default: %(default)s).",
    )
    ap.add_argument("--prf-hz", type=float, default=1000.0)
    ap.add_argument(
        "--profile",
        type=str,
        default="U",
        help="Band profile for Pf/Pa telemetry: U|S|L (default: %(default)s).",
    )
    ap.add_argument("--lt", type=int, default=64, help="STAP aperture Lt (default: %(default)s).")
    ap.add_argument("--tile-h", type=int, default=8)
    ap.add_argument("--tile-w", type=int, default=8)
    ap.add_argument("--tile-stride", type=int, default=3)
    ap.add_argument("--diag-load", type=float, default=0.07)
    ap.add_argument(
        "--kappa-shrink",
        type=float,
        default=200.0,
        help="Fast-path covariance shrinkage target kappa (default: %(default)s).",
    )
    ap.add_argument(
        "--kappa-msd",
        type=float,
        default=200.0,
        help="Fast-path whitening/loading target kappa (default: %(default)s).",
    )
    ap.add_argument("--cov-estimator", type=str, default="tyler_pca")
    ap.add_argument(
        "--fast-path",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Enable batched STAP fast path (STAP_FAST_PATH=1) (default: %(default)s).",
    )
    ap.add_argument(
        "--fast-pd-only",
        action=argparse.BooleanOptionalAction,
        default=False,
        help=(
            "Enable PD-only fast path (STAP_FAST_PD_ONLY=1). "
            "This is faster but does not return a meaningful STAP detector score (default: %(default)s)."
        ),
    )
    ap.add_argument(
        "--snapshot-stride",
        type=int,
        default=4,
        help="STAP_SNAPSHOT_STRIDE (training support throttle) (default: %(default)s).",
    )
    ap.add_argument(
        "--max-snapshots",
        type=int,
        default=64,
        help="STAP_MAX_SNAPSHOTS (training support cap after striding) (default: %(default)s).",
    )
    ap.add_argument(
        "--tyler-tol",
        type=float,
        default=None,
        help="Optional STAP_TYLER_TOL override for latency sweeps (default: unset; uses implementation default).",
    )
    ap.add_argument(
        "--tyler-max-iter",
        type=int,
        default=None,
        help="Optional STAP_TYLER_MAX_ITER override for latency sweeps (default: unset; uses implementation default).",
    )
    ap.add_argument(
        "--tyler-early-stop",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Control STAP_TYLER_EARLY_STOP (default: %(default)s).",
    )
    ap.add_argument(
        "--stap-device",
        type=str,
        default=None,
        help="STAP device: cpu, cuda, cuda:0, ... (default: auto).",
    )
    ap.add_argument(
        "--stap-conditional-enable",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Enable conditional STAP compute gating (default: %(default)s).",
    )
    ap.add_argument(
        "--reg-enable",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Enable rigid registration for baselines that support it (default: %(default)s).",
    )
    ap.add_argument("--reg-subpixel", type=int, default=4)
    ap.add_argument("--reg-reference", type=str, default="median")
    ap.add_argument(
        "--svd-energy-frac",
        type=float,
        default=0.97,
        help="MC-SVD baseline energy fraction removed (default: %(default)s).",
    )
    ap.add_argument("--rpca-max-iters", type=int, default=250)
    ap.add_argument(
        "--run-rpca",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Whether to run RPCA baseline bundles (default: %(default)s).",
    )
    ap.add_argument("--hosvd-spatial-downsample", type=int, default=2)
    ap.add_argument(
        "--hosvd-energy-fracs",
        type=str,
        default="0.99,0.99,0.99",
        help="HOSVD energy fracs as 'fT,fH,fW' (default: %(default)s).",
    )
    ap.add_argument(
        "--run-hosvd",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Whether to run HOSVD baseline bundles (default: %(default)s).",
    )
    ap.add_argument(
        "--alphas",
        type=str,
        default="1e-1,1e-2,3e-3,1e-3,3e-4,1e-4",
        help="Comma-separated background tail rates, e.g. '1e-4,3e-4,1e-3' (default: %(default)s).",
    )
    ap.add_argument("--connectivity", type=int, default=4, choices=[4, 8])
    ap.add_argument(
        "--out-root",
        type=Path,
        default=Path("runs/shin_ratbrain_baseline_matrix"),
        help="Output root for acceptance bundles (default: %(default)s).",
    )
    ap.add_argument(
        "--out-csv",
        type=Path,
        default=Path("reports/shin_ratbrain_baseline_matrix.csv"),
        help="Output CSV path (default: %(default)s).",
    )
    ap.add_argument(
        "--out-json",
        type=Path,
        default=Path("reports/shin_ratbrain_baseline_matrix_summary.json"),
        help="Output JSON summary (default: %(default)s).",
    )
    ap.add_argument(
        "--resume",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Skip bundle creation if meta.json already exists (default: %(default)s).",
    )
    ap.add_argument(
        "--flow-mask-union-default",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Union the pd_auto flow mask with the generic default mask (default: %(default)s).",
    )
    ap.add_argument(
        "--run-stap-raw",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Whether to run the STAP-only (raw IQ) bundle (default: %(default)s).",
    )
    args = ap.parse_args()

    # Performance knobs (used heavily in the repo's replay harnesses).
    if bool(args.fast_path):
        os.environ["STAP_FAST_PATH"] = "1"
    else:
        os.environ.pop("STAP_FAST_PATH", None)
    if bool(args.fast_pd_only):
        os.environ["STAP_FAST_PD_ONLY"] = "1"
    else:
        os.environ.pop("STAP_FAST_PD_ONLY", None)
    os.environ["STAP_SNAPSHOT_STRIDE"] = str(max(1, int(args.snapshot_stride)))
    os.environ["STAP_MAX_SNAPSHOTS"] = str(max(1, int(args.max_snapshots)))
    os.environ["STAP_KAPPA_SHRINK"] = str(float(args.kappa_shrink))
    os.environ["STAP_KAPPA_MSD"] = str(float(args.kappa_msd))
    if args.tyler_tol is not None:
        os.environ["STAP_TYLER_TOL"] = str(float(args.tyler_tol))
    else:
        os.environ.pop("STAP_TYLER_TOL", None)
    if args.tyler_max_iter is not None:
        os.environ["STAP_TYLER_MAX_ITER"] = str(int(args.tyler_max_iter))
    else:
        os.environ.pop("STAP_TYLER_MAX_ITER", None)
    if bool(args.tyler_early_stop):
        os.environ.pop("STAP_TYLER_EARLY_STOP", None)
    else:
        os.environ["STAP_TYLER_EARLY_STOP"] = "0"

    info = load_shin_metadata(args.data_root)
    flow_low_hz, flow_high_hz, alias_center_hz, alias_hw_hz = _profile_to_bands(str(args.profile))

    # Scenario matrix: files x frame windows.
    iq_spec = str(args.iq_files).strip()
    if iq_spec.lower() in {"subset10", "paper10"}:
        iq_files = _default_shin_subset10()
    elif iq_spec.lower() == "all":
        iq_files = sorted(p.name for p in args.data_root.glob("IQData*.dat"))
    else:
        iq_files = _parse_list(iq_spec)
    if not iq_files:
        raise ValueError("No IQ files selected.")

    frame_specs = _parse_list(str(args.frames_list))
    if not frame_specs:
        raise ValueError("No frame slices provided.")

    alphas: list[float] = []
    for part in _parse_list(str(args.alphas)):
        alphas.append(float(part))
    if not alphas:
        alphas = [1e-3]

    # Parse HOSVD energy fracs (optional).
    hosvd_energy_fracs: tuple[float, float, float] | None = None
    ef_spec = str(args.hosvd_energy_fracs or "").strip()
    if ef_spec:
        parts = [p.strip() for p in ef_spec.split(",") if p.strip()]
        if len(parts) != 3:
            raise ValueError("--hosvd-energy-fracs must be 'fT,fH,fW'")
        hosvd_energy_fracs = (float(parts[0]), float(parts[1]), float(parts[2]))

    out_root = Path(args.out_root)
    out_root.mkdir(parents=True, exist_ok=True)
    out_csv = Path(args.out_csv)
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    out_json = Path(args.out_json)
    out_json.parent.mkdir(parents=True, exist_ok=True)

    rows: list[dict[str, Any]] = []

    for frame_spec in frame_specs:
        frames, frame_tag = _parse_slice(frame_spec)
        for iq_file in iq_files:
            iq_path = args.data_root / iq_file
            if not iq_path.is_file():
                print(f"[shin-matrix] missing {iq_path}, skipping")
                continue
            Icube = load_shin_iq(iq_path, info, frames=frames)
            th, tw = int(args.tile_h), int(args.tile_w)

            common_meta_extra = {
                "orig_data": {
                    "dataset": "ShinRatBrain_Fig3",
                    "iq_file": str(iq_path),
                    "sizeinfo": asdict(info),
                    "frames_spec": frame_spec,
                },
                "shin_profile": {
                    "name": str(args.profile),
                    "flow_low_hz": float(flow_low_hz),
                    "flow_high_hz": float(flow_high_hz),
                    "alias_center_hz": float(alias_center_hz),
                    "alias_half_width_hz": float(alias_hw_hz),
                },
            }

            # 1) MC-SVD + STAP bundle (contains both MC-SVD baseline and STAP score maps)
            e_tag = f"e{float(args.svd_energy_frac):.3f}".replace(".", "p")
            dataset_name_stap = (
                f"shin_{iq_path.stem}_{frame_tag}_p{str(args.profile)}_Lt{int(args.lt)}_mcsvd_{e_tag}_stap"
            )
            stap_dir = out_root / dataset_name_stap
            stap_meta_path = stap_dir / "meta.json"
            if bool(args.resume) and stap_meta_path.is_file():
                paths_stap = {
                    "meta": str(stap_meta_path),
                    "mask_flow": str(stap_dir / "mask_flow.npy"),
                    "mask_bg": str(stap_dir / "mask_bg.npy"),
                    "score_base": str(stap_dir / "score_base.npy"),
                    "score_stap_preka": str(stap_dir / "score_stap_preka.npy"),
                }
            else:
                paths_stap = write_acceptance_bundle_from_icube(
                    out_root=out_root,
                    dataset_name=dataset_name_stap,
                    Icube=Icube,
                    prf_hz=float(args.prf_hz),
                    tile_hw=(th, tw),
                    tile_stride=int(args.tile_stride),
                    Lt=int(args.lt),
                    diag_load=float(args.diag_load),
                    cov_estimator=str(args.cov_estimator),
                    baseline_type="mc_svd",
                    svd_energy_frac=float(args.svd_energy_frac),
                    reg_enable=bool(args.reg_enable),
                    reg_subpixel=int(args.reg_subpixel),
                    reg_reference=str(args.reg_reference),
                    run_stap=True,
                    stap_device=args.stap_device,
                    stap_conditional_enable=bool(args.stap_conditional_enable),
                    score_mode="pd",
                    score_ka_v2_enable=False,
	                    band_ratio_flow_low_hz=float(flow_low_hz),
	                    band_ratio_flow_high_hz=float(flow_high_hz),
	                    band_ratio_alias_center_hz=float(alias_center_hz),
	                    band_ratio_alias_width_hz=float(alias_hw_hz),
	                    flow_mask_union_default=bool(args.flow_mask_union_default),
	                    meta_extra=dict(common_meta_extra),
	                )

            mask_flow = _load_npy(paths_stap["mask_flow"]).astype(bool, copy=False)
            mask_bg = _load_npy(paths_stap["mask_bg"]).astype(bool, copy=False)

            # 1b) STAP-only bundle (run STAP directly on raw/registered IQ; masks overridden)
            dataset_name_stap_raw = (
                f"shin_{iq_path.stem}_{frame_tag}_p{str(args.profile)}_Lt{int(args.lt)}_raw_stap"
            )
            stap_raw_dir = out_root / dataset_name_stap_raw
            stap_raw_meta_path = stap_raw_dir / "meta.json"
            if bool(args.run_stap_raw) and not (bool(args.resume) and stap_raw_meta_path.is_file()):
                write_acceptance_bundle_from_icube(
                    out_root=out_root,
                    dataset_name=dataset_name_stap_raw,
                    Icube=Icube,
                    prf_hz=float(args.prf_hz),
                    tile_hw=(th, tw),
                    tile_stride=int(args.tile_stride),
                    Lt=int(args.lt),
                    diag_load=float(args.diag_load),
                    cov_estimator=str(args.cov_estimator),
                    baseline_type="raw",
                    reg_enable=bool(args.reg_enable),
                    reg_subpixel=int(args.reg_subpixel),
                    reg_reference=str(args.reg_reference),
                    run_stap=True,
                    stap_device=args.stap_device,
                    stap_conditional_enable=bool(args.stap_conditional_enable),
                    score_mode="pd",
                    score_ka_v2_enable=False,
                    band_ratio_flow_low_hz=float(flow_low_hz),
                    band_ratio_flow_high_hz=float(flow_high_hz),
                    band_ratio_alias_center_hz=float(alias_center_hz),
                    band_ratio_alias_width_hz=float(alias_hw_hz),
                    mask_flow_override=mask_flow,
                    mask_bg_override=mask_bg,
                    flow_mask_union_default=bool(args.flow_mask_union_default),
                    meta_extra=dict(common_meta_extra),
                )

            # 2) Optional RPCA baseline bundle (baseline only; masks overridden to match MC-SVD proxy)
            rpca_dir: Path | None = None
            if bool(args.run_rpca):
                dataset_name_rpca = (
                    f"shin_{iq_path.stem}_{frame_tag}_p{str(args.profile)}_Lt{int(args.lt)}_rpca_stapoff"
                )
                rpca_dir = out_root / dataset_name_rpca
                rpca_meta_path = rpca_dir / "meta.json"
                if not (bool(args.resume) and rpca_meta_path.is_file()):
                    write_acceptance_bundle_from_icube(
                        out_root=out_root,
                        dataset_name=dataset_name_rpca,
                        Icube=Icube,
                        prf_hz=float(args.prf_hz),
                        tile_hw=(th, tw),
                        tile_stride=int(args.tile_stride),
                        Lt=int(args.lt),
                        diag_load=float(args.diag_load),
                        cov_estimator=str(args.cov_estimator),
                        baseline_type="rpca",
                        rpca_max_iters=int(args.rpca_max_iters),
                        reg_enable=bool(args.reg_enable),
                        reg_subpixel=int(args.reg_subpixel),
                        reg_reference=str(args.reg_reference),
                        run_stap=False,
                        stap_device=args.stap_device,
                        stap_conditional_enable=bool(args.stap_conditional_enable),
                        score_mode="pd",
                        score_ka_v2_enable=False,
                        band_ratio_flow_low_hz=float(flow_low_hz),
                        band_ratio_flow_high_hz=float(flow_high_hz),
                        band_ratio_alias_center_hz=float(alias_center_hz),
                        band_ratio_alias_width_hz=float(alias_hw_hz),
                        mask_flow_override=mask_flow,
                        mask_bg_override=mask_bg,
                        flow_mask_union_default=bool(args.flow_mask_union_default),
                        meta_extra=dict(common_meta_extra),
                    )

            # 3) Optional HOSVD baseline bundle (baseline only; masks overridden)
            hosvd_dir: Path | None = None
            if bool(args.run_hosvd):
                dataset_name_hosvd = (
                    f"shin_{iq_path.stem}_{frame_tag}_p{str(args.profile)}_Lt{int(args.lt)}_hosvd_stapoff"
                )
                hosvd_dir = out_root / dataset_name_hosvd
                hosvd_meta_path = hosvd_dir / "meta.json"
                if not (bool(args.resume) and hosvd_meta_path.is_file()):
                    write_acceptance_bundle_from_icube(
                        out_root=out_root,
                        dataset_name=dataset_name_hosvd,
                        Icube=Icube,
                        prf_hz=float(args.prf_hz),
                        tile_hw=(th, tw),
                        tile_stride=int(args.tile_stride),
                        Lt=int(args.lt),
                        diag_load=float(args.diag_load),
                        cov_estimator=str(args.cov_estimator),
                        baseline_type="hosvd",
                        hosvd_energy_fracs=hosvd_energy_fracs,
                        hosvd_spatial_downsample=int(args.hosvd_spatial_downsample),
                        reg_enable=bool(args.reg_enable),
                        reg_subpixel=int(args.reg_subpixel),
                        reg_reference=str(args.reg_reference),
                        run_stap=False,
                        stap_device=args.stap_device,
                        stap_conditional_enable=bool(args.stap_conditional_enable),
                        score_mode="pd",
                        score_ka_v2_enable=False,
                        band_ratio_flow_low_hz=float(flow_low_hz),
                        band_ratio_flow_high_hz=float(flow_high_hz),
                        band_ratio_alias_center_hz=float(alias_center_hz),
                        band_ratio_alias_width_hz=float(alias_hw_hz),
                        mask_flow_override=mask_flow,
                        mask_bg_override=mask_bg,
                        flow_mask_union_default=bool(args.flow_mask_union_default),
                        meta_extra=dict(common_meta_extra),
                    )

            # ---- Metrics ----
            # Common proxy masks come from MC-SVD baseline (flow mask derived from baseline PD).
            # Baselines use score_base = baseline PD score (right-tail). STAP uses score_stap_preka.
            score_mcsvd = _load_npy(stap_dir / "score_base.npy")
            score_stap = _load_npy(stap_dir / "score_stap_preka.npy")
            score_stap_raw = None
            if bool(args.run_stap_raw):
                score_stap_raw = _load_npy(stap_raw_dir / "score_stap_preka.npy")
            score_rpca = None
            if rpca_dir is not None:
                score_rpca = _load_npy(rpca_dir / "score_base.npy")
            score_hosvd = None
            if hosvd_dir is not None:
                score_hosvd = _load_npy(hosvd_dir / "score_base.npy")

            # Meta runtimes (best effort).
            def _load_ms(bundle_dir: Path) -> tuple[float | None, float | None]:
                try:
                    meta = json.loads((bundle_dir / "meta.json").read_text())
                except Exception:
                    return None, None
                tele = meta.get("stap_fallback_telemetry") or {}
                baseline_ms = tele.get("baseline_ms") if isinstance(tele, dict) else None
                stap_total_ms = tele.get("stap_total_ms") if isinstance(tele, dict) else None
                try:
                    baseline_ms = float(baseline_ms) if baseline_ms is not None else None
                except Exception:
                    baseline_ms = None
                try:
                    stap_total_ms = float(stap_total_ms) if stap_total_ms is not None else None
                except Exception:
                    stap_total_ms = None
                return baseline_ms, stap_total_ms

            ms_mcsvd, ms_stap_total = _load_ms(stap_dir)
            ms_raw = None
            ms_stap_raw_total = None
            if bool(args.run_stap_raw):
                ms_raw, ms_stap_raw_total = _load_ms(stap_raw_dir)
            ms_rpca = None
            if rpca_dir is not None:
                ms_rpca, _ = _load_ms(rpca_dir)
            ms_hosvd = None
            if hosvd_dir is not None:
                ms_hosvd, _ = _load_ms(hosvd_dir)

            common_fields = {
                "iq_file": iq_file,
                "frames_spec": frame_spec,
                "frame_tag": frame_tag,
                "profile": str(args.profile),
                "prf_hz": float(args.prf_hz),
                "Lt": int(args.lt),
                "tile_hw": f"{th}x{tw}",
                "tile_stride": int(args.tile_stride),
                "reg_enable": bool(args.reg_enable),
                "stap_conditional_enable": bool(args.stap_conditional_enable),
                "svd_energy_frac": float(args.svd_energy_frac),
                "rpca_max_iters": int(args.rpca_max_iters),
                "hosvd_spatial_downsample": int(args.hosvd_spatial_downsample),
                "hosvd_energy_fracs": ef_spec,
                "bundle_dir_stap": str(stap_dir),
                "bundle_dir_stap_raw": str(stap_raw_dir) if bool(args.run_stap_raw) else None,
                "bundle_dir_rpca": str(rpca_dir) if rpca_dir is not None else None,
                "bundle_dir_hosvd": str(hosvd_dir) if hosvd_dir is not None else None,
                "tyler_tol": float(args.tyler_tol) if args.tyler_tol is not None else None,
                "tyler_max_iter": int(args.tyler_max_iter) if args.tyler_max_iter is not None else None,
                "tyler_early_stop": bool(args.tyler_early_stop),
                "snapshot_stride": int(args.snapshot_stride),
                "max_snapshots": int(args.max_snapshots),
            }

            methods: list[tuple[str, str, np.ndarray, float | None, float | None]] = [
                ("mc_svd", "MC-SVD (baseline PD)", score_mcsvd, ms_mcsvd, None),
                ("stap", "STAP (matched-subspace)", score_stap, ms_mcsvd, ms_stap_total),
            ]
            if score_rpca is not None:
                methods.insert(1, ("rpca", "RPCA (baseline PD)", score_rpca, ms_rpca, None))
            if score_hosvd is not None:
                insert_idx = 2 if score_rpca is not None else 1
                methods.insert(insert_idx, ("hosvd", "HOSVD (baseline PD)", score_hosvd, ms_hosvd, None))
            if score_stap_raw is not None:
                methods.append(
                    ("stap_raw", "STAP-only (matched-subspace; raw IQ)", score_stap_raw, ms_raw, ms_stap_raw_total)
                )

            for key, label, score_map, baseline_ms, stap_total_ms in methods:
                auc = _auc_pos_vs_neg(score_map[mask_flow], score_map[mask_bg])
                row: dict[str, Any] = {
                    **common_fields,
                    "method_key": key,
                    "method": label,
                    "baseline_ms": baseline_ms,
                    "stap_total_ms": stap_total_ms,
                    "auc_flow_vs_bg": auc,
                }
                row.update(
                    _tail_metrics(
                        score=score_map,
                        mask_flow=mask_flow,
                        mask_bg=mask_bg,
                        alphas=alphas,
                        connectivity=int(args.connectivity),
                    )
                )
                rows.append(row)

            if len(rows) % 20 == 0:
                print(f"[shin-matrix] rows={len(rows)} (latest: {iq_file} {frame_spec})", flush=True)

    if not rows:
        print("[shin-matrix] no rows written; check data-root and inputs")
        return

    # Write CSV (stable field order: first row keys, then any new keys appended).
    fieldnames: list[str] = []
    for row in rows:
        for k in row.keys():
            if k not in fieldnames:
                fieldnames.append(k)
    with out_csv.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for row in rows:
            w.writerow(row)
    print(f"[shin-matrix] wrote {out_csv}")

    # Simple summary JSON.
    summary: dict[str, Any] = {
        "n_rows": len(rows),
        "n_scenarios": int(len({(r.get("iq_file"), r.get("frames_spec")) for r in rows})),
        "methods": sorted({str(r.get('method_key')) for r in rows}),
        "alphas": [float(a) for a in alphas],
        "profile": str(args.profile),
        "prf_hz": float(args.prf_hz),
        "Lt": int(args.lt),
        "tile_hw": [int(args.tile_h), int(args.tile_w)],
        "tile_stride": int(args.tile_stride),
        "diag_load": float(args.diag_load),
        "cov_estimator": str(args.cov_estimator),
        "fast_path": bool(args.fast_path),
        "fast_pd_only": bool(args.fast_pd_only),
        "snapshot_stride": int(args.snapshot_stride),
        "max_snapshots": int(args.max_snapshots),
        "kappa_shrink": float(args.kappa_shrink),
        "kappa_msd": float(args.kappa_msd),
        "flow_mask_union_default": bool(args.flow_mask_union_default),
        "svd_energy_frac": float(args.svd_energy_frac),
        "reg_enable": bool(args.reg_enable),
        "stap_conditional_enable": bool(args.stap_conditional_enable),
        "out_root": str(out_root),
        "out_csv": str(out_csv),
    }
    out_json.write_text(json.dumps(summary, indent=2))
    print(f"[shin-matrix] wrote {out_json}")


if __name__ == "__main__":
    main()
