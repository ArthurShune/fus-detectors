from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

import numpy as np


def _load_meta(meta_path: Path) -> dict[str, Any] | None:
    try:
        meta = json.loads(meta_path.read_text())
    except Exception:
        return None
    if not isinstance(meta, dict):
        return None
    return meta


def _iter_bundle_dirs(root: Path) -> list[Path]:
    out: list[Path] = []
    for meta_path in root.rglob("meta.json"):
        if meta_path.is_file():
            out.append(meta_path.parent)
    out.sort()
    return out


def _load_npy(bundle_dir: Path, name: str) -> np.ndarray | None:
    path = bundle_dir / f"{name}.npy"
    if not path.is_file():
        return None
    return np.load(path, allow_pickle=False)


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
        # Fallback: trivial scan (slow but fine for these map sizes).
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


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Compute label-free KA v2 \"map hygiene\" metrics on Shin RatBrain bundles.\n"
            "These metrics avoid ROC/labels: they measure background tail compression,\n"
            "background cluster counts, and protected-set invariance before/after KA."
        )
    )
    parser.add_argument("--root", type=Path, required=True, help="Root directory containing bundles.")
    parser.add_argument("--out-csv", type=Path, required=True, help="Output CSV path.")
    parser.add_argument(
        "--bg-tail-quantile",
        type=float,
        default=0.999,
        help="Quantile for background tail threshold on score S=-PD (default: %(default)s).",
    )
    parser.add_argument(
        "--connectivity",
        type=int,
        default=4,
        choices=[4, 8],
        help="Connectivity for cluster count (default: %(default)s).",
    )
    args = parser.parse_args()

    bundle_dirs = _iter_bundle_dirs(args.root)
    rows: list[dict[str, Any]] = []

    for bundle_dir in bundle_dirs:
        meta_path = bundle_dir / "meta.json"
        meta = _load_meta(meta_path)
        if meta is None:
            continue

        pd_post = _load_npy(bundle_dir, "pd_stap")
        pd_base = _load_npy(bundle_dir, "pd_base")
        mask_flow = _load_npy(bundle_dir, "mask_flow")
        mask_bg = _load_npy(bundle_dir, "mask_bg")
        if pd_post is None or mask_flow is None:
            continue
        if mask_bg is None:
            mask_bg = ~mask_flow.astype(bool)

        ka_scale = _load_npy(bundle_dir, "ka_scale_map")
        pd_pre = _load_npy(bundle_dir, "pd_stap_pre_ka")
        if pd_pre is None:
            if ka_scale is not None:
                eps = 1e-12
                pd_pre = pd_post.astype(np.float32, copy=False) / np.maximum(
                    ka_scale.astype(np.float32, copy=False), eps
                )
            else:
                pd_pre = pd_post

        pd_pre = np.asarray(pd_pre, dtype=np.float32)
        pd_post = np.asarray(pd_post, dtype=np.float32)
        bg = mask_bg.astype(bool)
        flow = mask_flow.astype(bool)

        # Score convention matches PD score_mode in hab_contract_check.py
        # (higher = more flow-like): S = -PD.
        s_pre = -pd_pre
        s_post = -pd_post

        q99_pre = _safe_quantile(s_pre[bg], 0.99)
        q999_pre = _safe_quantile(s_pre[bg], 0.999)
        q99_post = _safe_quantile(s_post[bg], 0.99)
        q999_post = _safe_quantile(s_post[bg], 0.999)

        thr = _safe_quantile(s_pre[bg], float(args.bg_tail_quantile))
        if thr is None:
            continue

        hit_pre = bg & (s_pre >= float(thr))
        hit_post = bg & (s_post >= float(thr))
        area_pre = int(np.sum(hit_pre))
        area_post = int(np.sum(hit_post))
        clust_pre = _connected_components(hit_pre, connectivity=int(args.connectivity))
        clust_post = _connected_components(hit_post, connectivity=int(args.connectivity))

        # Protected-set invariance (mask_flow + top-score pixels, if possible).
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

        # Metadata fields (best-effort).
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
            "bg_tail_quantile": float(args.bg_tail_quantile),
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
        rows.append(row)

    out_path = args.out_csv
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(rows[0].keys()) if rows else []
    with out_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)

    applied = sum(bool(r.get("score_ka_v2_applied")) for r in rows)
    print(f"[shin-hygiene] bundles={len(rows)} score_ka_v2_applied={applied}")
    print(f"[shin-hygiene] wrote {out_path}")


if __name__ == "__main__":
    main()
