#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

import numpy as np


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


def _score_metrics(
    score: np.ndarray,
    flow: np.ndarray,
    bg: np.ndarray,
    *,
    alpha: float,
    connectivity: int,
) -> dict[str, float | int | None]:
    score = np.asarray(score, dtype=np.float64)
    flow = np.asarray(flow, dtype=bool)
    bg = np.asarray(bg, dtype=bool)
    bg_vals = score[bg]
    bg_vals = bg_vals[np.isfinite(bg_vals)]
    out: dict[str, float | int | None] = {
        "bg_q99": _safe_quantile(score[bg], 0.99) if np.any(bg) else None,
        "bg_q999": _safe_quantile(score[bg], 0.999) if np.any(bg) else None,
    }
    if bg_vals.size == 0:
        out["thr"] = None
        out["hit_flow"] = None
        out["bg_area"] = None
        out["bg_clusters"] = None
        out["bg_excess_mass"] = None
        out["largest_cluster"] = None
        return out

    thr = float(np.quantile(bg_vals, 1.0 - float(alpha)))
    hit_bg = bg & (score >= thr)
    hit_flow = flow & (score >= thr)
    out["thr"] = thr
    out["hit_flow"] = float(np.mean(hit_flow[flow])) if np.any(flow) else None
    out["bg_area"] = int(np.sum(hit_bg))
    out["bg_clusters"] = int(_connected_components(hit_bg, connectivity=connectivity))
    excess = np.clip(score[bg] - thr, a_min=0.0, a_max=None)
    out["bg_excess_mass"] = float(np.sum(excess[np.isfinite(excess)]))
    if np.any(hit_bg):
        try:
            import scipy.ndimage as ndi  # type: ignore

            if connectivity == 8:
                structure = np.ones((3, 3), dtype=int)
            else:
                structure = np.array([[0, 1, 0], [1, 1, 1], [0, 1, 0]], dtype=int)
            labeled, n = ndi.label(hit_bg, structure=structure)
            if n > 0:
                sizes = ndi.sum(np.ones_like(hit_bg, dtype=np.int32), labeled, index=np.arange(1, n + 1))
                out["largest_cluster"] = int(np.max(sizes))
            else:
                out["largest_cluster"] = 0
        except Exception:
            out["largest_cluster"] = None
    else:
        out["largest_cluster"] = 0
    return out


def _median(rows: list[dict[str, Any]], key: str) -> float | None:
    vals = [r.get(key) for r in rows]
    arr = np.asarray([v for v in vals if isinstance(v, (int, float)) and np.isfinite(v)], dtype=float)
    if arr.size == 0:
        return None
    return float(np.median(arr))


def _mean(rows: list[dict[str, Any]], key: str) -> float | None:
    vals = [r.get(key) for r in rows]
    arr = np.asarray([v for v in vals if isinstance(v, (int, float)) and np.isfinite(v)], dtype=float)
    if arr.size == 0:
        return None
    return float(np.mean(arr))


def main() -> None:
    ap = argparse.ArgumentParser(
        description=(
            "Compare score-level Shin map hygiene on existing bundles. "
            "Uses score_base vs score_stap_preka on the same masks and reports "
            "background tail clusters, tail excess mass, and flow-proxy hit rates."
        )
    )
    ap.add_argument("--root", type=Path, required=True)
    ap.add_argument("--out-csv", type=Path, required=True)
    ap.add_argument("--out-json", type=Path, required=True)
    ap.add_argument("--alpha", type=float, default=1e-3)
    ap.add_argument("--connectivity", type=int, default=4, choices=[4, 8])
    args = ap.parse_args()

    rows: list[dict[str, Any]] = []
    for bundle_dir in _iter_bundle_dirs(args.root):
        meta_path = bundle_dir / "meta.json"
        try:
            meta = json.loads(meta_path.read_text())
        except Exception:
            continue
        score_base = _load_npy(bundle_dir, "score_base")
        score_stap = _load_npy(bundle_dir, "score_stap_preka")
        mask_flow = _load_npy(bundle_dir, "mask_flow")
        mask_bg = _load_npy(bundle_dir, "mask_bg")
        if score_base is None or score_stap is None or mask_flow is None:
            continue
        if mask_bg is None:
            mask_bg = ~mask_flow.astype(bool)

        base = _score_metrics(score_base, mask_flow, mask_bg, alpha=float(args.alpha), connectivity=int(args.connectivity))
        stap = _score_metrics(score_stap, mask_flow, mask_bg, alpha=float(args.alpha), connectivity=int(args.connectivity))
        tele = meta.get("stap_fallback_telemetry") or {}
        score_stats = meta.get("score_stats") or {}
        orig = meta.get("orig_data") or {}
        row = {
            "bundle": bundle_dir.name,
            "iq_file": Path(str(orig.get("iq_file") or "")).name or None,
            "profile": ((meta.get("shin_profile") or {}).get("name") if isinstance(meta.get("shin_profile"), dict) else None),
            "detector_variant": score_stats.get("stap_detector_variant") or tele.get("detector_variant"),
            "whiten_gamma": score_stats.get("stap_whiten_gamma") or tele.get("whiten_gamma"),
            "Lt": meta.get("Lt"),
            "alpha": float(args.alpha),
            "base_bg_q999": base.get("bg_q999"),
            "stap_bg_q999": stap.get("bg_q999"),
            "base_hit_flow": base.get("hit_flow"),
            "stap_hit_flow": stap.get("hit_flow"),
            "base_bg_clusters": base.get("bg_clusters"),
            "stap_bg_clusters": stap.get("bg_clusters"),
            "base_bg_area": base.get("bg_area"),
            "stap_bg_area": stap.get("bg_area"),
            "base_bg_excess_mass": base.get("bg_excess_mass"),
            "stap_bg_excess_mass": stap.get("bg_excess_mass"),
            "base_largest_cluster": base.get("largest_cluster"),
            "stap_largest_cluster": stap.get("largest_cluster"),
            "delta_hit_flow": (
                None if base.get("hit_flow") is None or stap.get("hit_flow") is None else float(stap["hit_flow"]) - float(base["hit_flow"])
            ),
            "delta_bg_clusters": (
                None if base.get("bg_clusters") is None or stap.get("bg_clusters") is None else int(stap["bg_clusters"]) - int(base["bg_clusters"])
            ),
            "delta_bg_area": (
                None if base.get("bg_area") is None or stap.get("bg_area") is None else int(stap["bg_area"]) - int(base["bg_area"])
            ),
            "delta_bg_excess_mass": (
                None
                if base.get("bg_excess_mass") is None or stap.get("bg_excess_mass") is None
                else float(stap["bg_excess_mass"]) - float(base["bg_excess_mass"])
            ),
            "delta_largest_cluster": (
                None
                if base.get("largest_cluster") is None or stap.get("largest_cluster") is None
                else int(stap["largest_cluster"]) - int(base["largest_cluster"])
            ),
        }
        rows.append(row)

    args.out_csv.parent.mkdir(parents=True, exist_ok=True)
    args.out_json.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = sorted({k for r in rows for k in r.keys()})
    with args.out_csv.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for r in rows:
            w.writerow(r)

    by_variant: dict[str, list[dict[str, Any]]] = {}
    for r in rows:
        key = f"{r.get('profile')}_Lt{r.get('Lt')}_{r.get('detector_variant')}"
        by_variant.setdefault(key, []).append(r)

    summary: dict[str, Any] = {}
    for key, group in sorted(by_variant.items()):
        better_clusters = sum(1 for r in group if isinstance(r.get("delta_bg_clusters"), (int, float)) and r["delta_bg_clusters"] < 0)
        better_excess = sum(1 for r in group if isinstance(r.get("delta_bg_excess_mass"), (int, float)) and r["delta_bg_excess_mass"] < 0)
        better_hit = sum(1 for r in group if isinstance(r.get("delta_hit_flow"), (int, float)) and r["delta_hit_flow"] > 0)
        summary[key] = {
            "n_bundles": len(group),
            "median_delta_hit_flow": _median(group, "delta_hit_flow"),
            "median_delta_bg_clusters": _median(group, "delta_bg_clusters"),
            "median_delta_bg_area": _median(group, "delta_bg_area"),
            "median_delta_bg_excess_mass": _median(group, "delta_bg_excess_mass"),
            "median_delta_largest_cluster": _median(group, "delta_largest_cluster"),
            "mean_base_bg_clusters": _mean(group, "base_bg_clusters"),
            "mean_stap_bg_clusters": _mean(group, "stap_bg_clusters"),
            "mean_base_bg_excess_mass": _mean(group, "base_bg_excess_mass"),
            "mean_stap_bg_excess_mass": _mean(group, "stap_bg_excess_mass"),
            "n_better_clusters": int(better_clusters),
            "n_better_excess_mass": int(better_excess),
            "n_better_hit_flow": int(better_hit),
        }

    payload = {
        "config": {
            "root": str(args.root),
            "alpha": float(args.alpha),
            "connectivity": int(args.connectivity),
        },
        "summary": summary,
    }
    args.out_json.write_text(json.dumps(payload, indent=2, sort_keys=True))
    print(f"[shin-score-hygiene] bundles={len(rows)} wrote {args.out_csv} and {args.out_json}")


if __name__ == "__main__":
    main()
