import argparse
import json
from pathlib import Path

import numpy as np


def _load_json(path: Path) -> dict:
    return json.loads(path.read_text())


def _neighbor_coherence(score_map: np.ndarray, eta: float = 0.5) -> np.ndarray:
    """
    Compute a simple 4-neighbor coherence metric per pixel:
    fraction of neighbors whose score >= eta * local score.
    """
    h, w = score_map.shape
    coh = np.zeros_like(score_map, dtype=np.float32)

    # Pad for convenience
    pad = np.pad(score_map, 1, mode="edge")

    for y in range(h):
        for x in range(w):
            s = score_map[y, x]
            if s <= 0.0:
                coh[y, x] = 0.0
                continue
            neighbors = [
                pad[y + 1 - 1, x + 1],  # up
                pad[y + 1 + 1, x + 1],  # down
                pad[y + 1, x + 1 - 1],  # left
                pad[y + 1, x + 1 + 1],  # right
            ]
            thresh = eta * s
            coh[y, x] = float(sum(n >= thresh for n in neighbors)) / 4.0

    return coh


def dump_features(bundle: Path, out_path: Path, eta: float = 0.5) -> None:
    """
    Dump per-pixel STAP features and labels for a single bundle.

    Features (per pixel):
      - S_base: STAP MSD score (stap_score_map.npy)
      - br: band ratio map (stap_band_ratio_map.npy) if present, else NaN
      - log_br: log(br) clipped, else NaN
      - depth_frac: normalized row index in [0,1]
      - neighbor_coh: 4-neighbor coherence of S_base with threshold eta

    Label:
      - y: 1 for flow (mask_flow), 0 for background (mask_bg)
    """
    meta = _load_json(bundle / "meta.json")

    score_map_path = bundle / meta.get("stap_score_map", "stap_score_map.npy")
    band_ratio_path = bundle / meta.get("stap_band_ratio_map", "stap_band_ratio_map.npy")
    mask_flow_path = bundle / meta.get("mask_flow", "mask_flow.npy")
    mask_bg_path = bundle / meta.get("mask_bg", "mask_bg.npy")

    score_map = np.load(score_map_path)  # (H,W)
    band_ratio_map = np.load(band_ratio_path) if band_ratio_path.exists() else None
    mask_flow = np.load(mask_flow_path).astype(bool)
    mask_bg = np.load(mask_bg_path).astype(bool)

    h, w = score_map.shape
    depth_indices = np.arange(h, dtype=np.float32) / max(h - 1, 1)
    depth_frac_map = np.repeat(depth_indices[:, None], w, axis=1)
    neighbor_coh_map = _neighbor_coherence(score_map, eta=eta)

    # Flatten
    S_base = score_map.reshape(-1)
    depth_frac = depth_frac_map.reshape(-1)
    neighbor_coh = neighbor_coh_map.reshape(-1)
    y_flow = mask_flow.reshape(-1)
    y_bg = mask_bg.reshape(-1)

    # Only keep pixels that are in either mask (ignore unlabeled)
    keep = y_flow | y_bg

    S_base = S_base[keep]
    depth_frac = depth_frac[keep]
    neighbor_coh = neighbor_coh[keep]
    y = y_flow[keep].astype(np.int8)  # 1 for flow, 0 for bg

    if band_ratio_map is not None:
        br = band_ratio_map.reshape(-1)[keep]
        with np.errstate(divide="ignore", invalid="ignore"):
            log_br = np.log(br + 1e-6)
    else:
        br = np.full_like(S_base, np.nan, dtype=np.float32)
        log_br = np.full_like(S_base, np.nan, dtype=np.float32)

    out = {
        "S_base": S_base.astype(np.float32),
        "br": br.astype(np.float32),
        "log_br": log_br.astype(np.float32),
        "depth_frac": depth_frac.astype(np.float32),
        "neighbor_coh": neighbor_coh.astype(np.float32),
        "y": y,
    }
    out_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out_path, **out)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Dump per-pixel STAP features and labels from a bundle"
    )
    parser.add_argument(
        "--bundle",
        type=Path,
        required=True,
        help="Path to a single STAP bundle (directory containing meta.json)",
    )
    parser.add_argument(
        "--out",
        type=Path,
        required=True,
        help="Output .npz path for dumped features",
    )
    parser.add_argument(
        "--eta",
        type=float,
        default=0.5,
        help="Threshold fraction for neighbor coherence (default 0.5)",
    )
    args = parser.parse_args()
    dump_features(args.bundle, args.out, eta=args.eta)


if __name__ == "__main__":
    main()
