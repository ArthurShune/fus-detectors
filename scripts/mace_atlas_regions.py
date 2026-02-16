#!/usr/bin/env python3
"""
Inspect Allen atlas region metadata for the Macé/Urban whole-brain fUS dataset.

This script parses:
  - `allen_brain_atlas.mat` to recover region acronyms/names,
  - `listSelectedRegions.txt` to obtain the subset of regions highlighted by the authors,
  - and reports which acronyms in the list are found in the atlas and how many
    voxels they occupy in the `Regions` volume.

It is a wiring / sanity check step before building full ROI masks and
aligning them with PD volumes.

Usage
-----
    PYTHONPATH=. python scripts/mace_atlas_regions.py
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np

from pipeline.realdata import mace_data_root
from pipeline.realdata.mace_wholebrain import (
    MaceAtlas,
    MaceRegionInfo,
    load_mace_atlas,
    load_mace_region_info,
)


def _parse_selected_regions(path: Path) -> List[str]:
    """
    Parse `listSelectedRegions.txt`, ignoring comments and groupings.

    The file contains lines with acronyms separated by spaces, optionally
    followed by inline '//' comments. Lines starting with '%' are comments.
    """

    selected: List[str] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("%"):
                continue
            if "//" in line:
                line = line.split("//", 1)[0]
            tokens = [tok.strip() for tok in line.split() if tok.strip()]
            selected.extend(tokens)
    # Deduplicate while preserving order
    seen: Dict[str, None] = {}
    out: List[str] = []
    for region in selected:
        if region not in seen:
            seen[region] = None
            out.append(region)
    return out


def _region_voxel_counts(
    atlas: MaceAtlas, region_info: MaceRegionInfo
) -> Dict[str, Tuple[int, int]]:
    """
    Return a mapping from acronym -> (label_value, voxel_count) for all labels present.
    """

    counts: Dict[str, Tuple[int, int]] = {}
    labels, freqs = np.unique(atlas.regions, return_counts=True)
    for label, nvox in zip(labels, freqs, strict=True):
        if label <= 0:
            continue
        # Labels are 1..N inclusive.
        if int(label) > len(region_info.acronyms):
            # Defensive: label outside metadata range
            continue
        # Atlas `Regions` stores 1-based indices into infoRegions.
        acr = region_info.acronyms[int(label) - 1]
        counts[acr] = (int(label), int(nvox))
    return counts


def main() -> None:
    root = mace_data_root()
    atlas = load_mace_atlas(root)
    region_info = load_mace_region_info(root)
    selected_path = root / "listSelectedRegions.txt"
    if not selected_path.exists():
        raise FileNotFoundError(selected_path)
    selected = _parse_selected_regions(selected_path)

    counts = _region_voxel_counts(atlas, region_info)

    print(f"Atlas Histology shape: {atlas.histology.shape}, Regions shape: {atlas.regions.shape}")
    print(f"Voxel size (um): {atlas.voxel_size_um}, Direction: {atlas.direction}")
    print(f"Total labeled regions (excluding background): {len(counts)}")
    print()
    print(f"Selected regions from listSelectedRegions.txt: {len(selected)} unique acronyms")

    found = []
    missing = []
    for acr in selected:
        if acr in counts:
            label_val, nvox = counts[acr]
            found.append((acr, label_val, nvox))
        else:
            missing.append(acr)

    if found:
        print("\nFound selected regions in atlas (first 20 shown):")
        for acr, label_val, nvox in found[:20]:
            print(f"  {acr:8s} -> label={label_val:3d}, voxels={nvox}")
    else:
        print("\nNo selected regions matched atlas labels.")

    if missing:
        print("\nSelected acronyms not found in atlas labels:")
        print(" ", ", ".join(sorted(missing)))


if __name__ == "__main__":
    main()
