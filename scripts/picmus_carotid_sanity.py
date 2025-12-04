#!/usr/bin/env python

from __future__ import annotations

import argparse

import numpy as np

from pipeline.realdata.picmus_carotid import load_picmus_carotid_uff


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--path",
        type=str,
        default="data/picmus/PICMUS_carotid_cross.uff",
        help="Path to PICMUS_carotid_cross.uff",
    )
    args = ap.parse_args()

    scan = load_picmus_carotid_uff(args.path)

    rf = scan.rf  # (T, S, C)
    T, S, C = rf.shape

    print(f"[picmus] loaded '{scan.name}' from {args.path}")
    print(f"[picmus] rf shape (T,S,C) = {rf.shape}")
    print(f"[picmus] fs = {scan.fs} Hz, prf = {scan.prf}, fc = {scan.center_freq}")
    print(f"[picmus] dtype = {rf.dtype}, abs(rf) mean/std = "
          f"{np.abs(rf).mean():.3e} / {np.abs(rf).std():.3e}")
    print(f"[picmus] meta keys: {sorted(list(scan.meta.keys()))[:12]} ...")


if __name__ == "__main__":
    main()
