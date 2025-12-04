#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from pipeline.stap.temporal import ka_prior_temporal_from_psd


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=str, required=True, help="Path to save .npy prior")
    ap.add_argument("--lt", type=int, default=4)
    ap.add_argument("--prf", type=float, default=3000.0)
    args = ap.parse_args()
    p = Path(args.out)
    p.parent.mkdir(parents=True, exist_ok=True)
    R0 = ka_prior_temporal_from_psd(
        Lt=int(args.lt),
        prf_hz=float(args.prf),
        f_peaks_hz=(0.0,),
        width_bins=1,
        add_deriv=True,
        device="cpu",
    )
    np.save(p, R0.cpu().numpy().astype(np.complex64))
    print("saved", p)


if __name__ == "__main__":
    main()
