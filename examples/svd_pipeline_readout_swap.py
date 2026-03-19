#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import numpy as np

# Allow direct execution from a source checkout.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from fus_detectors import DetectorConfig, score_residual_cube


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Swap the final PD readout for fus-detectors on an existing clutter-filtered residual cube."
    )
    parser.add_argument(
        "--input",
        type=Path,
        required=True,
        help="Complex-valued .npy clutter-filtered residual cube with shape (T, H, W).",
    )
    parser.add_argument("--prf-hz", type=float, required=True, help="Slow-time PRF in Hz.")
    parser.add_argument(
        "--variant",
        default="fixed",
        choices=["fixed", "adaptive", "whitened", "whitened_power"],
        help="Detector variant to run.",
    )
    parser.add_argument(
        "--device",
        default="cpu",
        help="Torch device for detector execution, for example 'cpu' or 'cuda'.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("swap_outputs"),
        help="Directory for saved baseline PD, detector readout, score map, and summary JSON.",
    )
    return parser.parse_args()


def _baseline_pd(residual_cube: np.ndarray) -> np.ndarray:
    residual_cube = np.asarray(residual_cube)
    return np.mean(np.abs(residual_cube) ** 2, axis=0, dtype=np.float64).astype(np.float32)


def main() -> None:
    args = parse_args()
    residual_cube = np.load(args.input)

    result = score_residual_cube(
        residual_cube,
        prf_hz=float(args.prf_hz),
        config=DetectorConfig(variant=str(args.variant), device=str(args.device)),
    )

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    baseline_pd = _baseline_pd(residual_cube)
    np.save(output_dir / "baseline_pd.npy", baseline_pd)
    np.save(output_dir / "detector_readout.npy", result.readout_map)
    np.save(output_dir / "detector_score.npy", result.score_map)

    summary = {
        "variant": str(args.variant),
        "prf_hz": float(args.prf_hz),
        "input": str(args.input),
        "baseline_pd_shape": list(baseline_pd.shape),
        "detector_readout_shape": list(result.readout_map.shape),
        "detector_score_shape": list(result.score_map.shape),
        "detector_summary": result.summary.to_dict(),
    }
    summary_path = output_dir / "swap_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2) + "\n")

    print("Saved same-residual outputs:")
    print(f"  baseline PD:       {output_dir / 'baseline_pd.npy'}")
    print(f"  detector readout:  {output_dir / 'detector_readout.npy'}")
    print(f"  detector score:    {output_dir / 'detector_score.npy'}")
    print(f"  summary:           {summary_path}")
    print()
    print("Interpretation:")
    print("  - baseline_pd.npy is the ordinary PD-style readout on the same residual cube")
    print("  - detector_readout.npy is the drop-in detector map for the chosen variant")
    print("  - detector_score.npy is the right-tail score used for thresholded comparisons")


if __name__ == "__main__":
    main()
