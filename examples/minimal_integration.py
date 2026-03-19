#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import numpy as np

# Allow direct execution from a source checkout without requiring pip install -e .
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from fus_detectors import DetectorConfig, score_residual_cube


def _demo_cube(
    *,
    prf_hz: float,
    t_len: int = 24,
    height: int = 16,
    width: int = 16,
    fd_hz: float = 350.0,
) -> np.ndarray:
    """Create a tiny synthetic residual cube for API smoke testing."""

    t = np.arange(t_len, dtype=np.float32)
    tone = np.exp(1j * 2.0 * np.pi * float(fd_hz) * t / float(prf_hz)).astype(np.complex64)
    cube = 0.02 * (
        np.random.randn(t_len, height, width).astype(np.float32)
        + 1j * np.random.randn(t_len, height, width).astype(np.float32)
    )
    cube[:, height // 2, width // 2] += 1.5 * tone
    return cube.astype(np.complex64)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Minimal fus-detectors integration example for clutter-filtered residual cubes."
    )
    parser.add_argument(
        "--input",
        type=Path,
        help="Path to a complex-valued .npy residual cube with shape (T, H, W).",
    )
    parser.add_argument(
        "--demo",
        action="store_true",
        help="Run a tiny self-contained demo cube instead of loading --input.",
    )
    parser.add_argument("--prf-hz", type=float, default=3000.0, help="Slow-time PRF in Hz.")
    parser.add_argument(
        "--variant",
        default="fixed",
        choices=["fixed", "adaptive", "whitened", "whitened_power"],
        help="Public detector variant.",
    )
    parser.add_argument(
        "--device",
        default="cpu",
        help="Torch device for the detector core, for example 'cpu' or 'cuda'.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("example_outputs"),
        help="Directory for the saved readout, score, and summary files.",
    )
    args = parser.parse_args()
    if not args.demo and args.input is None:
        parser.error("pass --demo or provide --input")
    return args


def main() -> None:
    args = parse_args()
    if args.demo:
        residual_cube = _demo_cube(prf_hz=float(args.prf_hz))
    else:
        residual_cube = np.load(args.input)

    result = score_residual_cube(
        residual_cube,
        prf_hz=float(args.prf_hz),
        config=DetectorConfig(
            variant=str(args.variant),
            device=str(args.device),
        ),
    )

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    np.save(output_dir / "readout_map.npy", result.readout_map)
    np.save(output_dir / "score_map.npy", result.score_map)
    summary_path = output_dir / "summary.json"
    summary_path.write_text(json.dumps(result.summary.to_dict(), indent=2) + "\n")

    print(f"saved readout map: {output_dir / 'readout_map.npy'}")
    print(f"saved score map:   {output_dir / 'score_map.npy'}")
    print(f"saved summary:     {summary_path}")
    print(json.dumps(result.summary.to_dict(), indent=2))


if __name__ == "__main__":
    main()
