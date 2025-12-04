import argparse
import json
from pathlib import Path

import numpy as np


def load_band_projectors(bundle_dir: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Load Pf, Pa, Po projectors from a baseline STAP bundle if present."""
    meta_path = bundle_dir / "meta.json"
    meta = json.loads(meta_path.read_text())
    tele = meta.get("stap_fallback_telemetry") or meta.get("stap_telemetry") or {}

    # Expect band projectors saved as real/imag parts in telemetry.
    Pf_real = tele.get("Pf_real")
    Pf_imag = tele.get("Pf_imag")
    Pa_real = tele.get("Pa_real")
    Pa_imag = tele.get("Pa_imag")
    if Pf_real is None or Pf_imag is None or Pa_real is None or Pa_imag is None:
        raise RuntimeError(
            "Band projectors Pf/Pa not found in telemetry; "
            "build_pial_prior expects Pf_real/Pf_imag and Pa_real/Pa_imag."
        )
    Pf_arr = np.asarray(Pf_real, dtype=np.float32) + 1j * np.asarray(Pf_imag, dtype=np.float32)
    Pa_arr = np.asarray(Pa_real, dtype=np.float32) + 1j * np.asarray(Pa_imag, dtype=np.float32)
    Po_arr = np.eye(Pf_arr.shape[0], dtype=np.complex64) - Pf_arr - Pa_arr
    return Pf_arr, Pa_arr, Po_arr


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Build a pial-specific KA prior R0 from a baseline STAP bundle by "
            "averaging band-restricted covariances on alias-heavy pial BG tiles."
        )
    )
    parser.add_argument(
        "--bundle",
        type=Path,
        required=True,
        help="Baseline STAP bundle to sample from (e.g. pial_mcsvd_k8_reg4_seed1/... ).",
    )
    parser.add_argument(
        "--out",
        type=Path,
        required=True,
        help="Output .npy path for the pial KA prior.",
    )
    args = parser.parse_args()

    bundle = args.bundle
    out_path = args.out

    meta_path = bundle / "meta.json"
    if not meta_path.exists():
        raise FileNotFoundError(f"Missing meta.json in {bundle}")
    meta = json.loads(meta_path.read_text())

    tele = meta.get("stap_fallback_telemetry") or meta.get("stap_telemetry") or {}
    Lt = int(tele.get("Lt") or 8)

    Pf, Pa, Po = load_band_projectors(bundle)

    # For now, approximate R0 as band-scalar: identity in each band with
    # slightly inflated alias block. This is a placeholder that can be
    # refined once band-restricted covariances are exposed.
    eye = np.eye(Lt, dtype=np.complex64)
    R0 = 0.0 * eye
    # Projectors are assumed Hermitian idempotent in the Lt basis.
    Pfh = Pf.astype(np.complex64, copy=False)
    Pah = Pa.astype(np.complex64, copy=False)
    Poh = Po.astype(np.complex64, copy=False)
    # Example band-scalar weights: w_f = w_o = 1.0, w_a = 1.5.
    w_f = 1.0
    w_a = 1.5
    w_o = 1.0
    R0 = w_f * Pfh + w_a * Pah + w_o * Poh
    # Ensure Hermitian.
    R0 = 0.5 * (R0 + R0.conj().T)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    np.save(out_path, R0.astype(np.complex64))
    print(f"[build_pial_prior] wrote {out_path} with shape {R0.shape}")


if __name__ == "__main__":
    main()
