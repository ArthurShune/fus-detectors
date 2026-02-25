import argparse
import json
import re
from pathlib import Path

import numpy as np
from PIL import Image

from pipeline.realdata.twinkling_artifact import (
    RawBCFPar,
    decode_rawbcf_bmode_iq,
    decode_rawbcf_cfm_cube,
    parse_rawbcf_par,
    read_rawbcf_frame,
)


def _slugify(text: str) -> str:
    s = re.sub(r"[\s/]+", "_", text.strip())
    s = re.sub(r"[^A-Za-z0-9_.-]+", "_", s)
    return s.strip("._-") or "seq"


def _to_uint8_log(img: np.ndarray) -> np.ndarray:
    x = np.asarray(img, dtype=np.float32)
    x = np.log1p(np.maximum(x, 0.0))
    x = (x - float(np.min(x))) / (float(np.max(x) - np.min(x)) + 1e-9)
    return (x * 255.0).astype(np.uint8)


def _corr(a: np.ndarray, b: np.ndarray) -> float:
    aa = a.astype(np.float64).ravel()
    bb = b.astype(np.float64).ravel()
    aa = (aa - aa.mean()) / (aa.std() + 1e-12)
    bb = (bb - bb.mean()) / (bb.std() + 1e-12)
    return float(np.mean(aa * bb))


def _mean_temporal_coherence(cube: np.ndarray) -> float:
    # cube: (T,H,W) complex
    x = cube.reshape(cube.shape[0], -1)
    vals = []
    for t in range(cube.shape[0] - 1):
        a = x[t]
        b = x[t + 1]
        num = np.vdot(a, b)
        den = np.sqrt(np.vdot(a, a) * np.vdot(b, b)) + 1e-12
        vals.append(float(np.abs(num / den)))
    return float(np.mean(vals)) if vals else float("nan")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Decode a Twinkling RawBCFCine .dat/.par frame and sanity-check B-mode vs picture.png."
    )
    parser.add_argument(
        "--seq-dir",
        type=str,
        required=True,
        help="Sequence directory containing RawBCFCine.dat / RawBCFCine.par (and optionally picture.png).",
    )
    parser.add_argument(
        "--par-path",
        type=str,
        default=None,
        help="Optional explicit .par path (default: auto-detect in seq-dir).",
    )
    parser.add_argument(
        "--dat-path",
        type=str,
        default=None,
        help="Optional explicit .dat path (default: auto-detect in seq-dir).",
    )
    parser.add_argument("--frame-idx", type=int, default=0, help="Cine frame index (default: 0).")
    parser.add_argument(
        "--picture-path",
        type=str,
        default=None,
        help=(
            "Optional explicit picture/PNG path for B-mode correlation (default: auto-detect picture.png or picture.*). "
            "Use this for sequence folders that store screenshots under different filenames."
        ),
    )
    parser.add_argument(
        "--out-dir",
        type=str,
        default="reports/twinkling_decode_sanity",
        help="Output directory root (default: reports/twinkling_decode_sanity).",
    )
    args = parser.parse_args()

    seq_dir = Path(args.seq_dir)
    if args.par_path is not None:
        par_path = Path(args.par_path)
    else:
        cand = seq_dir / "RawBCFCine.par"
        if cand.exists():
            par_path = cand
        else:
            pars = sorted(seq_dir.glob("*.par"))
            if len(pars) != 1:
                raise FileNotFoundError(f"Could not auto-detect a single .par in {seq_dir}: {pars}")
            par_path = pars[0]

    if args.dat_path is not None:
        dat_path = Path(args.dat_path)
    else:
        cand = seq_dir / "RawBCFCine.dat"
        if cand.exists():
            dat_path = cand
        else:
            dats = sorted(seq_dir.glob("*.dat"))
            if len(dats) != 1:
                raise FileNotFoundError(f"Could not auto-detect a single .dat in {seq_dir}: {dats}")
            dat_path = dats[0]

    if args.picture_path is not None:
        pic_path = Path(args.picture_path)
    else:
        pic_path = seq_dir / "picture.png"
        if not pic_path.exists():
            pics = sorted([p for p in seq_dir.glob("picture*") if p.is_file()])
            if pics:
                pic_path = pics[0]

    par_dict = parse_rawbcf_par(par_path)
    par = RawBCFPar.from_dict(par_dict)
    par.validate()

    frame = read_rawbcf_frame(dat_path, par, int(args.frame_idx))
    bmode = decode_rawbcf_bmode_iq(frame, par)  # (beams, samples)
    cfm = decode_rawbcf_cfm_cube(frame, par, order="beam_major")  # (shots, depth, beams)
    cfm_alt = decode_rawbcf_cfm_cube(frame, par, order="shot_major")

    out_root = Path(args.out_dir)
    rel = str(seq_dir)
    try:
        if seq_dir.is_relative_to(Path.cwd()):
            rel = str(seq_dir.relative_to(Path.cwd()))
    except Exception:
        pass
    out_seq = out_root / _slugify(f"{rel}__{par_path.stem}")
    out_seq.mkdir(parents=True, exist_ok=True)

    # Save decoded B-mode envelope.
    b_env = np.abs(bmode).T  # (samples, beams)
    b_img8 = _to_uint8_log(b_env)
    Image.fromarray(b_img8).save(out_seq / f"bmode_decoded_frame{int(args.frame_idx):03d}.png")

    # Save decoded CFM power (PD-like) for quick visual sanity.
    c_pow = np.mean(np.abs(cfm) ** 2, axis=0)  # (depth, beams)
    c_img8 = _to_uint8_log(c_pow)
    Image.fromarray(c_img8).save(out_seq / f"cfm_power_decoded_frame{int(args.frame_idx):03d}.png")

    report: dict[str, object] = {
        "seq_dir": str(seq_dir),
        "par_path": str(par_path),
        "dat_path": str(dat_path),
        "frame_idx": int(args.frame_idx),
        "picture_path": str(pic_path) if pic_path.exists() else None,
        "par": {
            "num_frames": int(par.num_frames),
            "raw_frame_size": int(par.raw_frame_size),
            "header_size": int(par.header_size),
            "sample_bytes": int(par.sample_bytes),
            "num_b_beams": int(par.num_b_beams),
            "b_beam_samples": int(par.b_beam_samples),
            "num_cfm_shots": int(par.num_cfm_shots),
            "num_cfm_beams": int(par.num_cfm_beams),
            "cfm_beam_samples": int(par.cfm_beam_samples),
            "num_sweeps": par.num_sweeps,
            "beams_in_sweep": par.beams_in_sweep,
            "num_first_cfm_sample": par.num_first_cfm_sample,
            "first_scan_cfm_beam": par.first_scan_cfm_beam,
        },
        "decoded_shapes": {
            "bmode_iq": list(bmode.shape),
            "cfm_iq": list(cfm.shape),
        },
        "cfm_order_sanity": {
            "selected": "beam_major",
            "mean_temporal_coherence": {
                "beam_major": _mean_temporal_coherence(cfm),
                "shot_major": _mean_temporal_coherence(cfm_alt),
            },
        },
        "bmode_picture_corr": None,
    }

    if pic_path.exists():
        pic = np.array(Image.open(pic_path).convert("L"), dtype=np.float32)
        # Many dataset screenshots have a final all-zero column; drop it to avoid skew.
        if pic.shape[1] >= 2 and float(np.mean(pic[:, -1])) == 0.0:
            pic = pic[:, :-1]
        b_resized = np.array(
            Image.fromarray(b_img8).resize((pic.shape[1], pic.shape[0]), resample=Image.BILINEAR),
            dtype=np.float32,
        )
        c0 = _corr(b_resized, pic)
        c1 = _corr(b_resized[:, ::-1], pic)
        report["bmode_picture_corr"] = {"direct": c0, "flip_lr": c1}

    (out_seq / "decode_report.json").write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
