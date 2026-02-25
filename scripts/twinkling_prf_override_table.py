from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

from pipeline.realdata.twinkling_artifact import RawBCFPar, parse_rawbcf_par


def _iter_meta_paths(runs_root: Path) -> list[Path]:
    out: list[Path] = []
    for p in runs_root.rglob("meta.json"):
        if p.is_file():
            out.append(p)
    out.sort()
    return out


def _load_json(path: Path) -> dict[str, Any] | None:
    try:
        obj = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None
    return obj if isinstance(obj, dict) else None


def _prf_source(seq_dir: str) -> str:
    s = (seq_dir or "").lower()
    if "flow in gammex phantom" in s:
        return "LeonovTwinklingDD2021 (Gammex example settings; RawBCF PRF not reliably encoded)"
    if "twinkling artifact on calculi" in s or "calculi" in s:
        return "Manual override (RawBCF PRF not reliably encoded; instrumentation-only regime)"
    return "Manual override (RawBCF PRF not reliably encoded)"


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Generate a PRF override table for Twinkling RawBCF sequences used in runs/real.\n"
            "Twinkling RawBCF sidecars do not reliably store Doppler PRF, so experiments use\n"
            "an explicit PRF override; this script extracts the per-sequence PRF values that\n"
            "were actually used in exported acceptance bundles."
        )
    )
    parser.add_argument(
        "--runs-root",
        type=Path,
        default=Path("runs/real"),
        help="Root containing Twinkling run directories (default: %(default)s).",
    )
    parser.add_argument(
        "--decode-sanity-root",
        type=Path,
        default=Path("reports/twinkling_decode_sanity"),
        help="Optional decode-sanity root to join correlation/coherence (default: %(default)s).",
    )
    parser.add_argument(
        "--out-csv",
        type=Path,
        default=Path("reports/twinkling_prf_override_table.csv"),
        help="Output CSV path (default: %(default)s).",
    )
    parser.add_argument(
        "--out-json",
        type=Path,
        default=Path("reports/twinkling_prf_override_table.json"),
        help="Output JSON path (default: %(default)s).",
    )
    args = parser.parse_args()

    # Optional decode-sanity join (by par_path).
    decode_by_par: dict[str, dict[str, Any]] = {}
    if args.decode_sanity_root.exists():
        for p in sorted(args.decode_sanity_root.rglob("decode_report.json")):
            rep = _load_json(p)
            if not rep:
                continue
            par_path = rep.get("par_path")
            if isinstance(par_path, str) and par_path:
                rep = dict(rep)
                rep["decode_report"] = str(p)
                decode_by_par[str(Path(par_path))] = rep

    meta_paths = _iter_meta_paths(args.runs_root)
    if not meta_paths:
        raise SystemExit(f"No meta.json files found under {args.runs_root}")

    # Collect unique sequences by (seq_dir, par_path).
    seqs: dict[tuple[str, str], dict[str, Any]] = {}
    for mp in meta_paths:
        meta = _load_json(mp)
        if not meta:
            continue
        tw = meta.get("twinkling_rawbcf")
        if not isinstance(tw, dict):
            continue
        seq_dir = tw.get("seq_dir")
        par_path = tw.get("par_path")
        dat_path = tw.get("dat_path")
        if not isinstance(seq_dir, str) or not seq_dir:
            continue
        if not isinstance(par_path, str) or not par_path:
            continue
        key = (seq_dir, par_path)
        rec = seqs.get(key)
        if rec is None:
            rec = {
                "seq_dir": seq_dir,
                "par_path": par_path,
                "dat_path": dat_path if isinstance(dat_path, str) else None,
                "prf_hz": meta.get("prf_hz"),
                "decode_cfm_order": tw.get("decode_cfm_order"),
                "prf_hz_note": tw.get("prf_hz_note"),
                "bundle_count": 0,
                "example_bundle": str(mp.parent),
                "run_root": str(mp.parents[3]) if len(mp.parents) >= 4 else None,
            }
            # Parse par for basic acquisition metadata (best-effort).
            try:
                par = RawBCFPar.from_dict(parse_rawbcf_par(Path(par_path)))
                rec.update(
                    {
                        "num_frames": int(par.num_frames),
                        "num_cfm_shots": int(par.num_cfm_shots),
                        "cfm_beam_samples": int(par.cfm_beam_samples),
                        "num_cfm_beams": int(par.num_cfm_beams),
                        "cfm_density": par.cfm_density,
                        "first_scan_cfm_beam": par.first_scan_cfm_beam,
                        "num_first_cfm_sample": par.num_first_cfm_sample,
                    }
                )
            except Exception:
                rec.update(
                    {
                        "num_frames": None,
                        "num_cfm_shots": None,
                        "cfm_beam_samples": None,
                        "num_cfm_beams": None,
                        "cfm_density": None,
                        "first_scan_cfm_beam": None,
                        "num_first_cfm_sample": None,
                    }
                )
            # Join decode sanity if available.
            drep = decode_by_par.get(str(Path(par_path)))
            if isinstance(drep, dict):
                corr = drep.get("bmode_picture_corr") if isinstance(drep.get("bmode_picture_corr"), dict) else {}
                coh = (
                    drep.get("cfm_order_sanity", {}).get("mean_temporal_coherence", {})
                    if isinstance(drep.get("cfm_order_sanity"), dict)
                    else {}
                )
                rec.update(
                    {
                        "decode_report": drep.get("decode_report"),
                        "bmode_picture_corr_direct": corr.get("direct") if isinstance(corr, dict) else None,
                        "bmode_picture_corr_flip_lr": corr.get("flip_lr") if isinstance(corr, dict) else None,
                        "coh_beam_major": coh.get("beam_major") if isinstance(coh, dict) else None,
                        "coh_shot_major": coh.get("shot_major") if isinstance(coh, dict) else None,
                    }
                )
            else:
                rec.update(
                    {
                        "decode_report": None,
                        "bmode_picture_corr_direct": None,
                        "bmode_picture_corr_flip_lr": None,
                        "coh_beam_major": None,
                        "coh_shot_major": None,
                    }
                )
            rec["prf_source"] = _prf_source(seq_dir)
            seqs[key] = rec
        rec["bundle_count"] = int(rec.get("bundle_count", 0)) + 1

    rows = sorted(seqs.values(), key=lambda r: (str(r.get("seq_dir") or ""), str(r.get("par_path") or "")))
    if not rows:
        raise SystemExit("No Twinkling sequences found in runs metadata.")

    args.out_csv.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = sorted({k for r in rows for k in r.keys()})
    with args.out_csv.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(rows)

    summary = {
        "runs_root": str(args.runs_root),
        "decode_sanity_root": str(args.decode_sanity_root),
        "sequence_count": int(len(rows)),
        "rows": rows,
    }
    args.out_json.parent.mkdir(parents=True, exist_ok=True)
    args.out_json.write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()

