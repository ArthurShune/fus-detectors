import argparse
import json
from pathlib import Path


def _load_json(path: Path) -> dict:
    try:
        return json.loads(path.read_text())
    except FileNotFoundError as exc:
        raise SystemExit(f"[guard] missing {path}") from exc


def _nneg_at_threshold(roc_data: list, threshold: float) -> int | None:
    for entry in roc_data:
        bundle = entry.get("bundle", "")
        if not bundle.startswith("aggregate:"):
            continue
        for result in entry.get("coverage_results", []):
            if abs(float(result.get("threshold", -1.0)) - threshold) < 1e-9:
                return int(result.get("n_neg", 0))
    return None


def check_negatives(roc_json: Path, min_neg: int) -> None:
    data = _load_json(roc_json)
    nn = _nneg_at_threshold(data, 0.5)

    if nn is None:
        raise SystemExit("[guard] Missing 0.5 threshold slice in aggregate JSON")
    if nn < min_neg:
        raise SystemExit(f"[guard] N_neg@0.5 too low for 1e-4: {nn} (<{min_neg})")
    print(f"[guard] N_neg@0.5 OK: {nn}")


def check_amplitude(seeds: list[str], base_dir: Path, dataset_prefix: str) -> None:
    ok = True
    for seed in seeds:
        ds = f"{dataset_prefix}{seed}"
        meta = base_dir / f"ka_seed{seed}" / ds / "meta.json"
        if not meta.exists():
            print(f"[guard] missing {meta}")
            ok = False
            continue
        tele = _load_json(meta).get("stap_fallback_telemetry", {})
        pd = tele.get("flow_pdmask_ratio_median")
        bg = tele.get("bg_var_ratio_actual")
        if pd is None or float(pd) < 0.90:
            print(f"[guard] PD-mask {pd} below 0.90 for seed {seed}")
            ok = False
        if bg is not None and abs(float(bg) - 1.0) > 0.15:
            print(f"[guard] BG var ratio {bg} off target for seed {seed}")
            ok = False

    if not ok:
        raise SystemExit(1)
    print("[guard] acceptance amplitude guardrails passed")


def recommend_targets(
    roc_json: Path,
    desired_fpr: float,
    desired_pauc: float,
    threshold: float,
    note_path: Path | None,
    out_json: Path | None,
) -> None:
    data = _load_json(roc_json)
    n_neg = _nneg_at_threshold(data, threshold)
    if n_neg is None or n_neg <= 0:
        raise SystemExit("[guard] Unable to read N_neg for aggregate bundle")

    fpr_floor = 1.0 / float(n_neg)
    effective_fpr = max(desired_fpr, fpr_floor)
    effective_pauc = max(desired_pauc, effective_fpr * 5.0)
    rerun = effective_fpr > desired_fpr + 1e-12 or effective_pauc > desired_pauc + 1e-12

    note_text = None
    if note_path:
        if rerun:
            note_text = (
                f"Requested FPR {desired_fpr:.2e} falls below the coverage floor "
                f"{fpr_floor:.2e} (N_neg={n_neg}). "
                f"Auto-upgraded to {effective_fpr:.2e}; "
                "add seeds/coverage to measure 1e-4 reliably."
            )
            note_path.parent.mkdir(parents=True, exist_ok=True)
            note_path.write_text(note_text + "\n")
        elif note_path.exists():
            note_path.unlink()

    result = {
        "n_neg": n_neg,
        "fpr_floor": fpr_floor,
        "effective_fpr": effective_fpr,
        "effective_pauc": effective_pauc,
        "rerun": rerun,
        "note": note_text,
    }
    if out_json:
        out_json.parent.mkdir(parents=True, exist_ok=True)
        out_json.write_text(json.dumps(result, indent=2))
    print(json.dumps(result))


def main() -> None:
    parser = argparse.ArgumentParser(description="Guard checks for k-Wave brain fUS runs")
    sub = parser.add_subparsers(dest="cmd", required=True)

    neg = sub.add_parser("negatives", help="ensure enough negatives at 50% slice")
    neg.add_argument("--roc-json", type=Path, required=True)
    neg.add_argument("--min-neg", type=int, default=10000)

    amp = sub.add_parser("amplitude", help="verify KA amplitude guardrails")
    amp.add_argument("--seeds", required=True, help="space-separated seed list")
    amp.add_argument("--base-dir", type=Path, required=True)
    amp.add_argument("--dataset-prefix", required=True)

    rec = sub.add_parser("recommend", help="suggest FPR/pAUC targets based on available negatives")
    rec.add_argument("--roc-json", type=Path, required=True)
    rec.add_argument("--desired-fpr", type=float, required=True)
    rec.add_argument("--desired-pauc", type=float, required=True)
    rec.add_argument("--threshold", type=float, default=0.5)
    rec.add_argument("--note-path", type=Path)
    rec.add_argument("--out-json", type=Path)

    args = parser.parse_args()
    if args.cmd == "negatives":
        check_negatives(args.roc_json, args.min_neg)
    elif args.cmd == "amplitude":
        seeds = args.seeds.split()
        check_amplitude(seeds, args.base_dir, args.dataset_prefix)
    else:
        recommend_targets(
            args.roc_json,
            args.desired_fpr,
            args.desired_pauc,
            args.threshold,
            args.note_path,
            args.out_json,
        )


if __name__ == "__main__":
    main()
