from __future__ import annotations

import argparse
from pathlib import Path

from sim.simus.pilot_pymust_simus import SUPPORTED_SIMUS_PROFILES, SimusConfig, resolve_config_from_args, write_simus_run


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Generate one or more canonical SIMUS/PyMUST runs.")
    ap.add_argument("--out-root", type=Path, required=True, help="Directory that will contain per-run subdirectories.")
    ap.add_argument("--seed", type=int, default=0, help="Starting seed.")
    ap.add_argument("--n-clips", type=int, default=1, help="Number of independent seeds/runs to generate.")
    ap.add_argument(
        "--profile",
        type=str,
        default=None,
        choices=list(SUPPORTED_SIMUS_PROFILES),
        help="Clinically aligned SIMUS profile. When set, this overrides --preset defaults.",
    )
    ap.add_argument(
        "--preset",
        type=str,
        default="microvascular_like",
        choices=["microvascular_like", "alias_stress"],
        help="Named SIMUS regime (legacy path).",
    )
    ap.add_argument("--tier", type=str, default="smoke", choices=["smoke", "paper", "functional"])
    ap.add_argument("--skip-bundle", action="store_true")
    ap.add_argument("--probe", type=str, default=None)
    ap.add_argument("--prf-hz", type=float, default=None)
    ap.add_argument("--T", type=int, default=None)
    ap.add_argument("--H", type=int, default=None)
    ap.add_argument("--W", type=int, default=None)
    ap.add_argument("--x-min-m", type=float, default=None)
    ap.add_argument("--x-max-m", type=float, default=None)
    ap.add_argument("--z-min-m", type=float, default=None)
    ap.add_argument("--z-max-m", type=float, default=None)
    ap.add_argument("--blood-vmax-mps", type=float, default=None)
    ap.add_argument("--blood-profile", type=str, default=None, choices=["plug", "poiseuille"])
    ap.add_argument("--vessel-radius-m", type=float, default=None)
    ap.add_argument("--tissue-count", type=int, default=None)
    ap.add_argument("--blood-count", type=int, default=None)
    return ap.parse_args()


def _run_slug(cfg: SimusConfig) -> str:
    if cfg.profile:
        return f"simus_{cfg.profile.lower().replace('-', '_')}_seed{cfg.seed}"
    return f"simus_{cfg.preset}_{cfg.tier}_seed{cfg.seed}"


def main() -> None:
    args = parse_args()
    out_root = Path(args.out_root)
    out_root.mkdir(parents=True, exist_ok=True)

    for idx in range(int(args.n_clips)):
        run_args = argparse.Namespace(**vars(args))
        run_args.seed = int(args.seed) + idx
        cfg = resolve_config_from_args(run_args)
        run_dir = out_root / _run_slug(cfg)
        outputs = write_simus_run(out_root=run_dir, cfg=cfg, skip_bundle=bool(args.skip_bundle))
        print(f"[simus-cli-generate] wrote {outputs['dataset_dir']}", flush=True)


if __name__ == "__main__":
    main()
