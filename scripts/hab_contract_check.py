import argparse
import json
import os
from typing import Dict, Sequence, Tuple

import numpy as np


def _load_bundle(bundle_dir: str) -> Dict:
    meta_path = os.path.join(bundle_dir, "meta.json")
    if not os.path.exists(meta_path):
        raise FileNotFoundError(f"meta.json not found in {bundle_dir}")
    with open(meta_path) as f:
        meta = json.load(f)

    def _load(name: str) -> np.ndarray:
        path = os.path.join(bundle_dir, name)
        if not os.path.exists(path):
            raise FileNotFoundError(path)
        return np.load(path)

    out: Dict[str, np.ndarray | Dict] = {"meta": meta}
    # Always-required maps.
    out["mask_flow"] = _load("mask_flow.npy")
    out["mask_bg"] = _load("mask_bg.npy")
    out["base_band_ratio"] = _load("base_band_ratio_map.npy")
    # Optional: explicit alias metric map (m_alias = log(Ea/Ef) up to a constant).
    base_m_alias_path = os.path.join(bundle_dir, "base_m_alias_map.npy")
    if os.path.exists(base_m_alias_path):
        out["base_m_alias"] = np.load(base_m_alias_path)
    out["base_score"] = _load("base_score_map.npy")
    out["stap_score"] = _load("stap_score_map.npy")
    # vNext primary score exports (preferred when present).
    score_base_path = os.path.join(bundle_dir, "score_base.npy")
    if os.path.exists(score_base_path):
        out["score_base"] = np.load(score_base_path)
    score_stap_preka_path = os.path.join(bundle_dir, "score_stap_preka.npy")
    if os.path.exists(score_stap_preka_path):
        out["score_stap_preka"] = np.load(score_stap_preka_path)
    score_stap_path = os.path.join(bundle_dir, "score_stap.npy")
    if os.path.exists(score_stap_path):
        out["score_stap"] = np.load(score_stap_path)
    score_name_path = os.path.join(bundle_dir, "score_name.txt")
    if os.path.exists(score_name_path):
        try:
            with open(score_name_path, "r", encoding="utf-8") as f:
                out["score_name"] = f.read()
        except Exception:
            pass
    # Optional maps used when we want to align R-3 with the actual
    # detector score used in the run (e.g. band-ratio or PD).
    for key, fname in (
        ("stap_score_pool", "stap_score_pool_map.npy"),
        ("pd_base", "pd_base.npy"),
        ("pd_stap", "pd_stap.npy"),
        ("score_pd_base", "score_pd_base.npy"),
        ("score_pd_stap", "score_pd_stap.npy"),
        ("stap_band_ratio", "stap_band_ratio_map.npy"),
    ):
        path = os.path.join(bundle_dir, fname)
        if os.path.exists(path):
            out[key] = np.load(path)

    return out


def _load_rf_tensor(bundle_dir: str, meta: Dict) -> np.ndarray:
    """
    Locate and load the rf_tensor corresponding to this bundle.

    For k-Wave pilot runs we expect rf_tensor.npy to live either one level
    above the bundle directory or at the original simulation root recorded
    in meta['orig_run'], e.g.:
      runs/pilot/r4c_kwave_hab_seed2/
        rf_tensor.npy
        pw_7.5MHz_5ang_5ens_320T_seed2/
          meta.json
          ...
    """

    parent = os.path.dirname(os.path.abspath(bundle_dir))
    candidates = [os.path.join(parent, "rf_tensor.npy")]

    # Replay/latency bundles (e.g. *_svdlit_stap_pd_clinical_fastpdonly_*)
    # record the original simulation root in meta['orig_run']. In that
    # case rf_tensor.npy lives under orig_run rather than under the
    # replay out_root, so fall back to that location when available.
    orig_run = meta.get("orig_run")
    if isinstance(orig_run, str) and orig_run:
        candidates.append(os.path.join(os.path.abspath(orig_run), "rf_tensor.npy"))

    rf_path = None
    for cand in candidates:
        if os.path.exists(cand):
            rf_path = cand
            break
    if rf_path is None:
        raise FileNotFoundError("rf_tensor.npy not found at any of: " + ", ".join(candidates))

    rf_raw = np.load(rf_path)

    # For k-Wave pilot runs we typically save rf_tensor with shape
    # (ensembles, angles, Nt, Nx). For the purposes of the R-1 PSD
    # check we can collapse the ensemble/angle axes into a single
    # ``height'' dimension and treat (Nt, H, W) as a slow-time cube.
    if rf_raw.ndim == 4:
        e, a, t_len, nx = rf_raw.shape
        rf = np.transpose(rf_raw, (2, 0, 1, 3)).reshape(t_len, e * a, nx)
    else:
        rf = rf_raw

    # Sanity-check against meta where possible, but do not enforce
    # equality: the raw k-Wave Nt can differ from the compounded
    # frame count used for STAP.
    t_expected = meta.get("total_frames")
    if t_expected is not None and rf.shape[0] != t_expected:
        print(
            f"[R-1] Warning: rf_tensor time length {rf.shape[0]} "
            f"differs from meta total_frames={t_expected}; proceeding with FFT "
            "on rf_tensor as-is."
        )
    return rf


def _tpr_at_fpr(scores_pos: np.ndarray, scores_neg: np.ndarray, fpr: float) -> Tuple[float, float]:
    """
    Compute (threshold, TPR) at the requested FPR using empirical quantiles.
    """

    if not (0.0 < fpr < 1.0):
        raise ValueError(f"fpr must be in (0,1), got {fpr}")

    scores_neg = np.asarray(scores_neg, dtype=float)
    scores_pos = np.asarray(scores_pos, dtype=float)

    if scores_neg.size == 0 or scores_pos.size == 0:
        return np.nan, np.nan

    neg_sorted = np.sort(scores_neg)
    n_neg = neg_sorted.size

    # Empirical (1 - fpr) quantile.
    q = 1.0 - fpr
    idx = int(np.floor(q * n_neg))
    idx = max(0, min(idx, n_neg - 1))
    thr = neg_sorted[idx]

    tpr = float((scores_pos >= thr).mean())
    return thr, tpr


def check_r1_band_occupancy(
    rf: np.ndarray,
    mask_flow: np.ndarray,
    mask_bg: np.ndarray,
    prf_hz: float,
    cf_hz: Tuple[float, float] = (30.0, 250.0),
    ca_hz: Tuple[float, float] = (400.0, 750.0),
) -> Dict[str, float]:
    """
    R-1: Band occupancy separation.

    We approximate S(f) via a simple FFT over the raw rf tensor.
    """

    t_len = rf.shape[0]
    # One-sided spectrum, frequencies in Hz.
    freqs = np.fft.rfftfreq(t_len, d=1.0 / prf_hz)
    spec = np.fft.rfft(rf, axis=0)
    psd = np.abs(spec) ** 2  # (F, H, W)

    # Peak frequency index per pixel.
    k_peak = psd.argmax(axis=0)
    f_peak = freqs[k_peak]

    cf_lo, cf_hi = cf_hz
    ca_lo, ca_hi = ca_hz

    h1 = mask_flow.astype(bool)
    h0 = mask_bg.astype(bool)

    # Require spatial alignment between rf-derived PSD and masks.
    if f_peak.shape != h1.shape:
        raise RuntimeError(
            f"rf_tensor PSD grid shape {f_peak.shape} does not match mask shape {h1.shape}"
        )

    in_cf_h1 = (f_peak >= cf_lo) & (f_peak <= cf_hi) & h1
    in_ca_h0 = (f_peak >= ca_lo) & (f_peak <= ca_hi) & h0

    p_f = float(in_cf_h1.sum() / max(1, h1.sum()))
    p_a = float(in_ca_h0.sum() / max(1, h0.sum()))

    return {
        "p_f_peak_in_Cf_given_H1": p_f,
        "p_a_peak_in_Ca_given_H0": p_a,
        "cf_lo_hz": cf_lo,
        "cf_hi_hz": cf_hi,
        "ca_lo_hz": ca_lo,
        "ca_hi_hz": ca_hi,
    }


def _check_r1_from_bandratio_telemetry(meta: Dict) -> Dict[str, float] | None:
    """
    Approximate R-1 using whitened band-ratio / PSD telemetry when a
    spatially aligned rf_tensor is not available.

    We reuse the band-ratio peak statistics recorded in
    stap_fallback_telemetry['band_ratio_stats'], which track:
      - br_flow_peak_fraction_nonbg: P(f_peak in Cf | non-bg tiles)
      - br_alias_peak_fraction_bg:  P(f_peak in Ca | bg tiles)

    The Cf/Ca bands are taken from 'band_ratio_bands_hz'.
    """

    tele = meta.get("stap_fallback_telemetry") or {}
    br_stats = tele.get("band_ratio_stats") or {}
    bands = tele.get("band_ratio_bands_hz") or {}

    if not br_stats or not bands:
        return None

    p_f = br_stats.get("br_flow_peak_fraction_nonbg")
    p_a = br_stats.get("br_alias_peak_fraction_bg")
    if p_f is None or p_a is None:
        return None

    cf_lo = float(bands.get("flow_low_hz", 0.0))
    cf_hi = float(bands.get("flow_high_hz", 0.0))
    alias_center = float(bands.get("alias_center_hz", 0.0))
    alias_width = float(bands.get("alias_width_hz", 0.0))
    ca_lo = alias_center - alias_width
    ca_hi = alias_center + alias_width

    return {
        "p_f_peak_in_Cf_given_H1": float(p_f),
        "p_a_peak_in_Ca_given_H0": float(p_a),
        "cf_lo_hz": cf_lo,
        "cf_hi_hz": cf_hi,
        "ca_lo_hz": ca_lo,
        "ca_hi_hz": ca_hi,
    }


def check_r2_alias_separation(
    m_alias: np.ndarray,
    mask_flow: np.ndarray,
    mask_bg: np.ndarray,
) -> Dict[str, float]:
    """
    R-2: Alias score class separation.

    We treat m_alias as an alias score that increases when alias-band
    energy dominates (e.g. log(E_a/E_f)).
    """
    m_alias = np.asarray(m_alias, dtype=float)
    h1 = mask_flow.astype(bool)
    h0 = mask_bg.astype(bool)

    alias_h0 = m_alias[h0]
    alias_h1 = m_alias[h1]

    if alias_h0.size == 0 or alias_h1.size == 0:
        return {
            "median_alias_H0": np.nan,
            "median_alias_H1": np.nan,
            "delta_median": np.nan,
        }

    med_h0 = float(np.median(alias_h0))
    med_h1 = float(np.median(alias_h1))
    delta = med_h0 - med_h1

    return {
        "median_alias_H0": med_h0,
        "median_alias_H1": med_h1,
        "delta_median": delta,
    }


def check_r3_headroom(
    base_score: np.ndarray,
    stap_score: np.ndarray,
    mask_flow: np.ndarray,
    mask_bg: np.ndarray,
    fprs: Sequence[float] = (1e-4, 3e-4, 1e-3),
) -> Dict[str, Dict[str, float]]:
    """
    R-3: Baseline headroom and STAP shoulder region at low FPR.
    """

    base_score = np.asarray(base_score, dtype=float)
    stap_score = np.asarray(stap_score, dtype=float)
    h1 = mask_flow.astype(bool)
    h0 = mask_bg.astype(bool)

    base_pos = base_score[h1]
    base_neg = base_score[h0]
    stap_pos = stap_score[h1]
    stap_neg = stap_score[h0]

    out: Dict[str, Dict[str, float]] = {}
    for fpr in fprs:
        thr_b, tpr_b = _tpr_at_fpr(base_pos, base_neg, fpr)
        thr_s, tpr_s = _tpr_at_fpr(stap_pos, stap_neg, fpr)
        out[f"fpr={fpr:g}"] = {
            "thr_base": thr_b,
            "tpr_base": tpr_b,
            "thr_stap": thr_s,
            "tpr_stap": tpr_s,
        }
    return out


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Check KA-friendly regime conditions (R-1/R-2/R-3) on a HAB k-Wave bundle.\n"
            "Bundle should be a directory containing meta.json, mask_flow.npy, "
            "mask_bg.npy, base_band_ratio_map.npy, base_score_map.npy, stap_score_map.npy "
            "(and optionally score_base.npy / score_stap.npy)."
        )
    )
    parser.add_argument(
        "bundle",
        help=(
            "Path to acceptance bundle "
            "(e.g. runs/pilot/r4c_kwave_hab_seed2/pw_7.5MHz_5ang_5ens_320T_seed2)"
        ),
    )
    parser.add_argument(
        "--cf",
        type=float,
        nargs=2,
        metavar=("F_LO", "F_HI"),
        default=(30.0, 250.0),
        help="Flow band Cf in Hz used for R-1 (default: 30 250).",
    )
    parser.add_argument(
        "--ca",
        type=float,
        nargs=2,
        metavar=("A_LO", "A_HI"),
        default=(400.0, 750.0),
        help="Alias band Ca in Hz used for R-1 (default: 400 750).",
    )
    parser.add_argument(
        "--fprs",
        type=float,
        nargs="+",
        default=[1e-4, 3e-4, 1e-3],
        help="FPR levels at which to report baseline/STAP TPR (default: 1e-4 3e-4 1e-3).",
    )
    parser.add_argument(
        "--score-mode",
        type=str,
        default="auto",
        choices=["auto", "msd", "stap", "pd", "band_ratio"],
        help=(
            "Score definition used for R-3 headroom. "
            "'auto' uses meta['score_stats']['mode'] when available, "
            "otherwise falls back to 'msd'."
        ),
    )
    args = parser.parse_args()

    bundle_dir = os.path.abspath(args.bundle)
    data = _load_bundle(bundle_dir)
    meta = data["meta"]

    prf_hz = meta.get("prf_hz")
    if prf_hz is None:
        raise RuntimeError("prf_hz missing from meta.json; cannot compute R-1.")

    print(f"# HAB contract check for bundle: {bundle_dir}")
    print(f"PRF: {prf_hz} Hz, Lt={meta.get('Lt')}, seed={meta.get('seed')}")

    # Optional baseline telemetry, e.g. from MC-SVD or HOSVD. When a
    # literature-style energy-fraction SVD baseline is used, this block
    # surfaces the chosen rank and energy fraction removed so we can
    # verify that the SVD thresholding behaves sensibly across regimes.
    baseline_stats_root = meta.get("baseline_stats") or {}
    baseline_stats = baseline_stats_root.get("pd") or meta.get("stap_fallback_telemetry") or {}
    baseline_type = baseline_stats.get("baseline_type") or baseline_stats_root.get("baseline_type")
    svd_rank_removed = baseline_stats.get("svd_rank_removed")
    svd_energy_removed = baseline_stats.get("svd_energy_removed_frac")
    if baseline_type:
        line = f"# Baseline: {baseline_type}"
        if svd_rank_removed is not None:
            line += f", svd_rank_removed={svd_rank_removed}"
        if svd_energy_removed is not None:
            line += f", svd_energy_removed_frac={svd_energy_removed:.3f}"
        print(line)
    # Band-ratio bin sanity (important for short-ensemble real datasets).
    br_stats = baseline_stats_root.get("band_ratio_stats") if isinstance(baseline_stats_root, dict) else None
    if isinstance(br_stats, dict) and br_stats.get("br_series_len") is not None:
        try:
            df_hz = br_stats.get("br_df_hz")
            flow_bins = br_stats.get("br_flow_bins")
            alias_bins = br_stats.get("br_alias_bins")
            flow_lo = br_stats.get("br_flow_bin_lo")
            flow_hi = br_stats.get("br_flow_bin_hi")
            alias_lo = br_stats.get("br_alias_bin_lo")
            alias_hi = br_stats.get("br_alias_bin_hi")
            overlap = br_stats.get("br_bins_overlap")
            flow_nodc = br_stats.get("br_flow_bins_nodc")
            alias_nodc = br_stats.get("br_alias_bins_nodc")
            peak_bin_p50 = br_stats.get("br_peak_bin_p50")
            peak_bin_p90 = br_stats.get("br_peak_bin_p90")
            print(
                "# Band-ratio bins: "
                f"T={br_stats.get('br_series_len')}, df≈{df_hz:.3g} Hz, "
                f"flow_bins={flow_bins} (bin {flow_lo}..{flow_hi}, nodc={flow_nodc}), "
                f"alias_bins={alias_bins} (bin {alias_lo}..{alias_hi}, nodc={alias_nodc}), "
                f"overlap={overlap}, peak_bin_p50={peak_bin_p50}, peak_bin_p90={peak_bin_p90}"
            )
            if bool(overlap) or int(flow_nodc or 0) <= 0 or int(alias_nodc or 0) <= 0:
                print("# Band-ratio sanity: FAIL (empty/overlap bins) -> treat KA contract as invalid")
        except Exception:
            pass

    # R-2: alias separation using band-ratio map.
    # Use an explicit alias metric map when present; otherwise invert the
    # baseline band-ratio map (which is log(Ef/(gamma*Ea))) so that larger
    # values correspond to more alias-like behavior.
    m_alias_map = data.get("base_m_alias")
    if m_alias_map is None:
        m_alias_map = -np.asarray(data["base_band_ratio"], dtype=float)

    r2 = check_r2_alias_separation(m_alias_map, data["mask_flow"], data["mask_bg"])
    print("\n[R-2] Alias score separation (m_alias = log(E_a/E_f))")
    print(
        f"  median(m_alias | H0) = {r2['median_alias_H0']:.3f}, "
        f"median(m_alias | H1) = {r2['median_alias_H1']:.3f}, "
        f"delta = {r2['delta_median']:.3f}"
    )

    # R-3: baseline headroom / STAP shoulder. Align the score with the
    # detector actually used in the run when possible.
    score_mode_meta = None
    score_stats = meta.get("score_stats") or {}
    if isinstance(score_stats, dict):
        score_mode_meta = score_stats.get("mode")
    score_mode = args.score_mode
    if score_mode == "auto":
        score_mode = score_mode_meta or meta.get("score_pool_default") or "msd"
    if score_mode not in {"msd", "pd", "band_ratio", "stap"}:
        raise RuntimeError(f"Unrecognized score_mode {score_mode!r}")
    if score_mode == "band_ratio":
        base_score = data["base_band_ratio"]
        stap_score = data.get("stap_score_pool")
        if stap_score is None:
            stap_score = data.get("stap_band_ratio")
        if stap_score is None:
            raise RuntimeError(
                "Band-ratio score_mode requested but neither stap_score_pool_map.npy "
                "nor stap_band_ratio_map.npy is present in the bundle."
            )
    elif score_mode == "pd":
        pd_base = data.get("pd_base")
        pd_stap = data.get("pd_stap")
        if pd_base is None or pd_stap is None:
            raise RuntimeError(
                "PD score_mode requested but pd_base.npy / pd_stap.npy are missing."
            )
        # PD-mode convention: ROC is evaluated by thresholding the right tail of
        # the exported PD score map `score_pd_*.npy` (higher = more flow evidence).
        #
        # Prefer explicit score maps when present to avoid sign ambiguity across
        # legacy bundles.
        base_score = data.get("score_pd_base")
        stap_score = data.get("score_pd_stap")
        if base_score is None:
            roc_conv = (meta.get("pd_mode") or {}).get("roc_convention") if isinstance(meta, dict) else None
            roc_conv = str(roc_conv or "").lower()
            base_score = (-pd_base) if "lower_tail_on_pd" in roc_conv else pd_base
        if stap_score is None:
            roc_conv = (meta.get("pd_mode") or {}).get("roc_convention") if isinstance(meta, dict) else None
            roc_conv = str(roc_conv or "").lower()
            stap_score = (-pd_stap) if "lower_tail_on_pd" in roc_conv else pd_stap
    elif score_mode == "stap":
        # vNext primary score exports (right-tail, higher = more flow evidence).
        base_score = data.get("score_base")
        if base_score is None:
            base_score = data["base_score"]
        stap_score = data.get("score_stap")
        if stap_score is None:
            stap_score = data.get("score_stap_preka")
        if stap_score is None:
            stap_score = data["stap_score"]
    else:  # "msd" (legacy stap_score_map)
        base_score = data["base_score"]
        stap_score = data["stap_score"]

    r3 = check_r3_headroom(
        base_score,
        stap_score,
        data["mask_flow"],
        data["mask_bg"],
        fprs=args.fprs,
    )
    print("\n[R-3] Baseline headroom and STAP shoulder at low FPR")
    print(f"  score_mode = {score_mode}")
    for key in sorted(r3.keys()):
        stats = r3[key]
        print(
            f"  {key}: "
            f"thr_base={stats['thr_base']:.4g}, tpr_base={stats['tpr_base']:.3f}, "
            f"thr_stap={stats['thr_stap']:.4g}, tpr_stap={stats['tpr_stap']:.3f}"
        )

    # R-1: band occupancy separation. Prefer rf_tensor when spatially aligned;
    # otherwise fall back to band-ratio / PSD telemetry if available.
    try:
        rf = _load_rf_tensor(bundle_dir, meta)
        cf_lo, cf_hi = args.cf
        ca_lo, ca_hi = args.ca
        r1 = check_r1_band_occupancy(
            rf,
            data["mask_flow"],
            data["mask_bg"],
            prf_hz=float(prf_hz),
            cf_hz=(cf_lo, cf_hi),
            ca_hz=(ca_lo, ca_hi),
        )
        print("\n[R-1] Band occupancy separation (fft-based PSD on rf_tensor)")
        print(
            f"  Cf = [{r1['cf_lo_hz']:.1f}, {r1['cf_hi_hz']:.1f}] Hz, "
            f"Ca = [{r1['ca_lo_hz']:.1f}, {r1['ca_hi_hz']:.1f}] Hz"
        )
        print(
            f"  P(f_peak∈Cf | H1) ≈ {r1['p_f_peak_in_Cf_given_H1']:.3f}, "
            f"P(f_peak∈Ca | H0) ≈ {r1['p_a_peak_in_Ca_given_H0']:.3f}"
        )
    except (FileNotFoundError, RuntimeError) as exc:
        approx = _check_r1_from_bandratio_telemetry(meta)
        if approx is None:
            print(
                f"\n[R-1] Band occupancy check skipped: {exc}. "
                "To enable this check, save a slow-time stack with spatial shape matching "
                "the masks one level above the bundle (rf_tensor.npy) or export PSD telemetry "
                "explicitly."
            )
        else:
            print(
                f"\n[R-1] Band occupancy separation (whitened PSD / band-ratio telemetry; "
                f"rf_tensor path failed: {exc})"
            )
            print(
                f"  Cf = [{approx['cf_lo_hz']:.1f}, {approx['cf_hi_hz']:.1f}] Hz, "
                f"Ca = [{approx['ca_lo_hz']:.1f}, {approx['ca_hi_hz']:.1f}] Hz"
            )
            print(
                f"  P(f_peak∈Cf | H1) ≈ {approx['p_f_peak_in_Cf_given_H1']:.3f}, "
                f"P(f_peak∈Ca | H0) ≈ {approx['p_a_peak_in_Ca_given_H0']:.3f}"
            )


if __name__ == "__main__":
    main()
