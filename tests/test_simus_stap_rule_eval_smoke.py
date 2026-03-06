import csv
import json
import sys


def test_simus_stap_rule_eval_smoke(tmp_path):
    from scripts import simus_stap_rule_eval as rule_eval

    rows = [
        {
            "case_key": "a",
            "simus_profile": "ClinIntraOp-Pf-v1",
            "seed": "1",
            "motion_scale": "0.25",
            "role": "baseline",
            "stap_profile": "",
            "reg_shift_rms": "1.0",
            "reg_shift_p90": "1.4",
            "reg_psr_median": "14.0",
            "auc_main_vs_bg": "0.60",
            "auc_main_vs_nuisance": "0.05",
            "fpr_nuisance_match@0p5": "1.0",
        },
        {
            "case_key": "a",
            "simus_profile": "ClinIntraOp-Pf-v1",
            "seed": "1",
            "motion_scale": "0.25",
            "role": "stap",
            "stap_profile": "Brain-SIMUS-Clin-MotionShort-v0",
            "reg_shift_rms": "1.0",
            "reg_shift_p90": "1.4",
            "reg_psr_median": "14.0",
            "auc_main_vs_bg": "0.70",
            "auc_main_vs_nuisance": "0.50",
            "fpr_nuisance_match@0p5": "0.5",
            "delta_auc_main_vs_bg": "0.10",
            "delta_auc_main_vs_nuisance": "0.45",
            "delta_fpr_nuisance_match@0p5": "-0.50",
        },
        {
            "case_key": "a",
            "simus_profile": "ClinIntraOp-Pf-v1",
            "seed": "1",
            "motion_scale": "0.25",
            "role": "stap",
            "stap_profile": "Brain-SIMUS-Clin-MotionMid-v0",
            "reg_shift_rms": "1.0",
            "reg_shift_p90": "1.4",
            "reg_psr_median": "14.0",
            "auc_main_vs_bg": "0.66",
            "auc_main_vs_nuisance": "0.42",
            "fpr_nuisance_match@0p5": "0.55",
            "delta_auc_main_vs_bg": "0.06",
            "delta_auc_main_vs_nuisance": "0.37",
            "delta_fpr_nuisance_match@0p5": "-0.45",
        },
        {
            "case_key": "a",
            "simus_profile": "ClinIntraOp-Pf-v1",
            "seed": "1",
            "motion_scale": "0.25",
            "role": "stap",
            "stap_profile": "Brain-SIMUS-Clin-MotionLong-v0",
            "reg_shift_rms": "1.0",
            "reg_shift_p90": "1.4",
            "reg_psr_median": "14.0",
            "auc_main_vs_bg": "0.58",
            "auc_main_vs_nuisance": "0.30",
            "fpr_nuisance_match@0p5": "0.70",
            "delta_auc_main_vs_bg": "-0.02",
            "delta_auc_main_vs_nuisance": "0.25",
            "delta_fpr_nuisance_match@0p5": "-0.30",
        },
        {
            "case_key": "a",
            "simus_profile": "ClinIntraOp-Pf-v1",
            "seed": "1",
            "motion_scale": "0.25",
            "role": "stap",
            "stap_profile": "Brain-SIMUS-Clin-MotionRobust-v0",
            "reg_shift_rms": "1.0",
            "reg_shift_p90": "1.4",
            "reg_psr_median": "14.0",
            "auc_main_vs_bg": "0.68",
            "auc_main_vs_nuisance": "0.48",
            "fpr_nuisance_match@0p5": "0.52",
            "delta_auc_main_vs_bg": "0.08",
            "delta_auc_main_vs_nuisance": "0.43",
            "delta_fpr_nuisance_match@0p5": "-0.48",
        },
        {
            "case_key": "b",
            "simus_profile": "ClinMobile-Pf-v1",
            "seed": "2",
            "motion_scale": "1.0",
            "role": "baseline",
            "stap_profile": "",
            "reg_shift_rms": "2.5",
            "reg_shift_p90": "3.1",
            "reg_psr_median": "9.0",
            "auc_main_vs_bg": "0.40",
            "auc_main_vs_nuisance": "0.02",
            "fpr_nuisance_match@0p5": "1.0",
        },
        {
            "case_key": "b",
            "simus_profile": "ClinMobile-Pf-v1",
            "seed": "2",
            "motion_scale": "1.0",
            "role": "stap",
            "stap_profile": "Brain-SIMUS-Clin-MotionShort-v0",
            "reg_shift_rms": "2.5",
            "reg_shift_p90": "3.1",
            "reg_psr_median": "9.0",
            "auc_main_vs_bg": "0.45",
            "auc_main_vs_nuisance": "0.30",
            "fpr_nuisance_match@0p5": "0.8",
            "delta_auc_main_vs_bg": "0.05",
            "delta_auc_main_vs_nuisance": "0.28",
            "delta_fpr_nuisance_match@0p5": "-0.2",
        },
        {
            "case_key": "b",
            "simus_profile": "ClinMobile-Pf-v1",
            "seed": "2",
            "motion_scale": "1.0",
            "role": "stap",
            "stap_profile": "Brain-SIMUS-Clin-MotionMid-v0",
            "reg_shift_rms": "2.5",
            "reg_shift_p90": "3.1",
            "reg_psr_median": "9.0",
            "auc_main_vs_bg": "0.50",
            "auc_main_vs_nuisance": "0.42",
            "fpr_nuisance_match@0p5": "0.55",
            "delta_auc_main_vs_bg": "0.10",
            "delta_auc_main_vs_nuisance": "0.40",
            "delta_fpr_nuisance_match@0p5": "-0.45",
        },
        {
            "case_key": "b",
            "simus_profile": "ClinMobile-Pf-v1",
            "seed": "2",
            "motion_scale": "1.0",
            "role": "stap",
            "stap_profile": "Brain-SIMUS-Clin-MotionLong-v0",
            "reg_shift_rms": "2.5",
            "reg_shift_p90": "3.1",
            "reg_psr_median": "9.0",
            "auc_main_vs_bg": "0.52",
            "auc_main_vs_nuisance": "0.50",
            "fpr_nuisance_match@0p5": "0.45",
            "delta_auc_main_vs_bg": "0.12",
            "delta_auc_main_vs_nuisance": "0.48",
            "delta_fpr_nuisance_match@0p5": "-0.55",
        },
        {
            "case_key": "b",
            "simus_profile": "ClinMobile-Pf-v1",
            "seed": "2",
            "motion_scale": "1.0",
            "role": "stap",
            "stap_profile": "Brain-SIMUS-Clin-MotionRobust-v0",
            "reg_shift_rms": "2.5",
            "reg_shift_p90": "3.1",
            "reg_psr_median": "9.0",
            "auc_main_vs_bg": "0.48",
            "auc_main_vs_nuisance": "0.40",
            "fpr_nuisance_match@0p5": "0.60",
            "delta_auc_main_vs_bg": "0.08",
            "delta_auc_main_vs_nuisance": "0.38",
            "delta_fpr_nuisance_match@0p5": "-0.40",
        },
    ]
    search_csv = tmp_path / "search.csv"
    fieldnames = []
    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)
    with search_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    out_csv = tmp_path / "rule.csv"
    out_json = tmp_path / "rule.json"
    argv_prev = sys.argv[:]
    sys.argv = [
        "simus_stap_rule_eval.py",
        "--search-csv",
        str(search_csv),
        "--out-csv",
        str(out_csv),
        "--out-json",
        str(out_json),
    ]
    try:
        rule_eval.main()
    finally:
        sys.argv = argv_prev

    payload = json.loads(out_json.read_text(encoding="utf-8"))
    assert "summary_by_rule" in payload
    assert "fixed_best" in payload["summary_by_rule"]
