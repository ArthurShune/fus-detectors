import csv
import json
import sys


def test_simus_policy_headline_smoke(tmp_path):
    from scripts import simus_policy_headline as headline

    rows = [
        {
            "case_key": "case_a",
            "seed": "1",
            "simus_profile": "ClinIntraOp-Pf-v1",
            "motion_scale": "0.25",
            "role": "baseline",
            "stap_profile": "",
            "motion_disp_rms_px": "1.5",
            "reg_shift_p90": "2.0",
        },
        {
            "case_key": "case_a",
            "seed": "1",
            "simus_profile": "ClinIntraOp-Pf-v1",
            "motion_scale": "0.25",
            "role": "stap",
            "stap_profile": "Brain-SIMUS-Clin",
            "motion_disp_rms_px": "1.5",
            "reg_shift_p90": "2.0",
            "auc_main_vs_bg": "0.50",
            "auc_main_vs_nuisance": "0.40",
            "fpr_nuisance_match@0p5": "0.60",
        },
        {
            "case_key": "case_a",
            "seed": "1",
            "simus_profile": "ClinIntraOp-Pf-v1",
            "motion_scale": "0.25",
            "role": "stap",
            "stap_profile": "Brain-SIMUS-Clin-MotionRobust-v0",
            "motion_disp_rms_px": "1.5",
            "reg_shift_p90": "2.0",
            "auc_main_vs_bg": "0.60",
            "auc_main_vs_nuisance": "0.55",
            "fpr_nuisance_match@0p5": "0.35",
        },
        {
            "case_key": "case_a",
            "seed": "1",
            "simus_profile": "ClinIntraOp-Pf-v1",
            "motion_scale": "0.25",
            "role": "stap",
            "stap_profile": "Brain-SIMUS-Clin-MotionMidRobust-v0",
            "motion_disp_rms_px": "1.5",
            "reg_shift_p90": "2.0",
            "auc_main_vs_bg": "0.58",
            "auc_main_vs_nuisance": "0.50",
            "fpr_nuisance_match@0p5": "0.40",
        },
        {
            "case_key": "case_b",
            "seed": "2",
            "simus_profile": "ClinMobile-Pf-v1",
            "motion_scale": "1.0",
            "role": "baseline",
            "stap_profile": "",
            "motion_disp_rms_px": "2.5",
            "reg_shift_p90": "2.3",
        },
        {
            "case_key": "case_b",
            "seed": "2",
            "simus_profile": "ClinMobile-Pf-v1",
            "motion_scale": "1.0",
            "role": "stap",
            "stap_profile": "Brain-SIMUS-Clin",
            "motion_disp_rms_px": "2.5",
            "reg_shift_p90": "2.3",
            "auc_main_vs_bg": "0.45",
            "auc_main_vs_nuisance": "0.30",
            "fpr_nuisance_match@0p5": "0.80",
        },
        {
            "case_key": "case_b",
            "seed": "2",
            "simus_profile": "ClinMobile-Pf-v1",
            "motion_scale": "1.0",
            "role": "stap",
            "stap_profile": "Brain-SIMUS-Clin-MotionRobust-v0",
            "motion_disp_rms_px": "2.5",
            "reg_shift_p90": "2.3",
            "auc_main_vs_bg": "0.50",
            "auc_main_vs_nuisance": "0.42",
            "fpr_nuisance_match@0p5": "0.55",
        },
        {
            "case_key": "case_b",
            "seed": "2",
            "simus_profile": "ClinMobile-Pf-v1",
            "motion_scale": "1.0",
            "role": "stap",
            "stap_profile": "Brain-SIMUS-Clin-MotionMidRobust-v0",
            "motion_disp_rms_px": "2.5",
            "reg_shift_p90": "2.3",
            "auc_main_vs_bg": "0.57",
            "auc_main_vs_nuisance": "0.60",
            "fpr_nuisance_match@0p5": "0.32",
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

    out_csv = tmp_path / "headline.csv"
    out_json = tmp_path / "headline.json"
    argv_prev = sys.argv[:]
    sys.argv = [
        "simus_policy_headline.py",
        "--search-csv",
        str(search_csv),
        "--out-csv",
        str(out_csv),
        "--out-json",
        str(out_json),
    ]
    try:
        headline.main()
    finally:
        sys.argv = argv_prev

    payload = json.loads(out_json.read_text(encoding="utf-8"))
    assert payload["summary_by_motion_scale"]["0.25"]["mean_policy_auc_main_vs_bg"] == 0.6
    assert payload["summary_by_motion_scale"]["1.0"]["mean_policy_auc_main_vs_nuisance"] == 0.6

    argv_prev = sys.argv[:]
    sys.argv = [
        "simus_policy_headline.py",
        "--search-csv",
        str(search_csv),
        "--policy",
        "Brain-SIMUS-Clin-RegShiftP90-v0",
        "--out-csv",
        str(out_csv),
        "--out-json",
        str(out_json),
    ]
    try:
        headline.main()
    finally:
        sys.argv = argv_prev
    payload = json.loads(out_json.read_text(encoding="utf-8"))
    rows = {row["case_key"]: row for row in payload["rows"]}
    assert rows["case_a"]["policy_profile"] == "Brain-SIMUS-Clin-MotionRobust-v0"
    assert rows["case_b"]["policy_profile"] == "Brain-SIMUS-Clin-MotionMidRobust-v0"
