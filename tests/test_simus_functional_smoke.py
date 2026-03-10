from __future__ import annotations

from pathlib import Path

import numpy as np

from sim.simus.functional import FunctionalDesignSpec, write_functional_case


def test_write_functional_case_smoke(tmp_path: Path) -> None:
    out = write_functional_case(
        out_root=tmp_path / "func_case",
        base_profile="ClinMobile-Pf-v2",
        tier="smoke",
        seed=0,
        null_run=False,
        design_spec=FunctionalDesignSpec(ensemble_count=4, off_length=1, on_length=1, activation_gain=0.10),
        max_workers=1,
        threads_per_worker=1,
        reuse_existing=False,
    )
    case_root = Path(out["case_root"])
    assert (case_root / "design.json").is_file()
    assert (case_root / "task_regressor.npy").is_file()
    assert (case_root / "hemo_regressor.npy").is_file()
    assert (case_root / "ensemble_table.csv").is_file()
    assert (case_root / "ensemble_000" / "dataset" / "icube.npy").is_file()
    roi = np.load(case_root / "mask_activation_roi.npy")
    assert roi.dtype == bool
    assert int(roi.sum()) > 0
