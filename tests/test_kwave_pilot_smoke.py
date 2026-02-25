# tests/test_kwave_pilot_smoke.py
import subprocess
import sys
from pathlib import Path


def test_pilot_r1_smoke(tmp_path):
    outdir = tmp_path / "pilot"
    cmd = [
        sys.executable,
        "sim/kwave/pilot_r1.py",
        "--out",
        str(outdir),
        "--Nx",
        "48",
        "--Ny",
        "48",
        "--ncycles",
        "1",
        "--pulses",
        "4",
        "--angles",
        "0",
        "--f0",
        "6.0",
        "--synthetic",
        "--stap-device",
        "cpu",
        "--force-cpu",
    ]
    subprocess.check_call(cmd)
    bundle_dir = next(Path(outdir).glob("pw_*"), None)
    assert bundle_dir is not None
    base_pos = bundle_dir / "base_pos.npy"
    pd_base = bundle_dir / "pd_base.npy"
    assert base_pos.exists() and pd_base.exists()
    assert (bundle_dir / "stap_score_map.npy").exists()
    assert (bundle_dir / "score_base.npy").exists()
    assert (bundle_dir / "score_stap_preka.npy").exists()
    assert (bundle_dir / "score_stap.npy").exists()
    assert (bundle_dir / "score_name.txt").exists()


def test_pilot_motion_smoke(tmp_path):
    outdir = tmp_path / "pilot"
    cmd = [
        sys.executable,
        "sim/kwave/pilot_motion.py",
        "--out",
        str(outdir),
        "--Nx",
        "48",
        "--Ny",
        "48",
        "--ncycles",
        "1",
        "--ensembles",
        "2",
        "--pulses",
        "3",
        "--angles",
        "0",
        "--f0",
        "6.0",
        "--synthetic",
        "--stap-device",
        "cpu",
        "--force-cpu",
    ]
    subprocess.check_call(cmd)
    bundle_dir = next(Path(outdir).glob("pw_*"), None)
    assert bundle_dir is not None
    assert (bundle_dir / "stap_pos.npy").exists()
    assert (bundle_dir / "mask_flow.npy").exists()
    assert (bundle_dir / "stap_score_map.npy").exists()
    assert (bundle_dir / "score_base.npy").exists()
    assert (bundle_dir / "score_stap_preka.npy").exists()
    assert (bundle_dir / "score_stap.npy").exists()
    assert (bundle_dir / "score_name.txt").exists()
