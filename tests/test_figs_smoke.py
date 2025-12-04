# tests/test_figs_smoke.py

import os
import subprocess
import sys
from pathlib import Path

ENV = dict(os.environ)
ENV["FAST_FIGS"] = "1"
ROOT = Path(__file__).resolve().parents[1]
ENV["PYTHONPATH"] = os.pathsep.join(
    [str(ROOT)] + ([ENV.get("PYTHONPATH")] if ENV.get("PYTHONPATH") else [])
)


def _run(script):
    cmd = [sys.executable, script]
    subprocess.check_call(cmd, env=ENV)


def test_fig1():
    _run("figs/fig1_before_after.py")


def test_fig2():
    _run("figs/fig2_roc_calibrated.py")


def test_fig4():
    _run("figs/fig4_latency_angle_trade.py")


def test_fig5():
    _run("figs/fig5_ablation_bars.py")


def test_fig6():
    _run("figs/fig6_telemetry_rank.py")
