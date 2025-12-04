import numpy as np

from pipeline.stap.temporal import _select_flow_grid


def test_select_flow_grid_caps_to_lt_minus_motion_rank() -> None:
    fd = np.linspace(-900.0, 900.0, 13)
    max_tones = 5  # e.g., Lt - motion_rank = 5
    selected = _select_flow_grid(fd, max_tones=max_tones, min_tones=3)
    assert selected.size <= max_tones
    assert selected.size % 2 == 1
    assert np.all(np.diff(selected) >= 0.0)


def test_select_flow_grid_expands_when_available() -> None:
    fd = np.array([-400.0, -200.0, 0.0, 200.0, 400.0])
    selected = _select_flow_grid(fd, max_tones=7, min_tones=3)
    assert selected.size == fd.size
    assert np.allclose(selected, fd)
