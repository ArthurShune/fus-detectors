"""Basic import smoke tests."""

import kwave

import pipeline


def test_imports() -> None:
    assert pipeline is not None
    assert kwave is not None
