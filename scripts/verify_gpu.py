"""Quick GPU stack smoke-test."""

from __future__ import annotations


def check_torch() -> None:
    print("=== Torch ===")
    try:
        import torch

        print("torch:", torch.__version__, "cuda:", torch.cuda.is_available())
        if torch.cuda.is_available():
            print("torch device:", torch.cuda.get_device_name(0))
    except Exception as exc:  # pragma: no cover
        print("Torch import failed:", exc)


def check_cupy() -> None:
    print("\n=== CuPy ===")
    try:
        import cupy as cp

        arr = cp.arange(4)
        print("cupy ok; device:", cp.cuda.runtime.getDevice())
        print("sample result:", cp.asnumpy(arr**2))
    except Exception as exc:  # pragma: no cover
        print("CuPy import failed:", exc)


def check_kwave() -> None:
    print("\n=== k-Wave-python ===")
    try:
        import kwave

        print("k-Wave import ok:", getattr(kwave, "__version__", "unknown"))
    except Exception as exc:  # pragma: no cover
        print("k-Wave import failed:", exc)


if __name__ == "__main__":
    check_torch()
    check_cupy()
    check_kwave()
