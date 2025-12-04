import os
import random

import numpy as np
import pytest

try:
    import torch
except Exception:  # pragma: no cover - torch optional
    torch = None


@pytest.fixture(autouse=True)
def _seed_everything():
    """
    Provide deterministic seeding across numpy/random/torch for each test.
    """
    seed = int(os.environ.get("TEST_SEED", "12345"))
    random.seed(seed)
    np.random.seed(seed)
    if torch is not None:
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        try:
            os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":16:8")
            torch.use_deterministic_algorithms(True)
        except Exception:
            pass
    yield
    if torch is not None:
        try:
            torch.use_deterministic_algorithms(False)
        except Exception:
            pass
