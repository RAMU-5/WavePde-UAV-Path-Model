"""Low-cost checks for the installable model package."""
from __future__ import annotations

import subprocess
import sys


def test_model_import_is_lazy() -> None:
    code = "import model, sys; assert 'jax' not in sys.modules; assert 'plan_path' in dir(model)"
    subprocess.run([sys.executable, "-c", code], check=True)


def test_public_api_is_declared() -> None:
    import model

    assert {
        "WaveConfig",
        "build_static_fields",
        "plan_path",
        "solve_backward",
        "solve_forward",
    }.issubset(model.__all__)
