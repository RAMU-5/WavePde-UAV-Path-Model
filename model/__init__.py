"""Public API for the WavePDE routing model.

Imports stay lazy because loading the solver initializes JAX. Lightweight
utilities can therefore import :mod:`model` without selecting a JAX backend.
"""
from __future__ import annotations

from importlib import import_module
from typing import Any

__all__ = [
    "PathResult",
    "StaticFields",
    "WaveConfig",
    "build_static_fields",
    "plan_path",
    "solve_backward",
    "solve_forward",
]

_PUBLIC_MODULES = {
    "PathResult": ".plan",
    "plan_path": ".plan",
    "StaticFields": ".wave_api",
    "WaveConfig": ".wave_api",
    "build_static_fields": ".wave_api",
    "solve_backward": ".wave_api",
    "solve_forward": ".wave_api",
}


def __getattr__(name: str) -> Any:
    """Load a public solver object on first access."""
    module_name = _PUBLIC_MODULES.get(name)
    if module_name is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    value = getattr(import_module(module_name, __name__), name)
    globals()[name] = value
    return value


def __dir__() -> list[str]:
    return sorted(set(globals()) | set(__all__))
