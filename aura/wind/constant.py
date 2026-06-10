"""Stage 1 wind field: spatially and temporally constant."""

from __future__ import annotations

import numpy as np

from aura.wind.base import IWindField


class ConstantWind(IWindField):
    def __init__(self, u: float = 3.0, v: float = 0.0, w: float = 0.0):
        self._uvw = np.asarray([u, v, w], dtype=np.float64)

    def sample(self, xyz: np.ndarray, t: float) -> np.ndarray:
        xyz = np.asarray(xyz)
        out_shape = xyz.shape
        return np.broadcast_to(self._uvw, out_shape).copy()

    def is_constant(self) -> bool:
        return True

    def uncertainty(self, xyz: np.ndarray, t: float) -> np.ndarray:
        return np.zeros_like(np.asarray(xyz))

    def __repr__(self) -> str:
        u, v, w = self._uvw
        return f"ConstantWind(u={u:.2f}, v={v:.2f}, w={w:.2f})"
