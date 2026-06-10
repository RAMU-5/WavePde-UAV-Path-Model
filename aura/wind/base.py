"""IWindField — abstract wind-field interface.

Every routing/control component in Aura that needs wind talks to this interface
only. Stage 1 ships ConstantWind. Stage 2 swaps in SCALEDWind without touching
any consumer code.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Tuple

import numpy as np


class IWindField(ABC):
    """Sample the 3D wind field at given world coordinates and time.

    Coordinates are in projected meters (EPSG:32650 for Stage 1). Time is
    seconds since simulation epoch. Returned vectors are (u, v, w) in m/s,
    same axes as the projected CRS (x = easting, y = northing, z = altitude).
    """

    @abstractmethod
    def sample(self, xyz: np.ndarray, t: float) -> np.ndarray:
        """xyz: (..., 3); returns (..., 3) wind vectors."""

    @abstractmethod
    def is_constant(self) -> bool:
        """If True, downstream code can precompute the metric tensor once."""

    @abstractmethod
    def uncertainty(self, xyz: np.ndarray, t: float) -> np.ndarray:
        """Epistemic uncertainty on the sampled vector (||sigma|| in m/s).

        Stage 1: returns zeros. Stage 2 surfaces SCALED's posterior std.
        """

    def constant_vector(self) -> Tuple[float, float, float] | None:
        """Convenience: if the field is constant, return its (u,v,w) tuple."""
        if not self.is_constant():
            return None
        sample = self.sample(np.zeros((1, 3)), 0.0)[0]
        return float(sample[0]), float(sample[1]), float(sample[2])
