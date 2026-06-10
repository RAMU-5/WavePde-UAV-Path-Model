"""Stage 2 placeholder for SCALED-driven wind field.

The real implementation will wrap `LatentInferenceOnlyGeometryDDP` from
SCALED-UrbanFlowv3, decode latent -> velocity, and serve queries by trilinear
interpolation. Left as a stub here so the import path exists today.
"""

from __future__ import annotations

import numpy as np

from aura.wind.base import IWindField


class SCALEDWind(IWindField):
    def __init__(self, *args, **kwargs):
        raise NotImplementedError(
            "SCALEDWind is deferred to Stage 2. Stage 1 uses ConstantWind. "
            "See plan: 'Out-of-scope (deferred to Stage 2/3)'."
        )

    def sample(self, xyz: np.ndarray, t: float) -> np.ndarray:
        raise NotImplementedError

    def is_constant(self) -> bool:
        return False

    def uncertainty(self, xyz: np.ndarray, t: float) -> np.ndarray:
        raise NotImplementedError
