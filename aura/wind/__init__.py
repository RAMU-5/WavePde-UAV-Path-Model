"""Wind field abstraction.

`IWindField` is the single integration point with the upstream CFD/SCALED
subsystem. Stage 1 ships `ConstantWind`; Stage 2 will ship `SCALEDWind`.
"""

from aura.wind.base import IWindField
from aura.wind.constant import ConstantWind

__all__ = ["IWindField", "ConstantWind"]
