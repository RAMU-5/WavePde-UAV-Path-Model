from __future__ import annotations

import numpy as np
import pytest

from model.wave_api import WaveConfig, solve_forward


def _cfg(D: float) -> WaveConfig:
    return WaveConfig(
        c0=5.0, cz_up=5.0, cz_down=5.0,
        dt=0.05, n_steps=80,
        pulse_t0_steps=6, pulse_tau_steps=2,
        sponge_layers=3, min_peak_amp=1e-8,
        reg_z_low=-2.0, reg_z_full=-1.0, reg_low_factor=1.0,
        diffusivity_m2_s=D,
    )


def test_zero_diffusion_solver_is_repeatable() -> None:
    mask = np.zeros((18, 18, 12), dtype=bool)
    src = (9, 9, 6)
    T_default = solve_forward(mask, 1.0, 1.0, 1.0, src, cfg=_cfg(0.0))
    T_zero = solve_forward(mask, 1.0, 1.0, 1.0, src, cfg=_cfg(0.0))
    np.testing.assert_array_equal(T_default, T_zero)


def test_unstable_diffusion_is_rejected() -> None:
    mask = np.zeros((8, 8, 8), dtype=bool)
    with pytest.raises(ValueError, match="diffusion is unstable"):
        solve_forward(mask, 1.0, 1.0, 1.0, (4, 4, 4), cfg=_cfg(20.0))
