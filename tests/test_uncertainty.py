from __future__ import annotations

import numpy as np

from model.uncertainty_diffusion import (
    confidence_radius,
    ensemble_collision_statistics,
    path_clearance,
    sample_trajectory_ensemble,
)


def test_confidence_radius_matches_chi_square_definition() -> None:
    tau = np.array([0.0, 1.0, 4.0])
    r95 = confidence_radius(tau)
    expected = np.sqrt(2.0 * tau * 7.814727903251179)
    np.testing.assert_allclose(r95, expected, rtol=1e-12)


def test_brownian_ensemble_has_expected_terminal_variance() -> None:
    n = 41
    t = np.linspace(0.0, 4.0, n)
    path = np.column_stack([5.0 * t, np.zeros(n), np.full(n, 10.0)])
    D = np.full(n, 0.75)
    samples = sample_trajectory_ensemble(path, t, D, n_samples=8000, seed=4)
    perturb = samples[:, -1] - path[-1]
    expected_var = 2.0 * 0.75 * 4.0
    np.testing.assert_allclose(perturb.var(axis=0), expected_var, rtol=0.05)


def test_zero_diffusion_has_zero_collision_probability() -> None:
    n = 21
    t = np.linspace(0.0, 2.0, n)
    path = np.column_stack([5.0 + 2.0 * t, np.full(n, 5.0), np.full(n, 5.0)])
    samples = sample_trajectory_ensemble(
        path, t, np.zeros(n), n_samples=100, seed=0,
    )
    mask = np.zeros((20, 20, 20), dtype=bool)
    stats = ensemble_collision_statistics(samples, mask, (1.0, 1.0, 1.0))
    assert stats["n_collided"] == 0
    assert stats["collision_probability"] == 0.0


def test_cell_centred_coordinates_map_to_the_expected_voxel() -> None:
    mask = np.zeros((4, 4, 4), dtype=bool)
    mask[1, 2, 3] = True
    centre = np.array([[1.5, 2.5, 3.5]])

    clearance = path_clearance(centre, mask, (1.0, 1.0, 1.0))
    samples = np.repeat(centre[None, :, :], 5, axis=0)
    stats = ensemble_collision_statistics(samples, mask, (1.0, 1.0, 1.0))

    np.testing.assert_allclose(clearance, 0.0)
    assert stats["n_collided"] == 5
    assert stats["n_obstacle_collided"] == 5
    assert stats["n_domain_exited"] == 0
