import numpy as np
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from rsvp_mips import ABPParameters, init_state, step_euler_maruyama
from rsvp_mips.fields import (
    Grid,
    reconstruct_density,
    reconstruct_density_and_current,
)


def make_params(**overrides):
    defaults = dict(
        N=200, L=20.0, sigma=1.0, epsilon=1.0, mobility=1.0,
        v0=5.0, Dt=0.1, Dr=0.5, dt=0.001,
    )
    defaults.update(overrides)
    return ABPParameters(**defaults)


def test_density_integrates_to_N():
    """
    rho_h(x) is a normalized kernel density estimate, so integrating it
    over the box (sum over grid cells * cell area) should recover N, as
    long as h is small relative to L (kernel mass isn't wrapping around
    and double-counting) and the grid is fine enough to resolve it.
    """
    params = make_params(N=300, L=30.0)
    rng = np.random.default_rng(0)
    state = init_state(params, rng)

    grid = Grid(L=params.L, n=64)
    h = 1.0
    rho_h = reconstruct_density(state, params, grid, h)

    integral = rho_h.sum() * grid.cell_area
    assert abs(integral - params.N) / params.N < 0.02


def test_density_integrates_to_N_across_kernel_widths():
    """Same check at a couple of different kernel widths h."""
    params = make_params(N=300, L=30.0)
    rng = np.random.default_rng(1)
    state = init_state(params, rng)
    grid = Grid(L=params.L, n=96)

    for h in (0.5, 1.0, 2.0):
        rho_h = reconstruct_density(state, params, grid, h)
        integral = rho_h.sum() * grid.cell_area
        assert abs(integral - params.N) / params.N < 0.03, f"failed at h={h}"


def test_velocity_field_matches_uniform_motion():
    """
    If every particle moves with the same velocity, the reconstructed
    velocity field should be uniform and equal to that velocity
    everywhere density is non-negligible, regardless of kernel width.
    """
    params = make_params(N=200, L=20.0, epsilon=0.0, v0=0.0, Dt=0.0, Dr=0.0)
    rng = np.random.default_rng(2)
    state = init_state(params, rng)

    # force every particle to have moved by the same (dx, dy) last step
    state.dx_det[:] = 0.0
    state.dx_stoch[:] = 0.0
    state.dx_det[:, 0] = 3.0 * params.dt  # uniform vx = 3.0
    state.dx_det[:, 1] = -1.5 * params.dt  # uniform vy = -1.5

    grid = Grid(L=params.L, n=32)
    rho_h, Jx_h, Jy_h, vx_h, vy_h = reconstruct_density_and_current(
        state, params, grid, h=1.5
    )

    # only check where density is appreciable (avoid the eps-regularized
    # near-vacuum regions, which are expected to be noisy per the module
    # docstring)
    mask = rho_h > 0.1 * rho_h.max()
    assert np.allclose(vx_h[mask], 3.0, atol=0.05)
    assert np.allclose(vy_h[mask], -1.5, atol=0.05)


def test_current_is_density_weighted_velocity():
    """J_h should equal rho_h * v_h by construction (up to the eps term)."""
    params = make_params(N=250, L=25.0)
    rng = np.random.default_rng(3)
    state = init_state(params, rng)
    for _ in range(50):
        step_euler_maruyama(state, params, rng)

    grid = Grid(L=params.L, n=40)
    rho_h, Jx_h, Jy_h, vx_h, vy_h = reconstruct_density_and_current(
        state, params, grid, h=1.0
    )

    assert np.allclose(Jx_h, vx_h * (rho_h + 1e-8), atol=1e-6)
    assert np.allclose(Jy_h, vy_h * (rho_h + 1e-8), atol=1e-6)
