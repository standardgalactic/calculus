import numpy as np
import pytest
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from rsvp_mips import ABPParameters, init_state, step_euler_maruyama
from rsvp_mips.fields import Grid, reconstruct_density_and_current
from rsvp_mips.capacity import reconstruct_phi_parallel
from rsvp_mips.dissipation import (
    effective_temperature,
    reconstruct_s_work,
    reconstruct_s_current,
)
from rsvp_mips.diagnostics import check_stability


def make_params(**overrides):
    defaults = dict(
        N=200, L=20.0, sigma=1.0, epsilon=1.0, mobility=1.0,
        v0=10.0, Dt=0.5, Dr=0.4, dt=0.0005,
    )
    defaults.update(overrides)
    return ABPParameters(**defaults)


def test_effective_temperature_requires_positive_Dt():
    params = make_params(Dt=0.0)
    with pytest.raises(ValueError):
        effective_temperature(params)


def test_s_work_requires_positive_Dt():
    params = make_params(Dt=0.0)
    rng = np.random.default_rng(0)
    state = init_state(params, rng)
    grid = Grid(L=params.L, n=16)
    with pytest.raises(ValueError):
        reconstruct_s_work(state, params, grid, h=1.5)


def test_s_work_is_exactly_proportional_to_phi_parallel():
    """
    Algebraic identity for this model: S_work = (v0 / Dt) * Phi_parallel,
    for the same velocity_source. This should hold to floating-point
    precision on any state, since it follows directly from
    F_active = (v0/mobility)*u and T_eff = Dt/mobility (the mobility
    cancels), not from any statistical property of the trajectory.
    """
    params = make_params(N=250, L=22.0, v0=12.0, Dt=0.3, mobility=1.7,
                          dt=0.0005)
    rng = np.random.default_rng(1)
    state = init_state(params, rng)
    for step in range(500):
        step_euler_maruyama(state, params, rng)
        check_stability(state, step=step)

    grid = Grid(L=params.L, n=20)
    for velocity_source in ("total", "deterministic"):
        s_work = reconstruct_s_work(state, params, grid, h=1.5,
                                     velocity_source=velocity_source)
        phi_par = reconstruct_phi_parallel(state, params, grid, h=1.5,
                                            velocity_source=velocity_source)
        expected = (params.v0 / params.Dt) * phi_par
        assert np.allclose(s_work, expected, rtol=1e-8, atol=1e-8), \
            f"failed for velocity_source={velocity_source}"


def test_s_current_is_nonnegative():
    params = make_params(N=250, L=20.0)
    rng = np.random.default_rng(2)
    state = init_state(params, rng)
    for step in range(500):
        step_euler_maruyama(state, params, rng)
        check_stability(state, step=step)

    grid = Grid(L=params.L, n=24)
    rho_h, Jx_h, Jy_h, _, _ = reconstruct_density_and_current(
        state, params, grid, h=1.5
    )
    s_current = reconstruct_s_current(rho_h, Jx_h, Jy_h, params)
    assert np.all(s_current >= 0.0)


def test_s_current_requires_positive_Dt():
    params = make_params(Dt=0.0)
    rng = np.random.default_rng(3)
    state = init_state(params, rng)
    grid = Grid(L=params.L, n=16)
    rho_h, Jx_h, Jy_h, _, _ = reconstruct_density_and_current(
        state, params, grid, h=1.5
    )
    with pytest.raises(ValueError):
        reconstruct_s_current(rho_h, Jx_h, Jy_h, params)


def test_s_current_is_not_proportional_to_s_work():
    """
    Unlike Phi_parallel, S_current should NOT be an exact multiple of
    S_work -- it is built from coarse-grained collective flow (rho_h, J_h)
    rather than per-particle force, so the two estimators should disagree
    in their spatial pattern on a real, structured trajectory. This is
    the genuine second, independent dissipation estimator the design
    called for.
    """
    params = make_params(N=300, L=18.0, v0=25.0, Dt=0.2, Dr=0.3, dt=0.0002)
    rng = np.random.default_rng(4)
    state = init_state(params, rng)
    for step in range(5000):
        step_euler_maruyama(state, params, rng)
        check_stability(state, step=step)

    grid = Grid(L=params.L, n=24)
    h = 1.5
    rho_h, Jx_h, Jy_h, _, _ = reconstruct_density_and_current(
        state, params, grid, h
    )
    s_current = reconstruct_s_current(rho_h, Jx_h, Jy_h, params)
    s_work = reconstruct_s_work(state, params, grid, h)

    mask = rho_h > 0.1 * rho_h.max()
    ratio = s_current[mask] / (s_work[mask] + 1e-8)
    # if they were proportional, the ratio would be ~constant everywhere;
    # a structured trajectory should show real spread in the ratio
    assert ratio.std() / (np.abs(ratio.mean()) + 1e-8) > 0.05
