import numpy as np
import pytest
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from rsvp_mips import (
    ABPParameters,
    init_state,
    step_euler_maruyama,
    compute_forces,
)
from rsvp_mips.observables import (
    orientation_autocorrelation_target,
    orientation_autocorrelation,
)


def make_params(**overrides):
    defaults = dict(
        N=100, L=20.0, sigma=1.0, epsilon=1.0, mobility=1.0,
        v0=0.0, Dt=0.0, Dr=0.5, dt=0.01,
    )
    defaults.update(overrides)
    return ABPParameters(**defaults)


def test_rotational_diffusion_matches_orientation_autocorrelation():
    """
    Corrected version of the naive Var(theta) test: measures
    C_u(t) = <u(0).u(t)> against the analytic e^{-Dr t}, which is well
    defined regardless of angle wrapping (unlike Var(theta_wrapped),
    which saturates at long times for the wrong reason).
    """
    params = make_params(v0=0.0, Dt=0.0, Dr=0.5, N=2000)
    rng = np.random.default_rng(123)
    state = init_state(params, rng)
    # start all particles pointing the same direction
    state.theta[:] = 0.0
    state.u[:, 0] = 1.0
    state.u[:, 1] = 0.0

    n_steps = 500
    u_history = [state.u.copy()]
    times = [state.t]
    for _ in range(n_steps):
        step_euler_maruyama(state, params, rng)
        u_history.append(state.u.copy())
        times.append(state.t)

    u_history = np.array(u_history)
    times = np.array(times)

    C_measured = orientation_autocorrelation(u_history)
    C_expected = orientation_autocorrelation_target(params.Dr, times)

    # compare only up to where the analytic curve hasn't decayed to noise
    keep = C_expected > 0.2
    assert np.allclose(C_measured[keep], C_expected[keep], atol=0.05)


def test_wca_force_vanishes_beyond_cutoff():
    params = make_params(N=2, L=20.0, sigma=1.0, epsilon=1.0)
    rng = np.random.default_rng(0)
    state = init_state(params, rng)

    rcut = 2.0 ** (1.0 / 6.0) * params.sigma
    state.x[0] = [5.0, 5.0]
    state.x[1] = [5.0 + rcut + 0.5, 5.0]  # well beyond cutoff
    compute_forces(state, params)
    assert np.allclose(state.force, 0.0)


def test_wca_force_repulsive_inside_cutoff():
    params = make_params(N=2, L=20.0, sigma=1.0, epsilon=1.0)
    rng = np.random.default_rng(0)
    state = init_state(params, rng)

    state.x[0] = [5.0, 5.0]
    state.x[1] = [5.3, 5.0]  # inside cutoff, particles overlapping
    compute_forces(state, params)
    # particle 0 should be pushed in -x, particle 1 in +x (repulsion)
    assert state.force[0, 0] < 0.0
    assert state.force[1, 0] > 0.0


def test_newtons_third_law_holds_on_average():
    params = make_params(N=50, L=10.0, sigma=1.0, epsilon=1.0)
    rng = np.random.default_rng(1)
    state = init_state(params, rng)
    compute_forces(state, params)
    total_force = state.force.sum(axis=0)
    assert np.allclose(total_force, 0.0, atol=1e-8)


def test_msd_uses_unwrapped_coordinates_not_wrapped():
    """
    Sanity check that x_unwrapped can exceed the box size L, confirming it
    is genuinely unwrapped rather than accidentally aliased to x. Uses
    epsilon=0 (no WCA interaction) so the test isolates the unwrapping
    mechanics from interaction-force behavior.
    """
    params = make_params(N=5, L=2.0, epsilon=0.0, v0=5.0, Dt=0.0, Dr=0.0, dt=0.1)
    rng = np.random.default_rng(2)
    state = init_state(params, rng)
    state.theta[:] = 0.0
    state.u[:, 0] = 1.0
    state.u[:, 1] = 0.0

    for _ in range(50):
        step_euler_maruyama(state, params, rng)

    # with v0=5, dt=0.1, 50 steps, no interactions: deterministic
    # displacement is exactly 25 in x, well beyond L=2.0, so unwrapped x
    # must exceed L while wrapped x must not.
    assert np.all(state.x_unwrapped[:, 0] > params.L)
    assert np.all(state.x[:, 0] < params.L)
