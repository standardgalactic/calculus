import numpy as np
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from rsvp_mips import ABPParameters, init_state, step_euler_maruyama
from rsvp_mips.fields import Grid
from rsvp_mips.capacity import (
    phi_input,
    reconstruct_phi_parallel,
    reconstruct_phi_plus,
)
from rsvp_mips.diagnostics import check_stability


def make_params(**overrides):
    defaults = dict(
        N=200, L=20.0, sigma=1.0, epsilon=1.0, mobility=1.0,
        v0=5.0, Dt=0.1, Dr=0.5, dt=0.001,
    )
    defaults.update(overrides)
    return ABPParameters(**defaults)


def test_phi_input_is_uniform_v0():
    params = make_params(v0=7.5)
    assert phi_input(params) == 7.5


def test_noninteracting_no_noise_collapses_all_three_variants():
    """
    With no interactions and no thermal noise, v_i = v0 * u_i exactly, so
    u_i . v_i = v0 for every particle regardless of orientation. All
    three capacity variants should therefore agree with Phi_input
    (up to floating point) everywhere -- the trivial limit in which the
    three competing definitions carry no information to distinguish them.
    """
    params = make_params(N=150, L=20.0, epsilon=0.0, v0=6.0, Dt=0.0, Dr=0.3,
                          dt=0.001)
    rng = np.random.default_rng(0)
    state = init_state(params, rng)
    step_euler_maruyama(state, params, rng)  # one step to populate dx_det

    grid = Grid(L=params.L, n=24)
    phi_par = reconstruct_phi_parallel(state, params, grid, h=1.5,
                                        velocity_source="deterministic")
    phi_plus = reconstruct_phi_plus(state, params, grid, h=1.5,
                                     velocity_source="deterministic")

    # only check where density is non-negligible
    from rsvp_mips.fields import reconstruct_density
    rho_h = reconstruct_density(state, params, grid, h=1.5)
    mask = rho_h > 0.1 * rho_h.max()

    assert np.allclose(phi_par[mask], params.v0, atol=1e-6)
    assert np.allclose(phi_plus[mask], params.v0, atol=1e-6)


def test_phi_plus_is_never_less_than_phi_parallel():
    """
    Phi_plus = <max(0, w_i)>_h >= <w_i>_h = Phi_parallel pointwise, since
    max(0, w) >= w for all w and kernel weights are non-negative. This
    should hold on any real, interacting, noisy trajectory.
    """
    params = make_params(N=250, L=22.0, epsilon=1.0, v0=15.0, Dt=0.1,
                          Dr=0.4, dt=0.001)
    rng = np.random.default_rng(1)
    state = init_state(params, rng)
    for step in range(500):
        step_euler_maruyama(state, params, rng)
        check_stability(state, step=step)

    grid = Grid(L=params.L, n=24)
    phi_par = reconstruct_phi_parallel(state, params, grid, h=1.5)
    phi_plus = reconstruct_phi_plus(state, params, grid, h=1.5)

    assert np.all(phi_plus >= phi_par - 1e-9)


def test_dense_interacting_system_suppresses_alignment_below_v0():
    """
    In a dense, interacting system, collisions push particles backward
    against their own orientation at least some of the time, so the
    density-weighted average realized alignment should be measurably
    below the imposed v0 -- capacity variants should show that
    interactions cost something, not leave Phi_parallel == Phi_input.
    """
    params = make_params(N=300, L=18.0, epsilon=1.0, v0=20.0, Dt=0.1,
                          Dr=0.3, dt=0.0002)
    rng = np.random.default_rng(2)
    state = init_state(params, rng)
    for step in range(7500):
        step_euler_maruyama(state, params, rng)
        check_stability(state, step=step)

    grid = Grid(L=params.L, n=24)
    phi_par = reconstruct_phi_parallel(state, params, grid, h=1.5)

    from rsvp_mips.fields import reconstruct_density
    rho_h = reconstruct_density(state, params, grid, h=1.5)
    mask = rho_h > 0.1 * rho_h.max()

    mean_alignment = phi_par[mask].mean()
    assert mean_alignment < params.v0 - 0.5


def test_velocity_source_total_vs_deterministic_differ():
    """
    With thermal noise present (Dt > 0), the total-velocity and
    deterministic-only variants of Phi_parallel should differ -- if they
    were identical, the velocity_source distinction would be doing
    nothing.

    Uses dt=0.0002: high Dt, like high v0, shortens the safe explicit-
    Euler timestep for the stiff WCA potential (see diagnostics.py). This
    was found empirically after an earlier version of this test at
    dt=0.001 produced force-blowup garbage (~1e11) that happened to
    satisfy "not allclose" for the wrong reason.
    """
    params = make_params(N=200, L=20.0, epsilon=1.0, v0=10.0, Dt=2.0,
                          Dr=0.3, dt=0.0002)
    rng = np.random.default_rng(3)
    state = init_state(params, rng)
    for step in range(2500):
        step_euler_maruyama(state, params, rng)
        check_stability(state, step=step)

    grid = Grid(L=params.L, n=20)
    phi_total = reconstruct_phi_parallel(state, params, grid, h=1.5,
                                          velocity_source="total")
    phi_det = reconstruct_phi_parallel(state, params, grid, h=1.5,
                                        velocity_source="deterministic")

    assert not np.allclose(phi_total, phi_det, atol=1e-3)
