import numpy as np
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from rsvp_mips import ABPParameters, init_state_random
from rsvp_mips.intervention import (
    LocalActivityPulse,
    affected_particle_mask,
    compute_v0_field,
    clone_state,
    run_counterfactual_pair,
)


def make_params(**overrides):
    defaults = dict(
        N=100, L=20.0, sigma=1.0, epsilon=1.0, mobility=1.0,
        v0=5.0, Dt=0.1, Dr=0.3, dt=0.0005,
    )
    defaults.update(overrides)
    return ABPParameters(**defaults)


# ---------------------------------------------------------------------------
# Spatial/temporal masking
# ---------------------------------------------------------------------------

def test_affected_particle_mask_identifies_particles_within_radius():
    params = make_params(N=10, L=20.0)
    rng = np.random.default_rng(0)
    state = init_state_random(params, rng, min_distance_factor=1.02)
    # place particles at known positions for a deterministic check
    state.x[:] = np.array([[10.0, 10.0]] * 10)
    state.x[0] = [10.0, 10.0]   # distance 0 -- inside
    state.x[1] = [11.0, 10.0]   # distance 1 -- inside radius=2
    state.x[2] = [13.0, 10.0]   # distance 3 -- outside radius=2

    intervention = LocalActivityPulse(center=(10.0, 10.0), radius=2.0,
                                       amplitude=0.5, start_time=0.0, duration=1.0)
    mask = affected_particle_mask(state, params, intervention)
    assert mask[0] and mask[1]
    assert not mask[2]


def test_affected_particle_mask_respects_periodic_wraparound():
    params = make_params(N=5, L=20.0)
    rng = np.random.default_rng(0)
    state = init_state_random(params, rng, min_distance_factor=1.02)
    # center near one edge, particle near the opposite edge -- should
    # still be "close" under periodic wraparound
    state.x[0] = [0.5, 10.0]
    state.x[1] = [19.5, 10.0]  # only 1.0 away from x[0] periodically

    intervention = LocalActivityPulse(center=(0.5, 10.0), radius=2.0,
                                       amplitude=0.5, start_time=0.0, duration=1.0)
    mask = affected_particle_mask(state, params, intervention)
    assert mask[0] and mask[1]


def test_compute_v0_field_outside_time_window_returns_uniform_v0():
    params = make_params(N=10, v0=7.0)
    rng = np.random.default_rng(0)
    state = init_state_random(params, rng, min_distance_factor=1.02)
    state.x[:] = np.array([[10.0, 10.0]] * 10)  # all inside any reasonable radius

    intervention = LocalActivityPulse(center=(10.0, 10.0), radius=5.0,
                                       amplitude=1.0, start_time=2.0, duration=1.0)
    # before window
    v0_before = compute_v0_field(state, params, intervention, t=1.0)
    assert np.allclose(v0_before, 7.0)
    # after window
    v0_after = compute_v0_field(state, params, intervention, t=3.5)
    assert np.allclose(v0_after, 7.0)
    # inside window
    v0_during = compute_v0_field(state, params, intervention, t=2.5)
    assert np.allclose(v0_during, 14.0)  # 7.0 * (1 + 1.0)


def test_compute_v0_field_only_affects_particles_in_radius():
    params = make_params(N=3, v0=5.0)
    rng = np.random.default_rng(0)
    state = init_state_random(params, rng, min_distance_factor=1.02)
    state.x[0] = [10.0, 10.0]  # inside
    state.x[1] = [10.0, 10.0]  # inside
    state.x[2] = [0.0, 0.0]    # far away -- outside

    intervention = LocalActivityPulse(center=(10.0, 10.0), radius=1.0,
                                       amplitude=2.0, start_time=0.0, duration=1.0)
    v0_field = compute_v0_field(state, params, intervention, t=0.5)
    assert np.isclose(v0_field[0], 15.0)  # 5 * 3
    assert np.isclose(v0_field[1], 15.0)
    assert np.isclose(v0_field[2], 5.0)   # unaffected


# ---------------------------------------------------------------------------
# Counterfactual pairing -- the core causal-inference mechanism
# ---------------------------------------------------------------------------

def simple_recorder(state, params):
    return {"mean_x": float(state.x[:, 0].mean())}


def test_zero_amplitude_gives_bit_identical_treatment_and_control():
    """
    The central correctness check for common random numbers: with
    amplitude=0, the intervention changes nothing about the dynamics, so
    treatment and control -- despite being independently-instantiated
    RNGs -- must produce EXACTLY identical trajectories, not just
    statistically similar ones.
    """
    params = make_params(N=50, L=20.0, v0=5.0, dt=0.0005)
    rng = np.random.default_rng(0)
    base_state = init_state_random(params, rng, min_distance_factor=1.02)

    intervention = LocalActivityPulse(center=(10.0, 10.0), radius=3.0,
                                       amplitude=0.0, start_time=0.0, duration=0.5)
    result = run_counterfactual_pair(
        base_state, params, seed=42, intervention=intervention,
        duration=0.5, record_every_time=0.1, recorder=simple_recorder,
    )
    assert not result.unstable
    assert np.array_equal(result.treatment_state.x, result.control_state.x)
    assert np.array_equal(result.treatment_state.theta, result.control_state.theta)
    assert result.treatment_history["mean_x"] == result.control_history["mean_x"]


def test_nonzero_amplitude_produces_measurable_divergence():
    """
    A real, sizeable perturbation should measurably change the treatment
    branch's local particle distribution relative to control, despite
    starting from the identical state and drawing the identical random
    numbers -- the divergence is attributable to the perturbation, not
    noise.
    """
    params = make_params(N=100, L=20.0, v0=10.0, dt=0.0005)
    rng = np.random.default_rng(1)
    base_state = init_state_random(params, rng, min_distance_factor=1.02)

    intervention = LocalActivityPulse(center=(10.0, 10.0), radius=3.0,
                                       amplitude=3.0, start_time=0.0, duration=1.0)
    result = run_counterfactual_pair(
        base_state, params, seed=7, intervention=intervention,
        duration=1.0, record_every_time=0.1, recorder=simple_recorder,
    )
    assert not result.unstable
    assert not np.array_equal(result.treatment_state.x, result.control_state.x)

    treatment_final = np.array(result.treatment_history["mean_x"])
    control_final = np.array(result.control_history["mean_x"])
    assert not np.allclose(treatment_final, control_final)


def test_clone_state_produces_independent_copies():
    params = make_params(N=10)
    rng = np.random.default_rng(0)
    state = init_state_random(params, rng, min_distance_factor=1.02)
    clone = clone_state(state)
    clone.x[0, 0] = 999.0
    assert state.x[0, 0] != 999.0  # original untouched


def test_counterfactual_pair_starts_from_identical_state():
    """Both branches' t=0 (well, first recorded frame) values should
    match exactly, since they're both clones of the same base_state."""
    params = make_params(N=30, dt=0.0005)
    rng = np.random.default_rng(0)
    base_state = init_state_random(params, rng, min_distance_factor=1.02)
    intervention = LocalActivityPulse(center=(10.0, 10.0), radius=2.0,
                                       amplitude=1.0, start_time=1.0, duration=0.5)
    result = run_counterfactual_pair(
        base_state, params, seed=3, intervention=intervention,
        duration=0.5, record_every_time=0.1, recorder=simple_recorder,
    )
    assert result.treatment_history["mean_x"][0] == result.control_history["mean_x"][0]


def test_intervention_activates_relative_to_run_start_not_absolute_state_time():
    """
    Regression test for a real bug: compute_v0_field must be checked
    against time RELATIVE to the counterfactual run's own start, not the
    state's absolute simulation clock. A state that has already run for
    several time units (e.g. post-burn-in, where state.t starts at 6.0
    rather than 0.0) must still have intervention.start_time=0.0 mean
    "immediately," not "at absolute t=0.0, which has already passed."
    Every other test in this file happens to use a fresh state (t=0), so
    this scenario is the only one that would have caught the bug.
    """
    params = make_params(N=50, L=20.0, v0=5.0, dt=0.0005)
    rng = np.random.default_rng(0)
    base_state = init_state_random(params, rng, min_distance_factor=1.02)
    base_state.t = 6.0  # simulate a post-burn-in state's absolute clock

    intervention = LocalActivityPulse(center=(10.0, 10.0), radius=3.0,
                                       amplitude=3.0, start_time=0.0, duration=1.0)
    result = run_counterfactual_pair(
        base_state, params, seed=7, intervention=intervention,
        duration=1.0, record_every_time=0.1, recorder=simple_recorder,
    )
    assert not result.unstable
    # if the bug were present, this would be array_equal (intervention
    # never actually activates)
    assert not np.array_equal(result.treatment_state.x, result.control_state.x)
