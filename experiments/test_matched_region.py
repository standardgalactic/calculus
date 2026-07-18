import numpy as np
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from rsvp_mips import ABPParameters, init_state_random
from rsvp_mips.fields import Grid
from rsvp_mips.matched_region import (
    RegionCandidate,
    _local_mean,
    _local_polarization,
    match_pairs_by_density,
    compute_response_metrics,
)
from rsvp_mips.intervention import LocalActivityPulse


def make_params(**overrides):
    defaults = dict(
        N=50, L=20.0, sigma=1.0, epsilon=1.0, mobility=1.0,
        v0=5.0, Dt=0.1, Dr=0.3, dt=0.0005,
    )
    defaults.update(overrides)
    return ABPParameters(**defaults)


# ---------------------------------------------------------------------------
# Local aggregation helpers
# ---------------------------------------------------------------------------

def test_local_mean_averages_only_cells_within_radius():
    L, n = 20.0, 20
    grid = Grid(L=L, n=n)
    field = np.zeros((n, n))
    field[:] = 1.0  # background
    # set a known patch near the center to a different value
    center_idx = n // 2
    field[center_idx, center_idx] = 5.0

    result = _local_mean(field, grid, center=(10.0, 10.0), radius=1.0)
    # radius=1.0 with grid spacing 1.0 should include only the center
    # cell or very few cells -- result should be pulled toward 5.0,
    # clearly above the 1.0 background
    assert result > 1.0


def test_local_mean_matches_full_field_mean_at_large_radius():
    L, n = 20.0, 16
    grid = Grid(L=L, n=n)
    rng = np.random.default_rng(0)
    field = rng.standard_normal((n, n))
    result = _local_mean(field, grid, center=(10.0, 10.0), radius=100.0)
    assert np.isclose(result, field.mean(), atol=1e-10)


def test_local_polarization_is_one_for_perfectly_aligned_particles():
    params = make_params(N=20, L=20.0)
    rng = np.random.default_rng(0)
    state = init_state_random(params, rng, min_distance_factor=1.02)
    state.x[:] = np.array([[10.0, 10.0]] * 20)  # all within any radius
    state.theta[:] = 0.0
    state.u[:, 0] = 1.0
    state.u[:, 1] = 0.0

    pol = _local_polarization(state, params, center=(10.0, 10.0), radius=1.0)
    assert np.isclose(pol, 1.0)


def test_local_polarization_is_near_zero_for_isotropic_particles():
    # init a small, easily-packable state first, then overwrite positions
    # directly (bypassing RSA's non-overlap constraint, which isn't
    # needed here -- this test only exercises orientation averaging)
    params = make_params(N=2000, L=20.0)
    rng = np.random.default_rng(0)
    small_params = make_params(N=10, L=20.0)
    state = init_state_random(small_params, rng, min_distance_factor=1.02)
    state.x = np.tile(np.array([10.0, 10.0]), (2000, 1))
    state.theta = np.zeros(2000)
    state.u = np.zeros((2000, 2))
    theta = rng.uniform(-np.pi, np.pi, 2000)
    state.theta[:] = theta
    state.u[:, 0] = np.cos(theta)
    state.u[:, 1] = np.sin(theta)

    pol = _local_polarization(state, params, center=(10.0, 10.0), radius=1.0)
    assert pol < 0.1  # should average out close to zero for large N


# ---------------------------------------------------------------------------
# Matching logic -- pure, testable with synthetic candidates
# ---------------------------------------------------------------------------

def make_candidate(center, density, capacity):
    return RegionCandidate(
        center=center, local_density=density, local_capacity=capacity,
        local_polarization=0.1, in_dense_domain=False,
        local_density_gradient_mag=0.0,
    )


def test_match_pairs_selects_min_max_capacity_within_density_bin():
    # all in the same narrow density bin (density ~1.0), varying capacity
    candidates = [
        make_candidate((0, 0), density=1.0, capacity=0.5),
        make_candidate((1, 0), density=1.01, capacity=5.0),
        make_candidate((2, 0), density=1.02, capacity=2.0),
    ]
    pairs = match_pairs_by_density(candidates, n_bins=1)
    assert len(pairs) == 1
    low_cap, high_cap = pairs[0]
    assert low_cap.local_capacity == 0.5
    assert high_cap.local_capacity == 5.0


def test_match_pairs_respects_density_binning():
    # two well-separated density clusters -- should NOT pair across them
    candidates = [
        make_candidate((0, 0), density=1.0, capacity=1.0),
        make_candidate((1, 0), density=1.0, capacity=9.0),
        make_candidate((2, 0), density=10.0, capacity=1.0),
        make_candidate((3, 0), density=10.0, capacity=9.0),
    ]
    pairs = match_pairs_by_density(candidates, n_bins=2)
    assert len(pairs) == 2
    for low_cap, high_cap in pairs:
        # matched pair should come from the SAME density neighborhood
        assert abs(low_cap.local_density - high_cap.local_density) < 5.0


def test_match_pairs_skips_low_contrast_bins():
    candidates = [
        make_candidate((0, 0), density=1.0, capacity=1.0),
        make_candidate((1, 0), density=1.0, capacity=1.01),  # negligible contrast
    ]
    pairs = match_pairs_by_density(candidates, n_bins=1, min_capacity_contrast=1.0)
    assert len(pairs) == 0


def test_match_pairs_empty_for_insufficient_candidates():
    assert match_pairs_by_density([], n_bins=3) == []
    assert match_pairs_by_density([make_candidate((0, 0), 1.0, 1.0)], n_bins=3) == []


# ---------------------------------------------------------------------------
# Response metrics -- constructed with known ground-truth shape
# ---------------------------------------------------------------------------

def test_response_metrics_on_a_known_dip_recover_overshoot_pattern():
    times = np.array([0.0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0])
    # treatment: starts at 20, dips to 2 during pulse, recovers, overshoots to 28
    treat_local = np.array([20, 10, 2, 8, 16, 20, 24, 28, 25])
    ctrl_local = np.array([20, 19, 20, 18, 19, 20, 19, 18, 19])
    treat_global = np.array([0.1] * 8 + [0.25])
    ctrl_global = np.array([0.1] * 9)

    intervention = LocalActivityPulse(center=(0, 0), radius=1.0, amplitude=2.0,
                                       start_time=0.0, duration=1.0)

    result = compute_response_metrics(
        times, treat_local, ctrl_local, treat_global, ctrl_global,
        intervention, region_label="test", local_capacity=3.0, local_density=1.0,
    )
    assert result.depletion_min == 2.0
    assert result.time_of_min == 1.0
    assert result.recovery_time is not None
    assert result.recovery_time >= 1.0
    assert result.overshoot_max == 28 - 18  # max diff after pulse ends at t=1.0
    assert np.isclose(result.delayed_global_diff, 0.25 - 0.1)
    assert result.local_capacity == 3.0
    assert result.local_density == 1.0


def test_response_metrics_recovery_none_if_never_recovers():
    times = np.array([0.0, 1.0, 2.0])
    treat_local = np.array([20.0, 5.0, 6.0])   # never catches back up to control
    ctrl_local = np.array([20.0, 19.0, 18.0])
    treat_global = np.array([0.1, 0.1, 0.1])
    ctrl_global = np.array([0.1, 0.1, 0.1])
    intervention = LocalActivityPulse(center=(0, 0), radius=1.0, amplitude=1.0,
                                       start_time=0.0, duration=0.5)
    result = compute_response_metrics(
        times, treat_local, ctrl_local, treat_global, ctrl_global,
        intervention, region_label="test", local_capacity=1.0, local_density=1.0,
    )
    assert result.recovery_time is None
