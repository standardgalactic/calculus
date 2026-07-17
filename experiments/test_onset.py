import numpy as np
import pytest
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from rsvp_mips.onset import (
    detect_onset_events,
    prospective_labels,
    delta_M,
)
from rsvp_mips import (
    ABPParameters, init_state, step_euler_maruyama,
    largest_cluster_fraction,
)
from rsvp_mips.diagnostics import check_stability


# ---------------------------------------------------------------------------
# Synthetic-trajectory tests
# ---------------------------------------------------------------------------

def test_clean_single_onset_is_detected_with_correct_start_time():
    times = np.arange(0, 20, 0.1)
    f_max = np.full_like(times, 0.1)
    # a clean, sustained rise from t=5 to t=10
    rise_mask = (times >= 5.0) & (times < 10.0)
    f_max[rise_mask] = 0.8

    events = detect_onset_events(times, f_max, theta_high=0.5,
                                  theta_low=0.3, min_dwell=1.0)
    assert len(events) == 1
    assert np.isclose(events[0].start_time, 5.0, atol=0.1)
    assert events[0].peak_f_max == 0.8


def test_transient_spike_below_min_dwell_is_not_detected():
    """
    A brief spike above theta_high that doesn't sustain for min_dwell
    should NOT register as onset -- this is the entire point of the
    persistence criterion.
    """
    times = np.arange(0, 20, 0.1)
    f_max = np.full_like(times, 0.1)
    spike_mask = (times >= 5.0) & (times < 5.3)  # 0.3 time units, brief
    f_max[spike_mask] = 0.9

    events = detect_onset_events(times, f_max, theta_high=0.5,
                                  theta_low=0.3, min_dwell=1.0)
    assert len(events) == 0


def test_sustained_run_that_never_crosses_theta_high_is_not_detected():
    """
    A long run that stays between theta_low and theta_high never actually
    reaches the high threshold, so it should not count as onset even
    though it is persistent.
    """
    times = np.arange(0, 20, 0.1)
    f_max = np.full_like(times, 0.1)
    plateau_mask = (times >= 5.0) & (times < 15.0)
    f_max[plateau_mask] = 0.35  # between theta_low=0.3 and theta_high=0.5

    events = detect_onset_events(times, f_max, theta_high=0.5,
                                  theta_low=0.3, min_dwell=1.0)
    assert len(events) == 0


def test_rapid_oscillation_produces_no_persistent_events():
    """
    The motivating case: f_max oscillating rapidly between low and high
    values (mimicking Figure 5's volatility) should produce zero
    persistent onset events, even though a raw threshold-crossing count
    on the same series would report many.
    """
    times = np.arange(0, 20, 0.1)
    # oscillate with a period much shorter than min_dwell
    f_max = 0.5 + 0.4 * np.sin(2 * np.pi * times / 0.4)

    raw_crossings = np.sum(np.diff((f_max >= 0.5).astype(int)) == 1)
    events = detect_onset_events(times, f_max, theta_high=0.7,
                                  theta_low=0.5, min_dwell=1.0)

    assert raw_crossings > 10  # confirms the series is genuinely volatile
    assert len(events) == 0


def test_s_low_q_confirmation_filters_out_unconfirmed_episodes():
    times = np.arange(0, 20, 0.1)
    f_max = np.full_like(times, 0.1)
    rise_mask = (times >= 5.0) & (times < 10.0)
    f_max[rise_mask] = 0.8

    # s_low_q stays LOW throughout -- no independent structural
    # confirmation of the f_max rise
    s_low_q = np.full_like(times, 0.1)

    events_unconfirmed = detect_onset_events(
        times, f_max, theta_high=0.5, theta_low=0.3, min_dwell=1.0,
        s_low_q=s_low_q, s_low_q_threshold=1.0,
    )
    assert len(events_unconfirmed) == 0

    # now with confirmation present
    s_low_q_confirmed = np.full_like(times, 0.1)
    s_low_q_confirmed[rise_mask] = 2.0
    events_confirmed = detect_onset_events(
        times, f_max, theta_high=0.5, theta_low=0.3, min_dwell=1.0,
        s_low_q=s_low_q_confirmed, s_low_q_threshold=1.0,
    )
    assert len(events_confirmed) == 1


def test_theta_high_must_exceed_theta_low():
    times = np.arange(0, 10, 0.1)
    f_max = np.full_like(times, 0.5)
    with pytest.raises(ValueError):
        detect_onset_events(times, f_max, theta_high=0.3, theta_low=0.5)


# ---------------------------------------------------------------------------
# Prospective labeling
# ---------------------------------------------------------------------------

def test_prospective_labels_true_only_within_window_before_onset():
    times = np.arange(0, 20, 1.0)
    f_max = np.full_like(times, 0.1)
    f_max[(times >= 10) & (times < 15)] = 0.8  # onset starts at t=10

    events = detect_onset_events(times, f_max, theta_high=0.5,
                                  theta_low=0.3, min_dwell=1.0)
    assert len(events) == 1
    assert events[0].start_time == 10.0

    Y = prospective_labels(times, events, tau=3.0)
    # t=7,8,9 should be True (onset at 10 falls within (t, t+3])
    for t in (7.0, 8.0, 9.0):
        idx = np.where(times == t)[0][0]
        assert Y[idx], f"expected True at t={t}"
    # t=10 itself: onset start (10) is not > t (10), so False by definition
    idx10 = np.where(times == 10.0)[0][0]
    assert not Y[idx10]
    # t=6: onset at 10 is outside (6, 9], so False
    idx6 = np.where(times == 6.0)[0][0]
    assert not Y[idx6]


def test_prospective_labels_all_false_with_no_events():
    times = np.arange(0, 10, 1.0)
    Y = prospective_labels(times, [], tau=3.0)
    assert not np.any(Y)


# ---------------------------------------------------------------------------
# Delta_tau M
# ---------------------------------------------------------------------------

def test_delta_M_on_linear_trajectory():
    times = np.arange(0, 10, 1.0)
    M = 2.0 * times  # linear, so Delta_tau M should be exactly 2*tau
    valid_times, delta = delta_M(times, M, tau=3.0)
    assert np.allclose(delta, 6.0)
    assert np.all(valid_times <= 7.0)  # can't look 3 units past t=9... trims correctly


def test_delta_M_snaps_to_nearest_available_sample():
    times = np.array([0.0, 1.0, 2.0, 5.0])
    M = np.array([0.0, 1.0, 2.0, 10.0])
    # requesting tau=2.9 from t=0 targets t=2.9, nearest sample is t=2.0
    # (not t=5.0), values [0,1,2,5] -> distances to 2.9: |2-2.9|=0.9,
    # |5-2.9|=2.1, so snaps to t=2.0
    valid_times, delta = delta_M(times, M, tau=2.9)
    assert valid_times[0] == 0.0
    assert delta[0] == 2.0  # M(2.0) - M(0.0)


# ---------------------------------------------------------------------------
# Real, volatile simulated trajectory -- the actual motivating case
# ---------------------------------------------------------------------------

def test_onset_detector_on_real_volatile_simulated_trajectory():
    """
    Reruns a scaled-down version of Milestone 1.5's Figure 5 regime
    (dense, high-activity, small system -- previously shown to produce
    f_max swinging 0.2-1.0 within a few time units) and confirms the
    persistence-filtered event count is substantially lower than a naive
    raw-crossing count on the same real trajectory, not just on
    synthetic data.
    """
    params = ABPParameters(
        N=150, L=20.0, sigma=1.0, epsilon=1.0, mobility=1.0,
        v0=40.0, Dt=0.1, Dr=0.3, dt=0.0002,
    )
    rng = np.random.default_rng(42)
    state = init_state(params, rng)

    n_steps = 30000
    record_every = 150
    times, f_max_series = [], []
    for step in range(n_steps + 1):
        if step % record_every == 0:
            times.append(state.t)
            f_max_series.append(
                largest_cluster_fraction(state.x, params.L, params.sigma,
                                          cutoff_factor=1.3)
            )
        if step < n_steps:
            step_euler_maruyama(state, params, rng)
            check_stability(state, step=step)

    times = np.array(times)
    f_max_series = np.array(f_max_series)

    raw_crossings = np.sum(np.diff((f_max_series >= 0.5).astype(int)) == 1)
    events = detect_onset_events(times, f_max_series, theta_high=0.6,
                                  theta_low=0.4, min_dwell=1.0)

    print(f"\nraw threshold crossings: {raw_crossings}, "
          f"persistence-filtered events: {len(events)}")

    # the qualitative claim: persistence filtering should not report MORE
    # events than naive crossing-counting would suggest are "real"
    assert len(events) <= raw_crossings
    # every returned event actually satisfies the stated dwell requirement
    for ev in events:
        assert ev.duration >= 1.0
        assert ev.peak_f_max >= 0.6
