"""
Onset detection and prospective labeling for MIPS.

M(t) = f_max(t) is volatile even in a system that never reaches a stable
answer to "has it clustered" -- Milestone 1.5's Figure 5 showed f_max
swinging 0.2-1.0 within a few time units at fixed cutoff. A raw threshold
crossing on f_max would label transient fluctuations as onset events and
corrupt every downstream label (Y_tau, Delta_tau M) before any regression
even runs.

This module implements a persistence-based definition instead: onset
requires crossing a high threshold, sustained above a lower threshold for
a minimum dwell time, optionally confirmed by elevated low-q structure
factor over the same window -- multiple independent structural signals
agreeing, per the original design intent ("persistent agreement among
cluster size...and density distribution").
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class OnsetEvent:
    start_time: float
    start_index: int
    end_time: float
    end_index: int
    peak_f_max: float
    duration: float


def _contiguous_true_runs(mask: np.ndarray):
    """Yields (start_idx, end_idx_inclusive) for each contiguous True run."""
    if not np.any(mask):
        return
    mask_int = mask.astype(int)
    diff = np.diff(mask_int)
    starts = list(np.where(diff == 1)[0] + 1)
    ends = list(np.where(diff == -1)[0])
    if mask[0]:
        starts = [0] + starts
    if mask[-1]:
        ends = ends + [len(mask) - 1]
    for s, e in zip(starts, ends):
        yield s, e


def detect_onset_events(times: np.ndarray, f_max: np.ndarray,
                         theta_high: float = 0.5, theta_low: float = 0.3,
                         min_dwell: float = 1.0,
                         s_low_q: np.ndarray | None = None,
                         s_low_q_threshold: float | None = None
                         ) -> list[OnsetEvent]:
    """
    Detects persistent MIPS-onset episodes in an f_max(t) trajectory.

    An episode qualifies as onset if, within one contiguous run of
    f_max >= theta_low:
      - the run's duration >= min_dwell, AND
      - the run's peak f_max >= theta_high (must actually cross the high
        threshold at some point, not just hover near the low one), AND
      - (if s_low_q and s_low_q_threshold are both given) the run's mean
        low-q structure factor >= s_low_q_threshold (independent
        structural confirmation).

    theta_high must be > theta_low. Returns one OnsetEvent per qualifying
    run, in time order. An event's start_time is the time f_max first
    crossed theta_low for that run -- the earliest defensible boundary of
    the elevated regime that turned out to be real, not the later moment
    it crossed theta_high.
    """
    if theta_high <= theta_low:
        raise ValueError("theta_high must be greater than theta_low")
    if len(times) != len(f_max):
        raise ValueError("times and f_max must be the same length")
    if s_low_q is not None and len(s_low_q) != len(f_max):
        raise ValueError("s_low_q must be the same length as f_max")

    mask = f_max >= theta_low
    events = []
    for s, e in _contiguous_true_runs(mask):
        duration = float(times[e] - times[s])
        peak = float(f_max[s:e + 1].max())
        if duration < min_dwell:
            continue
        if peak < theta_high:
            continue
        if s_low_q is not None and s_low_q_threshold is not None:
            if float(s_low_q[s:e + 1].mean()) < s_low_q_threshold:
                continue
        events.append(OnsetEvent(
            start_time=float(times[s]), start_index=int(s),
            end_time=float(times[e]), end_index=int(e),
            peak_f_max=peak, duration=duration,
        ))
    return events


def prospective_labels(times: np.ndarray, onset_events: list[OnsetEvent],
                        tau: float) -> np.ndarray:
    """
    Y_tau(t) = 1{some onset episode STARTS within (t, t+tau]}, for every t
    in `times`. Only counts the start of a qualifying onset episode (not
    every timestep within it) -- the prediction question is "will onset
    begin soon," not "are we currently inside an episode."
    """
    onset_starts = np.array([ev.start_time for ev in onset_events])
    Y = np.zeros(len(times), dtype=bool)
    if onset_starts.size == 0:
        return Y
    for i, t in enumerate(times):
        in_window = (onset_starts > t) & (onset_starts <= t + tau)
        Y[i] = bool(np.any(in_window))
    return Y


def delta_M(times: np.ndarray, M: np.ndarray, tau: float
            ) -> tuple[np.ndarray, np.ndarray]:
    """
    Delta_tau M(t) = M(t+tau) - M(t), computed at each t for which a
    recorded snapshot near t+tau exists. Snapshots are typically saved at
    fixed intervals rather than every simulation step, so this snaps to
    the nearest available sample to t+tau rather than interpolating.
    Returns (valid_times, delta_M_values), trimmed to only the t values
    where a target snapshot was found; the two arrays correspond
    one-to-one.
    """
    if len(times) != len(M):
        raise ValueError("times and M must be the same length")

    delta = np.full(len(times), np.nan)
    for i, t in enumerate(times):
        target = t + tau
        if target > times[-1]:
            continue
        j = int(np.searchsorted(times, target))
        if j >= len(times):
            j = len(times) - 1
        if j > 0 and abs(times[j - 1] - target) < abs(times[j] - target):
            j -= 1
        delta[i] = M[j] - M[i]

    valid = ~np.isnan(delta)
    return times[valid], delta[valid]
