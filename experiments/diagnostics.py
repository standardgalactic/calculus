"""
Numerical stability diagnostics for the Euler-Maruyama / WCA combination.

Explicit integration of a stiff repulsive potential (WCA, ~1/r^13 near
contact) is only conditionally stable: a close encounter combined with too
large a timestep can produce a runaway force that inflates positions to
nonsense values within a handful of steps, silently corrupting every
downstream statistic (MSD, g(r), S(q), cluster fraction) without raising
an error. This module makes that failure mode loud instead of silent.

Empirically, for sigma=1, epsilon=1, mobility=1: dt=0.001 stays stable
(max force ~O(10-100) at contact) across the parameter ranges explored in
Milestone 1.5; dt=0.002 was observed to blow up to force magnitudes of
1e13-1e24 within a few hundred steps for otherwise unremarkable ABP
parameters. This is not a fixed rule for all parameter combinations -- it
is a reminder to actually check, not skip the check because "it worked
last time."
"""
from __future__ import annotations

import numpy as np


class InstabilityDetected(RuntimeError):
    pass


def max_force_magnitude(state) -> float:
    return float(np.max(np.linalg.norm(state.force, axis=1)))


def check_stability(state, threshold: float = 1.0e4, step: int | None = None) -> None:
    """
    Raises InstabilityDetected if the maximum force magnitude in the
    current state exceeds `threshold`. Call this after compute_forces (or
    after step_euler_maruyama, which calls compute_forces internally)
    during any run whose results will be trusted downstream.

    threshold=1e4 is a loose bound -- physically reasonable WCA forces at
    contact for epsilon=O(1), sigma=O(1) are O(1-100). Anything above 1e4
    almost certainly indicates the integrator has already overshot into
    the unstable regime, even if the run does not visibly diverge to inf
    within the recorded window.
    """
    fmag = max_force_magnitude(state)
    if fmag > threshold:
        where = f" at step {step}" if step is not None else ""
        raise InstabilityDetected(
            f"Max force magnitude {fmag:.3e}{where} exceeds stability "
            f"threshold {threshold:.1e}. This almost always means dt is "
            f"too large for the given epsilon/sigma/mobility combination "
            f"under explicit Euler-Maruyama integration of WCA. Reduce dt "
            f"and re-run rather than trusting anything computed so far."
        )
