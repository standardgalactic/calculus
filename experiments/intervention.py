"""
Intervention protocol.

Everything upstream of this module is observational: reconstructed
fields, regression, classification, all tested for correlation and
incremental predictive value, none of it capable of establishing that
capacity fields are anything more than statistically redundant with
density. Per the essay's own Section 4, observational prediction can
justify "early-warning variable" language at best; only a surviving
intervention test justifies "driver" or "dynamical coupling" language.

The intervention: a spatially and temporally localized activity pulse,

    v_i(t) = v0 * (1 + amplitude) for particle i within `radius` of
             `center` (periodic minimum-image distance), during
             [start_time, start_time + duration); v0 otherwise.

paired with a counterfactual CONTROL run using the identical initial
state and an independently-instantiated RNG seeded identically, so both
runs draw the exact same sequence of random numbers at every step
(common random numbers) -- the intervention only ever changes the
deterministic drift term, never which random draws occur, so treatment
and control are bit-identical whenever amplitude=0, and differences
that do appear are attributable to the perturbation itself rather than
sampling noise. This is the standard variance-reduction design for
causal comparisons in stochastic simulation.
"""
from __future__ import annotations

import copy
from dataclasses import dataclass

import numpy as np

from .integrator import step_euler_maruyama
from .diagnostics import check_stability, InstabilityDetected


@dataclass(frozen=True)
class LocalActivityPulse:
    center: tuple  # (x, y)
    radius: float
    amplitude: float  # v_i = v0 * (1 + amplitude) for affected particles
    start_time: float
    duration: float

    @property
    def end_time(self) -> float:
        return self.start_time + self.duration


def _periodic_distance(x: np.ndarray, center: tuple, L: float) -> np.ndarray:
    diff = x - np.asarray(center)
    diff -= L * np.round(diff / L)
    return np.sqrt(np.sum(diff ** 2, axis=1))


def affected_particle_mask(state, params, intervention: LocalActivityPulse) -> np.ndarray:
    """Boolean (N,) mask of particles within intervention.radius of center."""
    d = _periodic_distance(state.x, intervention.center, params.L)
    return d < intervention.radius


def compute_v0_field(state, params, intervention: LocalActivityPulse, t: float
                      ) -> np.ndarray:
    """
    Returns a (N,) per-particle v0 array: params.v0 * (1+amplitude) for
    particles currently within the intervention's spatial and temporal
    window, params.v0 otherwise. Recomputed every step because
    membership in the affected region can change as particles move
    through it (the intervention is a fixed region in space, not a
    fixed set of particles).
    """
    v0_field = np.full(params.N, params.v0)
    if intervention.start_time <= t < intervention.end_time:
        mask = affected_particle_mask(state, params, intervention)
        v0_field[mask] = params.v0 * (1.0 + intervention.amplitude)
    return v0_field


def clone_state(state):
    """Deep-copies a state for use as an independent simulation branch."""
    return copy.deepcopy(state)


@dataclass
class CounterfactualResult:
    treatment_state: object
    control_state: object
    treatment_history: dict  # name -> list of per-recorded-step values
    control_history: dict
    times: np.ndarray
    intervention: LocalActivityPulse
    unstable: bool
    failure_reason: str


def run_counterfactual_pair(base_state, params, seed: int,
                             intervention: LocalActivityPulse,
                             duration: float, record_every_time: float,
                             recorder) -> CounterfactualResult:
    """
    Runs treatment (intervention applied) and control (no intervention)
    branches from the SAME starting state, for `duration` time units,
    using two independently-instantiated RNGs seeded identically so both
    branches draw the same random numbers at every step -- common random
    numbers. `recorder(state, params) -> dict` is called at every
    recorded frame on both branches; its return values are collected
    into treatment_history / control_history (dict of lists, one list
    per key `recorder` returns).

    Both branches share the SAME base_state as their t=0 condition
    (deep-copied, not aliased) -- this function does not run its own
    burn-in; pass an already-equilibrated state.
    """
    treatment_state = clone_state(base_state)
    control_state = clone_state(base_state)

    rng_treatment = np.random.default_rng(seed)
    rng_control = np.random.default_rng(seed)

    n_steps = int(duration / params.dt)
    record_every = max(1, int(record_every_time / params.dt))

    t0 = treatment_state.t  # both branches start at the same time
    times = []
    treatment_history: dict = {}
    control_history: dict = {}

    unstable = False
    failure_reason = ""

    for step in range(n_steps + 1):
        if step % record_every == 0:
            t_rel = treatment_state.t - t0
            times.append(t_rel)
            for key, val in recorder(treatment_state, params).items():
                treatment_history.setdefault(key, []).append(val)
            for key, val in recorder(control_state, params).items():
                control_history.setdefault(key, []).append(val)

        if step < n_steps:
            t_rel_now = treatment_state.t - t0
            v0_treatment = compute_v0_field(treatment_state, params, intervention, t_rel_now)
            try:
                step_euler_maruyama(treatment_state, params, rng_treatment,
                                     v0_override=v0_treatment)
                check_stability(treatment_state, step=step)
                step_euler_maruyama(control_state, params, rng_control)
                check_stability(control_state, step=step)
            except InstabilityDetected as exc:
                unstable = True
                failure_reason = str(exc)
                break

    return CounterfactualResult(
        treatment_state=treatment_state, control_state=control_state,
        treatment_history=treatment_history, control_history=control_history,
        times=np.array(times), intervention=intervention,
        unstable=unstable, failure_reason=failure_reason,
    )
