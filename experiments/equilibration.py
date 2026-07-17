"""
Burn-in / equilibration.

Advances a state for a fixed duration without recording anything, so
that whatever initial-condition structure was imposed by init_state or
init_state_random has a chance to decorrelate under the actual dynamics
before any statistic is treated as meaningful. Per the target-validity
scan findings: initialization choice alone does not guarantee the
recorded trajectory is free of preparation-method artifacts, so this
should be used and its effect checked explicitly (e.g. by comparing
statistics recorded with vs without burn-in), not assumed sufficient on
its own.
"""
from __future__ import annotations

import numpy as np

from .integrator import step_euler_maruyama
from .diagnostics import check_stability


def run_burn_in(state, params, rng: np.random.Generator, duration: float,
                 check_every: bool = True) -> None:
    """
    Advances `state` in place by `duration` time units (no return value;
    mutates state). If check_every is True (default), runs the
    numerical-stability check after every step -- burn-in periods are
    exactly the kind of long, unmonitored stretch where a silent
    force-blowup is easy to miss if nothing is being recorded to notice
    it by eye.
    """
    n_steps = int(duration / params.dt)
    for step in range(n_steps):
        step_euler_maruyama(state, params, rng)
        if check_every:
            check_stability(state, step=step)
