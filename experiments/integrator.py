"""
Euler-Maruyama integration of the overdamped ABP equations of motion.

Deliberately NOT Verlet: this is an overdamped (first-order) stochastic
system, not an inertial one, so velocities are not independent state
variables.

Deterministic and stochastic displacement components are tracked
separately (dx_det, dx_stoch) rather than only their sum. This matters
later: capacity fields built from dx_total would be kernel-width and
timestep dependent, contaminated by thermal noise. Keeping the components
separate lets the identification stage test whether any predictive signal
comes from active transport specifically, rather than from noise.
"""
from __future__ import annotations

import numpy as np

from .interactions import compute_forces


def step_euler_maruyama(state, params, rng: np.random.Generator) -> None:
    N, L, dt = params.N, params.L, params.dt
    v0, mu = params.v0, params.mobility

    # --- orientation update ---
    sqrt_2Dr_dt = np.sqrt(2.0 * params.Dr * dt)
    state.theta += sqrt_2Dr_dt * rng.standard_normal(N)
    # wrap into [-pi, pi] for numerical hygiene; u is derived, so wrapping
    # theta never contaminates displacement or orientation-correlation
    # statistics (see observables.orientation_autocorrelation_target).
    state.theta = np.mod(state.theta + np.pi, 2.0 * np.pi) - np.pi
    state.u[:, 0] = np.cos(state.theta)
    state.u[:, 1] = np.sin(state.theta)

    # --- interaction forces at current positions ---
    compute_forces(state, params)

    # --- deterministic and stochastic displacement, tracked separately ---
    state.dx_det[:] = (v0 * state.u + mu * state.force) * dt
    sqrt_2Dt_dt = np.sqrt(2.0 * params.Dt * dt)
    state.dx_stoch[:] = sqrt_2Dt_dt * rng.standard_normal((N, 2))

    dx = state.dx_det + state.dx_stoch

    # unwrapped positions accumulate true displacement (needed for MSD)
    state.x_unwrapped += dx
    # wrapped positions stay inside the periodic box (needed for WCA/RDF)
    state.x = np.mod(state.x + dx, L)

    state.t += dt
    state.step += 1
