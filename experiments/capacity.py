"""
Competing capacity hypotheses: Phi_input, Phi_parallel, Phi_plus.

This is the first module in the pipeline that is NOT theory-neutral. It
makes an interpretive choice -- what "capacity" means -- and, per the
project's own methodological essay ("Prediction Before Interpretation"),
that choice is not asserted here. Three competing operationalizations are
implemented side by side, precisely so none of them is adopted by fiat.
Which one (if any) earns predictive standing is a question for the
identification and prediction stages, not for this module.

Note on units: the reference engine (integrator.py) parameterizes
propulsion directly as a velocity v0, not as a force f0 divided through a
mobility. The original design sketch defined capacity as f0*v0 (an active
power). Since this engine has no independent f0, capacity here is
expressed in units of v0 itself (f0 is implicitly fixed to 1) -- a
modeling choice, not a physical claim, and worth stating plainly rather
than silently absorbing into the numbers.

  Phi_input:    the imposed propulsion speed, v0. Constant in space for
                this constant-speed ABP model, by construction -- it
                carries essentially no spatial information and functions
                as a control field, not a serious capacity candidate.

  Phi_parallel: signed realized propulsion, <u_i . v_i>_h, where v_i is
                the particle's actual (realized) instantaneous velocity.
                Can go negative where collisions push a particle backward
                against its own orientation.

  Phi_plus:     usable forward propulsion, <max(0, u_i . v_i)>_h -- the
                non-negative part only. By construction, Phi_plus >=
                Phi_parallel everywhere (pointwise max(0,w) >= w, and the
                kernel weights are non-negative, so this survives
                averaging).

Both Phi_parallel and Phi_plus accept `velocity_source`:
  "total"         -- uses state.dx_total / dt (includes thermal noise)
  "deterministic" -- uses state.dx_det / dt (active + interaction only,
                     excludes thermal noise)

This is the fix flagged during the design/review phase: capacity built
from the total velocity is contaminated by thermal fluctuations, and
whether any later predictive signal survives when noise is excluded is
itself part of what the identification stage needs to test. Both
variants are provided rather than picking one silently.
"""
from __future__ import annotations

import numpy as np

from .fields import Grid, gaussian_kernel_weights


def phi_input(params) -> float:
    """
    The imposed propulsion speed, v0. Spatially uniform for this
    constant-speed ABP model -- returned as a scalar rather than a grid,
    since reconstructing it onto a grid would just paint the same number
    everywhere. Use this as the control/null field: any capacity variant
    that fails to beat this trivial constant has not shown anything.
    """
    return params.v0


def _particle_velocity(state, params, velocity_source: str) -> np.ndarray:
    if velocity_source == "total":
        return state.dx_total / params.dt
    elif velocity_source == "deterministic":
        return state.dx_det / params.dt
    else:
        raise ValueError(
            f"velocity_source must be 'total' or 'deterministic', got "
            f"{velocity_source!r}"
        )


def _particle_alignment(state, params, velocity_source: str) -> np.ndarray:
    """Per-particle u_i . v_i, shape (N,)."""
    v = _particle_velocity(state, params, velocity_source)
    return np.sum(state.u * v, axis=1)


def reconstruct_phi_parallel(state, params, grid: Grid, h: float,
                              velocity_source: str = "total",
                              eps: float = 1e-8) -> np.ndarray:
    """
    Phi_parallel(x,t) = <u_i . v_i>_h, a density-weighted (kernel) average,
    NOT a raw kernel sum -- this is a per-particle quantity being
    smoothed, not a count. Signed: can go negative under collision-driven
    backward motion.
    """
    alignment = _particle_alignment(state, params, velocity_source)
    W = gaussian_kernel_weights(grid.points, state.x, params.L, h)  # (M, N)
    numerator = (W @ alignment).reshape(grid.n, grid.n)
    denominator = W.sum(axis=1).reshape(grid.n, grid.n)
    return numerator / (denominator + eps)


def reconstruct_phi_plus(state, params, grid: Grid, h: float,
                          velocity_source: str = "total",
                          eps: float = 1e-8) -> np.ndarray:
    """
    Phi_plus(x,t) = <max(0, u_i . v_i)>_h. Non-negative by construction,
    and Phi_plus >= Phi_parallel pointwise on the grid (see module
    docstring).
    """
    alignment = _particle_alignment(state, params, velocity_source)
    alignment_plus = np.maximum(0.0, alignment)
    W = gaussian_kernel_weights(grid.points, state.x, params.L, h)
    numerator = (W @ alignment_plus).reshape(grid.n, grid.n)
    denominator = W.sum(axis=1).reshape(grid.n, grid.n)
    return numerator / (denominator + eps)


def reconstruct_all_capacity_variants(state, params, grid: Grid, h: float,
                                       velocity_source: str = "total"):
    """
    Convenience wrapper returning (phi_input_scalar, phi_parallel_field,
    phi_plus_field) together, since the identification stage will want
    all three side by side rather than choosing one in advance.
    """
    return (
        phi_input(params),
        reconstruct_phi_parallel(state, params, grid, h, velocity_source),
        reconstruct_phi_plus(state, params, grid, h, velocity_source),
    )
