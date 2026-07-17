"""
Dissipation estimator family: S_work and S_current.

Per the design/review: "the apparently direct expression (1/T) F^active . v
is not automatically the unique or complete entropy-production rate."
Two estimators are implemented here precisely so the identification stage
can test whether any predictive signal survives a change in the
operational definition of dissipation, not just a change in capacity.

S_work: the direct force-times-velocity estimator.
    S_work_i = (F_active_i . v_i) / T_eff
  where F_active_i = (v0/mobility) * u_i is the active force implied by
  the overdamped relation mobility*F_active = v0*u (recovering a force
  from the velocity-parameterized propulsion this engine actually uses),
  and T_eff = Dt/mobility is an effective temperature via the Einstein
  relation (Dt = mobility*kB*T, kB=1 here -- an explicit modeling
  assumption, not a measured quantity).

  NOTE -- an algebraic identity worth stating plainly rather than
  discovering by surprise later: for THIS model, S_work is exactly
  proportional to Phi_parallel (same velocity_source):
      S_work(x,t) = (v0 / Dt) * Phi_parallel(x,t)
  This is a direct consequence of F_active being a constant multiple of
  u, and T_eff being a constant. It means S_work and Phi_parallel are NOT
  independent variables for the purposes of sparse regression -- they are
  perfectly collinear in this model, and including both in a candidate
  PDE library would be redundant, not informative. This is exactly the
  kind of thing Section 5 of the identification checklist (candidate
  library, not four hand-picked terms) is meant to catch, and it is
  caught here analytically rather than empirically.

S_current: an estimator built from the ALREADY-RECONSTRUCTED rho_h and
  J_h fields (kinematic.md/fields.py), not from a new per-particle force
  computation. Uses the standard Fokker-Planck local entropy-production-
  rate density for a probability current J in a medium with diffusivity D:
      S_current(x,t) = |J_h(x,t)|^2 / (Dt * rho_h(x,t) + eps)
  This is a genuinely different construction from S_work: it depends on
  the coarse-grained collective flow field, not on any individual
  particle's force. Non-negative by construction (it is a squared
  quantity over a positive weight).
"""
from __future__ import annotations

import numpy as np

from .fields import Grid, gaussian_kernel_weights


def effective_temperature(params) -> float:
    """T_eff = Dt / mobility (Einstein relation, kB=1)."""
    if params.Dt <= 0:
        raise ValueError(
            "effective_temperature requires Dt > 0 (S_work is undefined "
            "in the noiseless limit, where the Einstein relation gives "
            "T_eff = 0)."
        )
    return params.Dt / params.mobility


def _active_force(state, params) -> np.ndarray:
    """F_active_i = (v0 / mobility) * u_i, shape (N, 2)."""
    return (params.v0 / params.mobility) * state.u


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


def reconstruct_s_work(state, params, grid: Grid, h: float,
                        velocity_source: str = "total",
                        eps: float = 1e-8) -> np.ndarray:
    """
    S_work(x,t), a density-weighted (kernel) average of the per-particle
    work-based entropy-production estimator (F_active_i . v_i) / T_eff.
    """
    T_eff = effective_temperature(params)
    F_active = _active_force(state, params)
    v = _particle_velocity(state, params, velocity_source)
    per_particle = np.sum(F_active * v, axis=1) / T_eff

    W = gaussian_kernel_weights(grid.points, state.x, params.L, h)
    numerator = (W @ per_particle).reshape(grid.n, grid.n)
    denominator = W.sum(axis=1).reshape(grid.n, grid.n)
    return numerator / (denominator + eps)


def reconstruct_s_current(rho_h: np.ndarray, Jx_h: np.ndarray, Jy_h: np.ndarray,
                           params, eps: float = 1e-8) -> np.ndarray:
    """
    S_current(x,t) = |J_h|^2 / (Dt * rho_h + eps), built from the already-
    reconstructed density and current fields rather than any per-particle
    quantity. Non-negative by construction.

    Requires Dt > 0, for the same reason as S_work: this is a diffusive-
    medium entropy-production estimator and is undefined without thermal
    noise setting the reference diffusivity.
    """
    if params.Dt <= 0:
        raise ValueError(
            "reconstruct_s_current requires Dt > 0 (undefined without a "
            "reference diffusivity)."
        )
    J_sq = Jx_h ** 2 + Jy_h ** 2
    return J_sq / (params.Dt * rho_h + eps)
