"""
Kernel-based coarse-graining: density, current, and velocity fields.

This is the first stage that touches anything field-like, and it is
deliberately still theory-neutral. Density and current are purely
kinematic -- they require no interpretive choice about "capacity" or
"dissipation." The competing capacity hypotheses (Phi_input, Phi_parallel,
Phi_plus) are introduced only in a later module, built on top of these
fields, per the architectural principle: kinematics first, interpretation
after.

rho_h(x,t) = sum_i K_h(x - x_i)
J_h(x,t)   = sum_i xdot_i K_h(x - x_i)
v_h(x,t)   = J_h(x,t) / (rho_h(x,t) + eps)

K_h is a normalized 2D Gaussian kernel with periodic (minimum-image)
distance, so rho_h integrates to N over the box regardless of where mass
sits relative to the boundary.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class Grid:
    """A regular square grid of evaluation points covering the periodic box."""
    L: float
    n: int  # points per side

    @property
    def spacing(self) -> float:
        return self.L / self.n

    @property
    def points(self) -> np.ndarray:
        """(n*n, 2) array of grid-cell center coordinates."""
        coords = (np.arange(self.n) + 0.5) * self.spacing
        gx, gy = np.meshgrid(coords, coords, indexing="xy")
        return np.column_stack([gx.ravel(), gy.ravel()])

    @property
    def cell_area(self) -> float:
        return self.spacing ** 2


def _minimum_image_diff(a: np.ndarray, b: np.ndarray, L: float) -> np.ndarray:
    """
    a: (M, 2), b: (N, 2) -> (M, N, 2) minimum-image displacement a[m] - b[n].
    """
    diff = a[:, None, :] - b[None, :, :]
    return diff - L * np.round(diff / L)


def gaussian_kernel_weights(grid_points: np.ndarray, particle_positions: np.ndarray,
                             L: float, h: float) -> np.ndarray:
    """
    Returns (n_grid_points, N) weight matrix W[m, i] = K_h(x_m - x_i), using
    a normalized 2D Gaussian kernel and periodic minimum-image distance.

    K_h(r) = 1/(2 pi h^2) * exp(-|r|^2 / (2 h^2))

    This integrates to 1 over an infinite domain; on a finite periodic box
    it integrates to ~1 as long as h is small relative to L (checked by
    the density sanity test in scripts/validate_fields.py).
    """
    diff = _minimum_image_diff(grid_points, particle_positions, L)  # (M, N, 2)
    r_sq = np.sum(diff ** 2, axis=2)
    norm = 1.0 / (2.0 * np.pi * h * h)
    return norm * np.exp(-r_sq / (2.0 * h * h))


def reconstruct_density(state, params, grid: Grid, h: float) -> np.ndarray:
    """
    rho_h on the grid, shape (n, n).

    Purely kinematic: depends only on particle positions.
    """
    W = gaussian_kernel_weights(grid.points, state.x, params.L, h)  # (M, N)
    rho_flat = W.sum(axis=1)
    return rho_flat.reshape(grid.n, grid.n)


def reconstruct_density_and_current(state, params, grid: Grid, h: float,
                                     eps: float = 1e-8):
    """
    Returns (rho_h, Jx_h, Jy_h, vx_h, vy_h), each shape (n, n).

    Velocity here is the *total* instantaneous displacement over dt
    (state.dx_total / dt), i.e. the raw kinematic velocity including
    thermal noise. This is deliberate: J_h and v_h are purely kinematic
    fields. Separating "active" from "thermal" transport is exactly the
    kind of interpretive move deferred to the capacity-hypothesis stage
    (Phi_input / Phi_parallel / Phi_plus), which is NOT part of this
    module.
    """
    W = gaussian_kernel_weights(grid.points, state.x, params.L, h)  # (M, N)
    rho_flat = W.sum(axis=1)

    velocity = state.dx_total / params.dt  # (N, 2), instantaneous velocity
    Jx_flat = W @ velocity[:, 0]
    Jy_flat = W @ velocity[:, 1]

    rho_h = rho_flat.reshape(grid.n, grid.n)
    Jx_h = Jx_flat.reshape(grid.n, grid.n)
    Jy_h = Jy_flat.reshape(grid.n, grid.n)

    vx_h = Jx_h / (rho_h + eps)
    vy_h = Jy_h / (rho_h + eps)

    return rho_h, Jx_h, Jy_h, vx_h, vy_h
