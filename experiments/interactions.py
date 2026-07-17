"""
Pairwise Weeks-Chandler-Andersen (WCA) interaction forces under periodic
boundary conditions. Direct O(N^2) sweep — deliberately not cell-linked yet.

Per the milestone ordering: correctness before scale. A neighbor list is
introduced only after this reproduces known ABP/MIPS phenomenology.
"""
from __future__ import annotations

import numpy as np
from numba import njit

RCUT_FACTOR = 2.0 ** (1.0 / 6.0)  # 2^(1/6)


@njit(cache=True, fastmath=False)
def _compute_forces_kernel(x: np.ndarray, L: float, sigma: float,
                            epsilon: float) -> np.ndarray:
    N = x.shape[0]
    force = np.zeros((N, 2))
    rcut = RCUT_FACTOR * sigma
    rcut_sq = rcut * rcut
    sig_sq = sigma * sigma

    for i in range(N):
        xi0 = x[i, 0]
        xi1 = x[i, 1]
        for j in range(i + 1, N):
            dx0 = xi0 - x[j, 0]
            dx1 = xi1 - x[j, 1]
            # minimum image convention
            dx0 -= L * round(dx0 / L)
            dx1 -= L * round(dx1 / L)

            r_sq = dx0 * dx0 + dx1 * dx1
            if 1e-12 < r_sq < rcut_sq:
                inv_r2 = sig_sq / r_sq
                inv_r6 = inv_r2 * inv_r2 * inv_r2
                inv_r12 = inv_r6 * inv_r6
                # F = 24*eps/r^2 * [2*(sigma/r)^12 - (sigma/r)^6] * dx
                f_mag = (24.0 * epsilon / r_sq) * (2.0 * inv_r12 - inv_r6)
                fx = f_mag * dx0
                fy = f_mag * dx1
                force[i, 0] += fx
                force[i, 1] += fy
                force[j, 0] -= fx
                force[j, 1] -= fy

    return force


def compute_forces(state, params) -> None:
    """Computes WCA forces in-place on state.force."""
    state.force[:] = _compute_forces_kernel(
        state.x, params.L, params.sigma, params.epsilon
    )
