"""
Milestone 1.5 diagnostics: benchmark observables used to validate the
engine against known ABP/MIPS phenomenology *before* any RSVP field is
reconstructed. Nothing here knows about capacity, dissipation, or RSVP.
"""
from __future__ import annotations

import numpy as np


def orientation_autocorrelation_target(Dr: float, t: np.ndarray) -> np.ndarray:
    """
    Analytic prediction C_u(t) = <u(0).u(t)> = exp(-Dr * t) for free
    rotational diffusion. Used as the overlay/ground truth for the
    orientation-autocorrelation benchmark (Figure 1 of the validation
    milestone), NOT as a replacement for measuring C_u(t) from simulation.

    This replaces a naive Var(theta) test: wrapped angles saturate to a
    bounded variance at long times even though the underlying unwrapped
    process keeps growing, which makes Var(theta_wrapped) fail for the
    wrong reason. u(0).u(t) requires no unwrapping and is the physically
    relevant quantity in the first place.
    """
    return np.exp(-Dr * np.asarray(t))


def orientation_autocorrelation(u_history: np.ndarray) -> np.ndarray:
    """
    Measures C_u(t) = <u(0).u(t)> from a recorded trajectory of orientation
    vectors. u_history has shape (T, N, 2). Returns array of length T.
    """
    u0 = u_history[0]  # (N, 2)
    dots = np.einsum("tnk,nk->tn", u_history, u0)  # (T, N)
    return dots.mean(axis=1)


def mean_squared_displacement(x_unwrapped_history: np.ndarray) -> np.ndarray:
    """
    MSD(t) = <|x_unwrapped(t) - x_unwrapped(0)|^2>, using unwrapped
    coordinates. Using wrapped positions here silently corrupts the curve
    whenever a particle crosses a periodic boundary — this is a common and
    easy-to-miss bug, so unwrapped coordinates are the only accepted input.

    x_unwrapped_history has shape (T, N, 2).
    """
    disp = x_unwrapped_history - x_unwrapped_history[0]
    sq = np.sum(disp**2, axis=2)  # (T, N)
    return sq.mean(axis=1)


def radial_distribution_function(x: np.ndarray, L: float, sigma: float,
                                  r_max: float | None = None,
                                  n_bins: int = 100) -> tuple[np.ndarray, np.ndarray]:
    """
    Radial distribution function g(r) via minimum-image pair distances.
    For an ideal gas (v0=0, epsilon=0) this should validate to g(r) ~ 1
    for all admissible r — a required sanity check before trusting this
    on an interacting system.
    """
    N = x.shape[0]
    if r_max is None:
        r_max = L / 2.0

    diff = x[:, None, :] - x[None, :, :]
    diff -= L * np.round(diff / L)
    r = np.sqrt(np.sum(diff**2, axis=2))
    iu = np.triu_indices(N, k=1)
    r = r[iu]
    r = r[r < r_max]

    counts, edges = np.histogram(r, bins=n_bins, range=(0.0, r_max))
    r_centers = 0.5 * (edges[1:] + edges[:-1])
    shell_area = 2.0 * np.pi * r_centers * (edges[1] - edges[0])
    rho = N / (L * L)
    # each pair counted once; normalize per-particle by N, and factor of 2
    # for the double-counting convention standard to g(r)
    g = (2.0 * counts) / (N * rho * shell_area + 1e-30)
    return r_centers, g


def radial_structure_factor(x: np.ndarray, L: float, q_max_idx: int = 15,
                             n_shells: int = 20) -> tuple[np.ndarray, np.ndarray]:
    """
    Static structure factor S(q), radially averaged over shells |q| = const
    rather than sampled only in the positive quadrant. This makes MIPS
    diagnostics much less grid-dependent than a raw (qx, qy) grid.

    For an ideal gas (v0=0, epsilon=0), S(q) should validate to ~1 for all
    q > 0.
    """
    N = x.shape[0]
    dq = 2.0 * np.pi / L

    qx = np.arange(-q_max_idx, q_max_idx + 1) * dq
    qy = np.arange(-q_max_idx, q_max_idx + 1) * dq
    QX, QY = np.meshgrid(qx, qy)
    Qmag = np.sqrt(QX**2 + QY**2)
    mask = Qmag > 1e-12
    QX, QY, Qmag = QX[mask], QY[mask], Qmag[mask]

    # rho(q) = sum_j exp(-i q . x_j)
    phase = QX[:, None] * x[None, :, 0] + QY[:, None] * x[None, :, 1]
    rho_q = np.exp(-1j * phase).sum(axis=1)
    S_q = (np.abs(rho_q) ** 2) / N

    q_edges = np.linspace(0, Qmag.max(), n_shells + 1)
    q_centers = 0.5 * (q_edges[1:] + q_edges[:-1])
    S_radial = np.zeros(n_shells)
    for k in range(n_shells):
        in_shell = (Qmag >= q_edges[k]) & (Qmag < q_edges[k + 1])
        if np.any(in_shell):
            S_radial[k] = S_q[in_shell].mean()
        else:
            S_radial[k] = np.nan

    return q_centers, S_radial


def largest_cluster_fraction(x: np.ndarray, L: float, sigma: float,
                              cutoff_factor: float = 1.3) -> float:
    """
    Largest connected-cluster fraction f_max = C_max / N, via union-find
    on a minimum-image distance cutoff.

    cutoff_factor is a modeling choice, not a physical constant — the
    identification study downstream uses cluster onset as a prediction
    target, so this function's caller should sweep cutoff_factor across
    multiple values (e.g. 1.2, 1.3, 1.4, 1.5 sigma) and confirm the
    resulting transition times are stable before trusting any single
    value.
    """
    N = x.shape[0]
    cutoff_sq = (cutoff_factor * sigma) ** 2

    parent = np.arange(N)

    def find(i: int) -> int:
        root = i
        while parent[root] != root:
            root = parent[root]
        while parent[i] != root:
            parent[i], i = root, parent[i]
        return root

    def union(i: int, j: int) -> None:
        ri, rj = find(i), find(j)
        if ri != rj:
            parent[ri] = rj

    diff = x[:, None, :] - x[None, :, :]
    diff -= L * np.round(diff / L)
    r_sq = np.sum(diff**2, axis=2)
    iu = np.triu_indices(N, k=1)
    close = r_sq[iu] < cutoff_sq
    ii, jj = iu[0][close], iu[1][close]
    for i, j in zip(ii, jj):
        union(int(i), int(j))

    roots = np.array([find(i) for i in range(N)])
    _, counts = np.unique(roots, return_counts=True)
    return counts.max() / N
