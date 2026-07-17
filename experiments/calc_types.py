"""
Core state representation for the ABP engine.

Mirrors the Julia design's separation of concerns: this module knows nothing
about RSVP quantities (capacity, dissipation, etc). It only represents the
raw microscopic state of an active Brownian particle system.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


@dataclass(frozen=True)
class ABPParameters:
    N: int          # number of particles
    L: float        # box size (square, periodic)
    sigma: float    # WCA particle diameter
    epsilon: float  # WCA interaction strength
    mobility: float # mu
    v0: float       # self-propulsion speed
    Dt: float       # translational diffusion coefficient
    Dr: float       # rotational diffusion coefficient
    dt: float       # integration timestep

    @property
    def Pe(self) -> float:
        """Peclet number, v0 / (Dr * sigma)."""
        return self.v0 / (self.Dr * self.sigma)


@dataclass
class ABPState:
    x: np.ndarray            # (N, 2) wrapped positions, periodic box
    x_unwrapped: np.ndarray  # (N, 2) unwrapped positions, for correct MSD
    theta: np.ndarray        # (N,) orientation angle, wrapped to [-pi, pi]
    u: np.ndarray            # (N, 2) derived orientation unit vectors
    force: np.ndarray        # (N, 2) interaction forces (WCA)
    dx_det: np.ndarray       # (N, 2) deterministic displacement, last step
    dx_stoch: np.ndarray     # (N, 2) stochastic displacement, last step
    t: float = 0.0
    step: int = 0

    @property
    def dx_total(self) -> np.ndarray:
        return self.dx_det + self.dx_stoch


def init_state(params: ABPParameters, rng: np.random.Generator) -> ABPState:
    """
    Initialize particles on a grid (to avoid WCA overlap singularities at
    t=0) with uniformly random orientations.

    NOTE -- this initializer has a known limitation, discovered via the
    target-validity scan: at sufficiently high packing fraction (e.g.
    phi=0.5 at N=300), the grid spacing itself falls BELOW the cluster-
    detection cutoff distance, so particle-contact cluster fraction
    reads as fully "clustered" at t=0, before any dynamics -- a pure
    initialization artifact, not phase separation (confirmed
    independently via density bimodality, which does not track this
    false positive at all). For studies where the initial condition's
    structure could bias onset detection, prefer init_state_random below,
    or run a burn-in period (see equilibration.py) before treating
    recorded statistics as meaningful.
    """
    N, L = params.N, params.L
    grid_size = int(np.ceil(np.sqrt(N)))
    spacing = L / grid_size

    xs = np.zeros((N, 2), dtype=np.float64)
    for i in range(N):
        gx = (i % grid_size) * spacing + spacing / 2.0
        gy = (i // grid_size) * spacing + spacing / 2.0
        xs[i] = (gx, gy)

    theta = rng.uniform(-np.pi, np.pi, size=N)
    u = np.column_stack([np.cos(theta), np.sin(theta)])

    return ABPState(
        x=xs.copy(),
        x_unwrapped=xs.copy(),
        theta=theta,
        u=u,
        force=np.zeros((N, 2), dtype=np.float64),
        dx_det=np.zeros((N, 2), dtype=np.float64),
        dx_stoch=np.zeros((N, 2), dtype=np.float64),
        t=0.0,
        step=0,
    )


def init_state_random(params: ABPParameters, rng: np.random.Generator,
                       min_distance_factor: float = 1.05,
                       max_attempts_per_particle: int = 20000) -> ABPState:
    """
    Random sequential adsorption (RSA): places particles one at a time at
    uniformly random positions, rejecting any candidate closer than
    min_distance_factor * sigma (periodic minimum-image distance) to an
    already-placed particle, retrying until placed or a per-particle
    attempt budget is exhausted.

    Unlike init_state's grid placement, this does not impose a regular
    lattice structure that can itself satisfy a fixed contact-distance
    cluster criterion regardless of dynamics. It can still fail to place
    all N particles if the target packing fraction is too close to the
    RSA jamming limit for hard disks (~0.55 in 2D) -- in that regime,
    prefer a compression-from-dilute protocol instead (not implemented
    here), since rejection sampling becomes very inefficient near
    jamming.

    Raises RuntimeError if placement doesn't complete within the attempt
    budget, rather than silently returning an overlapping or incomplete
    configuration.
    """
    N, L, sigma = params.N, params.L, params.sigma
    min_dist_sq = (min_distance_factor * sigma) ** 2

    positions = np.zeros((N, 2), dtype=np.float64)
    placed = 0
    max_total_attempts = max_attempts_per_particle * N
    attempts = 0

    while placed < N and attempts < max_total_attempts:
        candidate = rng.uniform(0.0, L, size=2)
        attempts += 1
        if placed == 0:
            positions[0] = candidate
            placed = 1
            continue
        diff = positions[:placed] - candidate
        diff -= L * np.round(diff / L)
        dist_sq = np.sum(diff ** 2, axis=1)
        if np.all(dist_sq >= min_dist_sq):
            positions[placed] = candidate
            placed += 1

    if placed < N:
        raise RuntimeError(
            f"Random sequential placement failed to place all {N} "
            f"particles (placed {placed}) within {max_total_attempts} "
            f"attempts. The target packing fraction may be too close to "
            f"the RSA jamming limit for hard disks (~0.55 in 2D); "
            f"consider a compression-from-dilute protocol instead."
        )

    theta = rng.uniform(-np.pi, np.pi, size=N)
    u = np.column_stack([np.cos(theta), np.sin(theta)])

    return ABPState(
        x=positions.copy(),
        x_unwrapped=positions.copy(),
        theta=theta,
        u=u,
        force=np.zeros((N, 2), dtype=np.float64),
        dx_det=np.zeros((N, 2), dtype=np.float64),
        dx_stoch=np.zeros((N, 2), dtype=np.float64),
        t=0.0,
        step=0,
    )
