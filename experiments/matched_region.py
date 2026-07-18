"""
Matched-region intervention experiment.

The mechanism demonstration (intervention.py) established that a local
activity pulse is dynamically consequential -- but a single arbitrary
region cannot say whether the SIZE of that consequence depends on the
region's pre-intervention capacity reading. This module builds the
actual causal test: identify candidate regions from a real
(post-burn-in) system, match them on density (the covariate the
observational null result was never able to rule out as the sole
explanation), record several further covariates for transparency
(polarization, dense-domain membership, local density gradient
magnitude) without yet using them in matching, then apply the SAME
pulse to density-matched high-capacity/low-capacity region pairs and
ask whether the treatment EFFECT differs by capacity after
(approximately) controlling for density.

Honest scope note: matching here is on density alone. The original
design specified additionally matching on polarization, cluster
membership, boundary geometry, and local density gradient. Those four
are computed and recorded per candidate region so they're available for
a stricter match or as covariates in a later regression, but the
matching procedure implemented here does not yet condition on them.
Treat this as a first, density-only-matched pass, not the fully
controlled design.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np

from .fields import Grid, reconstruct_density
from .capacity import reconstruct_phi_parallel
from .density_domains import dense_domain_fraction, _periodic_grid_cluster_fraction
from .features import _gradient
from .intervention import LocalActivityPulse, run_counterfactual_pair
from .density_domains import composite_mips_indicators


@dataclass
class RegionCandidate:
    center: tuple
    local_density: float
    local_capacity: float
    local_polarization: float
    in_dense_domain: bool
    local_density_gradient_mag: float


def _local_mean(field: np.ndarray, grid: Grid, center: tuple, radius: float) -> float:
    """Mean of a grid field within `radius` of `center` (periodic)."""
    coords = grid.points  # (n*n, 2)
    diff = coords - np.asarray(center)
    diff -= grid.L * np.round(diff / grid.L)
    dist = np.sqrt(np.sum(diff ** 2, axis=1))
    mask = dist < radius
    values = field.ravel()
    if not np.any(mask):
        return float("nan")
    return float(values[mask].mean())


def _local_polarization(state, params, center: tuple, radius: float) -> float:
    """|<u_i>| for particles within radius of center -- polar order
    parameter, 0 (disordered) to 1 (perfectly aligned)."""
    diff = state.x - np.asarray(center)
    diff -= params.L * np.round(diff / params.L)
    dist = np.sqrt(np.sum(diff ** 2, axis=1))
    mask = dist < radius
    if not np.any(mask):
        return float("nan")
    mean_u = state.u[mask].mean(axis=0)
    return float(np.linalg.norm(mean_u))


def find_region_candidates(state, params, grid: Grid, h: float,
                            candidate_centers: list, region_radius: float,
                            rho_star_factor: float = 1.3) -> list:
    """
    Reconstructs density and capacity fields once, then characterizes
    each candidate region (a list of (x, y) centers -- e.g. a coarse
    grid of trial locations) by local density, local capacity, local
    polarization, dense-domain membership, and local density-gradient
    magnitude.
    """
    rho_h = reconstruct_density(state, params, grid, h)
    phi_par = reconstruct_phi_parallel(state, params, grid, h,
                                        velocity_source="deterministic")

    mean_rho = rho_h.mean()
    dense_mask = rho_h > (rho_star_factor * mean_rho) if mean_rho > 0 else np.zeros_like(rho_h, dtype=bool)

    spacing = grid.L / grid.n
    gx, gy = _gradient(rho_h, spacing)
    grad_mag = np.sqrt(gx ** 2 + gy ** 2)

    candidates = []
    for center in candidate_centers:
        local_density = _local_mean(rho_h, grid, center, region_radius)
        local_capacity = _local_mean(phi_par, grid, center, region_radius)
        local_grad = _local_mean(grad_mag, grid, center, region_radius)
        local_pol = _local_polarization(state, params, center, region_radius)

        # dense-domain membership: is the nearest grid cell to `center`
        # part of the thresholded dense mask?
        coords = grid.points
        diff = coords - np.asarray(center)
        diff -= grid.L * np.round(diff / grid.L)
        dist = np.sqrt(np.sum(diff ** 2, axis=1))
        nearest_idx = np.argmin(dist)
        in_dense = bool(dense_mask.ravel()[nearest_idx])

        candidates.append(RegionCandidate(
            center=center, local_density=local_density,
            local_capacity=local_capacity, local_polarization=local_pol,
            in_dense_domain=in_dense, local_density_gradient_mag=local_grad,
        ))
    return candidates


def match_pairs_by_density(candidates: list, n_bins: int = 5,
                            min_capacity_contrast: float = 0.0) -> list:
    """
    Bins candidates by local_density into n_bins equal-width bins, and
    within each bin that has at least 2 valid (finite) candidates,
    pairs the highest-capacity and lowest-capacity candidate. Returns a
    list of (low_capacity, high_capacity) RegionCandidate tuples, one
    per qualifying bin, restricted to pairs whose capacity difference is
    at least min_capacity_contrast (skip bins where the "contrast" is
    negligible -- a matched pair with near-identical capacity can't
    inform whether capacity matters).
    """
    valid = [c for c in candidates if np.isfinite(c.local_density) and np.isfinite(c.local_capacity)]
    if len(valid) < 2:
        return []

    densities = np.array([c.local_density for c in valid])
    d_min, d_max = densities.min(), densities.max()
    if d_max == d_min:
        bin_edges = np.array([d_min, d_min + 1e-9])
        n_bins = 1
    else:
        bin_edges = np.linspace(d_min, d_max, n_bins + 1)

    pairs = []
    for i in range(len(bin_edges) - 1):
        in_bin = [c for c in valid
                  if bin_edges[i] <= c.local_density <= bin_edges[i + 1]]
        if len(in_bin) < 2:
            continue
        in_bin_sorted = sorted(in_bin, key=lambda c: c.local_capacity)
        low_cap, high_cap = in_bin_sorted[0], in_bin_sorted[-1]
        contrast = high_cap.local_capacity - low_cap.local_capacity
        if contrast >= min_capacity_contrast and low_cap.center != high_cap.center:
            pairs.append((low_cap, high_cap))
    return pairs


@dataclass
class InterventionResponse:
    region_label: str
    local_capacity: float
    local_density: float
    depletion_min: float       # minimum local count during/after pulse
    time_of_min: float
    recovery_time: Optional[float]  # first time local count >= control's value at that time, after depletion_min; None if never
    overshoot_max: float       # max(treatment - control) after the pulse ends
    integrated_diff: float     # sum over observation window of (treatment - control) * dt_record
    delayed_global_diff: float  # f_dense_max(treatment) - f_dense_max(control) at final recorded time


def compute_response_metrics(times: np.ndarray, treat_local: np.ndarray,
                              ctrl_local: np.ndarray, treat_global: np.ndarray,
                              ctrl_global: np.ndarray,
                              intervention: LocalActivityPulse,
                              region_label: str, local_capacity: float,
                              local_density: float) -> InterventionResponse:
    diff = treat_local - ctrl_local
    dt_record = float(np.median(np.diff(times))) if len(times) > 1 else 0.0

    depletion_idx = int(np.argmin(treat_local))
    depletion_min = float(treat_local[depletion_idx])
    time_of_min = float(times[depletion_idx])

    recovery_time = None
    for i in range(depletion_idx, len(times)):
        if treat_local[i] >= ctrl_local[i]:
            recovery_time = float(times[i])
            break

    after_pulse = times >= intervention.end_time
    overshoot_max = float(np.max(diff[after_pulse])) if np.any(after_pulse) else float("nan")

    integrated_diff = float(np.sum(diff) * dt_record)
    delayed_global_diff = float(treat_global[-1] - ctrl_global[-1])

    return InterventionResponse(
        region_label=region_label, local_capacity=local_capacity,
        local_density=local_density, depletion_min=depletion_min,
        time_of_min=time_of_min, recovery_time=recovery_time,
        overshoot_max=overshoot_max, integrated_diff=integrated_diff,
        delayed_global_diff=delayed_global_diff,
    )
