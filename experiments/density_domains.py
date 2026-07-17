"""
Density-domain-based order parameters for MIPS.

Corrects a measured flaw in the particle-contact cluster fraction f_max:
at sufficiently high packing fraction, a FIXED distance cutoff on
particle contacts becomes nearly complete purely from packing geometry,
independent of whether real phase separation (dense clusters coexisting
with dilute voids) has occurred. Verified directly during the target-
validity scan: at phi=0.5, grid-initialized particles already satisfy
f_max=1.0 at t=0, before any dynamics, and density bimodality does not
track f_max at all in that regime (correlation near zero or negative) --
confirming no real dense/dilute coexistence is present despite f_max
reporting "complete clustering."

This module builds MIPS indicators from the coarse-grained density field
rho_h instead of particle contacts, separating three questions the old
metric collapsed into one:
  - local contact               (the old f_max; retained elsewhere as a
                                   diagnostic, NOT the primary order
                                   parameter)
  - dense-domain membership      (f_dense_max: largest connected region
                                   of rho_h above a dense threshold)
  - phase coexistence            (f_void, B_rho: is there ALSO a dilute
                                   complement, not just a dense region)

Thresholds are defined relative to the field's OWN mean density
(rho_star_factor * mean(rho_h), not an absolute value), so the
heterogeneity criteria don't fire just because overall packing fraction
is high. A perfectly homogeneous dense fluid has rho_h close to its own
mean everywhere and should show f_dense_max ~ 0 and f_void ~ 0, however
high the absolute density is -- what matters is CONTRAST relative to the
run's own mean, not the absolute density level.
"""
from __future__ import annotations

import numpy as np

from .features import field_structure_factor_low_q, bimodality_coefficient


def _periodic_grid_cluster_fraction(mask: np.ndarray) -> float:
    """
    Largest connected-component occupancy fraction of a boolean grid
    under periodic (toroidal) 4-connectivity, via union-find -- the grid
    analogue of largest_cluster_fraction for particles. Cells on opposite
    edges of the grid are treated as neighbors, consistent with the
    periodic simulation box.
    """
    n = mask.shape[0]
    total_true = int(mask.sum())
    if total_true == 0:
        return 0.0

    def idx(i, j):
        return i * n + j

    parent = np.arange(n * n)

    def find(x):
        root = x
        while parent[root] != root:
            root = parent[root]
        while parent[x] != root:
            parent[x], x = root, parent[x]
        return root

    def union(a, b):
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[ra] = rb

    for i in range(n):
        for j in range(n):
            if not mask[i, j]:
                continue
            jr = (j + 1) % n
            if mask[i, jr]:
                union(idx(i, j), idx(i, jr))
            ir = (i + 1) % n
            if mask[ir, j]:
                union(idx(i, j), idx(ir, j))

    true_cells = [idx(i, j) for i in range(n) for j in range(n) if mask[i, j]]
    roots = np.array([find(c) for c in true_cells])
    _, counts = np.unique(roots, return_counts=True)
    return float(counts.max()) / (n * n)


def dense_domain_fraction(rho_h: np.ndarray, rho_star_factor: float = 1.3) -> float:
    """
    f_dense_max(t): occupancy fraction of the LARGEST connected component
    of grid cells where rho_h exceeds rho_star_factor * mean(rho_h).
    """
    mean_rho = rho_h.mean()
    if mean_rho <= 0:
        return 0.0
    mask = rho_h > (rho_star_factor * mean_rho)
    return _periodic_grid_cluster_fraction(mask)


def void_fraction(rho_h: np.ndarray, rho_low_factor: float = 0.7) -> float:
    """
    f_void(t): fraction of grid cells with rho_h below
    rho_low_factor * mean(rho_h). Not restricted to a connected
    component -- total dilute area is what matters for the coexistence
    check, not whether the dilute region is contiguous.
    """
    mean_rho = rho_h.mean()
    if mean_rho <= 0:
        return float("nan")
    return float(np.mean(rho_h < (rho_low_factor * mean_rho)))


def composite_mips_indicators(rho_h: np.ndarray, L: float,
                               rho_star_factor: float = 1.3,
                               rho_low_factor: float = 0.7) -> dict:
    """
    Bundles the four indicators the design calls for: f_dense_max,
    S_rho(q_low), B_rho, f_void. Onset should require persistent
    agreement among at least two of these, not a single graph threshold.
    """
    return {
        "f_dense_max": dense_domain_fraction(rho_h, rho_star_factor),
        "f_void": void_fraction(rho_h, rho_low_factor),
        "S_rho_low_q": field_structure_factor_low_q(rho_h, L),
        "B_rho": bimodality_coefficient(rho_h),
    }
