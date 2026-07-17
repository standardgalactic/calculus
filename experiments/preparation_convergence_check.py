"""
Small preparation-dependence convergence check.

Not a full sweep -- answers one narrow question: after a fixed burn-in
(t=6), is the gap between grid-started and RSA-started trajectories
small relative to seed-to-seed variation, or comparable/larger? That
determines whether burn-in can honestly be described as having erased
the preparation method's memory, or whether initialization protocol is a
real covariate that training/test splits need to account for later.

D_prep = || (f_dense_max, f_void, S_low_q)_grid
           - (f_dense_max, f_void, S_low_q)_RSA ||
per seed, compared against the within-method seed-to-seed spread of the
same three-component vector.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from rsvp_mips import (  # noqa: E402
    ABPParameters, init_state, init_state_random, run_burn_in,
    reconstruct_density,
)
from rsvp_mips.fields import Grid  # noqa: E402
from rsvp_mips.density_domains import composite_mips_indicators  # noqa: E402

N, PHI, SIGMA = 300, 0.5, 1.0
L = np.sqrt(N * np.pi * (SIGMA / 2.0) ** 2 / PHI)
PARAMS = ABPParameters(N=N, L=L, sigma=SIGMA, epsilon=1.0, mobility=1.0,
                        v0=20.0, Dt=0.1, Dr=0.3, dt=0.0001)
BURNIN = 6.0
SEEDS = [0, 1, 2]


def indicators_for(init_fn, seed, **init_kwargs):
    rng = np.random.default_rng(seed)
    state = init_fn(PARAMS, rng, **init_kwargs)
    run_burn_in(state, PARAMS, rng, duration=BURNIN)
    grid = Grid(L=L, n=24)
    rho_h = reconstruct_density(state, PARAMS, grid, h=1.2)
    ind = composite_mips_indicators(rho_h, L)
    return np.array([ind["f_dense_max"], ind["f_void"], ind["S_rho_low_q"]])


def main():
    grid_vecs, rsa_vecs = [], []
    for seed in SEEDS:
        gv = indicators_for(init_state, seed)
        rv = indicators_for(init_state_random, seed, min_distance_factor=1.02)
        grid_vecs.append(gv)
        rsa_vecs.append(rv)
        print(f"seed={seed}  grid=(f_dense={gv[0]:.3f}, f_void={gv[1]:.3f}, "
              f"S_low_q={gv[2]:.3f})  "
              f"rsa=(f_dense={rv[0]:.3f}, f_void={rv[1]:.3f}, "
              f"S_low_q={rv[2]:.3f})")

    grid_vecs = np.array(grid_vecs)
    rsa_vecs = np.array(rsa_vecs)

    # normalize each component by its combined (grid+rsa) std across all
    # samples, so S_low_q's larger scale doesn't dominate the norm
    combined = np.vstack([grid_vecs, rsa_vecs])
    comp_std = combined.std(axis=0)
    comp_std[comp_std == 0] = 1.0

    def norm_vec(v):
        return v / comp_std

    D_prep = np.array([
        np.linalg.norm(norm_vec(g) - norm_vec(r))
        for g, r in zip(grid_vecs, rsa_vecs)
    ])

    # within-method seed-to-seed spread, as the comparison baseline
    def pairwise_spread(vecs):
        diffs = []
        for i in range(len(vecs)):
            for j in range(i + 1, len(vecs)):
                diffs.append(np.linalg.norm(norm_vec(vecs[i]) - norm_vec(vecs[j])))
        return np.array(diffs)

    grid_spread = pairwise_spread(grid_vecs)
    rsa_spread = pairwise_spread(rsa_vecs)

    print(f"\nD_prep (grid vs RSA, matched seed): {D_prep}")
    print(f"  mean D_prep = {D_prep.mean():.3f}")
    print(f"grid seed-to-seed spread: {grid_spread}")
    print(f"  mean = {grid_spread.mean():.3f}")
    print(f"rsa seed-to-seed spread:  {rsa_spread}")
    print(f"  mean = {rsa_spread.mean():.3f}")

    combined_seed_spread = np.concatenate([grid_spread, rsa_spread]).mean()
    ratio = D_prep.mean() / combined_seed_spread if combined_seed_spread > 0 else float("inf")
    print(f"\nD_prep / within-method seed spread ratio: {ratio:.2f}")
    if ratio < 0.5:
        verdict = "preparation dependence negligible relative to seed variation"
    elif ratio < 1.5:
        verdict = "preparation dependence comparable to seed variation -- borderline"
    else:
        verdict = "preparation dependence exceeds seed variation -- real covariate, not erased by this burn-in"
    print(f"verdict: {verdict}")


if __name__ == "__main__":
    main()
