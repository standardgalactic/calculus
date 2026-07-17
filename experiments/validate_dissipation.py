"""
Visual comparison of the two dissipation estimators (S_work, S_current)
and their relationship to Phi_parallel.

Demonstrates two things visually that the tests already proved
numerically:
  1. S_work is an exact rescaling of Phi_parallel (same spatial pattern,
     different colorbar) -- they are collinear, not independent.
  2. S_current has a visibly different spatial pattern -- built from
     collective flow (rho_h, J_h) rather than per-particle force, it is
     a genuinely distinct estimator.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from rsvp_mips import ABPParameters, init_state, step_euler_maruyama  # noqa: E402
from rsvp_mips.fields import Grid, reconstruct_density_and_current  # noqa: E402
from rsvp_mips.capacity import reconstruct_phi_parallel  # noqa: E402
from rsvp_mips.dissipation import reconstruct_s_work, reconstruct_s_current  # noqa: E402
from rsvp_mips.diagnostics import check_stability  # noqa: E402

RESULTS_DIR = Path(__file__).resolve().parents[1] / "results"
RESULTS_DIR.mkdir(exist_ok=True)


def main():
    N = 300
    sigma = 1.0
    phi_pack = 0.5
    L = np.sqrt(N * np.pi * (sigma / 2.0) ** 2 / phi_pack)
    params = ABPParameters(
        N=N, L=L, sigma=sigma, epsilon=1.0, mobility=1.0,
        v0=30.0, Dt=0.3, Dr=0.3, dt=0.0002,
    )
    print(f"Running: phi={phi_pack}, Pe={params.Pe:.1f}, dt={params.dt}")

    rng = np.random.default_rng(30)
    state = init_state(params, rng)
    n_steps = 20000
    for step in range(n_steps):
        step_euler_maruyama(state, params, rng)
        check_stability(state, step=step)

    grid = Grid(L=params.L, n=48)
    h = 1.2

    rho_h, Jx_h, Jy_h, _, _ = reconstruct_density_and_current(state, params, grid, h)
    phi_par = reconstruct_phi_parallel(state, params, grid, h)
    s_work = reconstruct_s_work(state, params, grid, h)
    s_current = reconstruct_s_current(rho_h, Jx_h, Jy_h, params)

    mask = rho_h > 0.05 * rho_h.max()

    # confirm and report the exact algebraic relationship
    predicted_s_work = (params.v0 / params.Dt) * phi_par
    max_rel_err = np.max(np.abs(s_work - predicted_s_work) /
                          (np.abs(predicted_s_work) + 1e-8))
    print(f"\nS_work vs (v0/Dt)*Phi_parallel: max relative error = "
          f"{max_rel_err:.2e} (should be ~machine precision)")

    correlation = np.corrcoef(s_work[mask].ravel(), s_current[mask].ravel())[0, 1]
    print(f"Correlation(S_work, S_current) over occupied cells: "
          f"{correlation:.3f}")
    print(f"S_current range: [{s_current[mask].min():.3f}, "
          f"{s_current[mask].max():.3f}], mean={s_current[mask].mean():.3f}")
    print(f"S_work range:    [{s_work[mask].min():.3f}, "
          f"{s_work[mask].max():.3f}], mean={s_work[mask].mean():.3f}")

    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    extent = [0, params.L, 0, params.L]

    im0 = axes[0].imshow(phi_par, origin="lower", extent=extent, cmap="RdBu_r",
                          vmin=-params.v0, vmax=params.v0)
    axes[0].set_title(r"$\Phi_\parallel$ (capacity)")
    fig.colorbar(im0, ax=axes[0], fraction=0.046)

    im1 = axes[1].imshow(s_work, origin="lower", extent=extent, cmap="RdBu_r")
    axes[1].set_title(r"$\dot S_{work}$ (exact rescaling of $\Phi_\parallel$)")
    fig.colorbar(im1, ax=axes[1], fraction=0.046)

    im2 = axes[2].imshow(s_current, origin="lower", extent=extent, cmap="magma")
    axes[2].set_title(r"$\dot S_{current}$ (independent estimator)")
    fig.colorbar(im2, ax=axes[2], fraction=0.046)

    fig.suptitle(f"Dissipation Estimators vs. Capacity "
                  f"(corr(S_work, S_current) = {correlation:.2f})")
    fig.tight_layout()
    fig.savefig(RESULTS_DIR / "dissipation_estimators_comparison.png", dpi=150)
    plt.close(fig)
    print(f"\nFigure written to "
          f"{RESULTS_DIR / 'dissipation_estimators_comparison.png'}")


if __name__ == "__main__":
    main()
