"""
Visual validation of kernel-based field reconstruction (rho_h, J_h, v_h).

Runs two systems -- a dilute, non-clustering one and a dense, clustering
one -- and plots the reconstructed density and velocity fields for each.
Purely a sanity check that the fields look like what the particle
configuration actually is: density should visibly track particle
clumping, and velocity vectors should visibly align with the local
propulsion direction in aligned clusters.

Still theory-neutral. No capacity/dissipation quantity appears here.
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
from rsvp_mips.diagnostics import check_stability  # noqa: E402

RESULTS_DIR = Path(__file__).resolve().parents[1] / "results"
RESULTS_DIR.mkdir(exist_ok=True)


def run_and_plot(params: ABPParameters, n_steps: int, seed: int, h: float,
                  grid_n: int, title: str, out_name: str) -> None:
    rng = np.random.default_rng(seed)
    state = init_state(params, rng)

    for step in range(n_steps):
        step_euler_maruyama(state, params, rng)
        check_stability(state, step=step)

    grid = Grid(L=params.L, n=grid_n)
    rho_h, Jx_h, Jy_h, vx_h, vy_h = reconstruct_density_and_current(
        state, params, grid, h=h
    )

    fig, axes = plt.subplots(1, 2, figsize=(11, 5))

    ax = axes[0]
    im = ax.imshow(rho_h, origin="lower", extent=[0, params.L, 0, params.L],
                    cmap="viridis")
    ax.scatter(state.x[:, 0], state.x[:, 1], s=4, c="white", alpha=0.5,
               linewidths=0)
    ax.set_title(r"$\rho_h(x,t)$ with particle positions")
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    fig.colorbar(im, ax=ax, label=r"$\rho_h$")

    ax = axes[1]
    # subsample the velocity field for a readable quiver plot
    step_q = max(1, grid_n // 20)
    coords = (np.arange(grid_n) + 0.5) * grid.spacing
    X, Y = np.meshgrid(coords, coords)
    im2 = ax.imshow(rho_h, origin="lower", extent=[0, params.L, 0, params.L],
                     cmap="Greys", alpha=0.4)
    ax.quiver(
        X[::step_q, ::step_q], Y[::step_q, ::step_q],
        vx_h[::step_q, ::step_q], vy_h[::step_q, ::step_q],
        color="crimson", scale_units="xy",
    )
    ax.set_title(r"$v_h(x,t)$ (velocity field)")
    ax.set_xlabel("x")
    ax.set_ylabel("y")

    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(RESULTS_DIR / out_name, dpi=150)
    plt.close(fig)

    print(f"  {title}: integral(rho_h) = {rho_h.sum() * grid.cell_area:.1f} "
          f"(N = {params.N})")


if __name__ == "__main__":
    print("Dilute, non-clustering system...")
    dilute = ABPParameters(
        N=250, L=np.sqrt(250 * np.pi * 0.25 / 0.2), sigma=1.0, epsilon=1.0,
        mobility=1.0, v0=10.0, Dt=0.1, Dr=0.5, dt=0.001,
    )
    run_and_plot(dilute, n_steps=8000, seed=10, h=1.2, grid_n=48,
                 title="Dilute system (phi=0.2, Pe=20)",
                 out_name="fields_dilute.png")

    print("\nDense, clustering system...")
    dense = ABPParameters(
        N=250, L=np.sqrt(250 * np.pi * 0.25 / 0.5), sigma=1.0, epsilon=1.0,
        mobility=1.0, v0=60.0, Dt=0.1, Dr=0.5, dt=0.0001,
    )
    run_and_plot(dense, n_steps=25000, seed=11, h=1.2, grid_n=48,
                 title="Dense, high-activity system (phi=0.5, Pe=120)",
                 out_name="fields_dense.png")

    print(f"\nFigures written to {RESULTS_DIR}")
