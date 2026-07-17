"""
Visual comparison of Phi_input, Phi_parallel, Phi_plus on a real,
clustering ABP system.

Phi_input is trivially uniform (=v0) by construction -- included as the
control/null field. The interesting comparison is Phi_parallel vs
Phi_plus, and total-velocity vs deterministic-only variants.

This script does not decide which variant is "capacity." It only makes
the three candidates visible side by side, per the project's own
identification criterion (comparability without adjudication).
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
from rsvp_mips.fields import Grid, reconstruct_density  # noqa: E402
from rsvp_mips.capacity import (  # noqa: E402
    phi_input,
    reconstruct_phi_parallel,
    reconstruct_phi_plus,
)
from rsvp_mips.diagnostics import check_stability  # noqa: E402

RESULTS_DIR = Path(__file__).resolve().parents[1] / "results"
RESULTS_DIR.mkdir(exist_ok=True)


def main():
    # dense, high-activity, clustering system -- the regime where capacity
    # variants have the best chance of showing daylight between them
    N = 300
    sigma = 1.0
    phi_pack = 0.5
    L = np.sqrt(N * np.pi * (sigma / 2.0) ** 2 / phi_pack)
    v0 = 40.0
    params = ABPParameters(
        N=N, L=L, sigma=sigma, epsilon=1.0, mobility=1.0,
        v0=v0, Dt=0.1, Dr=0.3, dt=0.0001,
    )
    print(f"Running dense system: phi={phi_pack}, Pe={params.Pe:.1f}, "
          f"dt={params.dt}")

    rng = np.random.default_rng(20)
    state = init_state(params, rng)
    n_steps = 30000
    for step in range(n_steps):
        step_euler_maruyama(state, params, rng)
        check_stability(state, step=step)

    grid = Grid(L=params.L, n=48)
    h = 1.2
    rho_h = reconstruct_density(state, params, grid, h)

    phi_in = phi_input(params)  # scalar
    phi_par_total = reconstruct_phi_parallel(state, params, grid, h,
                                              velocity_source="total")
    phi_plus_total = reconstruct_phi_plus(state, params, grid, h,
                                           velocity_source="total")
    phi_par_det = reconstruct_phi_parallel(state, params, grid, h,
                                            velocity_source="deterministic")
    phi_plus_det = reconstruct_phi_plus(state, params, grid, h,
                                         velocity_source="deterministic")

    mask = rho_h > 0.05 * rho_h.max()
    print(f"\nPhi_input (uniform):        {phi_in:.3f}")
    print(f"Phi_parallel (total),  mean: {phi_par_total[mask].mean():.3f}  "
          f"min: {phi_par_total[mask].min():.3f}  "
          f"max: {phi_par_total[mask].max():.3f}")
    print(f"Phi_plus (total),      mean: {phi_plus_total[mask].mean():.3f}  "
          f"min: {phi_plus_total[mask].min():.3f}  "
          f"max: {phi_plus_total[mask].max():.3f}")
    print(f"Phi_parallel (det),    mean: {phi_par_det[mask].mean():.3f}  "
          f"min: {phi_par_det[mask].min():.3f}  "
          f"max: {phi_par_det[mask].max():.3f}")
    print(f"Phi_plus (det),        mean: {phi_plus_det[mask].mean():.3f}  "
          f"min: {phi_plus_det[mask].min():.3f}  "
          f"max: {phi_plus_det[mask].max():.3f}")

    fig, axes = plt.subplots(2, 3, figsize=(15, 9))
    extent = [0, params.L, 0, params.L]
    vmin, vmax = -v0, v0

    panels = [
        (axes[0, 0], rho_h, r"$\rho_h$", "viridis", None, None),
        (axes[0, 1], phi_par_total, r"$\Phi_\parallel$ (total velocity)",
         "RdBu_r", vmin, vmax),
        (axes[0, 2], phi_plus_total, r"$\Phi_+$ (total velocity)",
         "viridis", 0, v0),
        (axes[1, 0], np.full_like(rho_h, phi_in),
         r"$\Phi_{input}$ (control, uniform)", "RdBu_r", vmin, vmax),
        (axes[1, 1], phi_par_det, r"$\Phi_\parallel$ (deterministic only)",
         "RdBu_r", vmin, vmax),
        (axes[1, 2], phi_plus_det, r"$\Phi_+$ (deterministic only)",
         "viridis", 0, v0),
    ]
    for ax, field, title, cmap, vlo, vhi in panels:
        im = ax.imshow(field, origin="lower", extent=extent, cmap=cmap,
                        vmin=vlo, vmax=vhi)
        ax.set_title(title)
        fig.colorbar(im, ax=ax, fraction=0.046)

    fig.suptitle(f"Competing Capacity Variants (phi={phi_pack}, "
                  f"Pe={params.Pe:.0f})")
    fig.tight_layout()
    fig.savefig(RESULTS_DIR / "capacity_variants_comparison.png", dpi=150)
    plt.close(fig)
    print(f"\nFigure written to {RESULTS_DIR / 'capacity_variants_comparison.png'}")


if __name__ == "__main__":
    main()
