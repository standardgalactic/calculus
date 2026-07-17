"""
Milestone 1.5: Physical Validation.

Confirms the engine reproduces known ABP/MIPS phenomenology *before* any
RSVP quantity is reconstructed. This script touches no capacity, transport,
or dissipation fields -- it only benchmarks the theory-neutral simulator
against standard active-matter diagnostics.

Produces:
  Figure 1 - orientation autocorrelation vs analytic exp(-Dr t)
  Figure 2 - MSD, log-log, showing ballistic -> diffusive crossover
  Figure 3 - radial distribution function g(r)
  Figure 4 - radially-averaged structure factor S(q)
  Figure 5 - largest-cluster fraction f_max(t) across four cutoff choices
  Figure 6 - coarse phase scan heatmap over (phi, Pe) -> final f_max

Plus two sanity checks run first and printed to stdout: the ideal-gas
limit (v0=0, epsilon=0) should give g(r) ~ 1 and S(q) ~ 1 everywhere.
These sanity checks are a precondition for trusting Figures 3-4 on the
interacting system; if they fail, nothing downstream should be trusted.
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from rsvp_mips import (  # noqa: E402
    ABPParameters,
    init_state,
    step_euler_maruyama,
    radial_structure_factor,
    radial_distribution_function,
    largest_cluster_fraction,
)
from rsvp_mips.observables import (  # noqa: E402
    orientation_autocorrelation,
    orientation_autocorrelation_target,
    mean_squared_displacement,
)
from rsvp_mips.diagnostics import check_stability, InstabilityDetected  # noqa: E402

RESULTS_DIR = Path(__file__).resolve().parents[1] / "results"
RESULTS_DIR.mkdir(exist_ok=True)


def packing_fraction(N: int, L: float, sigma: float) -> float:
    return N * np.pi * (sigma / 2.0) ** 2 / (L * L)


# ---------------------------------------------------------------------------
# Sanity checks: ideal-gas limit
# ---------------------------------------------------------------------------

def sanity_check_ideal_gas() -> bool:
    print("=" * 70)
    print("Sanity check: ideal-gas limit (v0=0, epsilon=0)")
    print("Expect g(r) ~ 1 and S(q) ~ 1 everywhere.")
    print("=" * 70)

    params = ABPParameters(
        N=400, L=30.0, sigma=1.0, epsilon=0.0, mobility=1.0,
        v0=0.0, Dt=0.5, Dr=0.5, dt=0.01,
    )
    rng = np.random.default_rng(0)
    state = init_state(params, rng)

    for _ in range(2000):
        step_euler_maruyama(state, params, rng)

    r, g = radial_distribution_function(state.x, params.L, params.sigma)
    q, S = radial_structure_factor(state.x, params.L, q_max_idx=10)

    # ignore the smallest-r / smallest-q bins, which are noisy at finite N
    g_mid = g[(r > 2.0) & (r < params.L / 3)]
    S_mid = S[~np.isnan(S)]

    g_ok = np.abs(np.nanmean(g_mid) - 1.0) < 0.15
    S_ok = np.abs(np.nanmean(S_mid) - 1.0) < 0.3

    print(f"  mean g(r) [mid-range]: {np.nanmean(g_mid):.3f} "
          f"({'OK' if g_ok else 'FAILED'})")
    print(f"  mean S(q):             {np.nanmean(S_mid):.3f} "
          f"({'OK' if S_ok else 'FAILED'})")

    passed = g_ok and S_ok
    print(f"  -> {'PASSED' if passed else 'FAILED'}\n")
    return passed


# ---------------------------------------------------------------------------
# Figure 1: orientation autocorrelation
# ---------------------------------------------------------------------------

def figure_1_orientation_autocorrelation():
    print("Figure 1: orientation autocorrelation...")
    params = ABPParameters(
        N=1000, L=50.0, sigma=1.0, epsilon=1.0, mobility=1.0,
        v0=0.0, Dt=0.0, Dr=1.0, dt=0.005,
    )
    rng = np.random.default_rng(1)
    state = init_state(params, rng)
    state.theta[:] = 0.0
    state.u[:, 0] = 1.0
    state.u[:, 1] = 0.0

    n_steps = 1500
    u_hist = np.empty((n_steps + 1, params.N, 2))
    times = np.empty(n_steps + 1)
    u_hist[0], times[0] = state.u.copy(), state.t

    for i in range(1, n_steps + 1):
        step_euler_maruyama(state, params, rng)
        u_hist[i], times[i] = state.u.copy(), state.t

    C_measured = orientation_autocorrelation(u_hist)
    C_theory = orientation_autocorrelation_target(params.Dr, times)

    fig, ax = plt.subplots(figsize=(6, 4.5))
    ax.plot(times, C_measured, label="measured", lw=2)
    ax.plot(times, C_theory, "--", label=r"$e^{-D_r t}$", lw=2)
    ax.set_xlabel("t")
    ax.set_ylabel(r"$C_u(t) = \langle u(0)\cdot u(t)\rangle$")
    ax.set_title("Figure 1: Orientation Autocorrelation")
    ax.legend()
    fig.tight_layout()
    fig.savefig(RESULTS_DIR / "figure1_orientation_autocorrelation.png", dpi=150)
    plt.close(fig)

    max_err = np.max(np.abs(C_measured[times < 3.0] - C_theory[times < 3.0]))
    print(f"  max |measured - theory| for t<3: {max_err:.4f}\n")


# ---------------------------------------------------------------------------
# Figure 2: MSD, ballistic -> diffusive crossover
# ---------------------------------------------------------------------------

def figure_2_msd():
    print("Figure 2: mean-squared displacement...")
    params = ABPParameters(
        N=300, L=40.0, sigma=1.0, epsilon=1.0, mobility=1.0,
        v0=8.0, Dt=0.05, Dr=0.5, dt=0.001,
    )
    rng = np.random.default_rng(2)
    state = init_state(params, rng)

    n_steps = 20000
    record_every = 5
    x_hist = [state.x_unwrapped.copy()]
    times = [state.t]

    for i in range(1, n_steps + 1):
        step_euler_maruyama(state, params, rng)
        check_stability(state, step=i)
        if i % record_every == 0:
            x_hist.append(state.x_unwrapped.copy())
            times.append(state.t)

    x_hist = np.array(x_hist)
    times = np.array(times)
    msd = mean_squared_displacement(x_hist)

    fig, ax = plt.subplots(figsize=(6, 4.5))
    mask = times > 0
    ax.loglog(times[mask], msd[mask], lw=2, label="measured MSD")
    # reference slopes
    t_ref = times[mask]
    ax.loglog(t_ref, msd[mask][0] * (t_ref / t_ref[0]) ** 2,
               ":", color="gray", label=r"slope 2 (ballistic)")
    ax.loglog(t_ref, msd[mask][-1] * (t_ref / t_ref[-1]) ** 1,
               "--", color="gray", label=r"slope 1 (diffusive)")
    ax.set_xlabel("t")
    ax.set_ylabel("MSD(t)")
    ax.set_title("Figure 2: MSD (log-log)")
    ax.legend()
    fig.tight_layout()
    fig.savefig(RESULTS_DIR / "figure2_msd.png", dpi=150)
    plt.close(fig)
    print(f"  Pe = {params.Pe:.2f}, final MSD = {msd[-1]:.2f}\n")


# ---------------------------------------------------------------------------
# Figure 3 & 4: g(r) and S(q) for an interacting, non-MIPS system
# ---------------------------------------------------------------------------

def figure_3_4_gr_sq():
    print("Figure 3/4: g(r) and S(q) for an interacting system...")
    params = ABPParameters(
        N=400, L=30.0, sigma=1.0, epsilon=1.0, mobility=1.0,
        v0=5.0, Dt=0.1, Dr=1.0, dt=0.001,
    )
    rng = np.random.default_rng(3)
    state = init_state(params, rng)

    for i in range(15000):
        step_euler_maruyama(state, params, rng)
        check_stability(state, step=i)

    r, g = radial_distribution_function(state.x, params.L, params.sigma)
    q, S = radial_structure_factor(state.x, params.L, q_max_idx=12)

    fig, ax = plt.subplots(figsize=(6, 4.5))
    ax.plot(r, g, lw=2)
    ax.axhline(1.0, color="gray", ls=":")
    ax.axvline(params.sigma, color="red", ls="--", alpha=0.5,
               label=r"$r=\sigma$")
    ax.set_xlabel("r")
    ax.set_ylabel("g(r)")
    ax.set_title("Figure 3: Radial Distribution Function")
    ax.legend()
    fig.tight_layout()
    fig.savefig(RESULTS_DIR / "figure3_rdf.png", dpi=150)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(6, 4.5))
    ax.plot(q, S, "o-", lw=2)
    ax.axhline(1.0, color="gray", ls=":")
    ax.set_xlabel("q")
    ax.set_ylabel("S(q)")
    ax.set_title("Figure 4: Radially-Averaged Structure Factor")
    fig.tight_layout()
    fig.savefig(RESULTS_DIR / "figure4_structure_factor.png", dpi=150)
    plt.close(fig)

    print(f"  phi = {packing_fraction(params.N, params.L, params.sigma):.3f}, "
          f"Pe = {params.Pe:.2f}")
    print(f"  g(r) near r=sigma (contact): {g[np.argmin(np.abs(r - params.sigma))]:.3f}\n")


# ---------------------------------------------------------------------------
# Figure 5: cluster fraction across cutoff choices
# ---------------------------------------------------------------------------

def figure_5_cluster_cutoff_sensitivity():
    print("Figure 5: largest-cluster fraction across cutoff choices...")
    params = ABPParameters(
        N=300, L=25.0, sigma=1.0, epsilon=1.0, mobility=1.0,
        v0=40.0, Dt=0.1, Dr=0.3, dt=0.0002,
    )
    rng = np.random.default_rng(4)
    state = init_state(params, rng)

    cutoffs = [1.2, 1.3, 1.4, 1.5]
    n_steps = 100000
    record_every = 2500
    times = []
    f_max_by_cutoff = {c: [] for c in cutoffs}

    for step in range(n_steps + 1):
        if step % record_every == 0:
            times.append(state.t)
            for c in cutoffs:
                f_max_by_cutoff[c].append(
                    largest_cluster_fraction(state.x, params.L, params.sigma,
                                              cutoff_factor=c)
                )
        if step < n_steps:
            step_euler_maruyama(state, params, rng)
            check_stability(state, step=step)

    fig, ax = plt.subplots(figsize=(6, 4.5))
    for c in cutoffs:
        ax.plot(times, f_max_by_cutoff[c], label=f"cutoff={c}σ", lw=2)
    ax.set_xlabel("t")
    ax.set_ylabel(r"$f_{max}(t)$")
    ax.set_title("Figure 5: Cluster Fraction, Cutoff Sensitivity")
    ax.legend()
    fig.tight_layout()
    fig.savefig(RESULTS_DIR / "figure5_cluster_cutoff_sensitivity.png", dpi=150)
    plt.close(fig)

    phi = packing_fraction(params.N, params.L, params.sigma)
    print(f"  phi = {phi:.3f}, Pe = {params.Pe:.2f}")
    spread = max(f_max_by_cutoff[c][-1] for c in cutoffs) - \
        min(f_max_by_cutoff[c][-1] for c in cutoffs)
    print(f"  final f_max spread across cutoffs: {spread:.3f}")
    print("  (large spread => cluster-onset time is sensitive to this "
          "modeling choice and should not be treated as a fixed target "
          "without stating which cutoff was used)\n")


# ---------------------------------------------------------------------------
# Figure 6: coarse phase scan over (phi, Pe)
# ---------------------------------------------------------------------------

def figure_6_phase_scan():
    print("Figure 6: coarse phase scan over (phi, Pe)...")
    N = 250
    sigma = 1.0
    phi_values = [0.2, 0.4, 0.6]
    Pe_values = [20.0, 60.0, 120.0]
    dt = 0.0001
    n_steps = 25000

    f_max_grid = np.zeros((len(phi_values), len(Pe_values)))

    t0 = time.time()
    for i, phi in enumerate(phi_values):
        L = np.sqrt(N * np.pi * (sigma / 2.0) ** 2 / phi)
        for j, Pe in enumerate(Pe_values):
            Dr = 0.5
            v0 = Pe * Dr * sigma
            params = ABPParameters(
                N=N, L=L, sigma=sigma, epsilon=1.0, mobility=1.0,
                v0=v0, Dt=0.1, Dr=Dr, dt=dt,
            )
            rng = np.random.default_rng(100 + i * 10 + j)
            state = init_state(params, rng)
            try:
                for step in range(n_steps):
                    step_euler_maruyama(state, params, rng)
                    check_stability(state, step=step)
                f_max_grid[i, j] = largest_cluster_fraction(
                    state.x, params.L, params.sigma, cutoff_factor=1.3
                )
                status = "OK"
            except InstabilityDetected as exc:
                f_max_grid[i, j] = np.nan
                status = f"UNSTABLE ({exc})"
            print(f"  phi={phi:.2f}, Pe={Pe:.0f} -> f_max={f_max_grid[i, j]:.3f} "
                  f"[{status}] [{time.time() - t0:.1f}s elapsed]")

    fig, ax = plt.subplots(figsize=(6, 4.5))
    im = ax.imshow(f_max_grid, origin="lower", aspect="auto", cmap="viridis",
                    vmin=0, vmax=1)
    ax.set_xticks(range(len(Pe_values)))
    ax.set_xticklabels([str(p) for p in Pe_values])
    ax.set_yticks(range(len(phi_values)))
    ax.set_yticklabels([str(p) for p in phi_values])
    ax.set_xlabel("Pe")
    ax.set_ylabel(r"$\phi$")
    ax.set_title(r"Figure 6: Phase Scan, final $f_{max}(\phi, Pe)$")
    fig.colorbar(im, ax=ax, label=r"$f_{max}$")
    fig.tight_layout()
    fig.savefig(RESULTS_DIR / "figure6_phase_scan.png", dpi=150)
    plt.close(fig)
    print()


if __name__ == "__main__":
    ideal_gas_ok = sanity_check_ideal_gas()
    if not ideal_gas_ok:
        print("WARNING: ideal-gas sanity check failed. Proceeding anyway "
              "to generate all figures, but results below should not be "
              "trusted until this is resolved.\n")

    figure_1_orientation_autocorrelation()
    figure_2_msd()
    figure_3_4_gr_sq()
    figure_5_cluster_cutoff_sensitivity()
    figure_6_phase_scan()

    print("=" * 70)
    print(f"All figures written to {RESULTS_DIR}")
    print("=" * 70)
