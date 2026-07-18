"""
First real demonstration of the intervention protocol: a local activity
pulse applied to the primary regime (N=200, phi=0.5, Pe=130), tracking
local density evolution in treatment vs. control.

This is a mechanism demonstration, not yet the full protocol the design
called for (matched regions with similar density but different
predicted "tension"/capacity -- that needs a library of candidate
regions selected from a reconstructed field, which is the natural next
step). This first pass answers a narrower question: does a real,
sizeable activity perturbation produce a measurable, reproducible
divergence in local density between treatment and control at all,
before attempting to correlate that divergence with pre-intervention
capacity readings.
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
    ABPParameters, init_state_random, run_burn_in,
    reconstruct_density, reconstruct_phi_parallel,
)
from rsvp_mips.fields import Grid  # noqa: E402
from rsvp_mips.intervention import LocalActivityPulse, run_counterfactual_pair  # noqa: E402
from rsvp_mips.density_domains import composite_mips_indicators  # noqa: E402

RESULTS_DIR = Path(__file__).resolve().parents[1] / "results"
RESULTS_DIR.mkdir(exist_ok=True)


def recorder_factory(grid, h, center, local_radius):
    """Returns a recorder tracking global f_dense_max plus local density
    within `local_radius` of `center` (the intervention region)."""
    def recorder(state, params):
        rho_h = reconstruct_density(state, params, grid, h)
        indicators = composite_mips_indicators(rho_h, params.L)
        d = state.x - np.asarray(center)
        d -= params.L * np.round(d / params.L)
        dist = np.sqrt(np.sum(d ** 2, axis=1))
        local_count = int(np.sum(dist < local_radius))
        return {
            "f_dense_max": indicators["f_dense_max"],
            "local_particle_count": local_count,
        }
    return recorder


def main():
    N, phi, Pe = 200, 0.5, 130.0
    sigma = 1.0
    L = np.sqrt(N * np.pi * (sigma / 2.0) ** 2 / phi)
    Dr = 0.3
    v0 = Pe * Dr * sigma
    dt = 0.0001
    params = ABPParameters(N=N, L=L, sigma=sigma, epsilon=1.0, mobility=1.0,
                            v0=v0, Dt=0.1, Dr=Dr, dt=dt)

    print(f"Preparing base state: N={N}, phi={phi}, Pe={Pe}, L={L:.2f}")
    rng = np.random.default_rng(0)
    base_state = init_state_random(params, rng, min_distance_factor=1.02)
    t0 = time.time()
    run_burn_in(base_state, params, rng, duration=6.0)
    print(f"Burn-in complete [{time.time()-t0:.0f}s]")

    center = (L / 2.0, L / 2.0)
    local_radius = 3.0
    intervention = LocalActivityPulse(
        center=center, radius=local_radius, amplitude=2.0,
        start_time=0.0, duration=2.0,
    )

    grid = Grid(L=L, n=24)
    recorder = recorder_factory(grid, 1.2, center, local_radius)

    print(f"Running counterfactual pair: amplitude={intervention.amplitude}, "
          f"radius={intervention.radius}, duration=2.0+3.0 (intervention+observation)")
    t0 = time.time()
    result = run_counterfactual_pair(
        base_state, params, seed=42, intervention=intervention,
        duration=5.0, record_every_time=0.1, recorder=recorder,
    )
    print(f"Counterfactual pair complete [{time.time()-t0:.0f}s], "
          f"unstable={result.unstable}")
    if result.unstable:
        print(f"Failure reason: {result.failure_reason}")
        return

    times = result.times
    treat_local = np.array(result.treatment_history["local_particle_count"])
    ctrl_local = np.array(result.control_history["local_particle_count"])
    treat_fdense = np.array(result.treatment_history["f_dense_max"])
    ctrl_fdense = np.array(result.control_history["f_dense_max"])

    print(f"\nLocal particle count at intervention end (t=2.0): "
          f"treatment={treat_local[times<=2.0][-1]}, control={ctrl_local[times<=2.0][-1]}")
    print(f"Local particle count at end of observation (t=5.0): "
          f"treatment={treat_local[-1]}, control={ctrl_local[-1]}")
    print(f"Global f_dense_max at end: treatment={treat_fdense[-1]:.3f}, "
          f"control={ctrl_fdense[-1]:.3f}")

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    ax = axes[0]
    ax.plot(times, treat_local, label="treatment", color="crimson")
    ax.plot(times, ctrl_local, label="control", color="steelblue")
    ax.axvspan(intervention.start_time, intervention.end_time, color="gray", alpha=0.2,
               label="intervention window")
    ax.set_xlabel("t (relative to intervention start)")
    ax.set_ylabel("local particle count (within radius of center)")
    ax.set_title("Local density: treatment vs. control")
    ax.legend()

    ax = axes[1]
    ax.plot(times, treat_fdense, label="treatment", color="crimson")
    ax.plot(times, ctrl_fdense, label="control", color="steelblue")
    ax.axvspan(intervention.start_time, intervention.end_time, color="gray", alpha=0.2)
    ax.set_xlabel("t (relative to intervention start)")
    ax.set_ylabel("global f_dense_max")
    ax.set_title("Global density-domain indicator: treatment vs. control")
    ax.legend()

    fig.suptitle(f"Intervention demo: amplitude={intervention.amplitude}, "
                  f"radius={intervention.radius}")
    fig.tight_layout()
    fig.savefig(RESULTS_DIR / "intervention_demo.png", dpi=150)
    plt.close(fig)
    print(f"\nFigure written to {RESULTS_DIR / 'intervention_demo.png'}")


if __name__ == "__main__":
    main()
