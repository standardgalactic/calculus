"""
Matched-region intervention experiment: the first experiment capable of
deciding whether capacity has causal content beyond density.

Honest scope: ONE base state, ONE seed, a handful of density-matched
pairs from a single snapshot -- a demonstration that the matching and
paired-intervention machinery works and produces interpretable output,
not yet a statistically powered study. A real claim needs multiple base
states/seeds and many more matched pairs (the natural next step,
exactly parallel to how the observational program went from "one run"
to "ten runs across four regimes").
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
    reconstruct_density,
)
from rsvp_mips.fields import Grid  # noqa: E402
from rsvp_mips.intervention import LocalActivityPulse, run_counterfactual_pair  # noqa: E402
from rsvp_mips.density_domains import composite_mips_indicators  # noqa: E402
from rsvp_mips.matched_region import (  # noqa: E402
    find_region_candidates, match_pairs_by_density, compute_response_metrics,
)

RESULTS_DIR = Path(__file__).resolve().parents[1] / "results"
RESULTS_DIR.mkdir(exist_ok=True)


def recorder_factory(grid, h, center, local_radius):
    def recorder(state, params):
        rho_h = reconstruct_density(state, params, grid, h)
        indicators = composite_mips_indicators(rho_h, params.L)
        d = state.x - np.asarray(center)
        d -= params.L * np.round(d / params.L)
        dist = np.sqrt(np.sum(d ** 2, axis=1))
        local_count = int(np.sum(dist < local_radius))
        return {"f_dense_max": indicators["f_dense_max"], "local_count": local_count}
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
    rng = np.random.default_rng(1)
    base_state = init_state_random(params, rng, min_distance_factor=1.02)
    t0 = time.time()
    run_burn_in(base_state, params, rng, duration=6.0)
    print(f"Burn-in complete [{time.time()-t0:.0f}s]")

    grid = Grid(L=L, n=24)
    h = 1.2
    local_radius = 3.0

    n_trial = 6
    trial_coords = (np.arange(n_trial) + 0.5) * (L / n_trial)
    candidate_centers = [(x, y) for x in trial_coords for y in trial_coords]

    print(f"\nEvaluating {len(candidate_centers)} candidate regions...")
    candidates = find_region_candidates(base_state, params, grid, h,
                                         candidate_centers, local_radius)
    for c in candidates:
        print(f"  center={c.center}: density={c.local_density:.3f}, "
              f"capacity={c.local_capacity:+.2f}, polarization={c.local_polarization:.3f}, "
              f"in_dense_domain={c.in_dense_domain}")

    pairs = match_pairs_by_density(candidates, n_bins=4, min_capacity_contrast=5.0)
    print(f"\nFound {len(pairs)} density-matched, capacity-contrasting pairs")
    if not pairs:
        print("No qualifying pairs found -- try more candidates or lower "
              "min_capacity_contrast.")
        return

    pairs = pairs[:4]

    amplitude = 2.0
    duration_pulse = 2.0
    duration_total = 5.0
    record_every_time = 0.1

    all_responses = []
    for pair_i, (low_cap, high_cap) in enumerate(pairs):
        print(f"\n=== Pair {pair_i}: density~{low_cap.local_density:.3f} "
              f"(bin match {high_cap.local_density:.3f}) ===")
        for label, region in [("low_capacity", low_cap), ("high_capacity", high_cap)]:
            intervention = LocalActivityPulse(
                center=region.center, radius=local_radius, amplitude=amplitude,
                start_time=0.0, duration=duration_pulse,
            )
            recorder = recorder_factory(grid, h, region.center, local_radius)
            t0 = time.time()
            result = run_counterfactual_pair(
                base_state, params, seed=100 + pair_i, intervention=intervention,
                duration=duration_total, record_every_time=record_every_time,
                recorder=recorder,
            )
            elapsed = time.time() - t0
            if result.unstable:
                print(f"  {label} (pair {pair_i}): UNSTABLE, skipping")
                continue

            times = result.times
            treat_local = np.array(result.treatment_history["local_count"])
            ctrl_local = np.array(result.control_history["local_count"])
            treat_global = np.array(result.treatment_history["f_dense_max"])
            ctrl_global = np.array(result.control_history["f_dense_max"])

            response = compute_response_metrics(
                times, treat_local, ctrl_local, treat_global, ctrl_global,
                intervention, region_label=f"pair{pair_i}_{label}",
                local_capacity=region.local_capacity,
                local_density=region.local_density,
            )
            all_responses.append(response)
            print(f"  {label}: capacity={region.local_capacity:+.2f} "
                  f"depletion_min={response.depletion_min:.0f} "
                  f"overshoot_max={response.overshoot_max:+.1f} "
                  f"integrated_diff={response.integrated_diff:+.2f} "
                  f"delayed_global_diff={response.delayed_global_diff:+.4f} "
                  f"[{elapsed:.0f}s]")

    print("\n=== Summary: within-pair comparison (high capacity - low capacity) ===")
    for pair_i in range(len(pairs)):
        low = next((r for r in all_responses if r.region_label == f"pair{pair_i}_low_capacity"), None)
        high = next((r for r in all_responses if r.region_label == f"pair{pair_i}_high_capacity"), None)
        if low is None or high is None:
            continue
        print(f"Pair {pair_i}: capacity contrast={high.local_capacity - low.local_capacity:+.2f}, "
              f"density diff={high.local_density - low.local_density:+.4f} | "
              f"overshoot diff={high.overshoot_max - low.overshoot_max:+.2f}, "
              f"integrated_diff diff={high.integrated_diff - low.integrated_diff:+.2f}, "
              f"global_diff diff={high.delayed_global_diff - low.delayed_global_diff:+.4f}")

    with open(RESULTS_DIR / "matched_region_experiment.txt", "w") as f:
        f.write("region_label,local_capacity,local_density,depletion_min,"
                "time_of_min,recovery_time,overshoot_max,integrated_diff,"
                "delayed_global_diff\n")
        for r in all_responses:
            f.write(f"{r.region_label},{r.local_capacity},{r.local_density},"
                     f"{r.depletion_min},{r.time_of_min},{r.recovery_time},"
                     f"{r.overshoot_max},{r.integrated_diff},{r.delayed_global_diff}\n")
    print(f"\nResults written to {RESULTS_DIR / 'matched_region_experiment.txt'}")


if __name__ == "__main__":
    main()
