"""
Additional packing-fraction points (phi=0.2, phi=0.45) to test whether
the phi=0.35 residualized-correlation pattern (weak but consistent
mean/skew signal in capacity_parallel/capacity_forward/s_work) is
monotonic/systematic across density, or a fluke of that one point.
5 seeds each first (exploratory), extend to 10 if a clear pattern
emerges.
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from rsvp_mips import ExportConfig, export_run  # noqa: E402

OUT_DIR = Path(__file__).resolve().parents[1] / "results" / "phi_sweep_runs"
OUT_DIR.mkdir(parents=True, exist_ok=True)


def main(phi, seeds):
    t0 = time.time()
    for i, seed in enumerate(seeds):
        config = ExportConfig(
            N=200, phi=phi, Pe=130.0, dt=0.0001,
            burnin_time=6.0, t_run=15.0, record_every_time=0.1,
            grid_n=24, kernel_h=1.2, seed=seed, save_particles=False,
            onset_theta_high=0.25, onset_theta_low=0.12, onset_min_dwell=1.0,
        )
        out_path = OUT_DIR / f"run_phi{phi}_seed{seed}.h5"
        export_run(config, str(out_path))
        print(f"[{i+1}/{len(seeds)}] phi={phi} seed={seed} done "
              f"[{time.time()-t0:.0f}s elapsed]")


if __name__ == "__main__":
    phi = float(sys.argv[1])
    seeds = [int(s) for s in sys.argv[2:]]
    main(phi, seeds)
