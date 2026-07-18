"""
Coarse-graining sensitivity check: same regime as the identification
pilot (N=200, phi=0.5, Pe=130), same 3 seeds, at two alternate kernel
bandwidths (0.8 and 1.8, vs the pilot's 1.2) -- narrow by design, per
the null-certification plan. Tests whether the clean-null regression
result is stable across a nuisance reconstruction parameter, not a
fluke of one bandwidth choice.
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from rsvp_mips import ExportConfig, export_run  # noqa: E402

OUT_DIR = Path(__file__).resolve().parents[1] / "results" / "coarse_graining_runs"
OUT_DIR.mkdir(parents=True, exist_ok=True)

SEEDS = [0, 1, 2]
KERNEL_H_VALUES = [0.8, 1.8]


def main():
    t0 = time.time()
    total = len(SEEDS) * len(KERNEL_H_VALUES)
    count = 0
    for h in KERNEL_H_VALUES:
        for seed in SEEDS:
            count += 1
            config = ExportConfig(
                N=200, phi=0.5, Pe=130.0, dt=0.0001,
                burnin_time=6.0, t_run=15.0, record_every_time=0.1,
                grid_n=24, kernel_h=h, seed=seed, save_particles=False,
                onset_theta_high=0.25, onset_theta_low=0.12, onset_min_dwell=1.0,
            )
            out_path = OUT_DIR / f"run_h{h}_seed{seed}.h5"
            export_run(config, str(out_path))
            print(f"[{count}/{total}] h={h} seed={seed} done "
                  f"[{time.time()-t0:.0f}s elapsed]")
    print(f"\nAll runs written to {OUT_DIR}")


if __name__ == "__main__":
    main()
