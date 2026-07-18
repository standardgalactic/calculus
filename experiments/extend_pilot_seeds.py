"""
Extends the primary identification pilot (N=200, phi=0.5, Pe=130,
kernel_h=1.2) from 5 to 10 seeds, per the last flagged next step in the
null-certification pass ("a larger seed count"). Run in small batches
(this script takes a seed range as CLI args) to stay safely under the
tool execution timeout that corrupted one export in the earlier
coarse-graining batch.
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from rsvp_mips import ExportConfig, export_run  # noqa: E402

OUT_DIR = Path(__file__).resolve().parents[1] / "results" / "identification_runs"
OUT_DIR.mkdir(parents=True, exist_ok=True)


def main(seeds):
    t0 = time.time()
    for i, seed in enumerate(seeds):
        config = ExportConfig(
            N=200, phi=0.5, Pe=130.0, dt=0.0001,
            burnin_time=6.0, t_run=15.0, record_every_time=0.1,
            grid_n=24, kernel_h=1.2, seed=seed, save_particles=False,
            onset_theta_high=0.25, onset_theta_low=0.12, onset_min_dwell=1.0,
        )
        out_path = OUT_DIR / f"run_seed{seed}.h5"
        export_run(config, str(out_path))
        print(f"[{i+1}/{len(seeds)}] seed={seed} done "
              f"[{time.time()-t0:.0f}s elapsed]")


if __name__ == "__main__":
    seeds = [int(s) for s in sys.argv[1:]]
    main(seeds)
