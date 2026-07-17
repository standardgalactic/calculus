"""
Runs the M0/M1/M2 nested sparse-identification comparison on a real
exported trajectory.

Honest framing up front: this is ONE run, at ONE seed, ONE parameter
combination. It is a demonstration that the pipeline works end to end
and produces sensible, non-degenerate output -- not a scientific
conclusion about whether any capacity variant carries real predictive
information. That would require many runs across seeds and the
parameter grid, exactly as the essay's own identification criterion
demands (comparability, stability across nuisance parameters, held-out
prediction that survives replication -- not a single lucky fit).
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from rsvp_mips.identification import (  # noqa: E402
    load_run, run_nested_comparison, print_comparison_table,
)


def main():
    run = load_run("/tmp/identification_demo_run.h5")
    print(f"Loaded run: {run.path}")
    print(f"  complete={run.complete}, n_frames={len(run.rel_time)}, "
          f"t_run={run.rel_time[-1]:.1f}")
    print(f"  f_dense_max: mean={run.targets['f_dense_max'].mean():.3f}, "
          f"std={run.targets['f_dense_max'].std():.3f}")

    tau = 1.0  # predict f_dense_max one time unit ahead
    print(f"\nRunning M0/M1/M2 nested comparison, tau={tau}, "
          f"target=f_dense_max, chronological 70/30 split...\n")

    results = run_nested_comparison(run, tau=tau, target_name="f_dense_max")
    print_comparison_table(results)

    print("\nSelected (non-zero coefficient) columns per model:")
    for r in results:
        print(f"  {r.branch}/{r.model_level}: {r.selected_columns}")

    # summarize the key comparison the essay actually cares about: does
    # adding a capacity variant (M1) improve held-out prediction over
    # the baseline (M0)?
    print("\n--- M1 vs M0 held-out R^2 improvement, per branch ---")
    by_key = {(r.branch, r.model_level): r for r in results}
    for branch in ("input", "parallel", "forward"):
        m0 = by_key.get((branch, "M0"))
        m1 = by_key.get((branch, "M1"))
        m2 = by_key.get((branch, "M2"))
        if m0 and m1:
            print(f"  {branch}: M0 test_R2={m0.test_r2:.4f}, "
                  f"M1 test_R2={m1.test_r2:.4f}, "
                  f"delta={m1.test_r2 - m0.test_r2:+.4f}"
                  + (f", M2 test_R2={m2.test_r2:.4f}" if m2 else ""))


if __name__ == "__main__":
    main()
