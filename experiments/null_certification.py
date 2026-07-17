"""
Null-certification pass on the real 5-run identification pilot.

Per the collaborator review: turns the "capacity is absorbed by
density" interpretation from an inference (Lasso coefficients happened
to hit zero) into an explicit, run-level-significance-tested claim.
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from rsvp_mips.identification import (  # noqa: E402
    load_run, residualized_correlation_test, permutation_test_delta_r2,
)

RUNS_DIR = Path(__file__).resolve().parents[1] / "results" / "identification_runs"
TAU = 2.0


def main():
    run_paths = sorted(RUNS_DIR.glob("run_seed*.h5"))
    print(f"Loading {len(run_paths)} runs...")
    runs = [load_run(str(p)) for p in run_paths]

    print("\n=== 1. Residualized correlation test (capacity_parallel vs baseline) ===")
    print("Tests: after removing what baseline density features predict about")
    print("each capacity column, does the residual still correlate with the")
    print("target? p-value from exact 5!=120-permutation test.\n")
    for group in ["capacity_parallel", "capacity_forward", "dissipation"]:
        print(f"--- {group} ---")
        results = residualized_correlation_test(runs, group, tau=TAU)
        for r in results:
            sig = "  <-- p<0.05" if r.permutation_p_value < 0.05 else ""
            print(f"  {r.column:25s} residual_corr={r.residual_pooled_corr:+.3f}  "
                  f"p={r.permutation_p_value:.3f}{sig}")

    print("\n=== 2. Run-level permutation test for Delta R^2 (M1 vs M0) ===")
    print("Tests: does the TRUE run-to-run pairing of capacity features with")
    print("their own run's target beat a random reassignment of which run's")
    print("capacity block gets used? p-value from exact 120-permutation test.\n")
    for branch, group in [("parallel", "capacity_parallel"),
                           ("forward", "capacity_forward"),
                           ("input", "capacity_input")]:
        result = permutation_test_delta_r2(
            runs, baseline_groups=["baseline"], extra_groups=[group], tau=TAU,
        )
        sig = "  <-- p<0.05" if result.p_value < 0.05 else ""
        print(f"  M1_{branch:10s} observed_delta_R2={result.observed_delta_r2:+.4f}  "
              f"p={result.p_value:.3f}  "
              f"null_range=[{result.null_delta_r2.min():.4f}, "
              f"{result.null_delta_r2.max():.4f}]{sig}")

    # also test dissipation + interactions together (approximating M2)
    result_m2 = permutation_test_delta_r2(
        runs, baseline_groups=["baseline"],
        extra_groups=["capacity_parallel", "dissipation", "interactions"],
        tau=TAU,
    )
    sig = "  <-- p<0.05" if result_m2.p_value < 0.05 else ""
    print(f"  M2_parallel-like observed_delta_R2={result_m2.observed_delta_r2:+.4f}  "
          f"p={result_m2.p_value:.3f}  "
          f"null_range=[{result_m2.null_delta_r2.min():.4f}, "
          f"{result_m2.null_delta_r2.max():.4f}]{sig}")

    print("\n=== Summary ===")
    print("If residual p-values are large AND permutation p-values are large,")
    print("the null-certification pass supports reading this as a genuine")
    print("(preliminary, single-regime) clean null rather than a Lasso/")
    print("regularization artifact.")


if __name__ == "__main__":
    main()
