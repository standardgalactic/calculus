import numpy as np
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from rsvp_mips.identification import (
    RunData,
    residualized_correlation_test,
    leave_one_run_out_r2_per_run,
    permutation_test_delta_r2,
)


def make_multirun(scenario: str, n_runs=5, n_frames=150, record_interval=0.1,
                   seed=0):
    """
    scenario:
      "informative" -- capacity_parallel's mean column is a genuine,
                        run-specific additional driver of the target,
                        independent of baseline.
      "redundant"   -- capacity_parallel's mean column is (almost)
                        perfectly explained by the baseline driver, so
                        residualizing against baseline should leave
                        nothing informative.
      "noise"       -- capacity_parallel is pure noise, unrelated to
                        anything.
    """
    rng = np.random.default_rng(seed)
    runs = []
    for run_i in range(n_runs):
        rel_time = np.arange(n_frames) * record_interval
        baseline_driver = (np.sin(2 * np.pi * rel_time / 5.0 + run_i) +
                            0.05 * rng.standard_normal(n_frames))

        if scenario == "informative":
            extra_driver = 0.7 * np.cos(2 * np.pi * rel_time / 3.3 + run_i * 0.5) \
                + 0.05 * rng.standard_normal(n_frames)
            beta_base, beta_extra = 1.5, 1.5
        elif scenario == "redundant":
            extra_driver = baseline_driver + 0.02 * rng.standard_normal(n_frames)
            beta_base, beta_extra = 1.5, 0.0  # target driven only by baseline
        elif scenario == "noise":
            extra_driver = rng.standard_normal(n_frames)
            beta_base, beta_extra = 1.5, 0.0
        else:
            raise ValueError(scenario)

        noise_M = 0.01 * rng.standard_normal(n_frames)
        increments = (beta_base * baseline_driver + beta_extra * extra_driver)[:-1] * record_interval
        M = np.concatenate([[0.0], np.cumsum(increments)]) + noise_M

        targets = {
            "f_dense_max": M,
            "f_void": 0.1 + 0.01 * rng.standard_normal(n_frames),
            "S_rho_low_q": 0.3 + 0.01 * rng.standard_normal(n_frames),
            "B_rho": 0.4 + 0.01 * rng.standard_normal(n_frames),
        }
        feature_tables = {
            "baseline": (
                ["rho_mean", "rho_var", "rho_driver"],
                np.column_stack([
                    np.full(n_frames, 0.6),  # rho_mean: zero-variance, as in real data
                    rng.standard_normal(n_frames),
                    baseline_driver,
                ]),
            ),
            "capacity_input": (["phi_input_value"], np.full((n_frames, 1), 10.0)),
            "capacity_parallel": (
                ["phi_parallel_mean", "phi_parallel_noise"],
                np.column_stack([extra_driver, rng.standard_normal(n_frames)]),
            ),
            "capacity_forward": (["phi_forward_mean"], rng.standard_normal((n_frames, 1))),
            "dissipation": (
                ["s_work_mean", "s_current_mean"],
                rng.standard_normal((n_frames, 2)),
            ),
            "interactions": (
                ["cov_rho_phi_parallel", "cov_rho_phi_forward"],
                rng.standard_normal((n_frames, 2)),
            ),
        }
        runs.append(RunData(
            path=f"<synthetic-{scenario}-{run_i}>", rel_time=rel_time,
            feature_tables=feature_tables, targets=targets, diagnostics={},
            complete=True, metadata={},
        ))
    return runs, record_interval


# ---------------------------------------------------------------------------
# Residualization test
# ---------------------------------------------------------------------------

def test_residualized_test_detects_genuine_independent_signal():
    runs, tau = make_multirun("informative", seed=1)
    results = residualized_correlation_test(runs, "capacity_parallel", tau=tau)
    driver_result = next(r for r in results if r.column == "phi_parallel_mean")
    assert driver_result.permutation_p_value < 0.05
    assert abs(driver_result.residual_pooled_corr) > 0.2


def test_residualized_test_rejects_redundant_column():
    """
    capacity_parallel's mean column is almost entirely explained by the
    baseline driver (it's a near-copy of it) -- after residualizing
    against baseline, essentially nothing should remain, and the
    permutation test should NOT find significance. This is the exact
    "absorbed by density" scenario the real pilot's null result is
    hypothesized to reflect.
    """
    runs, tau = make_multirun("redundant", seed=2)
    results = residualized_correlation_test(runs, "capacity_parallel", tau=tau)
    driver_result = next(r for r in results if r.column == "phi_parallel_mean")
    assert driver_result.permutation_p_value > 0.05


def test_residualized_test_rejects_pure_noise_column():
    runs, tau = make_multirun("noise", seed=3)
    results = residualized_correlation_test(runs, "capacity_parallel", tau=tau)
    driver_result = next(r for r in results if r.column == "phi_parallel_mean")
    assert driver_result.permutation_p_value > 0.05


def test_residualized_test_permutation_count_is_exact_5_factorial():
    runs, tau = make_multirun("noise", n_runs=5, seed=4)
    results = residualized_correlation_test(runs, "capacity_parallel", tau=tau)
    assert results[0].n_permutations == 120  # 5! exactly enumerated


# ---------------------------------------------------------------------------
# Run-level leave-one-out R^2 and permutation test for Delta R^2
# ---------------------------------------------------------------------------

def test_run_level_r2_has_one_value_per_run():
    runs, tau = make_multirun("informative", seed=5)
    r2_per_run = leave_one_run_out_r2_per_run(runs, ["baseline"], tau=tau)
    assert len(r2_per_run) == 5


def test_permutation_test_detects_genuine_incremental_signal():
    runs, tau = make_multirun("informative", seed=6)
    result = permutation_test_delta_r2(
        runs, baseline_groups=["baseline"], extra_groups=["capacity_parallel"],
        tau=tau,
    )
    assert result.observed_delta_r2 > 0
    assert result.p_value < 0.05
    assert result.n_permutations == 120


def test_permutation_test_null_for_redundant_extra_group():
    """
    The real-pilot-relevant case: extra group is highly correlated with
    the target marginally, but adds nothing once baseline is included,
    because it's redundant with the baseline driver. The permutation
    test should NOT reject the null that the extra group's run-specific
    content doesn't matter.
    """
    runs, tau = make_multirun("redundant", seed=7)
    result = permutation_test_delta_r2(
        runs, baseline_groups=["baseline"], extra_groups=["capacity_parallel"],
        tau=tau,
    )
    assert result.p_value > 0.05


def test_permutation_test_null_for_pure_noise_extra_group():
    runs, tau = make_multirun("noise", seed=8)
    result = permutation_test_delta_r2(
        runs, baseline_groups=["baseline"], extra_groups=["capacity_parallel"],
        tau=tau,
    )
    assert result.p_value > 0.05
