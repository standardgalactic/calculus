import numpy as np
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from rsvp_mips.identification import (
    RunData,
    build_design_matrix,
    chronological_split,
    fit_delta_M_model,
    run_nested_comparison,
)


def make_synthetic_run(seed=0, n_frames=200, record_interval=0.1,
                        beta=2.0, driver_period=5.0):
    """
    Builds a RunData with a KNOWN embedded relationship: the
    capacity_parallel branch's 'phi_parallel_mean' column is the true
    driver of f_dense_max's increments, at lag = one recording interval.
    Every other feature column is pure noise, unrelated to the target.
    This lets fit_delta_M_model be tested against ground truth rather
    than just "does it run."
    """
    rng = np.random.default_rng(seed)
    rel_time = np.arange(n_frames) * record_interval

    driver = np.sin(2 * np.pi * rel_time / driver_period) + 0.05 * rng.standard_normal(n_frames)
    noise_M = 0.01 * rng.standard_normal(n_frames)
    M = np.concatenate([[0.0], np.cumsum(beta * driver[:-1] * record_interval)]) + noise_M

    targets = {
        "f_dense_max": M,
        "f_void": 0.1 + 0.01 * rng.standard_normal(n_frames),
        "S_rho_low_q": 0.3 + 0.01 * rng.standard_normal(n_frames),
        "B_rho": 0.4 + 0.01 * rng.standard_normal(n_frames),
    }

    feature_tables = {
        "baseline": (
            ["rho_mean", "rho_var", "rho_noise1"],
            rng.standard_normal((n_frames, 3)),
        ),
        "capacity_input": (
            ["phi_input_value"],
            np.full((n_frames, 1), 10.0),  # constant, as in the real trivial control
        ),
        "capacity_parallel": (
            ["phi_parallel_mean", "phi_parallel_var", "phi_parallel_noise"],
            np.column_stack([
                driver,
                rng.standard_normal(n_frames),
                rng.standard_normal(n_frames),
            ]),
        ),
        "capacity_forward": (
            ["phi_forward_mean"],
            rng.standard_normal((n_frames, 1)),
        ),
        "dissipation": (
            ["s_work_mean", "s_current_mean"],
            rng.standard_normal((n_frames, 2)),
        ),
        "interactions": (
            ["cov_rho_phi_parallel", "cov_rho_phi_forward"],
            rng.standard_normal((n_frames, 2)),
        ),
    }

    return RunData(
        path="<synthetic>", rel_time=rel_time, feature_tables=feature_tables,
        targets=targets, diagnostics={}, complete=True, metadata={},
    ), record_interval


# ---------------------------------------------------------------------------
# Design matrix construction
# ---------------------------------------------------------------------------

def test_M0_uses_only_baseline_columns():
    run, _ = make_synthetic_run()
    X, cols = build_design_matrix(run, "M0", "parallel")
    assert cols == ["rho_mean", "rho_var", "rho_noise1"]
    assert X.shape == (200, 3)


def test_M1_adds_branch_capacity_columns():
    run, _ = make_synthetic_run()
    X, cols = build_design_matrix(run, "M1", "parallel")
    assert "phi_parallel_mean" in cols
    assert "phi_forward_mean" not in cols  # only this branch's capacity columns


def test_M2_excludes_other_branch_interaction_columns():
    run, _ = make_synthetic_run()
    X, cols = build_design_matrix(run, "M2", "parallel")
    assert "cov_rho_phi_parallel" in cols
    assert "cov_rho_phi_forward" not in cols  # belongs to the "forward" branch

    X2, cols2 = build_design_matrix(run, "M2", "forward")
    assert "cov_rho_phi_forward" in cols2
    assert "cov_rho_phi_parallel" not in cols2


def test_chronological_split_preserves_order():
    train_idx, test_idx = chronological_split(100, train_fraction=0.7)
    assert len(train_idx) == 70
    assert len(test_idx) == 30
    assert train_idx.max() < test_idx.min()  # no shuffling


# ---------------------------------------------------------------------------
# Ground-truth recovery -- the core validation
# ---------------------------------------------------------------------------

def test_M1_recovers_the_true_driver_and_predicts_well_held_out():
    run, record_interval = make_synthetic_run(beta=2.0)
    result = fit_delta_M_model(run, "M1", "parallel", tau=record_interval,
                                target_name="f_dense_max")
    assert result is not None
    assert "phi_parallel_mean" in result.selected_columns
    # a near-deterministic linear relationship should predict very well
    # even on the held-out chronological test split
    assert result.test_r2 > 0.8


def test_M0_without_the_true_driver_predicts_much_worse():
    """
    M0 has no access to the true driver (it's only in
    capacity_parallel), so its held-out R^2 should be far worse than
    M1's -- confirming the comparison actually distinguishes an
    informative branch from an uninformative one, not just that both
    fit noise equally.
    """
    run, record_interval = make_synthetic_run(beta=2.0)
    result_m0 = fit_delta_M_model(run, "M0", "parallel", tau=record_interval,
                                   target_name="f_dense_max")
    result_m1 = fit_delta_M_model(run, "M1", "parallel", tau=record_interval,
                                   target_name="f_dense_max")
    assert result_m0 is not None and result_m1 is not None
    assert result_m1.test_r2 > result_m0.test_r2 + 0.3


def test_forward_branch_does_not_spuriously_recover_parallel_driver():
    """
    The true driver lives only in capacity_parallel. The "forward"
    branch's M1 (which only sees capacity_forward's unrelated noise
    columns) should NOT achieve the same predictive quality.
    """
    run, record_interval = make_synthetic_run(beta=2.0)
    result_parallel = fit_delta_M_model(run, "M1", "parallel", tau=record_interval,
                                         target_name="f_dense_max")
    result_forward = fit_delta_M_model(run, "M1", "forward", tau=record_interval,
                                        target_name="f_dense_max")
    assert result_parallel.test_r2 > result_forward.test_r2 + 0.3


def test_run_nested_comparison_returns_all_cells():
    run, record_interval = make_synthetic_run()
    results = run_nested_comparison(run, tau=record_interval)
    levels_branches = {(r.model_level, r.branch) for r in results}
    for branch in ("input", "parallel", "forward"):
        for level in ("M0", "M1", "M2"):
            assert (level, branch) in levels_branches


def test_insufficient_data_returns_none_rather_than_unreliable_fit():
    run, record_interval = make_synthetic_run(n_frames=10)
    result = fit_delta_M_model(run, "M1", "parallel", tau=record_interval,
                                target_name="f_dense_max")
    assert result is None
