"""
Sparse identification stage: M0/M1/M2 nested model comparison across the
three capacity-variant branches (Phi_input, Phi_parallel, Phi_forward).

This is "sparse prospective time-series identification over spatial
field functionals," per the design record -- NOT literal spatial
PDE-SINDy (that remains a shelved auxiliary project). The target is
Delta_tau M(t) (continuous, sparse regression), built from the primary
composite indicator f_dense_max by default, using the aggregated scalar
features already stored in an exported HDF5 run record as the candidate
library Theta.

Nested models per branch:
  M0: baseline (density-only) features
  M1: M0 + this branch's capacity features
  M2: M1 + dissipation features + this branch's relevant interaction
      features (interaction columns naming the OTHER branch's capacity
      field are excluded, so a branch's M2 isn't silently informed by a
      competing capacity variant)

Evaluation is CHRONOLOGICAL (train on the first portion of the
trajectory, test on a later held-out portion) -- not random k-fold CV
across the whole series. Random CV would let autocorrelated future
frames leak into training folds, exactly the kind of leakage the
essay's own identification criterion (Section 2: "improves prediction on
data withheld from the fitting process") rules out. The inner
cross-validation used to select the Lasso regularization strength also
uses TimeSeriesSplit rather than ordinary KFold, for the same reason,
applied only within the training portion.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import h5py
import numpy as np
from sklearn.linear_model import LassoCV
from sklearn.model_selection import TimeSeriesSplit
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score

from .onset import delta_M

FEATURE_GROUPS = ["baseline", "capacity_input", "capacity_parallel",
                   "capacity_forward", "dissipation", "interactions"]

BRANCH_CAPACITY_GROUP = {
    "input": "capacity_input",
    "parallel": "capacity_parallel",
    "forward": "capacity_forward",
}

# Which OTHER branch's field name to exclude from a given branch's
# interaction columns, so e.g. the "forward" branch's M2 doesn't get
# silently informed by phi_parallel-named cross terms.
_EXCLUDE_OTHER_BRANCH_NAME = {
    "input": "phi_parallel",  # input has no interaction columns of its own anyway
    "parallel": "phi_forward",
    "forward": "phi_parallel",
}


@dataclass
class RunData:
    path: str
    rel_time: np.ndarray
    feature_tables: dict  # group_name -> (columns: list[str], data: np.ndarray)
    targets: dict          # name -> array
    diagnostics: dict
    complete: bool
    metadata: dict


def load_run(path: str) -> RunData:
    with h5py.File(path, "r") as f:
        rel_time = f["time/rel_time"][:]
        feature_tables = {}
        for group in FEATURE_GROUPS:
            ds = f[f"features/{group}"]
            feature_tables[group] = (list(ds.attrs["columns"]), ds[:])
        targets = {
            "f_dense_max": f["targets/f_dense_max"][:],
            "f_void": f["targets/f_void"][:],
            "S_rho_low_q": f["targets/S_rho_low_q"][:],
            "B_rho": f["targets/B_rho"][:],
        }
        diagnostics = {"f_contact_max": f["diagnostics/f_contact_max"][:]}
        complete = bool(f["run_metadata"].attrs["complete"])
        metadata = dict(f["run_metadata"].attrs)
    return RunData(path=path, rel_time=rel_time, feature_tables=feature_tables,
                   targets=targets, diagnostics=diagnostics, complete=complete,
                   metadata=metadata)


def _branch_interaction_mask(interaction_cols: list, branch: str) -> list:
    excl = _EXCLUDE_OTHER_BRANCH_NAME.get(branch)
    if excl is None:
        return [True] * len(interaction_cols)
    return [excl not in c for c in interaction_cols]


def build_design_matrix(run: RunData, model_level: str, branch: str
                         ) -> tuple[np.ndarray, list]:
    """model_level in {'M0','M1','M2'}; branch in {'input','parallel','forward'}."""
    if model_level not in ("M0", "M1", "M2"):
        raise ValueError("model_level must be 'M0', 'M1', or 'M2'")
    if branch not in BRANCH_CAPACITY_GROUP:
        raise ValueError("branch must be 'input', 'parallel', or 'forward'")

    base_cols, base_data = run.feature_tables["baseline"]
    columns = list(base_cols)
    blocks = [base_data]

    if model_level in ("M1", "M2"):
        cap_group = BRANCH_CAPACITY_GROUP[branch]
        cap_cols, cap_data = run.feature_tables[cap_group]
        columns += cap_cols
        blocks.append(cap_data)

    if model_level == "M2":
        diss_cols, diss_data = run.feature_tables["dissipation"]
        columns += diss_cols
        blocks.append(diss_data)

        inter_cols, inter_data = run.feature_tables["interactions"]
        mask = _branch_interaction_mask(inter_cols, branch)
        keep_idx = [i for i, keep in enumerate(mask) if keep]
        if keep_idx:
            columns += [inter_cols[i] for i in keep_idx]
            blocks.append(inter_data[:, keep_idx])

    X = np.hstack(blocks) if len(blocks) > 1 else np.asarray(blocks[0])
    return X, columns


def chronological_split(n: int, train_fraction: float = 0.7
                         ) -> tuple[np.ndarray, np.ndarray]:
    n_train = int(n * train_fraction)
    return np.arange(0, n_train), np.arange(n_train, n)


@dataclass
class FitResult:
    model_level: str
    branch: str
    tau: float
    target_name: str
    columns: list
    selected_columns: list
    coefficients: np.ndarray
    train_r2: float
    test_r2: float
    n_train: int
    n_test: int
    alpha: float


def fit_delta_M_model(run: RunData, model_level: str, branch: str, tau: float,
                       target_name: str = "f_dense_max",
                       train_fraction: float = 0.7,
                       cv_splits: int = 5) -> Optional[FitResult]:
    """
    Fits Delta_tau M(t) = f(Theta(t)) via LassoCV with a chronological
    train/test split. Returns None if there isn't enough data for a
    meaningful split (rather than fitting on too few points and
    reporting a misleadingly precise-looking R^2).
    """
    X_full, columns = build_design_matrix(run, model_level, branch)
    valid_times, delta = delta_M(run.rel_time, run.targets[target_name], tau)
    if len(valid_times) < 20:
        return None

    time_to_idx = {t: i for i, t in enumerate(run.rel_time)}
    row_idx = np.array([time_to_idx[t] for t in valid_times])
    X = X_full[row_idx]
    y = delta

    finite_rows = np.all(np.isfinite(X), axis=1) & np.isfinite(y)
    X, y = X[finite_rows], y[finite_rows]
    n = len(y)
    if n < 20:
        return None

    train_idx, test_idx = chronological_split(n, train_fraction)
    if len(train_idx) < 10 or len(test_idx) < 5:
        return None

    X_train, X_test = X[train_idx], X[test_idx]
    y_train, y_test = y[train_idx], y[test_idx]

    col_std = X_train.std(axis=0)
    keep = col_std > 1e-12
    if not np.any(keep):
        return None
    X_train, X_test = X_train[:, keep], X_test[:, keep]
    kept_columns = [c for c, k in zip(columns, keep) if k]

    scaler = StandardScaler().fit(X_train)
    X_train_s = scaler.transform(X_train)
    X_test_s = scaler.transform(X_test)

    n_splits = min(cv_splits, max(2, len(train_idx) // 10))
    tscv = TimeSeriesSplit(n_splits=n_splits)
    model = LassoCV(cv=tscv, max_iter=100000, alphas=50, tol=1e-6).fit(X_train_s, y_train)

    y_train_pred = model.predict(X_train_s)
    y_test_pred = model.predict(X_test_s)

    selected = [c for c, coef in zip(kept_columns, model.coef_) if abs(coef) > 1e-10]

    return FitResult(
        model_level=model_level, branch=branch, tau=tau, target_name=target_name,
        columns=kept_columns, selected_columns=selected,
        coefficients=model.coef_,
        train_r2=float(r2_score(y_train, y_train_pred)),
        test_r2=float(r2_score(y_test, y_test_pred)),
        n_train=len(train_idx), n_test=len(test_idx), alpha=float(model.alpha_),
    )


def run_nested_comparison(run: RunData, tau: float,
                           target_name: str = "f_dense_max",
                           branches: tuple = ("input", "parallel", "forward"),
                           train_fraction: float = 0.7) -> list:
    """
    Fits M0/M1/M2 for each branch and returns a flat list of FitResult
    (or skips a cell if fit_delta_M_model returns None for insufficient
    data). M0 is branch-independent but is fit once per branch here for
    a uniform table -- the fitted M0 model is identical across branches
    since it only uses baseline features; this is deliberate redundancy
    for reporting simplicity, not a bug.
    """
    results = []
    for branch in branches:
        for level in ("M0", "M1", "M2"):
            r = fit_delta_M_model(run, level, branch, tau,
                                   target_name=target_name,
                                   train_fraction=train_fraction)
            if r is not None:
                results.append(r)
    return results


def print_comparison_table(results: list) -> None:
    print(f"{'branch':10s} {'level':4s} {'n_train':8s} {'n_test':7s} "
          f"{'train_R2':9s} {'test_R2':9s} {'n_selected':10s} {'alpha':10s}")
    for r in results:
        print(f"{r.branch:10s} {r.model_level:4s} {r.n_train:<8d} {r.n_test:<7d} "
              f"{r.train_r2:9.4f} {r.test_r2:9.4f} {len(r.selected_columns):<10d} "
              f"{r.alpha:10.4g}")


# ---------------------------------------------------------------------------
# Null-certification pass: residualization, run-level uncertainty, and a
# run-level permutation test for Delta R^2. Frames within a run are
# strongly autocorrelated, so the effective independent sample size for
# any significance claim is the number of RUNS, not the number of
# frames -- everything below respects that.
# ---------------------------------------------------------------------------

def _extract_run_blocks(run: RunData, feature_groups: list, target_name: str,
                         tau: float) -> tuple:
    """
    Returns (X, y, row_idx) for one run: X assembled by horizontally
    stacking the named feature groups (each group is a (T, n_cols)
    array from run.feature_tables), y = Delta_tau target, row_idx = the
    original frame indices the valid (t, t+tau) pairs correspond to.
    """
    valid_times, delta = delta_M(run.rel_time, run.targets[target_name], tau)
    time_to_idx = {t: i for i, t in enumerate(run.rel_time)}
    row_idx = np.array([time_to_idx[t] for t in valid_times])

    blocks = []
    for g in feature_groups:
        cols, data = run.feature_tables[g]
        blocks.append(data[row_idx])
    X = np.hstack(blocks) if len(blocks) > 1 else blocks[0]
    return X, delta, row_idx


def residualize_against_baseline(baseline_X: np.ndarray, feature_col: np.ndarray,
                                  alphas=None) -> np.ndarray:
    """
    Fits RidgeCV(baseline_X -> feature_col) and returns the residual
    (feature_col minus its baseline-predictable part). This is a
    decorrelation step, not a predictive claim -- fit in-sample on the
    pooled data is appropriate here, since the question is "does this
    column carry information the baseline doesn't already have,"
    independent of any train/test split.
    """
    from sklearn.linear_model import RidgeCV
    if alphas is None:
        alphas = np.logspace(-3, 3, 20)
    model = RidgeCV(alphas=alphas).fit(baseline_X, feature_col)
    return feature_col - model.predict(baseline_X)


@dataclass
class ResidualTestResult:
    column: str
    residual_pooled_corr: float
    permutation_p_value: float
    n_permutations: int


def residualized_correlation_test(runs: list, feature_group: str,
                                   target_name: str = "f_dense_max",
                                   tau: float = 2.0,
                                   max_permutations: Optional[int] = None,
                                   rng: Optional[np.random.Generator] = None
                                   ) -> list:
    """
    For each column in `feature_group`, residualizes it against that
    run's own baseline features (pooled fit across all runs, since this
    is decorrelation not prediction), pools the residuals and targets
    across runs (each standardized per-run first, to remove any trivial
    run-level offset), and tests whether the residual correlates with
    the target using a run-level permutation test.

    With n_runs! permutations feasible (n_runs <= 7 or so), enumerates
    them EXACTLY. For larger n_runs, exact enumeration is intractable
    (10! = 3.6M), so max_permutations random permutations are sampled
    instead -- pass max_permutations explicitly for any n_runs > 7, or
    it defaults to exact enumeration and will be slow/impractical for
    large n_runs.

    This makes the "capacity is absorbed by density" interpretation
    explicit rather than inferred solely from which Lasso coefficients
    happened to hit exactly zero.
    """
    from itertools import permutations as _permutations

    n_runs = len(runs)
    base_blocks, feat_blocks, target_blocks = [], [], []
    col_names = None
    for run in runs:
        X_base, y, row_idx = _extract_run_blocks(run, ["baseline"], target_name, tau)
        cols, data = run.feature_tables[feature_group]
        col_names = cols
        X_feat = data[row_idx]
        base_blocks.append(X_base)
        feat_blocks.append(X_feat)
        target_blocks.append(y)

    X_base_pooled = np.vstack(base_blocks)
    X_feat_pooled = np.vstack(feat_blocks)

    import math
    exact_count = math.factorial(n_runs)
    use_exact = max_permutations is None or exact_count <= max_permutations
    if rng is None:
        rng = np.random.default_rng(0)

    if use_exact:
        perm_indices = list(_permutations(range(n_runs)))
    else:
        seen = set()
        perm_indices = []
        identity = tuple(range(n_runs))
        while len(perm_indices) < max_permutations:
            p = tuple(rng.permutation(n_runs).tolist())
            if p in seen:
                continue
            seen.add(p)
            perm_indices.append(p)
        if identity not in seen:
            perm_indices[0] = identity  # ensure the true pairing is included

    results = []
    for col_i, col in enumerate(col_names):
        feature_col = X_feat_pooled[:, col_i]
        finite = np.isfinite(feature_col) & np.isfinite(X_base_pooled).all(axis=1)
        if finite.sum() < 20:
            continue
        residual_pooled = np.full(len(feature_col), np.nan)
        residual_pooled[finite] = residualize_against_baseline(
            X_base_pooled[finite], feature_col[finite]
        )

        # split residual and target back into per-run blocks for
        # run-level (block) permutation
        residual_per_run, target_per_run = [], []
        offset = 0
        for y in target_blocks:
            n = len(y)
            residual_per_run.append(residual_pooled[offset:offset + n])
            target_per_run.append(y)
            offset += n

        def pooled_corr(res_blocks, tgt_blocks):
            r = np.concatenate(res_blocks)
            t = np.concatenate(tgt_blocks)
            m = np.isfinite(r) & np.isfinite(t)
            if m.sum() < 10 or np.std(r[m]) == 0 or np.std(t[m]) == 0:
                return 0.0
            return float(np.corrcoef(r[m], t[m])[0, 1])

        observed = pooled_corr(residual_per_run, target_per_run)

        null_corrs = []
        for perm in perm_indices:
            permuted_residuals = [residual_per_run[i] for i in perm]
            null_corrs.append(pooled_corr(permuted_residuals, target_per_run))
        null_corrs = np.array(null_corrs)
        p_value = float(np.mean(np.abs(null_corrs) >= abs(observed)))

        results.append(ResidualTestResult(
            column=col, residual_pooled_corr=observed,
            permutation_p_value=p_value, n_permutations=len(perm_indices),
        ))
    return results


def leave_one_run_out_r2_per_run(runs: list, feature_groups: list,
                                  target_name: str = "f_dense_max",
                                  tau: float = 2.0,
                                  feature_block_override: Optional[dict] = None,
                                  override_groups: Optional[list] = None
                                  ) -> np.ndarray:
    """
    Leave-one-run-out evaluation, returning ONE R^2 per held-out run
    (not pooled across runs) -- the effective independent sample size
    for uncertainty is the number of runs, not the number of frames.

    feature_block_override / override_groups: if given, the columns in
    `override_groups` are replaced, for run i, with run
    feature_block_override[i]'s own values instead of run i's own
    (used to implement the block-permutation test below; all other
    groups are left as run i's true data).
    """
    from sklearn.linear_model import Lasso

    n_runs = len(runs)
    Xs, ys = [], []
    for i, run in enumerate(runs):
        if feature_block_override is not None and override_groups is not None:
            base_groups = [g for g in feature_groups if g not in override_groups]
            override_run = runs[feature_block_override[i]]
            X_override, _, _ = _extract_run_blocks(
                override_run, override_groups, target_name, tau
            )
            if base_groups:
                X_base_true, y_true, _ = _extract_run_blocks(
                    run, base_groups, target_name, tau
                )
                X = np.hstack([X_base_true, X_override])
                y = y_true
            else:
                # no baseline groups requested -- target still comes
                # from this run's OWN series, just via a group that
                # happens to be entirely in override_groups
                _, y, _ = _extract_run_blocks(run, ["baseline"], target_name, tau)
                X = X_override
        else:
            X, y, _ = _extract_run_blocks(run, feature_groups, target_name, tau)
        Xs.append(X)
        ys.append(y)

    per_run_r2 = np.full(n_runs, np.nan)
    for test_i in range(n_runs):
        train_idx = [i for i in range(n_runs) if i != test_i]
        X_train = np.vstack([Xs[i] for i in train_idx])
        y_train = np.concatenate([ys[i] for i in train_idx])
        X_test, y_test = Xs[test_i], ys[test_i]

        finite_train = np.all(np.isfinite(X_train), axis=1) & np.isfinite(y_train)
        finite_test = np.all(np.isfinite(X_test), axis=1) & np.isfinite(y_test)
        if finite_train.sum() < 10 or finite_test.sum() < 3:
            continue
        X_train, y_train = X_train[finite_train], y_train[finite_train]
        X_test, y_test = X_test[finite_test], y_test[finite_test]

        col_std = X_train.std(axis=0)
        keep = col_std > 1e-12
        if not np.any(keep):
            per_run_r2[test_i] = 0.0
            continue
        X_train, X_test = X_train[:, keep], X_test[:, keep]

        scaler = StandardScaler().fit(X_train)
        X_train_s, X_test_s = scaler.transform(X_train), scaler.transform(X_test)

        model = Lasso(alpha=0.01, max_iter=20000).fit(X_train_s, y_train)
        pred = model.predict(X_test_s)
        per_run_r2[test_i] = r2_score(y_test, pred)

    return per_run_r2


@dataclass
class PermutationTestResult:
    observed_delta_r2: float
    null_delta_r2: np.ndarray
    p_value: float
    n_permutations: int


def permutation_test_delta_r2(runs: list, baseline_groups: list, extra_groups: list,
                               target_name: str = "f_dense_max", tau: float = 2.0,
                               max_permutations: Optional[int] = None,
                               rng: Optional[np.random.Generator] = None
                               ) -> PermutationTestResult:
    """
    Tests whether the TRUE pairing of each run's extra_groups features
    (e.g. capacity_parallel) with that same run's baseline+target
    produces a better held-out Delta R^2 than assigning a DIFFERENT
    run's extra_groups block to it -- i.e., whether the extra features'
    run-specific content matters, beyond whatever generic structure
    they share with density across runs.

    With n_runs! permutations feasible (n_runs <= 7 or so), enumerates
    them EXACTLY (120 for 5 runs). For larger n_runs, exact enumeration
    is intractable AND each permutation requires refitting n_runs Lasso
    models, so this is expensive even with sampling -- pass
    max_permutations explicitly (e.g. 100-200) for any n_runs > 7.
    """
    from itertools import permutations as _permutations
    import math

    n_runs = len(runs)
    m0_per_run = leave_one_run_out_r2_per_run(runs, baseline_groups, target_name, tau)
    r2_m0 = float(np.nanmean(m0_per_run))

    all_groups = baseline_groups + extra_groups
    m1_true_per_run = leave_one_run_out_r2_per_run(runs, all_groups, target_name, tau)
    r2_m1_true = float(np.nanmean(m1_true_per_run))
    observed_delta = r2_m1_true - r2_m0

    exact_count = math.factorial(n_runs)
    use_exact = max_permutations is None or exact_count <= max_permutations
    if rng is None:
        rng = np.random.default_rng(0)

    if use_exact:
        perms = list(_permutations(range(n_runs)))
    else:
        seen = set()
        perms = []
        identity = tuple(range(n_runs))
        while len(perms) < max_permutations:
            p = tuple(rng.permutation(n_runs).tolist())
            if p in seen:
                continue
            seen.add(p)
            perms.append(p)
        if identity not in seen:
            perms[0] = identity

    null_deltas = []
    for perm in perms:
        override = {i: perm[i] for i in range(n_runs)}
        m1_perm_per_run = leave_one_run_out_r2_per_run(
            runs, all_groups, target_name, tau,
            feature_block_override=override, override_groups=extra_groups,
        )
        r2_m1_perm = float(np.nanmean(m1_perm_per_run))
        null_deltas.append(r2_m1_perm - r2_m0)

    null_deltas = np.array(null_deltas)
    p_value = float(np.mean(null_deltas >= observed_delta))

    return PermutationTestResult(
        observed_delta_r2=observed_delta, null_delta_r2=null_deltas,
        p_value=p_value, n_permutations=len(perms),
    )


# ---------------------------------------------------------------------------
# Prospective classification: Y_tau(t) (onset-within-window), evaluated via
# ROC-AUC / PR-AUC / Brier score, per the essay's Section 5 design. The
# regression side (Delta_tau M) has been extensively null-certified; this
# is the corresponding check for the classification framing, which the
# original 5-run pilot found underpowered and never revisited.
# ---------------------------------------------------------------------------

@dataclass
class ClassificationResult:
    branch: str
    model_level: str
    n_pos: int
    n_total: int
    roc_auc: float
    pr_auc: float
    brier: float
    n_runs_used: int


def _build_classification_xy(run: RunData, feature_groups: list, tau: float,
                              branch: Optional[str] = None,
                              onset_series_name: str = "f_dense_max"
                              ) -> tuple:
    """
    Builds (X, y) for prospective classification: X = feature vector at
    time t, y = Y_tau(t) = 1{onset starts within (t, t+tau]}, using this
    run's OWN stored onset labels (the exporter already computed these
    from f_dense_max with persistence filtering -- reused here rather
    than recomputed, so classification and the exported labels can never
    silently drift apart).

    If "interactions" is in feature_groups and branch is given, applies
    the same other-branch-column exclusion used in build_design_matrix
    for regression (e.g. the "forward" branch never sees
    phi_parallel-named interaction columns), so the two evaluation paths
    stay consistent rather than silently diverging.
    """
    from .onset import detect_onset_events, prospective_labels

    events = detect_onset_events(
        run.rel_time, run.targets[onset_series_name],
        theta_high=0.25, theta_low=0.12, min_dwell=1.0,
    )
    Y = prospective_labels(run.rel_time, events, tau=tau)

    blocks = []
    for g in feature_groups:
        cols, data = run.feature_tables[g]
        if g == "interactions" and branch is not None:
            mask = _branch_interaction_mask(cols, branch)
            keep_idx = [i for i, keep in enumerate(mask) if keep]
            data = data[:, keep_idx] if keep_idx else data[:, :0]
        blocks.append(data)
    X = np.hstack(blocks) if len(blocks) > 1 else blocks[0]
    return X, Y


def leave_one_run_out_classification(runs: list, feature_groups: list,
                                      branch: str, model_level: str,
                                      tau: float = 2.0) -> Optional[ClassificationResult]:
    """
    Leave-one-run-out prospective classification, pooling held-out
    predictions across all runs before computing ROC-AUC/PR-AUC/Brier
    (rather than averaging per-run AUCs, since with only a handful of
    positive events per run, a per-run AUC is often undefined or wildly
    unstable -- pooling first is standard practice for rare-event
    classification with few runs).

    Returns None if there are too few positive examples overall for the
    metrics to be meaningful (fewer than 5 positive windows pooled
    across all held-out runs), rather than reporting a number computed
    from 1-2 positive examples.
    """
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import roc_auc_score, average_precision_score, brier_score_loss

    n_runs = len(runs)
    Xs, ys = [], []
    for run in runs:
        X, y = _build_classification_xy(run, feature_groups, tau, branch=branch)
        Xs.append(X)
        ys.append(y.astype(int))

    all_y_true, all_y_score = [], []
    for test_i in range(n_runs):
        train_idx = [i for i in range(n_runs) if i != test_i]
        X_train = np.vstack([Xs[i] for i in train_idx])
        y_train = np.concatenate([ys[i] for i in train_idx])
        X_test, y_test = Xs[test_i], ys[test_i]

        finite_train = np.all(np.isfinite(X_train), axis=1)
        finite_test = np.all(np.isfinite(X_test), axis=1)
        X_train, y_train = X_train[finite_train], y_train[finite_train]
        X_test, y_test = X_test[finite_test], y_test[finite_test]

        if len(np.unique(y_train)) < 2 or len(X_test) == 0:
            continue

        col_std = X_train.std(axis=0)
        keep = col_std > 1e-12
        if not np.any(keep):
            continue
        X_train, X_test = X_train[:, keep], X_test[:, keep]

        scaler = StandardScaler().fit(X_train)
        X_train_s, X_test_s = scaler.transform(X_train), scaler.transform(X_test)

        model = LogisticRegression(
            penalty="l1", solver="liblinear", C=1.0, max_iter=5000,
            class_weight="balanced",
        ).fit(X_train_s, y_train)
        scores = model.predict_proba(X_test_s)[:, 1]

        all_y_true.append(y_test)
        all_y_score.append(scores)

    if not all_y_true:
        return None
    y_true = np.concatenate(all_y_true)
    y_score = np.concatenate(all_y_score)

    n_pos = int(y_true.sum())
    if n_pos < 5 or n_pos == len(y_true):
        return None

    return ClassificationResult(
        branch=branch, model_level=model_level, n_pos=n_pos,
        n_total=len(y_true),
        roc_auc=float(roc_auc_score(y_true, y_score)),
        pr_auc=float(average_precision_score(y_true, y_score)),
        brier=float(brier_score_loss(y_true, y_score)),
        n_runs_used=n_runs,
    )


def run_classification_comparison(runs: list, tau: float = 2.0,
                                   branches: tuple = ("input", "parallel", "forward")
                                   ) -> list:
    """Classification analogue of run_nested_comparison."""
    results = []
    for branch in branches:
        for level in ("M0", "M1", "M2"):
            base_cols, _ = runs[0].feature_tables["baseline"]
            groups = ["baseline"]
            if level in ("M1", "M2"):
                groups.append(BRANCH_CAPACITY_GROUP[branch])
            if level == "M2":
                groups.append("dissipation")
                groups.append("interactions")  # branch-filtered internally
            r = leave_one_run_out_classification(runs, groups, branch, level, tau=tau)
            if r is not None:
                results.append(r)
    return results


def print_classification_table(results: list) -> None:
    print(f"{'branch':10s} {'level':4s} {'n_pos':6s} {'n_total':8s} "
          f"{'ROC_AUC':8s} {'PR_AUC':8s} {'Brier':8s}")
    for r in results:
        print(f"{r.branch:10s} {r.model_level:4s} {r.n_pos:<6d} {r.n_total:<8d} "
              f"{r.roc_auc:8.4f} {r.pr_auc:8.4f} {r.brier:8.4f}")
