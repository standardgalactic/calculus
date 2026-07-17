"""
Post-export sanity check: NOT sparse regression yet. Confirms the
exported feature time series have sensible variation, finite values, and
no accidental leakage before any identification work is trusted.
"""
from __future__ import annotations

import sys
from pathlib import Path

import h5py
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from rsvp_mips import ExportConfig, export_run  # noqa: E402
from rsvp_mips.diagnostics import check_stability  # noqa: E402
from rsvp_mips import ABPParameters, init_state_random, step_euler_maruyama  # noqa: E402


def verify_stability_first(config: ExportConfig, probe_time: float = 1.0):
    """Quick pre-flight probe before committing to the full export."""
    params = config.to_abp_params()
    rng = np.random.default_rng(config.seed)
    state = init_state_random(params, rng, min_distance_factor=config.min_distance_factor)
    n_probe = int(probe_time / config.dt)
    for step in range(n_probe):
        step_euler_maruyama(state, params, rng)
        check_stability(state, step=step)
    print(f"stability probe OK for {probe_time} time units at dt={config.dt}")


def check_variation_and_finiteness(f: h5py.File):
    print("\n--- Variation and finiteness check ---")
    issues = []

    def check_table(path, columns_attr="columns"):
        ds = f[path]
        cols = list(ds.attrs[columns_attr])
        data = ds[:]
        for i, col in enumerate(cols):
            series = data[:, i]
            n_nan = np.sum(np.isnan(series))
            n_inf = np.sum(np.isinf(series))
            std = np.nanstd(series)
            if n_inf > 0:
                issues.append(f"{path}/{col}: {n_inf} Inf values")
            if n_nan > 0:
                print(f"  NOTE {path}/{col}: {n_nan}/{len(series)} NaN "
                      f"(expected for e.g. xi when correlation fit fails)")
            if std == 0 and col != "phi_input_value":
                issues.append(f"{path}/{col}: zero variance (degenerate feature)")
        return cols, data

    for group in ["features/baseline", "features/capacity_input",
                  "features/capacity_parallel", "features/capacity_forward",
                  "features/dissipation", "features/interactions"]:
        check_table(group)

    for path in ["targets/f_dense_max", "targets/f_void",
                 "targets/S_rho_low_q", "targets/B_rho",
                 "diagnostics/f_contact_max"]:
        series = f[path][:]
        n_nan = np.sum(np.isnan(series))
        n_inf = np.sum(np.isinf(series))
        std = np.nanstd(series)
        if n_inf > 0:
            issues.append(f"{path}: {n_inf} Inf values")
        if n_nan > 0:
            print(f"  NOTE {path}: {n_nan}/{len(series)} NaN")
        if std == 0:
            issues.append(f"{path}: zero variance")
        print(f"  {path}: mean={np.nanmean(series):.4f}, std={std:.4f}, "
              f"range=[{np.nanmin(series):.4f}, {np.nanmax(series):.4f}]")

    if issues:
        print("\n  ISSUES FOUND:")
        for issue in issues:
            print(f"    - {issue}")
    else:
        print("\n  No zero-variance or Inf issues found.")
    return issues


def check_finite_derivatives(f: h5py.File):
    print("\n--- Finite-derivative check (no huge jumps between frames) ---")
    rel_t = f["time/rel_time"][:]
    dt_record = np.diff(rel_t)
    issues = []
    for path in ["targets/f_dense_max", "targets/f_void", "targets/S_rho_low_q"]:
        series = f[path][:]
        d_series = np.diff(series) / dt_record
        max_abs_deriv = np.nanmax(np.abs(d_series))
        print(f"  {path}: max |d/dt| = {max_abs_deriv:.3f}")
        if not np.isfinite(max_abs_deriv):
            issues.append(f"{path}: non-finite derivative")
    return issues


def check_no_leakage(f: h5py.File):
    print("\n--- Leakage check ---")
    # (1) do any /features columns literally duplicate a /targets series?
    # (expected/benign case: rho_S_low_q in baseline features is computed
    # by the SAME function as targets/S_rho_low_q, both from rho_h at the
    # same frame t -- this is a legitimate lagged/autoregressive
    # predictor, not future leakage, since both are evaluated at time t)
    baseline_cols = list(f["features/baseline"].attrs["columns"])
    baseline_data = f["features/baseline"][:]
    slowq_target = f["targets/S_rho_low_q"][:]

    idx = baseline_cols.index("rho_S_low_q")
    identical = np.allclose(baseline_data[:, idx], slowq_target, rtol=1e-10)
    print(f"  features/baseline/rho_S_low_q == targets/S_rho_low_q: {identical} "
          f"(expected True -- same formula, same field, same frame t; this "
          f"is a legitimate contemporaneous predictor, not leakage, since "
          f"labels Y_tau look FORWARD from t while this feature is AT t)")

    # (2) confirm the onset LABEL is not present, in any form, among the
    # feature columns (a feature column should never be computed FROM
    # the label array)
    onset_mask = f["labels/onset"][:]
    leak_found = False
    for group in ["features/baseline", "features/capacity_parallel",
                  "features/dissipation", "features/interactions"]:
        data = f[group][:]
        for col_i in range(data.shape[1]):
            series = data[:, col_i]
            if series.shape == onset_mask.shape:
                # a feature that happens to perfectly correlate with the
                # boolean onset mask would be suspicious (though for a
                # continuous feature, exact equality is the real red
                # flag, not correlation, since correlation is expected
                # and desired)
                if np.array_equal(series.astype(bool), onset_mask):
                    leak_found = True
                    print(f"  WARNING: {group} column {col_i} exactly "
                          f"equals the onset label mask")
    if not leak_found:
        print("  No feature column found to be a verbatim copy of the "
              "onset label. Good.")

    # (3) confirm feature values at frame t don't depend on frame t+1
    # by construction -- structural check: recompute frame t's baseline
    # features using ONLY frame t's stored rho grid, confirm exact match
    # (already covered by test_exporter.py's acceptance test, repeated
    # here as a live sanity check on this specific real run)
    from rsvp_mips.features import compute_field_features
    L = f["run_metadata"].attrs["L"]
    t_check = len(f["time/rel_time"]) // 2
    rho_h = f["fields/rho"][t_check]
    recomputed = compute_field_features(rho_h, L, "rho")
    stored = dict(zip(baseline_cols, baseline_data[t_check]))
    all_match = all(
        (np.isnan(stored[k]) and np.isnan(recomputed[k])) or
        np.isclose(stored[k], recomputed[k], rtol=1e-10)
        for k in stored
    )
    print(f"  Frame {t_check}: recomputed baseline features from stored "
          f"grid match exactly: {all_match}")

    return leak_found


def main():
    config = ExportConfig(
        N=250, phi=0.3, Pe=60.0, dt=0.0002,
        burnin_time=6.0, t_run=15.0, record_every_time=0.1,
        grid_n=24, seed=0, save_particles=False,
    )

    print("Running stability probe before committing to full export...")
    verify_stability_first(config, probe_time=1.0)

    out_path = "/tmp/sanity_check_run.h5"
    print(f"\nRunning full export to {out_path}...")
    export_run(config, out_path)

    with h5py.File(out_path, "r") as f:
        print(f"\nrun complete: {bool(f['run_metadata'].attrs['complete'])}")
        print(f"n_frames: {f['run_metadata'].attrs['n_frames']}")
        print(f"n onset events: {f['labels/onset_definition'].attrs['n_events']}")

        variation_issues = check_variation_and_finiteness(f)
        derivative_issues = check_finite_derivatives(f)
        leak_found = check_no_leakage(f)

    print("\n=== SUMMARY ===")
    print(f"variation/finiteness issues: {len(variation_issues)}")
    print(f"derivative issues: {len(derivative_issues)}")
    print(f"leakage found: {leak_found}")


if __name__ == "__main__":
    main()
