import numpy as np
import h5py
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from rsvp_mips import ExportConfig, export_run
from rsvp_mips.features import compute_field_features, compute_cross_field_features
from rsvp_mips.density_domains import composite_mips_indicators
from rsvp_mips.dissipation import reconstruct_s_current
from rsvp_mips.fields import reconstruct_density


def small_config(tmp_seed=0, **overrides):
    defaults = dict(
        N=80, phi=0.3, Pe=60.0, dt=0.0005,
        burnin_time=0.5, t_run=1.5, record_every_time=0.25,
        grid_n=16, seed=tmp_seed, save_particles=True,
    )
    defaults.update(overrides)
    return ExportConfig(**defaults)


def test_export_completes_and_marks_run_complete(tmp_path):
    out_path = tmp_path / "run.h5"
    config = small_config()
    export_run(config, str(out_path))

    with h5py.File(out_path, "r") as f:
        assert bool(f["run_metadata"].attrs["complete"]) is True
        assert bool(f["run_metadata"].attrs["run_failed"]) is False


def test_all_datasets_have_consistent_frame_count(tmp_path):
    out_path = tmp_path / "run.h5"
    config = small_config()
    export_run(config, str(out_path))

    with h5py.File(out_path, "r") as f:
        n_frames = f["run_metadata"].attrs["n_frames"]
        assert n_frames > 0

        expected_first_dim = {
            "time/abs_time": n_frames,
            "time/rel_time": n_frames,
            "fields/rho": n_frames,
            "fields/current_x": n_frames,
            "fields/capacity/parallel": n_frames,
            "fields/capacity/input": n_frames,
            "fields/dissipation/work": n_frames,
            "fields/dissipation/current": n_frames,
            "features/baseline": n_frames,
            "features/capacity_parallel": n_frames,
            "features/dissipation": n_frames,
            "features/interactions": n_frames,
            "targets/f_dense_max": n_frames,
            "targets/f_void": n_frames,
            "targets/S_rho_low_q": n_frames,
            "targets/B_rho": n_frames,
            "diagnostics/f_contact_max": n_frames,
            "labels/onset": n_frames,
            "particles/x": n_frames,
        }
        for path, expected in expected_first_dim.items():
            actual = f[path].shape[0]
            assert actual == expected, f"{path}: expected {expected}, got {actual}"


def test_grid_shapes_match_config():
    pass  # covered implicitly below; kept as a marker for the intent


def test_field_grid_dimensions_match_grid_n(tmp_path):
    out_path = tmp_path / "run.h5"
    config = small_config(grid_n=20)
    export_run(config, str(out_path))
    with h5py.File(out_path, "r") as f:
        assert f["fields/rho"].shape[1:] == (20, 20)
        assert f["fields/capacity/parallel"].shape[1:] == (20, 20)
        assert f["fields/dissipation/current"].shape[1:] == (20, 20)


def test_rel_time_starts_near_zero_and_abs_time_offset_by_burnin(tmp_path):
    out_path = tmp_path / "run.h5"
    config = small_config(burnin_time=2.0)
    export_run(config, str(out_path))
    with h5py.File(out_path, "r") as f:
        rel_t = f["time/rel_time"][:]
        abs_t = f["time/abs_time"][:]
        assert rel_t[0] == 0.0
        # abs_time should be offset from rel_time by exactly the burn-in
        # duration (t_origin_abs), removing any ambiguity about what t=0
        # means
        t_origin = f["run_metadata"].attrs["t_origin_abs"]
        assert np.allclose(abs_t - rel_t, t_origin)
        assert np.isclose(t_origin, config.burnin_time, atol=config.dt)


def test_recomputed_features_exactly_match_stored_features(tmp_path):
    """
    The core acceptance test: reopen the file, recompute a handful of
    saved features directly from the stored grids using the same
    feature functions the exporter used, and confirm exact (not
    approximate-up-to-a-different-formula) agreement. This tests the
    data contract -- that what's stored in /features is genuinely
    derivable from what's stored in /fields -- not just that the script
    ran without crashing.
    """
    out_path = tmp_path / "run.h5"
    config = small_config()
    export_run(config, str(out_path))

    with h5py.File(out_path, "r") as f:
        L = f["run_metadata"].attrs["L"]
        n_frames = f["run_metadata"].attrs["n_frames"]
        frame_idx = n_frames // 2  # check a middle frame, not just frame 0

        rho_h = f["fields/rho"][frame_idx]
        stored_baseline = dict(zip(
            f["features/baseline"].attrs["columns"],
            f["features/baseline"][frame_idx],
        ))
        recomputed = compute_field_features(rho_h, L, "rho")

        for key, stored_val in stored_baseline.items():
            recomputed_val = recomputed[key]
            if np.isnan(stored_val) and np.isnan(recomputed_val):
                continue
            assert np.isclose(stored_val, recomputed_val, rtol=1e-10), (
                f"{key}: stored={stored_val}, recomputed={recomputed_val}"
            )

        # also recompute a target indicator directly from the stored
        # density grid and confirm it matches /targets
        indicators = composite_mips_indicators(
            rho_h, L,
            rho_star_factor=f["run_metadata"].attrs["rho_star_factor"],
            rho_low_factor=f["run_metadata"].attrs["rho_low_factor"],
        )
        assert np.isclose(indicators["f_dense_max"],
                           f["targets/f_dense_max"][frame_idx], rtol=1e-10)
        assert np.isclose(indicators["f_void"],
                           f["targets/f_void"][frame_idx], rtol=1e-10)
        assert np.isclose(indicators["B_rho"],
                           f["targets/B_rho"][frame_idx], rtol=1e-10)

        # and confirm an interaction feature (cross-field) recomputes
        # exactly from the stored grids too
        phi_par = f["fields/capacity/parallel"][frame_idx]
        cross = compute_cross_field_features(rho_h, phi_par, L, "rho", "phi_parallel")
        stored_inter = dict(zip(
            f["features/interactions"].attrs["columns"],
            f["features/interactions"][frame_idx],
        ))
        assert np.isclose(cross["cov_rho_phi_parallel"],
                           stored_inter["cov_rho_phi_parallel"], rtol=1e-10)


def test_grid_init_mode_available_but_distinct_from_rsa(tmp_path):
    out_path_grid = tmp_path / "run_grid.h5"
    out_path_rsa = tmp_path / "run_rsa.h5"
    export_run(small_config(init_method="grid"), str(out_path_grid))
    export_run(small_config(init_method="rsa"), str(out_path_rsa))

    with h5py.File(out_path_grid, "r") as fg, h5py.File(out_path_rsa, "r") as fr:
        assert fg["run_metadata"].attrs["init_method"] == "grid"
        assert fr["run_metadata"].attrs["init_method"] == "rsa"


def test_incomplete_run_is_marked_as_such(tmp_path):
    """
    A config guaranteed to be numerically unstable (dt far too large for
    this v0) should still produce a file, but with complete=False and a
    recorded failure reason -- not a file that looks like valid finished
    data.
    """
    out_path = tmp_path / "run_unstable.h5"
    config = small_config(N=200, phi=0.5, Pe=130.0, dt=0.002,
                           burnin_time=0.0, t_run=3.0)
    export_run(config, str(out_path))

    with h5py.File(out_path, "r") as f:
        assert bool(f["run_metadata"].attrs["complete"]) is False
        assert bool(f["run_metadata"].attrs["run_failed"]) is True
        assert f["run_metadata"].attrs["failure_reason"] != ""
