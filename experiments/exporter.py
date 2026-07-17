"""
Trajectory exporter.

Ties simulation -> preparation -> burn-in -> reconstruction -> features
-> targets -> labels into one canonical HDF5 run record per run, per the
finalized identification data contract:

  - One physical dataset of shared fields (density, current, velocity,
    dissipation, capacity fields), stored once -- not duplicated per
    capacity-variant branch.
  - Separate /features/* groups for baseline, capacity_input,
    capacity_parallel, capacity_forward, dissipation, and interactions,
    so the three-branch analysis (Phi_input / Phi_parallel / Phi_plus)
    doesn't require three duplicated field datasets or risk branch-
    specific preprocessing drift.
  - Both full grids (for later feature-library revision without
    rerunning the simulation) AND derived scalar features (for the
    current identification pass) are saved.
  - The composite target M(t) = (f_dense_max, f_void, S_rho_low_q,
    B_rho) is stored as four independently interpretable series, not
    collapsed into one scalar. The old contact-based cluster fraction is
    stored ONLY under /diagnostics/f_contact_max, clearly separated from
    /targets/, so it cannot be accidentally reused as the primary target.
  - Preparation parameters (init_method, min_distance_factor,
    burnin_time, seed, preparation_version) and reconstruction
    provenance (kernel bandwidth, grid dimensions, dt, sampling
    interval, exporter version, git commit) are first-class metadata.
  - Both absolute simulation time and burn-in-relative time are stored,
    so "t=0" is never ambiguous between "initialization" and "start of
    recorded data."
  - Written incrementally with run_metadata.attrs['complete'] set to
    True only after the entire run finishes without error -- an
    interrupted run is visibly incomplete, not silently passed off as
    finished data.

Capacity and dissipation fields use velocity_source="deterministic" by
default (see capacity.py / dissipation.py): the earlier validation work
(scripts/validate_capacity.py) showed total-velocity fields are visibly
noise-contaminated relative to the deterministic-only variant. This
choice is recorded in run_metadata rather than silently assumed by a
downstream reader; it can be overridden via ExportConfig.
"""
from __future__ import annotations

import subprocess
from dataclasses import dataclass, asdict, fields as dataclass_fields
from typing import Optional

import h5py
import numpy as np

from .types import ABPParameters, init_state, init_state_random
from .integrator import step_euler_maruyama
from .equilibration import run_burn_in
from .diagnostics import check_stability, InstabilityDetected
from .fields import Grid, reconstruct_density, reconstruct_density_and_current
from .capacity import phi_input, reconstruct_phi_parallel, reconstruct_phi_plus
from .dissipation import reconstruct_s_work, reconstruct_s_current
from .density_domains import composite_mips_indicators
from .observables import largest_cluster_fraction
from .features import compute_field_features, compute_cross_field_features
from .onset import detect_onset_events

EXPORTER_VERSION = "0.1.0"


def _git_commit() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], stderr=subprocess.DEVNULL
        ).decode().strip()
    except Exception:
        return "unknown"


@dataclass
class ExportConfig:
    N: int
    phi: float
    Pe: float
    dt: Optional[float] = None  # required; see __post_init__
    Dr: float = 0.3
    Dt: float = 0.1
    mobility: float = 1.0
    sigma: float = 1.0
    epsilon: float = 1.0

    init_method: str = "rsa"  # "rsa" (production default) or "grid" (validation mode only)
    min_distance_factor: float = 1.02
    burnin_time: float = 6.0
    seed: int = 0
    preparation_version: str = "v1-rsa-burnin6"

    t_run: float = 20.0
    record_every_time: float = 0.1

    grid_n: int = 24
    kernel_h: float = 1.2
    capacity_velocity_source: str = "deterministic"
    contact_cutoff_factor: float = 1.3  # for the DIAGNOSTIC f_contact_max only
    rho_star_factor: float = 1.3
    rho_low_factor: float = 0.7

    save_particles: bool = True

    onset_theta_high: float = 0.3
    onset_theta_low: float = 0.15
    onset_min_dwell: float = 1.0
    onset_s_low_q_threshold: Optional[float] = None

    def __post_init__(self):
        if self.dt is None:
            raise ValueError(
                "dt must be set explicitly -- the stable timestep for "
                "explicit Euler-Maruyama under WCA depends on v0, Dt, "
                "and packing fraction (see diagnostics.py); there is no "
                "safe default. Probe stability for this specific "
                "configuration before exporting, and pass the result "
                "in."
            )
        if self.init_method not in ("rsa", "grid"):
            raise ValueError("init_method must be 'rsa' or 'grid'")

    @property
    def L(self) -> float:
        return float(np.sqrt(self.N * np.pi * (self.sigma / 2.0) ** 2 / self.phi))

    @property
    def v0(self) -> float:
        return self.Pe * self.Dr * self.sigma

    def to_abp_params(self) -> ABPParameters:
        return ABPParameters(
            N=self.N, L=self.L, sigma=self.sigma, epsilon=self.epsilon,
            mobility=self.mobility, v0=self.v0, Dt=self.Dt, Dr=self.Dr,
            dt=self.dt,
        )


def _create_resizable(group: h5py.Group, name: str, frame_shape: tuple,
                       dtype=np.float64) -> h5py.Dataset:
    return group.create_dataset(
        name, shape=(0,) + frame_shape, maxshape=(None,) + frame_shape,
        dtype=dtype, chunks=(1,) + frame_shape if frame_shape else (64,),
        compression="gzip", compression_opts=4,
    )


def _append(ds: h5py.Dataset, value) -> None:
    n = ds.shape[0]
    ds.resize(n + 1, axis=0)
    ds[n] = value


def _create_feature_table(group: h5py.Group, name: str, columns: list[str]) -> h5py.Dataset:
    ds = group.create_dataset(
        name, shape=(0, len(columns)), maxshape=(None, len(columns)),
        dtype=np.float64, chunks=(64, len(columns)),
        compression="gzip", compression_opts=4,
    )
    ds.attrs["columns"] = columns
    return ds


def _append_row(ds: h5py.Dataset, feature_dict: dict, columns: list[str]) -> None:
    n = ds.shape[0]
    ds.resize(n + 1, axis=0)
    ds[n] = [feature_dict[c] for c in columns]


def export_run(config: ExportConfig, out_path: str) -> None:
    params = config.to_abp_params()
    rng = np.random.default_rng(config.seed)

    if config.init_method == "rsa":
        state = init_state_random(params, rng,
                                   min_distance_factor=config.min_distance_factor)
    else:
        state = init_state(params, rng)

    with h5py.File(out_path, "w") as f:
        meta = f.create_group("run_metadata")
        meta.attrs["complete"] = False
        meta.attrs["exporter_version"] = EXPORTER_VERSION
        meta.attrs["git_commit"] = _git_commit()
        for field_ in dataclass_fields(config):
            val = getattr(config, field_.name)
            if val is None:
                continue
            meta.attrs[field_.name] = val
        meta.attrs["L"] = config.L
        meta.attrs["v0"] = config.v0
        f.flush()

        # --- burn-in (unrecorded) ---
        burnin_failed = False
        try:
            run_burn_in(state, params, rng, duration=config.burnin_time)
        except InstabilityDetected as exc:
            burnin_failed = True
            meta.attrs["burnin_instability"] = str(exc)

        t_origin = state.t  # absolute sim time at start of recorded data

        # --- dataset creation ---
        gn = config.grid_n
        time_grp = f.create_group("time")
        ds_abs_t = _create_resizable(time_grp, "abs_time", ())
        ds_rel_t = _create_resizable(time_grp, "rel_time", ())

        fields_grp = f.create_group("fields")
        ds_rho = _create_resizable(fields_grp, "rho", (gn, gn))
        ds_Jx = _create_resizable(fields_grp, "current_x", (gn, gn))
        ds_Jy = _create_resizable(fields_grp, "current_y", (gn, gn))
        ds_vx = _create_resizable(fields_grp, "velocity_x", (gn, gn))
        ds_vy = _create_resizable(fields_grp, "velocity_y", (gn, gn))

        diss_grp = fields_grp.create_group("dissipation")
        ds_s_work = _create_resizable(diss_grp, "work", (gn, gn))
        ds_s_current = _create_resizable(diss_grp, "current", (gn, gn))

        cap_grp = fields_grp.create_group("capacity")
        # Phi_input is spatially uniform for this constant-speed model
        # (see capacity.py) -- stored as a scalar-per-frame series, not a
        # wasted constant grid. Documented explicitly here so this
        # deviation from a literal (n_frames, gn, gn) shape isn't a
        # silent surprise to a downstream reader.
        ds_phi_input = _create_resizable(cap_grp, "input", ())
        cap_grp["input"].attrs["note"] = (
            "Spatially uniform (= v0) for this constant-speed ABP model; "
            "stored as one scalar per frame, not a (gn, gn) grid."
        )
        ds_phi_par = _create_resizable(cap_grp, "parallel", (gn, gn))
        ds_phi_plus = _create_resizable(cap_grp, "forward", (gn, gn))

        particles_grp = None
        ds_px = ds_ptheta = None
        if config.save_particles:
            particles_grp = f.create_group("particles")
            ds_px = _create_resizable(particles_grp, "x", (config.N, 2))
            ds_ptheta = _create_resizable(particles_grp, "theta", (config.N,))

        baseline_cols = [f"rho_{k}" for k in
                         ("mean", "var", "skew", "q90", "grad2_mean",
                          "lap2_mean", "S_low_q", "xi")]
        capacity_input_cols = ["phi_input_value"]
        capacity_par_cols = [f"phi_parallel_{k}" for k in
                              ("mean", "var", "skew", "q90", "grad2_mean",
                               "lap2_mean", "S_low_q", "xi")]
        capacity_fwd_cols = [f"phi_forward_{k}" for k in
                              ("mean", "var", "skew", "q90", "grad2_mean",
                               "lap2_mean", "S_low_q", "xi")]
        dissipation_cols = (
            [f"s_work_{k}" for k in
             ("mean", "var", "skew", "q90", "grad2_mean", "lap2_mean",
              "S_low_q", "xi")] +
            [f"s_current_{k}" for k in
             ("mean", "var", "skew", "q90", "grad2_mean", "lap2_mean",
              "S_low_q", "xi")]
        )
        interaction_pairs = [
            ("rho", "phi_parallel"), ("rho", "phi_forward"),
            ("rho", "s_work"), ("rho", "s_current"),
            ("phi_parallel", "s_work"), ("phi_parallel", "s_current"),
        ]
        interaction_cols = []
        for a, b in interaction_pairs:
            interaction_cols += [f"cov_{a}_{b}", f"meanprod_{a}_{b}",
                                  f"grad_align_{a}_{b}"]

        feat_grp = f.create_group("features")
        ds_feat_baseline = _create_feature_table(feat_grp, "baseline", baseline_cols)
        ds_feat_cap_in = _create_feature_table(feat_grp, "capacity_input", capacity_input_cols)
        ds_feat_cap_par = _create_feature_table(feat_grp, "capacity_parallel", capacity_par_cols)
        ds_feat_cap_fwd = _create_feature_table(feat_grp, "capacity_forward", capacity_fwd_cols)
        ds_feat_diss = _create_feature_table(feat_grp, "dissipation", dissipation_cols)
        ds_feat_inter = _create_feature_table(feat_grp, "interactions", interaction_cols)

        targets_grp = f.create_group("targets")
        ds_t_fdense = _create_resizable(targets_grp, "f_dense_max", ())
        ds_t_fvoid = _create_resizable(targets_grp, "f_void", ())
        ds_t_slowq = _create_resizable(targets_grp, "S_rho_low_q", ())
        ds_t_brho = _create_resizable(targets_grp, "B_rho", ())

        diag_grp = f.create_group("diagnostics")
        ds_diag_fcontact = _create_resizable(diag_grp, "f_contact_max", ())

        grid = Grid(L=config.L, n=gn)
        h = config.kernel_h
        vsrc = config.capacity_velocity_source

        n_steps = int(config.t_run / config.dt)
        record_every = max(1, int(config.record_every_time / config.dt))

        run_failed = burnin_failed
        failure_reason = meta.attrs.get("burnin_instability", "")

        f_dense_series = []
        s_low_q_series = []
        rel_time_series = []

        if not burnin_failed:
            for step in range(n_steps + 1):
                if step % record_every == 0:
                    rho_h = reconstruct_density(state, params, grid, h)
                    _, Jx_h, Jy_h, vx_h, vy_h = reconstruct_density_and_current(
                        state, params, grid, h
                    )
                    s_work = reconstruct_s_work(state, params, grid, h, velocity_source=vsrc)
                    s_current_field = reconstruct_s_current(rho_h, Jx_h, Jy_h, params)
                    p_input = phi_input(params)
                    p_par = reconstruct_phi_parallel(state, params, grid, h, velocity_source=vsrc)
                    p_fwd = reconstruct_phi_plus(state, params, grid, h, velocity_source=vsrc)

                    abs_t = state.t
                    rel_t = state.t - t_origin
                    _append(ds_abs_t, abs_t)
                    _append(ds_rel_t, rel_t)

                    _append(ds_rho, rho_h)
                    _append(ds_Jx, Jx_h)
                    _append(ds_Jy, Jy_h)
                    _append(ds_vx, vx_h)
                    _append(ds_vy, vy_h)
                    _append(ds_s_work, s_work)
                    _append(ds_s_current, s_current_field)
                    _append(ds_phi_input, p_input)
                    _append(ds_phi_par, p_par)
                    _append(ds_phi_plus, p_fwd)

                    if config.save_particles:
                        _append(ds_px, state.x)
                        _append(ds_ptheta, state.theta)

                    feats_rho = compute_field_features(rho_h, config.L, "rho")
                    _append_row(ds_feat_baseline, feats_rho, baseline_cols)
                    _append_row(ds_feat_cap_in, {"phi_input_value": p_input}, capacity_input_cols)
                    feats_par = compute_field_features(p_par, config.L, "phi_parallel")
                    _append_row(ds_feat_cap_par, feats_par, capacity_par_cols)
                    feats_fwd = compute_field_features(p_fwd, config.L, "phi_forward")
                    _append_row(ds_feat_cap_fwd, feats_fwd, capacity_fwd_cols)
                    feats_work = compute_field_features(s_work, config.L, "s_work")
                    feats_current = compute_field_features(s_current_field, config.L, "s_current")
                    _append_row(ds_feat_diss, {**feats_work, **feats_current}, dissipation_cols)

                    inter_feats = {}
                    field_lookup = {"rho": rho_h, "phi_parallel": p_par,
                                     "phi_forward": p_fwd, "s_work": s_work,
                                     "s_current": s_current_field}
                    for a, b in interaction_pairs:
                        inter_feats.update(
                            compute_cross_field_features(field_lookup[a], field_lookup[b],
                                                          config.L, a, b)
                        )
                    _append_row(ds_feat_inter, inter_feats, interaction_cols)

                    indicators = composite_mips_indicators(
                        rho_h, config.L, rho_star_factor=config.rho_star_factor,
                        rho_low_factor=config.rho_low_factor,
                    )
                    _append(ds_t_fdense, indicators["f_dense_max"])
                    _append(ds_t_fvoid, indicators["f_void"])
                    _append(ds_t_slowq, indicators["S_rho_low_q"])
                    _append(ds_t_brho, indicators["B_rho"])

                    f_contact = largest_cluster_fraction(
                        state.x, config.L, config.sigma,
                        cutoff_factor=config.contact_cutoff_factor,
                    )
                    _append(ds_diag_fcontact, f_contact)

                    f_dense_series.append(indicators["f_dense_max"])
                    s_low_q_series.append(indicators["S_rho_low_q"])
                    rel_time_series.append(rel_t)

                    if len(rel_time_series) % 50 == 0:
                        f.flush()

                if step < n_steps:
                    try:
                        step_euler_maruyama(state, params, rng)
                        check_stability(state, step=step)
                    except InstabilityDetected as exc:
                        run_failed = True
                        failure_reason = str(exc)
                        break

        # --- onset labels, computed once over the full recorded series ---
        labels_grp = f.create_group("labels")
        rel_time_arr = np.array(rel_time_series)
        f_dense_arr = np.array(f_dense_series)
        s_low_q_arr = np.array(s_low_q_series)

        if len(rel_time_arr) >= 2:
            events = detect_onset_events(
                rel_time_arr, f_dense_arr,
                theta_high=config.onset_theta_high,
                theta_low=config.onset_theta_low,
                min_dwell=config.onset_min_dwell,
                s_low_q=s_low_q_arr if config.onset_s_low_q_threshold is not None else None,
                s_low_q_threshold=config.onset_s_low_q_threshold,
            )
        else:
            events = []

        onset_mask = np.zeros(len(rel_time_arr), dtype=bool)
        for ev in events:
            onset_mask |= (rel_time_arr >= ev.start_time) & (rel_time_arr <= ev.end_time)
        labels_grp.create_dataset("onset", data=onset_mask)

        onset_def = labels_grp.create_group("onset_definition")
        onset_def.attrs["theta_high"] = config.onset_theta_high
        onset_def.attrs["theta_low"] = config.onset_theta_low
        onset_def.attrs["min_dwell"] = config.onset_min_dwell
        onset_def.attrs["s_low_q_threshold"] = (
            config.onset_s_low_q_threshold if config.onset_s_low_q_threshold is not None else np.nan
        )
        onset_def.attrs["series_used"] = "f_dense_max"
        onset_def.attrs["confirmation_series"] = (
            "S_rho_low_q" if config.onset_s_low_q_threshold is not None else "none"
        )
        onset_def.attrs["n_events"] = len(events)

        if events:
            event_dtype = np.dtype([("start_time", "f8"), ("end_time", "f8"),
                                     ("peak_value", "f8"), ("duration", "f8")])
            event_arr = np.array(
                [(ev.start_time, ev.end_time, ev.peak_f_max, ev.duration)
                 for ev in events], dtype=event_dtype,
            )
            labels_grp.create_dataset("onset_events", data=event_arr)

        # --- finalize ---
        meta.attrs["run_failed"] = run_failed
        meta.attrs["failure_reason"] = failure_reason
        meta.attrs["n_frames"] = len(rel_time_arr)
        meta.attrs["t_origin_abs"] = t_origin
        meta.attrs["complete"] = not run_failed
        f.flush()
