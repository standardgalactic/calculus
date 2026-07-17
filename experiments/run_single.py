"""
Milestone 1 verification run.

Initializes a MIPS-susceptible ABP system (N=256, Pe~100), integrates it,
records structural diagnostics periodically, and serializes the full
trajectory plus metadata (seed, params, git commit) to HDF5.

This script does not touch RSVP quantities. It exists only to confirm the
engine is a trustworthy, theory-neutral generator of ABP dynamics before
Milestone 1.5 (physical validation against published benchmarks) and any
later field reconstruction.
"""
from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import h5py
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from rsvp_mips import (  # noqa: E402
    ABPParameters,
    init_state,
    step_euler_maruyama,
    radial_structure_factor,
    largest_cluster_fraction,
)


def git_commit() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], stderr=subprocess.DEVNULL
        ).decode().strip()
    except Exception:
        return "unknown"


def run_simulation(seed: int = 42, steps: int = 10_000,
                    record_every: int = 2000) -> None:
    params = ABPParameters(
        N=256, L=32.0, sigma=1.0, epsilon=1.0, mobility=1.0,
        v0=20.0, Dt=0.1, Dr=0.2, dt=0.001,
    )
    rng = np.random.default_rng(seed)
    state = init_state(params, rng)

    print("Starting ABP verification engine...")
    print(f"Parameters: N={params.N}, L={params.L}, Pe={params.Pe:.2f}")

    x_history = [state.x_unwrapped.copy()]
    u_history = [state.u.copy()]
    times = [state.t]

    for step in range(1, steps + 1):
        step_euler_maruyama(state, params, rng)

        if step % record_every == 0:
            x_history.append(state.x_unwrapped.copy())
            u_history.append(state.u.copy())
            times.append(state.t)

            f_max = largest_cluster_fraction(state.x, params.L, params.sigma)
            _, S = radial_structure_factor(state.x, params.L, q_max_idx=5)
            mean_S = np.nanmean(S)
            print(f"Step: {step} | Time: {state.t:.2f} | "
                  f"Max Cluster Frac: {f_max:.3f} | Mean S(q): {mean_S:.3f}")

    out_dir = Path(__file__).resolve().parents[1] / "results"
    out_dir.mkdir(exist_ok=True)
    out_path = out_dir / "verification_trajectory.h5"

    with h5py.File(out_path, "w") as f:
        f.create_dataset("x_unwrapped", data=np.array(x_history))
        f.create_dataset("u", data=np.array(u_history))
        f.create_dataset("t", data=np.array(times))
        f.create_dataset("final_x", data=state.x)
        f.create_dataset("final_theta", data=state.theta)
        meta = f.create_group("metadata")
        meta.attrs["seed"] = seed
        meta.attrs["git_commit"] = git_commit()
        for field_name in params.__dataclass_fields__:
            meta.attrs[field_name] = getattr(params, field_name)

    print(f"Simulation finished. Saved to {out_path}")


if __name__ == "__main__":
    run_simulation()
