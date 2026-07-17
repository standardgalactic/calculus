"""
Visual demonstration of persistence-based onset detection on a real
simulated trajectory, at the same scale as Milestone 1.5's Figure 5
(the run that first showed f_max swinging 0.2-1.0 within a few time
units at fixed cutoff).

Compares naive raw-threshold-crossing "events" against the
persistence-filtered onset events this module actually returns.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from rsvp_mips import (  # noqa: E402
    ABPParameters, init_state, step_euler_maruyama,
    largest_cluster_fraction,
)
from rsvp_mips.onset import detect_onset_events, prospective_labels  # noqa: E402
from rsvp_mips.diagnostics import check_stability  # noqa: E402

RESULTS_DIR = Path(__file__).resolve().parents[1] / "results"
RESULTS_DIR.mkdir(exist_ok=True)


def main():
    params = ABPParameters(
        N=300, L=25.0, sigma=1.0, epsilon=1.0, mobility=1.0,
        v0=40.0, Dt=0.1, Dr=0.3, dt=0.0002,
    )
    print(f"Running: N={params.N}, phi (approx)={params.N * np.pi * 0.25 / params.L**2:.3f}, "
          f"Pe={params.Pe:.1f}")

    rng = np.random.default_rng(7)
    state = init_state(params, rng)

    n_steps = 60000
    record_every = 300
    times, f_max_series = [], []
    for step in range(n_steps + 1):
        if step % record_every == 0:
            times.append(state.t)
            f_max_series.append(
                largest_cluster_fraction(state.x, params.L, params.sigma,
                                          cutoff_factor=1.3)
            )
        if step < n_steps:
            step_euler_maruyama(state, params, rng)
            check_stability(state, step=step)

    times = np.array(times)
    f_max = np.array(f_max_series)

    theta_high, theta_low, min_dwell = 0.6, 0.4, 1.0
    events = detect_onset_events(times, f_max, theta_high=theta_high,
                                  theta_low=theta_low, min_dwell=min_dwell)
    raw_crossings_idx = np.where(np.diff((f_max >= theta_high).astype(int)) == 1)[0] + 1

    print(f"\nraw threshold crossings (f_max >= {theta_high}): "
          f"{len(raw_crossings_idx)}")
    print(f"persistence-filtered onset events (min_dwell={min_dwell}): "
          f"{len(events)}")
    for ev in events:
        print(f"  event: start={ev.start_time:.2f}, end={ev.end_time:.2f}, "
              f"duration={ev.duration:.2f}, peak={ev.peak_f_max:.3f}")

    Y = prospective_labels(times, events, tau=2.0)

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(times, f_max, lw=1.5, color="steelblue", label=r"$f_{max}(t)$")
    ax.axhline(theta_high, color="crimson", ls="--", alpha=0.6,
               label=f"theta_high={theta_high}")
    ax.axhline(theta_low, color="orange", ls="--", alpha=0.6,
               label=f"theta_low={theta_low}")
    for t in times[raw_crossings_idx]:
        ax.axvline(t, color="gray", alpha=0.2, lw=1)
    for ev in events:
        ax.axvspan(ev.start_time, ev.end_time, color="crimson", alpha=0.2)
    ax.fill_between(times, 0, 0.03, where=Y, color="green", alpha=0.5,
                     label=r"$Y_\tau(t)=1$ ($\tau=2$)", transform=ax.get_xaxis_transform())
    ax.set_xlabel("t")
    ax.set_ylabel(r"$f_{max}(t)$")
    ax.set_title("Onset Detection: raw crossings (gray) vs. persistence-"
                  "filtered events (red spans)")
    ax.legend(loc="upper right", fontsize=8)
    fig.tight_layout()
    fig.savefig(RESULTS_DIR / "onset_detection_demo.png", dpi=150)
    plt.close(fig)
    print(f"\nFigure written to {RESULTS_DIR / 'onset_detection_demo.png'}")


if __name__ == "__main__":
    main()
