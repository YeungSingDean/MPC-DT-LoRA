"""Generate exactly four EACT trajectories at the seasonal typical weeks.

These are added to the offline dataset so both pretraining and V3
post-training see the seasonal distributions that the evaluation uses.
"""

from __future__ import annotations

import argparse
import os
import sys

import numpy as np

_PROJECT_ROOT = os.path.abspath(
    os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..")
)
sys.path.insert(0, _PROJECT_ROOT)

from coastaldc_env import CoastalDCContinuousEnv  # noqa: E402
from coastaldc_env.rollout import run_episode, save_trajectories  # noqa: E402
from controllers.eact_mpc import EACTMPCController  # noqa: E402


SEASONAL_TIMESTAMPS = (
    "2025-01-15 00:00:00",
    "2025-04-15 00:00:00",
    "2025-07-15 00:00:00",
    "2025-10-15 00:00:00",
)


def resolve_start_hours(env, timestamps):
    import pandas as pd

    parsed = pd.to_datetime(env.data["timestamp"])
    lookup = {ts: i for i, ts in enumerate(parsed)}
    min_start = (
        int(env._causal_min_history) if env.forecast_mode == "causal" else 0
    )
    out = []
    for value in timestamps:
        ts = pd.Timestamp(value)
        if ts not in lookup:
            raise ValueError(f"timestamp missing from data: {value}")
        out.append(int(lookup[ts]))
    return out


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--country", default="NLD")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--forecast-dir", default=None)
    parser.add_argument(
        "--out", default=None,
        help="Output npz (defaults to data/trajectories/eact_<COUNTRY>_seasonal.npz)",
    )
    args = parser.parse_args()

    forecast_dir = args.forecast_dir or os.path.join(
        _PROJECT_ROOT, "results", "causal_forecasts_v3_gated_bias"
    )

    # IMPORTANT: forecast_mode must match the training/eval setup.
    env = CoastalDCContinuousEnv(
        country=args.country,
        seed=args.seed,
        forecast_mode="causal",
        forecast_dir=forecast_dir,
    )

    start_hours = resolve_start_hours(env, SEASONAL_TIMESTAMPS)
    print(f"[info] seasonal start hours: {start_hours}")

    ctrl = EACTMPCController(
        env,
        forecast_dir=forecast_dir,
        horizon=24,
        maxiter=20,
        control_block_hours=6,
    )

    trajs = []
    for ep, start_hour in enumerate(start_hours):
        ctrl.reset()
        traj = run_episode(
            env, ctrl, seed=3000 + ep, start_hour=start_hour,
        )
        trajs.append(traj)
        print(
            f"seasonal {ep} (start_hour={start_hour}): "
            f"return={traj['episode_return']:.2f}, "
            f"max_t_inlet={float(np.max(traj['t_inlet'])):.3f}, "
            f"thermal_hours={int((traj['t_inlet'] > 27.0).sum())}"
        )

    out = args.out or os.path.join(
        _PROJECT_ROOT, "data", "trajectories",
        f"eact_{args.country}_seasonal.npz",
    )
    os.makedirs(os.path.dirname(out), exist_ok=True)
    save_trajectories(out, trajs, source="eact")
    print(f"saved {len(trajs)} seasonal trajectories -> {out}")


if __name__ == "__main__":
    main()