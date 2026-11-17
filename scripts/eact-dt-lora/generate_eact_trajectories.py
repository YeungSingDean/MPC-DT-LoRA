"""Generate EACT-MPC expert trajectories for offline training.

Usage:
    python scripts\\eact-dt-lora\\generate_eact_trajectories.py --country JPN --episodes 40 --horizon 24
"""

from __future__ import annotations

import argparse
import os
import sys

# This file lives in scripts/eact-dt-lora/, so the project root is two levels up.
_PROJECT_ROOT = os.path.abspath(
    os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..")
)
sys.path.insert(0, _PROJECT_ROOT)

from coastaldc_env import CoastalDCContinuousEnv  # noqa: E402
from coastaldc_env.rollout import run_episode, save_trajectories  # noqa: E402
from controllers.eact_mpc import EACTMPCController  # noqa: E402


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--country", default="JPN")
    parser.add_argument("--episodes", type=int, default=250)
    parser.add_argument("--horizon", type=int, default=24)
    parser.add_argument(
        "--control-block-hours",
        type=int,
        default=6,
        help="Hold each MPC action for this many hours (EACT replans every step).",
    )
    parser.add_argument("--maxiter", type=int, default=20)
    parser.add_argument("--confidence", type=float, default=0.90)
    parser.add_argument(
        "--gamma", type=float, default=0.995,
        help="Discount factor used inside the MPC cost roll-out.",
    )
    parser.add_argument(
        "--out",
        default=None,
        help="Output .npz path (defaults to data/trajectories/eact_<COUNTRY>.npz).",
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--forecast-dir",
        default=None,
        help=(
            "Directory containing causal_ridge_<COUNTRY>.npz and "
            "calibration_residuals_<COUNTRY>.npz."
        ),
    )
    args = parser.parse_args()

    forecast_dir = args.forecast_dir or os.path.join(
        _PROJECT_ROOT,
        "results",
        "causal_forecasts_v3_gated_bias",
    )

    # EACT runs its own causal ridge forecaster + adaptive residual adaptor,
    # so the environment does not need a forecast_mode / forecast_dir here.
    env = CoastalDCContinuousEnv(
        country=args.country,
        seed=args.seed,
    )
    ctrl = EACTMPCController(
        env,
        forecast_dir=forecast_dir,
        horizon=args.horizon,
        maxiter=args.maxiter,
        control_block_hours=args.control_block_hours,
        gamma=args.gamma,
        confidence=args.confidence,
    )

    trajs = []
    for ep in range(args.episodes):
        # Clear adaptive residual windows / solver history between episodes.
        ctrl.reset()
        traj = run_episode(env, ctrl, seed=2000 + ep)
        trajs.append(traj)
        print(f"episode {ep}: return={traj['episode_return']:.2f}")

    out = args.out or os.path.join(
        _PROJECT_ROOT,
        "data",
        "trajectories",
        f"eact_{args.country}.npz",
    )
    os.makedirs(os.path.dirname(out), exist_ok=True)
    save_trajectories(out, trajs, source="eact")
    print(f"saved {len(trajs)} trajectories -> {out}")


if __name__ == "__main__":
    main()