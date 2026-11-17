"""Train the continuous Decision Transformer on the mixed offline dataset.

Recommended mixture:
    40% EACT-MPC + 30% rule-based + 20% safe-random (+10% RL expert if available).

Before running this script, regenerate the trajectory files with the updated
rollout.py so that both requested_actions and applied_actions are stored.

This script lives in scripts/eact-dt-lora/, and expects the EACT expert pool
to be data/trajectories/eact_<COUNTRY>.npz. If that file is missing, the loader
falls back to the legacy mpc_<COUNTRY>.npz for backward compatibility.

Examples (run from this directory):
    python generate_eact_trajectories.py --country JPN --episodes 40 --horizon 24
    python train_decision_transformer.py --country JPN --epochs 20
    python train_decision_transformer.py --country JPN --forecast-mode oracle
    python train_decision_transformer.py --country JPN --target-percentile 90
"""

from __future__ import annotations

import argparse
import glob
import os
import sys

import numpy as np
import torch

# scripts/eact-dt-lora/ -> project root is two levels up.
_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.abspath(
    os.path.join(_SCRIPT_DIR, "..", "..")
)
sys.path.insert(0, _PROJECT_ROOT)

from coastaldc_env import CoastalDCContinuousEnv  # noqa: E402
from coastaldc_env.rollout import load_trajectories, run_episode  # noqa: E402
from controllers.decision_transformer_continuous import (  # noqa: E402
    ContinuousDecisionTransformer,
    DecisionTransformerController,
    train_dt,
)

MIXTURE = {
    "eact": 0.4,
    "rule_based": 0.3,
    "safe_random": 0.2,
    "rl": 0.1,
}

# If the preferred pool file is missing, try these legacy stems instead.
LEGACY_EXPERT_ALIASES: dict[str, tuple[str, ...]] = {
    "eact": ("mpc",),
}


def build_mixed_dataset(
    traj_dir: str,
    country: str,
    rng: np.random.Generator,
) -> list[dict]:
    """Load source trajectory pools and build the requested mixed dataset."""
    pools: dict[str, list[dict]] = {}

    for source in MIXTURE:
        paths = glob.glob(os.path.join(traj_dir, f"{source}_{country}.npz"))
        # Seasonal EACT trajectories are concatenated into the EACT pool
        if source == "eact":
            seasonal = os.path.join(
                traj_dir, f"eact_{country}_seasonal.npz")
            if os.path.isfile(seasonal):
                paths.append(seasonal)
                print(f"[info] + seasonal EACT: {seasonal}")
        actual_source = source

        if not paths:
            for alias in LEGACY_EXPERT_ALIASES.get(source, ()):
                alt = glob.glob(
                    os.path.join(traj_dir, f"{alias}_{country}.npz"))
                if alt:
                    paths = alt
                    actual_source = alias
                    print(
                        f"[info] {source}_{country}.npz not found; "
                        f"falling back to {alias}_{country}.npz"
                    )
                    break

        if paths:
            # Updated rollout.py defaults to use_applied_actions=True.
            pools[source] = load_trajectories(paths)

            # Keep the logical pool name (e.g. "eact") even if we loaded
            # a legacy file (e.g. mpc_*.npz).
            for traj in pools[source]:
                traj["source"] = source

            if actual_source != source:
                print(
                    f"[info] loaded {len(pools[source])} trajectories from "
                    f"{actual_source} into pool '{source}'"
                )

    if not pools:
        raise FileNotFoundError(
            f"No trajectory files found in {traj_dir} for {country}. "
            "Run generate_eact_trajectories.py (or another generate_* script) first."
        )

    total = sum(len(v) for v in pools.values())
    mixed: list[dict] = []

    present = {source: weight for source, weight in MIXTURE.items() if source in pools}
    weight_sum = sum(present.values())

    for source, trajs in pools.items():
        n = max(1, int(round(total * present[source] / weight_sum)))
        idx = rng.choice(
            len(trajs),
            size=min(n, len(trajs)),
            replace=False,
        )
        mixed.extend(trajs[i] for i in idx)

    rng.shuffle(mixed)

    print(
        {source: len(trajs) for source, trajs in pools.items()},
        "-> mixed:",
        len(mixed),
    )
    return mixed


def trajectory_returns(trajectories: list[dict]) -> np.ndarray:
    """Return one episode return per trajectory in original reward units."""
    return np.asarray(
        [
            float(np.asarray(traj["rewards"], dtype=np.float64).sum())
            for traj in trajectories
        ],
        dtype=np.float64,
    )


def print_one_return_distribution(
    returns: np.ndarray,
    title: str,
) -> None:
    """Print summary statistics for a non-empty return vector."""
    if len(returns) == 0:
        print(f"\n=== {title} ===")
        print("count  : 0")
        return

    print(f"\n=== {title} ===")
    print(f"count  : {len(returns)}")
    print(f"min    : {returns.min():.2f}")
    print(f"25%    : {np.percentile(returns, 25):.2f}")
    print(f"median : {np.median(returns):.2f}")
    print(f"75%    : {np.percentile(returns, 75):.2f}")
    print(f"90%    : {np.percentile(returns, 90):.2f}")
    print(f"max    : {returns.max():.2f}")
    print(f"mean   : {returns.mean():.2f}")
    print(f"std    : {returns.std():.2f}")


def print_source_return_statistics(trajectories: list[dict]) -> None:
    """Print return distributions separately for each trajectory source."""
    print("\n==============================================")
    print("Return distribution by trajectory source")
    print("==============================================")

    preferred_order = ["eact", "mpc", "rule_based", "safe_random", "rl"]

    observed_sources = {
        str(traj.get("source", "unknown"))
        for traj in trajectories
    }

    ordered_sources = [
        source for source in preferred_order
        if source in observed_sources
    ]
    ordered_sources.extend(
        sorted(observed_sources.difference(ordered_sources))
    )

    for source in ordered_sources:
        source_trajs = [
            traj
            for traj in trajectories
            if str(traj.get("source", "unknown")) == source
        ]
        source_returns = trajectory_returns(source_trajs)
        print_one_return_distribution(
            source_returns,
            f"{source} trajectory returns",
        )

    print("\n==============================================\n")


def print_overall_return_statistics(returns: np.ndarray) -> None:
    """Print return distribution of the complete mixed offline dataset."""
    print_one_return_distribution(
        returns,
        "Offline mixed trajectory return distribution",
    )
    print("==============================================\n")


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--country", default="JPN")
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--context-len", type=int, default=24)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--traj-dir", default=None)
    parser.add_argument("--eval-episodes", type=int, default=5)

    parser.add_argument(
        "--forecast-mode",
        choices=("oracle", "causal", "none"),
        default="causal",
        help="Forecast features used during DT evaluation (default: causal).",
    )
    parser.add_argument(
        "--forecast-dir",
        default=None,
        help="Directory containing causal_ridge_<COUNTRY>.npz models.",
    )

    parser.add_argument(
        "--target-percentile",
        type=float,
        default=90.0,
        help=(
            "Percentile of offline episode returns used as the DT target return. "
            "Use 100 to reproduce the old max(returns) behavior."
        ),
    )

    parser.add_argument(
        "--return-scale",
        type=float,
        default=100.0,
        help=(
            "Positive scale factor for Decision Transformer return-to-go tokens. "
            "The same scale is used during training and rollout."
        ),
    )

    # Output location override (default: scripts/eact-dt-lora/results/models)
    parser.add_argument(
        "--out-dir",
        default=None,
        help=(
            "Directory for the trained checkpoint. Default: "
            "scripts/eact-dt-lora/results/models/"
        ),
    )

    args = parser.parse_args()

    if not (0.0 <= args.target_percentile <= 100.0):
        parser.error("--target-percentile must be between 0 and 100")

    if args.return_scale <= 0:
        parser.error("--return-scale must be > 0")

    root = _PROJECT_ROOT
    traj_dir = args.traj_dir or os.path.join(
        root,
        "data",
        "trajectories",
    )

    device = "cuda" if torch.cuda.is_available() else "cpu"
    rng = np.random.default_rng(args.seed)

    # ------------------------------------------------------------------
    # 1. Load the mixed offline dataset.
    # ------------------------------------------------------------------
    trajs = build_mixed_dataset(
        traj_dir,
        args.country,
        rng,
    )

    obs_dim = trajs[0]["observations"].shape[1]

    # ------------------------------------------------------------------
    # 2. Diagnose the quality of each data source separately.
    # ------------------------------------------------------------------
    print_source_return_statistics(trajs)

    # ------------------------------------------------------------------
    # 3. Diagnose the return distribution of the full mixed dataset.
    # ------------------------------------------------------------------
    returns = trajectory_returns(trajs)
    print_overall_return_statistics(returns)

    # ------------------------------------------------------------------
    # 4. Choose a supported target return instead of always max(returns).
    # ------------------------------------------------------------------
    target = float(
        np.percentile(
            returns,
            args.target_percentile,
        )
    )

    print(
        f"DT target return: {target:.2f} "
        f"({args.target_percentile:g}th percentile)"
    )
    print(f"RTG scale: {args.return_scale:g}")
    print(f"Device: {device}\n")

    # ------------------------------------------------------------------
    # 5. Train Decision Transformer.
    # ------------------------------------------------------------------
    model = ContinuousDecisionTransformer(
        obs_dim,
        context_len=args.context_len,
    ).to(device)

    losses = train_dt(
        model,
        trajs,
        epochs=args.epochs,
        device=device,
        seed=args.seed,
        return_scale=args.return_scale,
    )

    print(f"final MSE loss: {losses[-1]}")

    # ------------------------------------------------------------------
    # 6. Save trained model.
    #
    # Output goes under THIS script's folder:
    #   scripts/eact-dt-lora/results/models/
    # so the V3 post-training and evaluation scripts can find it without
    # guessing between two possible trees.
    # ------------------------------------------------------------------
    out_dir = (
        os.path.abspath(args.out_dir)
        if args.out_dir
        else os.path.join(_SCRIPT_DIR, "results", "models")
    )
    os.makedirs(out_dir, exist_ok=True)

    model_path = os.path.join(
        out_dir,
        f"dt_{args.country}_seed{args.seed}.pt",
    )

    torch.save(
        model.state_dict(),
        model_path,
    )
    print(f"saved -> {model_path}")

    # ------------------------------------------------------------------
    # 7. Closed-loop evaluation.
    # ------------------------------------------------------------------
    env = CoastalDCContinuousEnv(
        country=args.country,
        seed=args.seed,
        forecast_mode=args.forecast_mode,
        forecast_dir=args.forecast_dir,
    )

    ctrl = DecisionTransformerController(
        obs_dim,
        target_return=target,
        context_len=args.context_len,
        device=device,
        return_scale=args.return_scale,
    )
    ctrl.model.load_state_dict(model.state_dict())

    eval_returns: list[float] = []

    for ep in range(args.eval_episodes):
        traj = run_episode(
            env,
            ctrl,
            seed=9000 + ep,
        )

        ep_return = float(traj["episode_return"])
        eval_returns.append(ep_return)

        print(
            f"eval episode {ep}: "
            f"return={ep_return:.2f} "
            f"(target {target:.2f})"
        )

    if eval_returns:
        eval_returns_array = np.asarray(
            eval_returns,
            dtype=np.float64,
        )

        print(
            f"eval mean return={eval_returns_array.mean():.2f}, "
            f"std={eval_returns_array.std():.2f}, "
            f"best={eval_returns_array.max():.2f}, "
            f"worst={eval_returns_array.min():.2f}"
        )


if __name__ == "__main__":
    main()