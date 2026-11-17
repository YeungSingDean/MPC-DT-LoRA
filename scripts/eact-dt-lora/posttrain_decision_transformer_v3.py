"""Decision Transformer post-training V3: dual-pool / dual-objective training.

Core idea
---------
Use TWO different data pools because performance alignment and constraint
alignment need different kinds of samples.

Pool A: Performance pool
    High-return EACT-MPC trajectories (default: return >= P75).
    Objective:
        L_perf = ordinary applied-action MSE over sampled context windows.

Pool B: Constraint-boundary pool
    Two sources, deduplicated per timestep:

    (a) Thermal boundary:
        timesteps where the EACT trajectory's T_inlet >= thermal_boundary.
        These are the samples that sit right on the 27 C constraint edge.

    (b) Full action correction:
        timesteps where ||requested_action - applied_action||_inf > eps.
        EACT's 27 C constraint is enforced on setpoint (dim 1) and pump (dim 2),
        so the boundary signal must look at all three channels, not just
        workload (dim 0).

    A boundary sample is built so that the boundary timestep is the LAST
    timestep in the Transformer context.

    Objective:
        L_boundary =
            channel-weighted MSE(predicted action at the boundary timestep,
                                applied action at that timestep)
        with channel weights [1.0, 3.0, 3.0] so that setpoint and pump, which
        actually enforce the thermal constraint, dominate the gradient.

    IMPORTANT: boundary targets are the APPLIED actions, not the requested
    ones. DT is a pure behaviour-cloning policy over the closed-loop
    transition (obs_t, applied_action_t) -> (reward_t, obs_{t+1}), so the
    target must be the action that actually generated the stored reward and
    next state. Training on requested_actions would break the action-state
    causal chain.

Total post-training objective
-----------------------------
    L_total = L_perf + lambda_boundary * L_boundary

Evaluation
----------
After training, the script evaluates THREE controllers on the SAME four
seasonal "typical week" episodes:

    Pre-DT  : the pretrained V3 checkpoint (input to this script)
    Post-DT : the V3 checkpoint produced by this script
    EACT    : the EACT-MPC expert controller, used as a reference

No heuristic safety wrapper is applied to DT. The intent is a faithful clone
of EACT's closed-loop behaviour; any safety property must come from the
training signal, not from a runtime post-hoc projection.

The four seasonal typical weeks are pinned via
env.reset(options={"start_hour": ...}) using these fixed timestamps:
    2025-01-15 / 04-15 / 07-15 / 10-15 at 00:00 UTC
Pass --no-seasonal to fall back to the legacy random-start evaluation.

Thermal metrics (avg_t_inlet / p95_t_inlet / max_t_inlet /
thermal_exposure_c_h / thermal_event_hours / thermal_event_rate_pct /
cold_exposure_sq_c2_h / below_cold_threshold_hours) are computed from a
per-step T_inlet trace collected during evaluation, using the same
definitions as evaluate_lora_vs_eact_season.py.

Paths
-----
This file lives in scripts/eact-dt-lora/. Therefore:
    ROOT       = project root (two levels up)
    SCRIPT_DIR = scripts/eact-dt-lora/

Both the pretrained DT checkpoint and the V3 output live under this
script's own results tree (SCRIPT_DIR/results/models/), so both training
stages share a single well-defined location and do not pollute the
shared ROOT/results/ tree.

Inputs:
    data/trajectories/eact_<COUNTRY>.npz (or mpc_<COUNTRY>.npz fallback)
    data/trajectories/eact_<COUNTRY>_seasonal.npz  (OPTIONAL, appended if present)
    scripts/eact-dt-lora/results/models/dt_<COUNTRY>_seed*.pt (pretrained DT)
    results/causal_forecasts_v3_gated_bias/causal_ridge_<COUNTRY>.npz
    results/causal_forecasts_v3_gated_bias/calibration_residuals_<COUNTRY>.npz

Outputs:
    scripts/eact-dt-lora/results/models/
        dt_<COUNTRY>_post_v3_dualpool_eact<N>_seed<S>.pt
        dt_<COUNTRY>_post_v3_dualpool_eact<N>_seed<S>_history.json

Recommended first run (from scripts/eact-dt-lora/)
--------------------------------------------------
python posttrain_decision_transformer_v3.py ^
    --country NLD ^
    --elite-percentile 75 ^
    --lambda-boundary 1.0 ^
    --thermal-boundary 26.5 ^
    --epochs 10 ^
    --lr 1e-5
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import sys
from dataclasses import asdict, dataclass

import numpy as np
import pandas as pd
import torch
import torch.nn as nn


# ---------------------------------------------------------------------------
# Path resolution
# ---------------------------------------------------------------------------
# This file lives in scripts/eact-dt-lora/. Therefore:
#   ROOT       = project root (two levels up)
#   SCRIPT_DIR = scripts/eact-dt-lora/ (this script's folder)
#
# ROOT is used to locate the code package and shared input artifacts
# (data/, results/causal_forecasts_*).
#
# SCRIPT_DIR is used for BOTH the pretrained DT input and the V3 output,
# so the whole DT pipeline keeps its artifacts under one tree:
#   SCRIPT_DIR/results/models/
# ---------------------------------------------------------------------------

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, "..", ".."))
sys.path.insert(0, ROOT)

from coastaldc_env import CoastalDCContinuousEnv  # noqa: E402
from coastaldc_env.rollout import load_trajectories  # noqa: E402
from controllers.decision_transformer_continuous import (  # noqa: E402
    ContinuousDecisionTransformer,
    DecisionTransformerController,
)
from controllers.eact_mpc import EACTMPCController  # noqa: E402


# Trajectories live under the project root.
DEFAULT_TRAJ_DIR = os.path.join(ROOT, "data", "trajectories")

# Both pretrained DT and V3 output live under this script's own tree.
DEFAULT_PRETRAINED_DIR = os.path.join(SCRIPT_DIR, "results", "models")
DEFAULT_OUTPUT_DIR = os.path.join(SCRIPT_DIR, "results", "models")

# Forecast artifacts remain shared under the project root.
DEFAULT_FORECAST_DIR = os.path.join(
    ROOT,
    "results",
    "causal_forecasts_v3_gated_bias",
)

DEFAULT_SEASONAL_TIMESTAMPS = (
    "2025-01-15 00:00:00",
    "2025-04-15 00:00:00",
    "2025-07-15 00:00:00",
    "2025-10-15 00:00:00",
)
SEASONAL_LABELS = ("winter", "spring", "summer", "autumn")


# Channel weights for the boundary loss.  Setpoint (dim 1) and pump (dim 2)
# are the channels that actually enforce the 27 C constraint in EACT-MPC, so
# they receive higher weight than workload (dim 0).
BOUNDARY_CHANNEL_WEIGHTS = (1.0, 3.0, 3.0)


# =============================================================================
# Default trajectory resolution (EACT-first, MPC fallback, optional seasonal)
# =============================================================================

def _resolve_default_traj_paths(country: str) -> tuple[str, str | None]:
    """Resolve the default trajectory file(s) for a country.

    Resolution order for the *primary* trajectory file:
      1. data/trajectories/eact_<COUNTRY>.npz
      2. data/trajectories/mpc_<COUNTRY>.npz (legacy fallback)

    An optional *seasonal* trajectory file
        data/trajectories/eact_<COUNTRY>_seasonal.npz
    is treated as an ADDITIONAL source when present: it is loaded in
    addition to (not instead of) the primary file. If it is absent, only the
    primary file is used.

    Returns
    -------
    primary_path : str
        The primary trajectory file. Existence is NOT checked here so the
        caller can emit an informative FileNotFoundError.
    seasonal_path : str | None
        The optional seasonal EACT trajectory file, or None if absent.
    """
    preferred = os.path.join(DEFAULT_TRAJ_DIR, f"eact_{country}.npz")
    legacy = os.path.join(DEFAULT_TRAJ_DIR, f"mpc_{country}.npz")

    if os.path.isfile(preferred):
        primary = preferred
    elif os.path.isfile(legacy):
        print(
            f"[info] eact_{country}.npz not found; "
            f"falling back to mpc_{country}.npz"
        )
        primary = legacy
    else:
        primary = preferred  # informational path for the error message

    seasonal_candidate = os.path.join(
        DEFAULT_TRAJ_DIR, f"eact_{country}_seasonal.npz"
    )
    seasonal = seasonal_candidate if os.path.isfile(seasonal_candidate) else None

    return primary, seasonal


def _newest_by_mtime(candidates: list[str]) -> str | None:
    if not candidates:
        return None
    candidates.sort(key=os.path.getmtime, reverse=True)
    return candidates[0]


def _resolve_default_pretrained_path(country: str) -> str:
    """Resolve the pretrained DT checkpoint passed to this script.

    Resolution order:
      1. Plain pretrained DT produced by train_decision_transformer.py:
             dt_<COUNTRY>_seed*.pt
         (newest by mtime)
      2. EACT-trained V3 checkpoint from a previous post-training run:
             dt_<COUNTRY>_post_v3_dualpool_eact*_seed*.pt
         (newest by name)
      3. Legacy mpc250 V3 checkpoint:
             dt_<COUNTRY>_post_v3_dualpool_mpc250_seed0.pt
      4. Fall back to the expected plain-pretrained path so the
         FileNotFoundError message is informative.
    """
    # 1. Plain pretrained DT.
    candidates = glob.glob(
        os.path.join(DEFAULT_PRETRAINED_DIR, f"dt_{country}_seed*.pt")
    )
    newest = _newest_by_mtime(candidates)
    if newest is not None:
        return newest

    # 2. EACT-trained V3 checkpoint.
    candidates = sorted(
        glob.glob(
            os.path.join(
                DEFAULT_PRETRAINED_DIR,
                f"dt_{country}_post_v3_dualpool_eact*_seed*.pt",
            )
        )
    )
    if candidates:
        return candidates[-1]

    # 3. Legacy mpc250 V3 checkpoint.
    legacy = os.path.join(
        DEFAULT_PRETRAINED_DIR,
        f"dt_{country}_post_v3_dualpool_mpc250_seed0.pt",
    )
    if os.path.isfile(legacy):
        print(
            f"[info] EACT V3 checkpoint not found; "
            f"falling back to legacy {os.path.basename(legacy)}"
        )
        return legacy

    # 4. Informative fallback path.
    return os.path.join(
        DEFAULT_PRETRAINED_DIR,
        f"dt_{country}_seed0.pt",
    )


# =============================================================================
# Reproducibility
# =============================================================================

def set_seed(seed: int) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# =============================================================================
# Seasonal start-hour resolution
# =============================================================================

def resolve_seasonal_start_hours(
    data: pd.DataFrame,
    timestamps,
    *,
    episode_hours: int,
    min_start: int = 0,
) -> list[int]:
    if "timestamp" not in data.columns:
        raise ValueError(
            "Seasonal start timestamps require a 'timestamp' column "
            "in the environment data."
        )

    parsed = pd.to_datetime(data["timestamp"], errors="raise")
    if parsed.duplicated().any():
        raise ValueError("Environment data contain duplicate timestamps.")

    lookup = {ts: i for i, ts in enumerate(parsed)}
    max_start = len(data) - episode_hours - 24

    out: list[int] = []
    for value in timestamps:
        ts = pd.Timestamp(value)
        if ts not in lookup:
            raise ValueError(
                "Seasonal timestamp is absent from environment data: "
                f"{value}"
            )
        idx = int(lookup[ts])
        if idx < min_start or idx > max_start:
            raise ValueError(
                f"Seasonal timestamp {value} maps to start_hour={idx}, "
                f"outside valid range [{min_start}, {max_start}]."
            )
        out.append(idx)
    return out


# =============================================================================
# Trajectory utilities
# =============================================================================

def trajectory_return(traj: dict) -> float:
    return float(np.asarray(traj["rewards"], dtype=np.float64).sum())


def trajectory_returns(trajectories: list[dict]) -> np.ndarray:
    return np.asarray(
        [trajectory_return(traj) for traj in trajectories],
        dtype=np.float64,
    )


def workload_correction(traj: dict) -> np.ndarray:
    """Legacy workload-only correction. Kept for backward compatibility.

    New code should use ``action_correction`` instead: EACT's 27 C constraint
    is enforced by setpoint (dim 1) and pump (dim 2), so a workload-only
    correction misses every constraint-critical timestep.
    """
    requested = np.asarray(traj["requested_actions"], dtype=np.float32)
    applied = np.asarray(traj["applied_actions"], dtype=np.float32)

    if requested.shape != applied.shape:
        raise ValueError(
            "requested_actions and applied_actions have different shapes: "
            f"{requested.shape} vs {applied.shape}"
        )

    if requested.ndim != 2 or requested.shape[1] != 3:
        raise ValueError(
            "Expected requested/applied actions with shape (T, 3), "
            f"got {requested.shape}"
        )

    return np.abs(requested[:, 0] - applied[:, 0]).astype(np.float32)


def action_correction(traj: dict) -> np.ndarray:
    """Per-timestep L-infinity correction across ALL 3 action channels.

    EACT's 27 C constraint is enforced by setpoint (dim 1) and pump (dim 2),
    not by workload (dim 0). Restricting the boundary signal to workload
    misses the constraint-critical samples entirely.
    """
    requested = np.asarray(traj["requested_actions"], dtype=np.float32)
    applied = np.asarray(traj["applied_actions"], dtype=np.float32)

    if requested.shape != applied.shape:
        raise ValueError(
            "requested_actions and applied_actions have different shapes: "
            f"{requested.shape} vs {applied.shape}"
        )

    if requested.ndim != 2 or requested.shape[1] != 3:
        raise ValueError(
            "Expected requested/applied actions with shape (T, 3), "
            f"got {requested.shape}"
        )

    return np.abs(requested - applied).max(axis=1).astype(np.float32)


def validate_trajectory_schema(trajectories: list[dict]) -> None:
    if not trajectories:
        raise ValueError("No trajectories were loaded.")

    no_true_applied = 0
    no_t_inlet = 0
    for idx, traj in enumerate(trajectories):
        required = (
            "observations",
            "actions",
            "requested_actions",
            "applied_actions",
            "rewards",
        )
        for key in required:
            if key not in traj:
                raise ValueError(
                    f"Trajectory {idx} is missing required key {key!r}."
                )
        if not bool(traj.get("has_true_applied_actions", True)):
            no_true_applied += 1
        if traj.get("t_inlet") is None:
            no_t_inlet += 1

    if no_true_applied:
        raise RuntimeError(
            f"{no_true_applied}/{len(trajectories)} trajectories do not "
            "contain true applied_actions. Regenerate EACT trajectories "
            "with the updated rollout.py before V3 post-training."
        )

    if no_t_inlet:
        print(
            f"[warning] {no_t_inlet}/{len(trajectories)} trajectories do not "
            "contain a 't_inlet' trace. The thermal boundary source will be "
            "unavailable for them; only action-correction samples will be used. "
            "Regenerate with the updated rollout.py to enable thermal boundaries."
        )


# =============================================================================
# Pool construction
# =============================================================================

@dataclass
class PoolInfo:
    n_all_trajectories: int
    n_elite_trajectories: int
    elite_percentile: float
    elite_threshold_return: float
    mean_return_all: float
    mean_return_elite: float
    total_timesteps: int
    n_boundary_timesteps: int
    n_thermal_boundary_timesteps: int
    n_action_correction_timesteps: int
    boundary_timestep_rate_pct: float
    boundary_correction_mean: float
    boundary_correction_median: float
    boundary_correction_p90: float
    boundary_correction_max: float
    n_trajectories_with_boundary: int
    thermal_boundary_threshold: float


def build_performance_pool(
    trajectories: list[dict],
    elite_percentile: float,
) -> tuple[list[dict], float]:
    returns = trajectory_returns(trajectories)
    threshold = float(np.percentile(returns, elite_percentile))
    elite = [
        traj for traj, ret in zip(trajectories, returns)
        if ret >= threshold
    ]

    if not elite:
        raise RuntimeError(
            "Performance-pool selection produced zero trajectories."
        )

    return elite, threshold


def build_boundary_pool(
    trajectories: list[dict],
    correction_eps: float,
    thermal_threshold: float = 26.5,
) -> tuple[list[tuple[int, int]], np.ndarray, np.ndarray, np.ndarray]:
    """Build the constraint-boundary pool.

    Samples are drawn from two sources and deduplicated per (traj_idx, t):

      (a) Thermal boundary: T_inlet >= thermal_threshold.
      (b) Full action correction: max_c |requested_c - applied_c| > eps.

    Both sources contribute the SAME target: the applied action at that
    timestep. The boundary pool is a selection mechanism over the closed-loop
    data, not a change of supervision target.

    Returns
    -------
    boundary_indices : list of (traj_idx, timestep)
    boundary_magnitudes : (N,) float
        Per-sample scalar weight (>= eps). Used by magnitude sampling and by
        diagnostic stats.
    boundary_targets : (N, 3) float
        The applied action at each boundary timestep. Regression target for
        the boundary loss.
    is_thermal : (N,) bool
        True when the sample comes from the thermal source. Purely diagnostic.
    """
    boundary_map: dict[tuple[int, int], tuple[float, np.ndarray, bool]] = {}

    for traj_idx, traj in enumerate(trajectories):
        applied = np.asarray(traj["applied_actions"], dtype=np.float32)
        n_steps = applied.shape[0]

        # (a) thermal boundary
        t_inlet = traj.get("t_inlet")
        if t_inlet is not None:
            t_inlet = np.asarray(t_inlet, dtype=np.float32).reshape(-1)
            if t_inlet.shape[0] != n_steps:
                raise ValueError(
                    f"Trajectory {traj_idx}: t_inlet length {t_inlet.shape[0]} "
                    f"does not match applied_actions length {n_steps}."
                )
            hot = np.flatnonzero(t_inlet >= thermal_threshold)
            for ts in hot:
                key = (traj_idx, int(ts))
                mag = float(t_inlet[ts] - thermal_threshold) + 1e-3
                boundary_map[key] = (mag, applied[ts], True)

        # (b) full action correction
        correction = action_correction(traj)
        corr_idx = np.flatnonzero(correction > correction_eps)
        for ts in corr_idx:
            key = (traj_idx, int(ts))
            mag = float(correction[ts])
            if key in boundary_map:
                prev_mag, prev_target, prev_thermal = boundary_map[key]
                boundary_map[key] = (
                    max(prev_mag, mag), prev_target, prev_thermal)
            else:
                boundary_map[key] = (mag, applied[ts], False)

    if not boundary_map:
        raise RuntimeError(
            "No boundary samples were found. Check that EACT trajectories "
            "carry 't_inlet' (for the thermal source) and that "
            "--correction-eps is small enough (for the action source)."
        )

    boundary_indices: list[tuple[int, int]] = []
    magnitudes: list[float] = []
    targets: list[np.ndarray] = []
    is_thermal: list[bool] = []

    for key, (mag, target, thermal) in boundary_map.items():
        boundary_indices.append(key)
        magnitudes.append(mag)
        targets.append(np.asarray(target, dtype=np.float32))
        is_thermal.append(bool(thermal))

    return (
        boundary_indices,
        np.asarray(magnitudes, dtype=np.float64),
        np.asarray(targets, dtype=np.float32),
        np.asarray(is_thermal, dtype=bool),
    )


def build_pool_info(
    all_trajs: list[dict],
    elite_trajs: list[dict],
    elite_percentile: float,
    elite_threshold: float,
    boundary_indices: list[tuple[int, int]],
    boundary_magnitudes: np.ndarray,
    is_thermal: np.ndarray,
    thermal_threshold: float,
) -> PoolInfo:
    all_returns = trajectory_returns(all_trajs)
    elite_returns = trajectory_returns(elite_trajs)
    total_timesteps = int(sum(len(traj["rewards"]) for traj in all_trajs))
    boundary_traj_ids = {traj_idx for traj_idx, _ in boundary_indices}

    n_thermal = int(np.sum(is_thermal)) if is_thermal.size else 0
    n_action = int(is_thermal.size - n_thermal)

    return PoolInfo(
        n_all_trajectories=len(all_trajs),
        n_elite_trajectories=len(elite_trajs),
        elite_percentile=float(elite_percentile),
        elite_threshold_return=float(elite_threshold),
        mean_return_all=float(all_returns.mean()),
        mean_return_elite=float(elite_returns.mean()),
        total_timesteps=total_timesteps,
        n_boundary_timesteps=len(boundary_indices),
        n_thermal_boundary_timesteps=n_thermal,
        n_action_correction_timesteps=n_action,
        boundary_timestep_rate_pct=(
            100.0 * len(boundary_indices) / max(total_timesteps, 1)
        ),
        boundary_correction_mean=float(boundary_magnitudes.mean()),
        boundary_correction_median=float(np.median(boundary_magnitudes)),
        boundary_correction_p90=float(np.percentile(boundary_magnitudes, 90)),
        boundary_correction_max=float(boundary_magnitudes.max()),
        n_trajectories_with_boundary=len(boundary_traj_ids),
        thermal_boundary_threshold=float(thermal_threshold),
    )


def print_pool_diagnostics(info: PoolInfo) -> None:
    print("\n" + "=" * 88)
    print("Decision Transformer V3 dual-pool diagnostics")
    print("=" * 88)
    print(f"all EACT trajectories              : {info.n_all_trajectories}")
    print(f"all-data mean return               : {info.mean_return_all:.2f}")
    print(f"elite percentile                   : P{info.elite_percentile:g}")
    print(f"elite return threshold             : {info.elite_threshold_return:.2f}")
    print(f"performance-pool trajectories      : {info.n_elite_trajectories}")
    print(f"performance-pool mean return       : {info.mean_return_elite:.2f}")
    print(f"total EACT timesteps               : {info.total_timesteps}")
    print(f"constraint-boundary timesteps      : {info.n_boundary_timesteps}")
    print(f"  thermal boundary (T >= {info.thermal_boundary_threshold:.2f} C) "
          f": {info.n_thermal_boundary_timesteps}")
    print(f"  action-correction samples        : {info.n_action_correction_timesteps}")
    print(f"boundary timestep rate (%)         : {info.boundary_timestep_rate_pct:.2f}")
    print(f"trajectories containing boundary   : {info.n_trajectories_with_boundary}")
    print(f"boundary magnitude mean            : {info.boundary_correction_mean:.6f}")
    print(f"boundary magnitude median          : {info.boundary_correction_median:.6f}")
    print(f"boundary magnitude P90             : {info.boundary_correction_p90:.6f}")
    print(f"boundary magnitude max             : {info.boundary_correction_max:.6f}")
    print("=" * 88 + "\n")


# =============================================================================
# Batch sampling
# =============================================================================

def compute_rtgs(
    trajectories: list[dict],
    return_scale: float,
) -> list[np.ndarray]:
    rtgs_all = []
    for traj in trajectories:
        rewards = np.asarray(traj["rewards"], dtype=np.float32)
        rtg = (
            np.cumsum(rewards[::-1])[::-1]
            / float(return_scale)
        ).astype(np.float32)
        rtgs_all.append(rtg)
    return rtgs_all


def left_pad_window(
    *,
    states: np.ndarray,
    actions: np.ndarray,
    rtgs: np.ndarray,
    timesteps: np.ndarray,
    K: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    valid = np.ones(len(states), dtype=np.float32)
    pad = K - len(states)

    if pad <= 0:
        return states, actions, rtgs, timesteps, valid

    states = np.concatenate(
        [np.zeros((pad, states.shape[1]), dtype=np.float32), states], axis=0)
    actions = np.concatenate(
        [np.zeros((pad, actions.shape[1]), dtype=np.float32), actions], axis=0)
    rtgs = np.concatenate([np.zeros(pad, dtype=np.float32), rtgs], axis=0)
    timesteps = np.concatenate(
        [np.zeros(pad, dtype=np.int64), timesteps], axis=0)
    valid = np.concatenate([np.zeros(pad, dtype=np.float32), valid], axis=0)

    return states, actions, rtgs, timesteps, valid


def sample_performance_batch(
    *,
    trajectories: list[dict],
    rtgs_all: list[np.ndarray],
    K: int,
    batch_size: int,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    batch_states = []
    batch_actions = []
    batch_rtgs = []
    batch_timesteps = []
    batch_valid = []

    for _ in range(batch_size):
        traj_idx = int(rng.integers(len(trajectories)))
        traj = trajectories[traj_idx]
        rtg = rtgs_all[traj_idx]

        T = len(traj["rewards"])
        max_start = max(T - K + 1, 1)
        start = int(rng.integers(0, max_start))
        end = min(start + K, T)

        states = np.asarray(traj["observations"][start:end], dtype=np.float32)
        actions = np.asarray(traj["actions"][start:end], dtype=np.float32)
        rtgs = np.asarray(rtg[start:end], dtype=np.float32)
        timesteps = np.arange(start, end, dtype=np.int64)

        states, actions, rtgs, timesteps, valid = left_pad_window(
            states=states,
            actions=actions,
            rtgs=rtgs,
            timesteps=timesteps,
            K=K,
        )

        batch_states.append(states)
        batch_actions.append(actions)
        batch_rtgs.append(rtgs)
        batch_timesteps.append(timesteps)
        batch_valid.append(valid)

    return (
        np.stack(batch_states),
        np.stack(batch_actions),
        np.stack(batch_rtgs),
        np.stack(batch_timesteps),
        np.stack(batch_valid),
    )


def sample_boundary_batch(
    *,
    all_trajectories: list[dict],
    all_rtgs: list[np.ndarray],
    boundary_indices: list[tuple[int, int]],
    boundary_magnitudes: np.ndarray,
    K: int,
    batch_size: int,
    rng: np.random.Generator,
    sampling: str,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Sample a batch of boundary windows.

    Returns
    -------
    states, actions, rtgs, timesteps : (B, K, ...) context tensors
    final_targets : (B, 3) float
        The full applied action at the boundary timestep. All three channels
        are targets; channel weights are applied inside the loss.
    """
    if sampling == "uniform":
        sampled_positions = rng.integers(
            0, len(boundary_indices), size=batch_size)
    elif sampling == "magnitude":
        probs = boundary_magnitudes / boundary_magnitudes.sum()
        sampled_positions = rng.choice(
            len(boundary_indices), size=batch_size, replace=True, p=probs)
    else:
        raise ValueError(f"Unknown boundary sampling mode: {sampling}")

    batch_states = []
    batch_actions = []
    batch_rtgs = []
    batch_timesteps = []
    final_targets = []

    for pool_position in sampled_positions:
        traj_idx, boundary_t = boundary_indices[int(pool_position)]
        traj = all_trajectories[traj_idx]
        rtg = all_rtgs[traj_idx]

        end = boundary_t + 1
        start = max(0, end - K)

        states = np.asarray(traj["observations"][start:end], dtype=np.float32)
        actions = np.asarray(traj["actions"][start:end], dtype=np.float32)
        rtgs = np.asarray(rtg[start:end], dtype=np.float32)
        timesteps = np.arange(start, end, dtype=np.int64)

        states, actions, rtgs, timesteps, _valid = left_pad_window(
            states=states,
            actions=actions,
            rtgs=rtgs,
            timesteps=timesteps,
            K=K,
        )

        batch_states.append(states)
        batch_actions.append(actions)
        batch_rtgs.append(rtgs)
        batch_timesteps.append(timesteps)
        # Full 3-channel target at the boundary timestep.
        final_targets.append(actions[-1, :].copy())

    return (
        np.stack(batch_states),
        np.stack(batch_actions),
        np.stack(batch_rtgs),
        np.stack(batch_timesteps),
        np.asarray(final_targets, dtype=np.float32),
    )


# =============================================================================
# Post-training loop
# =============================================================================

def posttrain_dt_v3(
    model: ContinuousDecisionTransformer,
    *,
    performance_pool: list[dict],
    performance_rtgs: list[np.ndarray],
    all_trajectories: list[dict],
    all_rtgs: list[np.ndarray],
    boundary_indices: list[tuple[int, int]],
    boundary_magnitudes: np.ndarray,
    epochs: int,
    steps_per_epoch: int,
    perf_batch_size: int,
    boundary_batch_size: int,
    lr: float,
    device: str,
    seed: int,
    lambda_boundary: float,
    boundary_sampling: str,
) -> list[dict[str, float]]:
    if lambda_boundary < 0:
        raise ValueError("lambda_boundary must be >= 0")

    rng = np.random.default_rng(seed)
    K = model.context_len

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=lr,
        weight_decay=1e-4,
    )

    channel_weights = torch.tensor(
        BOUNDARY_CHANNEL_WEIGHTS, dtype=torch.float32, device=device,
    ).view(1, 3)

    model.train()
    history: list[dict[str, float]] = []

    for epoch in range(1, epochs + 1):
        total_loss_sum = 0.0
        perf_loss_sum = 0.0
        boundary_loss_sum = 0.0

        for _ in range(steps_per_epoch):
            (
                perf_states_np,
                perf_actions_np,
                perf_rtgs_np,
                perf_ts_np,
                perf_valid_np,
            ) = sample_performance_batch(
                trajectories=performance_pool,
                rtgs_all=performance_rtgs,
                K=K,
                batch_size=perf_batch_size,
                rng=rng,
            )

            perf_states = torch.as_tensor(perf_states_np, dtype=torch.float32, device=device)
            perf_actions = torch.as_tensor(perf_actions_np, dtype=torch.float32, device=device)
            perf_rtgs = torch.as_tensor(perf_rtgs_np, dtype=torch.float32, device=device)[..., None]
            perf_ts = torch.as_tensor(perf_ts_np, dtype=torch.int64, device=device)
            perf_valid = torch.as_tensor(perf_valid_np, dtype=torch.float32, device=device)

            perf_pred = model(perf_states, perf_actions, perf_rtgs, perf_ts)

            perf_per_step_mse = ((perf_pred - perf_actions) ** 2).mean(dim=-1)
            perf_loss = (
                perf_per_step_mse * perf_valid
            ).sum() / perf_valid.sum().clamp_min(1e-8)

            if lambda_boundary > 0:
                (
                    bound_states_np,
                    bound_actions_np,
                    bound_rtgs_np,
                    bound_ts_np,
                    bound_target_np,
                ) = sample_boundary_batch(
                    all_trajectories=all_trajectories,
                    all_rtgs=all_rtgs,
                    boundary_indices=boundary_indices,
                    boundary_magnitudes=boundary_magnitudes,
                    K=K,
                    batch_size=boundary_batch_size,
                    rng=rng,
                    sampling=boundary_sampling,
                )

                bound_states = torch.as_tensor(bound_states_np, dtype=torch.float32, device=device)
                bound_actions = torch.as_tensor(bound_actions_np, dtype=torch.float32, device=device)
                bound_rtgs = torch.as_tensor(bound_rtgs_np, dtype=torch.float32, device=device)[..., None]
                bound_ts = torch.as_tensor(bound_ts_np, dtype=torch.int64, device=device)
                # (B, 3)
                bound_target = torch.as_tensor(
                    bound_target_np, dtype=torch.float32, device=device)

                bound_pred = model(bound_states, bound_actions, bound_rtgs, bound_ts)
                bound_pred_last = bound_pred[:, -1, :]  # (B, 3)

                # Channel-weighted MSE: setpoint and pump dominate so the
                # network actually learns to push them at the boundary.
                boundary_loss = (
                    channel_weights * (bound_pred_last - bound_target) ** 2
                ).mean()
            else:
                boundary_loss = torch.zeros((), dtype=torch.float32, device=device)

            total_loss = perf_loss + float(lambda_boundary) * boundary_loss

            optimizer.zero_grad()
            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5)
            optimizer.step()

            total_loss_sum += float(total_loss.item())
            perf_loss_sum += float(perf_loss.item())
            boundary_loss_sum += float(boundary_loss.item())

        record = {
            "epoch": epoch,
            "total_loss": total_loss_sum / steps_per_epoch,
            "performance_loss": perf_loss_sum / steps_per_epoch,
            "boundary_loss": boundary_loss_sum / steps_per_epoch,
        }
        history.append(record)

        print(
            f"post-train [V3 dual-pool] "
            f"epoch {epoch:02d}/{epochs}: "
            f"total={record['total_loss']:.6f}, "
            f"perf={record['performance_loss']:.6f}, "
            f"boundary={record['boundary_loss']:.6f}"
        )

    model.eval()
    return history


# =============================================================================
# Evaluation
# =============================================================================

EVAL_METRICS = (
    "co2_kg",
    "e_grid_mwh",
    "e_total_mwh",
    "sla_violation_mwh",
    "thermal_violation_hours",
    "workload_interventions",
    "safety_interventions",
    "mean_action_override",
    "wind_utilization_pct",
    "max_t_inlet",
)


def thermal_metrics_from_temps(
    temperatures: list[float],
    *,
    thermal_reference_c: float,
    cold_threshold_c: float,
) -> dict[str, float]:
    """Compute thermal metrics from a per-step T_inlet trace.

    Definitions match evaluate_lora_vs_eact_season.py so the numbers line
    up across scripts.
    """
    x = np.asarray(temperatures, dtype=np.float64)
    x = x[np.isfinite(x)]

    if x.size == 0:
        return {
            "avg_t_inlet": float("nan"),
            "p95_t_inlet": float("nan"),
            "max_t_inlet": float("nan"),
            "thermal_exposure_c_h": float("nan"),
            "thermal_event_hours": float("nan"),
            "thermal_event_rate_pct": float("nan"),
            "cold_exposure_sq_c2_h": float("nan"),
            "below_cold_threshold_hours": float("nan"),
        }

    hot_excess = np.maximum(x - thermal_reference_c, 0.0)
    true_event = x > thermal_reference_c
    cold_excess = np.maximum(cold_threshold_c - x, 0.0)

    return {
        "avg_t_inlet": float(np.mean(x)),
        "p95_t_inlet": float(np.percentile(x, 95)),
        "max_t_inlet": float(np.max(x)),
        "thermal_exposure_c_h": float(hot_excess.sum()),
        "thermal_event_hours": float(true_event.sum()),
        "thermal_event_rate_pct": float(100.0 * true_event.mean()),
        "cold_exposure_sq_c2_h": float(np.square(cold_excess).sum()),
        "below_cold_threshold_hours": float((x < cold_threshold_c).sum()),
    }


def make_env(
    country: str,
    seed: int,
    forecast_mode: str,
    forecast_dir: str | None,
):
    return CoastalDCContinuousEnv(
        country=country,
        seed=seed,
        forecast_mode=forecast_mode,
        forecast_dir=forecast_dir,
    )


def _collect_record(
    traj: dict,
    *,
    temperatures: list[float],
    thermal_reference_c: float,
    cold_threshold_c: float,
    n_steps_override: int | None = None,
) -> dict[str, float]:
    metrics = traj.get("metrics", {})
    steps = max(len(traj["rewards"]), 1)
    if n_steps_override is not None:
        steps = max(int(n_steps_override), 1)

    record = {"return": float(traj["episode_return"])}
    for key in EVAL_METRICS:
        try:
            record[key] = float(metrics.get(key, np.nan))
        except (TypeError, ValueError):
            record[key] = np.nan

    # Overwrite max_t_inlet with the true trace max so it always matches
    # the thermal block below, regardless of what metrics reports.
    thermal = thermal_metrics_from_temps(
        temperatures,
        thermal_reference_c=thermal_reference_c,
        cold_threshold_c=cold_threshold_c,
    )
    record.update(thermal)

    record["workload_intervention_rate_pct"] = (
        100.0 * record["workload_interventions"] / steps
    )
    record["safety_intervention_rate_pct"] = (
        100.0 * record["safety_interventions"] / steps
    )
    return record


def _run_episode_with_temp_trace(
    *,
    env,
    controller,
    seed: int,
    start_hour: int | None,
) -> tuple[dict, list[float]]:
    """Run one episode, returning a run_episode-like dict plus a T_inlet trace.

    Mirrors run_episode's reward accounting but also records per-step inlet
    temperature so thermal metrics can be computed afterwards.
    """
    options = (
        None if start_hour is None
        else {"start_hour": int(start_hour)}
    )

    obs, info = env.reset(seed=seed, options=options)
    if hasattr(controller, "reset"):
        controller.reset()

    done = False
    rewards: list[float] = []
    temperatures: list[float] = []

    while not done:
        action = np.asarray(controller.act(obs, info), dtype=np.float32)

        next_obs, reward, terminated, truncated, next_info = env.step(action)

        applied = np.asarray(
            next_info.get("applied_action", action),
            dtype=np.float32,
        )

        # DT / EACT controllers don't define these; the guard keeps it safe.
        if hasattr(controller, "update_applied_action"):
            controller.update_applied_action(applied)
        if hasattr(controller, "update_reward"):
            controller.update_reward(reward)

        rewards.append(float(reward))
        temperatures.append(float(next_info.get("t_inlet", np.nan)))

        obs = next_obs
        info = next_info
        done = bool(terminated or truncated)

    summary = env.episode_summary()
    traj = {
        "rewards": np.asarray(rewards, dtype=np.float64),
        "episode_return": float(np.sum(rewards)),
        "metrics": dict(summary),
    }
    return traj, temperatures


def evaluate_dt_checkpoint(
    *,
    model_path: str,
    country: str,
    target_return: float,
    context_len: int,
    return_scale: float,
    forecast_mode: str,
    forecast_dir: str | None,
    device: str,
    start_hours: list[int],
    base_seed: int,
    thermal_reference_c: float,
    cold_threshold_c: float,
) -> list[dict[str, float]]:
    records = []

    for ep, start_hour in enumerate(start_hours):
        seed = base_seed + ep

        env = make_env(country, seed, forecast_mode, forecast_dir)
        obs_dim = int(env.observation_space.shape[0])

        controller = DecisionTransformerController(
            obs_dim=obs_dim,
            target_return=target_return,
            context_len=context_len,
            model_path=model_path,
            device=device,
            return_scale=return_scale,
        )

        traj, temperatures = _run_episode_with_temp_trace(
            env=env,
            controller=controller,
            seed=seed,
            start_hour=start_hour,
        )

        records.append(
            _collect_record(
                traj,
                temperatures=temperatures,
                thermal_reference_c=thermal_reference_c,
                cold_threshold_c=cold_threshold_c,
            )
        )

    return records


def evaluate_eact_checkpoint(
    *,
    country: str,
    forecast_dir: str | None,
    forecast_mode: str,
    horizon: int,
    maxiter: int,
    control_block_hours: int,
    confidence: float,
    gamma: float,
    start_hours: list[int],
    base_seed: int,
    device: str,
    thermal_reference_c: float,
    cold_threshold_c: float,
) -> list[dict[str, float]]:
    """Run EACT-MPC on exactly the same (seed, start_hour) episodes."""
    records = []

    for ep, start_hour in enumerate(start_hours):
        seed = base_seed + ep

        env = make_env(country, seed, forecast_mode, forecast_dir)

        controller = EACTMPCController(
            env,
            forecast_dir=forecast_dir,
            horizon=horizon,
            maxiter=maxiter,
            control_block_hours=control_block_hours,
            gamma=gamma,
            confidence=confidence,
        )

        traj, temperatures = _run_episode_with_temp_trace(
            env=env,
            controller=controller,
            seed=seed,
            start_hour=start_hour,
        )

        records.append(
            _collect_record(
                traj,
                temperatures=temperatures,
                thermal_reference_c=thermal_reference_c,
                cold_threshold_c=cold_threshold_c,
            )
        )

    return records


def _rows_for_table() -> tuple:
    return (
        ("Return", "return", 3),
        ("CO2 (kg)", "co2_kg", 2),
        ("Grid energy (MWh)", "e_grid_mwh", 3),
        ("Total energy (MWh)", "e_total_mwh", 3),
        (
            "Workload intervention (%)",
            "workload_intervention_rate_pct",
            3,
        ),
        (
            "Safety intervention (%)",
            "safety_intervention_rate_pct",
            3,
        ),
        ("Mean action override", "mean_action_override", 5),
        ("Wind utilization (%)", "wind_utilization_pct", 3),
        ("Avg inlet temp (C)", "avg_t_inlet", 3),
        ("P95 inlet temp (C)", "p95_t_inlet", 3),
        ("Max inlet temp (C)", "max_t_inlet", 3),
        ("Thermal exposure (C*h)", "thermal_exposure_c_h", 3),
        ("Thermal event hours (>27C)", "thermal_event_hours", 3),
        ("Thermal event rate (%)", "thermal_event_rate_pct", 3),
        ("Cold exposure (C^2*h)", "cold_exposure_sq_c2_h", 4),
        ("Below cold threshold (h)", "below_cold_threshold_hours", 2),
        ("SLA violation (MWh)", "sla_violation_mwh", 4),
    )


def print_three_way_comparison(
    pre: list[dict[str, float]],
    post: list[dict[str, float]],
    eact: list[dict[str, float]] | None,
) -> None:
    rows = _rows_for_table()

    print("\n" + "=" * 116)
    print("Matched evaluation: Pre-DT vs Post-DT (V3) vs EACT-MPC on the same episodes")
    print("=" * 116)

    if eact:
        print(
            f"{'Metric':34s} "
            f"{'Pre-DT':>16s} "
            f"{'Post-DT':>16s} "
            f"{'EACT':>16s} "
            f"{'Post-Pre':>16s} "
            f"{'Post-EACT':>16s}"
        )
    else:
        print(
            f"{'Metric':34s} "
            f"{'Pre-DT':>16s} "
            f"{'Post-DT':>16s} "
            f"{'Post-Pre':>16s}"
        )

    for label, key, digits in rows:
        pre_values = np.asarray([r[key] for r in pre], dtype=np.float64)
        post_values = np.asarray([r[key] for r in post], dtype=np.float64)

        pre_mean = float(np.nanmean(pre_values))
        post_mean = float(np.nanmean(post_values))
        delta_pp = post_mean - pre_mean

        if eact:
            eact_values = np.asarray([r[key] for r in eact], dtype=np.float64)
            eact_mean = float(np.nanmean(eact_values))
            delta_pe = post_mean - eact_mean
            print(
                f"{label:34s} "
                f"{pre_mean:>16.{digits}f} "
                f"{post_mean:>16.{digits}f} "
                f"{eact_mean:>16.{digits}f} "
                f"{delta_pp:>+16.{digits}f} "
                f"{delta_pe:>+16.{digits}f}"
            )
        else:
            print(
                f"{label:34s} "
                f"{pre_mean:>16.{digits}f} "
                f"{post_mean:>16.{digits}f} "
                f"{delta_pp:>+16.{digits}f}"
            )

    pre_r = np.asarray([r["return"] for r in pre], dtype=np.float64)
    post_r = np.asarray([r["return"] for r in post], dtype=np.float64)

    print("-" * 116)
    gaps_pp = post_r - pre_r
    print(
        "Post vs Pre-DT return win rate : "
        f"{100.0 * np.mean(gaps_pp > 0):.1f}%"
    )
    print(f"Mean Post-Pre return gap      : {gaps_pp.mean():+.3f}")
    print(f"Median Post-Pre return gap    : {np.median(gaps_pp):+.3f}")

    if eact:
        eact_r = np.asarray([r["return"] for r in eact], dtype=np.float64)
        gaps_pe = post_r - eact_r
        print(
            "Post-DT vs EACT return win rate: "
            f"{100.0 * np.mean(gaps_pe > 0):.1f}%"
        )
        print(f"Mean Post-EACT return gap     : {gaps_pe.mean():+.3f}")
        print(f"Median Post-EACT return gap   : {np.median(gaps_pe):+.3f}")
        print(
            "EACT return reference          : "
            f"{eact_r.mean():.3f} (mean), "
            f"{eact_r.min():.3f} (min), "
            f"{eact_r.max():.3f} (max)"
        )

    print("=" * 116)


# =============================================================================
# Main
# =============================================================================

def main() -> None:
    parser = argparse.ArgumentParser()

    parser.add_argument("--country", default="JPN")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--traj-path", default=None)
    parser.add_argument("--pretrained-path", default=None)
    parser.add_argument("--output-path", default=None)

    parser.add_argument("--context-len", type=int, default=24)
    parser.add_argument("--embed-dim", type=int, default=128)
    parser.add_argument("--n-layers", type=int, default=3)
    parser.add_argument("--n-heads", type=int, default=4)
    parser.add_argument("--dropout", type=float, default=0.1)

    parser.add_argument("--elite-percentile", type=float, default=75.0)
    parser.add_argument("--correction-eps", type=float, default=1e-6)
    parser.add_argument(
        "--thermal-boundary",
        type=float,
        default=26.5,
        help=(
            "T_inlet >= this value is treated as a thermal boundary sample. "
            "Should be slightly below the 27 C constraint. Default 26.5 C."
        ),
    )
    parser.add_argument(
        "--boundary-sampling",
        choices=("uniform", "magnitude"),
        default="uniform",
    )
    parser.add_argument("--lambda-boundary", type=float, default=1.0)

    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--steps-per-epoch", type=int, default=300)
    parser.add_argument("--perf-batch-size", type=int, default=64)
    parser.add_argument("--boundary-batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-5)
    parser.add_argument("--return-scale", type=float, default=100.0)

    parser.add_argument("--target-percentile", type=float, default=90.0)
    parser.add_argument(
        "--forecast-mode",
        choices=("oracle", "causal", "none"),
        default="causal",
    )
    parser.add_argument("--forecast-dir", default=None)

    # Evaluation schedule: four seasonal typical weeks by default.
    parser.add_argument(
        "--seasonal-timestamps",
        nargs="*",
        default=list(DEFAULT_SEASONAL_TIMESTAMPS),
        help=(
            "Fixed start timestamps for the seasonal typical-week evaluation. "
            "Default: 2025-01-15 / 04-15 / 07-15 / 10-15 at 00:00 UTC."
        ),
    )
    parser.add_argument(
        "--no-seasonal",
        action="store_true",
        help=(
            "Fall back to the legacy random-start evaluation using "
            "--eval-episodes random episodes."
        ),
    )
    parser.add_argument(
        "--eval-episodes",
        type=int,
        default=20,
        help="Only used when --no-seasonal is set (legacy random-start mode).",
    )
    parser.add_argument("--base-seed", type=int, default=9000)

    parser.add_argument(
        "--thermal-reference",
        type=float,
        default=27.0,
        help="T > this value is a true thermal event. Default 27.0 C.",
    )
    parser.add_argument(
        "--cold-threshold",
        type=float,
        default=25.5,
        help=(
            "T below this value gets a cold-exposure penalty in the reported "
            "metrics. Default 25.5 C."
        ),
    )

    # EACT reference options.
    parser.add_argument(
        "--compare-eact",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Also evaluate EACT-MPC on the same episodes (default: on).",
    )
    parser.add_argument("--eact-horizon", type=int, default=24)
    parser.add_argument("--eact-maxiter", type=int, default=20)
    parser.add_argument("--eact-block-hours", type=int, default=6)
    parser.add_argument("--eact-confidence", type=float, default=0.90)
    parser.add_argument("--eact-gamma", type=float, default=0.995)

    parser.add_argument("--skip-eval", action="store_true")

    args = parser.parse_args()

    for name, value in (
        ("elite-percentile", args.elite_percentile),
        ("target-percentile", args.target_percentile),
    ):
        if not (0.0 <= value <= 100.0):
            parser.error(f"--{name} must be between 0 and 100")

    if args.return_scale <= 0:
        parser.error("--return-scale must be > 0")
    if args.lr <= 0:
        parser.error("--lr must be > 0")
    if args.lambda_boundary < 0:
        parser.error("--lambda-boundary must be >= 0")
    if args.perf_batch_size <= 0:
        parser.error("--perf-batch-size must be > 0")
    if args.boundary_batch_size <= 0:
        parser.error("--boundary-batch-size must be > 0")
    if args.no_seasonal and args.eval_episodes <= 0:
        parser.error(
            "--no-seasonal requires --eval-episodes > 0 "
            "(otherwise there is nothing to evaluate)."
        )
    if args.cold_threshold >= args.thermal_reference:
        parser.error("--cold-threshold must be below --thermal-reference")
    if args.thermal_boundary >= args.thermal_reference:
        parser.error("--thermal-boundary must be below --thermal-reference")

    set_seed(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # ------------------------------------------------------------------
    # Resolve input paths.
    #
    # Primary trajectory (optionally overridden by --traj-path). When the
    # default resolution is used, an OPTIONAL seasonal EACT trajectory file
    # (data/trajectories/eact_<COUNTRY>_seasonal.npz) is appended if present.
    # ------------------------------------------------------------------
    if args.traj_path is not None:
        traj_paths: list[str] = [os.path.abspath(args.traj_path)]
    else:
        primary_traj_path, seasonal_traj_path = _resolve_default_traj_paths(
            args.country
        )
        traj_paths = [primary_traj_path]
        if seasonal_traj_path is not None:
            traj_paths.append(seasonal_traj_path)
            print(
                f"[info] Including seasonal EACT trajectories: "
                f"{seasonal_traj_path}"
            )

    # Keep a convenience alias for logging / history (primary file only).
    traj_path = traj_paths[0]

    pretrained_path = (
        os.path.abspath(args.pretrained_path)
        if args.pretrained_path is not None
        else _resolve_default_pretrained_path(args.country)
    )

    for path in traj_paths:
        if not os.path.isfile(path):
            raise FileNotFoundError(
                f"EACT trajectory file not found:\n  {path}"
            )

    if not os.path.isfile(pretrained_path):
        raise FileNotFoundError(
            f"Pretrained DT checkpoint not found:\n  {pretrained_path}\n"
            f"Expected location: {DEFAULT_PRETRAINED_DIR}\n"
            "Run train_decision_transformer.py first, or pass "
            "--pretrained-path explicitly."
        )

    # ------------------------------------------------------------------
    # Load trajectories and build pools.
    #
    # use_applied_actions=True is REQUIRED: the stored reward and next state
    # were produced by the applied action, so DT must be trained on the same
    # action that generated them. Training on requested_actions would break
    # the action-state causal chain.
    # ------------------------------------------------------------------
    all_trajs = load_trajectories(traj_paths, use_applied_actions=True)
    validate_trajectory_schema(all_trajs)

    all_returns = trajectory_returns(all_trajs)
    target_return = float(np.percentile(all_returns, args.target_percentile))

    performance_pool, elite_threshold = build_performance_pool(
        all_trajs, args.elite_percentile)

    (
        boundary_indices,
        boundary_magnitudes,
        boundary_targets,
        is_thermal,
    ) = build_boundary_pool(
        all_trajs,
        args.correction_eps,
        thermal_threshold=args.thermal_boundary,
    )

    pool_info = build_pool_info(
        all_trajs,
        performance_pool,
        args.elite_percentile,
        elite_threshold,
        boundary_indices,
        boundary_magnitudes,
        is_thermal,
        args.thermal_boundary,
    )
    print_pool_diagnostics(pool_info)

    all_rtgs = compute_rtgs(all_trajs, args.return_scale)
    performance_rtgs = compute_rtgs(performance_pool, args.return_scale)

    obs_dim = int(all_trajs[0]["observations"].shape[1])

    # ------------------------------------------------------------------
    # Build and load pretrained model.
    # ------------------------------------------------------------------
    model = ContinuousDecisionTransformer(
        obs_dim=obs_dim,
        context_len=args.context_len,
        embed_dim=args.embed_dim,
        n_layers=args.n_layers,
        n_heads=args.n_heads,
        dropout=args.dropout,
    ).to(device)

    state_dict = torch.load(pretrained_path, map_location=device)
    model.load_state_dict(state_dict)

    print("V3 post-training configuration")
    print("-" * 88)
    print(f"device                    : {device}")
    print(f"country                   : {args.country}")
    print(f"all trajectories          : {len(all_trajs)}")
    print(f"performance pool          : {len(performance_pool)} trajectories")
    print(f"constraint boundary pool  : {len(boundary_indices)} timesteps")
    print(f"  thermal boundary        : {pool_info.n_thermal_boundary_timesteps}")
    print(f"  action correction       : {pool_info.n_action_correction_timesteps}")
    print(f"elite percentile          : P{args.elite_percentile:g}")
    print(f"lambda boundary           : {args.lambda_boundary:g}")
    print(f"boundary sampling         : {args.boundary_sampling}")
    print(f"thermal boundary threshold: {args.thermal_boundary:g} C")
    print(f"boundary channel weights  : {BOUNDARY_CHANNEL_WEIGHTS}")
    print(f"training action source    : applied_actions")
    print(f"performance batch size    : {args.perf_batch_size}")
    print(f"boundary batch size       : {args.boundary_batch_size}")
    print(f"epochs                    : {args.epochs}")
    print(f"steps/epoch               : {args.steps_per_epoch}")
    print(f"learning rate             : {args.lr:g}")
    print(
        f"target return             : {target_return:.2f} "
        f"(P{args.target_percentile:g}, all EACT data)"
    )
    print(
        "trajectory file(s)        : "
        + ", ".join(traj_paths)
    )
    print(f"pretrained checkpoint     : {pretrained_path}")
    print("-" * 88)

    # ------------------------------------------------------------------
    # Post-train.
    # ------------------------------------------------------------------
    history = posttrain_dt_v3(
        model,
        performance_pool=performance_pool,
        performance_rtgs=performance_rtgs,
        all_trajectories=all_trajs,
        all_rtgs=all_rtgs,
        boundary_indices=boundary_indices,
        boundary_magnitudes=boundary_magnitudes,
        epochs=args.epochs,
        steps_per_epoch=args.steps_per_epoch,
        perf_batch_size=args.perf_batch_size,
        boundary_batch_size=args.boundary_batch_size,
        lr=args.lr,
        device=device,
        seed=args.seed,
        lambda_boundary=args.lambda_boundary,
        boundary_sampling=args.boundary_sampling,
    )

    os.makedirs(DEFAULT_OUTPUT_DIR, exist_ok=True)

    if args.output_path:
        output_path = os.path.abspath(args.output_path)
        parent = os.path.dirname(output_path)
        if parent:
            os.makedirs(parent, exist_ok=True)
    else:
        output_path = os.path.join(
            DEFAULT_OUTPUT_DIR,
            (
                f"dt_{args.country}_post_v3_dualpool_"
                f"eact{len(all_trajs)}_seed{args.seed}.pt"
            ),
        )

    torch.save(model.state_dict(), output_path)

    history_path = os.path.splitext(output_path)[0] + "_history.json"
    with open(history_path, "w", encoding="utf-8") as f:
        json.dump(
            {
                "country": args.country,
                "trajectory_path": traj_path,
                "trajectory_paths": list(traj_paths),
                "pretrained_path": pretrained_path,
                "output_path": output_path,
                "target_percentile": args.target_percentile,
                "target_return": target_return,
                "return_scale": args.return_scale,
                "pool_info": asdict(pool_info),
                "lambda_boundary": args.lambda_boundary,
                "boundary_sampling": args.boundary_sampling,
                "boundary_channel_weights": list(BOUNDARY_CHANNEL_WEIGHTS),
                "correction_eps": args.correction_eps,
                "thermal_boundary": args.thermal_boundary,
                "training_action_source": "applied_actions",
                "epochs": args.epochs,
                "steps_per_epoch": args.steps_per_epoch,
                "perf_batch_size": args.perf_batch_size,
                "boundary_batch_size": args.boundary_batch_size,
                "lr": args.lr,
                "history": history,
            },
            f,
            ensure_ascii=False,
            indent=2,
        )

    print(f"\nV3 post-trained model saved -> {output_path}")
    print(f"Training history saved       -> {history_path}")

    # ------------------------------------------------------------------
    # Resolve forecast dir.
    # ------------------------------------------------------------------
    if args.forecast_dir is None:
        forecast_dir = (
            DEFAULT_FORECAST_DIR
            if args.forecast_mode == "causal"
            else None
        )
    else:
        forecast_dir = os.path.abspath(args.forecast_dir)

    if args.forecast_mode == "causal":
        expected_forecast = os.path.join(
            forecast_dir, f"causal_ridge_{args.country}.npz")
        if not os.path.isfile(expected_forecast):
            raise FileNotFoundError(
                "Causal forecast model not found:\n"
                f"  {expected_forecast}"
            )

    if args.skip_eval:
        return

    # ------------------------------------------------------------------
    # Resolve evaluation start hours (seasonal by default).
    # ------------------------------------------------------------------
    if args.no_seasonal:
        start_hours = []
        for ep in range(args.eval_episodes):
            seed = args.base_seed + ep
            probe_env = make_env(
                args.country, seed, args.forecast_mode, forecast_dir)
            _, info = probe_env.reset(seed=seed)
            start_hours.append(int(info["data_hour"]))
        print("\nLegacy random-start evaluation seeds:")
        print([args.base_seed + i for i in range(args.eval_episodes)])
    else:
        probe_env = make_env(
            args.country, args.base_seed, args.forecast_mode, forecast_dir)
        causal_min_history = (
            int(probe_env._causal_min_history)
            if args.forecast_mode == "causal"
            else 0
        )
        start_hours = resolve_seasonal_start_hours(
            probe_env.data,
            args.seasonal_timestamps,
            episode_hours=int(probe_env.episode_hours),
            min_start=causal_min_history,
        )
        print("\nMatched evaluation seasonal typical weeks:")
        for label, ts, sh in zip(
            SEASONAL_LABELS, args.seasonal_timestamps, start_hours
        ):
            print(f"  - {label:8s} | {ts} | start_hour={sh}")

    # ------------------------------------------------------------------
    # Pre-DT vs Post-DT.
    # ------------------------------------------------------------------
    pre_results = evaluate_dt_checkpoint(
        model_path=pretrained_path,
        country=args.country,
        target_return=target_return,
        context_len=args.context_len,
        return_scale=args.return_scale,
        forecast_mode=args.forecast_mode,
        forecast_dir=forecast_dir,
        device=device,
        start_hours=start_hours,
        base_seed=args.base_seed,
        thermal_reference_c=args.thermal_reference,
        cold_threshold_c=args.cold_threshold,
    )

    post_results = evaluate_dt_checkpoint(
        model_path=output_path,
        country=args.country,
        target_return=target_return,
        context_len=args.context_len,
        return_scale=args.return_scale,
        forecast_mode=args.forecast_mode,
        forecast_dir=forecast_dir,
        device=device,
        start_hours=start_hours,
        base_seed=args.base_seed,
        thermal_reference_c=args.thermal_reference,
        cold_threshold_c=args.cold_threshold,
    )

    # ------------------------------------------------------------------
    # EACT reference (same episodes).
    # ------------------------------------------------------------------
    eact_results = None
    if args.compare_eact:
        if args.forecast_mode != "causal":
            print(
                "[warn] EACT comparison requires --forecast-mode causal; "
                "skipping EACT evaluation."
            )
        else:
            print("\nRunning EACT-MPC reference on the same episodes ...")
            eact_results = evaluate_eact_checkpoint(
                country=args.country,
                forecast_dir=forecast_dir,
                forecast_mode=args.forecast_mode,
                horizon=args.eact_horizon,
                maxiter=args.eact_maxiter,
                control_block_hours=args.eact_block_hours,
                confidence=args.eact_confidence,
                gamma=args.eact_gamma,
                start_hours=start_hours,
                base_seed=args.base_seed,
                device=device,
                thermal_reference_c=args.thermal_reference,
                cold_threshold_c=args.cold_threshold,
            )

    print_three_way_comparison(pre_results, post_results, eact_results)


if __name__ == "__main__":
    main()