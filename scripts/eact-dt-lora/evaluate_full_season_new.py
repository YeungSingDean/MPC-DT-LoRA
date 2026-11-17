"""Paired evaluation: LoRA-SAC V6 vs EACT MPC vs Pre-DT vs Post-DT vs SAC.

Runs five (or more) controllers on the exact same episodes:

    eact     : EACT-MPC expert reference
    pre_dt   : pretrained Decision Transformer checkpoint
    post_dt  : V3 post-trained Decision Transformer checkpoint
    lora     : LoRA-SAC V6 RL-finetuned actor
    sac      : SAC baseline from scripts/RL/train_sac.py (MLP actor
               + twin Q + optional Lagrange cost critics)

Episode schedule
----------------
By default this script evaluates ONLY the four fixed seasonal "typical week"
episodes:

    2025-01-15 / 04-15 / 07-15 / 10-15 at 00:00 UTC

pinned via env.reset(options={"start_hour": ...}). Each controller runs on
the SAME (seed, start_hour) pair for every episode, so per-episode
differences between algorithms are well-defined.

A legacy random-start mode is still available: pass --episodes N > 0 to
also append N random-start episodes with seeds base_seed..base_seed+N-1.

Alignment with posttrain_decision_transformer_v3.py
---------------------------------------------------
This script is deliberately aligned with
scripts/eact-dt-lora/posttrain_decision_transformer_v3.py so that the EACT
reference numbers reported here match the EACT reference numbers reported
there on the same episodes:

  * Environment construction passes exactly the same arguments
    (country, seed, forecast_mode, forecast_dir) and lets
    CoastalDCContinuousEnv use its own defaults for reward weights and
    cooling configuration.

  * EACTMPCController is constructed with exactly the same keyword set:
        forecast_dir, horizon, maxiter, control_block_hours, gamma, confidence
    and nothing else.

  * Seasonal episode seeds follow base_seed + i for i in range(len(seasonal)),
    matching posttrain's base_seed + ep.

If you change any of these, the EACT reference in this script will silently
diverge from the one printed by posttrain.

Paths
-----
This script lives in scripts/eact-dt-lora/. Therefore:
    ROOT       = project root (two levels up)
    SCRIPT_DIR = scripts/eact-dt-lora/

Outputs go to <SCRIPT_DIR>/full_evaluate_results/ (kept separate from the
original paired-eval results directory to avoid mixing runs).

Checkpoint auto-resolution
--------------------------
If --pre-dt-checkpoint is omitted, the script searches
    <SCRIPT_DIR>/results/models/ and <ROOT>/results/models/
for dt_<COUNTRY>_seed*.pt (newest by mtime).

If --post-dt-checkpoint is omitted, the script searches the same dirs for
dt_<COUNTRY>_post_v3_dualpool_eact*_seed*.pt (falls back to mpc*).

If --checkpoint is omitted while "lora" is in --controllers, the script
searches <SCRIPT_DIR>/results/lora_thermal_rl/<COUNTRY>_*/best_lora_thermal_dt.pt.

If --sac-checkpoint is omitted while "sac" is in --controllers, the script
searches scripts/RL/results/<COUNTRY>_v3_sac_lagrange_season/best_sac.pt
and the vanilla variant.

DT target return
----------------
Both pre_dt and post_dt are conditioned on a target return. If
--dt-target-return is not passed, the script loads the expert trajectory at
--expert-trajectories (default data/trajectories/eact_<COUNTRY>.npz) and uses
the --dt-target-percentile (default 90) percentile of per-episode returns.

Outputs
-------
episode_results.csv : one row per (algorithm, episode)
paired_results.csv  : one row per episode, EACT minus each non-EACT algorithm
step_traces.csv     : one row per (algorithm, episode, step), with the
                      full observation vector and both requested and applied
                      actions expanded into obs_0..obs_{N-1},
                      action_requested_0..2, action_applied_0..2, plus
                      per-step reward, t_inlet, co2_kg, e_grid_mwh,
                      e_total_mwh.
summary.json        : aggregate + per-controller + paired + per-season stats
"""

from __future__ import annotations

import argparse
import csv
import glob
import json
import os
import sys

import numpy as np
import pandas as pd
import torch
import torch.nn as nn


# ---------------------------------------------------------------------------
# Path resolution
# ---------------------------------------------------------------------------

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, "..", ".."))
sys.path.insert(0, ROOT)


from coastaldc_env import CoastalDCContinuousEnv  # noqa: E402
from coastaldc_env.rollout import load_trajectories  # noqa: E402

from controllers.decision_transformer_continuous import (  # noqa: E402
    DecisionTransformerController,
)
from controllers.lora_joint_forecast_v6 import (  # noqa: E402
    FULL_FORECAST_DIM,
    ForecastJointLoRAActor,
    ForecastJointLoRADTController,
)
from controllers.eact_mpc import (  # noqa: E402
    EACTMPCController,
    EACTNoBiasMPCController,
    EACTNoInterventionMPCController,
    NominalCausalMPCController,
    OracleConstrainedMPCController,
    StaticRobustMPCController,
)
from controllers.no_control import NoControlController  # noqa: E402
from controllers.rule_based import RuleBasedController  # noqa: E402


DEFAULT_FORECAST_DIR = os.path.join(
    ROOT, "results", "causal_forecasts_v3_gated_bias",
)
DEFAULT_MODEL_DIRS = (
    os.path.join(SCRIPT_DIR, "results", "models"),
    os.path.join(ROOT, "results", "models"),
)
DEFAULT_TRAJ_DIR = os.path.join(ROOT, "data", "trajectories")
DEFAULT_LORA_ROOT = os.path.join(SCRIPT_DIR, "results", "lora_thermal_rl")

# SAC checkpoints live next to scripts/RL/train_sac.py, whose SCRIPT_DIR is
# scripts/RL/, so its results root is scripts/RL/results/.
RL_SCRIPT_DIR = os.path.abspath(os.path.join(SCRIPT_DIR, "..", "RL"))
DEFAULT_SAC_ROOT = os.path.join(RL_SCRIPT_DIR, "results")

# NOTE: output root is intentionally different from the original paired
# evaluation script to avoid mixing results.
DEFAULT_OUTPUT_ROOT = os.path.join(SCRIPT_DIR, "full_evaluate_results")

DEFAULT_SEASONAL_TIMESTAMPS = (
    "2025-01-15 00:00:00",
    "2025-04-15 00:00:00",
    "2025-07-15 00:00:00",
    "2025-10-15 00:00:00",
)
SEASONAL_LABELS = ("winter", "spring", "summer", "autumn")

EACT_CONTROLLER_NAMES = (
    "no_control",
    "rule_based",
    "nominal",
    "static",
    "eact",
    "eact_no_bias",
    "eact_no_intervention",
    "oracle",
)
DT_CONTROLLER_NAMES = ("pre_dt", "post_dt")
ALL_CONTROLLER_NAMES = (
    EACT_CONTROLLER_NAMES + ("lora", "sac") + DT_CONTROLLER_NAMES
)


# =============================================================================
# SAC actor definition
# =============================================================================
# Re-declared locally so that evaluate_full_season_new.py does not have to
# import scripts/RL/train_sac.py (which would pull in extra module-level side
# effects and couples the two scripts).  The definition must remain
# bit-for-bit compatible with the one used at training time so that
# ``actor.load_state_dict(ckpt["actor_state"])`` succeeds.

LOG_STD_MIN = -20.0
LOG_STD_MAX = 2.0


class GaussianActor(nn.Module):
    """MLP Gaussian policy with tanh squashing onto a Box action space."""

    def __init__(self, obs_dim, act_dim, hidden_dim, action_low, action_high):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )
        self.mean_layer = nn.Linear(hidden_dim, act_dim)
        self.log_std_layer = nn.Linear(hidden_dim, act_dim)
        self.register_buffer(
            "action_low", torch.as_tensor(action_low, dtype=torch.float32))
        self.register_buffer(
            "action_high", torch.as_tensor(action_high, dtype=torch.float32))

    def forward(self, obs):
        h = self.net(obs)
        mean = self.mean_layer(h)
        log_std = torch.clamp(self.log_std_layer(h), LOG_STD_MIN, LOG_STD_MAX)
        return mean, log_std

    def sample(self, obs, deterministic: bool = False):
        mean, log_std = self.forward(obs)
        std = log_std.exp()
        normal = torch.distributions.Normal(mean, std)
        x = mean if deterministic else normal.rsample()
        y = torch.tanh(x)
        scale = (self.action_high - self.action_low) * 0.5
        action = self.action_low + (y + 1.0) * scale
        log_prob = normal.log_prob(x) - torch.log(1.0 - y.pow(2) + 1e-6)
        log_prob = log_prob - torch.log(scale + 1e-6)
        log_prob = log_prob.sum(dim=-1, keepdim=True)
        return action, log_prob


class SACController:
    """Thin controller wrapper around a trained SAC GaussianActor.

    Exposes the same ``act(obs, info)`` / ``reset()`` interface used by the
    evaluation driver.  Actions are deterministic (mean of the Gaussian) to
    match the deterministic seasonal evaluation used in train_sac.py.
    """

    def __init__(self, *, actor: GaussianActor, device: str,
                 deterministic: bool = True):
        self.actor = actor
        self.device = device
        self.deterministic = bool(deterministic)

    def reset(self) -> None:  # noqa: D401 (interface parity with other controllers)
        return None

    def act(self, obs, info=None) -> np.ndarray:
        with torch.no_grad():
            obs_t = torch.as_tensor(
                np.asarray(obs, dtype=np.float32), device=self.device,
            ).unsqueeze(0)
            action, _ = self.actor.sample(obs_t, deterministic=self.deterministic)
            action_np = action.squeeze(0).detach().cpu().numpy()
        return action_np.astype(np.float32)


# =============================================================================
# Checkpoint resolution
# =============================================================================

def _newest(candidates: list[str]) -> str | None:
    if not candidates:
        return None
    candidates.sort(key=os.path.getmtime, reverse=True)
    return candidates[0]


def _resolve_default_lora_checkpoint(country: str) -> str | None:
    if not os.path.isdir(DEFAULT_LORA_ROOT):
        return None
    candidates = sorted(
        glob.glob(os.path.join(DEFAULT_LORA_ROOT, f"{country}_*",
                               "best_lora_thermal_dt.pt"))
    )
    if candidates:
        return candidates[-1]
    candidates = sorted(
        glob.glob(os.path.join(DEFAULT_LORA_ROOT, "*",
                               "best_lora_thermal_dt.pt"))
    )
    if candidates:
        return candidates[-1]
    return None


def _resolve_default_sac_checkpoint(country: str) -> str | None:
    """Find best_sac.pt under scripts/RL/results/<country>_{tag}/.

    Priority: lagrange variant first, then vanilla.
    """
    if not os.path.isdir(DEFAULT_SAC_ROOT):
        return None
    candidates: list[str] = []
    for tag in ("v3_sac_lagrange_season", "v3_sac_vanilla_season"):
        path = os.path.join(DEFAULT_SAC_ROOT, f"{country}_{tag}", "best_sac.pt")
        if os.path.isfile(path):
            candidates.append(path)
    if candidates:
        return candidates[0]

    # Fallback: any best_sac.pt under matching country dirs.
    fallback = sorted(
        glob.glob(os.path.join(DEFAULT_SAC_ROOT, f"{country}_*", "best_sac.pt"))
    )
    if fallback:
        return fallback[-1]
    return None


def _resolve_dt_checkpoint(
    checkpoint_arg: str | None,
    country: str,
    kind: str,  # "pre" | "post"
) -> str | None:
    if checkpoint_arg:
        return os.path.abspath(checkpoint_arg)

    if kind == "pre":
        patterns = [f"dt_{country}_seed*.pt"]
    elif kind == "post":
        patterns = [
            f"dt_{country}_post_v3_dualpool_eact*_seed*.pt",
            f"dt_{country}_post_v3_dualpool_mpc*_seed*.pt",
        ]
    else:
        raise ValueError(f"Unknown DT kind: {kind}")

    candidates: list[str] = []
    for d in DEFAULT_MODEL_DIRS:
        if not os.path.isdir(d):
            continue
        for p in patterns:
            candidates.extend(glob.glob(os.path.join(d, p)))

    return _newest(candidates)


def _compute_target_return(
    *,
    traj_path: str,
    percentile: float,
) -> float:
    trajectories = load_trajectories([traj_path], use_applied_actions=True)
    returns = np.asarray(
        [
            float(np.asarray(t["rewards"], dtype=np.float64).sum())
            for t in trajectories
        ],
        dtype=np.float64,
    )
    return float(np.percentile(returns, percentile))


def _load_sac_bundle(checkpoint_path: str, device: str):
    """Load a SAC checkpoint produced by scripts/RL/train_sac.py."""
    ckpt = torch.load(checkpoint_path, map_location="cpu")
    required = ("actor_config", "actor_state")
    missing = [k for k in required if k not in ckpt]
    if missing:
        raise KeyError(
            f"SAC checkpoint missing fields: {missing} (found keys="
            f"{sorted(ckpt.keys())})"
        )

    actor_config = dict(ckpt["actor_config"])
    actor = GaussianActor(
        obs_dim=int(actor_config["obs_dim"]),
        act_dim=int(actor_config["act_dim"]),
        hidden_dim=int(actor_config["hidden_dim"]),
        action_low=actor_config["action_low"],
        action_high=actor_config["action_high"],
    )
    actor.load_state_dict(ckpt["actor_state"])
    actor.to(device)
    actor.eval()

    return actor, {
        "path": os.path.abspath(checkpoint_path),
        "use_lagrange": bool(ckpt.get("use_lagrange", False)),
        "actor_config": actor_config,
        "metrics": ckpt.get("metrics", {}),
    }


# =============================================================================
# Environment
# =============================================================================
# NOTE: this mirrors posttrain_decision_transformer_v3.py's make_env EXACTLY.
# Only (country, seed, forecast_mode, forecast_dir) are passed, and the
# environment uses its own defaults for reward weights and cooling config.
# =============================================================================

def make_env(
    *,
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
# Per-episode metrics
# =============================================================================

def thermal_metrics(
    temperatures: list[float],
    *,
    thermal_reference_c: float,
    event_margin_start_c: float,
    event_margin_width_c: float,
    cold_threshold_c: float,
) -> dict[str, float]:
    x = np.asarray(temperatures, dtype=np.float64)
    x = x[np.isfinite(x)]

    if len(x) == 0:
        raise ValueError("No finite inlet-temperature values were recorded.")

    hot_excess = np.maximum(x - thermal_reference_c, 0.0)
    true_event = x > thermal_reference_c
    margin_risk = np.clip(
        (x - event_margin_start_c) / event_margin_width_c, 0.0, 1.0,
    )
    cold_excess = np.maximum(cold_threshold_c - x, 0.0)

    return {
        "avg_t_inlet": float(np.mean(x)),
        "p95_t_inlet": float(np.percentile(x, 95)),
        "max_t_inlet": float(np.max(x)),
        "thermal_exposure_c_h": float(hot_excess.sum()),
        "thermal_event_hours": float(true_event.sum()),
        "thermal_event_rate_pct": float(100.0 * true_event.mean()),
        "margin_risk_sum": float(margin_risk.sum()),
        "margin_risk_mean": float(margin_risk.mean()),
        "cold_exposure_sq_c2_h": float(np.square(cold_excess).sum()),
        "below_cold_threshold_hours": float((x < cold_threshold_c).sum()),
    }


def _summary_get(summary: dict, key: str) -> float:
    if key not in summary:
        return float("nan")
    value = summary[key]
    if value is None:
        return float("nan")
    try:
        return float(value)
    except (TypeError, ValueError):
        return float("nan")


# =============================================================================
# Controller factories
# =============================================================================

def _shared_eact_kwargs(args) -> dict:
    """EACT-MPC kwargs, aligned EXACTLY with posttrain_decision_transformer_v3.py.

    posttrain's evaluate_eact_checkpoint builds EACTMPCController with:
        forecast_dir, horizon, maxiter, control_block_hours, gamma, confidence
    and nothing else. This function must return the same key set so that the
    EACT reference reported here matches the one printed by posttrain on the
    same episodes.
    """
    return dict(
        forecast_dir=args.forecast_dir,
        horizon=args.horizon,
        maxiter=args.maxiter,
        control_block_hours=args.block_hours,
        gamma=args.gamma,
        confidence=args.confidence,
    )


def build_controller_factories(
    args,
    lora_bundle,
    dt_bundles: dict[str, dict | None],
    sac_bundle=None,
) -> list[tuple[str, callable]]:
    """Return [(name, factory(env) -> controller), ...].

    dt_bundles: {"pre_dt": {...} | None, "post_dt": {...} | None}
      each bundle: {"path": str, "target_return": float,
                    "return_scale": float, "context_len": int}

    sac_bundle: (actor, info_dict) | None
    """
    factories: list[tuple[str, callable]] = []
    shared_eact = _shared_eact_kwargs(args)
    device = args.device

    for name in args.controllers:
        if name == "lora":
            if lora_bundle is None:
                raise ValueError(
                    "LoRA controller requested but no checkpoint was loaded"
                )
            actor, target_return, return_scale = lora_bundle
            deterministic = bool(args.lora_deterministic)

            def make_lora(
                env,
                _actor=actor,
                _tr=target_return,
                _rs=return_scale,
                _det=deterministic,
            ):
                return ForecastJointLoRADTController(
                    actor=_actor,
                    env=env,
                    target_return=_tr,
                    return_scale=_rs,
                    deterministic=_det,
                )

            factories.append(("lora", make_lora))

        elif name == "sac":
            if sac_bundle is None:
                raise ValueError(
                    "SAC controller requested but no checkpoint was loaded. "
                    "Pass --sac-checkpoint or train with scripts/RL/train_sac.py."
                )
            sac_actor, _info = sac_bundle
            sac_det = bool(args.sac_deterministic)

            def make_sac(env, _actor=sac_actor, _dev=device, _det=sac_det):
                return SACController(
                    actor=_actor, device=_dev, deterministic=_det,
                )

            factories.append(("sac", make_sac))

        elif name in ("pre_dt", "post_dt"):
            bundle = dt_bundles.get(name)
            if bundle is None:
                raise ValueError(
                    f"{name} requested but no checkpoint was loaded. "
                    f"Pass --{'pre' if name == 'pre_dt' else 'post'}-dt-checkpoint."
                )

            def make_dt(
                env,
                _b=bundle,
                _dev=device,
            ):
                return DecisionTransformerController(
                    obs_dim=int(env.observation_space.shape[0]),
                    target_return=float(_b["target_return"]),
                    context_len=int(_b["context_len"]),
                    model_path=_b["path"],
                    device=_dev,
                    return_scale=float(_b["return_scale"]),
                )

            factories.append((name, make_dt))

        elif name == "no_control":
            factories.append(("no_control", lambda env: NoControlController()))

        elif name == "rule_based":
            factories.append(("rule_based", lambda env: RuleBasedController()))

        elif name == "nominal":
            factories.append(
                ("nominal",
                 lambda env: NominalCausalMPCController(env, **shared_eact))
            )
        elif name == "static":
            factories.append(
                ("static",
                 lambda env: StaticRobustMPCController(env, **shared_eact))
            )
        elif name == "eact":
            factories.append(
                ("eact", lambda env: EACTMPCController(env, **shared_eact))
            )
        elif name == "eact_no_bias":
            factories.append(
                ("eact_no_bias",
                 lambda env: EACTNoBiasMPCController(env, **shared_eact))
            )
        elif name == "eact_no_intervention":
            factories.append(
                ("eact_no_intervention",
                 lambda env: EACTNoInterventionMPCController(env, **shared_eact))
            )
        elif name == "oracle":
            factories.append(
                ("oracle",
                 lambda env: OracleConstrainedMPCController(env, **shared_eact))
            )
        else:
            raise ValueError(f"Unknown controller: {name}")

    return factories


# =============================================================================
# Episode runner
# =============================================================================

def _run_episode(
    *,
    env,
    controller,
    seed: int,
    start_hour_override: int | None,
    thermal_reference_c: float,
    event_margin_start_c: float,
    event_margin_width_c: float,
    cold_threshold_c: float,
    is_lora: bool,
) -> tuple[dict, list[dict]]:
    options = (
        None if start_hour_override is None
        else {"start_hour": int(start_hour_override)}
    )

    obs, info = env.reset(seed=seed, options=options)
    if hasattr(controller, "reset"):
        controller.reset()

    start_hour = info.get("data_hour", np.nan)

    done = False
    rewards: list[float] = []
    temperatures: list[float] = []

    requested_actions = []
    applied_actions = []
    workload_residuals = []
    setpoint_residuals = []
    pump_residuals = []

    sla_step_mwh: list[float] = []
    backlog_step_mwh: list[float] = []

    step_traces: list[dict] = []
    step_idx = 0

    while not done:
        obs_snapshot = np.asarray(obs, dtype=np.float32).copy()

        requested = np.asarray(controller.act(obs, info), dtype=np.float32)

        if is_lora and hasattr(controller, "last_base_action"):
            base_action = np.asarray(
                controller.last_base_action, dtype=np.float64,
            )
            workload_residuals.append(float(requested[0] - base_action[0]))
            setpoint_residuals.append(float(requested[1] - base_action[1]))
            pump_residuals.append(float(requested[2] - base_action[2]))

        next_obs, reward, terminated, truncated, next_info = env.step(requested)

        applied = np.asarray(
            next_info.get("applied_action", requested), dtype=np.float32,
        )
        t_inlet_step = float(next_info.get("t_inlet", np.nan))

        # ---- per-step energy / carbon signals -------------------------
        # These may be either instantaneous (per-step) values or running
        # cumulative values depending on the environment.  We store what
        # the environment exposes verbatim; if a key is missing, NaN is
        # written instead of raising.
        co2_kg_step = float(next_info.get("co2_kg", np.nan))
        e_grid_step = float(next_info.get("e_grid_mwh", np.nan))
        e_total_step = float(next_info.get("e_total_mwh", np.nan))

        step_traces.append({
            "step": int(step_idx),
            "obs": obs_snapshot,
            "action_requested": requested.copy(),
            "action_applied": applied.copy(),
            "reward": float(reward),
            "t_inlet": t_inlet_step,
            "co2_kg": co2_kg_step,
            "e_grid_mwh": e_grid_step,
            "e_total_mwh": e_total_step,
        })
        step_idx += 1

        if hasattr(controller, "update_applied_action"):
            controller.update_applied_action(applied)
        if hasattr(controller, "update_reward"):
            controller.update_reward(reward)

        rewards.append(float(reward))
        temperatures.append(t_inlet_step)
        requested_actions.append(requested)
        applied_actions.append(applied)

        sla_step_mwh.append(float(next_info.get("sla_violation_mwh", np.nan)))
        backlog_step_mwh.append(float(next_info.get("backlog_mwh", np.nan)))

        obs = next_obs
        info = next_info
        done = bool(terminated or truncated)

    summary = env.episode_summary()

    req = np.asarray(requested_actions, dtype=np.float64)
    app = np.asarray(applied_actions, dtype=np.float64)

    if req.shape == app.shape:
        mean_override_direct = float(np.mean(np.abs(req - app)))
    else:
        mean_override_direct = float("nan")

    thermal = thermal_metrics(
        temperatures,
        thermal_reference_c=thermal_reference_c,
        event_margin_start_c=event_margin_start_c,
        event_margin_width_c=event_margin_width_c,
        cold_threshold_c=cold_threshold_c,
    )

    n_steps = max(len(rewards), 1)

    sla_steps_arr = np.asarray(sla_step_mwh, dtype=np.float64)
    backlog_arr = np.asarray(backlog_step_mwh, dtype=np.float64)

    sla_step_finite = sla_steps_arr[np.isfinite(sla_steps_arr)]
    if sla_step_finite.size > 0:
        sla_violation_mwh_from_steps = float(sla_step_finite.sum())
        sla_violation_hours = float(np.sum(sla_step_finite > 1e-9))
    else:
        sla_violation_mwh_from_steps = float("nan")
        sla_violation_hours = float("nan")

    backlog_finite = backlog_arr[np.isfinite(backlog_arr)]
    avg_backlog_mwh = (
        float(backlog_finite.mean()) if backlog_finite.size > 0 else float("nan")
    )
    max_backlog_mwh = (
        float(backlog_finite.max()) if backlog_finite.size > 0 else float("nan")
    )

    sla_violation_mwh_summary = _summary_get(summary, "sla_violation_mwh")
    sla_violation_hours_summary = _summary_get(summary, "sla_violation_hours")
    terminal_unserved_mwh = _summary_get(summary, "terminal_unserved_mwh")
    final_backlog_mwh = _summary_get(summary, "final_backlog_mwh")

    sla_violation_mwh = (
        sla_violation_mwh_from_steps
        if np.isfinite(sla_violation_mwh_from_steps)
        else sla_violation_mwh_summary
    )
    sla_violation_rate_pct = (
        100.0 * sla_violation_hours / n_steps
        if np.isfinite(sla_violation_hours)
        else float("nan")
    )

    def _safe_mean(values):
        if not values:
            return float("nan")
        arr = np.asarray(values, dtype=np.float64)
        if not np.any(np.isfinite(arr)):
            return float("nan")
        return float(np.nanmean(np.abs(arr)))

    mean_abs_workload_residual = _safe_mean(workload_residuals)
    mean_abs_setpoint_residual = _safe_mean(setpoint_residuals)
    mean_abs_pump_residual = _safe_mean(pump_residuals)

    mean_action_override = float(
        summary.get("mean_action_override", mean_override_direct)
    )

    metrics = {
        "start_hour": float(start_hour) if start_hour is not None else np.nan,
        "n_steps": int(n_steps),
        "return": float(np.sum(rewards)),
        "co2_kg": _summary_get(summary, "co2_kg"),
        "e_grid_mwh": _summary_get(summary, "e_grid_mwh"),
        "e_total_mwh": _summary_get(summary, "e_total_mwh"),
        "e_cooling_mwh": _summary_get(summary, "e_cooling_mwh"),
        "e_pump_mwh": _summary_get(summary, "e_pump_mwh"),
        "e_cooling_system_mwh": (
            _summary_get(summary, "e_cooling_mwh")
            + _summary_get(summary, "e_pump_mwh")
        ),
        "workload_intervention_rate_pct": (
            100.0 * _summary_get(summary, "workload_interventions") / n_steps
        ),
        "mean_action_override": mean_action_override,
        "mean_abs_workload_residual": mean_abs_workload_residual,
        "mean_abs_setpoint_residual": mean_abs_setpoint_residual,
        "mean_abs_pump_residual": mean_abs_pump_residual,
        "sla_violation_mwh": sla_violation_mwh,
        "sla_violation_mwh_summary": sla_violation_mwh_summary,
        "sla_violation_hours": sla_violation_hours,
        "sla_violation_hours_summary": sla_violation_hours_summary,
        "sla_violation_rate_pct": sla_violation_rate_pct,
        "terminal_unserved_mwh": terminal_unserved_mwh,
        "final_backlog_mwh": final_backlog_mwh,
        "avg_backlog_mwh": avg_backlog_mwh,
        "max_backlog_mwh": max_backlog_mwh,
        **thermal,
    }

    return metrics, step_traces


# =============================================================================
# Aggregation helpers
# =============================================================================

EXCLUDED_FROM_MEAN = {
    "episode", "seed", "start_hour",
    "episode_kind", "season_label", "algorithm",
}


def _aggregate_rows(rows: list[dict]) -> dict[str, float]:
    out: dict[str, float] = {}
    if not rows:
        return out

    keys = set()
    for row in rows:
        keys.update(row.keys())

    for key in keys:
        if key in EXCLUDED_FROM_MEAN:
            continue
        values = []
        for row in rows:
            v = row.get(key, np.nan)
            try:
                values.append(float(v))
            except (TypeError, ValueError):
                values.append(np.nan)
        arr = np.asarray(values, dtype=np.float64)
        with np.errstate(invalid="ignore"):
            out[key] = float(np.nanmean(arr))
    return out


PAIRED_METRIC_COLUMNS = (
    "return",
    "co2_kg",
    "e_grid_mwh",
    "e_total_mwh",
    "e_cooling_mwh",
    "e_pump_mwh",
    "e_cooling_system_mwh",
    "workload_intervention_rate_pct",
    "mean_action_override",
    "sla_violation_mwh",
    "sla_violation_hours",
    "sla_violation_rate_pct",
    "terminal_unserved_mwh",
    "final_backlog_mwh",
    "avg_backlog_mwh",
    "max_backlog_mwh",
    "avg_t_inlet",
    "p95_t_inlet",
    "max_t_inlet",
    "thermal_exposure_c_h",
    "thermal_event_hours",
    "thermal_event_rate_pct",
    "margin_risk_sum",
    "cold_exposure_sq_c2_h",
)


def build_paired_rows(
    rows: list[dict],
    *,
    left: str = "eact",
    right: str = "lora",
) -> list[dict]:
    """left minus right on shared episodes."""
    index: dict[tuple, dict] = {}
    for row in rows:
        key = (row["episode_kind"], row["season_label"], int(row["episode"]))
        index.setdefault(key, {})[row["algorithm"]] = row

    out: list[dict] = []
    for key, per_algo in index.items():
        if left not in per_algo or right not in per_algo:
            continue
        a = per_algo[left]
        b = per_algo[right]
        row = {
            "episode_kind": key[0],
            "season_label": key[1],
            "episode": key[2],
            "seed": int(a["seed"]),
            "start_hour": float(a["start_hour"]),
        }
        for col in PAIRED_METRIC_COLUMNS:
            if col not in a or col not in b:
                continue
            va = a[col]
            vb = b[col]
            row[f"{col}_{left}"] = float(va) if va is not None else float("nan")
            row[f"{col}_{right}"] = float(vb) if vb is not None else float("nan")
            try:
                row[f"{col}_diff"] = float(va) - float(vb)
            except (TypeError, ValueError):
                row[f"{col}_diff"] = float("nan")
        out.append(row)
    return out


# =============================================================================
# Evaluation driver
# =============================================================================

def evaluate(
    *,
    controller_factories: list[tuple[str, callable]],
    country: str,
    forecast_mode: str,
    forecast_dir: str | None,
    base_seed: int,
    episodes: int,
    thermal_reference_c: float,
    event_margin_start_c: float,
    event_margin_width_c: float,
    cold_threshold_c: float,
    verbose_episodes: bool,
    seasonal_start_hours: list[int] | None,
    seasonal_labels: list[str] | None,
) -> tuple[list[dict], list[dict]]:
    tasks: list[dict] = []

    for ep in range(episodes):
        tasks.append({
            "kind": "regular", "label": "",
            "seed": int(base_seed + ep), "start_hour": None,
        })

    # Seasonal seeds use base_seed + i, matching posttrain's base_seed + ep.
    if seasonal_start_hours:
        labels = (
            seasonal_labels if seasonal_labels is not None
            else [f"season_{i}" for i in range(len(seasonal_start_hours))]
        )
        for i, sh in enumerate(seasonal_start_hours):
            tasks.append({
                "kind": "seasonal",
                "label": labels[i] if i < len(labels) else f"season_{i}",
                "seed": int(base_seed + i),
                "start_hour": int(sh),
            })

    if not tasks:
        raise ValueError(
            "No episodes to evaluate: --episodes is 0 and no seasonal "
            "start hours were resolved."
        )

    rows: list[dict] = []
    all_step_traces: list[dict] = []

    for task_idx, task in enumerate(tasks):
        seed = task["seed"]
        start_hour_override = task["start_hour"]
        episode_kind = task["kind"]
        season_label = task["label"]

        for algo_name, factory in controller_factories:
            env = make_env(
                country=country,
                seed=seed,
                forecast_mode=forecast_mode,
                forecast_dir=forecast_dir,
            )

            controller = factory(env)

            try:
                metrics, step_traces = _run_episode(
                    env=env,
                    controller=controller,
                    seed=seed,
                    start_hour_override=start_hour_override,
                    thermal_reference_c=thermal_reference_c,
                    event_margin_start_c=event_margin_start_c,
                    event_margin_width_c=event_margin_width_c,
                    cold_threshold_c=cold_threshold_c,
                    is_lora=(algo_name == "lora"),
                )
            finally:
                del controller

            row = {
                "algorithm": algo_name,
                "episode": int(task_idx),
                "episode_kind": episode_kind,
                "season_label": season_label,
                "seed": int(seed),
                **metrics,
            }
            rows.append(row)

            start_hour_for_trace = (
                float(start_hour_override)
                if start_hour_override is not None
                else float("nan")
            )
            for trace in step_traces:
                all_step_traces.append({
                    "algorithm": algo_name,
                    "episode": int(task_idx),
                    "episode_kind": episode_kind,
                    "season_label": season_label,
                    "seed": int(seed),
                    "start_hour": start_hour_for_trace,
                    **trace,
                })

            if verbose_episodes:
                tag = (
                    f"[{episode_kind}"
                    + (f":{season_label}" if season_label else "")
                    + "]"
                )
                print(
                    f"ep {task_idx + 1:03d}/{len(tasks)} {tag:20s} "
                    f"| {algo_name:22s} | "
                    f"seed={seed} | "
                    f"R={row['return']:.2f} | "
                    f"P95={row['p95_t_inlet']:.3f} | "
                    f"Tex={row['thermal_exposure_c_h']:.3f} | "
                    f"EventH={row['thermal_event_hours']:.1f} | "
                    f"SLA={row['sla_violation_mwh']:.4f} | "
                    f"FinalBacklog={row['final_backlog_mwh']:.4f}"
                )

    return rows, all_step_traces


# =============================================================================
# Saving
# =============================================================================

def save_csv(path: str, rows: list[dict]) -> None:
    if not rows:
        return
    os.makedirs(os.path.dirname(path), exist_ok=True)

    fieldnames = []
    seen = set()
    for row in rows:
        for key in row:
            if key not in seen:
                seen.add(key)
                fieldnames.append(key)

    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def save_step_traces_csv(
    path: str,
    step_traces: list[dict],
    *,
    max_rows: int | None = None,
) -> None:
    if not step_traces:
        print("[warning] step_traces is empty; skipping step_traces.csv")
        return

    os.makedirs(os.path.dirname(path), exist_ok=True)

    first = step_traces[0]
    obs_dim = int(np.asarray(first["obs"]).reshape(-1).shape[0])
    act_dim = int(np.asarray(first["action_requested"]).reshape(-1).shape[0])

    meta_cols = ["algorithm", "episode", "episode_kind", "season_label",
                 "seed", "start_hour", "step"]
    scalar_cols = ["reward", "t_inlet", "co2_kg", "e_grid_mwh", "e_total_mwh"]
    obs_cols = [f"obs_{i}" for i in range(obs_dim)]
    act_req_cols = [f"action_requested_{i}" for i in range(act_dim)]
    act_app_cols = [f"action_applied_{i}" for i in range(act_dim)]

    fieldnames = meta_cols + scalar_cols + obs_cols + act_req_cols + act_app_cols

    rows_to_write = step_traces
    if max_rows is not None and max_rows > 0:
        rows_to_write = step_traces[:max_rows]

    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()

        for trace in rows_to_write:
            obs_vec = np.asarray(trace["obs"]).reshape(-1)
            req_vec = np.asarray(trace["action_requested"]).reshape(-1)
            app_vec = np.asarray(trace["action_applied"]).reshape(-1)

            if obs_vec.shape[0] != obs_dim:
                raise ValueError(
                    "Inconsistent obs dimension in step_traces: "
                    f"expected {obs_dim}, got {obs_vec.shape[0]}"
                )
            if req_vec.shape[0] != act_dim or app_vec.shape[0] != act_dim:
                raise ValueError(
                    "Inconsistent action dimension in step_traces: "
                    f"expected {act_dim}, "
                    f"got requested={req_vec.shape[0]}, "
                    f"applied={app_vec.shape[0]}"
                )

            row = {
                "algorithm": trace["algorithm"],
                "episode": trace["episode"],
                "episode_kind": trace["episode_kind"],
                "season_label": trace["season_label"],
                "seed": trace["seed"],
                "start_hour": trace["start_hour"],
                "step": trace["step"],
                "reward": trace["reward"],
                "t_inlet": trace["t_inlet"],
                # New per-step energy / carbon columns.  .get() keeps
                # backwards compatibility with traces produced by older
                # versions of _run_episode that did not populate these.
                "co2_kg": trace.get("co2_kg", float("nan")),
                "e_grid_mwh": trace.get("e_grid_mwh", float("nan")),
                "e_total_mwh": trace.get("e_total_mwh", float("nan")),
            }
            for i, v in enumerate(obs_vec):
                row[f"obs_{i}"] = float(v)
            for i, v in enumerate(req_vec):
                row[f"action_requested_{i}"] = float(v)
            for i, v in enumerate(app_vec):
                row[f"action_applied_{i}"] = float(v)

            writer.writerow(row)

    print(
        f"[info] wrote {len(rows_to_write)} step traces to {path} "
        f"(obs_dim={obs_dim}, act_dim={act_dim})"
    )


# =============================================================================
# Main
# =============================================================================

def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Paired evaluation: EACT MPC vs Pre-DT vs Post-DT vs LoRA-SAC V6 "
            "vs SAC baseline on the four fixed seasonal typical weeks. "
            "EACT controller and environment are aligned with "
            "posttrain_decision_transformer_v3.py."
        ),
    )

    # --- Checkpoints ---------------------------------------------------
    parser.add_argument("--checkpoint", default=None,
                        help="LoRA-SAC checkpoint (best_lora_thermal_dt.pt).")
    parser.add_argument("--pre-dt-checkpoint", default=None,
                        help="Pretrained DT checkpoint (dt_<C>_seed*.pt).")
    parser.add_argument("--post-dt-checkpoint", default=None,
                        help="V3 post-trained DT checkpoint.")
    parser.add_argument("--sac-checkpoint", default=None,
                        help=("SAC baseline checkpoint produced by "
                              "scripts/RL/train_sac.py (best_sac.pt)."))

    # --- DT inference --------------------------------------------------
    parser.add_argument("--dt-target-return", type=float, default=None,
                        help=("Target return for DT conditioning. If None, "
                              "computed from --expert-trajectories at "
                              "--dt-target-percentile."))
    parser.add_argument("--dt-target-percentile", type=float, default=90.0)
    parser.add_argument("--dt-return-scale", type=float, default=100.0)
    parser.add_argument("--dt-context-len", type=int, default=24)
    parser.add_argument("--expert-trajectories", default=None,
                        help=("Expert trajectory used to compute DT target "
                              "return. Default: data/trajectories/eact_<C>.npz"))

    # --- Run -----------------------------------------------------------
    parser.add_argument("--country", default="JPN")
    parser.add_argument("--episodes", type=int, default=0)
    parser.add_argument("--base-seed", type=int, default=9000)

    parser.add_argument(
        "--controllers",
        nargs="*",
        choices=ALL_CONTROLLER_NAMES,
        default=["eact", "pre_dt", "post_dt", "lora", "sac"],
    )
    parser.add_argument(
        "--lora-deterministic",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument(
        "--sac-deterministic",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=("Use deterministic (mean) SAC actions. Matches the "
              "deterministic seasonal evaluation used inside train_sac.py."),
    )

    parser.add_argument("--thermal-reference", type=float, default=27.0)
    parser.add_argument("--event-margin-start", type=float, default=26.9)
    parser.add_argument("--event-margin-width", type=float, default=0.1)
    parser.add_argument("--cold-threshold", type=float, default=25.5)

    parser.add_argument("--forecast-mode",
                        choices=("oracle", "causal", "none"),
                        default="causal")
    parser.add_argument("--forecast-dir", default=None)

    # EACT-specific options (aligned with posttrain_decision_transformer_v3.py).
    # Only these six are forwarded to EACTMPCController and its subclasses.
    parser.add_argument("--horizon", type=int, default=24)
    parser.add_argument("--maxiter", type=int, default=20)
    parser.add_argument("--block-hours", type=int, default=6)
    parser.add_argument("--gamma", type=float, default=0.995)
    parser.add_argument("--confidence", type=float, default=0.90)

    parser.add_argument(
        "--seasonal-timestamps", nargs="*",
        default=list(DEFAULT_SEASONAL_TIMESTAMPS),
    )
    parser.add_argument("--no-seasonal", action="store_true")

    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--verbose-episodes", action="store_true")

    parser.add_argument(
        "--save-step-traces",
        action=argparse.BooleanOptionalAction, default=True,
    )
    parser.add_argument("--step-trace-max-rows", type=int, default=0)

    args = parser.parse_args()

    if args.episodes < 0:
        raise ValueError("--episodes must be >= 0")
    if args.event_margin_width <= 0:
        raise ValueError("--event-margin-width must be > 0")
    if args.event_margin_start >= args.thermal_reference:
        raise ValueError("--event-margin-start must be below --thermal-reference")
    if args.cold_threshold >= args.thermal_reference:
        raise ValueError("--cold-threshold must be below --thermal-reference")
    if args.no_seasonal and args.episodes == 0:
        raise ValueError(
            "Nothing to evaluate: --no-seasonal was set and --episodes is 0."
        )

    # ------------------------------------------------------------------
    # Resolve device early (needed by factory).
    # ------------------------------------------------------------------
    device = "cuda" if torch.cuda.is_available() else "cpu"
    args.device = device

    # ------------------------------------------------------------------
    # Auto-resolve checkpoints.
    # ------------------------------------------------------------------
    if "lora" in args.controllers and not args.checkpoint:
        resolved = _resolve_default_lora_checkpoint(args.country)
        if resolved is not None:
            args.checkpoint = resolved
            print(f"[info] auto-resolved LoRA checkpoint: {resolved}")
        else:
            raise FileNotFoundError(
                "--checkpoint is required when 'lora' is in --controllers, "
                "and no best_lora_thermal_dt.pt was found under "
                f"{DEFAULT_LORA_ROOT}/<COUNTRY>_*/. "
            )

    if "sac" in args.controllers and not args.sac_checkpoint:
        resolved = _resolve_default_sac_checkpoint(args.country)
        if resolved is not None:
            args.sac_checkpoint = resolved
            print(f"[info] auto-resolved SAC checkpoint: {resolved}")
        else:
            raise FileNotFoundError(
                "--sac-checkpoint is required when 'sac' is in "
                "--controllers, and no best_sac.pt was found under "
                f"{DEFAULT_SAC_ROOT}/<COUNTRY>_v3_sac_*_season/. "
                "Train it with scripts/RL/train_sac.py first."
            )

    if "pre_dt" in args.controllers and not args.pre_dt_checkpoint:
        resolved = _resolve_dt_checkpoint(None, args.country, "pre")
        if resolved is not None:
            args.pre_dt_checkpoint = resolved
            print(f"[info] auto-resolved pre_dt checkpoint: {resolved}")
        else:
            raise FileNotFoundError(
                "No pre-DT checkpoint found. Pass --pre-dt-checkpoint, or "
                "run train_decision_transformer.py first."
            )

    if "post_dt" in args.controllers and not args.post_dt_checkpoint:
        resolved = _resolve_dt_checkpoint(None, args.country, "post")
        if resolved is not None:
            args.post_dt_checkpoint = resolved
            print(f"[info] auto-resolved post_dt checkpoint: {resolved}")
        else:
            raise FileNotFoundError(
                "No post-DT (V3) checkpoint found. Pass --post-dt-checkpoint, "
                "or run posttrain_decision_transformer_v3.py first."
            )

    # ------------------------------------------------------------------
    # Compute DT target return if not provided.
    # ------------------------------------------------------------------
    dt_target_return = args.dt_target_return
    if (("pre_dt" in args.controllers or "post_dt" in args.controllers)
            and dt_target_return is None):
        traj_path = (
            os.path.abspath(args.expert_trajectories)
            if args.expert_trajectories
            else os.path.join(DEFAULT_TRAJ_DIR, f"eact_{args.country}.npz")
        )
        if not os.path.isfile(traj_path):
            raise FileNotFoundError(
                f"Cannot compute DT target return: expert trajectory not "
                f"found at {traj_path}. Pass --dt-target-return explicitly "
                f"or --expert-trajectories."
            )
        dt_target_return = _compute_target_return(
            traj_path=traj_path,
            percentile=args.dt_target_percentile,
        )
        print(
            f"[info] DT target return = {dt_target_return:.3f} "
            f"(P{args.dt_target_percentile:g} of {traj_path})"
        )
    args.dt_target_return = dt_target_return

    # ------------------------------------------------------------------
    # Forecast dir.
    # ------------------------------------------------------------------
    forecast_dir = (
        os.path.abspath(args.forecast_dir)
        if args.forecast_dir
        else DEFAULT_FORECAST_DIR
    )
    args.forecast_dir = forecast_dir

    if any(
        name in args.controllers
        for name in ("nominal", "static", "eact", "eact_no_bias",
                     "eact_no_intervention", "oracle")
    ):
        required_files = [
            os.path.join(forecast_dir, f"causal_ridge_{args.country}.npz"),
            os.path.join(forecast_dir,
                         f"calibration_residuals_{args.country}.npz"),
        ]
        missing = [p for p in required_files if not os.path.isfile(p)]
        if missing:
            raise FileNotFoundError(
                "EACT MPC requires causal forecast files, missing:\n  "
                + "\n  ".join(missing)
            )

    # ------------------------------------------------------------------
    # Load LoRA bundle.
    # ------------------------------------------------------------------
    lora_bundle = None
    if "lora" in args.controllers:
        checkpoint_path = os.path.abspath(args.checkpoint)
        if not os.path.isfile(checkpoint_path):
            raise FileNotFoundError(
                f"LoRA checkpoint not found: {checkpoint_path}"
            )
        ckpt = torch.load(checkpoint_path, map_location="cpu")
        required = ("actor_config", "v3_checkpoint",
                    "trainable_actor_state", "target_return", "return_scale")
        missing = [k for k in required if k not in ckpt]
        if missing:
            raise KeyError(f"LoRA checkpoint missing fields: {missing}")

        actor_config = dict(ckpt["actor_config"])
        actor = ForecastJointLoRAActor(
            v3_checkpoint=ckpt["v3_checkpoint"],
            device=device,
            **actor_config,
        )
        actor.load_trainable_state_dict(ckpt["trainable_actor_state"])
        actor.eval()
        lora_bundle = (
            actor, float(ckpt["target_return"]), float(ckpt["return_scale"]),
        )

        print(f"LoRA checkpoint : {checkpoint_path}")
        print(f"LoRA target     : {lora_bundle[1]:.3f}")
        print(f"LoRA scale      : {lora_bundle[2]:.3f}")

    # ------------------------------------------------------------------
    # Load SAC bundle (baseline).
    # ------------------------------------------------------------------
    sac_bundle = None
    if "sac" in args.controllers:
        sac_path = os.path.abspath(args.sac_checkpoint)
        if not os.path.isfile(sac_path):
            raise FileNotFoundError(f"SAC checkpoint not found: {sac_path}")
        sac_actor, sac_info = _load_sac_bundle(sac_path, device)
        sac_bundle = (sac_actor, sac_info)
        print(f"SAC checkpoint  : {sac_info['path']}")
        print(f"SAC use_lagrange: {sac_info['use_lagrange']}")

    # ------------------------------------------------------------------
    # Build DT bundles.
    # ------------------------------------------------------------------
    dt_bundles: dict[str, dict | None] = {"pre_dt": None, "post_dt": None}

    if "pre_dt" in args.controllers:
        path = os.path.abspath(args.pre_dt_checkpoint)
        if not os.path.isfile(path):
            raise FileNotFoundError(f"pre_dt checkpoint not found: {path}")
        dt_bundles["pre_dt"] = {
            "path": path,
            "target_return": float(args.dt_target_return),
            "return_scale": float(args.dt_return_scale),
            "context_len": int(args.dt_context_len),
        }
        print(f"pre_dt checkpoint : {path}")
        print(f"pre_dt target     : {args.dt_target_return:.3f}")

    if "post_dt" in args.controllers:
        path = os.path.abspath(args.post_dt_checkpoint)
        if not os.path.isfile(path):
            raise FileNotFoundError(f"post_dt checkpoint not found: {path}")
        dt_bundles["post_dt"] = {
            "path": path,
            "target_return": float(args.dt_target_return),
            "return_scale": float(args.dt_return_scale),
            "context_len": int(args.dt_context_len),
        }
        print(f"post_dt checkpoint: {path}")
        print(f"post_dt target    : {args.dt_target_return:.3f}")

    controller_factories = build_controller_factories(
        args, lora_bundle, dt_bundles, sac_bundle,
    )

    # ------------------------------------------------------------------
    # Resolve seasonal start hours.
    # ------------------------------------------------------------------
    seasonal_start_hours: list[int] = []
    seasonal_labels: list[str] = []

    if not args.no_seasonal:
        probe_env = make_env(
            country=args.country,
            seed=args.base_seed,
            forecast_mode=args.forecast_mode,
            forecast_dir=forecast_dir,
        )
        causal_min_history = (
            int(probe_env._causal_min_history)
            if args.forecast_mode == "causal" else 0
        )
        seasonal_start_hours = resolve_seasonal_start_hours(
            probe_env.data,
            args.seasonal_timestamps,
            episode_hours=int(probe_env.episode_hours),
            min_start=causal_min_history,
        )
        for i in range(len(seasonal_start_hours)):
            seasonal_labels.append(
                SEASONAL_LABELS[i] if i < len(SEASONAL_LABELS)
                else f"seasonal_{i}"
            )

    # ------------------------------------------------------------------
    # Evaluate.
    # ------------------------------------------------------------------
    rows, all_step_traces = evaluate(
        controller_factories=controller_factories,
        country=args.country,
        forecast_mode=args.forecast_mode,
        forecast_dir=forecast_dir,
        base_seed=args.base_seed,
        episodes=args.episodes,
        thermal_reference_c=args.thermal_reference,
        event_margin_start_c=args.event_margin_start,
        event_margin_width_c=args.event_margin_width,
        cold_threshold_c=args.cold_threshold,
        verbose_episodes=args.verbose_episodes,
        seasonal_start_hours=seasonal_start_hours or None,
        seasonal_labels=seasonal_labels or None,
    )

    # ------------------------------------------------------------------
    # Aggregations.
    # ------------------------------------------------------------------
    controller_names = [name for name, _ in controller_factories]

    per_controller: dict[str, dict] = {}
    for name in controller_names:
        sub = [r for r in rows if r["algorithm"] == name]
        per_controller[name] = {
            "all": _aggregate_rows(sub),
            "regular": _aggregate_rows(
                [r for r in sub if r["episode_kind"] == "regular"]
            ),
            "seasonal": _aggregate_rows(
                [r for r in sub if r["episode_kind"] == "seasonal"]
            ),
        }

    # Paired: EACT minus every other algorithm.
    paired: dict[str, list[dict]] = {}
    paired_aggregate: dict[str, dict] = {}
    if "eact" in controller_names:
        for name in controller_names:
            if name == "eact":
                continue
            pr = build_paired_rows(rows, left="eact", right=name)
            paired[name] = pr
            paired_aggregate[name] = {
                "all": _aggregate_rows(pr),
                "regular": _aggregate_rows(
                    [r for r in pr if r["episode_kind"] == "regular"]
                ),
                "seasonal": _aggregate_rows(
                    [r for r in pr if r["episode_kind"] == "seasonal"]
                ),
            }

    # Per-season breakdown.
    per_season: dict[str, dict[str, dict]] = {}
    for name in controller_names:
        per_season[name] = {}
        for label in SEASONAL_LABELS:
            sub = [
                r for r in rows
                if r["algorithm"] == name
                and r["episode_kind"] == "seasonal"
                and r["season_label"] == label
            ]
            if sub:
                per_season[name][label] = _aggregate_rows(sub)

    # ------------------------------------------------------------------
    # Save.
    # ------------------------------------------------------------------
    if args.output_dir is None:
        if args.episodes > 0:
            run_tag = f"{args.episodes}ep_seed{args.base_seed}"
        else:
            run_tag = f"seasonal_seed{args.base_seed}"
        output_dir = os.path.join(
            DEFAULT_OUTPUT_ROOT, f"full_{args.country}_{run_tag}",
        )
    else:
        output_dir = os.path.abspath(args.output_dir)

    os.makedirs(output_dir, exist_ok=True)

    episode_csv = os.path.join(output_dir, "episode_results.csv")
    paired_csv = os.path.join(output_dir, "paired_results.csv")
    step_trace_csv = os.path.join(output_dir, "step_traces.csv")
    summary_json = os.path.join(output_dir, "summary.json")

    save_csv(episode_csv, rows)

    # Save one paired CSV per non-EACT algorithm.
    for name, pr in paired.items():
        save_csv(
            os.path.join(output_dir, f"paired_results_eact_vs_{name}.csv"),
            pr,
        )
    # Keep legacy name for EACT-vs-LoRA (if present).
    if "lora" in paired:
        save_csv(paired_csv, paired["lora"])

    if args.save_step_traces:
        save_step_traces_csv(
            step_trace_csv,
            all_step_traces,
            max_rows=(args.step_trace_max_rows or None),
        )
    else:
        print("[info] --no-save-step-traces set; skipping step_traces.csv")

    with open(summary_json, "w", encoding="utf-8") as f:
        json.dump(
            {
                "country": args.country,
                "checkpoint": (
                    os.path.abspath(args.checkpoint) if args.checkpoint else None
                ),
                "pre_dt_checkpoint": (
                    os.path.abspath(args.pre_dt_checkpoint)
                    if args.pre_dt_checkpoint else None
                ),
                "post_dt_checkpoint": (
                    os.path.abspath(args.post_dt_checkpoint)
                    if args.post_dt_checkpoint else None
                ),
                "sac_checkpoint": (
                    os.path.abspath(args.sac_checkpoint)
                    if args.sac_checkpoint else None
                ),
                "dt_target_return": float(args.dt_target_return)
                if args.dt_target_return is not None else None,
                "dt_target_percentile": float(args.dt_target_percentile),
                "dt_return_scale": float(args.dt_return_scale),
                "dt_context_len": int(args.dt_context_len),
                "controllers": controller_names,
                "random_episodes": args.episodes,
                "base_seed": args.base_seed,
                "forecast_mode": args.forecast_mode,
                "forecast_dir": forecast_dir,
                "thermal_reference_c": args.thermal_reference,
                "event_margin_start_c": args.event_margin_start,
                "event_margin_width_c": args.event_margin_width,
                "cold_threshold_c": args.cold_threshold,
                "eact_kwargs": {
                    "horizon": args.horizon,
                    "maxiter": args.maxiter,
                    "control_block_hours": args.block_hours,
                    "gamma": args.gamma,
                    "confidence": args.confidence,
                },
                "seasonal": {
                    "enabled": bool(seasonal_start_hours),
                    "timestamps": list(args.seasonal_timestamps),
                    "labels": list(seasonal_labels),
                    "start_hours": [int(v) for v in seasonal_start_hours],
                },
                "per_controller": per_controller,
                "per_season": per_season,
                "paired": {
                    "left": "eact",
                    "aggregate": paired_aggregate,
                },
                "step_traces": {
                    "enabled": bool(args.save_step_traces),
                    "n_rows": len(all_step_traces),
                    "path": (
                        step_trace_csv if args.save_step_traces else None
                    ),
                    "max_rows": int(args.step_trace_max_rows),
                },
                "output_dir": output_dir,
            },
            f,
            ensure_ascii=False,
            indent=2,
            default=str,
        )

    # ------------------------------------------------------------------
    # Console summary.
    # ------------------------------------------------------------------
    print("=" * 96)
    print("Full evaluation: EACT vs Pre-DT vs Post-DT vs LoRA vs SAC")
    print("(EACT controller and env aligned with posttrain_decision_transformer_v3.py)")
    print("=" * 96)
    print(f"country                 : {args.country}")
    print(f"controllers             : {controller_names}")
    print(f"random episodes         : {args.episodes}")
    if seasonal_start_hours:
        print(f"seasonal typical weeks  : {len(seasonal_start_hours)}")
        for label, ts, sh in zip(
            seasonal_labels, args.seasonal_timestamps, seasonal_start_hours,
        ):
            print(f"  - {label:8s} | {ts} | start_hour={sh}")
    print(f"total rows              : {len(rows)}")
    print(f"total step traces       : {len(all_step_traces)}")
    print("-" * 96)

    preferred_order = (
        "return", "co2_kg", "e_grid_mwh", "e_total_mwh",
        "e_cooling_mwh", "e_pump_mwh", "e_cooling_system_mwh",
        "workload_intervention_rate_pct", "mean_action_override",
        "mean_abs_workload_residual",
        "mean_abs_setpoint_residual",
        "mean_abs_pump_residual",
        "sla_violation_mwh", "sla_violation_hours", "sla_violation_rate_pct",
        "terminal_unserved_mwh", "final_backlog_mwh",
        "avg_backlog_mwh", "max_backlog_mwh",
        "avg_t_inlet", "p95_t_inlet", "max_t_inlet",
        "thermal_exposure_c_h", "thermal_event_hours",
        "thermal_event_rate_pct", "margin_risk_sum",
        "cold_exposure_sq_c2_h", "below_cold_threshold_hours",
    )

    for name in controller_names:
        block_all = per_controller[name]["all"]
        block_sea = per_controller[name]["seasonal"]
        print(f"\n[{name}] aggregate over ALL episodes")
        shown = set()
        for key in preferred_order:
            if key in block_all:
                print(f"  {key:36s}: {block_all[key]:.6f}")
                shown.add(key)
        for key, value in block_all.items():
            if key not in shown:
                print(f"  {key:36s}: {value:.6f}")

        if block_sea:
            print(f"\n[{name}] aggregate over SEASONAL episodes")
            shown = set()
            for key in preferred_order:
                if key in block_sea:
                    print(f"  {key:36s}: {block_sea[key]:.6f}")
                    shown.add(key)

    # Per-season breakdown.
    if seasonal_start_hours:
        print("\n" + "=" * 96)
        print("Per-season typical-week metrics (seasonal episodes only)")
        print("=" * 96)
        for name in controller_names:
            if not per_season.get(name):
                continue
            print(f"\n[{name}]")
            print(
                f"  {'season':8s} | {'return':>9s} | {'P95':>6s} | "
                f"{'Tex_h':>7s} | {'EventH':>7s} | {'SLA_MWh':>9s} | "
                f"{'SLA_H':>6s} | {'TermUnserv':>10s}"
            )
            for label in SEASONAL_LABELS:
                b = per_season[name].get(label)
                if not b:
                    continue
                print(
                    f"  {label:8s} | "
                    f"{b.get('return', float('nan')):9.2f} | "
                    f"{b.get('p95_t_inlet', float('nan')):6.3f} | "
                    f"{b.get('thermal_exposure_c_h', float('nan')):7.3f} | "
                    f"{b.get('thermal_event_hours', float('nan')):7.1f} | "
                    f"{b.get('sla_violation_mwh', float('nan')):9.4f} | "
                    f"{b.get('sla_violation_hours', float('nan')):6.0f} | "
                    f"{b.get('terminal_unserved_mwh', float('nan')):10.4f}"
                )

    if paired_aggregate:
        print("\n" + "=" * 96)
        print("Paired comparison: EACT minus <other> (negative = other better)")
        print("=" * 96)
        for other_name, agg in paired_aggregate.items():
            print(f"\n[EACT minus {other_name} | ALL episodes]")
            for col in PAIRED_METRIC_COLUMNS:
                diff_key = f"{col}_diff"
                if diff_key in agg.get("all", {}):
                    print(f"  {col:36s}: {agg['all'][diff_key]:+.6f}")

    print("-" * 96)
    print(f"episode CSV : {episode_csv}")
    for name in paired:
        print(f"paired CSV  : paired_results_eact_vs_{name}.csv")
    if args.save_step_traces:
        print(f"step CSV    : {step_trace_csv}")
    print(f"summary JSON: {summary_json}")
    print("=" * 96)


if __name__ == "__main__":
    main()