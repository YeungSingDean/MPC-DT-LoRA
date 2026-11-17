"""full_season.py CLI/output with per-controller protocol switching.

Protocol semantics
------------------
--protocol {full_season, paired}      default protocol for every controller.
--protocol-overrides n1=p1 n2=p2 ...  per-controller overrides.

Example:
    --protocol paired --protocol-overrides lora=full_season

means: run every controller on the paired protocol, except LoRA which
runs on the full_season protocol (i.e. byte-for-byte identical to
evaluate_full_season.py for that controller).

EACT's hyperparameters (horizon, maxiter, gamma, confidence,
constraint_tolerance, intervention_weight, adaptive_beta_floor,
forecast_stress*, reward weights, cooling config) are NEVER touched by
either switch -- only the environment / reset protocol changes.
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


SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, "..", ".."))
sys.path.insert(0, ROOT)


from coastaldc_env import CoastalDCContinuousEnv  # noqa: E402
from coastaldc_env.reward import RewardWeights  # noqa: E402
from coastaldc_env.rollout import load_trajectories  # noqa: E402
from coastaldc_env.swhp_cooling import CoolingConfig  # noqa: E402

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

RL_SCRIPT_DIR = os.path.abspath(os.path.join(SCRIPT_DIR, "..", "RL"))
DEFAULT_SAC_ROOT = os.path.join(RL_SCRIPT_DIR, "results")

PAIRED_DEFAULT_DATA_DIR = os.path.join(
    ROOT, "data", "processed_multiyear", "2025",
)
PAIRED_DEFAULT_SEASONAL_HOURS = 168

DEFAULT_OUTPUT_ROOT_FULL = os.path.join(SCRIPT_DIR, "full_evaluate_results")
DEFAULT_OUTPUT_ROOT_PAIRED = os.path.join(
    SCRIPT_DIR, "full_evaluate_results_paired",
)

DEFAULT_SEASONAL_TIMESTAMPS = (
    "2025-01-15 00:00:00",
    "2025-04-15 00:00:00",
    "2025-07-15 00:00:00",
    "2025-10-15 00:00:00",
)
SEASONAL_LABELS = ("winter", "spring", "summer", "autumn")

EACT_CONTROLLER_NAMES = (
    "no_control", "rule_based", "nominal", "static", "eact",
    "eact_no_bias", "eact_no_intervention", "oracle",
)
DT_CONTROLLER_NAMES = ("pre_dt", "post_dt")
ALL_CONTROLLER_NAMES = (
    EACT_CONTROLLER_NAMES + ("lora", "sac") + DT_CONTROLLER_NAMES
)
VALID_PROTOCOLS = ("full_season", "paired")


# =============================================================================
# SAC actor (unchanged)
# =============================================================================

LOG_STD_MIN = -20.0
LOG_STD_MAX = 2.0


class GaussianActor(nn.Module):
    def __init__(self, obs_dim, act_dim, hidden_dim, action_low, action_high):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim), nn.ReLU(),
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
    def __init__(self, *, actor, device, deterministic=True):
        self.actor = actor
        self.device = device
        self.deterministic = bool(deterministic)

    def reset(self):
        return None

    def act(self, obs, info=None):
        with torch.no_grad():
            obs_t = torch.as_tensor(
                np.asarray(obs, dtype=np.float32), device=self.device,
            ).unsqueeze(0)
            action, _ = self.actor.sample(
                obs_t, deterministic=self.deterministic)
            action_np = action.squeeze(0).detach().cpu().numpy()
        return action_np.astype(np.float32)


# =============================================================================
# Checkpoint resolution (unchanged)
# =============================================================================

def _newest(candidates):
    if not candidates:
        return None
    candidates.sort(key=os.path.getmtime, reverse=True)
    return candidates[0]


def _resolve_default_lora_checkpoint(country):
    if not os.path.isdir(DEFAULT_LORA_ROOT):
        return None
    candidates = sorted(glob.glob(os.path.join(
        DEFAULT_LORA_ROOT, f"{country}_*", "best_lora_thermal_dt.pt")))
    if candidates:
        return candidates[-1]
    candidates = sorted(glob.glob(os.path.join(
        DEFAULT_LORA_ROOT, "*", "best_lora_thermal_dt.pt")))
    return candidates[-1] if candidates else None


def _resolve_default_sac_checkpoint(country):
    if not os.path.isdir(DEFAULT_SAC_ROOT):
        return None
    for tag in ("v3_sac_lagrange_season", "v3_sac_vanilla_season"):
        path = os.path.join(
            DEFAULT_SAC_ROOT, f"{country}_{tag}", "best_sac.pt")
        if os.path.isfile(path):
            return path
    fallback = sorted(glob.glob(os.path.join(
        DEFAULT_SAC_ROOT, f"{country}_*", "best_sac.pt")))
    return fallback[-1] if fallback else None


def _resolve_dt_checkpoint(checkpoint_arg, country, kind):
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
    candidates = []
    for d in DEFAULT_MODEL_DIRS:
        if not os.path.isdir(d):
            continue
        for p in patterns:
            candidates.extend(glob.glob(os.path.join(d, p)))
    return _newest(candidates)


def _compute_target_return(*, traj_path, percentile):
    trajectories = load_trajectories([traj_path], use_applied_actions=True)
    returns = np.asarray(
        [float(np.asarray(t["rewards"], dtype=np.float64).sum())
         for t in trajectories], dtype=np.float64)
    return float(np.percentile(returns, percentile))


def _load_sac_bundle(checkpoint_path, device):
    ckpt = torch.load(checkpoint_path, map_location="cpu")
    required = ("actor_config", "actor_state")
    missing = [k for k in required if k not in ckpt]
    if missing:
        raise KeyError(
            f"SAC checkpoint missing fields: {missing} (found keys="
            f"{sorted(ckpt.keys())})")
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

def make_env(*, protocol, country, seed, forecast_mode, forecast_dir,
             reward_weights, cooling_config,
             data_dir=None, episode_hours=None):
    base = dict(
        country=country, seed=seed,
        forecast_mode=forecast_mode, forecast_dir=forecast_dir,
        reward_weights=reward_weights, cooling_config=cooling_config,
    )
    if protocol == "paired":
        base.update(
            data_dir=os.path.abspath(data_dir),
            episode_hours=int(episode_hours),
            random_episode_start=False,
            use_oracle_workload_projection=False,
            use_oracle_forecast_observations=False,
        )
    return CoastalDCContinuousEnv(**base)


def resolve_seasonal_start_hours(data, timestamps, *, episode_hours):
    """Use the strict (full_season) bound so every protocol accepts the
    resolved starts.  For the four 2025 seasonal timestamps this makes no
    difference -- they are far from the boundary."""
    if "timestamp" not in data.columns:
        raise ValueError(
            "Seasonal start timestamps require a 'timestamp' column "
            "in the environment data.")
    parsed = pd.to_datetime(data["timestamp"], errors="raise")
    if parsed.duplicated().any():
        raise ValueError("Environment data contain duplicate timestamps.")
    lookup = {ts: i for i, ts in enumerate(parsed)}
    max_start = len(data) - int(episode_hours) - 24
    out = []
    for value in timestamps:
        ts = pd.Timestamp(value)
        if ts not in lookup:
            raise ValueError(
                f"Seasonal timestamp is absent from environment data: {value}")
        idx = int(lookup[ts])
        if idx < 0 or idx > max_start:
            raise ValueError(
                f"Seasonal timestamp {value} maps to start_hour={idx}, "
                f"outside valid range [0, {max_start}].")
        out.append(idx)
    return out


# =============================================================================
# Per-episode metrics (unchanged)
# =============================================================================

def thermal_metrics(temperatures, *, thermal_reference_c,
                    event_margin_start_c, event_margin_width_c,
                    cold_threshold_c):
    x = np.asarray(temperatures, dtype=np.float64)
    x = x[np.isfinite(x)]
    if len(x) == 0:
        raise ValueError("No finite inlet-temperature values were recorded.")
    hot_excess = np.maximum(x - thermal_reference_c, 0.0)
    true_event = x > thermal_reference_c
    margin_risk = np.clip(
        (x - event_margin_start_c) / event_margin_width_c, 0.0, 1.0)
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


def _summary_get(summary, key):
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
# Controller factories (unchanged)
# =============================================================================

def _shared_eact_kwargs(args):
    return dict(
        forecast_dir=args.forecast_dir,
        horizon=args.horizon, maxiter=args.maxiter,
        control_block_hours=args.block_hours,
        gamma=args.gamma, confidence=args.confidence,
        constraint_tolerance=args.constraint_tolerance,
        intervention_weight=args.intervention_weight,
        adaptive_beta_floor=args.adaptive_beta_floor,
        forecast_stress_mode=args.forecast_stress,
        forecast_stress_scale=args.forecast_stress_scale,
        forecast_stress_start_step=args.forecast_stress_start_step,
        forecast_stress_seed=args.forecast_stress_seed,
    )


def build_controller_factories(args, lora_bundle, dt_bundles, sac_bundle=None):
    factories = []
    shared_eact = _shared_eact_kwargs(args)
    device = args.device

    for name in args.controllers:
        if name == "lora":
            if lora_bundle is None:
                raise ValueError(
                    "LoRA controller requested but no checkpoint was loaded")
            actor, target_return, return_scale = lora_bundle
            deterministic = bool(args.lora_deterministic)

            def make_lora(env, _actor=actor, _tr=target_return,
                          _rs=return_scale, _det=deterministic):
                return ForecastJointLoRADTController(
                    actor=_actor, env=env,
                    target_return=_tr, return_scale=_rs, deterministic=_det)

            factories.append(("lora", make_lora))

        elif name == "sac":
            if sac_bundle is None:
                raise ValueError(
                    "SAC controller requested but no checkpoint was loaded.")
            sac_actor, _info = sac_bundle
            sac_det = bool(args.sac_deterministic)

            def make_sac(env, _actor=sac_actor, _dev=device, _det=sac_det):
                return SACController(actor=_actor, device=_dev,
                                     deterministic=_det)

            factories.append(("sac", make_sac))

        elif name in ("pre_dt", "post_dt"):
            bundle = dt_bundles.get(name)
            if bundle is None:
                raise ValueError(f"{name} requested but no checkpoint loaded.")

            def make_dt(env, _b=bundle, _dev=device):
                return DecisionTransformerController(
                    obs_dim=int(env.observation_space.shape[0]),
                    target_return=float(_b["target_return"]),
                    context_len=int(_b["context_len"]),
                    model_path=_b["path"], device=_dev,
                    return_scale=float(_b["return_scale"]))

            factories.append((name, make_dt))

        elif name == "no_control":
            factories.append(("no_control", lambda env: NoControlController()))
        elif name == "rule_based":
            factories.append(("rule_based", lambda env: RuleBasedController()))
        elif name == "nominal":
            factories.append(("nominal",
                lambda env: NominalCausalMPCController(env, **shared_eact)))
        elif name == "static":
            factories.append(("static",
                lambda env: StaticRobustMPCController(env, **shared_eact)))
        elif name == "eact":
            factories.append(("eact",
                lambda env: EACTMPCController(env, **shared_eact)))
        elif name == "eact_no_bias":
            factories.append(("eact_no_bias",
                lambda env: EACTNoBiasMPCController(env, **shared_eact)))
        elif name == "eact_no_intervention":
            factories.append(("eact_no_intervention",
                lambda env: EACTNoInterventionMPCController(env, **shared_eact)))
        elif name == "oracle":
            factories.append(("oracle",
                lambda env: OracleConstrainedMPCController(env, **shared_eact)))
        else:
            raise ValueError(f"Unknown controller: {name}")

    return factories


# =============================================================================
# Episode runner (unchanged)
# =============================================================================

def _run_episode(*, env, controller, seed, start_hour_override,
                 thermal_reference_c, event_margin_start_c,
                 event_margin_width_c, cold_threshold_c, is_lora):
    options = (None if start_hour_override is None
               else {"start_hour": int(start_hour_override)})
    obs, info = env.reset(seed=seed, options=options)
    if hasattr(controller, "reset"):
        controller.reset()
    start_hour = info.get("data_hour", np.nan)

    done = False
    rewards, temperatures = [], []
    requested_actions, applied_actions = [], []
    workload_residuals, setpoint_residuals, pump_residuals = [], [], []
    sla_step_mwh, backlog_step_mwh = [], []
    step_traces = []
    step_idx = 0

    while not done:
        obs_snapshot = np.asarray(obs, dtype=np.float32).copy()
        requested = np.asarray(controller.act(obs, info), dtype=np.float32)

        if is_lora and hasattr(controller, "last_base_action"):
            base = np.asarray(controller.last_base_action, dtype=np.float64)
            workload_residuals.append(float(requested[0] - base[0]))
            setpoint_residuals.append(float(requested[1] - base[1]))
            pump_residuals.append(float(requested[2] - base[2]))

        next_obs, reward, terminated, truncated, next_info = env.step(requested)
        applied = np.asarray(
            next_info.get("applied_action", requested), dtype=np.float32)
        t_inlet_step = float(next_info.get("t_inlet", np.nan))

        step_traces.append({
            "step": int(step_idx),
            "obs": obs_snapshot,
            "action_requested": requested.copy(),
            "action_applied": applied.copy(),
            "reward": float(reward),
            "t_inlet": t_inlet_step,
            "co2_kg": float(next_info.get("co2_kg", np.nan)),
            "e_grid_mwh": float(next_info.get("e_grid_mwh", np.nan)),
            "e_total_mwh": float(next_info.get("e_total_mwh", np.nan)),
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
    mean_override_direct = (float(np.mean(np.abs(req - app)))
                            if req.shape == app.shape else float("nan"))

    thermal = thermal_metrics(
        temperatures,
        thermal_reference_c=thermal_reference_c,
        event_margin_start_c=event_margin_start_c,
        event_margin_width_c=event_margin_width_c,
        cold_threshold_c=cold_threshold_c)

    n_steps = max(len(rewards), 1)
    sla_arr = np.asarray(sla_step_mwh, dtype=np.float64)
    backlog_arr = np.asarray(backlog_step_mwh, dtype=np.float64)
    sla_finite = sla_arr[np.isfinite(sla_arr)]
    if sla_finite.size > 0:
        sla_violation_mwh_from_steps = float(sla_finite.sum())
        sla_violation_hours = float(np.sum(sla_finite > 1e-9))
    else:
        sla_violation_mwh_from_steps = float("nan")
        sla_violation_hours = float("nan")
    backlog_finite = backlog_arr[np.isfinite(backlog_arr)]
    avg_backlog_mwh = (float(backlog_finite.mean())
                       if backlog_finite.size > 0 else float("nan"))
    max_backlog_mwh = (float(backlog_finite.max())
                       if backlog_finite.size > 0 else float("nan"))

    sla_violation_mwh_summary = _summary_get(summary, "sla_violation_mwh")
    sla_violation_hours_summary = _summary_get(summary, "sla_violation_hours")
    terminal_unserved_mwh = _summary_get(summary, "terminal_unserved_mwh")
    final_backlog_mwh = _summary_get(summary, "final_backlog_mwh")
    sla_violation_mwh = (sla_violation_mwh_from_steps
                        if np.isfinite(sla_violation_mwh_from_steps)
                        else sla_violation_mwh_summary)
    sla_violation_rate_pct = (100.0 * sla_violation_hours / n_steps
                             if np.isfinite(sla_violation_hours)
                             else float("nan"))

    def _safe_mean(values):
        if not values:
            return float("nan")
        arr = np.asarray(values, dtype=np.float64)
        if not np.any(np.isfinite(arr)):
            return float("nan")
        return float(np.nanmean(np.abs(arr)))

    metrics = {
        "start_hour": float(start_hour) if start_hour is not None else np.nan,
        "n_steps": int(n_steps),
        "return": float(np.sum(rewards)),
        "co2_kg": _summary_get(summary, "co2_kg"),
        "e_grid_mwh": _summary_get(summary, "e_grid_mwh"),
        "e_total_mwh": _summary_get(summary, "e_total_mwh"),
        "e_cooling_mwh": _summary_get(summary, "e_cooling_mwh"),
        "e_pump_mwh": _summary_get(summary, "e_pump_mwh"),
        "e_cooling_system_mwh": (_summary_get(summary, "e_cooling_mwh")
                                 + _summary_get(summary, "e_pump_mwh")),
        "workload_intervention_rate_pct": (
            100.0 * _summary_get(summary, "workload_interventions") / n_steps),
        "mean_action_override": float(
            summary.get("mean_action_override", mean_override_direct)),
        "mean_abs_workload_residual": _safe_mean(workload_residuals),
        "mean_abs_setpoint_residual": _safe_mean(setpoint_residuals),
        "mean_abs_pump_residual": _safe_mean(pump_residuals),
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
# Aggregation (unchanged)
# =============================================================================

EXCLUDED_FROM_MEAN = {"episode", "seed", "start_hour",
                     "episode_kind", "season_label", "algorithm"}


def _aggregate_rows(rows):
    out = {}
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
    "return", "co2_kg", "e_grid_mwh", "e_total_mwh",
    "e_cooling_mwh", "e_pump_mwh", "e_cooling_system_mwh",
    "workload_intervention_rate_pct", "mean_action_override",
    "sla_violation_mwh", "sla_violation_hours", "sla_violation_rate_pct",
    "terminal_unserved_mwh", "final_backlog_mwh",
    "avg_backlog_mwh", "max_backlog_mwh",
    "avg_t_inlet", "p95_t_inlet", "max_t_inlet",
    "thermal_exposure_c_h", "thermal_event_hours", "thermal_event_rate_pct",
    "margin_risk_sum", "cold_exposure_sq_c2_h",
)


def build_paired_rows(rows, *, left="eact", right="lora"):
    index = {}
    for row in rows:
        key = (row["episode_kind"], row["season_label"], int(row["episode"]))
        index.setdefault(key, {})[row["algorithm"]] = row
    out = []
    for key, per_algo in index.items():
        if left not in per_algo or right not in per_algo:
            continue
        a, b = per_algo[left], per_algo[right]
        row = {"episode_kind": key[0], "season_label": key[1],
               "episode": key[2], "seed": int(a["seed"]),
               "start_hour": float(a["start_hour"])}
        for col in PAIRED_METRIC_COLUMNS:
            if col not in a or col not in b:
                continue
            va, vb = a[col], b[col]
            row[f"{col}_{left}"] = float(va) if va is not None else float("nan")
            row[f"{col}_{right}"] = float(vb) if vb is not None else float("nan")
            try:
                row[f"{col}_diff"] = float(va) - float(vb)
            except (TypeError, ValueError):
                row[f"{col}_diff"] = float("nan")
        out.append(row)
    return out


# =============================================================================
# Evaluation driver (per-controller protocol)
# =============================================================================

def evaluate(*, args, controller_factories, protocol_for, country,
             forecast_mode, forecast_dir, reward_weights, cooling_config,
             data_dir, episode_hours, base_seed, episodes,
             thermal_reference_c, event_margin_start_c, event_margin_width_c,
             cold_threshold_c, verbose_episodes,
             seasonal_start_hours, seasonal_labels):
    """protocol_for: dict {controller_name -> "full_season"|"paired"}."""
    tasks = []
    for ep in range(episodes):
        tasks.append({"kind": "regular", "label": "",
                      "seed": int(base_seed + ep), "start_hour": None})
    if seasonal_start_hours:
        labels = (seasonal_labels if seasonal_labels is not None
                  else [f"season_{i}" for i in range(len(seasonal_start_hours))])
        for i, sh in enumerate(seasonal_start_hours):
            tasks.append({"kind": "seasonal",
                          "label": labels[i] if i < len(labels) else f"season_{i}",
                          "seed": int(base_seed + episodes + i),
                          "start_hour": int(sh)})
    if not tasks:
        raise ValueError("No episodes to evaluate.")

    # Env cache: paired envs are shared per (country, episode_hours);
    # full_season envs are fresh per episode (matches evaluate_full_season.py).
    paired_env_cache = {}

    def _get_env(proto):
        if proto == "paired":
            key = (country, int(episode_hours) if episode_hours else -1)
            if key not in paired_env_cache:
                paired_env_cache[key] = make_env(
                    protocol="paired", country=country, seed=base_seed,
                    forecast_mode=forecast_mode, forecast_dir=forecast_dir,
                    reward_weights=reward_weights, cooling_config=cooling_config,
                    data_dir=data_dir, episode_hours=episode_hours)
            return paired_env_cache[key]
        return make_env(
            protocol="full_season", country=country, seed=base_seed,
            forecast_mode=forecast_mode, forecast_dir=forecast_dir,
            reward_weights=reward_weights, cooling_config=cooling_config,
            data_dir=data_dir, episode_hours=episode_hours)

    rows, all_step_traces = [], []

    for task_idx, task in enumerate(tasks):
        seed = task["seed"]
        start_hour_override = task["start_hour"]
        episode_kind = task["kind"]
        season_label = task["label"]

        for algo_name, factory in controller_factories:
            proto = protocol_for.get(algo_name, args.protocol)
            env = _get_env(proto)
            controller = factory(env)

            try:
                metrics, step_traces = _run_episode(
                    env=env, controller=controller, seed=seed,
                    start_hour_override=start_hour_override,
                    thermal_reference_c=thermal_reference_c,
                    event_margin_start_c=event_margin_start_c,
                    event_margin_width_c=event_margin_width_c,
                    cold_threshold_c=cold_threshold_c,
                    is_lora=(algo_name == "lora"))
            finally:
                del controller

            row = {"algorithm": algo_name, "episode": int(task_idx),
                   "episode_kind": episode_kind, "season_label": season_label,
                   "seed": int(seed), "protocol": proto, **metrics}
            rows.append(row)

            start_hour_for_trace = (float(start_hour_override)
                                    if start_hour_override is not None
                                    else float("nan"))
            for trace in step_traces:
                all_step_traces.append({
                    "algorithm": algo_name, "episode": int(task_idx),
                    "episode_kind": episode_kind, "season_label": season_label,
                    "seed": int(seed), "start_hour": start_hour_for_trace,
                    **trace})

            if verbose_episodes:
                tag = f"[{episode_kind}" + (f":{season_label}" if season_label else "") + "]"
                print(f"ep {task_idx+1:03d}/{len(tasks)} {tag:20s} "
                      f"| {algo_name:22s} | proto={proto:11s} | "
                      f"seed={seed} | R={row['return']:.2f} | "
                      f"P95={row['p95_t_inlet']:.3f} | "
                      f"Tex={row['thermal_exposure_c_h']:.3f} | "
                      f"EventH={row['thermal_event_hours']:.1f} | "
                      f"SLA={row['sla_violation_mwh']:.4f} | "
                      f"FinalBacklog={row['final_backlog_mwh']:.4f}")

    return rows, all_step_traces


# =============================================================================
# Saving (unchanged)
# =============================================================================

def save_csv(path, rows):
    if not rows:
        return
    os.makedirs(os.path.dirname(path), exist_ok=True)
    fieldnames, seen = [], set()
    for row in rows:
        for key in row:
            if key not in seen:
                seen.add(key)
                fieldnames.append(key)
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def save_step_traces_csv(path, step_traces, *, max_rows=None):
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
            row = {
                "algorithm": trace["algorithm"], "episode": trace["episode"],
                "episode_kind": trace["episode_kind"],
                "season_label": trace["season_label"], "seed": trace["seed"],
                "start_hour": trace["start_hour"], "step": trace["step"],
                "reward": trace["reward"], "t_inlet": trace["t_inlet"],
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
    print(f"[info] wrote {len(rows_to_write)} step traces to {path} "
          f"(obs_dim={obs_dim}, act_dim={act_dim})")


# =============================================================================
# Main
# =============================================================================

def _parse_protocol_overrides(items):
    """Parse ["lora=full_season", "eact=paired"] -> {"lora": "full_season", ...}."""
    overrides = {}
    for item in items or []:
        if "=" not in item:
            raise ValueError(
                f"--protocol-overrides entries must be <controller>=<protocol>, "
                f"got {item!r}")
        name, proto = item.split("=", 1)
        name, proto = name.strip(), proto.strip()
        if proto not in VALID_PROTOCOLS:
            raise ValueError(
                f"Unknown protocol {proto!r} in --protocol-overrides; "
                f"valid: {VALID_PROTOCOLS}")
        if name not in ALL_CONTROLLER_NAMES:
            raise ValueError(
                f"Unknown controller {name!r} in --protocol-overrides; "
                f"valid: {ALL_CONTROLLER_NAMES}")
        overrides[name] = proto
    return overrides


def main():
    parser = argparse.ArgumentParser(
        description="full_season.py CLI/output with per-controller protocol.",
    )

    # --- Checkpoints ---
    parser.add_argument("--checkpoint", default=None)
    parser.add_argument("--pre-dt-checkpoint", default=None)
    parser.add_argument("--post-dt-checkpoint", default=None)
    parser.add_argument("--sac-checkpoint", default=None)

    # --- DT inference ---
    parser.add_argument("--dt-target-return", type=float, default=None)
    parser.add_argument("--dt-target-percentile", type=float, default=90.0)
    parser.add_argument("--dt-return-scale", type=float, default=100.0)
    parser.add_argument("--dt-context-len", type=int, default=24)
    parser.add_argument("--expert-trajectories", default=None)

    # --- Run ---
    parser.add_argument("--country", default="JPN")
    parser.add_argument("--episodes", type=int, default=0)
    parser.add_argument("--base-seed", type=int, default=9000)
    parser.add_argument("--controllers", nargs="*",
                        choices=ALL_CONTROLLER_NAMES,
                        default=["eact", "pre_dt", "post_dt", "lora", "sac"])
    parser.add_argument("--lora-deterministic",
                        action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--sac-deterministic",
                        action=argparse.BooleanOptionalAction, default=True)

    parser.add_argument("--thermal-reference", type=float, default=27.0)
    parser.add_argument("--event-margin-start", type=float, default=26.9)
    parser.add_argument("--event-margin-width", type=float, default=0.1)
    parser.add_argument("--cold-threshold", type=float, default=25.5)

    # --- Protocol (global + per-controller overrides) ---
    parser.add_argument(
        "--protocol", choices=VALID_PROTOCOLS, default="full_season",
        help=("Default protocol for all controllers.  "
              "full_season (default): byte-for-byte identical to "
              "evaluate_full_season.py.  paired: matches "
              "evaluate_paired_eact_vs_lora.py."))
    parser.add_argument(
        "--protocol-overrides", nargs="*", default=[],
        help=("Per-controller protocol overrides.  Each entry is "
              "<controller>=<full_season|paired>.  Unlisted controllers "
              "use --protocol.  Example: --protocol paired "
              "--protocol-overrides lora=full_season"))

    parser.add_argument("--data-dir", default=PAIRED_DEFAULT_DATA_DIR,
                        help="Only used by controllers on the paired protocol.")
    parser.add_argument("--seasonal-hours", type=int,
                        default=PAIRED_DEFAULT_SEASONAL_HOURS)

    parser.add_argument("--forecast-mode",
                        choices=("oracle", "causal", "none"),
                        default="causal")
    parser.add_argument("--forecast-dir", default=None)

    # EACT-specific (unchanged)
    parser.add_argument("--horizon", type=int, default=24)
    parser.add_argument("--maxiter", type=int, default=20)
    parser.add_argument("--block-hours", type=int, default=6)
    parser.add_argument("--gamma", type=float, default=0.995)
    parser.add_argument("--confidence", type=float, default=0.90)
    parser.add_argument("--constraint-tolerance", type=float, default=1e-4)
    parser.add_argument("--intervention-weight", type=float, default=0.1)
    parser.add_argument("--adaptive-beta-floor", type=float, default=0.10)
    parser.add_argument("--forecast-stress",
                        choices=("none", "biased", "noisy", "bias_drift"),
                        default="none")
    parser.add_argument("--forecast-stress-scale", type=float, default=0.0)
    parser.add_argument("--forecast-stress-start-step", type=int, default=0)
    parser.add_argument("--forecast-stress-seed", type=int, default=20260717)

    # Reward weights (unchanged)
    parser.add_argument("--weight-grid", type=float, default=1.0)
    parser.add_argument("--weight-co2", type=float, default=2.0)
    parser.add_argument("--weight-total", type=float, default=0.2)
    parser.add_argument("--weight-smooth", type=float, default=0.5)

    parser.add_argument("--cooling-conductance-multiplier",
                        type=float, default=1.0)
    parser.add_argument("--thermal-safety-shield",
                        action=argparse.BooleanOptionalAction, default=True)

    parser.add_argument("--seasonal-timestamps", nargs="*",
                        default=list(DEFAULT_SEASONAL_TIMESTAMPS))
    parser.add_argument("--no-seasonal", action="store_true")

    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--verbose-episodes", action="store_true")
    parser.add_argument("--save-step-traces",
                        action=argparse.BooleanOptionalAction, default=True)
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
        raise ValueError("Nothing to evaluate.")

    # --- Per-controller protocol map ---
    overrides = _parse_protocol_overrides(args.protocol_overrides)
    # Only honour overrides for controllers actually in this run.
    protocol_for = {name: overrides.get(name, args.protocol)
                    for name in args.controllers}

    # --- Device ---
    device = "cuda" if torch.cuda.is_available() else "cpu"
    args.device = device

    # --- Checkpoint auto-resolve (unchanged) ---
    if "lora" in args.controllers and not args.checkpoint:
        resolved = _resolve_default_lora_checkpoint(args.country)
        if resolved is None:
            raise FileNotFoundError(
                "--checkpoint required for lora; nothing found under "
                f"{DEFAULT_LORA_ROOT}/<COUNTRY>_*/")
        args.checkpoint = resolved
        print(f"[info] auto-resolved LoRA checkpoint: {resolved}")

    if "sac" in args.controllers and not args.sac_checkpoint:
        resolved = _resolve_default_sac_checkpoint(args.country)
        if resolved is None:
            raise FileNotFoundError("--sac-checkpoint required for sac.")
        args.sac_checkpoint = resolved
        print(f"[info] auto-resolved SAC checkpoint: {resolved}")

    if "pre_dt" in args.controllers and not args.pre_dt_checkpoint:
        resolved = _resolve_dt_checkpoint(None, args.country, "pre")
        if resolved is None:
            raise FileNotFoundError("No pre-DT checkpoint found.")
        args.pre_dt_checkpoint = resolved
        print(f"[info] auto-resolved pre_dt checkpoint: {resolved}")

    if "post_dt" in args.controllers and not args.post_dt_checkpoint:
        resolved = _resolve_dt_checkpoint(None, args.country, "post")
        if resolved is None:
            raise FileNotFoundError("No post-DT (V3) checkpoint found.")
        args.post_dt_checkpoint = resolved
        print(f"[info] auto-resolved post_dt checkpoint: {resolved}")

    # --- DT target return ---
    dt_target_return = args.dt_target_return
    if (("pre_dt" in args.controllers or "post_dt" in args.controllers)
            and dt_target_return is None):
        traj_path = (os.path.abspath(args.expert_trajectories)
                     if args.expert_trajectories
                     else os.path.join(DEFAULT_TRAJ_DIR, f"eact_{args.country}.npz"))
        if not os.path.isfile(traj_path):
            raise FileNotFoundError(
                f"Cannot compute DT target return: expert trajectory not "
                f"found at {traj_path}.")
        dt_target_return = _compute_target_return(
            traj_path=traj_path, percentile=args.dt_target_percentile)
        print(f"[info] DT target return = {dt_target_return:.3f} "
              f"(P{args.dt_target_percentile:g} of {traj_path})")
    args.dt_target_return = dt_target_return

    # --- Reward / cooling ---
    reward_weights = RewardWeights(
        w_grid=args.weight_grid, w_co2=args.weight_co2,
        w_total=args.weight_total, w_smooth=args.weight_smooth)
    nominal_cooling = CoolingConfig()
    cooling_config = CoolingConfig(
        conductance_mw_per_k=(nominal_cooling.conductance_mw_per_k
                              * float(args.cooling_conductance_multiplier)),
        enforce_thermal_safety=bool(args.thermal_safety_shield))

    forecast_dir = (os.path.abspath(args.forecast_dir) if args.forecast_dir
                    else DEFAULT_FORECAST_DIR)
    args.forecast_dir = forecast_dir
    data_dir = os.path.abspath(args.data_dir) if args.data_dir else None

    if any(name in args.controllers
           for name in ("nominal", "static", "eact", "eact_no_bias",
                        "eact_no_intervention", "oracle")):
        required = [
            os.path.join(forecast_dir, f"causal_ridge_{args.country}.npz"),
            os.path.join(forecast_dir,
                         f"calibration_residuals_{args.country}.npz")]
        missing = [p for p in required if not os.path.isfile(p)]
        if missing:
            raise FileNotFoundError(
                "EACT MPC requires causal forecast files, missing:\n  "
                + "\n  ".join(missing))

    # --- LoRA bundle ---
    lora_bundle = None
    if "lora" in args.controllers:
        checkpoint_path = os.path.abspath(args.checkpoint)
        if not os.path.isfile(checkpoint_path):
            raise FileNotFoundError(f"LoRA checkpoint not found: {checkpoint_path}")
        ckpt = torch.load(checkpoint_path, map_location="cpu")
        required = ("actor_config", "v3_checkpoint",
                    "trainable_actor_state", "target_return", "return_scale")
        missing = [k for k in required if k not in ckpt]
        if missing:
            raise KeyError(f"LoRA checkpoint missing fields: {missing}")
        actor = ForecastJointLoRAActor(
            v3_checkpoint=ckpt["v3_checkpoint"], device=device,
            **dict(ckpt["actor_config"]))
        actor.load_trainable_state_dict(ckpt["trainable_actor_state"])
        actor.eval()
        lora_bundle = (actor, float(ckpt["target_return"]),
                       float(ckpt["return_scale"]))
        print(f"LoRA checkpoint : {checkpoint_path}")
        print(f"LoRA target     : {lora_bundle[1]:.3f}")
        print(f"LoRA scale      : {lora_bundle[2]:.3f}")

    # --- SAC bundle ---
    sac_bundle = None
    if "sac" in args.controllers:
        sac_path = os.path.abspath(args.sac_checkpoint)
        if not os.path.isfile(sac_path):
            raise FileNotFoundError(f"SAC checkpoint not found: {sac_path}")
        sac_actor, sac_info = _load_sac_bundle(sac_path, device)
        sac_bundle = (sac_actor, sac_info)
        print(f"SAC checkpoint  : {sac_info['path']}")
        print(f"SAC use_lagrange: {sac_info['use_lagrange']}")

    # --- DT bundles ---
    dt_bundles = {"pre_dt": None, "post_dt": None}
    if "pre_dt" in args.controllers:
        path = os.path.abspath(args.pre_dt_checkpoint)
        if not os.path.isfile(path):
            raise FileNotFoundError(f"pre_dt checkpoint not found: {path}")
        dt_bundles["pre_dt"] = {
            "path": path, "target_return": float(args.dt_target_return),
            "return_scale": float(args.dt_return_scale),
            "context_len": int(args.dt_context_len)}
        print(f"pre_dt checkpoint : {path}")
        print(f"pre_dt target     : {args.dt_target_return:.3f}")
    if "post_dt" in args.controllers:
        path = os.path.abspath(args.post_dt_checkpoint)
        if not os.path.isfile(path):
            raise FileNotFoundError(f"post_dt checkpoint not found: {path}")
        dt_bundles["post_dt"] = {
            "path": path, "target_return": float(args.dt_target_return),
            "return_scale": float(args.dt_return_scale),
            "context_len": int(args.dt_context_len)}
        print(f"post_dt checkpoint: {path}")
        print(f"post_dt target    : {args.dt_target_return:.3f}")

    controller_factories = build_controller_factories(
        args, lora_bundle, dt_bundles, sac_bundle)

    # --- Seasonal start hours ---
    seasonal_start_hours, seasonal_labels = [], []
    episode_hours = None
    if not args.no_seasonal:
        # Probe with paired protocol (most permissive) to get the timestamp
        # data and the reference episode_hours.
        probe = make_env(
            protocol="paired", country=args.country, seed=args.base_seed,
            forecast_mode=args.forecast_mode, forecast_dir=forecast_dir,
            reward_weights=reward_weights, cooling_config=cooling_config,
            data_dir=data_dir, episode_hours=int(args.seasonal_hours))
        episode_hours = int(probe.episode_hours)
        seasonal_start_hours = resolve_seasonal_start_hours(
            probe.data, args.seasonal_timestamps, episode_hours=episode_hours)
        del probe
        seasonal_labels = [
            SEASONAL_LABELS[i] if i < len(SEASONAL_LABELS) else f"seasonal_{i}"
            for i in range(len(seasonal_start_hours))]

        # If any controller uses full_season, verify its default episode
        # length matches, otherwise the two protocols will see different
        # episode windows.
        if any(p == "full_season" for p in protocol_for.values()):
            fs_probe = make_env(
                protocol="full_season", country=args.country, seed=args.base_seed,
                forecast_mode=args.forecast_mode, forecast_dir=forecast_dir,
                reward_weights=reward_weights, cooling_config=cooling_config,
                data_dir=data_dir, episode_hours=int(args.seasonal_hours))
            fs_hours = int(fs_probe.episode_hours)
            del fs_probe
            if fs_hours != episode_hours:
                print(f"[warning] full_season default episode_hours={fs_hours} "
                      f"differs from --seasonal-hours={episode_hours}; "
                      f"controllers on different protocols will see "
                      f"different episode lengths.")

    # --- Evaluate ---
    rows, all_step_traces = evaluate(
        args=args, controller_factories=controller_factories,
        protocol_for=protocol_for,
        country=args.country, forecast_mode=args.forecast_mode,
        forecast_dir=forecast_dir, reward_weights=reward_weights,
        cooling_config=cooling_config, data_dir=data_dir,
        episode_hours=episode_hours, base_seed=args.base_seed,
        episodes=args.episodes, thermal_reference_c=args.thermal_reference,
        event_margin_start_c=args.event_margin_start,
        event_margin_width_c=args.event_margin_width,
        cold_threshold_c=args.cold_threshold,
        verbose_episodes=args.verbose_episodes,
        seasonal_start_hours=seasonal_start_hours or None,
        seasonal_labels=seasonal_labels or None)

    # --- Aggregations ---
    controller_names = [name for name, _ in controller_factories]
    per_controller = {}
    for name in controller_names:
        sub = [r for r in rows if r["algorithm"] == name]
        per_controller[name] = {
            "protocol": protocol_for[name],
            "all": _aggregate_rows(sub),
            "regular": _aggregate_rows(
                [r for r in sub if r["episode_kind"] == "regular"]),
            "seasonal": _aggregate_rows(
                [r for r in sub if r["episode_kind"] == "seasonal"]),
        }

    paired, paired_aggregate = {}, {}
    if "eact" in controller_names:
        for name in controller_names:
            if name == "eact":
                continue
            pr = build_paired_rows(rows, left="eact", right=name)
            paired[name] = pr
            paired_aggregate[name] = {
                "all": _aggregate_rows(pr),
                "regular": _aggregate_rows(
                    [r for r in pr if r["episode_kind"] == "regular"]),
                "seasonal": _aggregate_rows(
                    [r for r in pr if r["episode_kind"] == "seasonal"]),
            }

    per_season = {}
    for name in controller_names:
        per_season[name] = {}
        for label in SEASONAL_LABELS:
            sub = [r for r in rows
                   if r["algorithm"] == name
                   and r["episode_kind"] == "seasonal"
                   and r["season_label"] == label]
            if sub:
                per_season[name][label] = _aggregate_rows(sub)

    # --- Save ---
    if args.output_dir is None:
        run_tag = (f"{args.episodes}ep_seed{args.base_seed}"
                   if args.episodes > 0 else f"seasonal_seed{args.base_seed}")
        # Pick a root based on whether any controller runs paired.
        any_paired = any(p == "paired" for p in protocol_for.values())
        root = DEFAULT_OUTPUT_ROOT_PAIRED if any_paired else DEFAULT_OUTPUT_ROOT_FULL
        output_dir = os.path.join(root, f"full_{args.country}_{run_tag}")
    else:
        output_dir = os.path.abspath(args.output_dir)
    os.makedirs(output_dir, exist_ok=True)

    episode_csv = os.path.join(output_dir, "episode_results.csv")
    paired_csv = os.path.join(output_dir, "paired_results.csv")
    step_trace_csv = os.path.join(output_dir, "step_traces.csv")
    summary_json = os.path.join(output_dir, "summary.json")

    save_csv(episode_csv, rows)
    for name, pr in paired.items():
        save_csv(os.path.join(
            output_dir, f"paired_results_eact_vs_{name}.csv"), pr)
    if "lora" in paired:
        save_csv(paired_csv, paired["lora"])

    if args.save_step_traces:
        save_step_traces_csv(step_trace_csv, all_step_traces,
                             max_rows=(args.step_trace_max_rows or None))
    else:
        print("[info] --no-save-step-traces set; skipping step_traces.csv")

    with open(summary_json, "w", encoding="utf-8") as f:
        json.dump({
            "default_protocol": args.protocol,
            "protocol_overrides": overrides,
            "protocol_for": protocol_for,
            "country": args.country,
            "checkpoint": (os.path.abspath(args.checkpoint)
                           if args.checkpoint else None),
            "pre_dt_checkpoint": (os.path.abspath(args.pre_dt_checkpoint)
                                  if args.pre_dt_checkpoint else None),
            "post_dt_checkpoint": (os.path.abspath(args.post_dt_checkpoint)
                                   if args.post_dt_checkpoint else None),
            "sac_checkpoint": (os.path.abspath(args.sac_checkpoint)
                               if args.sac_checkpoint else None),
            "dt_target_return": (float(args.dt_target_return)
                                 if args.dt_target_return is not None else None),
            "dt_target_percentile": float(args.dt_target_percentile),
            "dt_return_scale": float(args.dt_return_scale),
            "dt_context_len": int(args.dt_context_len),
            "controllers": controller_names,
            "random_episodes": args.episodes,
            "base_seed": args.base_seed,
            "forecast_mode": args.forecast_mode,
            "forecast_dir": forecast_dir,
            "data_dir": data_dir,
            "episode_hours": (int(episode_hours)
                              if episode_hours is not None else None),
            "thermal_reference_c": args.thermal_reference,
            "event_margin_start_c": args.event_margin_start,
            "event_margin_width_c": args.event_margin_width,
            "cold_threshold_c": args.cold_threshold,
            "seasonal": {
                "enabled": bool(seasonal_start_hours),
                "timestamps": list(args.seasonal_timestamps),
                "labels": list(seasonal_labels),
                "start_hours": [int(v) for v in seasonal_start_hours],
            },
            "per_controller": per_controller,
            "per_season": per_season,
            "paired": {"left": "eact", "aggregate": paired_aggregate},
            "step_traces": {
                "enabled": bool(args.save_step_traces),
                "n_rows": len(all_step_traces),
                "path": (step_trace_csv if args.save_step_traces else None),
                "max_rows": int(args.step_trace_max_rows),
            },
            "output_dir": output_dir,
        }, f, ensure_ascii=False, indent=2, default=str)

    # --- Console summary ---
    print("=" * 96)
    print("Full evaluation (per-controller protocol)")
    print("=" * 96)
    print(f"country                 : {args.country}")
    print(f"default protocol        : {args.protocol}")
    if overrides:
        print(f"protocol overrides      : {overrides}")
    print("effective protocol map  :")
    for name in controller_names:
        print(f"  - {name:24s} -> {protocol_for[name]}")
    print(f"random episodes         : {args.episodes}")
    if seasonal_start_hours:
        print(f"seasonal typical weeks  : {len(seasonal_start_hours)}")
        for label, ts, sh in zip(
                seasonal_labels, args.seasonal_timestamps, seasonal_start_hours):
            print(f"  - {label:8s} | {ts} | start_hour={sh}")
    print(f"total rows              : {len(rows)}")
    print(f"total step traces       : {len(all_step_traces)}")
    print("-" * 96)

    preferred_order = (
        "return", "co2_kg", "e_grid_mwh", "e_total_mwh",
        "e_cooling_mwh", "e_pump_mwh", "e_cooling_system_mwh",
        "workload_intervention_rate_pct", "mean_action_override",
        "mean_abs_workload_residual", "mean_abs_setpoint_residual",
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
        proto = protocol_for[name]
        block_all = per_controller[name]["all"]
        block_sea = per_controller[name]["seasonal"]
        print(f"\n[{name}] protocol={proto} | aggregate over ALL episodes")
        shown = set()
        for key in preferred_order:
            if key in block_all:
                print(f"  {key:36s}: {block_all[key]:.6f}")
                shown.add(key)
        for key, value in block_all.items():
            if key not in shown:
                print(f"  {key:36s}: {value:.6f}")
        if block_sea:
            print(f"\n[{name}] protocol={proto} | aggregate over SEASONAL episodes")
            shown = set()
            for key in preferred_order:
                if key in block_sea:
                    print(f"  {key:36s}: {block_sea[key]:.6f}")
                    shown.add(key)

    if seasonal_start_hours:
        print("\n" + "=" * 96)
        print("Per-season typical-week metrics (seasonal episodes only)")
        print("=" * 96)
        for name in controller_names:
            if not per_season.get(name):
                continue
            print(f"\n[{name}] protocol={protocol_for[name]}")
            print(f"  {'season':8s} | {'return':>9s} | {'P95':>6s} | "
                  f"{'Tex_h':>7s} | {'EventH':>7s} | {'SLA_MWh':>9s} | "
                  f"{'SLA_H':>6s} | {'TermUnserv':>10s}")
            for label in SEASONAL_LABELS:
                b = per_season[name].get(label)
                if not b:
                    continue
                print(f"  {label:8s} | "
                      f"{b.get('return', float('nan')):9.2f} | "
                      f"{b.get('p95_t_inlet', float('nan')):6.3f} | "
                      f"{b.get('thermal_exposure_c_h', float('nan')):7.3f} | "
                      f"{b.get('thermal_event_hours', float('nan')):7.1f} | "
                      f"{b.get('sla_violation_mwh', float('nan')):9.4f} | "
                      f"{b.get('sla_violation_hours', float('nan')):6.0f} | "
                      f"{b.get('terminal_unserved_mwh', float('nan')):10.4f}")

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