"""Ablation evaluation: LoRA-SAC V6 baseline vs LoRA-SAC ablation.

Two controllers, both ForecastJointLoRADTController, only the frozen
backbone checkpoint differs:

    lora_v6
        LoRA-SAC V6 checkpoint (trained on top of the V3 post-trained DT):
            scripts/eact-dt-lora/results/lora_thermal_rl/<C>_*/best_lora_thermal_dt.pt
        backbone = dt_<C>_post_v3_dualpool_*.pt

    lora_ablation
        LoRA-SAC ablation checkpoint (trained directly on the pretrained DT,
        skipping posttrain_decision_transformer_v3.py):
            scripts/eact-dt-lora/ablation_results/<C>_lora_sac_direct_from_pretrain/best_lora_thermal_dt.pt
        backbone = dt_<C>_seed*.pt

Reward / cooling config
-----------------------
This script preserves the same RewardWeights / CoolingConfig plumbing as
evaluate_full_season.py, so the environment dynamics are identical to the
V6 baseline evaluation. All --weight-* and --cooling-* flags are forwarded
to the env.

Episode schedule
----------------
Default: the four fixed seasonal "typical week" episodes
(2025-01-15 / 04-15 / 07-15 / 10-15 at 00:00 UTC), pinned via
env.reset(options={"start_hour": ...}). Both controllers run on the SAME
(seed, start_hour) pair for every episode.

Optional: --episodes N > 0 appends N random-start episodes with seeds
base_seed..base_seed+N-1.  Seasonal episode seeds are
base_seed + episodes + i (matching evaluate_full_season.py).

Outputs
-------
All files go to scripts/eact-dt-lora/ablation_eval_results/
    <COUNTRY>_v6_vs_ablation_<run_tag>/

    episode_results.csv                  : one row per (algorithm, episode)
    paired_results_v6_minus_ablation.csv : one row per episode, <col>_diff = v6 - ablation
    ablation_metric_summary.csv          : compact 5-metric table
                                           metric, algorithm, mean_abs, reference,
                                           reference_mean_abs, mean_diff
    step_traces.csv                      : one row per (algorithm, episode, step)
    summary.json                         : aggregate + per-controller + per-season + paired
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


# ---------------------------------------------------------------------------
# Path resolution
# ---------------------------------------------------------------------------

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, "..", ".."))
sys.path.insert(0, ROOT)


from coastaldc_env import CoastalDCContinuousEnv  # noqa: E402
from coastaldc_env.reward import RewardWeights  # noqa: E402
from coastaldc_env.swhp_cooling import CoolingConfig  # noqa: E402

from controllers.lora_joint_forecast_v6 import (  # noqa: E402
    ForecastJointLoRAActor,
    ForecastJointLoRADTController,
)


DEFAULT_FORECAST_DIR = os.path.join(
    ROOT, "results", "causal_forecasts_v3_gated_bias",
)

# V6 baseline outputs.
DEFAULT_V6_ROOT = os.path.join(SCRIPT_DIR, "results", "lora_thermal_rl")

# Ablation outputs (from ablation_lora_sac_direct_from_pretrain.py).
DEFAULT_ABLATION_ROOT = os.path.join(SCRIPT_DIR, "ablation_results")

# Isolated output for this comparison.
DEFAULT_OUTPUT_ROOT = os.path.join(SCRIPT_DIR, "ablation_eval_results")

DEFAULT_SEASONAL_TIMESTAMPS = (
    "2025-01-15 00:00:00",
    "2025-04-15 00:00:00",
    "2025-07-15 00:00:00",
    "2025-10-15 00:00:00",
)
SEASONAL_LABELS = ("winter", "spring", "summer", "autumn")

CONTROLLER_NAMES = ("lora_v6", "lora_ablation")


# =============================================================================
# Checkpoint resolution
# =============================================================================

def _newest(candidates: list[str]) -> str | None:
    if not candidates:
        return None
    candidates.sort(key=os.path.getmtime, reverse=True)
    return candidates[0]


def _resolve_v6_checkpoint(country: str) -> str | None:
    if not os.path.isdir(DEFAULT_V6_ROOT):
        return None
    candidates = sorted(
        glob.glob(
            os.path.join(DEFAULT_V6_ROOT, f"{country}_*",
                         "best_lora_thermal_dt.pt")
        )
    )
    if candidates:
        return candidates[-1]
    return _newest(
        glob.glob(
            os.path.join(DEFAULT_V6_ROOT, "*", "best_lora_thermal_dt.pt")
        )
    )


def _resolve_ablation_checkpoint(country: str) -> str | None:
    if not os.path.isdir(DEFAULT_ABLATION_ROOT):
        return None
    preferred = os.path.join(
        DEFAULT_ABLATION_ROOT,
        f"{country}_lora_sac_direct_from_pretrain",
        "best_lora_thermal_dt.pt",
    )
    if os.path.isfile(preferred):
        return preferred

    candidates = sorted(
        glob.glob(
            os.path.join(
                DEFAULT_ABLATION_ROOT, f"{country}_*",
                "best_lora_thermal_dt.pt",
            )
        )
    )
    if candidates:
        return candidates[-1]

    return _newest(
        glob.glob(
            os.path.join(DEFAULT_ABLATION_ROOT, "*",
                         "best_lora_thermal_dt.pt")
        )
    )


def _load_lora_bundle(
    checkpoint_path: str,
    device: str,
    *,
    label: str,
):
    if not os.path.isfile(checkpoint_path):
        raise FileNotFoundError(f"{label} checkpoint not found: {checkpoint_path}")

    ckpt = torch.load(checkpoint_path, map_location="cpu")
    required = (
        "actor_config",
        "v3_checkpoint",   # naming kept for loader compat
        "trainable_actor_state",
        "target_return",
        "return_scale",
    )
    missing = [k for k in required if k not in ckpt]
    if missing:
        raise KeyError(
            f"{label} checkpoint missing fields {missing} "
            f"(path={checkpoint_path})"
        )

    backbone = os.path.abspath(ckpt["v3_checkpoint"])
    if not os.path.isfile(backbone):
        raise FileNotFoundError(
            f"{label} checkpoint references a backbone that does not exist:\n"
            f"  {backbone}\n"
            f"(checkpoint: {checkpoint_path})"
        )

    actor_config = dict(ckpt["actor_config"])
    actor = ForecastJointLoRAActor(
        v3_checkpoint=backbone,
        device=device,
        **actor_config,
    )
    actor.load_trainable_state_dict(ckpt["trainable_actor_state"])
    actor.eval()

    return actor, {
        "path": os.path.abspath(checkpoint_path),
        "backbone": backbone,
        "target_return": float(ckpt["target_return"]),
        "return_scale": float(ckpt["return_scale"]),
        "format": str(ckpt.get("format", "unknown")),
        "metrics": ckpt.get("metrics", {}),
    }


# =============================================================================
# Environment
# =============================================================================

def make_env(
    *,
    country: str,
    seed: int,
    forecast_mode: str,
    forecast_dir: str | None,
    reward_weights: RewardWeights | None,
    cooling_config: CoolingConfig | None,
):
    return CoastalDCContinuousEnv(
        country=country,
        seed=seed,
        forecast_mode=forecast_mode,
        forecast_dir=forecast_dir,
        reward_weights=reward_weights,
        cooling_config=cooling_config,
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
                f"Seasonal timestamp is absent from environment data: {value}"
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

def build_controller_factories(
    args,
    bundles: dict[str, tuple[ForecastJointLoRAActor, dict]],
) -> list[tuple[str, callable]]:
    factories: list[tuple[str, callable]] = []
    deterministic = bool(args.lora_deterministic)

    for name in args.controllers:
        if name not in bundles or bundles[name] is None:
            raise ValueError(
                f"{name} requested but no checkpoint was loaded. "
                f"Pass --{'v6' if name == 'lora_v6' else 'ablation'}-checkpoint."
            )
        actor, info = bundles[name]
        target_return = info["target_return"]
        return_scale = info["return_scale"]

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

        factories.append((name, make_lora))

    return factories


# =============================================================================
# Episode runner (unchanged from evaluate_full_season.py)
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


# ---------------------------------------------------------------------------
# Seasonal summary table (compact 5-metric CSV)
# ---------------------------------------------------------------------------
SEASONAL_SUMMARY_METRICS = (
    "co2_kg",
    "e_grid_mwh",
    "e_total_mwh",
    "p95_t_inlet",
    "thermal_event_hours",
)
SEASONAL_SUMMARY_REFERENCE = "lora_v6"
SEASONAL_SUMMARY_COMPARE = ("lora_v6", "lora_ablation")


def build_paired_rows(
    rows: list[dict],
    *,
    left: str = "lora_v6",
    right: str = "lora_ablation",
) -> list[dict]:
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


def build_seasonal_summary_rows(
    per_controller: dict[str, dict],
    *,
    metrics=SEASONAL_SUMMARY_METRICS,
    reference_name: str = SEASONAL_SUMMARY_REFERENCE,
    compare_names=SEASONAL_SUMMARY_COMPARE,
) -> list[dict]:
    """Compact ablation summary: one row per (metric, algorithm).

    mean_abs           : that algorithm's seasonal mean (seasonal only)
    reference          : fixed reference name ("lora_v6")
    reference_mean_abs : reference's seasonal mean for the same metric
    mean_diff          : mean_abs - reference_mean_abs
                         (i.e. algorithm minus lora)
    """
    if reference_name not in per_controller:
        return []

    ref_block = per_controller[reference_name].get("seasonal", {})

    rows: list[dict] = []
    for metric in metrics:
        ref_val = ref_block.get(metric, float("nan"))
        try:
            ref_val_f = float(ref_val)
        except (TypeError, ValueError):
            ref_val_f = float("nan")

        for name in compare_names:
            if name not in per_controller:
                continue
            block = per_controller[name].get("seasonal", {})
            raw = block.get(metric, float("nan"))
            try:
                val_f = float(raw)
            except (TypeError, ValueError):
                val_f = float("nan")

            mean_diff = (
                val_f - ref_val_f
                if (np.isfinite(val_f) and np.isfinite(ref_val_f))
                else float("nan")
            )

            rows.append({
                "metric": metric,
                "algorithm": name,
                "mean_abs": val_f,
                "reference": reference_name,
                "reference_mean_abs": ref_val_f,
                "mean_diff": mean_diff,
            })

    return rows


# =============================================================================
# Evaluation driver
# =============================================================================

def evaluate(
    *,
    controller_factories: list[tuple[str, callable]],
    country: str,
    forecast_mode: str,
    forecast_dir: str | None,
    reward_weights: RewardWeights | None,
    cooling_config: CoolingConfig | None,
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

    if seasonal_start_hours:
        labels = (
            seasonal_labels if seasonal_labels is not None
            else [f"season_{i}" for i in range(len(seasonal_start_hours))]
        )
        # Seasonal seeds: base_seed + episodes + i
        # (matching evaluate_full_season.py)
        for i, sh in enumerate(seasonal_start_hours):
            tasks.append({
                "kind": "seasonal",
                "label": labels[i] if i < len(labels) else f"season_{i}",
                "seed": int(base_seed + episodes + i),
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
                reward_weights=reward_weights,
                cooling_config=cooling_config,
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
                    is_lora=True,  # both controllers are LoRA
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
                    f"| {algo_name:16s} | "
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
                    f"Inconsistent obs dimension in step_traces: "
                    f"expected {obs_dim}, got {obs_vec.shape[0]}"
                )
            if req_vec.shape[0] != act_dim or app_vec.shape[0] != act_dim:
                raise ValueError(
                    f"Inconsistent action dimension in step_traces: "
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
            "Ablation evaluation: LoRA-SAC V6 baseline vs LoRA-SAC ablation "
            "on the four fixed seasonal typical weeks. "
            "Reward / cooling config are forwarded exactly as in "
            "evaluate_full_season.py."
        ),
    )

    # --- Checkpoints ---------------------------------------------------
    parser.add_argument("--v6-checkpoint", default=None,
                        help=("LoRA-SAC V6 baseline checkpoint. Default: "
                              "results/lora_thermal_rl/<C>_*/best_lora_thermal_dt.pt"))
    parser.add_argument("--ablation-checkpoint", default=None,
                        help=("Ablation checkpoint. Default: "
                              "ablation_results/<C>_lora_sac_direct_from_pretrain/"
                              "best_lora_thermal_dt.pt"))

    # --- Run -----------------------------------------------------------
    parser.add_argument("--country", default="JPN")
    parser.add_argument("--episodes", type=int, default=0)
    parser.add_argument("--base-seed", type=int, default=9000)

    parser.add_argument(
        "--controllers",
        nargs="*",
        choices=CONTROLLER_NAMES,
        default=list(CONTROLLER_NAMES),
    )
    parser.add_argument(
        "--lora-deterministic",
        action=argparse.BooleanOptionalAction,
        default=True,
    )

    parser.add_argument("--thermal-reference", type=float, default=27.0)
    parser.add_argument("--event-margin-start", type=float, default=26.9)
    parser.add_argument("--event-margin-width", type=float, default=0.1)
    parser.add_argument("--cold-threshold", type=float, default=25.5)

    parser.add_argument("--forecast-mode",
                        choices=("oracle", "causal", "none"),
                        default="causal")
    parser.add_argument("--forecast-dir", default=None)

    # --- Reward weights (forwarded to env, same as evaluate_full_season.py) ---
    parser.add_argument("--weight-grid", type=float, default=1.0)
    parser.add_argument("--weight-co2", type=float, default=2.0)
    parser.add_argument("--weight-total", type=float, default=0.2)
    parser.add_argument("--weight-smooth", type=float, default=0.5)

    # --- Cooling config (forwarded to env) ------------------------------
    parser.add_argument("--cooling-conductance-multiplier",
                        type=float, default=1.0)
    parser.add_argument("--thermal-safety-shield",
                        action=argparse.BooleanOptionalAction,
                        default=True)

    # --- Seasonal schedule ----------------------------------------------
    parser.add_argument(
        "--seasonal-timestamps", nargs="*",
        default=list(DEFAULT_SEASONAL_TIMESTAMPS),
    )
    parser.add_argument("--no-seasonal", action="store_true")

    # --- Output / tracing ----------------------------------------------
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
    if not args.controllers:
        raise ValueError("--controllers must contain at least one entry")

    # ------------------------------------------------------------------
    # Device
    # ------------------------------------------------------------------
    device = "cuda" if torch.cuda.is_available() else "cpu"
    args.device = device

    # ------------------------------------------------------------------
    # Auto-resolve checkpoints
    # ------------------------------------------------------------------
    if "lora_v6" in args.controllers and not args.v6_checkpoint:
        resolved = _resolve_v6_checkpoint(args.country)
        if resolved is None:
            raise FileNotFoundError(
                "--v6-checkpoint is required when 'lora_v6' is in "
                "--controllers, and no best_lora_thermal_dt.pt was found "
                f"under {DEFAULT_V6_ROOT}/<COUNTRY>_*/. "
                "Train it with posttrain_dt_lora_sac_thermal_v6_backlog_season.py."
            )
        args.v6_checkpoint = resolved
        print(f"[info] auto-resolved V6 checkpoint: {resolved}")

    if "lora_ablation" in args.controllers and not args.ablation_checkpoint:
        resolved = _resolve_ablation_checkpoint(args.country)
        if resolved is None:
            raise FileNotFoundError(
                "--ablation-checkpoint is required when 'lora_ablation' is in "
                "--controllers, and no best_lora_thermal_dt.pt was found "
                f"under {DEFAULT_ABLATION_ROOT}/<COUNTRY>_*/. "
                "Train it with ablation_lora_sac_direct_from_pretrain.py."
            )
        args.ablation_checkpoint = resolved
        print(f"[info] auto-resolved ablation checkpoint: {resolved}")

    # ------------------------------------------------------------------
    # Forecast dir
    # ------------------------------------------------------------------
    forecast_dir = (
        os.path.abspath(args.forecast_dir)
        if args.forecast_dir
        else DEFAULT_FORECAST_DIR
    )
    args.forecast_dir = forecast_dir

    if args.forecast_mode == "causal":
        required = [
            os.path.join(forecast_dir, f"causal_ridge_{args.country}.npz"),
        ]
        missing = [p for p in required if not os.path.isfile(p)]
        if missing:
            raise FileNotFoundError(
                "Causal forecast files are missing:\n  "
                + "\n  ".join(missing)
            )

    # ------------------------------------------------------------------
    # Reward / cooling config (same as evaluate_full_season.py)
    # ------------------------------------------------------------------
    reward_weights = RewardWeights(
        w_grid=args.weight_grid,
        w_co2=args.weight_co2,
        w_total=args.weight_total,
        w_smooth=args.weight_smooth,
    )

    nominal_cooling = CoolingConfig()
    cooling_config = CoolingConfig(
        conductance_mw_per_k=(
            nominal_cooling.conductance_mw_per_k
            * float(args.cooling_conductance_multiplier)
        ),
        enforce_thermal_safety=bool(args.thermal_safety_shield),
    )

    # ------------------------------------------------------------------
    # Load both LoRA bundles
    # ------------------------------------------------------------------
    bundles: dict[str, tuple[ForecastJointLoRAActor, dict]] = {}

    if "lora_v6" in args.controllers:
        actor, info = _load_lora_bundle(
            os.path.abspath(args.v6_checkpoint), device, label="lora_v6",
        )
        bundles["lora_v6"] = (actor, info)
        print(f"lora_v6 checkpoint : {info['path']}")
        print(f"lora_v6 backbone   : {info['backbone']}")
        print(f"lora_v6 target     : {info['target_return']:.3f}")
        print(f"lora_v6 scale      : {info['return_scale']:.3f}")
        print(f"lora_v6 format     : {info['format']}")

    if "lora_ablation" in args.controllers:
        actor, info = _load_lora_bundle(
            os.path.abspath(args.ablation_checkpoint), device,
            label="lora_ablation",
        )
        bundles["lora_ablation"] = (actor, info)
        print(f"lora_ablation ckpt : {info['path']}")
        print(f"lora_ablation bb   : {info['backbone']}")
        print(f"lora_ablation tgt  : {info['target_return']:.3f}")
        print(f"lora_ablation scale: {info['return_scale']:.3f}")
        print(f"lora_ablation fmt  : {info['format']}")

    if (
        "lora_v6" in bundles
        and "lora_ablation" in bundles
        and bundles["lora_v6"][1]["path"] == bundles["lora_ablation"][1]["path"]
    ):
        print(
            "[WARNING] lora_v6 and lora_ablation resolved to the SAME "
            "checkpoint file. The comparison will be trivial."
        )

    controller_factories = build_controller_factories(args, bundles)

    # ------------------------------------------------------------------
    # Resolve seasonal start hours
    # ------------------------------------------------------------------
    seasonal_start_hours: list[int] = []
    seasonal_labels: list[str] = []

    if not args.no_seasonal:
        probe_env = make_env(
            country=args.country,
            seed=args.base_seed,
            forecast_mode=args.forecast_mode,
            forecast_dir=forecast_dir,
            reward_weights=reward_weights,
            cooling_config=cooling_config,
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
    # Evaluate
    # ------------------------------------------------------------------
    rows, all_step_traces = evaluate(
        controller_factories=controller_factories,
        country=args.country,
        forecast_mode=args.forecast_mode,
        forecast_dir=forecast_dir,
        reward_weights=reward_weights,
        cooling_config=cooling_config,
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
    # Aggregations
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

    paired_rows = build_paired_rows(rows, left="lora_v6", right="lora_ablation")
    paired_aggregate = {
        "all": _aggregate_rows(paired_rows),
        "regular": _aggregate_rows(
            [r for r in paired_rows if r["episode_kind"] == "regular"]
        ),
        "seasonal": _aggregate_rows(
            [r for r in paired_rows if r["episode_kind"] == "seasonal"]
        ),
    }

    seasonal_summary_rows = build_seasonal_summary_rows(per_controller)

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
    # Save
    # ------------------------------------------------------------------
    if args.output_dir is None:
        if args.episodes > 0:
            run_tag = f"{args.episodes}ep_seed{args.base_seed}"
        else:
            run_tag = f"seasonal_seed{args.base_seed}"
        output_dir = os.path.join(
            DEFAULT_OUTPUT_ROOT,
            f"{args.country}_v6_vs_ablation_{run_tag}",
        )
    else:
        output_dir = os.path.abspath(args.output_dir)

    os.makedirs(output_dir, exist_ok=True)

    episode_csv = os.path.join(output_dir, "episode_results.csv")
    paired_csv = os.path.join(output_dir, "paired_results_v6_minus_ablation.csv")
    seasonal_summary_csv = os.path.join(
        output_dir, "ablation_metric_summary.csv",
    )
    step_trace_csv = os.path.join(output_dir, "step_traces.csv")
    summary_json = os.path.join(output_dir, "summary.json")

    save_csv(episode_csv, rows)
    save_csv(paired_csv, paired_rows)
    save_csv(seasonal_summary_csv, seasonal_summary_rows)

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
                "experiment": "ablation_vs_v6",
                "country": args.country,
                "v6_checkpoint": (
                    os.path.abspath(args.v6_checkpoint)
                    if args.v6_checkpoint else None
                ),
                "v6_backbone": (
                    bundles["lora_v6"][1]["backbone"]
                    if "lora_v6" in bundles else None
                ),
                "ablation_checkpoint": (
                    os.path.abspath(args.ablation_checkpoint)
                    if args.ablation_checkpoint else None
                ),
                "ablation_backbone": (
                    bundles["lora_ablation"][1]["backbone"]
                    if "lora_ablation" in bundles else None
                ),
                "controllers": controller_names,
                "random_episodes": args.episodes,
                "base_seed": args.base_seed,
                "forecast_mode": args.forecast_mode,
                "forecast_dir": forecast_dir,
                "reward_weights": {
                    "w_grid": float(args.weight_grid),
                    "w_co2": float(args.weight_co2),
                    "w_total": float(args.weight_total),
                    "w_smooth": float(args.weight_smooth),
                },
                "cooling_config": {
                    "conductance_mw_per_k": float(
                        cooling_config.conductance_mw_per_k
                    ),
                    "enforce_thermal_safety": bool(
                        cooling_config.enforce_thermal_safety
                    ),
                    "cooling_conductance_multiplier": float(
                        args.cooling_conductance_multiplier
                    ),
                },
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
                "paired": {
                    "left": "lora_v6",
                    "right": "lora_ablation",
                    "diff_definition": "lora_v6 - lora_ablation",
                    "aggregate": paired_aggregate,
                },
                "seasonal_summary": {
                    "metrics": list(SEASONAL_SUMMARY_METRICS),
                    "reference": SEASONAL_SUMMARY_REFERENCE,
                    "diff_definition": "algorithm - reference",
                    "rows": seasonal_summary_rows,
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
    # Console summary
    # ------------------------------------------------------------------
    print("=" * 96)
    print("Ablation evaluation: LoRA-SAC V6 baseline vs LoRA-SAC ablation")
    print("(both ForecastJointLoRADTController; only the frozen backbone differs)")
    print("(reward weights / cooling config forwarded as in evaluate_full_season.py)")
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
            for key in preferred_order:
                if key in block_sea:
                    print(f"  {key:36s}: {block_sea[key]:.6f}")

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

    print("\n" + "=" * 96)
    print("Paired comparison: lora_v6 minus lora_ablation "
          "(positive = V6 larger)")
    print("=" * 96)
    for col in PAIRED_METRIC_COLUMNS:
        diff_key = f"{col}_diff"
        if diff_key in paired_aggregate["all"]:
            print(
                f"  {col:36s}: {paired_aggregate['all'][diff_key]:+.6f}"
            )

    if seasonal_summary_rows:
        print("\n" + "=" * 96)
        print(
            "Seasonal summary (reference = lora_v6, "
            "mean_diff = algorithm - lora_v6)"
        )
        print("=" * 96)
        print(
            f"  {'metric':20s} | {'algorithm':16s} | {'mean_abs':>12s} | "
            f"{'reference':10s} | {'ref_mean_abs':>12s} | {'mean_diff':>12s}"
        )
        for row in seasonal_summary_rows:
            print(
                f"  {row['metric']:20s} | "
                f"{row['algorithm']:16s} | "
                f"{row['mean_abs']:>12.4f} | "
                f"{row['reference']:10s} | "
                f"{row['reference_mean_abs']:>12.4f} | "
                f"{row['mean_diff']:>+12.4f}"
            )
    else:
        print(
            "\n[info] seasonal summary table is empty "
            "(seasonal episodes were not evaluated, or reference is missing)."
        )

    print("-" * 96)
    print(f"episode CSV         : {episode_csv}")
    print(f"paired CSV          : {paired_csv}")
    print(f"metric summary CSV  : {seasonal_summary_csv}")
    if args.save_step_traces:
        print(f"step CSV            : {step_trace_csv}")
    print(f"summary JSON        : {summary_json}")
    print("=" * 96)


if __name__ == "__main__":
    main()