"""ABLATION: full-forecast joint LoRA-SAC starting DIRECTLY from the
pretrained Decision Transformer, SKIPPING posttrain_decision_transformer_v3.py.

Design
------
This is a controlled ablation of
    posttrain_dt_lora_sac_thermal_v6_backlog_season.py

The ONLY experimental variable is the LoRA-SAC backbone checkpoint:

    baseline V6 : dt_<COUNTRY>_post_v3_dualpool_eact*_seed*.pt   (V3 output)
    ablation    : dt_<COUNTRY>_seed*.pt                          (pretrained)

Everything else is copied verbatim from V6:
    - ContextReplayBuffer
    - 5x QNetwork critics (reward / exposure / true-event / margin)
    - dual ascent
    - shaped reward (cold / mechanical-cooling / backlog)
    - seasonal fixed-start-hour evaluation
    - thermal-feasibility-first checkpoint selection

Isolation
---------
Outputs go under:
    scripts/eact-dt-lora/ablation_results/<COUNTRY>_lora_sac_direct_from_pretrain/

The script NEVER writes into:
    scripts/eact-dt-lora/results/models/     (pretrained / V3 checkpoints)
    scripts/eact-dt-lora/results/lora_thermal_rl/  (V6 outputs)

Path layout
-----------
    SCRIPT_DIR = scripts/eact-dt-lora/
    ROOT       = project root (two levels up)

    Pretrained backbone search dir:
        SCRIPT_DIR/results/models/dt_<COUNTRY>_seed*.pt
        (this is where train_decision_transformer.py saves)

    Trajectory dir (shared):
        ROOT/data/trajectories/

    Forecast dir (shared):
        ROOT/results/causal_forecasts_v3_gated_bias/

    Output dir (ABLATION, isolated):
        SCRIPT_DIR/ablation_results/

Examples (run from scripts/eact-dt-lora/)
-----------------------------------------
# Default: resolve newest dt_<COUNTRY>_seed*.pt under results/models/
python ablation_lora_sac_direct_from_pretrain.py --country JPN

# Explicit pretrained checkpoint
python ablation_lora_sac_direct_from_pretrain.py ^
    --country JPN ^
    --pretrain-model .\results\models\dt_JPN_seed0.pt

# Full aligned command (with your custom hyper-parameters)
python ablation_lora_sac_direct_from_pretrain.py ^
    --country JPN ^
    --setpoint-residual-limit 1 ^
    --pump-residual-limit 1 ^
    --workload-residual-limit 1 ^
    --mechanical-cooling-penalty-weight 0.2 ^
    --backlog-penalty-weight 0.05 ^
    --target-band-low 26.3 ^
    --cold-penalty-weight 0.25 ^
    --margin-aux-beta 0.01 ^
    --train-episodes 300 ^
    --selection-event-hours-target 1.0
"""

from __future__ import annotations

import argparse
import copy
import csv
import glob
import json
import os
import random
import sys
from dataclasses import dataclass

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F


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
# SCRIPT_DIR is used for BOTH the pretrained DT lookup and the ablation
# output tree, so the whole ablation keeps its artifacts under one folder.
# ---------------------------------------------------------------------------

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, "..", ".."))
sys.path.insert(0, ROOT)

from coastaldc_env.continuous_env import CoastalDCContinuousEnv  # noqa: E402
from coastaldc_env.rollout import load_trajectories  # noqa: E402
from controllers.lora_thermal_dt import DTContext  # noqa: E402
from controllers.lora_joint_forecast_v6 import (  # noqa: E402
    FULL_FORECAST_DIM,
    ForecastJointLoRAActor,
    ForecastJointLoRADTController,
    build_full_forecast_features,
)


# Pretrained DT lives under scripts/eact-dt-lora/results/models/
# (same tree as V3 and LoRA-SAC outputs; NOT ROOT/results/models, which
# only holds causal forecasts).
DEFAULT_MODEL_DIR = os.path.join(SCRIPT_DIR, "results", "models")

# Shared data & forecast artifacts stay under ROOT.
DEFAULT_TRAJ_DIR = os.path.join(ROOT, "data", "trajectories")
DEFAULT_FORECAST_DIR = os.path.join(
    ROOT,
    "results",
    "causal_forecasts_v3_gated_bias",
)

# ABLATION output root: isolated so we never touch results/ or V3 checkpoints.
DEFAULT_OUTPUT_ROOT = os.path.join(SCRIPT_DIR, "ablation_results")

DEFAULT_SEASONAL_TIMESTAMPS = (
    "2025-01-15 00:00:00",
    "2025-04-15 00:00:00",
    "2025-07-15 00:00:00",
    "2025-10-15 00:00:00",
)

SEASONAL_LABELS = (
    "winter",
    "spring",
    "summer",
    "autumn",
)


# =============================================================================
# ABLATION: default backbone = PRETRAINED dt_<COUNTRY>_seed*.pt
# =============================================================================

def _resolve_default_pretrain_model(country: str) -> str:
    """Newest pretrained DT checkpoint produced by train_decision_transformer.py.

    Search dir: scripts/eact-dt-lora/results/models/

    We deliberately DO NOT look at any *_post_v3_*.pt file, because the
    entire point of this ablation is to skip V3 post-training. The glob
    pattern already only matches ``dt_<country>_seed*.pt``, and we also
    defensively filter out any ``_post_v3_`` filenames.
    """
    candidates = glob.glob(
        os.path.join(DEFAULT_MODEL_DIR, f"dt_{country}_seed*.pt")
    )
    candidates = [
        c for c in candidates
        if "_post_v3_" not in os.path.basename(c)
    ]

    if candidates:
        candidates.sort(key=os.path.getmtime, reverse=True)
        chosen = candidates[0]
        print(f"[ablation] pretrained backbone resolved -> {chosen}")
        print(f"[ablation]   (search dir: {DEFAULT_MODEL_DIR})")
        return chosen

    # Informative fallback path so the FileNotFoundError names the file we
    # actually expected.
    return os.path.join(DEFAULT_MODEL_DIR, f"dt_{country}_seed0.pt")


def _resolve_default_traj_path(country: str) -> str:
    """Prefer the EACT expert pool; fall back to the legacy mpc pool."""
    preferred = os.path.join(DEFAULT_TRAJ_DIR, f"eact_{country}.npz")
    if os.path.isfile(preferred):
        return preferred

    legacy = os.path.join(DEFAULT_TRAJ_DIR, f"mpc_{country}.npz")
    if os.path.isfile(legacy):
        print(
            f"[info] eact_{country}.npz not found; "
            f"falling back to mpc_{country}.npz"
        )
        return legacy

    return preferred


# =============================================================================
# Reproducibility
# =============================================================================

def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# =============================================================================
# Seasonal start-hour resolution (verbatim from V6)
# =============================================================================

def resolve_seasonal_start_hours(
    data: pd.DataFrame,
    timestamps,
    *,
    episode_hours: int,
    min_start: int = 0,
) -> list[int]:
    """Map seasonal start timestamps to start-hour indices in the data."""
    if "timestamp" not in data.columns:
        raise ValueError(
            "Seasonal start timestamps require a 'timestamp' column "
            "in the environment data."
        )

    parsed = pd.to_datetime(data["timestamp"], errors="raise")
    if parsed.duplicated().any():
        raise ValueError(
            "Environment data contain duplicate timestamps."
        )

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
# Replay buffer (verbatim from V6)
# =============================================================================

class ContextReplayBuffer:
    """Memory-efficient replay storing fixed-K DT contexts in float16."""

    def __init__(
        self,
        *,
        capacity: int,
        context_len: int,
        obs_dim: int,
        forecast_dim: int,
    ):
        self.capacity = int(capacity)
        self.K = int(context_len)
        self.obs_dim = int(obs_dim)
        self.forecast_dim = int(forecast_dim)

        self.states = np.zeros(
            (capacity, context_len, obs_dim),
            dtype=np.float16,
        )
        self.actions_ctx = np.zeros(
            (capacity, context_len, 3),
            dtype=np.float16,
        )
        self.rtgs = np.zeros(
            (capacity, context_len, 1),
            dtype=np.float16,
        )
        self.timesteps = np.zeros(
            (capacity, context_len),
            dtype=np.int16,
        )
        self.lengths = np.zeros(capacity, dtype=np.uint8)

        self.next_states = np.zeros_like(self.states)
        self.next_actions_ctx = np.zeros_like(self.actions_ctx)
        self.next_rtgs = np.zeros_like(self.rtgs)
        self.next_timesteps = np.zeros_like(self.timesteps)
        self.next_lengths = np.zeros_like(self.lengths)

        self.forecast_features = np.zeros(
            (capacity, forecast_dim),
            dtype=np.float16,
        )
        self.next_forecast_features = np.zeros_like(self.forecast_features)

        self.env_actions = np.zeros((capacity, 3), dtype=np.float32)
        self.rewards = np.zeros((capacity, 1), dtype=np.float32)
        self.exposure_costs = np.zeros((capacity, 1), dtype=np.float32)
        self.event_costs = np.zeros((capacity, 1), dtype=np.float32)
        self.margin_costs = np.zeros((capacity, 1), dtype=np.float32)
        self.cold_costs = np.zeros((capacity, 1), dtype=np.float32)
        self.mechanical_cooling_costs = np.zeros(
            (capacity, 1), dtype=np.float32)
        self.backlog_costs = np.zeros((capacity, 1), dtype=np.float32)
        self.dones = np.zeros((capacity, 1), dtype=np.float32)

        self.ptr = 0
        self.size = 0

    def __len__(self):
        return self.size

    @staticmethod
    def _write_context(
        destination,
        action_destination,
        rtg_destination,
        timestep_destination,
        length_destination,
        idx,
        context: DTContext,
    ):
        destination[idx] = context.states.astype(np.float16)
        action_destination[idx] = context.actions.astype(np.float16)
        rtg_destination[idx] = context.rtgs.astype(np.float16)
        timestep_destination[idx] = context.timesteps.astype(np.int16)
        length_destination[idx] = np.uint8(context.length)

    def add(
        self,
        *,
        context: DTContext,
        forecast_features: np.ndarray,
        env_action: np.ndarray,
        reward: float,
        exposure_cost: float,
        event_cost: float,
        margin_cost: float,
        cold_cost: float,
        mechanical_cooling_cost: float,
        backlog_cost: float,
        next_context: DTContext,
        next_forecast_features: np.ndarray,
        done: bool,
    ):
        i = self.ptr

        self._write_context(
            self.states,
            self.actions_ctx,
            self.rtgs,
            self.timesteps,
            self.lengths,
            i,
            context,
        )
        self._write_context(
            self.next_states,
            self.next_actions_ctx,
            self.next_rtgs,
            self.next_timesteps,
            self.next_lengths,
            i,
            next_context,
        )

        self.forecast_features[i] = np.asarray(
            forecast_features, dtype=np.float16,
        ).reshape(self.forecast_dim)
        self.next_forecast_features[i] = np.asarray(
            next_forecast_features, dtype=np.float16,
        ).reshape(self.forecast_dim)

        self.env_actions[i] = np.asarray(env_action, dtype=np.float32)
        self.rewards[i, 0] = float(reward)
        self.exposure_costs[i, 0] = float(exposure_cost)
        self.event_costs[i, 0] = float(event_cost)
        self.margin_costs[i, 0] = float(margin_cost)
        self.cold_costs[i, 0] = float(cold_cost)
        self.mechanical_cooling_costs[i, 0] = float(mechanical_cooling_cost)
        self.backlog_costs[i, 0] = float(backlog_cost)
        self.dones[i, 0] = float(done)

        self.ptr = (self.ptr + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)

    def sample(self, batch_size: int, *, device: str) -> dict[str, torch.Tensor]:
        idx = np.random.randint(0, self.size, size=batch_size)

        def f32(array):
            return torch.as_tensor(
                array[idx], dtype=torch.float32, device=device,
            )

        return {
            "states": f32(self.states),
            "actions_ctx": f32(self.actions_ctx),
            "rtgs": f32(self.rtgs),
            "timesteps": torch.as_tensor(
                self.timesteps[idx], dtype=torch.int64, device=device,
            ),
            "lengths": torch.as_tensor(
                self.lengths[idx], dtype=torch.int64, device=device,
            ),
            "next_states": f32(self.next_states),
            "next_actions_ctx": f32(self.next_actions_ctx),
            "next_rtgs": f32(self.next_rtgs),
            "next_timesteps": torch.as_tensor(
                self.next_timesteps[idx], dtype=torch.int64, device=device,
            ),
            "next_lengths": torch.as_tensor(
                self.next_lengths[idx], dtype=torch.int64, device=device,
            ),
            "forecast_features": f32(self.forecast_features),
            "next_forecast_features": f32(self.next_forecast_features),
            "env_actions": f32(self.env_actions),
            "rewards": f32(self.rewards),
            "exposure_costs": f32(self.exposure_costs),
            "event_costs": f32(self.event_costs),
            "margin_costs": f32(self.margin_costs),
            "cold_costs": f32(self.cold_costs),
            "mechanical_cooling_costs": f32(self.mechanical_cooling_costs),
            "backlog_costs": f32(self.backlog_costs),
            "dones": f32(self.dones),
        }


# =============================================================================
# Critics (verbatim from V6)
# =============================================================================

class QNetwork(nn.Module):
    def __init__(self, obs_dim: int, act_dim: int = 3, hidden_dim: int = 256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim + act_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, obs: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        return self.net(torch.cat([obs, action], dim=-1))


def soft_update(target: nn.Module, source: nn.Module, tau: float) -> None:
    with torch.no_grad():
        for target_p, source_p in zip(target.parameters(), source.parameters()):
            target_p.data.mul_(1.0 - tau).add_(tau * source_p.data)


# =============================================================================
# Dual variables (verbatim from V6)
# =============================================================================

@dataclass
class DualState:
    exposure_lambda: float
    event_lambda: float
    exposure_gap_normalized: float
    event_gap_normalized: float


class EpisodeLagrangeController:
    """Symmetric clipped episode-level dual ascent."""

    def __init__(
        self,
        *,
        exposure_budget: float,
        event_hours_budget: float,
        exposure_gap_scale: float,
        event_gap_scale: float,
        dual_lr: float,
        init_exposure_lambda: float,
        init_event_lambda: float,
        max_lambda: float,
    ):
        if exposure_budget < 0 or event_hours_budget < 0:
            raise ValueError("Thermal budgets must be non-negative")
        if exposure_gap_scale <= 0 or event_gap_scale <= 0:
            raise ValueError("Dual gap scales must be positive")
        if dual_lr < 0:
            raise ValueError("dual_lr must be non-negative")

        self.exposure_budget = float(exposure_budget)
        self.event_hours_budget = float(event_hours_budget)
        self.exposure_gap_scale = float(exposure_gap_scale)
        self.event_gap_scale = float(event_gap_scale)
        self.dual_lr = float(dual_lr)
        self.max_lambda = float(max_lambda)

        self.exposure_lambda = float(init_exposure_lambda)
        self.event_lambda = float(init_event_lambda)

    def update(
        self,
        *,
        episode_exposure: float,
        episode_event_hours: float,
        episode_length: int | None = None,
    ) -> DualState:
        del episode_length

        exp_gap = float(
            np.clip(
                (float(episode_exposure) - self.exposure_budget)
                / self.exposure_gap_scale,
                -1.0,
                1.0,
            )
        )
        event_gap = float(
            np.clip(
                (float(episode_event_hours) - self.event_hours_budget)
                / self.event_gap_scale,
                -1.0,
                1.0,
            )
        )

        self.exposure_lambda = float(
            np.clip(
                self.exposure_lambda + self.dual_lr * exp_gap,
                0.0,
                self.max_lambda,
            )
        )
        self.event_lambda = float(
            np.clip(
                self.event_lambda + self.dual_lr * event_gap,
                0.0,
                self.max_lambda,
            )
        )

        return DualState(
            exposure_lambda=self.exposure_lambda,
            event_lambda=self.event_lambda,
            exposure_gap_normalized=exp_gap,
            event_gap_normalized=event_gap,
        )


# =============================================================================
# Constrained SAC (verbatim from V6)
# =============================================================================

class ConstrainedSAC:
    def __init__(
        self,
        *,
        actor: ForecastJointLoRAActor,
        obs_dim: int,
        forecast_dim: int,
        device: str,
        actor_lr: float,
        critic_lr: float,
        alpha_lr: float,
        gamma: float,
        tau: float,
        target_entropy: float,
        reward_scale: float,
        cold_penalty_weight: float,
        mechanical_cooling_penalty_weight: float,
        backlog_penalty_weight: float,
        residual_reg_beta: float,
        margin_aux_beta: float,
    ):
        self.actor = actor
        self.device = device
        self.obs_dim = int(obs_dim)
        self.forecast_dim = int(forecast_dim)
        self.critic_state_dim = self.obs_dim + self.forecast_dim
        self.gamma = float(gamma)
        self.tau = float(tau)
        self.reward_scale = float(reward_scale)
        self.target_entropy = float(target_entropy)
        self.cold_penalty_weight = float(cold_penalty_weight)
        self.mechanical_cooling_penalty_weight = float(
            mechanical_cooling_penalty_weight
        )
        if self.mechanical_cooling_penalty_weight < 0:
            raise ValueError(
                "mechanical_cooling_penalty_weight must be non-negative"
            )
        self.backlog_penalty_weight = float(backlog_penalty_weight)
        if self.backlog_penalty_weight < 0:
            raise ValueError("backlog_penalty_weight must be non-negative")
        self.residual_reg_beta = float(residual_reg_beta)
        self.margin_aux_beta = float(margin_aux_beta)
        if self.margin_aux_beta < 0:
            raise ValueError("margin_aux_beta must be non-negative")

        def critic_pair():
            q1 = QNetwork(self.critic_state_dim).to(device)
            q2 = QNetwork(self.critic_state_dim).to(device)
            return q1, q2

        self.q1, self.q2 = critic_pair()
        self.q1_t = copy.deepcopy(self.q1)
        self.q2_t = copy.deepcopy(self.q2)

        self.ce1, self.ce2 = critic_pair()
        self.ce1_t = copy.deepcopy(self.ce1)
        self.ce2_t = copy.deepcopy(self.ce2)

        self.cv1, self.cv2 = critic_pair()
        self.cv1_t = copy.deepcopy(self.cv1)
        self.cv2_t = copy.deepcopy(self.cv2)

        self.cm1, self.cm2 = critic_pair()
        self.cm1_t = copy.deepcopy(self.cm1)
        self.cm2_t = copy.deepcopy(self.cm2)

        for module in (
            self.q1_t, self.q2_t,
            self.ce1_t, self.ce2_t,
            self.cv1_t, self.cv2_t,
            self.cm1_t, self.cm2_t,
        ):
            for p in module.parameters():
                p.requires_grad = False

        self.actor_optimizer = torch.optim.Adam(
            self.actor.trainable_parameters(), lr=actor_lr,
        )
        self.q_optimizer = torch.optim.Adam(
            list(self.q1.parameters()) + list(self.q2.parameters()),
            lr=critic_lr,
        )
        self.ce_optimizer = torch.optim.Adam(
            list(self.ce1.parameters()) + list(self.ce2.parameters()),
            lr=critic_lr,
        )
        self.cv_optimizer = torch.optim.Adam(
            list(self.cv1.parameters()) + list(self.cv2.parameters()),
            lr=critic_lr,
        )
        self.cm_optimizer = torch.optim.Adam(
            list(self.cm1.parameters()) + list(self.cm2.parameters()),
            lr=critic_lr,
        )

        self.log_alpha = torch.tensor(
            np.log(0.1), dtype=torch.float32, device=device,
            requires_grad=True,
        )
        self.alpha_optimizer = torch.optim.Adam([self.log_alpha], lr=alpha_lr)

    @property
    def alpha(self):
        return self.log_alpha.exp()

    @staticmethod
    def _last_obs(states: torch.Tensor) -> torch.Tensor:
        return states[:, -1, :]

    def _actor_sample(self, batch, *, next_context, deterministic=False):
        prefix = "next_" if next_context else ""
        return self.actor.sample(
            states=batch[prefix + "states"],
            actions=batch[prefix + "actions_ctx"],
            rtgs=batch[prefix + "rtgs"],
            timesteps=batch[prefix + "timesteps"],
            lengths=batch[prefix + "lengths"],
            forecast_features=batch[prefix + "forecast_features"],
            deterministic=deterministic,
        )

    def update(self, batch, *, exposure_lambda, event_lambda):
        self.actor.train(True)

        obs = torch.cat(
            [self._last_obs(batch["states"]), batch["forecast_features"]],
            dim=-1,
        )
        next_obs = torch.cat(
            [self._last_obs(batch["next_states"]),
             batch["next_forecast_features"]],
            dim=-1,
        )

        action = batch["env_actions"]
        reward = batch["rewards"] * self.reward_scale
        exposure_cost = batch["exposure_costs"]
        event_cost = batch["event_costs"]
        margin_cost = batch["margin_costs"]
        cold_cost = batch["cold_costs"]
        mechanical_cooling_cost = batch["mechanical_cooling_costs"]
        backlog_cost = batch["backlog_costs"]
        done = batch["dones"]

        shaped_reward = (
            reward
            - self.cold_penalty_weight * cold_cost
            - self.mechanical_cooling_penalty_weight * mechanical_cooling_cost
            - self.backlog_penalty_weight * backlog_cost
        )

        with torch.no_grad():
            next_action, next_log_prob, _, _ = self._actor_sample(
                batch, next_context=True, deterministic=False,
            )

            next_q = torch.minimum(
                self.q1_t(next_obs, next_action),
                self.q2_t(next_obs, next_action),
            )
            reward_target = (
                shaped_reward
                + self.gamma * (1.0 - done)
                * (next_q - self.alpha.detach() * next_log_prob)
            )

            next_ce = torch.clamp_min(
                torch.maximum(
                    self.ce1_t(next_obs, next_action),
                    self.ce2_t(next_obs, next_action),
                ), 0.0,
            )
            exposure_target = (
                exposure_cost + self.gamma * (1.0 - done) * next_ce
            )

            next_cv = torch.clamp_min(
                torch.maximum(
                    self.cv1_t(next_obs, next_action),
                    self.cv2_t(next_obs, next_action),
                ), 0.0,
            )
            event_target = (
                event_cost + self.gamma * (1.0 - done) * next_cv
            )

            next_cm = torch.clamp_min(
                torch.maximum(
                    self.cm1_t(next_obs, next_action),
                    self.cm2_t(next_obs, next_action),
                ), 0.0,
            )
            margin_target = (
                margin_cost + self.gamma * (1.0 - done) * next_cm
            )

        # Reward critic
        q1 = self.q1(obs, action)
        q2 = self.q2(obs, action)
        q_loss = F.mse_loss(q1, reward_target) + F.mse_loss(q2, reward_target)
        self.q_optimizer.zero_grad()
        q_loss.backward()
        self.q_optimizer.step()

        # Exposure critic
        ce1 = self.ce1(obs, action)
        ce2 = self.ce2(obs, action)
        ce_loss = F.mse_loss(ce1, exposure_target) + F.mse_loss(ce2, exposure_target)
        self.ce_optimizer.zero_grad()
        ce_loss.backward()
        self.ce_optimizer.step()

        # True-event critic
        cv1 = self.cv1(obs, action)
        cv2 = self.cv2(obs, action)
        cv_loss = F.mse_loss(cv1, event_target) + F.mse_loss(cv2, event_target)
        self.cv_optimizer.zero_grad()
        cv_loss.backward()
        self.cv_optimizer.step()

        # Margin auxiliary critic
        cm1 = self.cm1(obs, action)
        cm2 = self.cm2(obs, action)
        cm_loss = F.mse_loss(cm1, margin_target) + F.mse_loss(cm2, margin_target)
        self.cm_optimizer.zero_grad()
        cm_loss.backward()
        self.cm_optimizer.step()

        # Actor
        policy_action, log_prob, _base_action, residual = self._actor_sample(
            batch, next_context=False, deterministic=False,
        )
        q_pi = torch.minimum(
            self.q1(obs, policy_action), self.q2(obs, policy_action),
        )
        exposure_pi = torch.clamp_min(
            torch.maximum(
                self.ce1(obs, policy_action), self.ce2(obs, policy_action),
            ), 0.0,
        )
        event_pi = torch.clamp_min(
            torch.maximum(
                self.cv1(obs, policy_action), self.cv2(obs, policy_action),
            ), 0.0,
        )
        margin_pi = torch.clamp_min(
            torch.maximum(
                self.cm1(obs, policy_action), self.cm2(obs, policy_action),
            ), 0.0,
        )

        residual_action = torch.stack(
            [
                residual[:, 0] * float(self.actor.residual_workload_limit),
                residual[:, 1] * float(self.actor.residual_setpoint_limit),
                residual[:, 2] * float(self.actor.residual_pump_limit),
            ],
            dim=-1,
        )
        residual_reg = residual_action.pow(2).mean()

        actor_loss = (
            self.alpha.detach() * log_prob
            - q_pi
            + float(exposure_lambda) * exposure_pi
            + float(event_lambda) * event_pi
            + self.margin_aux_beta * margin_pi
        ).mean() + self.residual_reg_beta * residual_reg

        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        torch.nn.utils.clip_grad_norm_(
            self.actor.trainable_parameters(), 1.0,
        )
        self.actor_optimizer.step()

        # Entropy temperature
        alpha_loss = -(
            self.log_alpha * (log_prob + self.target_entropy).detach()
        ).mean()
        self.alpha_optimizer.zero_grad()
        alpha_loss.backward()
        self.alpha_optimizer.step()

        # Target networks
        for target, source in (
            (self.q1_t, self.q1), (self.q2_t, self.q2),
            (self.ce1_t, self.ce1), (self.ce2_t, self.ce2),
            (self.cv1_t, self.cv1), (self.cv2_t, self.cv2),
            (self.cm1_t, self.cm1), (self.cm2_t, self.cm2),
        ):
            soft_update(target, source, self.tau)

        return {
            "q_loss": float(q_loss.item()),
            "exposure_critic_loss": float(ce_loss.item()),
            "event_critic_loss": float(cv_loss.item()),
            "margin_critic_loss": float(cm_loss.item()),
            "actor_loss": float(actor_loss.item()),
            "alpha_loss": float(alpha_loss.item()),
            "alpha": float(self.alpha.detach().item()),
            "q_pi": float(q_pi.detach().mean().item()),
            "exposure_q": float(exposure_pi.detach().mean().item()),
            "event_q": float(event_pi.detach().mean().item()),
            "margin_q": float(margin_pi.detach().mean().item()),
            "residual_abs_mean": float(residual.detach().abs().mean().item()),
            "residual_action_l2": float(residual_reg.detach().item()),
            "true_event_cost_batch_mean": float(event_cost.detach().mean().item()),
            "margin_cost_batch_mean": float(margin_cost.detach().mean().item()),
            "cold_cost_batch_mean": float(cold_cost.detach().mean().item()),
            "mechanical_cooling_cost_batch_mean": float(
                mechanical_cooling_cost.detach().mean().item()
            ),
            "backlog_cost_batch_mean": float(backlog_cost.detach().mean().item()),
            "shaped_reward_batch_mean": float(shaped_reward.detach().mean().item()),
        }


# =============================================================================
# Target return / environment / thermal stats (verbatim from V6)
# =============================================================================

def compute_target_return(*, traj_path: str, percentile: float) -> float:
    trajectories = load_trajectories([traj_path], use_applied_actions=True)
    returns = np.asarray(
        [
            float(np.asarray(t["rewards"], dtype=np.float64).sum())
            for t in trajectories
        ],
        dtype=np.float64,
    )
    return float(np.percentile(returns, percentile))


def make_env(*, country, seed, forecast_mode, forecast_dir):
    return CoastalDCContinuousEnv(
        country=country,
        seed=seed,
        forecast_mode=forecast_mode,
        forecast_dir=forecast_dir,
    )


def episode_thermal_stats(
    temperatures,
    *,
    reference_temp_c: float,
    target_band_low_c: float = 26.6,
) -> dict[str, float]:
    x = np.asarray(temperatures, dtype=np.float64)
    x = x[np.isfinite(x)]

    excess = np.maximum(x - reference_temp_c, 0.0)
    event = (x > reference_temp_c)
    cold_excess = np.maximum(target_band_low_c - x, 0.0)
    below_band = (x < target_band_low_c)

    return {
        "p95_t_inlet": float(np.percentile(x, 95)),
        "thermal_exposure_c_h": float(excess.sum()),
        "thermal_event_hours": float(event.sum()),
        "thermal_event_rate_pct": float(100.0 * event.mean()),
        "cold_exposure_sq_c2_h": float(np.square(cold_excess).sum()),
        "below_target_band_hours": float(below_band.sum()),
    }


# =============================================================================
# Deterministic evaluation (verbatim from V6)
# =============================================================================

def evaluate_actor(
    *,
    actor: ForecastJointLoRAActor,
    target_return: float,
    return_scale: float,
    country: str,
    forecast_mode: str,
    forecast_dir: str | None,
    base_seed: int,
    episodes: int,
    thermal_reference_c: float,
    target_band_low_c: float = 26.6,
    seasonal_start_hours: list[int] | None = None,
) -> dict[str, float]:
    rows = []
    actor.eval()

    if seasonal_start_hours is not None and len(seasonal_start_hours) > 0:
        eval_schedule = [
            (int(base_seed) + i, int(sh), str(i))
            for i, sh in enumerate(seasonal_start_hours)
        ]
    else:
        eval_schedule = [
            (int(base_seed) + ep, None, str(ep))
            for ep in range(int(episodes))
        ]

    for seed, start_hour_override, _tag in eval_schedule:
        env = make_env(
            country=country, seed=seed,
            forecast_mode=forecast_mode, forecast_dir=forecast_dir,
        )
        controller = ForecastJointLoRADTController(
            actor=actor, env=env,
            target_return=target_return,
            return_scale=return_scale,
            deterministic=True,
        )

        reset_options = (
            None if start_hour_override is None
            else {"start_hour": int(start_hour_override)}
        )
        obs, info = env.reset(seed=seed, options=reset_options)
        controller.reset()

        done = False
        rewards, temps = [], []
        while not done:
            action = controller.act(obs, info)
            next_obs, reward, terminated, truncated, next_info = env.step(action)
            applied = np.asarray(
                next_info.get("applied_action", action), dtype=np.float32,
            )
            controller.update_applied_action(applied)
            controller.update_reward(reward)
            rewards.append(float(reward))
            temps.append(float(next_info.get("t_inlet", np.nan)))
            obs, info = next_obs, next_info
            done = terminated or truncated

        summary = env.episode_summary()
        thermal = episode_thermal_stats(
            temps, reference_temp_c=thermal_reference_c,
            target_band_low_c=target_band_low_c,
        )

        rows.append(
            {
                "return": float(np.sum(rewards)),
                "co2_kg": float(summary["co2_kg"]),
                "e_grid_mwh": float(summary["e_grid_mwh"]),
                "e_total_mwh": float(summary["e_total_mwh"]),
                "workload_intervention_rate_pct": (
                    100.0 * float(summary["workload_interventions"])
                    / max(len(rewards), 1)
                ),
                "mean_action_override": float(summary["mean_action_override"]),
                **thermal,
            }
        )

    out = {}
    for key in rows[0]:
        values = np.asarray([r[key] for r in rows], dtype=np.float64)
        out[key] = float(values.mean())
    return out


def thermal_selection_tuple(
    metrics: dict[str, float],
    *,
    selection_p95_limit_c: float,
    selection_exposure_limit: float,
    selection_event_hours_limit: float,
    p95_target_c: float,
    selection_exposure_target: float,
    selection_event_hours_target: float,
):
    p95 = float(metrics["p95_t_inlet"])
    exposure = float(metrics["thermal_exposure_c_h"])
    event_hours = float(metrics["thermal_event_hours"])

    feasibility_violation = (
        max(p95 - selection_p95_limit_c, 0.0) / 0.10
        + max(exposure - selection_exposure_limit, 0.0)
        / max(selection_exposure_limit, 0.25)
        + max(event_hours - selection_event_hours_limit, 0.0)
        / max(selection_event_hours_limit, 1.0)
    )

    thermal_tie_break = (
        abs(p95 - p95_target_c) / 0.10
        + 0.25 * abs(exposure - selection_exposure_target)
        / max(selection_exposure_target, 0.25)
        + 0.25 * abs(event_hours - selection_event_hours_target)
        / max(selection_event_hours_target, 1.0)
    )

    return (
        float(feasibility_violation),
        float(-metrics["return"]),
        float(thermal_tie_break),
    )


# =============================================================================
# Saving
# =============================================================================

def save_actor_checkpoint(
    *,
    path: str,
    actor: ForecastJointLoRAActor,
    base_checkpoint: str,
    target_return: float,
    return_scale: float,
    actor_config: dict,
    training_metrics: dict | None,
):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.save(
        {
            "format": "lora_thermal_dt_v6_ablation_no_v3",
            # Field name kept as "v3_checkpoint" for loader compatibility with
            # ForecastJointLoRAActor's checkpoint loader. In this ablation, it
            # holds the PRETRAINED dt_<COUNTRY>_seed*.pt path, NOT a V3 output.
            "v3_checkpoint": os.path.abspath(base_checkpoint),
            "target_return": float(target_return),
            "return_scale": float(return_scale),
            "actor_config": actor_config,
            "trainable_actor_state": actor.trainable_state_dict(),
            "metrics": training_metrics,
        },
        path,
    )


# =============================================================================
# Main
# =============================================================================

def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--country", default="JPN")

    # ABLATION: backbone defaults to PRETRAINED, not V3.
    parser.add_argument(
        "--pretrain-model",
        "--v3-model",  # deprecated alias, kept for backward compatibility
        dest="pretrain_model",
        default=None,
        help=(
            "Pretrained DT checkpoint produced by "
            "train_decision_transformer.py. Defaults to the newest "
            "scripts/eact-dt-lora/results/models/dt_<COUNTRY>_seed*.pt. "
            "posttrain_decision_transformer_v3.py is deliberately SKIPPED."
        ),
    )
    parser.add_argument(
        "--expert-trajectories",
        "--mpc-trajectories",
        dest="expert_trajectories",
        default=None,
    )

    parser.add_argument("--context-len", type=int, default=24)
    parser.add_argument("--embed-dim", type=int, default=128)
    parser.add_argument("--n-layers", type=int, default=3)
    parser.add_argument("--n-heads", type=int, default=4)
    parser.add_argument("--dropout", type=float, default=0.1)

    parser.add_argument("--lora-rank", type=int, default=4)
    parser.add_argument("--lora-alpha", type=float, default=8.0)
    parser.add_argument("--lora-dropout", type=float, default=0.0)

    parser.add_argument("--workload-residual-limit", type=float, default=0.10)
    parser.add_argument("--setpoint-residual-limit", type=float, default=0.10)
    parser.add_argument("--pump-residual-limit", type=float, default=0.10)
    parser.add_argument("--forecast-hidden-dim", type=int, default=64)
    parser.add_argument("--residual-hidden-dim", type=int, default=96)
    parser.add_argument("--initial-log-std", type=float, default=-1.0)

    parser.add_argument("--target-percentile", type=float, default=90.0)
    parser.add_argument("--return-scale", type=float, default=100.0)

    parser.add_argument("--thermal-reference", type=float, default=27.0)
    parser.add_argument("--target-band-low", type=float, default=25.5)
    parser.add_argument("--p95-target", type=float, default=26.91)
    parser.add_argument("--exposure-budget", type=float, default=0.50)
    parser.add_argument("--event-hours-budget", type=float, default=2.0)
    parser.add_argument("--event-margin-start", type=float, default=26.9)
    parser.add_argument("--event-margin-width", type=float, default=0.1)
    parser.add_argument("--selection-p95-limit", type=float, default=26.95)
    parser.add_argument("--selection-exposure-limit", type=float, default=0.50)
    parser.add_argument("--selection-event-hours-limit", type=float, default=2.0)
    parser.add_argument("--selection-exposure-target", type=float, default=0.155)
    parser.add_argument("--selection-event-hours-target", type=float, default=1.73)
    parser.add_argument("--cold-penalty-weight", type=float, default=0.25)
    parser.add_argument(
        "--mechanical-cooling-penalty-weight", type=float, default=0.20,
    )
    parser.add_argument("--backlog-penalty-weight", type=float, default=0.10)
    parser.add_argument("--residual-reg-beta", type=float, default=0.05)
    parser.add_argument("--margin-aux-beta", type=float, default=0.05)

    parser.add_argument("--train-episodes", type=int, default=300)
    parser.add_argument("--train-seed-start", type=int, default=30000)
    parser.add_argument("--learning-starts", type=int, default=1500)
    parser.add_argument("--updates-per-step", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--replay-capacity", type=int, default=50000)

    parser.add_argument("--actor-lr", type=float, default=1e-4)
    parser.add_argument("--critic-lr", type=float, default=3e-4)
    parser.add_argument("--alpha-lr", type=float, default=3e-4)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--tau", type=float, default=0.005)
    parser.add_argument("--reward-scale", type=float, default=1.0)
    parser.add_argument("--target-entropy", type=float, default=-3.0)

    parser.add_argument("--dual-lr", type=float, default=0.05)
    parser.add_argument("--dual-exposure-gap-scale", type=float, default=0.50)
    parser.add_argument("--dual-event-gap-scale", type=float, default=2.0)
    parser.add_argument("--init-exposure-lambda", type=float, default=0.01)
    parser.add_argument("--init-event-lambda", type=float, default=0.01)
    parser.add_argument("--max-dual-lambda", type=float, default=10.0)

    parser.add_argument(
        "--forecast-mode", choices=("oracle", "causal", "none"),
        default="causal",
    )
    parser.add_argument("--forecast-dir", default=None)

    parser.add_argument("--eval-every", type=int, default=5)
    parser.add_argument(
        "--seasonal-timestamps", nargs="*",
        default=list(DEFAULT_SEASONAL_TIMESTAMPS),
    )
    parser.add_argument("--eval-episodes", type=int, default=50)  # legacy
    parser.add_argument("--eval-base-seed", type=int, default=9500)

    parser.add_argument("--seed", type=int, default=0)

    # ABLATION: isolated output directory.
    parser.add_argument(
        "--output-dir", default=None,
        help=(
            "Defaults to scripts/eact-dt-lora/ablation_results/"
            "<COUNTRY>_lora_sac_direct_from_pretrain/"
        ),
    )

    args = parser.parse_args()

    if args.event_margin_width <= 0:
        raise ValueError("--event-margin-width must be > 0")
    if args.mechanical_cooling_penalty_weight < 0:
        raise ValueError("--mechanical-cooling-penalty-weight must be >= 0")
    if args.backlog_penalty_weight < 0:
        raise ValueError("--backlog-penalty-weight must be >= 0")
    if args.event_margin_start >= args.thermal_reference:
        raise ValueError(
            "--event-margin-start must be below --thermal-reference."
        )
    if args.target_band_low >= args.thermal_reference:
        raise ValueError(
            "--target-band-low must be below --thermal-reference."
        )
    if args.selection_p95_limit > args.thermal_reference:
        raise ValueError(
            "--selection-p95-limit should not exceed --thermal-reference."
        )
    if args.selection_exposure_limit < 0:
        raise ValueError("--selection-exposure-limit must be >= 0")
    if args.selection_event_hours_limit < 0:
        raise ValueError("--selection-event-hours-limit must be >= 0")
    if args.eval_every <= 0:
        raise ValueError("--eval-every must be > 0")
    if len(args.seasonal_timestamps) == 0:
        raise ValueError(
            "--seasonal-timestamps must contain at least one timestamp"
        )

    set_seed(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # ------------------------------------------------------------------
    # Resolve backbone + trajectories
    # ------------------------------------------------------------------
    base_ckpt = (
        os.path.abspath(args.pretrain_model)
        if args.pretrain_model is not None
        else _resolve_default_pretrain_model(args.country)
    )
    traj_path = (
        os.path.abspath(args.expert_trajectories)
        if args.expert_trajectories is not None
        else _resolve_default_traj_path(args.country)
    )

    if not os.path.isfile(base_ckpt):
        raise FileNotFoundError(
            "Pretrained DT checkpoint not found:\n"
            f"  {base_ckpt}\n"
            f"Expected location: {DEFAULT_MODEL_DIR}\n"
            "Run train_decision_transformer.py first (this ablation does NOT "
            "use posttrain_decision_transformer_v3.py), or pass "
            "--pretrain-model explicitly."
        )

    if not os.path.isfile(traj_path):
        raise FileNotFoundError(
            "EACT expert trajectory file not found:\n"
            f"  {traj_path}\n"
            "Run generate_eact_trajectories.py first, or pass "
            "--expert-trajectories explicitly."
        )

    # Defensive check: reject V3-style checkpoints in this ablation.
    if "_post_v3_" in os.path.basename(base_ckpt):
        print(
            f"[WARNING] --pretrain-model points to a V3 post-trained file:\n"
            f"  {base_ckpt}\n"
            "This ablation is designed to skip V3 post-training. "
            "Proceeding anyway, but the ablation is no longer clean."
        )

    forecast_dir = (
        os.path.abspath(args.forecast_dir)
        if args.forecast_dir is not None
        else DEFAULT_FORECAST_DIR
    )
    if args.forecast_mode == "causal":
        forecast_file = os.path.join(
            forecast_dir, f"causal_ridge_{args.country}.npz",
        )
        if not os.path.isfile(forecast_file):
            raise FileNotFoundError(
                f"Causal forecast model not found:\n  {forecast_file}"
            )

    target_return = compute_target_return(
        traj_path=traj_path, percentile=args.target_percentile,
    )

    # ------------------------------------------------------------------
    # Env shape + seasonal start hours
    # ------------------------------------------------------------------
    probe_env = make_env(
        country=args.country,
        seed=args.train_seed_start,
        forecast_mode=args.forecast_mode,
        forecast_dir=forecast_dir,
    )
    obs_dim = int(probe_env.observation_space.shape[0])
    episode_hours = int(probe_env.episode_hours)
    causal_min_history = (
        int(probe_env._causal_min_history)
        if args.forecast_mode == "causal" else 0
    )

    seasonal_start_hours = resolve_seasonal_start_hours(
        probe_env.data, args.seasonal_timestamps,
        episode_hours=episode_hours, min_start=causal_min_history,
    )
    seasonal_labels = [
        (SEASONAL_LABELS[i] if i < len(SEASONAL_LABELS)
         else f"seasonal_{i}")
        for i in range(len(seasonal_start_hours))
    ]

    actor_config = {
        "obs_dim": obs_dim,
        "context_len": args.context_len,
        "embed_dim": args.embed_dim,
        "n_layers": args.n_layers,
        "n_heads": args.n_heads,
        "dropout": args.dropout,
        "lora_rank": args.lora_rank,
        "lora_alpha": args.lora_alpha,
        "lora_dropout": args.lora_dropout,
        "forecast_dim": FULL_FORECAST_DIM,
        "forecast_hidden_dim": args.forecast_hidden_dim,
        "residual_hidden_dim": args.residual_hidden_dim,
        "residual_workload_limit": args.workload_residual_limit,
        "residual_setpoint_limit": args.setpoint_residual_limit,
        "residual_pump_limit": args.pump_residual_limit,
        "initial_log_std": args.initial_log_std,
    }

    # ------------------------------------------------------------------
    # Actor / replay / SAC / dual
    # ------------------------------------------------------------------
    actor = ForecastJointLoRAActor(
        v3_checkpoint=base_ckpt,  # PRETRAINED in this ablation
        device=device,
        **actor_config,
    )

    replay = ContextReplayBuffer(
        capacity=args.replay_capacity,
        context_len=args.context_len,
        obs_dim=obs_dim,
        forecast_dim=FULL_FORECAST_DIM,
    )

    sac = ConstrainedSAC(
        actor=actor,
        obs_dim=obs_dim,
        forecast_dim=FULL_FORECAST_DIM,
        device=device,
        actor_lr=args.actor_lr,
        critic_lr=args.critic_lr,
        alpha_lr=args.alpha_lr,
        gamma=args.gamma,
        tau=args.tau,
        target_entropy=args.target_entropy,
        reward_scale=args.reward_scale,
        cold_penalty_weight=args.cold_penalty_weight,
        mechanical_cooling_penalty_weight=args.mechanical_cooling_penalty_weight,
        backlog_penalty_weight=args.backlog_penalty_weight,
        residual_reg_beta=args.residual_reg_beta,
        margin_aux_beta=args.margin_aux_beta,
    )

    dual = EpisodeLagrangeController(
        exposure_budget=args.exposure_budget,
        event_hours_budget=args.event_hours_budget,
        exposure_gap_scale=args.dual_exposure_gap_scale,
        event_gap_scale=args.dual_event_gap_scale,
        dual_lr=args.dual_lr,
        init_exposure_lambda=args.init_exposure_lambda,
        init_event_lambda=args.init_event_lambda,
        max_lambda=args.max_dual_lambda,
    )

    # ------------------------------------------------------------------
    # Output paths (isolated under ablation_results/)
    # ------------------------------------------------------------------
    output_dir = (
        os.path.abspath(args.output_dir)
        if args.output_dir is not None
        else os.path.join(
            DEFAULT_OUTPUT_ROOT,
            f"{args.country}_lora_sac_direct_from_pretrain",
        )
    )
    os.makedirs(output_dir, exist_ok=True)

    best_path = os.path.join(output_dir, "best_lora_thermal_dt.pt")
    last_path = os.path.join(output_dir, "last_lora_thermal_dt.pt")
    log_csv = os.path.join(output_dir, "training_log.csv")
    validation_csv = os.path.join(output_dir, "validation_pareto.csv")
    config_path = os.path.join(output_dir, "config.json")

    # ------------------------------------------------------------------
    # Header
    # ------------------------------------------------------------------
    print("=" * 100)
    print("ABLATION: FULL-FORECAST JOINT LoRA-SAC  (DIRECT from PRETRAINED DT)")
    print("            >>>  NO posttrain_decision_transformer_v3.py  <<<")
    print("=" * 100)
    print(f"device                    : {device}")
    print(f"country                   : {args.country}")
    print(f"expert trajectory pool    : {traj_path}")
    print(f"PRETRAIN backbone         : {base_ckpt}")
    print(f"target return             : {target_return:.3f}")
    print(f"LoRA layers               : {actor.n_lora_layers}")
    print(f"LoRA rank / alpha         : {args.lora_rank} / {args.lora_alpha:g}")
    print(f"workload residual limit   : +/- {args.workload_residual_limit:g}")
    print(f"setpoint residual limit   : +/- {args.setpoint_residual_limit:g}")
    print(f"pump residual limit       : +/- {args.pump_residual_limit:g}")
    print(f"full forecast dimension   : {FULL_FORECAST_DIM}")
    print(f"cold-protection threshold : {args.target_band_low:.3f} C")
    print(f"true event threshold      : {args.thermal_reference:.3f} C")
    print(f"training exposure budget  : {args.exposure_budget:.3f} C*h/ep")
    print(f"training event budget     : {args.event_hours_budget:.3f} h/ep")
    print(
        f"checkpoint limits         : "
        f"P95<={args.selection_p95_limit:.3f} C, "
        f"Exposure<={args.selection_exposure_limit:.3f}, "
        f"EventH<={args.selection_event_hours_limit:.3f}"
    )
    print(f"cold penalty weight       : {args.cold_penalty_weight:g}")
    print(f"mechanical cooling beta   : {args.mechanical_cooling_penalty_weight:g}")
    print(f"backlog penalty beta      : {args.backlog_penalty_weight:g}")
    print(f"residual reg beta         : {args.residual_reg_beta:g}")
    print(f"margin auxiliary beta     : {args.margin_aux_beta:g}")
    print(f"train episodes (total)    : {args.train_episodes}")
    print(f"eval every                : {args.eval_every}")
    print("seasonal evaluation starts:")
    for label, ts, sh in zip(
        seasonal_labels, args.seasonal_timestamps, seasonal_start_hours,
    ):
        print(f"  - {label:8s} | {ts} | start_hour={sh}")
    print(f"output                    : {output_dir}")
    print("=" * 100)

    # ------------------------------------------------------------------
    # Initial deterministic eval (should match pretrained DT closely)
    # ------------------------------------------------------------------
    initial_eval = evaluate_actor(
        actor=actor,
        target_return=target_return,
        return_scale=args.return_scale,
        country=args.country,
        forecast_mode=args.forecast_mode,
        forecast_dir=forecast_dir,
        base_seed=args.eval_base_seed,
        episodes=args.eval_episodes,
        thermal_reference_c=args.thermal_reference,
        target_band_low_c=args.target_band_low,
        seasonal_start_hours=seasonal_start_hours,
    )

    print("\nInitial deterministic evaluation (PRETRAINED-DT baseline)")
    for key, value in initial_eval.items():
        print(f"  {key:36s}: {value:.5f}")

    best_score = thermal_selection_tuple(
        initial_eval,
        selection_p95_limit_c=args.selection_p95_limit,
        selection_exposure_limit=args.selection_exposure_limit,
        selection_event_hours_limit=args.selection_event_hours_limit,
        p95_target_c=args.p95_target,
        selection_exposure_target=args.selection_exposure_target,
        selection_event_hours_target=args.selection_event_hours_target,
    )
    best_metrics = dict(initial_eval)

    save_actor_checkpoint(
        path=best_path,
        actor=actor,
        base_checkpoint=base_ckpt,
        target_return=target_return,
        return_scale=args.return_scale,
        actor_config=actor_config,
        training_metrics=best_metrics,
    )

    log_rows = []
    validation_rows = []
    total_env_steps = 0
    update_count = 0
    latest_update_metrics = {}

    # ------------------------------------------------------------------
    # Training loop
    # ------------------------------------------------------------------
    for episode in range(1, args.train_episodes + 1):
        reset_options = None
        seed = args.train_seed_start + episode - 1

        env = make_env(
            country=args.country,
            seed=seed,
            forecast_mode=args.forecast_mode,
            forecast_dir=forecast_dir,
        )
        controller = ForecastJointLoRADTController(
            actor=actor,
            env=env,
            target_return=target_return,
            return_scale=args.return_scale,
            deterministic=False,
        )

        obs, info = env.reset(seed=seed, options=reset_options)
        controller.reset()

        done = False
        episode_return = 0.0
        episode_exposure = 0.0
        episode_event_hours = 0.0
        episode_margin_risk = 0.0
        episode_cold_cost = 0.0
        episode_mechanical_cooling_cost = 0.0
        episode_backlog_cost = 0.0
        episode_backlog_sum_mwh = 0.0
        episode_max_backlog_mwh = 0.0
        episode_temps = []
        latest_update_metrics = {}

        while not done:
            action = controller.act(obs, info)
            current_context = controller.last_context
            current_forecast = np.asarray(
                controller.last_forecast, dtype=np.float32,
            ).copy()

            next_obs, reward, terminated, truncated, next_info = env.step(action)
            done = bool(terminated or truncated)

            applied = np.asarray(
                next_info.get("applied_action", action), dtype=np.float32,
            )
            controller.update_applied_action(applied)
            controller.update_reward(reward)

            try:
                t_inlet = float(next_info.get("t_inlet", np.nan))
            except (TypeError, ValueError):
                t_inlet = np.nan

            e_cooling_step_mwh = float(next_info.get("e_cooling_mwh", 0.0))
            mechanical_cooling_cost = (
                e_cooling_step_mwh
                / max(float(env.reward_fn.n.total_scale), 1e-9)
            )

            backlog_after_mwh = float(
                next_info.get("backlog_mwh", env.workload.backlog_mwh)
            )
            backlog_scale_mwh = max(
                float(env.wl_cfg.it_capacity_mw)
                * float(env.wl_cfg.max_delay_hours),
                1e-9,
            )
            backlog_cost = float(
                np.clip(backlog_after_mwh / backlog_scale_mwh, 0.0, 1.0)
            )

            if np.isfinite(t_inlet):
                exposure_cost = max(t_inlet - args.thermal_reference, 0.0)
                true_event = float(t_inlet > args.thermal_reference)
                margin_cost = float(
                    np.clip(
                        (t_inlet - args.event_margin_start)
                        / args.event_margin_width,
                        0.0, 1.0,
                    )
                )
                cold_cost = max(args.target_band_low - t_inlet, 0.0) ** 2
            else:
                exposure_cost = 0.0
                true_event = 0.0
                margin_cost = 0.0
                cold_cost = 0.0

            next_context = controller.preview_context(next_obs)
            if done:
                next_forecast = np.zeros(FULL_FORECAST_DIM, dtype=np.float32)
            else:
                next_forecast = build_full_forecast_features(env)

            replay.add(
                context=current_context,
                forecast_features=current_forecast,
                env_action=action,
                reward=reward,
                exposure_cost=exposure_cost,
                event_cost=true_event,
                margin_cost=margin_cost,
                cold_cost=cold_cost,
                mechanical_cooling_cost=mechanical_cooling_cost,
                backlog_cost=backlog_cost,
                next_context=next_context,
                next_forecast_features=next_forecast,
                done=done,
            )

            episode_return += float(reward)
            episode_exposure += float(exposure_cost)
            episode_event_hours += float(true_event)
            episode_margin_risk += float(margin_cost)
            episode_cold_cost += float(cold_cost)
            episode_mechanical_cooling_cost += float(mechanical_cooling_cost)
            episode_backlog_cost += float(backlog_cost)
            episode_backlog_sum_mwh += float(backlog_after_mwh)
            episode_max_backlog_mwh = max(
                episode_max_backlog_mwh, float(backlog_after_mwh),
            )
            episode_temps.append(t_inlet)
            total_env_steps += 1

            if len(replay) >= args.learning_starts:
                for _ in range(args.updates_per_step):
                    batch = replay.sample(args.batch_size, device=device)
                    latest_update_metrics = sac.update(
                        batch,
                        exposure_lambda=dual.exposure_lambda,
                        event_lambda=dual.event_lambda,
                    )
                    update_count += 1

            obs = next_obs
            info = next_info

        dual_state = dual.update(
            episode_exposure=episode_exposure,
            episode_event_hours=episode_event_hours,
            episode_length=len(episode_temps),
        )
        thermal = episode_thermal_stats(
            episode_temps,
            reference_temp_c=args.thermal_reference,
            target_band_low_c=args.target_band_low,
        )
        summary = env.episode_summary()

        row = {
            "episode": episode,
            "seed": seed,
            "env_steps": total_env_steps,
            "updates": update_count,
            "return": episode_return,
            "co2_kg": float(summary["co2_kg"]),
            "e_grid_mwh": float(summary["e_grid_mwh"]),
            "e_total_mwh": float(summary["e_total_mwh"]),
            "e_cooling_mwh": float(summary["e_cooling_mwh"]),
            "e_pump_mwh": float(summary["e_pump_mwh"]),
            "workload_intervention_rate_pct": (
                100.0 * float(summary["workload_interventions"])
                / max(len(episode_temps), 1)
            ),
            **thermal,
            "exposure_lambda": dual_state.exposure_lambda,
            "event_lambda": dual_state.event_lambda,
            "exposure_gap_normalized": dual_state.exposure_gap_normalized,
            "event_gap_normalized": dual_state.event_gap_normalized,
            "episode_margin_risk": episode_margin_risk,
            "episode_cold_cost": episode_cold_cost,
            "episode_mechanical_cooling_cost": episode_mechanical_cooling_cost,
            "episode_backlog_cost": episode_backlog_cost,
            "avg_backlog_mwh": (
                episode_backlog_sum_mwh / max(len(episode_temps), 1)
            ),
            "max_backlog_mwh": episode_max_backlog_mwh,
            **latest_update_metrics,
        }
        log_rows.append(row)

        if episode == 1 or episode % 5 == 0:
            print(
                f"[ABLATION] ep {episode:04d}/{args.train_episodes} | "
                f"R={episode_return:.2f} | "
                f"P95={thermal['p95_t_inlet']:.3f} | "
                f"Exp={episode_exposure:.3f} | "
                f"EvtH={episode_event_hours:.1f} | "
                f"Cold={episode_cold_cost:.3f} | "
                f"Mech={episode_mechanical_cooling_cost:.2f} | "
                f"B={episode_backlog_sum_mwh / max(len(episode_temps), 1):.2f}MWh | "
                f"lam_exp={dual.exposure_lambda:.3f} | "
                f"lam_evt={dual.event_lambda:.3f} | "
                f"buf={len(replay)}"
            )

        # --------------------------------------------------------------
        # Fixed matched deterministic validation
        # --------------------------------------------------------------
        if episode % args.eval_every == 0 or episode == args.train_episodes:
            metrics = evaluate_actor(
                actor=actor,
                target_return=target_return,
                return_scale=args.return_scale,
                country=args.country,
                forecast_mode=args.forecast_mode,
                forecast_dir=forecast_dir,
                base_seed=args.eval_base_seed,
                episodes=args.eval_episodes,
                thermal_reference_c=args.thermal_reference,
                target_band_low_c=args.target_band_low,
                seasonal_start_hours=seasonal_start_hours,
            )

            score = thermal_selection_tuple(
                metrics,
                selection_p95_limit_c=args.selection_p95_limit,
                selection_exposure_limit=args.selection_exposure_limit,
                selection_event_hours_limit=args.selection_event_hours_limit,
                p95_target_c=args.p95_target,
                selection_exposure_target=args.selection_exposure_target,
                selection_event_hours_target=args.selection_event_hours_target,
            )

            print("\n" + "-" * 100)
            print(
                f"[ABLATION] seasonal eval @ episode {episode} "
                f"({len(seasonal_start_hours)} fixed start-hours)"
            )
            print(
                f"Return={metrics['return']:.3f} | "
                f"CO2={metrics['co2_kg']:.1f} | "
                f"Grid={metrics['e_grid_mwh']:.3f} | "
                f"P95={metrics['p95_t_inlet']:.3f} | "
                f"Exp={metrics['thermal_exposure_c_h']:.3f} | "
                f"EvtH={metrics['thermal_event_hours']:.3f} | "
                f"ColdSq={metrics['cold_exposure_sq_c2_h']:.3f} | "
                f"BelowBandH={metrics['below_target_band_hours']:.2f} | "
                f"WorkInt={metrics['workload_intervention_rate_pct']:.3f}%"
            )
            print(f"selection score={score}")
            print("-" * 100 + "\n")

            periodic_path = os.path.join(
                output_dir, f"checkpoint_ep{episode:03d}.pt",
            )
            save_actor_checkpoint(
                path=periodic_path,
                actor=actor,
                base_checkpoint=base_ckpt,
                target_return=target_return,
                return_scale=args.return_scale,
                actor_config=actor_config,
                training_metrics=metrics,
            )

            validation_rows.append(
                {
                    "episode": int(episode),
                    "checkpoint": periodic_path,
                    "return": float(metrics["return"]),
                    "co2_kg": float(metrics["co2_kg"]),
                    "e_grid_mwh": float(metrics["e_grid_mwh"]),
                    "e_total_mwh": float(metrics["e_total_mwh"]),
                    "p95_t_inlet": float(metrics["p95_t_inlet"]),
                    "thermal_exposure_c_h": float(metrics["thermal_exposure_c_h"]),
                    "thermal_event_hours": float(metrics["thermal_event_hours"]),
                    "thermal_event_rate_pct": float(metrics["thermal_event_rate_pct"]),
                    "workload_intervention_rate_pct": float(
                        metrics["workload_intervention_rate_pct"]
                    ),
                    "mean_action_override": float(metrics["mean_action_override"]),
                    "selection_feasibility_violation": float(score[0]),
                    "selection_negative_return": float(score[1]),
                    "selection_thermal_tie_break": float(score[2]),
                    "exposure_lambda": float(dual.exposure_lambda),
                    "event_lambda": float(dual.event_lambda),
                }
            )

            with open(validation_csv, "w", newline="", encoding="utf-8") as f:
                writer = csv.DictWriter(
                    f, fieldnames=list(validation_rows[0].keys()),
                )
                writer.writeheader()
                writer.writerows(validation_rows)

            if score < best_score:
                best_score = score
                best_metrics = dict(metrics)
                save_actor_checkpoint(
                    path=best_path,
                    actor=actor,
                    base_checkpoint=base_ckpt,
                    target_return=target_return,
                    return_scale=args.return_scale,
                    actor_config=actor_config,
                    training_metrics=best_metrics,
                )
                print(f"[best][ABLATION] updated -> {best_path}")

            save_actor_checkpoint(
                path=last_path,
                actor=actor,
                base_checkpoint=base_ckpt,
                target_return=target_return,
                return_scale=args.return_scale,
                actor_config=actor_config,
                training_metrics=metrics,
            )

            if log_rows:
                fields, seen = [], set()
                for log_row in log_rows:
                    for key in log_row:
                        if key not in seen:
                            seen.add(key)
                            fields.append(key)
                with open(log_csv, "w", newline="", encoding="utf-8") as f:
                    writer = csv.DictWriter(f, fieldnames=fields)
                    writer.writeheader()
                    writer.writerows(log_rows)

    # ------------------------------------------------------------------
    # Final save
    # ------------------------------------------------------------------
    save_actor_checkpoint(
        path=last_path,
        actor=actor,
        base_checkpoint=base_ckpt,
        target_return=target_return,
        return_scale=args.return_scale,
        actor_config=actor_config,
        training_metrics=(log_rows[-1] if log_rows else None),
    )

    with open(config_path, "w", encoding="utf-8") as f:
        json.dump(
            {
                "experiment": "ablation_no_v3_posttrain",
                "note": (
                    "LoRA-SAC backbone is the pretrained dt_<COUNTRY>_seed*.pt; "
                    "posttrain_decision_transformer_v3.py is NOT used."
                ),
                "args": vars(args),
                "device": device,
                "target_return": target_return,
                "pretrain_model": base_ckpt,
                "expert_trajectories": traj_path,
                "actor_config": actor_config,
                "best_metrics": best_metrics,
                "best_score": list(best_score),
                "seasonal_start_hours": [int(v) for v in seasonal_start_hours],
                "seasonal_labels": list(seasonal_labels),
                "eval_every": int(args.eval_every),
                "evaluation_schedule": "seasonal_fixed_start_hours",
                "evaluation_episode_count": int(len(seasonal_start_hours)),
                "output_dir": output_dir,
            },
            f, ensure_ascii=False, indent=2,
        )

    print("\n[ABLATION] Training complete")
    print(f"best checkpoint : {best_path}")
    print(f"last checkpoint : {last_path}")
    print(f"training log    : {log_csv}")
    print(f"validation log  : {validation_csv}")
    print(f"config          : {config_path}")
    print("\nBest deterministic metrics:")
    for key, value in best_metrics.items():
        print(f"  {key:36s}: {value:.6f}")


if __name__ == "__main__":
    main()