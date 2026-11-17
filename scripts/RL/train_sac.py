"""train_sac.py — SAC baseline for CoastalDCContinuousEnv (fair architecture comparison).

Location
--------
scripts/RL/train_sac.py

Purpose: fair algorithmic-architecture comparison to
``scripts/eact-dt-lora/posttrain_dt_lora_sac_thermal_v6_backlog_season.py``.

The reward and constraint structure is IDENTICAL to V6.  The only differences
are the function approximators:

    V6 LoRA-SAC (ours)   : DT-context + LoRA actor, twin Q, 3 cost critics
    This SAC baseline    : MLP Gaussian actor,     twin Q, 3 cost critics

Both scripts share:
    [Layer 1] Q-level shaped reward
        shaped_reward = reward_env
                        - beta_cold   * cold_cost
                        - beta_mech   * (E_cooling / reward.total_scale)
                        - beta_backlog * backlog_cost
    [Layer 2] Actor-level constraint shaping
        actor_loss += lambda_exposure * Q_exposure(s, pi(s))
                    + lambda_event    * Q_event(s, pi(s))
                    + beta_margin     * Q_margin(s, pi(s))
        with (lambda_exposure, lambda_event) updated by symmetric clipped
        episode-level dual ascent.
    - Same env, same 29-D obs, same 3-D Box action space
    - Same fixed seasonal deterministic evaluation
    - Same checkpoint-selection tuple:
          (thermal feasibility violation, -return, thermal tie-break)
    - Same output layout

Use ``--no-lagrange`` to disable Layer 2 (pure-vanilla ablation).

Usage (from scripts/RL/)
------------------------
    python train_sac.py --country JPN --train-episodes 300                (Layer 1 + Layer 2)
    python train_sac.py --country JPN --train-episodes 300 --no-lagrange  (Layer 1 only)
"""

from __future__ import annotations

import argparse
import copy
import csv
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
# Path resolution: this file lives in scripts/RL/, so ROOT is two levels up.
# ---------------------------------------------------------------------------
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, "..", ".."))
sys.path.insert(0, ROOT)

from coastaldc_env.continuous_env import CoastalDCContinuousEnv  # noqa: E402


DEFAULT_SEASONAL_TIMESTAMPS = (
    "2025-01-15 00:00:00",
    "2025-04-15 00:00:00",
    "2025-07-15 00:00:00",
    "2025-10-15 00:00:00",
)
SEASONAL_LABELS = ("winter", "spring", "summer", "autumn")
DEFAULT_OUTPUT_ROOT = os.path.join(SCRIPT_DIR, "results")


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
# Seasonal start-hour resolution (identical to V6)
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
                f"Seasonal timestamp is absent from environment data: {value}")
        idx = int(lookup[ts])
        if idx < min_start or idx > max_start:
            raise ValueError(
                f"Seasonal timestamp {value} maps to start_hour={idx}, "
                f"outside valid range [{min_start}, {max_start}].")
        out.append(idx)
    return out


def make_env(*, country, seed, forecast_mode, forecast_dir):
    return CoastalDCContinuousEnv(
        country=country,
        seed=seed,
        forecast_mode=forecast_mode,
        forecast_dir=forecast_dir,
    )


# =============================================================================
# Replay buffer (with cost signals)
# =============================================================================
class ReplayBuffer:
    """obs / action / shaped_reward / cost signals / next_obs / done."""

    def __init__(self, *, capacity: int, obs_dim: int, act_dim: int):
        self.capacity = int(capacity)
        self.obs = np.zeros((capacity, obs_dim), dtype=np.float32)
        self.next_obs = np.zeros((capacity, obs_dim), dtype=np.float32)
        self.actions = np.zeros((capacity, act_dim), dtype=np.float32)
        self.rewards = np.zeros((capacity, 1), dtype=np.float32)
        self.exposure_costs = np.zeros((capacity, 1), dtype=np.float32)
        self.event_costs = np.zeros((capacity, 1), dtype=np.float32)
        self.margin_costs = np.zeros((capacity, 1), dtype=np.float32)
        self.dones = np.zeros((capacity, 1), dtype=np.float32)
        self.ptr = 0
        self.size = 0

    def __len__(self):
        return self.size

    def add(self, obs, action, reward, next_obs, done,
            exposure_cost=0.0, event_cost=0.0, margin_cost=0.0):
        i = self.ptr
        self.obs[i] = np.asarray(obs, dtype=np.float32).reshape(-1)
        self.actions[i] = np.asarray(action, dtype=np.float32).reshape(-1)
        self.rewards[i, 0] = float(reward)
        self.next_obs[i] = np.asarray(next_obs, dtype=np.float32).reshape(-1)
        self.dones[i, 0] = float(done)
        self.exposure_costs[i, 0] = float(exposure_cost)
        self.event_costs[i, 0] = float(event_cost)
        self.margin_costs[i, 0] = float(margin_cost)
        self.ptr = (self.ptr + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)

    def sample(self, batch_size: int, *, device: str) -> dict[str, torch.Tensor]:
        idx = np.random.randint(0, self.size, size=batch_size)

        def f32(a):
            return torch.as_tensor(a[idx], dtype=torch.float32, device=device)

        return {
            "obs": f32(self.obs),
            "actions": f32(self.actions),
            "rewards": f32(self.rewards),
            "next_obs": f32(self.next_obs),
            "dones": f32(self.dones),
            "exposure_costs": f32(self.exposure_costs),
            "event_costs": f32(self.event_costs),
            "margin_costs": f32(self.margin_costs),
        }


# =============================================================================
# Networks
# =============================================================================
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


class QNetwork(nn.Module):
    def __init__(self, obs_dim, act_dim, hidden_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim + act_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, obs, action):
        return self.net(torch.cat([obs, action], dim=-1))


def soft_update(target, source, tau):
    with torch.no_grad():
        for tp, sp in zip(target.parameters(), source.parameters()):
            tp.data.mul_(1.0 - tau).add_(tau * sp.data)


# =============================================================================
# Episode-level Lagrange dual ascent (identical to V6)
# =============================================================================
@dataclass
class DualState:
    exposure_lambda: float
    event_lambda: float
    exposure_gap_normalized: float
    event_gap_normalized: float


class EpisodeLagrangeController:
    """Symmetric clipped episode-level dual ascent (same as V6)."""

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
        self.exposure_budget = float(exposure_budget)
        self.event_hours_budget = float(event_hours_budget)
        self.exposure_gap_scale = float(exposure_gap_scale)
        self.event_gap_scale = float(event_gap_scale)
        self.dual_lr = float(dual_lr)
        self.max_lambda = float(max_lambda)
        self.exposure_lambda = float(init_exposure_lambda)
        self.event_lambda = float(init_event_lambda)

    def update(self, *, episode_exposure: float, episode_event_hours: float) -> DualState:
        exp_gap = float(np.clip(
            (float(episode_exposure) - self.exposure_budget)
            / self.exposure_gap_scale, -1.0, 1.0))
        event_gap = float(np.clip(
            (float(episode_event_hours) - self.event_hours_budget)
            / self.event_gap_scale, -1.0, 1.0))
        self.exposure_lambda = float(np.clip(
            self.exposure_lambda + self.dual_lr * exp_gap, 0.0, self.max_lambda))
        self.event_lambda = float(np.clip(
            self.event_lambda + self.dual_lr * event_gap, 0.0, self.max_lambda))
        return DualState(
            exposure_lambda=self.exposure_lambda,
            event_lambda=self.event_lambda,
            exposure_gap_normalized=exp_gap,
            event_gap_normalized=event_gap,
        )


# =============================================================================
# SAC agent (with optional V6-style Lagrange heads)
# =============================================================================
class SACAgent:
    """SAC with Layer-1 shaped reward (always) + Layer-2 Lagrange (optional).

    When ``use_lagrange=True`` (default), three additional cost critics are
    trained with standard Bellman backups, and the actor loss includes the
    Lagrange terms with externally-supplied multipliers.  This mirrors the
    constraint structure of the LoRA-SAC V6 script exactly.
    """

    def __init__(
        self,
        *,
        obs_dim: int,
        act_dim: int,
        action_low,
        action_high,
        device: str,
        hidden_dim: int,
        actor_lr: float,
        critic_lr: float,
        alpha_lr: float,
        gamma: float,
        tau: float,
        target_entropy: float,
        reward_scale: float,
        use_lagrange: bool,
        margin_aux_beta: float,
    ):
        self.device = device
        self.gamma = float(gamma)
        self.tau = float(tau)
        self.reward_scale = float(reward_scale)
        self.target_entropy = float(target_entropy)
        self.use_lagrange = bool(use_lagrange)
        self.margin_aux_beta = float(margin_aux_beta)

        self.actor = GaussianActor(
            obs_dim, act_dim, hidden_dim, action_low, action_high).to(device)
        self.q1 = QNetwork(obs_dim, act_dim, hidden_dim).to(device)
        self.q2 = QNetwork(obs_dim, act_dim, hidden_dim).to(device)
        self.q1_t = copy.deepcopy(self.q1)
        self.q2_t = copy.deepcopy(self.q2)
        for m in (self.q1_t, self.q2_t):
            for p in m.parameters():
                p.requires_grad = False

        self.actor_opt = torch.optim.Adam(self.actor.parameters(), lr=actor_lr)
        self.q_opt = torch.optim.Adam(
            list(self.q1.parameters()) + list(self.q2.parameters()),
            lr=critic_lr,
        )

        if self.use_lagrange:
            self.ce1 = QNetwork(obs_dim, act_dim, hidden_dim).to(device)
            self.ce2 = QNetwork(obs_dim, act_dim, hidden_dim).to(device)
            self.cv1 = QNetwork(obs_dim, act_dim, hidden_dim).to(device)
            self.cv2 = QNetwork(obs_dim, act_dim, hidden_dim).to(device)
            self.cm1 = QNetwork(obs_dim, act_dim, hidden_dim).to(device)
            self.cm2 = QNetwork(obs_dim, act_dim, hidden_dim).to(device)
            self.ce1_t = copy.deepcopy(self.ce1); self.ce2_t = copy.deepcopy(self.ce2)
            self.cv1_t = copy.deepcopy(self.cv1); self.cv2_t = copy.deepcopy(self.cv2)
            self.cm1_t = copy.deepcopy(self.cm1); self.cm2_t = copy.deepcopy(self.cm2)
            for m in (self.ce1_t, self.ce2_t, self.cv1_t, self.cv2_t,
                      self.cm1_t, self.cm2_t):
                for p in m.parameters():
                    p.requires_grad = False
            self.ce_opt = torch.optim.Adam(
                list(self.ce1.parameters()) + list(self.ce2.parameters()),
                lr=critic_lr)
            self.cv_opt = torch.optim.Adam(
                list(self.cv1.parameters()) + list(self.cv2.parameters()),
                lr=critic_lr)
            self.cm_opt = torch.optim.Adam(
                list(self.cm1.parameters()) + list(self.cm2.parameters()),
                lr=critic_lr)

        self.log_alpha = torch.tensor(
            np.log(0.1), dtype=torch.float32, device=device, requires_grad=True)
        self.alpha_opt = torch.optim.Adam([self.log_alpha], lr=alpha_lr)

    @property
    def alpha(self):
        return self.log_alpha.exp()

    def act(self, obs, *, deterministic: bool = False) -> np.ndarray:
        with torch.no_grad():
            obs_t = torch.as_tensor(
                obs, dtype=torch.float32, device=self.device).unsqueeze(0)
            action, _ = self.actor.sample(obs_t, deterministic=deterministic)
            return action.squeeze(0).cpu().numpy()

    # ---------------------------------------------------------------- update
    def update(self, batch, *, exposure_lambda: float = 0.0,
               event_lambda: float = 0.0) -> dict[str, float]:
        obs = batch["obs"]
        actions = batch["actions"]
        rewards = batch["rewards"] * self.reward_scale
        next_obs = batch["next_obs"]
        dones = batch["dones"]

        # ---- Reward Q critics --------------------------------------------
        self.q_opt.zero_grad(set_to_none=True)
        with torch.no_grad():
            next_actions, next_log_probs = self.actor.sample(next_obs)
            next_q = torch.minimum(
                self.q1_t(next_obs, next_actions),
                self.q2_t(next_obs, next_actions))
            target_q = rewards + self.gamma * (1.0 - dones) * (
                next_q - self.alpha.detach() * next_log_probs)

        q1 = self.q1(obs, actions)
        q2 = self.q2(obs, actions)
        q_loss = F.mse_loss(q1, target_q) + F.mse_loss(q2, target_q)
        q_loss.backward()
        self.q_opt.step()

        # ---- Cost critics (Layer 2) -------------------------------------
        ce_loss = cv_loss = cm_loss = torch.zeros((), device=self.device)
        if self.use_lagrange:
            with torch.no_grad():
                # Re-use next_actions from Q update (same on-policy target).
                next_ce = torch.clamp_min(torch.maximum(
                    self.ce1_t(next_obs, next_actions),
                    self.ce2_t(next_obs, next_actions)), 0.0)
                exp_target = batch["exposure_costs"] + self.gamma * (1.0 - dones) * next_ce

                next_cv = torch.clamp_min(torch.maximum(
                    self.cv1_t(next_obs, next_actions),
                    self.cv2_t(next_obs, next_actions)), 0.0)
                evt_target = batch["event_costs"] + self.gamma * (1.0 - dones) * next_cv

                next_cm = torch.clamp_min(torch.maximum(
                    self.cm1_t(next_obs, next_actions),
                    self.cm2_t(next_obs, next_actions)), 0.0)
                mgn_target = batch["margin_costs"] + self.gamma * (1.0 - dones) * next_cm

            self.ce_opt.zero_grad(set_to_none=True)
            ce_loss = (F.mse_loss(self.ce1(obs, actions), exp_target)
                       + F.mse_loss(self.ce2(obs, actions), exp_target))
            ce_loss.backward(); self.ce_opt.step()

            self.cv_opt.zero_grad(set_to_none=True)
            cv_loss = (F.mse_loss(self.cv1(obs, actions), evt_target)
                       + F.mse_loss(self.cv2(obs, actions), evt_target))
            cv_loss.backward(); self.cv_opt.step()

            self.cm_opt.zero_grad(set_to_none=True)
            cm_loss = (F.mse_loss(self.cm1(obs, actions), mgn_target)
                       + F.mse_loss(self.cm2(obs, actions), mgn_target))
            cm_loss.backward(); self.cm_opt.step()

        # ---- Actor -------------------------------------------------------
        self.actor_opt.zero_grad(set_to_none=True)
        self.q_opt.zero_grad(set_to_none=True)

        new_actions, log_probs = self.actor.sample(obs)
        q_pi = torch.minimum(
            self.q1(obs, new_actions), self.q2(obs, new_actions))
        actor_loss = (self.alpha.detach() * log_probs - q_pi).mean()

        exp_pi = evt_pi = mgn_pi = torch.zeros((), device=self.device)
        if self.use_lagrange:
            exp_pi = torch.clamp_min(torch.maximum(
                self.ce1(obs, new_actions), self.ce2(obs, new_actions)), 0.0)
            evt_pi = torch.clamp_min(torch.maximum(
                self.cv1(obs, new_actions), self.cv2(obs, new_actions)), 0.0)
            mgn_pi = torch.clamp_min(torch.maximum(
                self.cm1(obs, new_actions), self.cm2(obs, new_actions)), 0.0)
            actor_loss = actor_loss + (
                float(exposure_lambda) * exp_pi
                + float(event_lambda) * evt_pi
                + self.margin_aux_beta * mgn_pi
            ).mean()

        actor_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.actor.parameters(), 1.0)
        self.actor_opt.step()

        # ---- Temperature --------------------------------------------------
        self.alpha_opt.zero_grad(set_to_none=True)
        alpha_loss = -(self.log_alpha * (
            log_probs + self.target_entropy).detach()).mean()
        alpha_loss.backward()
        self.alpha_opt.step()

        # ---- Soft target updates ------------------------------------------
        soft_update(self.q1_t, self.q1, self.tau)
        soft_update(self.q2_t, self.q2, self.tau)
        if self.use_lagrange:
            soft_update(self.ce1_t, self.ce1, self.tau)
            soft_update(self.ce2_t, self.ce2, self.tau)
            soft_update(self.cv1_t, self.cv1, self.tau)
            soft_update(self.cv2_t, self.cv2, self.tau)
            soft_update(self.cm1_t, self.cm1, self.tau)
            soft_update(self.cm2_t, self.cm2, self.tau)

        out = {
            "q_loss": float(q_loss.item()),
            "actor_loss": float(actor_loss.item()),
            "alpha_loss": float(alpha_loss.item()),
            "alpha": float(self.alpha.detach().item()),
            "q_pi": float(q_pi.detach().mean().item()),
            "log_prob_mean": float(log_probs.detach().mean().item()),
        }
        if self.use_lagrange:
            out.update({
                "exposure_critic_loss": float(ce_loss.item()),
                "event_critic_loss": float(cv_loss.item()),
                "margin_critic_loss": float(cm_loss.item()),
                "exposure_q": float(exp_pi.detach().mean().item()),
                "event_q": float(evt_pi.detach().mean().item()),
                "margin_q": float(mgn_pi.detach().mean().item()),
            })
        return out


# =============================================================================
# Thermal statistics (identical to V6)
# =============================================================================
def episode_thermal_stats(temperatures, *, reference_temp_c: float,
                          target_band_low_c: float = 26.6) -> dict[str, float]:
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
# Deterministic seasonal evaluation (identical protocol to V6)
# =============================================================================
def evaluate_policy(*, act_fn, country, forecast_mode, forecast_dir, base_seed,
                    thermal_reference_c, target_band_low_c,
                    seasonal_start_hours) -> dict[str, float]:
    rows = []
    for i, start_hour in enumerate(seasonal_start_hours):
        seed = int(base_seed) + i
        env = make_env(country=country, seed=seed,
                       forecast_mode=forecast_mode, forecast_dir=forecast_dir)
        obs, info = env.reset(seed=seed, options={"start_hour": int(start_hour)})
        done = False
        rewards, temps = [], []
        while not done:
            action = act_fn(obs)
            next_obs, reward, terminated, truncated, next_info = env.step(action)
            rewards.append(float(reward))
            temps.append(float(next_info.get("t_inlet", np.nan)))
            obs = next_obs
            done = terminated or truncated

        summary = env.episode_summary()
        thermal = episode_thermal_stats(
            temps, reference_temp_c=thermal_reference_c,
            target_band_low_c=target_band_low_c)
        rows.append({
            "return": float(np.sum(rewards)),
            "co2_kg": float(summary["co2_kg"]),
            "e_grid_mwh": float(summary["e_grid_mwh"]),
            "e_total_mwh": float(summary["e_total_mwh"]),
            "workload_intervention_rate_pct":
                100.0 * float(summary["workload_interventions"])
                / max(len(rewards), 1),
            "mean_action_override": float(summary["mean_action_override"]),
            **thermal,
        })

    out = {}
    for key in rows[0]:
        values = np.asarray([r[key] for r in rows], dtype=np.float64)
        out[key] = float(values.mean())
    return out


def thermal_selection_tuple(metrics, *, selection_p95_limit_c,
                            selection_exposure_limit, selection_event_hours_limit,
                            p95_target_c, selection_exposure_target,
                            selection_event_hours_target):
    """Same checkpoint-selection priority as V6."""
    p95 = float(metrics["p95_t_inlet"])
    exposure = float(metrics["thermal_exposure_c_h"])
    event_hours = float(metrics["thermal_event_hours"])
    feasibility_violation = (
        max(p95 - selection_p95_limit_c, 0.0) / 0.10
        + max(exposure - selection_exposure_limit, 0.0)
        / max(selection_exposure_limit, 0.25)
        + max(event_hours - selection_event_hours_limit, 0.0)
        / max(selection_event_hours_limit, 1.0))
    thermal_tie_break = (
        abs(p95 - p95_target_c) / 0.10
        + 0.25 * abs(exposure - selection_exposure_target)
        / max(selection_exposure_target, 0.25)
        + 0.25 * abs(event_hours - selection_event_hours_target)
        / max(selection_event_hours_target, 1.0))
    return (float(feasibility_violation),
            float(-metrics["return"]),
            float(thermal_tie_break))


# =============================================================================
# Checkpoint I/O
# =============================================================================
def save_checkpoint(*, path, agent, actor_config, training_metrics, use_lagrange):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    payload = {
        "format": "sac_lagrange_v1",
        "use_lagrange": bool(use_lagrange),
        "actor_config": actor_config,
        "actor_state": agent.actor.state_dict(),
        "q1_state": agent.q1.state_dict(),
        "q2_state": agent.q2.state_dict(),
        "metrics": training_metrics,
    }
    if use_lagrange:
        payload.update({
            "ce1_state": agent.ce1.state_dict(),
            "ce2_state": agent.ce2.state_dict(),
            "cv1_state": agent.cv1.state_dict(),
            "cv2_state": agent.cv2.state_dict(),
            "cm1_state": agent.cm1.state_dict(),
            "cm2_state": agent.cm2.state_dict(),
        })
    torch.save(payload, path)


# =============================================================================
# Main
# =============================================================================
def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--country", default="JPN")

    # ---- SAC hyperparameters -----------------------------------------------
    parser.add_argument("--hidden-dim", type=int, default=256)
    parser.add_argument("--actor-lr", type=float, default=3e-4)
    parser.add_argument("--critic-lr", type=float, default=3e-4)
    parser.add_argument("--alpha-lr", type=float, default=3e-4)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--tau", type=float, default=0.005)
    parser.add_argument("--reward-scale", type=float, default=1.0)
    parser.add_argument("--target-entropy", type=float, default=-3.0)

    # ---- Training loop ------------------------------------------------------
    parser.add_argument("--train-episodes", type=int, default=300)
    parser.add_argument("--train-seed-start", type=int, default=30000)
    parser.add_argument("--learning-starts", type=int, default=1500)
    parser.add_argument("--updates-per-step", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--replay-capacity", type=int, default=50000)

    # ---- Layer-1 shaped reward (matches V6) --------------------------------
    parser.add_argument("--thermal-reference", type=float, default=27.0)
    parser.add_argument("--target-band-low", type=float, default=25.5)
    parser.add_argument("--cold-penalty-weight", type=float, default=0.25)
    parser.add_argument("--mechanical-cooling-penalty-weight",
                        type=float, default=0.20)
    parser.add_argument("--backlog-penalty-weight", type=float, default=0.10)

    # ---- Layer-2 Lagrange constraints (matches V6) -------------------------
    parser.add_argument("--use-lagrange", dest="use_lagrange",
                        action="store_true", default=True,
                        help="Enable V6-style Lagrange constraint heads (default).")
    parser.add_argument("--no-lagrange", dest="use_lagrange",
                        action="store_false",
                        help="Disable Layer 2 (pure Layer-1 ablation).")
    parser.add_argument("--exposure-budget", type=float, default=0.50)
    parser.add_argument("--event-hours-budget", type=float, default=2.0)
    parser.add_argument("--event-margin-start", type=float, default=26.9)
    parser.add_argument("--event-margin-width", type=float, default=0.1)
    parser.add_argument("--margin-aux-beta", type=float, default=0.05)
    parser.add_argument("--dual-lr", type=float, default=0.05)
    parser.add_argument("--dual-exposure-gap-scale", type=float, default=0.50)
    parser.add_argument("--dual-event-gap-scale", type=float, default=2.0)
    parser.add_argument("--init-exposure-lambda", type=float, default=0.01)
    parser.add_argument("--init-event-lambda", type=float, default=0.01)
    parser.add_argument("--max-dual-lambda", type=float, default=10.0)

    # ---- Checkpoint selection ---------------------------------------------
    parser.add_argument("--p95-target", type=float, default=26.91)
    parser.add_argument("--selection-p95-limit", type=float, default=26.95)
    parser.add_argument("--selection-exposure-limit", type=float, default=0.50)
    parser.add_argument("--selection-event-hours-limit", type=float, default=2.0)
    parser.add_argument("--selection-exposure-target", type=float, default=0.155)
    parser.add_argument("--selection-event-hours-target", type=float, default=1.73)

    # ---- Environment --------------------------------------------------------
    parser.add_argument("--forecast-mode",
                        choices=("oracle", "causal", "none"), default="causal")
    parser.add_argument("--forecast-dir", default=None)

    # ---- Eval schedule ------------------------------------------------------
    parser.add_argument("--eval-every", type=int, default=5)
    parser.add_argument("--seasonal-timestamps", nargs="*",
                        default=list(DEFAULT_SEASONAL_TIMESTAMPS))
    parser.add_argument("--eval-episodes", type=int, default=50,
                        help="Legacy; ignored (seasonal schedule is fixed).")
    parser.add_argument("--eval-base-seed", type=int, default=9500)

    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--output-dir", default=None)

    args = parser.parse_args()

    if args.eval_every <= 0:
        raise ValueError("--eval-every must be > 0")
    if len(args.seasonal_timestamps) == 0:
        raise ValueError("--seasonal-timestamps must contain at least one entry")
    if args.target_band_low >= args.thermal_reference:
        raise ValueError("--target-band-low must be below --thermal-reference")
    if args.selection_p95_limit > args.thermal_reference:
        raise ValueError(
            "--selection-p95-limit should not exceed --thermal-reference.")
    if args.event_margin_width <= 0:
        raise ValueError("--event-margin-width must be > 0")
    if args.event_margin_start >= args.thermal_reference:
        raise ValueError("--event-margin-start must be below --thermal-reference")

    set_seed(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # ---- Environment probe --------------------------------------------------
    forecast_dir = os.path.abspath(args.forecast_dir) if args.forecast_dir else None
    probe_env = make_env(
        country=args.country, seed=args.train_seed_start,
        forecast_mode=args.forecast_mode, forecast_dir=forecast_dir)
    obs_dim = int(probe_env.observation_space.shape[0])
    act_dim = int(probe_env.action_space.shape[0])
    episode_hours = int(probe_env.episode_hours)
    action_low = np.asarray(probe_env.action_space.low, dtype=np.float32)
    action_high = np.asarray(probe_env.action_space.high, dtype=np.float32)

    causal_min_history = (
        int(probe_env._causal_min_history)
        if args.forecast_mode == "causal" else 0)
    seasonal_start_hours = resolve_seasonal_start_hours(
        probe_env.data, args.seasonal_timestamps,
        episode_hours=episode_hours, min_start=causal_min_history)
    seasonal_labels = [
        SEASONAL_LABELS[i] if i < len(SEASONAL_LABELS) else f"seasonal_{i}"
        for i in range(len(seasonal_start_hours))]

    # ---- Agent / replay / duals --------------------------------------------
    agent = SACAgent(
        obs_dim=obs_dim, act_dim=act_dim,
        action_low=action_low, action_high=action_high,
        device=device, hidden_dim=args.hidden_dim,
        actor_lr=args.actor_lr, critic_lr=args.critic_lr, alpha_lr=args.alpha_lr,
        gamma=args.gamma, tau=args.tau,
        target_entropy=args.target_entropy, reward_scale=args.reward_scale,
        use_lagrange=args.use_lagrange, margin_aux_beta=args.margin_aux_beta)

    replay = ReplayBuffer(
        capacity=args.replay_capacity, obs_dim=obs_dim, act_dim=act_dim)

    dual = EpisodeLagrangeController(
        exposure_budget=args.exposure_budget,
        event_hours_budget=args.event_hours_budget,
        exposure_gap_scale=args.dual_exposure_gap_scale,
        event_gap_scale=args.dual_event_gap_scale,
        dual_lr=args.dual_lr,
        init_exposure_lambda=args.init_exposure_lambda,
        init_event_lambda=args.init_event_lambda,
        max_lambda=args.max_dual_lambda)

    actor_config = {
        "obs_dim": obs_dim, "act_dim": act_dim, "hidden_dim": args.hidden_dim,
        "action_low": action_low.tolist(), "action_high": action_high.tolist(),
        "use_lagrange": bool(args.use_lagrange),
    }

    # ---- Output dir ---------------------------------------------------------
    tag = "v3_sac_lagrange_season" if args.use_lagrange else "v3_sac_vanilla_season"
    output_dir = (
        os.path.abspath(args.output_dir)
        if args.output_dir is not None
        else os.path.join(DEFAULT_OUTPUT_ROOT, f"{args.country}_{tag}"))
    os.makedirs(output_dir, exist_ok=True)

    best_path = os.path.join(output_dir, "best_sac.pt")
    last_path = os.path.join(output_dir, "last_sac.pt")
    log_csv = os.path.join(output_dir, "training_log.csv")
    validation_csv = os.path.join(output_dir, "validation_pareto.csv")
    config_path = os.path.join(output_dir, "config.json")

    print("=" * 100)
    print("SAC BASELINE  |  "
          + ("Layer-1 shaped reward + Layer-2 Lagrange"
             if args.use_lagrange else "Layer-1 shaped reward ONLY"))
    print("=" * 100)
    print(f"device                    : {device}")
    print(f"country                   : {args.country}")
    print(f"obs_dim / act_dim         : {obs_dim} / {act_dim}")
    print(f"hidden_dim                : {args.hidden_dim}")
    print(f"actor_lr / critic_lr      : {args.actor_lr} / {args.critic_lr}")
    print(f"alpha_lr                  : {args.alpha_lr}")
    print(f"gamma / tau               : {args.gamma} / {args.tau}")
    print(f"target entropy            : {args.target_entropy}")
    print(f"batch size                : {args.batch_size}")
    print(f"replay capacity           : {args.replay_capacity}")
    print(f"learning starts           : {args.learning_starts}")
    print(f"updates per step          : {args.updates_per_step}")
    print("--- Layer-1 shaped reward (matches V6) ---")
    print(f"cold penalty weight       : {args.cold_penalty_weight:g}")
    print(f"mechanical cooling beta   : {args.mechanical_cooling_penalty_weight:g}")
    print(f"backlog penalty beta      : {args.backlog_penalty_weight:g}")
    print("--- Layer-2 actor-level Lagrange ---")
    print(f"use_lagrange              : {args.use_lagrange}")
    if args.use_lagrange:
        print(f"exposure budget           : {args.exposure_budget:.3f} C*h/episode")
        print(f"event hours budget        : {args.event_hours_budget:.3f} h/episode")
        print(f"margin aux beta           : {args.margin_aux_beta:g}")
        print(f"dual lr / gap scales      : "
              f"{args.dual_lr:g} / ({args.dual_exposure_gap_scale:g}, "
              f"{args.dual_event_gap_scale:g})")
        print(f"max dual lambda           : {args.max_dual_lambda:g}")
    print("--- evaluation / selection ---")
    print(f"thermal reference         : {args.thermal_reference:.3f} C")
    print(f"target band low           : {args.target_band_low:.3f} C")
    print(f"P95 selection target      : {args.p95_target:.3f} C")
    print(f"checkpoint limits         : "
          f"P95<={args.selection_p95_limit:.3f} C, "
          f"Exposure<={args.selection_exposure_limit:.3f}, "
          f"EventH<={args.selection_event_hours_limit:.3f}")
    print(f"train episodes            : {args.train_episodes}")
    print(f"eval every                : {args.eval_every}")
    print("seasonal evaluation starts:")
    for label, ts, sh in zip(seasonal_labels, args.seasonal_timestamps,
                             seasonal_start_hours):
        print(f"  - {label:8s} | {ts} | start_hour={sh}")
    print(f"output                    : {output_dir}")
    print("=" * 100)

    # ---- Initial deterministic evaluation (fresh policy; reference only) ---
    initial_eval = evaluate_policy(
        act_fn=lambda obs: agent.act(obs, deterministic=True),
        country=args.country, forecast_mode=args.forecast_mode,
        forecast_dir=forecast_dir, base_seed=args.eval_base_seed,
        thermal_reference_c=args.thermal_reference,
        target_band_low_c=args.target_band_low,
        seasonal_start_hours=seasonal_start_hours)
    print("\nPre-training deterministic evaluation "
          "(randomly-initialised SAC policy -- reference only)")
    for key, value in initial_eval.items():
        print(f"  {key:36s}: {value:.5f}")

    # NOTE: best_score is NOT initialised from the random policy.  We only
    # accept a checkpoint as "best" once we have evaluated a trained policy.
    best_score = None
    best_metrics: dict = {}

    log_rows: list[dict] = []
    validation_rows: list[dict] = []
    total_env_steps = 0
    update_count = 0

    # ---- Training loop ------------------------------------------------------
    for episode in range(1, args.train_episodes + 1):
        seed = args.train_seed_start + episode - 1
        env = make_env(country=args.country, seed=seed,
                       forecast_mode=args.forecast_mode, forecast_dir=forecast_dir)
        obs, info = env.reset(seed=seed)
        done = False

        episode_return = 0.0
        episode_exposure = 0.0
        episode_event_hours = 0.0
        episode_cold_cost = 0.0
        episode_mech_cool = 0.0
        episode_backlog_cost = 0.0
        episode_backlog_sum = 0.0
        episode_max_backlog = 0.0
        episode_temps: list[float] = []
        latest_update_metrics: dict[str, float] = {}

        while not done:
            if len(replay) < args.learning_starts:
                action = env.action_space.sample()
            else:
                action = agent.act(obs, deterministic=False)
            action = np.clip(
                action, env.action_space.low, env.action_space.high
            ).astype(np.float32)

            next_obs, reward, terminated, truncated, next_info = env.step(action)
            done = bool(terminated or truncated)

            try:
                t_inlet = float(next_info.get("t_inlet", np.nan))
            except (TypeError, ValueError):
                t_inlet = np.nan

            # --- Layer-1 shaping -----------------------------------------
            e_cooling_step_mwh = float(next_info.get("e_cooling_mwh", 0.0))
            mechanical_cooling_cost = (
                e_cooling_step_mwh
                / max(float(env.reward_fn.n.total_scale), 1e-9))
            backlog_after_mwh = float(
                next_info.get("backlog_mwh", env.workload.backlog_mwh))
            backlog_scale_mwh = max(
                float(env.wl_cfg.it_capacity_mw)
                * float(env.wl_cfg.max_delay_hours), 1e-9)
            backlog_cost = float(np.clip(
                backlog_after_mwh / backlog_scale_mwh, 0.0, 1.0))

            # --- Cost signals (Layer-2) ----------------------------------
            if np.isfinite(t_inlet):
                exposure_cost = max(t_inlet - args.thermal_reference, 0.0)
                true_event = float(t_inlet > args.thermal_reference)
                margin_cost = float(np.clip(
                    (t_inlet - args.event_margin_start) / args.event_margin_width,
                    0.0, 1.0))
                cold_cost = max(args.target_band_low - t_inlet, 0.0) ** 2
            else:
                exposure_cost = 0.0
                true_event = 0.0
                margin_cost = 0.0
                cold_cost = 0.0

            shaped_reward = (
                float(reward)
                - args.cold_penalty_weight * cold_cost
                - args.mechanical_cooling_penalty_weight * mechanical_cooling_cost
                - args.backlog_penalty_weight * backlog_cost)

            replay.add(obs, action, shaped_reward, next_obs, done,
                       exposure_cost=exposure_cost,
                       event_cost=true_event,
                       margin_cost=margin_cost)

            episode_return += float(reward)
            episode_exposure += float(exposure_cost)
            episode_event_hours += float(true_event)
            episode_cold_cost += float(cold_cost)
            episode_mech_cool += float(mechanical_cooling_cost)
            episode_backlog_cost += float(backlog_cost)
            episode_backlog_sum += float(backlog_after_mwh)
            episode_max_backlog = max(episode_max_backlog,
                                      float(backlog_after_mwh))
            episode_temps.append(t_inlet)

            total_env_steps += 1

            if len(replay) >= args.learning_starts:
                for _ in range(args.updates_per_step):
                    batch = replay.sample(args.batch_size, device=device)
                    latest_update_metrics = agent.update(
                        batch,
                        exposure_lambda=dual.exposure_lambda,
                        event_lambda=dual.event_lambda)
                    update_count += 1

            obs = next_obs
            info = next_info

        # ---- Episode-end dual ascent -------------------------------------
        dual_state = dual.update(
            episode_exposure=episode_exposure,
            episode_event_hours=episode_event_hours)

        thermal = episode_thermal_stats(
            episode_temps, reference_temp_c=args.thermal_reference,
            target_band_low_c=args.target_band_low)
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
            "workload_intervention_rate_pct":
                100.0 * float(summary["workload_interventions"])
                / max(len(episode_temps), 1),
            **thermal,
            "episode_exposure": episode_exposure,
            "episode_event_hours": episode_event_hours,
            "episode_cold_cost": episode_cold_cost,
            "episode_mechanical_cooling_cost": episode_mech_cool,
            "episode_backlog_cost": episode_backlog_cost,
            "avg_backlog_mwh": episode_backlog_sum / max(len(episode_temps), 1),
            "max_backlog_mwh": episode_max_backlog,
            "exposure_lambda": dual_state.exposure_lambda,
            "event_lambda": dual_state.event_lambda,
            "exposure_gap_normalized": dual_state.exposure_gap_normalized,
            "event_gap_normalized": dual_state.event_gap_normalized,
            **latest_update_metrics,
        }
        log_rows.append(row)

        if episode == 1 or episode % 5 == 0:
            print(
                f"episode {episode:04d}/{args.train_episodes} | "
                f"R={episode_return:.2f} | "
                f"P95={thermal['p95_t_inlet']:.3f} | "
                f"Exposure={episode_exposure:.3f} | "
                f"EventH={episode_event_hours:.1f} | "
                f"Cold={episode_cold_cost:.3f} | "
                f"MechCool={episode_mech_cool:.2f} | "
                f"Backlog={episode_backlog_sum / max(len(episode_temps), 1):.2f}MWh | "
                f"lam_exp={dual_state.exposure_lambda:.3f} | "
                f"lam_evt={dual_state.event_lambda:.3f} | "
                f"alpha={latest_update_metrics.get('alpha', float('nan')):.4f} | "
                f"buffer={len(replay)}")

        # ---- Seasonal deterministic evaluation --------------------------
        if episode % args.eval_every == 0 or episode == args.train_episodes:
            metrics = evaluate_policy(
                act_fn=lambda obs: agent.act(obs, deterministic=True),
                country=args.country, forecast_mode=args.forecast_mode,
                forecast_dir=forecast_dir, base_seed=args.eval_base_seed,
                thermal_reference_c=args.thermal_reference,
                target_band_low_c=args.target_band_low,
                seasonal_start_hours=seasonal_start_hours)

            score = thermal_selection_tuple(
                metrics,
                selection_p95_limit_c=args.selection_p95_limit,
                selection_exposure_limit=args.selection_exposure_limit,
                selection_event_hours_limit=args.selection_event_hours_limit,
                p95_target_c=args.p95_target,
                selection_exposure_target=args.selection_exposure_target,
                selection_event_hours_target=args.selection_event_hours_target)

            print("\n" + "-" * 100)
            print(f"deterministic seasonal eval @ episode {episode} "
                  f"({len(seasonal_start_hours)} fixed seasonal start-hours)")
            print(
                f"Return={metrics['return']:.3f} | "
                f"CO2={metrics['co2_kg']:.1f} | "
                f"Grid={metrics['e_grid_mwh']:.3f} | "
                f"P95={metrics['p95_t_inlet']:.3f} | "
                f"Exposure={metrics['thermal_exposure_c_h']:.3f} | "
                f"EventH={metrics['thermal_event_hours']:.3f} | "
                f"ColdSq={metrics['cold_exposure_sq_c2_h']:.3f} | "
                f"BelowBandH={metrics['below_target_band_hours']:.2f} | "
                f"WorkInt={metrics['workload_intervention_rate_pct']:.3f}%")
            print(f"selection score={score}")
            print("-" * 100 + "\n")

            periodic_path = os.path.join(output_dir,
                                         f"checkpoint_ep{episode:03d}.pt")
            save_checkpoint(path=periodic_path, agent=agent,
                            actor_config=actor_config, training_metrics=metrics,
                            use_lagrange=args.use_lagrange)

            validation_rows.append({
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
                    metrics["workload_intervention_rate_pct"]),
                "mean_action_override": float(metrics["mean_action_override"]),
                "selection_feasibility_violation": float(score[0]),
                "selection_negative_return": float(score[1]),
                "selection_thermal_tie_break": float(score[2]),
                "alpha": float(agent.alpha.detach().item()),
                "exposure_lambda": float(dual.exposure_lambda),
                "event_lambda": float(dual.event_lambda),
            })

            with open(validation_csv, "w", newline="", encoding="utf-8") as f:
                writer = csv.DictWriter(
                    f, fieldnames=list(validation_rows[0].keys()))
                writer.writeheader()
                writer.writerows(validation_rows)

            # Best-checkpoint update; first trained eval always wins.
            if best_score is None or score < best_score:
                best_score = score
                best_metrics = dict(metrics)
                save_checkpoint(path=best_path, agent=agent,
                                actor_config=actor_config,
                                training_metrics=best_metrics,
                                use_lagrange=args.use_lagrange)
                print(f"[best] updated -> {best_path}")

            save_checkpoint(path=last_path, agent=agent,
                            actor_config=actor_config,
                            training_metrics=metrics,
                            use_lagrange=args.use_lagrange)

            # Flush training log.
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

    # ---- Final artefacts ---------------------------------------------------
    save_checkpoint(path=last_path, agent=agent,
                    actor_config=actor_config,
                    training_metrics=(log_rows[-1] if log_rows else None),
                    use_lagrange=args.use_lagrange)

    with open(config_path, "w", encoding="utf-8") as f:
        json.dump({
            "args": vars(args),
            "device": device,
            "actor_config": actor_config,
            "best_metrics": best_metrics,
            "best_score": list(best_score) if best_score is not None else None,
            "seasonal_start_hours": [int(v) for v in seasonal_start_hours],
            "seasonal_labels": list(seasonal_labels),
            "eval_every": int(args.eval_every),
            "evaluation_schedule": "seasonal_fixed_start_hours",
            "evaluation_episode_count": int(len(seasonal_start_hours)),
            "output_dir": output_dir,
            "algorithm": "sac_lagrange" if args.use_lagrange else "vanilla_sac",
        }, f, ensure_ascii=False, indent=2)

    print("\nTraining complete")
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