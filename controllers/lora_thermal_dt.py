"""LoRA residual thermal adaptation for a V3 Decision Transformer.

This module is designed to sit AFTER the existing V3 dual-pool post-training.

Policy structure
----------------
1. Frozen V3 Decision Transformer provides the base action:
       a_base = [a_work, a_setpoint, a_pump]

2. A second copy of the same V3 DT receives low-rank LoRA adapters in the
   Transformer FFN linear layers. All original DT parameters remain frozen.

3. A small stochastic residual head reads the adapted final state-token
   representation and produces ONLY two residuals:
       delta_setpoint, delta_pump

4. Final requested action:
       a_work     = a_base_work                  # strictly frozen
       a_setpoint = clip(a_base_setpoint + delta_setpoint, -1, 1)
       a_pump     = clip(a_base_pump + delta_pump, 0, 1)

Thus RL cannot directly destroy the workload behavior learned by V3.

Checkpoint format
-----------------
Only trainable LoRA/residual parameters are saved. The V3 checkpoint is still
required when loading the thermal adapter.
"""

from __future__ import annotations

import copy
from dataclasses import dataclass

import numpy as np
import torch
import torch.nn as nn

from controllers.decision_transformer_continuous import (
    ContinuousDecisionTransformer,
)


# =============================================================================
# LoRA
# =============================================================================

class LoRALinear(nn.Module):
    """Frozen Linear layer plus trainable low-rank update."""

    def __init__(
        self,
        base: nn.Linear,
        *,
        rank: int = 4,
        alpha: float = 8.0,
        dropout: float = 0.0,
    ):
        super().__init__()

        if rank <= 0:
            raise ValueError("LoRA rank must be > 0")

        self.base = base
        self.rank = int(rank)
        self.alpha = float(alpha)
        self.scale = self.alpha / self.rank
        self.dropout = nn.Dropout(float(dropout))

        for p in self.base.parameters():
            p.requires_grad = False

        self.lora_A = nn.Linear(
            base.in_features,
            rank,
            bias=False,
        )
        self.lora_B = nn.Linear(
            rank,
            base.out_features,
            bias=False,
        )

        # Standard LoRA initialization: zero initial update.
        nn.init.kaiming_uniform_(
            self.lora_A.weight,
            a=np.sqrt(5),
        )
        nn.init.zeros_(
            self.lora_B.weight
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return (
            self.base(x)
            + self.scale
            * self.lora_B(
                self.lora_A(
                    self.dropout(x)
                )
            )
        )


def inject_ffn_lora(
    model: ContinuousDecisionTransformer,
    *,
    rank: int,
    alpha: float,
    dropout: float,
) -> int:
    """Insert LoRA into the two FFN Linear layers of every Transformer block."""
    count = 0

    for block in model.blocks:
        if not isinstance(block.mlp[0], nn.Linear):
            raise TypeError(
                "Expected block.mlp[0] to be nn.Linear before LoRA injection"
            )
        if not isinstance(block.mlp[2], nn.Linear):
            raise TypeError(
                "Expected block.mlp[2] to be nn.Linear before LoRA injection"
            )

        block.mlp[0] = LoRALinear(
            block.mlp[0],
            rank=rank,
            alpha=alpha,
            dropout=dropout,
        )
        block.mlp[2] = LoRALinear(
            block.mlp[2],
            rank=rank,
            alpha=alpha,
            dropout=dropout,
        )
        count += 2

    return count


def freeze_module(module: nn.Module) -> None:
    for p in module.parameters():
        p.requires_grad = False


# =============================================================================
# DT hidden representation
# =============================================================================

def forward_state_hidden(
    model: ContinuousDecisionTransformer,
    states: torch.Tensor,
    actions: torch.Tensor,
    rtgs: torch.Tensor,
    timesteps: torch.Tensor,
) -> torch.Tensor:
    """Return DT hidden representations at state-token locations."""
    B, T, _ = states.shape

    time_embedding = model.embed_time(
        timesteps
    )

    rtg_tokens = (
        model.embed_rtg(rtgs)
        + time_embedding
    )
    state_tokens = (
        model.embed_state(states)
        + time_embedding
    )
    action_tokens = (
        model.embed_action(actions)
        + time_embedding
    )

    x = torch.stack(
        [
            rtg_tokens,
            state_tokens,
            action_tokens,
        ],
        dim=2,
    ).reshape(
        B,
        3 * T,
        -1,
    )

    x = model.ln_in(x)

    for block in model.blocks:
        x = block(x)

    x = model.ln_out(x)

    return x[:, 1::3]


# =============================================================================
# Context container
# =============================================================================

@dataclass
class DTContext:
    states: np.ndarray
    actions: np.ndarray
    rtgs: np.ndarray
    timesteps: np.ndarray
    length: int


def pad_context(
    *,
    states: list[np.ndarray],
    actions: list[np.ndarray],
    rtgs: list[float],
    timesteps: list[int],
    context_len: int,
    obs_dim: int,
) -> DTContext:
    """Left-pad history to fixed K for replay storage."""
    K = int(context_len)

    s = np.asarray(
        states[-K:],
        dtype=np.float32,
    )
    a = np.asarray(
        actions[-K:],
        dtype=np.float32,
    )
    r = np.asarray(
        rtgs[-K:],
        dtype=np.float32,
    )[:, None]
    t = np.asarray(
        timesteps[-K:],
        dtype=np.int64,
    )

    length = len(s)
    pad = K - length

    if pad > 0:
        s = np.concatenate(
            [
                np.zeros(
                    (pad, obs_dim),
                    dtype=np.float32,
                ),
                s,
            ],
            axis=0,
        )
        a = np.concatenate(
            [
                np.zeros(
                    (pad, 3),
                    dtype=np.float32,
                ),
                a,
            ],
            axis=0,
        )
        r = np.concatenate(
            [
                np.zeros(
                    (pad, 1),
                    dtype=np.float32,
                ),
                r,
            ],
            axis=0,
        )
        t = np.concatenate(
            [
                np.zeros(
                    pad,
                    dtype=np.int64,
                ),
                t,
            ],
            axis=0,
        )

    return DTContext(
        states=s,
        actions=a,
        rtgs=r,
        timesteps=t,
        length=int(length),
    )


# =============================================================================
# LoRA residual actor
# =============================================================================

class LoRAThermalResidualActor(nn.Module):
    """Frozen V3 DT + LoRA-adapted hidden state + stochastic cooling residual."""

    def __init__(
        self,
        *,
        obs_dim: int,
        v3_checkpoint: str,
        context_len: int = 24,
        embed_dim: int = 128,
        n_layers: int = 3,
        n_heads: int = 4,
        dropout: float = 0.1,
        lora_rank: int = 4,
        lora_alpha: float = 8.0,
        lora_dropout: float = 0.0,
        residual_setpoint_limit: float = 0.5,
        residual_pump_limit: float = 0.5,
        residual_hidden_dim: int = 64,
        initial_log_std: float = -1.0,
        device: str = "cpu",
    ):
        super().__init__()

        self.obs_dim = int(obs_dim)
        self.context_len = int(context_len)
        self.device_name = str(device)

        self.residual_setpoint_limit = float(
            residual_setpoint_limit
        )
        self.residual_pump_limit = float(
            residual_pump_limit
        )

        model_kwargs = dict(
            obs_dim=obs_dim,
            context_len=context_len,
            embed_dim=embed_dim,
            n_layers=n_layers,
            n_heads=n_heads,
            dropout=dropout,
        )

        # Frozen exact V3 policy used for the workload/base action.
        self.base_model = ContinuousDecisionTransformer(
            **model_kwargs
        )
        state = torch.load(
            v3_checkpoint,
            map_location="cpu",
        )
        self.base_model.load_state_dict(
            state
        )
        freeze_module(
            self.base_model
        )
        self.base_model.eval()

        # Adaptable copy: same V3 weights + LoRA in FFNs.
        self.adapted_model = copy.deepcopy(
            self.base_model
        )
        freeze_module(
            self.adapted_model
        )

        self.n_lora_layers = inject_ffn_lora(
            self.adapted_model,
            rank=lora_rank,
            alpha=lora_alpha,
            dropout=lora_dropout,
        )
        self.adapted_model.eval()

        # Stochastic residual head for [setpoint, pump] only.
        self.residual_backbone = nn.Sequential(
            nn.Linear(
                embed_dim,
                residual_hidden_dim,
            ),
            nn.GELU(),
            nn.Linear(
                residual_hidden_dim,
                residual_hidden_dim,
            ),
            nn.GELU(),
        )

        self.mu_head = nn.Linear(
            residual_hidden_dim,
            2,
        )
        self.log_std_head = nn.Linear(
            residual_hidden_dim,
            2,
        )

        # Start deterministic policy exactly at V3: residual mean = 0.
        nn.init.zeros_(
            self.mu_head.weight
        )
        nn.init.zeros_(
            self.mu_head.bias
        )

        nn.init.zeros_(
            self.log_std_head.weight
        )
        nn.init.constant_(
            self.log_std_head.bias,
            float(initial_log_std),
        )

        self.log_std_min = -5.0
        self.log_std_max = 1.0

        self.to(
            device
        )

    def train(self, mode: bool = True):
        # Keep frozen/adapted DT dropout disabled for stable online RL.
        super().train(mode)
        self.base_model.eval()
        self.adapted_model.eval()
        return self

    def trainable_parameters(self):
        return [
            p
            for p in self.parameters()
            if p.requires_grad
        ]

    def trainable_state_dict(self) -> dict[str, torch.Tensor]:
        names = {
            name
            for name, p in self.named_parameters()
            if p.requires_grad
        }

        full = self.state_dict()

        return {
            key: value.detach().cpu()
            for key, value in full.items()
            if key in names
        }

    def load_trainable_state_dict(
        self,
        state: dict[str, torch.Tensor],
    ) -> None:
        own = self.state_dict()

        unknown = [
            key
            for key in state
            if key not in own
        ]
        if unknown:
            raise KeyError(
                f"Unknown thermal-adapter keys: {unknown[:10]}"
            )

        own.update(
            state
        )
        self.load_state_dict(
            own,
            strict=True,
        )

    def _model_outputs_grouped(
        self,
        *,
        states: torch.Tensor,
        actions: torch.Tensor,
        rtgs: torch.Tensor,
        timesteps: torch.Tensor,
        lengths: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Exact variable-length DT evaluation from fixed-size replay contexts."""
        B = states.shape[0]
        device = states.device

        base_actions = torch.empty(
            (B, 3),
            dtype=states.dtype,
            device=device,
        )
        adapted_hidden = torch.empty(
            (
                B,
                self.adapted_model.action_head.in_features,
            ),
            dtype=states.dtype,
            device=device,
        )

        unique_lengths = torch.unique(
            lengths.detach()
        ).tolist()

        for length_value in unique_lengths:
            L = int(
                length_value
            )
            idx = torch.nonzero(
                lengths == L,
                as_tuple=False,
            ).squeeze(-1)

            s = states[
                idx,
                -L:,
            ]
            a = actions[
                idx,
                -L:,
            ]
            r = rtgs[
                idx,
                -L:,
            ]
            t = timesteps[
                idx,
                -L:,
            ]

            with torch.no_grad():
                base_pred = self.base_model(
                    s,
                    a,
                    r,
                    t,
                )[:, -1]

            hidden = forward_state_hidden(
                self.adapted_model,
                s,
                a,
                r,
                t,
            )[:, -1]

            base_actions[
                idx
            ] = base_pred

            adapted_hidden[
                idx
            ] = hidden

        return (
            base_actions,
            adapted_hidden,
        )

    def sample(
        self,
        *,
        states: torch.Tensor,
        actions: torch.Tensor,
        rtgs: torch.Tensor,
        timesteps: torch.Tensor,
        lengths: torch.Tensor,
        deterministic: bool = False,
    ) -> tuple[
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
    ]:
        """Return final action, log_prob, base V3 action and normalized residual."""
        base_action, hidden = self._model_outputs_grouped(
            states=states,
            actions=actions,
            rtgs=rtgs,
            timesteps=timesteps,
            lengths=lengths,
        )

        z = self.residual_backbone(
            hidden
        )

        mu = self.mu_head(
            z
        )

        log_std = torch.clamp(
            self.log_std_head(z),
            self.log_std_min,
            self.log_std_max,
        )

        std = torch.exp(
            log_std
        )

        if deterministic:
            pre_tanh = mu
        else:
            dist = torch.distributions.Normal(
                mu,
                std,
            )
            pre_tanh = dist.rsample()

        normalized_residual = torch.tanh(
            pre_tanh
        )

        if deterministic:
            log_prob = torch.zeros(
                (
                    states.shape[0],
                    1,
                ),
                dtype=states.dtype,
                device=states.device,
            )
        else:
            dist = torch.distributions.Normal(
                mu,
                std,
            )

            log_prob = (
                dist.log_prob(
                    pre_tanh
                )
                - torch.log(
                    1.0
                    - normalized_residual.pow(2)
                    + 1e-6
                )
            ).sum(
                dim=-1,
                keepdim=True,
            )

        delta_setpoint = (
            normalized_residual[
                :,
                0
            ]
            * self.residual_setpoint_limit
        )

        delta_pump = (
            normalized_residual[
                :,
                1
            ]
            * self.residual_pump_limit
        )

        workload = base_action[
            :,
            0
        ]

        setpoint = torch.clamp(
            base_action[
                :,
                1
            ]
            + delta_setpoint,
            -1.0,
            1.0,
        )

        pump = torch.clamp(
            base_action[
                :,
                2
            ]
            + delta_pump,
            0.0,
            1.0,
        )

        final_action = torch.stack(
            [
                workload,
                setpoint,
                pump,
            ],
            dim=-1,
        )

        return (
            final_action,
            log_prob,
            base_action,
            normalized_residual,
        )


# =============================================================================
# Rollout controller
# =============================================================================

class LoRAThermalDTController:
    name = "lora_thermal_dt"

    def __init__(
        self,
        *,
        actor: LoRAThermalResidualActor,
        target_return: float,
        return_scale: float = 100.0,
        deterministic: bool = True,
    ):
        if return_scale <= 0:
            raise ValueError(
                "return_scale must be > 0"
            )

        self.actor = actor
        self.context_len = actor.context_len
        self.obs_dim = actor.obs_dim

        self.target_return = float(
            target_return
        )
        self.return_scale = float(
            return_scale
        )
        self.deterministic = bool(
            deterministic
        )

        self.device = next(
            actor.parameters()
        ).device

        self.reset()

    def reset(self):
        self._states = []
        self._actions = []
        self._rtgs = []
        self._ts = []

        self._rtg = (
            self.target_return
            / self.return_scale
        )
        self._t = 0
        self.last_context = None

    def _context_from_lists(
        self,
        states,
        actions,
        rtgs,
        timesteps,
    ) -> DTContext:
        return pad_context(
            states=states,
            actions=actions,
            rtgs=rtgs,
            timesteps=timesteps,
            context_len=self.context_len,
            obs_dim=self.obs_dim,
        )

    def preview_context(
        self,
        obs: np.ndarray,
    ) -> DTContext:
        states = list(
            self._states
        )
        actions = list(
            self._actions
        )
        rtgs = list(
            self._rtgs
        )
        ts = list(
            self._ts
        )

        states.append(
            np.asarray(
                obs,
                dtype=np.float32,
            )
        )
        actions.append(
            np.zeros(
                3,
                dtype=np.float32,
            )
        )
        rtgs.append(
            float(
                self._rtg
            )
        )
        ts.append(
            int(
                self._t
            )
        )

        return self._context_from_lists(
            states,
            actions,
            rtgs,
            ts,
        )

    @staticmethod
    def _context_to_torch(
        context: DTContext,
        device,
    ):
        return dict(
            states=torch.as_tensor(
                context.states,
                dtype=torch.float32,
                device=device,
            )[None],
            actions=torch.as_tensor(
                context.actions,
                dtype=torch.float32,
                device=device,
            )[None],
            rtgs=torch.as_tensor(
                context.rtgs,
                dtype=torch.float32,
                device=device,
            )[None],
            timesteps=torch.as_tensor(
                context.timesteps,
                dtype=torch.int64,
                device=device,
            )[None],
            lengths=torch.as_tensor(
                [context.length],
                dtype=torch.int64,
                device=device,
            ),
        )

    def act(
        self,
        obs: np.ndarray,
        info: dict | None = None,
    ) -> np.ndarray:
        self._states.append(
            np.asarray(
                obs,
                dtype=np.float32,
            )
        )
        self._actions.append(
            np.zeros(
                3,
                dtype=np.float32,
            )
        )
        self._rtgs.append(
            float(
                self._rtg
            )
        )
        self._ts.append(
            int(
                self._t
            )
        )

        context = self._context_from_lists(
            self._states,
            self._actions,
            self._rtgs,
            self._ts,
        )

        self.last_context = context

        tensors = self._context_to_torch(
            context,
            self.device,
        )

        with torch.no_grad():
            action, _, _, _ = self.actor.sample(
                **tensors,
                deterministic=self.deterministic,
            )

        requested = (
            action[
                0
            ]
            .cpu()
            .numpy()
            .astype(
                np.float32
            )
        )

        self._actions[
            -1
        ] = requested.copy()

        self._t += 1

        return requested

    def update_applied_action(
        self,
        action: np.ndarray,
    ):
        if not self._actions:
            return

        applied = np.asarray(
            action,
            dtype=np.float32,
        ).reshape(
            -1
        )

        if applied.shape != (3,):
            raise ValueError(
                f"Applied action must have shape (3,), got {applied.shape}"
            )

        self._actions[
            -1
        ] = applied.copy()

    def update_reward(
        self,
        reward: float,
    ):
        self._rtg -= (
            float(reward)
            / self.return_scale
        )
