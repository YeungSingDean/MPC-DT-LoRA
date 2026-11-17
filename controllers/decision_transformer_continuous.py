"""Continuous-action Decision Transformer.

Differences from the discrete DT used in the previous DPT paper:
  - the action head is a regression head (MSE loss), not cross-entropy;
  - predicted actions are tanh-squashed and affine-mapped into the env action box.

Token order per timestep: (return-to-go, state, action); action is predicted from
the state token with full causal context.

This version also supports:
  - return-to-go scaling for numerically stable conditioning;
  - correct sampling of the final valid context window;
  - replacing the latest requested action in rollout history with the action
    actually applied by the environment after safety / feasibility projection.
"""

from __future__ import annotations

import math

import numpy as np
import torch
import torch.nn as nn

from controllers.behavior_cloning import squash_to_action_range


class CausalSelfAttention(nn.Module):
    def __init__(
        self,
        embed_dim: int,
        n_heads: int,
        max_tokens: int,
        dropout: float,
    ):
        super().__init__()

        self.attn = nn.MultiheadAttention(
            embed_dim,
            n_heads,
            dropout=dropout,
            batch_first=True,
        )

        mask = torch.triu(
            torch.ones(
                max_tokens,
                max_tokens,
                dtype=torch.bool,
            ),
            diagonal=1,
        )

        self.register_buffer(
            "causal_mask",
            mask,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        T = x.size(1)

        out, _ = self.attn(
            x,
            x,
            x,
            attn_mask=self.causal_mask[:T, :T],
            need_weights=False,
        )

        return out


class Block(nn.Module):
    def __init__(
        self,
        embed_dim: int,
        n_heads: int,
        max_tokens: int,
        dropout: float,
    ):
        super().__init__()

        self.ln1 = nn.LayerNorm(embed_dim)

        self.attn = CausalSelfAttention(
            embed_dim,
            n_heads,
            max_tokens,
            dropout,
        )

        self.ln2 = nn.LayerNorm(embed_dim)

        self.mlp = nn.Sequential(
            nn.Linear(
                embed_dim,
                4 * embed_dim,
            ),
            nn.GELU(),
            nn.Linear(
                4 * embed_dim,
                embed_dim,
            ),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(
            self.ln1(x)
        )

        x = x + self.mlp(
            self.ln2(x)
        )

        return x


class ContinuousDecisionTransformer(nn.Module):
    def __init__(
        self,
        obs_dim: int,
        act_dim: int = 3,
        context_len: int = 24,
        embed_dim: int = 128,
        n_layers: int = 3,
        n_heads: int = 4,
        dropout: float = 0.1,
        max_ep_len: int = 200,
    ):
        super().__init__()

        self.obs_dim = obs_dim
        self.act_dim = act_dim
        self.context_len = context_len

        max_tokens = 3 * context_len

        self.embed_rtg = nn.Linear(
            1,
            embed_dim,
        )

        self.embed_state = nn.Linear(
            obs_dim,
            embed_dim,
        )

        self.embed_action = nn.Linear(
            act_dim,
            embed_dim,
        )

        self.embed_time = nn.Embedding(
            max_ep_len,
            embed_dim,
        )

        self.ln_in = nn.LayerNorm(
            embed_dim
        )

        self.blocks = nn.ModuleList([
            Block(
                embed_dim,
                n_heads,
                max_tokens,
                dropout,
            )
            for _ in range(n_layers)
        ])

        self.ln_out = nn.LayerNorm(
            embed_dim
        )

        # Continuous action regression head.
        self.action_head = nn.Linear(
            embed_dim,
            act_dim,
        )

    def forward(
        self,
        states: torch.Tensor,
        actions: torch.Tensor,
        rtgs: torch.Tensor,
        timesteps: torch.Tensor,
    ) -> torch.Tensor:
        """Forward pass.

        Parameters
        ----------
        states:
            Shape (B, T, obs_dim).

        actions:
            Shape (B, T, act_dim).

        rtgs:
            Shape (B, T, 1). These values are expected to already be scaled
            consistently with the return_scale used during training / rollout.

        timesteps:
            Shape (B, T).

        Returns
        -------
        torch.Tensor
            Predicted continuous actions with shape (B, T, act_dim).
        """
        B, T, _ = states.shape

        time_embedding = self.embed_time(
            timesteps
        )

        rtg_tokens = (
            self.embed_rtg(rtgs)
            + time_embedding
        )

        state_tokens = (
            self.embed_state(states)
            + time_embedding
        )

        action_tokens = (
            self.embed_action(actions)
            + time_embedding
        )

        # Token order:
        # (RTG_t, state_t, action_t) for each timestep.
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

        x = self.ln_in(x)

        for block in self.blocks:
            x = block(x)

        x = self.ln_out(x)

        # Action prediction is generated from the state-token positions.
        hidden_state_tokens = x[:, 1::3]

        raw_actions = self.action_head(
            hidden_state_tokens
        )

        return squash_to_action_range(
            raw_actions
        )


class DecisionTransformerController:
    name = "decision_transformer"

    def __init__(
        self,
        obs_dim: int,
        target_return: float,
        context_len: int = 24,
        model_path: str | None = None,
        device: str = "cpu",
        return_scale: float = 100.0,
        **model_kwargs,
    ):
        """Decision Transformer rollout controller.

        Parameters
        ----------
        obs_dim:
            Observation dimension.

        target_return:
            Desired episode return in the environment's original reward units.

        context_len:
            Number of timesteps in the Transformer context.

        model_path:
            Optional model checkpoint.

        device:
            PyTorch device.

        return_scale:
            Positive scaling factor applied to return-to-go values. It must match
            the return_scale used by train_dt().
        """
        if return_scale <= 0:
            raise ValueError(
                "return_scale must be > 0"
            )

        self.device = device

        self.model = ContinuousDecisionTransformer(
            obs_dim,
            context_len=context_len,
            **model_kwargs,
        ).to(device)

        if model_path:
            self.model.load_state_dict(
                torch.load(
                    model_path,
                    map_location=device,
                )
            )

        self.model.eval()

        self.target_return = float(
            target_return
        )

        self.context_len = int(
            context_len
        )

        self.return_scale = float(
            return_scale
        )

        self.reset()

    def reset(self):
        """Reset rollout history at the beginning of an episode."""
        self._states = []
        self._actions = []
        self._rtgs = []
        self._ts = []

        # Internally the DT receives scaled RTG values.
        self._rtg = (
            self.target_return
            / self.return_scale
        )

        self._t = 0

    def act(
        self,
        obs: np.ndarray,
        info: dict | None = None,
    ) -> np.ndarray:
        """Predict the next requested action."""
        self._states.append(
            np.asarray(
                obs,
                dtype=np.float32,
            )
        )

        # Placeholder for the current action token.
        # Because action_t follows state_t in the causal token sequence,
        # this placeholder cannot leak into the prediction for action_t.
        self._actions.append(
            np.zeros(
                3,
                dtype=np.float32,
            )
        )

        self._rtgs.append(
            float(self._rtg)
        )

        self._ts.append(
            int(self._t)
        )

        K = self.context_len

        states = torch.as_tensor(
            np.stack(
                self._states[-K:]
            ),
            dtype=torch.float32,
            device=self.device,
        )[None]

        actions = torch.as_tensor(
            np.stack(
                self._actions[-K:]
            ),
            dtype=torch.float32,
            device=self.device,
        )[None]

        rtgs = torch.as_tensor(
            np.asarray(
                self._rtgs[-K:],
                dtype=np.float32,
            ),
            dtype=torch.float32,
            device=self.device,
        )[None, :, None]

        timesteps = torch.as_tensor(
            np.asarray(
                self._ts[-K:],
                dtype=np.int64,
            ),
            dtype=torch.int64,
            device=self.device,
        )[None]

        with torch.no_grad():
            pred = self.model(
                states,
                actions,
                rtgs,
                timesteps,
            )

        requested_action = (
            pred[0, -1]
            .cpu()
            .numpy()
            .astype(np.float32)
        )

        # Temporarily store the requested action. After env.step(), rollout.py
        # calls update_applied_action() so this history entry is replaced by
        # the action actually executed by the environment.
        self._actions[-1] = (
            requested_action.copy()
        )

        self._t += 1

        return requested_action

    def update_applied_action(
        self,
        action: np.ndarray,
    ):
        """Replace the latest requested action with the actually applied action.

        The environment may change the raw controller output through workload
        feasibility projection, thermal safety projection, rate limiting, etc.

        DT training uses applied_actions by default, so rollout history must use
        the same action semantics to avoid train-evaluation mismatch.
        """
        if not self._actions:
            return

        applied_action = np.asarray(
            action,
            dtype=np.float32,
        ).reshape(-1)

        if applied_action.shape != (3,):
            raise ValueError(
                "Applied action must have shape (3,), "
                f"got {applied_action.shape}"
            )

        self._actions[-1] = (
            applied_action.copy()
        )

    def update_reward(
        self,
        reward: float,
    ):
        """Update the scaled return-to-go after env.step().

        Original:
            RTG_{t+1} = RTG_t - reward_t

        Because the internally stored RTG is scaled:
            scaled_RTG_{t+1}
                = scaled_RTG_t - reward_t / return_scale
        """
        self._rtg -= (
            float(reward)
            / self.return_scale
        )


# ----------------------------------------------------------------------
# Training
# ----------------------------------------------------------------------

def train_dt(
    model: ContinuousDecisionTransformer,
    trajectories: list[dict],
    epochs: int = 20,
    steps_per_epoch: int = 500,
    batch_size: int = 64,
    lr: float = 1e-4,
    device: str = "cpu",
    seed: int = 0,
    return_scale: float = 100.0,
) -> list[float]:
    """Train the continuous Decision Transformer.

    Parameters
    ----------
    model:
        ContinuousDecisionTransformer.

    trajectories:
        List of trajectory dictionaries containing:
        - observations: (T, obs_dim)
        - actions:      (T, 3)
        - rewards:      (T,)

        With the updated rollout loader, ``actions`` should normally contain
        applied actions.

    epochs:
        Number of training epochs.

    steps_per_epoch:
        Number of gradient updates per epoch.

    batch_size:
        Number of sampled sequence windows per gradient update.

    lr:
        AdamW learning rate.

    device:
        PyTorch device.

    seed:
        RNG seed.

    return_scale:
        Positive scale applied to all return-to-go values. This must match
        DecisionTransformerController(return_scale=...).

    Returns
    -------
    list[float]
        Mean MSE loss for each epoch.
    """
    if return_scale <= 0:
        raise ValueError(
            "return_scale must be > 0"
        )

    if not trajectories:
        raise ValueError(
            "trajectories must not be empty"
        )

    rng = np.random.default_rng(
        seed
    )

    K = model.context_len

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=lr,
        weight_decay=1e-4,
    )

    model.train()

    # --------------------------------------------------------------
    # Pre-compute scaled return-to-go for every trajectory.
    #
    # RTG_t = r_t + r_{t+1} + ... + r_T
    # --------------------------------------------------------------
    rtgs_all = []

    for traj in trajectories:
        rewards = np.asarray(
            traj["rewards"],
            dtype=np.float32,
        )

        rtg = np.cumsum(
            rewards[::-1]
        )[::-1]

        rtg = (
            rtg
            / float(return_scale)
        ).astype(np.float32)

        rtgs_all.append(
            rtg
        )

    losses: list[float] = []

    for _ in range(epochs):
        epoch_loss = 0.0

        for _ in range(steps_per_epoch):
            batch_states = []
            batch_actions = []
            batch_rtgs = []
            batch_timesteps = []

            for _ in range(batch_size):
                traj_idx = int(
                    rng.integers(
                        len(trajectories)
                    )
                )

                traj = trajectories[
                    traj_idx
                ]

                rtg = rtgs_all[
                    traj_idx
                ]

                T = len(
                    traj["rewards"]
                )

                # --------------------------------------------------
                # Correct valid start range.
                #
                # Example:
                # T = 168, K = 24
                #
                # Valid starts are:
                # 0, 1, ..., 144
                #
                # Hence upper bound must be T-K+1.
                # --------------------------------------------------
                max_start = max(
                    T - K + 1,
                    1,
                )

                start = int(
                    rng.integers(
                        0,
                        max_start,
                    )
                )

                end = min(
                    start + K,
                    T,
                )

                states = np.asarray(
                    traj["observations"][
                        start:end
                    ],
                    dtype=np.float32,
                )

                actions = np.asarray(
                    traj["actions"][
                        start:end
                    ],
                    dtype=np.float32,
                )

                returns_to_go = np.asarray(
                    rtg[
                        start:end
                    ],
                    dtype=np.float32,
                )

                timesteps = np.arange(
                    start,
                    end,
                    dtype=np.int64,
                )

                # --------------------------------------------------
                # Left padding for trajectories shorter than K.
                # Standard 168-step episodes normally do not need this.
                # --------------------------------------------------
                pad = (
                    K
                    - len(states)
                )

                if pad > 0:
                    states = np.concatenate(
                        [
                            np.zeros(
                                (
                                    pad,
                                    states.shape[1],
                                ),
                                dtype=np.float32,
                            ),
                            states,
                        ],
                        axis=0,
                    )

                    actions = np.concatenate(
                        [
                            np.zeros(
                                (
                                    pad,
                                    actions.shape[1],
                                ),
                                dtype=np.float32,
                            ),
                            actions,
                        ],
                        axis=0,
                    )

                    returns_to_go = np.concatenate(
                        [
                            np.zeros(
                                pad,
                                dtype=np.float32,
                            ),
                            returns_to_go,
                        ],
                        axis=0,
                    )

                    timesteps = np.concatenate(
                        [
                            np.zeros(
                                pad,
                                dtype=np.int64,
                            ),
                            timesteps,
                        ],
                        axis=0,
                    )

                batch_states.append(
                    states
                )

                batch_actions.append(
                    actions
                )

                batch_rtgs.append(
                    returns_to_go
                )

                batch_timesteps.append(
                    timesteps
                )

            states_tensor = torch.as_tensor(
                np.stack(
                    batch_states
                ),
                dtype=torch.float32,
                device=device,
            )

            actions_tensor = torch.as_tensor(
                np.stack(
                    batch_actions
                ),
                dtype=torch.float32,
                device=device,
            )

            rtgs_tensor = torch.as_tensor(
                np.stack(
                    batch_rtgs
                ),
                dtype=torch.float32,
                device=device,
            )[..., None]

            timesteps_tensor = torch.as_tensor(
                np.stack(
                    batch_timesteps
                ),
                dtype=torch.int64,
                device=device,
            )

            predicted_actions = model(
                states_tensor,
                actions_tensor,
                rtgs_tensor,
                timesteps_tensor,
            )

            loss = nn.functional.mse_loss(
                predicted_actions,
                actions_tensor,
            )

            optimizer.zero_grad()
            loss.backward()

            torch.nn.utils.clip_grad_norm_(
                model.parameters(),
                0.5,
            )

            optimizer.step()

            epoch_loss += float(
                loss.item()
            )

        losses.append(
            epoch_loss
            / steps_per_epoch
        )

    model.eval()

    return losses
