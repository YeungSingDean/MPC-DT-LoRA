"""Forecast-aware joint LoRA residual policy for V6.

V6 keeps the supervised V3 Decision Transformer intact while adding two things
that the V1-V5 thermal policies did not have:

1) full causal 24 h forecast context for
       fixed load, flexible arrival, SST, wind, carbon
   compressed into 4 x 6 h means per variable (5 x 4 = 20 features);

2) a small trainable workload residual in addition to cooling residuals.

The frozen V3 DT still consumes the original 29-dimensional observation, so an
existing V3 checkpoint remains directly usable.  The 20-dimensional forecast
vector is encoded by a separate trainable MLP and fused only into the LoRA
residual policy.

Final action:
    workload = clip(a_V3_work + delta_work, -1, 1)
    setpoint = clip(a_V3_set  + delta_set,  -1, 1)
    pump     = clip(a_V3_pump + delta_pump,  0, 1)

Default residual limits:
    workload +/- 0.15
    setpoint +/- 0.50
    pump     +/- 0.50

Forecast feature order:
    fixed_load[4],
    flexible_arrival[4],
    SST[4],
    wind[4],
    carbon[4]

All causal forecasts use only history through the current hour.
"""

from __future__ import annotations

import copy
from collections.abc import Mapping

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

from controllers.decision_transformer_continuous import (
    ContinuousDecisionTransformer,
)
from controllers.lora_thermal_dt import (
    DTContext,
    freeze_module,
    forward_state_hidden,
    inject_ffn_lora,
    pad_context,
)


FORECAST_BUCKETS = 4
FORECAST_HOURS = 24
FORECAST_COLUMNS = (
    "fixed_load_mw",
    "flexible_arrival_mw",
    "sst_c",
    "wind_mw",
    "carbon_kg_per_mwh",
)
FULL_FORECAST_DIM = len(FORECAST_COLUMNS) * FORECAST_BUCKETS


def _extract_forecast_values(
    forecast,
    column: str,
) -> np.ndarray:
    if isinstance(forecast, pd.DataFrame):
        if column not in forecast.columns:
            raise KeyError(
                f"Full causal forecast output missing column '{column}'."
            )
        values = forecast[column].to_numpy(dtype=float)
    elif isinstance(forecast, Mapping):
        if column not in forecast:
            raise KeyError(
                f"Full causal forecast output missing key '{column}'."
            )
        values = np.asarray(
            forecast[column],
            dtype=float,
        )
    else:
        raise TypeError(
            "Causal forecaster must return a mapping or DataFrame, got "
            f"{type(forecast).__name__}."
        )

    values = values.reshape(-1)

    if values.size < FORECAST_HOURS:
        raise ValueError(
            f"Forecast for '{column}' contains only {values.size} values; "
            f"{FORECAST_HOURS} are required."
        )

    values = values[:FORECAST_HOURS]

    if not np.all(
        np.isfinite(values)
    ):
        raise ValueError(
            f"Forecast for '{column}' contains non-finite values."
        )

    return values


def _bucket_means(
    values: np.ndarray,
) -> np.ndarray:
    values = np.asarray(
        values,
        dtype=float,
    ).reshape(
        FORECAST_HOURS
    )

    return values.reshape(
        FORECAST_BUCKETS,
        FORECAST_HOURS // FORECAST_BUCKETS,
    ).mean(
        axis=1
    )


def _normalize_full_forecast(
    env,
    raw: dict[str, np.ndarray],
) -> np.ndarray:
    cap = float(
        env.wl_cfg.it_capacity_mw
    )
    wind_cap = max(
        float(
            env.wind_cfg.wind_capacity_mw
        ),
        1e-9,
    )

    parts = [
        raw[
            "fixed_load_mw"
        ] / max(
            cap,
            1e-9,
        ),
        raw[
            "flexible_arrival_mw"
        ] / max(
            cap,
            1e-9,
        ),
        raw[
            "sst_c"
        ] / 30.0,
        raw[
            "wind_mw"
        ] / wind_cap,
        raw[
            "carbon_kg_per_mwh"
        ] / 1000.0,
    ]

    out = np.concatenate(
        parts,
        axis=0,
    ).astype(
        np.float32
    )

    if out.shape != (
        FULL_FORECAST_DIM,
    ):
        raise RuntimeError(
            "Unexpected full forecast feature shape: "
            f"{out.shape}; expected {(FULL_FORECAST_DIM,)}."
        )

    return out


def build_full_forecast_features(
    env,
) -> np.ndarray:
    """Return the V6 20-D full forecast vector for the current env state.

    ``causal``:
        Uses env.forecaster.predict(history, origin=current) and therefore has
        no future-data leakage.

    ``oracle``:
        Uses the realised next 24 h directly.  This is an upper-bound mode and
        should not be used for the main causal V6 result.

    ``none``:
        Returns zeros.
    """
    mode = str(
        getattr(
            env,
            "forecast_mode",
            "none",
        )
    ).lower()

    global_t = int(
        env._t0
        + env._step_idx
    )

    cache = getattr(
        env,
        "_v6_full_forecast_cache",
        None,
    )
    if cache is None:
        cache = {}
        setattr(
            env,
            "_v6_full_forecast_cache",
            cache,
        )

    key = (
        mode,
        global_t,
    )

    cached = cache.get(
        key
    )
    if cached is not None:
        return cached.copy()

    if mode == "none":
        features = np.zeros(
            FULL_FORECAST_DIM,
            dtype=np.float32,
        )
        cache[
            key
        ] = features
        return features.copy()

    if mode == "oracle":
        future = env.data.iloc[
            global_t + 1:
            global_t + 1 + FORECAST_HOURS
        ]

        if len(
            future
        ) < FORECAST_HOURS:
            raise ValueError(
                "Not enough future rows for a 24 h oracle V6 forecast at "
                f"data hour {global_t}."
            )

        raw = {
            col: _bucket_means(
                future[
                    col
                ].to_numpy(
                    dtype=float
                )
            )
            for col in FORECAST_COLUMNS
        }

    elif mode == "causal":
        forecaster = getattr(
            env,
            "forecaster",
            None,
        )
        if forecaster is None:
            raise RuntimeError(
                "V6 full causal forecast requested but env.forecaster is None."
            )

        history = env.data.iloc[
            :global_t + 1
        ].copy()

        try:
            forecast = forecaster.predict(
                history,
                origin=len(
                    history
                )
                - 1,
            )
        except Exception as exc:
            raise RuntimeError(
                "Failed to build the V6 full causal forecast at data hour "
                f"{global_t}. Ensure the fitted causal model predicts "
                f"{FORECAST_COLUMNS}."
            ) from exc

        raw = {
            col: _bucket_means(
                _extract_forecast_values(
                    forecast,
                    col,
                )
            )
            for col in FORECAST_COLUMNS
        }

    else:
        raise ValueError(
            f"Unsupported forecast_mode={mode!r}."
        )

    features = _normalize_full_forecast(
        env,
        raw,
    )

    cache[
        key
    ] = features.copy()

    return features


class ForecastJointLoRAActor(nn.Module):
    """Frozen V3 DT + LoRA hidden + full-forecast encoder + 3-D residual."""

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
        forecast_dim: int = FULL_FORECAST_DIM,
        forecast_hidden_dim: int = 64,
        residual_hidden_dim: int = 96,
        residual_workload_limit: float = 0.15,
        residual_setpoint_limit: float = 0.50,
        residual_pump_limit: float = 0.50,
        initial_log_std: float = -1.0,
        device: str = "cpu",
    ):
        super().__init__()

        if forecast_dim != FULL_FORECAST_DIM:
            raise ValueError(
                f"V6 expects forecast_dim={FULL_FORECAST_DIM}, "
                f"got {forecast_dim}."
            )

        for name, value in (
            (
                "residual_workload_limit",
                residual_workload_limit,
            ),
            (
                "residual_setpoint_limit",
                residual_setpoint_limit,
            ),
            (
                "residual_pump_limit",
                residual_pump_limit,
            ),
        ):
            if value < 0:
                raise ValueError(
                    f"{name} must be non-negative."
                )

        self.obs_dim = int(
            obs_dim
        )
        self.context_len = int(
            context_len
        )
        self.forecast_dim = int(
            forecast_dim
        )

        self.residual_workload_limit = float(
            residual_workload_limit
        )
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

        self.forecast_encoder = nn.Sequential(
            nn.Linear(
                forecast_dim,
                forecast_hidden_dim,
            ),
            nn.LayerNorm(
                forecast_hidden_dim
            ),
            nn.GELU(),
            nn.Linear(
                forecast_hidden_dim,
                forecast_hidden_dim,
            ),
            nn.GELU(),
        )

        self.residual_backbone = nn.Sequential(
            nn.Linear(
                embed_dim
                + forecast_hidden_dim,
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
            3,
        )
        self.log_std_head = nn.Linear(
            residual_hidden_dim,
            3,
        )

        # Deterministic initial policy exactly reproduces V3.
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
            float(
                initial_log_std
            ),
        )

        self.log_std_min = -5.0
        self.log_std_max = 1.0

        self.to(
            device
        )

    def train(
        self,
        mode: bool = True,
    ):
        super().train(
            mode
        )
        self.base_model.eval()
        self.adapted_model.eval()
        return self

    def trainable_parameters(
        self,
    ):
        return [
            p
            for p in self.parameters()
            if p.requires_grad
        ]

    def trainable_state_dict(
        self,
    ) -> dict[str, torch.Tensor]:
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
                f"Unknown V6 adapter keys: {unknown[:10]}"
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
        B = states.shape[
            0
        ]
        device = states.device

        base_actions = torch.empty(
            (
                B,
                3,
            ),
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

        for length_value in torch.unique(
            lengths.detach()
        ).tolist():
            L = int(
                length_value
            )

            idx = torch.nonzero(
                lengths
                == L,
                as_tuple=False,
            ).squeeze(
                -1
            )

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
                )[
                    :,
                    -1,
                ]

            hidden = forward_state_hidden(
                self.adapted_model,
                s,
                a,
                r,
                t,
            )[
                :,
                -1,
            ]

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
        forecast_features: torch.Tensor,
        deterministic: bool = False,
    ):
        if forecast_features.ndim != 2:
            raise ValueError(
                "forecast_features must have shape (B, forecast_dim)."
            )

        if forecast_features.shape[
            1
        ] != self.forecast_dim:
            raise ValueError(
                "Wrong forecast feature dimension: "
                f"{forecast_features.shape[1]} vs {self.forecast_dim}."
            )

        base_action, hidden = self._model_outputs_grouped(
            states=states,
            actions=actions,
            rtgs=rtgs,
            timesteps=timesteps,
            lengths=lengths,
        )

        forecast_hidden = self.forecast_encoder(
            forecast_features
        )

        fused = torch.cat(
            [
                hidden,
                forecast_hidden,
            ],
            dim=-1,
        )

        z = self.residual_backbone(
            fused
        )

        mu = self.mu_head(
            z
        )

        log_std = torch.clamp(
            self.log_std_head(
                z
            ),
            self.log_std_min,
            self.log_std_max,
        )

        std = torch.exp(
            log_std
        )

        dist = torch.distributions.Normal(
            mu,
            std,
        )

        if deterministic:
            pre_tanh = mu
        else:
            pre_tanh = dist.rsample()

        normalized_residual = torch.tanh(
            pre_tanh
        )

        if deterministic:
            log_prob = torch.zeros(
                (
                    states.shape[
                        0
                    ],
                    1,
                ),
                dtype=states.dtype,
                device=states.device,
            )
        else:
            log_prob = (
                dist.log_prob(
                    pre_tanh
                )
                - torch.log(
                    1.0
                    - normalized_residual.pow(
                        2
                    )
                    + 1e-6
                )
            ).sum(
                dim=-1,
                keepdim=True,
            )

        delta_work = (
            normalized_residual[
                :,
                0
            ]
            * self.residual_workload_limit
        )
        delta_setpoint = (
            normalized_residual[
                :,
                1
            ]
            * self.residual_setpoint_limit
        )
        delta_pump = (
            normalized_residual[
                :,
                2
            ]
            * self.residual_pump_limit
        )

        workload = torch.clamp(
            base_action[
                :,
                0
            ]
            + delta_work,
            -1.0,
            1.0,
        )

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


class ForecastJointLoRADTController:
    name = "forecast_joint_lora_dt_v6"

    def __init__(
        self,
        *,
        actor: ForecastJointLoRAActor,
        env,
        target_return: float,
        return_scale: float = 100.0,
        deterministic: bool = True,
    ):
        if return_scale <= 0:
            raise ValueError(
                "return_scale must be > 0."
            )

        self.actor = actor
        self.env = env
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

    def reset(
        self,
    ):
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
        self.last_forecast = None
        self.last_base_action = None
        self.last_normalized_residual = None

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
            )[
                None
            ],
            actions=torch.as_tensor(
                context.actions,
                dtype=torch.float32,
                device=device,
            )[
                None
            ],
            rtgs=torch.as_tensor(
                context.rtgs,
                dtype=torch.float32,
                device=device,
            )[
                None
            ],
            timesteps=torch.as_tensor(
                context.timesteps,
                dtype=torch.int64,
                device=device,
            )[
                None
            ],
            lengths=torch.as_tensor(
                [
                    context.length
                ],
                dtype=torch.int64,
                device=device,
            ),
        )

    def act(
        self,
        obs: np.ndarray,
        info: dict | None = None,
    ) -> np.ndarray:
        del info

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

        forecast = build_full_forecast_features(
            self.env
        )

        self.last_context = context
        self.last_forecast = forecast.copy()

        tensors = self._context_to_torch(
            context,
            self.device,
        )

        forecast_tensor = torch.as_tensor(
            forecast,
            dtype=torch.float32,
            device=self.device,
        )[
            None
        ]

        with torch.no_grad():
            (
                action,
                _,
                base_action,
                residual,
            ) = self.actor.sample(
                **tensors,
                forecast_features=forecast_tensor,
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

        self.last_base_action = (
            base_action[
                0
            ]
            .cpu()
            .numpy()
            .astype(
                np.float32
            )
        )
        self.last_normalized_residual = (
            residual[
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

        if applied.shape != (
            3,
        ):
            raise ValueError(
                f"Applied action must have shape (3,), got {applied.shape}."
            )

        self._actions[
            -1
        ] = applied.copy()

    def update_reward(
        self,
        reward: float,
    ):
        self._rtg -= (
            float(
                reward
            )
            / self.return_scale
        )
