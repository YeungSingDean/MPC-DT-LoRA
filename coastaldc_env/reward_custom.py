"""Training-only pump bonus; leave coastaldc_env/reward.py unchanged.

Usage in the RL collection loop (after env.step):
    pump_reward = PumpBonusReward(weight=0.05)   # construct once
    replay_reward, info = pump_reward(reward, next_info["applied_action"])
    replay.add(..., reward=replay_reward, ...)

Continue to pass the ORIGINAL reward to DT.update_reward(), validation,
checkpoint selection, and reported episode returns. The existing SAC cold
penalty, entropy term, and thermal cost critics are not replaced here.

Bonus = weight * applied_action[2], where the actual pump flow action is in
[0, 1]. This explicitly introduces a preference for more pump flow; it does
not prove greater cooling efficiency and can encourage excessive pumping.
It rewards neither pump electricity nor a request rejected by the environment.
"""

from __future__ import annotations

from dataclasses import dataclass
import math

import numpy as np


@dataclass(frozen=True)
class PumpBonusReward:
    """Add one bounded positive pump-flow bonus to a base step reward.

    weight=0 is the no-bonus control; a positive weight encourages higher
    actual pump flow. With weight=0.05, the bonus is at most 0.05 per step.
    No target action, EACT label, or SST-dependent rule is used.
    """

    weight: float = 0.05

    def __post_init__(self) -> None:
        weight = float(self.weight)
        if not math.isfinite(weight) or weight < 0.0:
            raise ValueError("Pump bonus weight must be finite and >= 0.")
        object.__setattr__(self, "weight", weight)

    def __call__(
        self,
        base_reward: float,
        applied_action: np.ndarray,
    ) -> tuple[float, dict[str, float]]:
        base_reward = float(base_reward)
        if not math.isfinite(base_reward):
            raise ValueError("base_reward must be finite.")

        action = np.asarray(applied_action, dtype=np.float64)
        if action.shape != (3,) or not np.all(np.isfinite(action)):
            raise ValueError("applied_action must be a finite shape-(3,) vector.")

        pump = float(action[2])
        if pump < -1e-6 or pump > 1.0 + 1e-6:
            raise ValueError(f"Applied pump must be in [0, 1], got {pump}.")
        pump = float(np.clip(pump, 0.0, 1.0))  # tolerate round-off only

        bonus = self.weight * pump
        replay_reward = base_reward + bonus
        return replay_reward, {
            "base_reward": base_reward,
            "pump_applied": pump,
            "pump_bonus": bonus,
            "replay_reward_before_cold": replay_reward,
        }
