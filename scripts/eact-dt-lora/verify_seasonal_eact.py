"""Verify EACT on the four seasonal typical weeks (self-contained).

Also optionally writes an expert trajectory .npz for those four seasonal
weeks in exactly the same format as
generate_eact_trajectories_parallel.py, so it can be fed straight into the
DT / LoRA training pipeline.

Seasonal schedule (matches evaluate_lora_vs_eact_season.py):

    seed_i       = base_seed + i
    start_hour_i = data index of seasonal timestamp i
    env.reset(seed=seed_i, options={"start_hour": start_hour_i})

EACT config is the LEGACY one (same as evaluate_lora_vs_eact_season.py):
    env  : RewardWeights(1.0, 2.0, 0.2, 0.5), CoolingConfig(1.0, True)
    ctrl : horizon/maxiter/control_block_hours/gamma/confidence +
           constraint_tolerance=1e-4, intervention_weight=0.1,
           adaptive_beta_floor=0.10, forecast_stress_*=legacy defaults

Usage:
    python verify_seasonal_eact.py ^
        --country PRT ^
        --base-seed 9000 ^
        --save-trajectories ^
        --traj-out data\\trajectories\\eact_PRT_seasonal.npz
"""

from __future__ import annotations

import argparse
import os
import sys

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")

import numpy as np
import pandas as pd

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.abspath(os.path.join(_HERE, "..", ".."))
sys.path.insert(0, _ROOT)

from coastaldc_env import CoastalDCContinuousEnv  # noqa: E402
from coastaldc_env.reward import RewardWeights  # noqa: E402
from coastaldc_env.swhp_cooling import CoolingConfig  # noqa: E402
from coastaldc_env.rollout import (  # noqa: E402
    run_episode,
    save_trajectories,
)
from controllers.eact_mpc import EACTMPCController  # noqa: E402


# --- Legacy defaults (must mirror evaluate_lora_vs_eact_season.py) ---------
DEFAULT_WEIGHT_GRID = 1.0
DEFAULT_WEIGHT_CO2 = 2.0
DEFAULT_WEIGHT_TOTAL = 0.2
DEFAULT_WEIGHT_SMOOTH = 0.5

DEFAULT_COOLING_CONDUCTANCE_MULTIPLIER = 1.0
DEFAULT_THERMAL_SAFETY_SHIELD = True

DEFAULT_CONSTRAINT_TOLERANCE = 1e-4
DEFAULT_INTERVENTION_WEIGHT = 0.1
DEFAULT_ADAPTIVE_BETA_FLOOR = 0.10
DEFAULT_FORECAST_STRESS = "none"
DEFAULT_FORECAST_STRESS_SCALE = 0.0
DEFAULT_FORECAST_STRESS_START_STEP = 0
DEFAULT_FORECAST_STRESS_SEED = 20260717

DEFAULT_SEASONAL_TIMESTAMPS = (
    "2025-01-15 00:00:00",
    "2025-04-15 00:00:00",
    "2025-07-15 00:00:00",
    "2025-10-15 00:00:00",
)
SEASONAL_LABELS = ("winter", "spring", "summer", "autumn")
DEFAULT_FORECAST_DIR = os.path.join(
    _ROOT, "results", "causal_forecasts_v3_gated_bias"
)
DEFAULT_TRAJ_OUT_DIR = os.path.join(_ROOT, "data", "trajectories")


# =============================================================================
# Builders (legacy-aligned)
# =============================================================================

def _build_env(country, seed, forecast_mode, forecast_dir):
    nominal = CoolingConfig()
    return CoastalDCContinuousEnv(
        country=country,
        seed=seed,
        forecast_mode=forecast_mode,
        forecast_dir=forecast_dir,
        reward_weights=RewardWeights(
            w_grid=DEFAULT_WEIGHT_GRID,
            w_co2=DEFAULT_WEIGHT_CO2,
            w_total=DEFAULT_WEIGHT_TOTAL,
            w_smooth=DEFAULT_WEIGHT_SMOOTH,
        ),
        cooling_config=CoolingConfig(
            conductance_mw_per_k=(
                nominal.conductance_mw_per_k
                * DEFAULT_COOLING_CONDUCTANCE_MULTIPLIER
            ),
            enforce_thermal_safety=DEFAULT_THERMAL_SAFETY_SHIELD,
        ),
    )


def _build_eact_ctrl(env, forecast_dir, horizon, maxiter,
                     control_block_hours, gamma, confidence):
    return EACTMPCController(
        env,
        forecast_dir=forecast_dir,
        horizon=horizon,
        maxiter=maxiter,
        control_block_hours=control_block_hours,
        gamma=gamma,
        confidence=confidence,
        constraint_tolerance=DEFAULT_CONSTRAINT_TOLERANCE,
        intervention_weight=DEFAULT_INTERVENTION_WEIGHT,
        adaptive_beta_floor=DEFAULT_ADAPTIVE_BETA_FLOOR,
        forecast_stress_mode=DEFAULT_FORECAST_STRESS,
        forecast_stress_scale=DEFAULT_FORECAST_STRESS_SCALE,
        forecast_stress_start_step=DEFAULT_FORECAST_STRESS_START_STEP,
        forecast_stress_seed=DEFAULT_FORECAST_STRESS_SEED,
    )


# =============================================================================
# Start-hour wrapper so run_episode can be reused as-is
# =============================================================================

class _StartHourEnv:
    """Transparent wrapper that forces env.reset(options={start_hour: ...}).

    run_episode internally calls env.reset(seed=...) with no options, so we
    inject start_hour here. All other attribute access is forwarded verbatim
    to the wrapped env, so run_episode / episode_summary behave exactly as
    they would on the raw env.
    """

    def __init__(self, env, start_hour: int):
        self._env = env
        self._start_hour = int(start_hour)

    def reset(self, *args, **kwargs):
        options = dict(kwargs.get("options") or {})
        options["start_hour"] = self._start_hour
        kwargs["options"] = options
        return self._env.reset(*args, **kwargs)

    def __getattr__(self, name):
        return getattr(self._env, name)

    def __setattr__(self, name, value):
        if name in ("_env", "_start_hour"):
            object.__setattr__(self, name, value)
        else:
            setattr(self._env, name, value)


# =============================================================================
# Seasonal timestamp -> start_hour
# =============================================================================

def _resolve_seasonal_start_hours(data, timestamps, episode_hours, min_start):
    if "timestamp" not in data.columns:
        raise ValueError("Environment data lacks a 'timestamp' column.")
    parsed = pd.to_datetime(data["timestamp"], errors="raise")
    if parsed.duplicated().any():
        raise ValueError("Environment data contain duplicate timestamps.")
    lookup = {ts: i for i, ts in enumerate(parsed)}
    max_start = len(data) - episode_hours - 24
    out = []
    for value in timestamps:
        ts = pd.Timestamp(value)
        if ts not in lookup:
            raise ValueError(f"Timestamp absent from env data: {value}")
        idx = int(lookup[ts])
        if idx < min_start or idx > max_start:
            raise ValueError(
                f"{value} -> start_hour={idx} outside [{min_start}, {max_start}]"
            )
        out.append(idx)
    return out


# =============================================================================
# Episode metrics (for console + CSV comparison)
# =============================================================================

def _thermal(x, ref):
    x = np.asarray(x, dtype=np.float64)
    x = x[np.isfinite(x)]
    if x.size == 0:
        raise ValueError("No finite inlet temperatures recorded.")
    return {
        "avg_t_inlet": float(x.mean()),
        "p95_t_inlet": float(np.percentile(x, 95)),
        "max_t_inlet": float(x.max()),
        "thermal_exposure_c_h": float(np.maximum(x - ref, 0.0).sum()),
        "thermal_event_hours": float((x > ref).sum()),
    }


def _metrics_from_traj(traj, ref):
    rewards = np.asarray(traj["rewards"], dtype=np.float64)
    temps = np.asarray(
        traj.get("temperatures", traj.get("t_inlet", [])),
        dtype=np.float64,
    )
    thermal = _thermal(temps, ref) if temps.size else {}
    return {
        "return": float(rewards.sum()),
        **thermal,
    }


# =============================================================================
# Main
# =============================================================================

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--country", default="PRT")
    p.add_argument("--base-seed", type=int, default=9000)
    p.add_argument("--forecast-mode",
                   choices=("oracle", "causal", "none"), default="causal")
    p.add_argument("--forecast-dir", default=None)
    p.add_argument("--seasonal-timestamps", nargs="*",
                   default=list(DEFAULT_SEASONAL_TIMESTAMPS))
    p.add_argument("--thermal-reference", type=float, default=27.0)
    p.add_argument("--horizon", type=int, default=24)
    p.add_argument("--maxiter", type=int, default=20)
    p.add_argument("--block-hours", type=int, default=6)
    p.add_argument("--gamma", type=float, default=0.995)
    p.add_argument("--confidence", type=float, default=0.90)
    p.add_argument("--eval-csv", default=None,
                   help="Optional episode_results.csv from the evaluator.")
    p.add_argument("--atol", type=float, default=1e-3)

    # --- Trajectory saving ---------------------------------------------
    p.add_argument("--save-trajectories",
                   action=argparse.BooleanOptionalAction, default=True,
                   help="Save the four seasonal weeks as an expert .npz.")
    p.add_argument("--traj-out", default=None,
                   help=("Output .npz path. Default: "
                         "data/trajectories/eact_<COUNTRY>_seasonal.npz"))
    p.add_argument("--traj-source", default="eact",
                   help="Value stored as `source` inside the .npz.")
    args = p.parse_args()

    forecast_dir = args.forecast_dir or DEFAULT_FORECAST_DIR

    probe = _build_env(
        args.country, args.base_seed, args.forecast_mode, forecast_dir,
    )
    causal_min_history = (
        int(probe._causal_min_history) if args.forecast_mode == "causal"
        else 0
    )
    episode_hours = int(probe.episode_hours)

    start_hours = _resolve_seasonal_start_hours(
        probe.data, args.seasonal_timestamps,
        episode_hours=episode_hours, min_start=causal_min_history,
    )

    rows = []
    trajs = []

    for i, sh in enumerate(start_hours):
        label = SEASONAL_LABELS[i] if i < len(SEASONAL_LABELS) else f"s{i}"
        seed = args.base_seed + i

        env = _build_env(
            args.country, seed, args.forecast_mode, forecast_dir,
        )
        ctrl = _build_eact_ctrl(
            env, forecast_dir, args.horizon, args.maxiter,
            args.block_hours, args.gamma, args.confidence,
        )

        # Wrap the env so run_episode's internal env.reset() gets start_hour.
        wrapped = _StartHourEnv(env, sh)
        ctrl.reset()
        traj = run_episode(wrapped, ctrl, seed=int(seed))
        trajs.append(traj)

        m = _metrics_from_traj(traj, args.thermal_reference)
        row = {
            "season": label,
            "seed": seed,
            "start_hour": sh,
            **m,
        }
        rows.append(row)

        print(
            f"[{label:6s}] seed={seed} start_hour={sh} | "
            f"R={row['return']:.2f} | "
            f"P95={row.get('p95_t_inlet', float('nan')):.3f} | "
            f"Tex={row.get('thermal_exposure_c_h', float('nan')):.3f} | "
            f"EventH={row.get('thermal_event_hours', float('nan')):.1f}"
        )

    # --- Save trajectories ---------------------------------------------
    if args.save_trajectories:
        out = args.traj_out or os.path.join(
            DEFAULT_TRAJ_OUT_DIR,
            f"eact_{args.country}_seasonal.npz",
        )
        os.makedirs(os.path.dirname(out), exist_ok=True)
        save_trajectories(out, trajs, source=args.traj_source)
        print(f"\n[info] saved {len(trajs)} seasonal trajectories -> {out}")

    # --- Compare against evaluator CSV ---------------------------------
    if args.eval_csv:
        print("\n--- comparison against evaluator CSV ---")
        ev = pd.read_csv(args.eval_csv)
        ev_eact = ev[ev["algorithm"] == "eact"].copy()
        if ev_eact.empty:
            print(f"[warn] no 'eact' rows found in {args.eval_csv}")
            return

        compare_cols = (
            "return", "p95_t_inlet",
            "thermal_exposure_c_h", "thermal_event_hours",
            "sla_violation_mwh", "sla_violation_hours",
            "terminal_unserved_mwh", "final_backlog_mwh",
            "co2_kg", "e_grid_mwh", "e_cooling_mwh", "e_pump_mwh",
        )
        for row in rows:
            m = ev_eact
            if "season_label" in m.columns:
                m = m[m["season_label"] == row["season"]]
            if "start_hour" in m.columns and len(m) > 1:
                m = m[np.isclose(
                    m["start_hour"].astype(float),
                    float(row["start_hour"]),
                )]
            if m.empty:
                print(f"[warn] no evaluator row for {row['season']}")
                continue
            e = m.iloc[0]
            for col in compare_cols:
                if col not in e or col not in row:
                    continue
                a = float(row[col])
                b = float(e[col])
                ok = abs(a - b) <= args.atol
                print(
                    f"  {row['season']:6s} {col:24s}: "
                    f"gen={a:+.6f} eval={b:+.6f} "
                    f"diff={a - b:+.2e} "
                    f"{'OK' if ok else 'MISMATCH'}"
                )


if __name__ == "__main__":
    main()