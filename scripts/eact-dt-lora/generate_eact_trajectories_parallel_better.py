"""Generate EACT-MPC expert trajectories for offline training (multiprocess).

This generator is configured to match the LEGACY paired-evaluation script
`scripts/eact-dt-lora/evaluate_lora_vs_eact_season.py` exactly:

  * Environment is built with the same reward_weights and cooling_config
    (see DEFAULT_* constants below), plus forecast_mode / forecast_dir,
    mirroring the legacy `make_env`.

  * EACTMPCController is built with the same 13-keyword set as the legacy
    `_shared_eact_kwargs` (including constraint_tolerance, intervention_weight,
    adaptive_beta_floor and the forecast_stress_* knobs).

If you change any of these, trajectories generated here will silently
diverge from the EACT reference printed by the legacy evaluator.

Usage:
    python -u scripts\\eact-dt-lora\\generate_eact_trajectories_parallel.py ^
        --country JPN --episodes 300 --horizon 24 --workers 4
"""

from __future__ import annotations

import argparse
import os
import sys

# --- 限制 BLAS 线程数：必须在 numpy/scipy 之前设置 -------------------------
# 多进程已经把核占满了，如果每个子进程里的 BLAS 再开多线程，会严重过订阅。
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")

import time
from concurrent.futures import ProcessPoolExecutor, as_completed

# This file lives in scripts/eact-dt-lora/, so the project root is two levels up.
_PROJECT_ROOT = os.path.abspath(
    os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..")
)
sys.path.insert(0, _PROJECT_ROOT)

from coastaldc_env import CoastalDCContinuousEnv  # noqa: E402
from coastaldc_env.reward import RewardWeights  # noqa: E402
from coastaldc_env.rollout import run_episode, save_trajectories  # noqa: E402
from coastaldc_env.swhp_cooling import CoolingConfig  # noqa: E402
from controllers.eact_mpc import EACTMPCController  # noqa: E402


# =============================================================================
# Legacy-compatible defaults
# =============================================================================
# These mirror `evaluate_lora_vs_eact_season.py` exactly so that the EACT
# reference in this generator is identical to the legacy evaluator's EACT.
# =============================================================================

# -- Reward weights (legacy evaluator defaults) ------------------------------
DEFAULT_WEIGHT_GRID = 1.0
DEFAULT_WEIGHT_CO2 = 2.0
DEFAULT_WEIGHT_TOTAL = 0.2
DEFAULT_WEIGHT_SMOOTH = 0.5

# -- Cooling config (legacy evaluator defaults) ------------------------------
DEFAULT_COOLING_CONDUCTANCE_MULTIPLIER = 1.0
DEFAULT_THERMAL_SAFETY_SHIELD = True

# -- Extra EACT-MPC kwargs (legacy evaluator defaults) -----------------------
DEFAULT_CONSTRAINT_TOLERANCE = 1e-4
DEFAULT_INTERVENTION_WEIGHT = 0.1
DEFAULT_ADAPTIVE_BETA_FLOOR = 0.10
DEFAULT_FORECAST_STRESS = "none"
DEFAULT_FORECAST_STRESS_SCALE = 0.0
DEFAULT_FORECAST_STRESS_START_STEP = 0
DEFAULT_FORECAST_STRESS_SEED = 20260717


def _build_reward_weights() -> RewardWeights:
    return RewardWeights(
        w_grid=DEFAULT_WEIGHT_GRID,
        w_co2=DEFAULT_WEIGHT_CO2,
        w_total=DEFAULT_WEIGHT_TOTAL,
        w_smooth=DEFAULT_WEIGHT_SMOOTH,
    )


def _build_cooling_config() -> CoolingConfig:
    nominal = CoolingConfig()
    return CoolingConfig(
        conductance_mw_per_k=(
            nominal.conductance_mw_per_k
            * DEFAULT_COOLING_CONDUCTANCE_MULTIPLIER
        ),
        enforce_thermal_safety=DEFAULT_THERMAL_SAFETY_SHIELD,
    )


def _build_env(country, seed, forecast_mode, forecast_dir):
    """Build env exactly the way the legacy evaluator does."""
    return CoastalDCContinuousEnv(
        country=country,
        seed=seed,
        forecast_mode=forecast_mode,
        forecast_dir=forecast_dir,
        reward_weights=_build_reward_weights(),
        cooling_config=_build_cooling_config(),
    )


def _build_eact_ctrl(env, forecast_dir, horizon, maxiter,
                     control_block_hours, gamma, confidence):
    """Build EACT-MPC exactly the way the legacy evaluator does."""
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


# ---- 每个子进程只建一次 env / ctrl ------------------------------------------
_WORKER: dict = {}


def _worker_init(country, seed, horizon, maxiter, control_block_hours,
                 gamma, confidence, forecast_dir, forecast_mode):
    env = _build_env(country, seed, forecast_mode, forecast_dir)
    ctrl = _build_eact_ctrl(
        env, forecast_dir, horizon, maxiter,
        control_block_hours, gamma, confidence,
    )
    _WORKER["env"] = env
    _WORKER["ctrl"] = ctrl


def _worker_run(ep: int):
    env = _WORKER["env"]
    ctrl = _WORKER["ctrl"]
    t0 = time.time()
    ctrl.reset()                       # 清空自适应残差窗口 / solver 历史
    traj = run_episode(env, ctrl, seed=2000 + ep)
    dt = time.time() - t0
    return ep, traj, dt


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--country", default="JPN")
    parser.add_argument("--episodes", type=int, default=250)
    parser.add_argument("--horizon", type=int, default=24)
    parser.add_argument(
        "--control-block-hours",
        type=int,
        default=6,
        help="Hold each MPC action for this many hours (EACT replans every step).",
    )
    parser.add_argument("--maxiter", type=int, default=20)
    parser.add_argument("--confidence", type=float, default=0.90)
    parser.add_argument(
        "--gamma", type=float, default=0.995,
        help="Discount factor used inside the MPC cost roll-out.",
    )
    parser.add_argument(
        "--out",
        default=None,
        help="Output .npz path (defaults to data/trajectories/eact_<COUNTRY>.npz).",
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--forecast-mode",
        choices=("oracle", "causal", "none"),
        default="causal",
        help="Forecast mode passed to the environment (legacy default: causal).",
    )
    parser.add_argument(
        "--forecast-dir",
        default=None,
        help=(
            "Directory containing causal_ridge_<COUNTRY>.npz and "
            "calibration_residuals_<COUNTRY>.npz."
        ),
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=max(1, (os.cpu_count() or 2) // 2),
        help="并行进程数。1 = 退回串行（方便调试）。默认 = CPU 核数的一半。",
    )
    parser.add_argument(
        "--start-episode",
        type=int,
        default=0,
        help="起始 episode 编号（用于跨机器分片，多机合并时保持一致 seed）。",
    )
    args = parser.parse_args()

    forecast_dir = args.forecast_dir or os.path.join(
        _PROJECT_ROOT,
        "results",
        "causal_forecasts_v3_gated_bias",
    )

    # Same preflight check as the legacy evaluator when causal forecasts are needed.
    if args.forecast_mode == "causal":
        required_files = [
            os.path.join(forecast_dir, f"causal_ridge_{args.country}.npz"),
            os.path.join(
                forecast_dir, f"calibration_residuals_{args.country}.npz"
            ),
        ]
        missing = [p for p in required_files if not os.path.isfile(p)]
        if missing:
            raise FileNotFoundError(
                "Causal forecasts required but missing:\n  "
                + "\n  ".join(missing)
            )

    episode_ids = list(
        range(args.start_episode, args.start_episode + args.episodes)
    )
    trajs = [None] * len(episode_ids)

    # ---- 串行路径（--workers 1）: 保留原始行为，方便调试复现 ----------------
    if args.workers <= 1:
        print(f"[serial] episodes={len(episode_ids)}", flush=True)
        env = _build_env(
            args.country, args.seed, args.forecast_mode, forecast_dir,
        )
        ctrl = _build_eact_ctrl(
            env, forecast_dir, args.horizon, args.maxiter,
            args.control_block_hours, args.gamma, args.confidence,
        )
        for i, ep in enumerate(episode_ids):
            t0 = time.time()
            ctrl.reset()
            traj = run_episode(env, ctrl, seed=2000 + ep)
            trajs[i] = traj
            dt = time.time() - t0
            print(
                f"[{i + 1}/{len(episode_ids)}] episode {ep}: "
                f"return={traj['episode_return']:.2f} dt={dt:.1f}s",
                flush=True,
            )

    # ---- 并行路径 ---------------------------------------------------------
    else:
        print(
            f"[parallel] workers={args.workers} "
            f"(cpu_count={os.cpu_count()}), episodes={len(episode_ids)}",
            flush=True,
        )
        t_all = time.time()
        with ProcessPoolExecutor(
            max_workers=args.workers,
            initializer=_worker_init,
            initargs=(
                args.country, args.seed, args.horizon, args.maxiter,
                args.control_block_hours, args.gamma, args.confidence,
                forecast_dir, args.forecast_mode,
            ),
        ) as ex:
            futures = {
                ex.submit(_worker_run, ep): i
                for i, ep in enumerate(episode_ids)
            }
            done = 0
            for fut in as_completed(futures):
                i = futures[fut]
                ep, traj, dt = fut.result()
                trajs[i] = traj
                done += 1
                elapsed = time.time() - t_all
                print(
                    f"[{done}/{len(episode_ids)}] episode {ep}: "
                    f"return={traj['episode_return']:.2f} "
                    f"dt={dt:.1f}s elapsed={elapsed:.0f}s",
                    flush=True,
                )

    out = args.out or os.path.join(
        _PROJECT_ROOT,
        "data",
        "trajectories",
        f"eact_{args.country}.npz",
    )
    os.makedirs(os.path.dirname(out), exist_ok=True)
    save_trajectories(out, trajs, source="eact")
    print(f"saved {len(trajs)} trajectories -> {out}", flush=True)


if __name__ == "__main__":
    main()