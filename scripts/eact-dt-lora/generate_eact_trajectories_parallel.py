"""Generate EACT-MPC expert trajectories for offline training (multiprocess).

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
from coastaldc_env.rollout import run_episode, save_trajectories  # noqa: E402
from controllers.eact_mpc import EACTMPCController  # noqa: E402


# ---- 每个子进程只建一次 env / ctrl ------------------------------------------
_WORKER: dict = {}


def _worker_init(country, seed, horizon, maxiter, control_block_hours,
                 gamma, confidence, forecast_dir):
    env = CoastalDCContinuousEnv(country=country, seed=seed)
    ctrl = EACTMPCController(
        env,
        forecast_dir=forecast_dir,
        horizon=horizon,
        maxiter=maxiter,
        control_block_hours=control_block_hours,
        gamma=gamma,
        confidence=confidence,
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

    episode_ids = list(
        range(args.start_episode, args.start_episode + args.episodes)
    )
    trajs = [None] * len(episode_ids)

    # ---- 串行路径（--workers 1）: 保留原始行为，方便调试复现 ----------------
    if args.workers <= 1:
        print(f"[serial] episodes={len(episode_ids)}", flush=True)
        env = CoastalDCContinuousEnv(country=args.country, seed=args.seed)
        ctrl = EACTMPCController(
            env,
            forecast_dir=forecast_dir,
            horizon=args.horizon,
            maxiter=args.maxiter,
            control_block_hours=args.control_block_hours,
            gamma=args.gamma,
            confidence=args.confidence,
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
                forecast_dir,
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