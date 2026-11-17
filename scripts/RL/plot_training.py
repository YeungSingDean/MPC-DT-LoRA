"""plot_training.py — per-episode training curves from training_log.csv.

Usage:
    python plot_training.py --log results\\JPN_v3_sac_lagrange_season\\training_log.csv
    python plot_training.py --log <sac_log> --compare <lora_log>
"""

import argparse
import os

import matplotlib.pyplot as plt
import pandas as pd


def smooth(x, k=5):
    return pd.Series(x).rolling(k, min_periods=1).mean().to_numpy()


def plot_single(log_path, out_path):
    df = pd.read_csv(log_path)

    fig, axes = plt.subplots(2, 2, figsize=(13, 8))

    # (1) env return vs shaped return
    ax = axes[0, 0]
    env_col = "env_return" if "env_return" in df.columns else "return"
    ax.plot(df["episode"], df[env_col], alpha=0.25, color="C0")
    ax.plot(df["episode"], smooth(df[env_col]), color="C0",
            label=f"{env_col} (5-ep MA)")
    if "shaped_return" in df.columns:
        ax.plot(df["episode"], df["shaped_return"], alpha=0.25, color="C1")
        ax.plot(df["episode"], smooth(df["shaped_return"]), color="C1",
                label="shaped_return (5-ep MA)")
    ax.set_xlabel("episode")
    ax.set_ylabel("return per episode")
    ax.set_title("Env return vs shaped return")
    ax.legend()
    ax.grid(alpha=0.3)

    # (2) thermal P95 + event hours
    ax = axes[0, 1]
    ax.plot(df["episode"], df["p95_t_inlet"], color="C3", alpha=0.6)
    ax.plot(df["episode"], smooth(df["p95_t_inlet"]), color="C3",
            label="P95 T_inlet (5-ep MA)")
    ax.axhline(27.0, ls="--", color="k", lw=0.8, label="27 C reference")
    ax.set_xlabel("episode")
    ax.set_ylabel("P95 [C]")
    ax2 = ax.twinx()
    ax2.plot(df["episode"], df["thermal_event_hours"], color="C4", alpha=0.35)
    ax2.plot(df["episode"], smooth(df["thermal_event_hours"]), color="C4",
             label="event hours (5-ep MA)")
    ax2.set_ylabel("event hours / episode")
    ax.set_title("Thermal metrics (training)")
    ax.grid(alpha=0.3)
    lines, labels = ax.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax.legend(lines + lines2, labels + labels2, loc="best")

    # (3) Lagrange multipliers
    ax = axes[1, 0]
    if "exposure_lambda" in df.columns:
        ax.plot(df["episode"], df["exposure_lambda"],
                color="C2", label="lambda_exposure")
    if "event_lambda" in df.columns:
        ax.plot(df["episode"], df["event_lambda"],
                color="C5", label="lambda_event")
    ax.set_xlabel("episode")
    ax.set_ylabel("lambda")
    ax.set_title("Lagrange multipliers")
    ax.legend()
    ax.grid(alpha=0.3)

    # (4) alpha + log_prob
    ax = axes[1, 1]
    if "alpha" in df.columns:
        ax.plot(df["episode"], df["alpha"], color="C6", label="alpha")
    ax.set_xlabel("episode")
    ax.set_ylabel("alpha")
    ax.set_title("Entropy temperature")
    ax.legend()
    ax.grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    print(f"saved: {out_path}")


def plot_compare(sac_log, lora_log, out_path):
    """Overlay SAC vs LoRA-SAC on the same axes."""
    df_sac = pd.read_csv(sac_log)
    df_lora = pd.read_csv(lora_log)

    fig, axes = plt.subplots(1, 3, figsize=(16, 4.5))

    # env return
    ax = axes[0]
    s = "env_return" if "env_return" in df_sac.columns else "return"
    ax.plot(df_sac["episode"], smooth(df_sac[s]), label="SAC (MLP)")
    ax.plot(df_lora["episode"], smooth(df_lora["return"]), label="LoRA-SAC (V6)")
    ax.set_xlabel("episode"); ax.set_ylabel("env return (5-ep MA)")
    ax.set_title("Return"); ax.legend(); ax.grid(alpha=0.3)

    # shaped return
    ax = axes[1]
    if "shaped_return" in df_sac.columns:
        ax.plot(df_sac["episode"], smooth(df_sac["shaped_return"]),
                label="SAC shaped (5-ep MA)")
    # V6 的 shaped_return 需要按同样公式重算（V6 没有直接存这个列）
    lora_shaped = (df_lora["return"]
                   - 0.25 * df_lora["episode_cold_cost"]
                   - 0.20 * df_lora["episode_mechanical_cooling_cost"]
                   - 0.10 * df_lora["episode_backlog_cost"])
    ax.plot(df_lora["episode"], smooth(lora_shaped), label="V6 shaped (5-ep MA)")
    ax.set_xlabel("episode"); ax.set_ylabel("shaped return (5-ep MA)")
    ax.set_title("Shaped return"); ax.legend(); ax.grid(alpha=0.3)

    # P95
    ax = axes[2]
    ax.plot(df_sac["episode"], smooth(df_sac["p95_t_inlet"]), label="SAC (MLP)")
    ax.plot(df_lora["episode"], smooth(df_lora["p95_t_inlet"]), label="LoRA-SAC (V6)")
    ax.axhline(27.0, ls="--", color="k", lw=0.8)
    ax.set_xlabel("episode"); ax.set_ylabel("P95 T_inlet [C] (5-ep MA)")
    ax.set_title("Thermal P95"); ax.legend(); ax.grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    print(f"saved: {out_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--log", required=True, help="SAC training_log.csv")
    parser.add_argument("--compare", default=None,
                        help="Optional LoRA-SAC training_log.csv to overlay.")
    parser.add_argument("--out", default=None,
                        help="Output PNG path (default: alongside log).")
    args = parser.parse_args()

    log_dir = os.path.dirname(os.path.abspath(args.log))

    out_single = args.out or os.path.join(log_dir, "training_curves.png")
    plot_single(args.log, out_single)

    if args.compare:
        out_cmp = os.path.join(log_dir, "compare_sac_vs_lora.png")
        plot_compare(args.log, args.compare, out_cmp)


if __name__ == "__main__":
    main()