import numpy as np

PATH = r"C:\Users\TRY\PycharmProjects\coastaldc-eact-mpc-main\data\trajectories\eact_MYS.npz"

d = np.load(PATH, allow_pickle=True)
obs = d["observations"].astype(np.float64)
req = d["requested_actions"].astype(np.float64)
app = d["applied_actions"].astype(np.float64)
rwd = d["rewards"].astype(np.float64)

print("=" * 78)
print("1. Looking for t_inlet-like dim in obs (mean in [24, 31])")
print("=" * 78)
found = False
for i in range(obs.shape[-1]):
    x = obs[..., i]
    if 24.0 <= x.mean() <= 31.0:
        found = True
        print(
            f"  dim {i:>2}: mean={x.mean():.4f}  std={x.std():.4f}  "
            f"min={x.min():.4f}  max={x.max():.4f}  "
            f"p95={np.percentile(x, 95):.4f}"
        )
if not found:
    print("  (no dim looks like a temperature; obs min/max per dim below)")
    for i in range(obs.shape[-1]):
        x = obs[..., i]
        print(f"  dim {i:>2}: min={x.min():.4f} max={x.max():.4f}")

print()
print("=" * 78)
print("2. Requested vs applied correction per dim")
print("=" * 78)
for i, name in enumerate(["workload", "setpoint", "pump"]):
    x = np.abs(req[..., i] - app[..., i])
    print(f"[{name}]")
    print(
        f"  mean={x.mean():.6f}  median={np.median(x):.6f}  "
        f"p90={np.percentile(x, 90):.6f}  "
        f"p99={np.percentile(x, 99):.6f}  max={x.max():.6f}"
    )
    print(
        f"  frac > 1e-6: {(x > 1e-6).mean() * 100:.2f}%   "
        f"> 1e-3: {(x > 1e-3).mean() * 100:.2f}%   "
        f"> 1e-2: {(x > 1e-2).mean() * 100:.2f}%"
    )

print()
print("=" * 78)
print("3. Thermal pool size (max of setpoint/pump correction)")
print("=" * 78)
d_cool = np.maximum(
    np.abs(req[..., 1] - app[..., 1]),
    np.abs(req[..., 2] - app[..., 2]),
)
for eps in [1e-6, 1e-5, 1e-4, 1e-3, 1e-2]:
    n = int((d_cool > eps).sum())
    print(f"  eps={eps:.0e}: {n:>8d} / {d_cool.size} = {100.0 * n / d_cool.size:.3f}%")

print()
print("=" * 78)
print("4. Return stats")
print("=" * 78)
returns = rwd.sum(axis=1)
print(
    f"  mean={returns.mean():.2f}  std={returns.std():.2f}  "
    f"min={returns.min():.2f}  "
    f"P75={np.percentile(returns, 75):.2f}  "
    f"max={returns.max():.2f}"
)