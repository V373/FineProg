"""
bench_triplet_pairwise.py
─────────────────────────
Compare OLD per-triplet-gather distance vs NEW pairwise-matrix distance
inside TemporalTripletLoss, for fractions 0.1 and 0.5.

Both approaches are benchmarked on the same synthetic clip (T=20, D=128),
with triplet indices pre-computed by _enumerate_valid_triplets so that
only the distance-computation step is timed.

Run:
    conda run -n fineprog python scripts/bench_triplet_pairwise.py
"""

import sys
import time
from pathlib import Path

import torch

# ── project root on path ───────────────────────────────────────────────────
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from algos.loss.contrastive.loss_temporal_triplet import TemporalTripletLoss

# ── config ─────────────────────────────────────────────────────────────────
T         = 20
D         = 128
FRACTIONS = [0.1, 0.5]
WARMUP    = 20
REPS      = 100
DEVICE    = torch.device("cuda" if torch.cuda.is_available() else "cpu")
DTYPE     = torch.float32

# ── helper: old distance path ───────────────────────────────────────────────
def dist_old(z, a_idx, p_idx, n_idx):
    """Original per-triplet gather + element-wise subtraction."""
    z_a = z[a_idx]
    z_p = z[p_idx]
    z_n = z[n_idx]
    d_ap = ((z_a - z_p) ** 2).sum(dim=-1)
    d_an = ((z_a - z_n) ** 2).sum(dim=-1)
    return d_ap, d_an


# ── helper: new distance path ────────────────────────────────────────────────
def dist_new(z, a_idx, p_idx, n_idx):
    """Full [T,T] pairwise matrix + index lookup."""
    pdist2 = TemporalTripletLoss._squared_l2_dist_full(z)
    d_ap   = pdist2[a_idx, p_idx]
    d_an   = pdist2[a_idx, n_idx]
    return d_ap, d_an


# ── timing helper ────────────────────────────────────────────────────────────
def bench_fn(fn, args, warmup, reps, device):
    """Return mean elapsed time in ms over `reps` iterations."""
    if device.type == "cuda":
        torch.cuda.synchronize()
    # warmup
    for _ in range(warmup):
        _ = fn(*args)
    if device.type == "cuda":
        torch.cuda.synchronize()

    t0 = time.perf_counter()
    for _ in range(reps):
        _ = fn(*args)
    if device.type == "cuda":
        torch.cuda.synchronize()
    t1 = time.perf_counter()
    return (t1 - t0) / reps * 1e3


# ── main ─────────────────────────────────────────────────────────────────────
def main():
    print(f"device={DEVICE}  T={T}  D={D}  warmup={WARMUP}  reps={REPS}")
    print()

    torch.manual_seed(0)
    z = torch.randn(T, D, dtype=DTYPE, device=DEVICE)

    # Build integer time-steps (consecutive, as in a real clip)
    s = torch.arange(T, device=DEVICE)

    # Enumerate all valid triplets once
    result = TemporalTripletLoss._enumerate_valid_triplets(s)
    assert result is not None, "No valid triplets found for T=20 consecutive steps"
    a_all, p_all, n_all = result
    N_valid = a_all.shape[0]
    print(f"N_valid triplets = {N_valid}")

    # Verify numerical equivalence before benchmarking
    d_ap_old, d_an_old = dist_old(z, a_all, p_all, n_all)
    d_ap_new, d_an_new = dist_new(z, a_all, p_all, n_all)
    err_ap = float((d_ap_old - d_ap_new).abs().max())
    err_an = float((d_an_old - d_an_new).abs().max())
    assert err_ap < 1e-4 and err_an < 1e-4, \
        f"Numerical mismatch!  max_err_ap={err_ap:.2e}  max_err_an={err_an:.2e}"
    print(f"Numerical check PASSED  (max_err_ap={err_ap:.2e}  max_err_an={err_an:.2e})\n")

    header = f"{'fraction':>10}  {'K sampled':>10}  {'OLD (ms)':>10}  {'NEW (ms)':>10}  {'speedup':>8}"
    print(header)
    print("-" * len(header))

    for frac in FRACTIONS:
        K = max(1, round(N_valid * frac))
        K = min(K, N_valid)
        perm  = torch.randperm(N_valid, device=DEVICE)
        sel   = perm[:K]
        a_idx = a_all[sel]
        p_idx = p_all[sel]
        n_idx = n_all[sel]

        args = (z, a_idx, p_idx, n_idx)
        t_old = bench_fn(dist_old, args, WARMUP, REPS, DEVICE)
        t_new = bench_fn(dist_new, args, WARMUP, REPS, DEVICE)
        speedup = t_old / t_new if t_new > 0 else float("inf")
        print(f"{frac:>10.1f}  {K:>10d}  {t_old:>10.3f}  {t_new:>10.3f}  {speedup:>8.2f}x")

    print()
    print("Done.")


if __name__ == "__main__":
    main()
