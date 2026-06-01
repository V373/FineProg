"""
Benchmark: sequential vs vectorized stochastic TCC.
Also compares against deterministic for small configs only.
Run with: conda run -n fineprog python scripts/bench_tcc.py
"""
import sys, time
sys.path.insert(0, '.')

import torch
from algos.loss.tcc.deterministic_alignment import compute_deterministic_alignment_loss
from algos.loss.tcc.stochastic_alignment import (
    compute_stochastic_alignment_loss,
    gen_cycles, _align, _align_vectorized,
)
from algos.loss.tcc.loss_head import regression_loss

LOSS_TYPE = 'regression_mse_var'


def bench(fn, reps=20):
    for _ in range(3): fn()
    t0 = time.perf_counter()
    for _ in range(reps): fn()
    return (time.perf_counter() - t0) / reps * 1000


def row(label, ms, ref=None):
    suffix = f"  ({ms/ref:.2f}x vs det)" if ref else ""
    print(f"  {label:<52s} {ms:8.2f} ms{suffix}")


# ── 1. Correctness smoke test ─────────────────────────────────────────
print("── Correctness smoke test ──────────────────────────────────────────")
torch.manual_seed(0)
B, T, D = 4, 8, 16
steps = torch.arange(T).unsqueeze(0).expand(B, -1).clone()
seq_lens = torch.full((B,), T, dtype=torch.long)
for loss_type in ['classification', 'regression_mse', 'regression_mse_var', 'regression_huber']:
    e = torch.randn(B, T, D, requires_grad=True)
    l = compute_stochastic_alignment_loss(e, steps, seq_lens, num_cycles=20,
                                          loss_type=loss_type)
    assert l.dim() == 0 and torch.isfinite(l), f"FAIL {loss_type}"
    l.backward()
    assert e.grad is not None
    print(f"  {loss_type:<30s}  loss={l.item():.4f}  grad_norm={e.grad.norm():.4f}  OK")
print()

# ── 2. Stochastic: sequential vs vectorized across num_cycles ─────────
B, T, D = 8, 20, 128
steps = torch.arange(T).unsqueeze(0).expand(B, -1).clone()
seq_lens = torch.full((B,), T, dtype=torch.long)

print("── Stochastic sequential vs vectorized  (B=8 T=20 D=128) ──────────")
print(f"  {'num_cycles':<12} {'sequential':>14} {'vectorized':>14} {'speedup':>10}")
print(f"  {'-'*12} {'-'*14} {'-'*14} {'-'*10}")
for nc in [10, 20, 40, 80, 160, 320]:
    cycles = gen_cycles(nc, B, cycle_length=2)

    def seq(n=nc, cy=cycles):
        e = torch.randn(B, T, D, requires_grad=True)
        cy2 = gen_cycles(n, B, cycle_length=2)
        lo, la = _align(cy2, e, T, n, 2, 'l2', 0.1)
        regression_loss(lo, la, T, steps[cy2[:,0]], seq_lens[cy2[:,0]],
                        LOSS_TYPE, True, 0.001, 0.1).backward()

    def vec(n=nc):
        e = torch.randn(B, T, D, requires_grad=True)
        compute_stochastic_alignment_loss(e, steps, seq_lens, num_cycles=n,
                                          loss_type=LOSS_TYPE).backward()

    ms_s = bench(seq, reps=15)
    ms_v = bench(vec, reps=15)
    print(f"  {nc:<12d} {ms_s:>13.2f}ms {ms_v:>13.2f}ms {ms_s/ms_v:>9.1f}x")
print()

# ── 3. Stochastic vs Deterministic (small config only) ───────────────
B, T, D = 8, 20, 128
nc_frac1 = B * T      # fraction=1.0 → 160
nc_25pct = B * T // 4  # fraction=0.25 → 40
steps = torch.arange(T).unsqueeze(0).expand(B, -1).clone()
seq_lens = torch.full((B,), T, dtype=torch.long)

print("── Stochastic vs Deterministic  (B=8 T=20 D=128) ──────────────────")

def det():
    e = torch.randn(B, T, D, requires_grad=True)
    compute_deterministic_alignment_loss(e, steps, seq_lens,
                                         loss_type=LOSS_TYPE).backward()

ms_det = bench(det, reps=15)
row(f"Deterministic  (pairs={B*(B-1)})", ms_det)

for nc in [nc_25pct, nc_frac1]:
    def vec(n=nc):
        e = torch.randn(B, T, D, requires_grad=True)
        compute_stochastic_alignment_loss(e, steps, seq_lens, num_cycles=n,
                                          loss_type=LOSS_TYPE).backward()
    ms_v = bench(vec, reps=15)
    row(f"Stochastic-vec (num_cycles={nc})", ms_v, ref=ms_det)
print()

# ── 4. gen_cycles: loop vs vectorized ────────────────────────────────
print("── gen_cycles: loop vs vectorized ──────────────────────────────────")
from algos.loss.tcc.stochastic_alignment import gen_cycles

# loop version (old)
def gen_cycles_loop(num_cycles, batch_size, cycle_length=2, device=None):
    import torch as _t
    cycles = []
    for _ in range(num_cycles):
        perm = _t.randperm(batch_size, device=device)[:cycle_length]
        cycles.append(_t.cat([perm, perm[:1]]))
    return _t.stack(cycles, dim=0)

for nc in [20, 80, 160, 320]:
    t0 = time.perf_counter()
    for _ in range(200): gen_cycles_loop(nc, 8)
    ms_loop = (time.perf_counter()-t0)/200*1000

    t0 = time.perf_counter()
    for _ in range(200): gen_cycles(nc, 8)
    ms_vec = (time.perf_counter()-t0)/200*1000

    print(f"  nc={nc:<4d}  loop={ms_loop:.4f}ms  vec={ms_vec:.4f}ms  speedup={ms_loop/ms_vec:.1f}x")

import sys
import time
import cProfile
import pstats
import io

sys.path.insert(0, '.')

import torch
from algos.loss.tcc.deterministic_alignment import compute_deterministic_alignment_loss
from algos.loss.tcc.stochastic_alignment import (
    compute_stochastic_alignment_loss,
    gen_cycles, _align, _align_vectorized,
)

REPEATS = 30
LOSS_TYPE = 'regression_mse_var'


def bench(fn, name, reps=REPEATS):
    for _ in range(3):
        fn()
    t0 = time.perf_counter()
    for _ in range(reps):
        fn()
    ms = (time.perf_counter() - t0) / reps * 1000
    print(f"  {name:60s}: {ms:8.2f} ms/iter")
    return ms


# ──────────────────────────────────────────────────────────────────────
# 1. Medium realistic config  B=8, T=20, D=128, num_cycles=160
# ──────────────────────────────────────────────────────────────────────
B, T, D = 8, 20, 128
nc = int(B * T * 1.0)  # 160

steps    = torch.arange(T).unsqueeze(0).expand(B, -1).clone()
seq_lens = torch.full((B,), T, dtype=torch.long)

print("=" * 75)
print(f"Config A: B={B} T={T} D={D}  num_cycles={nc} (fraction=1.0)")
print("=" * 75)

def run_det():
    e = torch.randn(B, T, D, requires_grad=True)
    compute_deterministic_alignment_loss(e, steps, seq_lens, loss_type=LOSS_TYPE).backward()

def run_sto_seq():
    e = torch.randn(B, T, D, requires_grad=True)
    cycles = gen_cycles(nc, B, cycle_length=2, device=None)
    logits, labels = _align(cycles, e, T, nc, 2, 'l2', 0.1)
    from algos.loss.tcc.loss_head import regression_loss
    start = cycles[:, 0]
    regression_loss(logits, labels, T, steps[start], seq_lens[start],
                    LOSS_TYPE, True, 0.001, 0.1).backward()

def run_sto_vec():
    e = torch.randn(B, T, D, requires_grad=True)
    compute_stochastic_alignment_loss(e, steps, seq_lens,
                                      num_cycles=nc, loss_type=LOSS_TYPE).backward()

ms_det     = bench(run_det,     f"Deterministic          (pairs={B*(B-1)}, work~{B*(B-1)*T*T} sims)")
ms_seq     = bench(run_sto_seq, f"Stochastic sequential  (Python loop × {nc})")
ms_vec     = bench(run_sto_vec, f"Stochastic vectorized  (Python loop × cycle_length=2)")

print(f"\n  Sequential vs Deterministic : {ms_seq/ms_det:.2f}x")
print(f"  Vectorized vs Deterministic : {ms_vec/ms_det:.2f}x")
print(f"  Vectorized vs Sequential    : {ms_seq/ms_vec:.1f}x speedup\n")

# ──────────────────────────────────────────────────────────────────────
# 2. Small config where sequential was potentially slower
# ──────────────────────────────────────────────────────────────────────
print("=" * 75)
print("Config B: varying B and T, num_cycles = B*T (fraction=1.0)")
print("=" * 75)
configs = [
    (2,  5,   8,   "tiny  "),
    (4,  8,  16,   "small "),
    (8,  20, 128,  "medium"),
    (16, 32, 256,  "large "),
]
for Bc, Tc, Dc, tag in configs:
    nc2 = Bc * Tc
    st = torch.arange(Tc).unsqueeze(0).expand(Bc, -1).clone()
    sl = torch.full((Bc,), Tc, dtype=torch.long)

    def _det(b=Bc, t=Tc, d=Dc, s=st, l=sl):
        e = torch.randn(b, t, d, requires_grad=True)
        compute_deterministic_alignment_loss(e, s, l, loss_type=LOSS_TYPE).backward()

    def _seq(b=Bc, t=Tc, d=Dc, n=nc2, s=st, l=sl):
        e = torch.randn(b, t, d, requires_grad=True)
        cy = gen_cycles(n, b, cycle_length=2)
        lo, la = _align(cy, e, t, n, 2, 'l2', 0.1)
        from algos.loss.tcc.loss_head import regression_loss
        regression_loss(lo, la, t, s[cy[:,0]], l[cy[:,0]],
                        LOSS_TYPE, True, 0.001, 0.1).backward()

    def _vec(b=Bc, t=Tc, d=Dc, n=nc2, s=st, l=sl):
        e = torch.randn(b, t, d, requires_grad=True)
        compute_stochastic_alignment_loss(e, s, l, num_cycles=n,
                                          loss_type=LOSS_TYPE).backward()

    md = bench(_det, f"[{tag}] B={Bc:2d} T={Tc:2d} D={Dc:3d} nc={nc2:4d}  det", reps=10)
    ms2 = bench(_seq, f"[{tag}] B={Bc:2d} T={Tc:2d} D={Dc:3d} nc={nc2:4d}  sto-seq", reps=10)
    mv  = bench(_vec, f"[{tag}] B={Bc:2d} T={Tc:2d} D={Dc:3d} nc={nc2:4d}  sto-vec", reps=10)
    print(f"          seq/det={ms2/md:.2f}x  vec/det={mv/md:.2f}x  seq/vec={ms2/mv:.1f}x speedup\n")

# ──────────────────────────────────────────────────────────────────────
# 3. Scaling with num_cycles (vectorized only)
# ──────────────────────────────────────────────────────────────────────
B, T, D = 8, 20, 128
steps    = torch.arange(T).unsqueeze(0).expand(B, -1).clone()
seq_lens = torch.full((B,), T, dtype=torch.long)

print("=" * 75)
print("Config C: vectorized scaling with num_cycles  (B=8 T=20 D=128)")
print("=" * 75)
for nc3 in [1, 5, 10, 20, 40, 80, 160, 320, 640]:
    def _v(n=nc3):
        e = torch.randn(B, T, D, requires_grad=True)
        compute_stochastic_alignment_loss(e, steps, seq_lens, num_cycles=n,
                                          loss_type=LOSS_TYPE).backward()
    bench(_v, f"vectorized num_cycles={nc3:4d}", reps=15)
