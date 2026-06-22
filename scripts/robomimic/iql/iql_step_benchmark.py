"""
iql_step_benchmark.py
─────────────────────
Per-step timing benchmark for the IQL training pipeline used by

    python robomimic/scripts/train.py \\
        --config robomimic/exps/templates/iql.json \\
        --dataset datasets/can/mh/image_2view_v15_reward_labeled_original.hdf5

The script reuses the exact same config, dataset, model, and DataLoader
parameters that train.py uses for one gradient step.  It then decomposes
a single IQL train_on_batch into a fine-grained timing breakdown:

  • Data_Loading   – fetching one batch from the DataLoader
  • Process_Batch  – dtype conversion + .to(device)
  • Q_Forward      – online critic forward (2 Q networks)
  • V_Next_Forward – target V forward on next_obs (for Q target)
  • V_Forward      – online V forward
  • Q_Target_Forward – target-critic forward (used for V loss)
  • Loss_Compute   – MSE + expectile + AWR (CPU side)
  • Q_Backward     – backprop + soft Polyak update for both Q
  • V_Backward     – backprop on V
  • Actor_Forward  – GaussianActorNetwork forward_train
  • Actor_Backward – backprop on actor
  • Log_Info       – tensor board log dict assembly

Output: a printed table (mean / p50 / p95 / min / max per stage) plus
a single-line throughput estimate (samples/s).

Usage (in conda env `fineprog`):

    conda run -n fineprog python \\
        /home/user/zhangzk/projects/fineprog/scripts/robomimic/iql_step_benchmark.py

Optional env vars:
    N_WARMUP   – warmup steps before timing (default 10)
    N_TIMED    – timed steps (default 50)
"""

import os
import sys
import json
import time
import math
import statistics
from collections import OrderedDict, defaultdict

# ─── Make robomimic importable from the local checkout ────────────────────────
ROBOMIMIC_ROOT = "/home/user/zhangzk/projects/fineprog/third_party/robomimic"
# The user's command runs from inside ROBOMIMIC_ROOT and uses a *relative*
# dataset path – we resolve it relative to the same cwd.
DATASET_REL    = "datasets/can/mh/image_2view_v15_reward_labeled_original.hdf5"
DATASET_ABS    = os.path.join(ROBOMIMIC_ROOT, DATASET_REL)
CONFIG_PATH    = os.path.join(
    ROBOMIMIC_ROOT, "robomimic/exps/templates/iql.json"
)

sys.path.insert(0, ROBOMIMIC_ROOT)

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

import robomimic
import robomimic.utils.obs_utils as ObsUtils
import robomimic.utils.file_utils as FileUtils
import robomimic.utils.train_utils as TrainUtils
import robomimic.utils.torch_utils as TorchUtils
from robomimic.config import config_factory
from robomimic.algo import algo_factory


# ─── config ──────────────────────────────────────────────────────────────────
N_WARMUP = int(os.environ.get("N_WARMUP", "10"))
N_TIMED  = int(os.environ.get("N_TIMED",  "50"))


# ─── helpers ─────────────────────────────────────────────────────────────────
class Timer:
    """CPU+GPU accurate timer using torch.cuda.Event when CUDA is available."""
    def __init__(self, use_cuda_events: bool):
        self.use_cuda_events = use_cuda_events and torch.cuda.is_available()
        self.cpu_t0 = 0.0
        self.cuda_t0 = None
        self.cuda_t1 = None

    def __enter__(self):
        if self.use_cuda_events:
            self.cuda_t0 = torch.cuda.Event(enable_timing=True)
            self.cuda_t1 = torch.cuda.Event(enable_timing=True)
            self.cuda_t0.record()
        self.cpu_t0 = time.perf_counter()
        return self

    def __exit__(self, *args):
        self.cpu_dt = time.perf_counter() - self.cpu_t0
        if self.use_cuda_events:
            self.cuda_t1.record()
            torch.cuda.synchronize()
            self.cuda_dt = self.cuda_t0.elapsed_time(self.cuda_t1) / 1000.0
        else:
            self.cuda_dt = None

    def dt(self) -> float:
        # Use the larger of CPU / GPU time to capture the real "wall-clock"
        # contribution of this stage (whichever is the bottleneck).
        if self.cuda_dt is not None:
            return max(self.cpu_dt, self.cuda_dt)
        return self.cpu_dt


def percentile(xs, p):
    if not xs:
        return float("nan")
    xs = sorted(xs)
    k = (len(xs) - 1) * p / 100.0
    f, c = math.floor(k), math.ceil(k)
    if f == c:
        return xs[int(k)]
    return xs[f] + (xs[c] - xs[f]) * (k - f)


def fmt_sec(s: float) -> str:
    if s >= 1.0:
        return f"{s:7.3f} s "
    if s >= 1e-3:
        return f"{s*1e3:7.3f} ms"
    return f"{s*1e6:7.3f} us"


# ─── 1. Build config, dataset, model exactly like train.py ───────────────────
def build():
    print("=" * 80)
    print("Loading config + dataset + model (mirrors train.py)...")
    print("=" * 80)

    # Load default config for the algo, then overlay the iql.json template
    config = config_factory("iql")
    with open(CONFIG_PATH, "r") as f:
        ext_cfg = json.load(f)
    # The default config_factory already merges a base.  train.py applies the
    # json via config.update(ext_cfg) inside a values_unlocked() block:
    with config.values_unlocked():
        config.update(ext_cfg)
        # Provide the dataset path the way --dataset on the CLI does
        config.train.data = [{"path": DATASET_ABS}]
        # Keep config small: we only time training, no rollout
        config.experiment.rollout.enabled = False
        # Make sure caching is what the production command uses
        config.train.hdf5_cache_mode = "all"
        config.train.hdf5_use_swmr   = True
        config.train.num_data_workers = 0   # iql.json default → keep

    # Initialise obs utils the way train.py does
    ObsUtils.initialize_obs_utils_with_config(config)

    # Read shape metadata from the dataset
    env_meta = FileUtils.get_env_metadata_from_dataset(
        dataset_path=DATASET_ABS
    )
    shape_meta = FileUtils.get_shape_metadata_from_dataset(
        dataset_config={"path": DATASET_ABS},
        action_keys=config.train.action_keys,
        all_obs_keys=config.all_obs_keys,
        verbose=True,
    )

    # Build train set + sampler + dataloader
    trainset, _ = TrainUtils.load_data_for_training(
        config, obs_keys=shape_meta["all_obs_keys"]
    )
    train_sampler = trainset.get_dataset_sampler()
    train_loader = DataLoader(
        dataset=trainset,
        sampler=train_sampler,
        batch_size=config.train.batch_size,
        shuffle=(train_sampler is None),
        num_workers=config.train.num_data_workers,
        drop_last=True,
    )
    print(f"  dataset = {DATASET_ABS}")
    print(f"  filter  = IQL_expert   # {len(trainset)} transitions")
    print(f"  batch   = {config.train.batch_size}   num_steps/epoch = {config.experiment.epoch_every_n_steps}")
    print(f"  num_data_workers = {config.train.num_data_workers}   hdf5_cache_mode = {config.train.hdf5_cache_mode}")
    print(f"  device  = {'cuda' if torch.cuda.is_available() else 'cpu'}")

    # Build IQL model
    device = TorchUtils.get_torch_device(try_to_use_cuda=config.train.cuda)
    model  = algo_factory(
        algo_name="iql",
        config=config,
        obs_key_shapes=shape_meta["all_shapes"],
        ac_dim=shape_meta["ac_dim"],
        device=device,
    )
    model.set_train()
    model.on_epoch_end(0)  # initialise LR schedulers etc.
    print(f"  model   = IQL  (actor, vf, critic×2, critic_target×2)")
    n_params = sum(p.numel() for p in model.nets.parameters())
    print(f"  total params = {n_params/1e6:.2f} M")
    print("=" * 80)

    return config, model, train_loader, device


# ─── 2. Hand-decomposed IQL train_on_batch (mirrors iql.py) ──────────────────
def time_one_step(model, data_loader, use_cuda_events: bool, stage_buf):
    """
    Run one IQL gradient step and record per-stage wall time.
    Mirrors IQL.train_on_batch / _compute_critic_loss / _compute_actor_loss /
    _update_critic / _update_actor exactly, but with explicit timing around
    every sub-stage.
    """
    stages = {}

    # ── (a) Data loading ────────────────────────────────────────────────
    with Timer(use_cuda_events) as t:
        batch = next(data_loader)
    stages["Data_Loading"] = t.dt()

    # ── (b) Process batch for training (CPU→GPU + dtype) ────────────────
    with Timer(use_cuda_events) as t:
        input_batch = model.process_batch_for_training(batch)
        input_batch = model.postprocess_batch_for_training(
            input_batch, obs_normalization_stats=None
        )
    stages["Process_Batch"] = t.dt()

    obs          = input_batch["obs"]
    next_obs     = input_batch["next_obs"]
    goal_obs     = input_batch.get("goal_obs", None)
    actions      = input_batch["actions"]
    rewards      = torch.unsqueeze(input_batch["rewards"], 1)
    dones        = torch.unsqueeze(input_batch["dones"],   1)
    discount     = model.algo_config.discount
    vf_quantile  = model.algo_config.vf_quantile
    target_tau   = model.algo_config.target_tau
    beta         = model.algo_config.adv.beta

    # ── (c) Online Q forward (2 critics) ─────────────────────────────────
    with Timer(use_cuda_events) as t:
        pred_qs = [critic(obs_dict=obs, acts=actions, goal_dict=goal_obs)
                   for critic in model.nets["critic"]]
    stages["Q_Forward"] = t.dt()

    # ── (d) Target V forward on next_obs (for Q target) ──────────────────
    with Timer(use_cuda_events) as t:
        target_vf_pred = model.nets["vf"](
            obs_dict=next_obs, goal_dict=goal_obs
        ).detach()
        q_target = rewards + (1.0 - dones) * discount * target_vf_pred
        q_target = q_target.detach()
    stages["V_Next_Forward"] = t.dt()

    # ── (e) Q losses on CPU side ────────────────────────────────────────
    with Timer(use_cuda_events) as t:
        td_loss_fcn = nn.MSELoss()
        critic_losses = [td_loss_fcn(q, q_target) for q in pred_qs]
    stages["Q_Loss_Compute"] = t.dt()

    # ── (f) Q target forward + online V forward (for V expectile loss) ──
    with Timer(use_cuda_events) as t:
        with torch.no_grad():
            target_qs = [critic(obs_dict=obs, acts=actions, goal_dict=goal_obs)
                         for critic in model.nets["critic_target"]]
        q_pred, _ = torch.cat(target_qs, dim=1).min(dim=1, keepdim=True)
        q_pred = q_pred.detach()
        vf_pred = model.nets["vf"](obs_dict=obs, goal_dict=goal_obs)
    stages["Q_Target_V_Forward"] = t.dt()

    # ── (g) V expectile loss computation ────────────────────────────────
    with Timer(use_cuda_events) as t:
        vf_err    = vf_pred - q_pred
        vf_sign   = (vf_err > 0).float()
        vf_weight = (1 - vf_sign) * vf_quantile + vf_sign * (1 - vf_quantile)
        vf_loss   = (vf_weight * (vf_err ** 2)).mean()
    stages["V_Loss_Compute"] = t.dt()

    # ── (h) Actor forward (Gaussian forward_train) ──────────────────────
    with Timer(use_cuda_events) as t:
        dist     = model.nets["actor"].forward_train(
            obs_dict=obs, goal_dict=goal_obs
        )
        log_prob = dist.log_prob(actions)
    stages["Actor_Forward"] = t.dt()

    # ── (i) Actor AWR loss computation ──────────────────────────────────
    with Timer(use_cuda_events) as t:
        adv     = q_pred - vf_pred
        weights = torch.exp(adv / beta).clamp(-100.0, 100.0)
        actor_loss = (-log_prob * weights.detach()).mean()
    stages["Actor_Loss_Compute"] = t.dt()

    # ── (j) Q backward (2 critics, each with its own optimizer) ─────────
    with Timer(use_cuda_events) as t:
        for (critic_loss, critic, critic_target, optimizer) in zip(
            critic_losses, model.nets["critic"], model.nets["critic_target"],
            model.optimizers["critic"]
        ):
            TorchUtils.backprop_for_loss(
                net=critic, optim=optimizer, loss=critic_loss,
                max_grad_norm=model.algo_config.critic.max_gradient_norm,
                retain_graph=False,
            )
            with torch.no_grad():
                TorchUtils.soft_update(
                    source=critic, target=critic_target, tau=target_tau
                )
    stages["Q_Backward_SoftUpdate"] = t.dt()

    # ── (k) V backward ──────────────────────────────────────────────────
    with Timer(use_cuda_events) as t:
        TorchUtils.backprop_for_loss(
            net=model.nets["vf"], optim=model.optimizers["vf"],
            loss=vf_loss,
            max_grad_norm=model.algo_config.critic.max_gradient_norm,
            retain_graph=False,
        )
    stages["V_Backward"] = t.dt()

    # ── (l) Actor backward ──────────────────────────────────────────────
    with Timer(use_cuda_events) as t:
        TorchUtils.backprop_for_loss(
            net=model.nets["actor"], optim=model.optimizers["actor"],
            loss=actor_loss,
            max_grad_norm=model.algo_config.actor.max_gradient_norm,
        )
    stages["Actor_Backward"] = t.dt()

    # ── (m) log_info (bookkeeping for tensorboard) ──────────────────────
    with Timer(use_cuda_events) as t:
        info = OrderedDict()
        info["actor/log_prob"]        = log_prob.mean()
        info["actor/loss"]            = actor_loss
        info["critic/critic1_pred"]   = pred_qs[0].mean()
        info["critic/critic1_loss"]   = critic_losses[0]
        info["vf/v_loss"]             = vf_loss
        info["vf/q_pred"]             = q_pred
        info["vf/v_pred"]             = vf_pred
        info["adv/adv"]               = adv
        info["adv/adv_weight"]        = weights
        _ = model.log_info(info)
    stages["Log_Info"] = t.dt()

    for k, v in stages.items():
        stage_buf[k].append(v)
    return stages


# ─── 3. main ─────────────────────────────────────────────────────────────────
def main():
    config, model, train_loader, device = build()
    use_cuda_events = (device.type == "cuda")
    data_iter = iter(train_loader)

    # Warmup
    print(f"\nWarmup: {N_WARMUP} steps …")
    for _ in range(N_WARMUP):
        time_one_step(model, data_iter, use_cuda_events, defaultdict(list))
    if device.type == "cuda":
        torch.cuda.synchronize()

    # Timed
    print(f"Timing:  {N_TIMED} steps …\n")
    stage_buf = defaultdict(list)
    total_dt = []
    for _ in range(N_TIMED):
        t0 = time.perf_counter()
        time_one_step(model, data_iter, use_cuda_events, stage_buf)
        if device.type == "cuda":
            torch.cuda.synchronize()
        total_dt.append(time.perf_counter() - t0)

    # ── Report ──────────────────────────────────────────────────────────
    print("=" * 80)
    print(" Per-stage timing (mean / p50 / p95 / min / max)")
    print("=" * 80)
    fmt = "{:<22}  {:>10}  {:>10}  {:>10}  {:>10}  {:>10}"
    print(fmt.format("stage", "mean", "p50", "p95", "min", "max"))
    print("-" * 80)
    stage_order = [
        "Data_Loading",
        "Process_Batch",
        "Q_Forward",
        "V_Next_Forward",
        "Q_Loss_Compute",
        "Q_Target_V_Forward",
        "V_Loss_Compute",
        "Actor_Forward",
        "Actor_Loss_Compute",
        "Q_Backward_SoftUpdate",
        "V_Backward",
        "Actor_Backward",
        "Log_Info",
    ]
    sum_mean = 0.0
    for s in stage_order:
        xs = stage_buf[s]
        if not xs:
            continue
        m = statistics.mean(xs)
        sum_mean += m
        print(fmt.format(
            s,
            fmt_sec(m).strip(),
            fmt_sec(percentile(xs, 50)).strip(),
            fmt_sec(percentile(xs, 95)).strip(),
            fmt_sec(min(xs)).strip(),
            fmt_sec(max(xs)).strip(),
        ))
    print("-" * 80)
    print(fmt.format("Σ stages", fmt_sec(sum_mean).strip(),
                      "", "", "", ""))
    print(fmt.format("wall (incl. sync)", fmt_sec(statistics.mean(total_dt)).strip(),
                      fmt_sec(percentile(total_dt, 50)).strip(),
                      fmt_sec(percentile(total_dt, 95)).strip(),
                      fmt_sec(min(total_dt)).strip(),
                      fmt_sec(max(total_dt)).strip()))
    print("=" * 80)

    # ── Throughput & bottleneck ─────────────────────────────────────────
    bs   = config.train.batch_size
    mean = statistics.mean(total_dt)
    sps  = bs / mean
    eps  = 1.0 / mean
    print(f" throughput = {sps:7.1f} samples/s  ({eps:5.2f} steps/s,  batch={bs})")
    print()
    print(" Bottleneck ranking (descending mean wall time):")
    ranked = sorted(stage_order, key=lambda s: statistics.mean(stage_buf[s]), reverse=True)
    for s in ranked:
        m = statistics.mean(stage_buf[s])
        print(f"   {m*100/mean:5.1f}%   {fmt_sec(m).strip()}   {s}")
    print("=" * 80)

    # ── GPU memory snapshot (best effort) ────────────────────────────────
    if device.type == "cuda":
        alloc = torch.cuda.memory_allocated(device) / 1024**2
        rsrv  = torch.cuda.memory_reserved(device)  / 1024**2
        peak  = torch.cuda.max_memory_allocated(device) / 1024**2
        print(f" GPU mem: allocated={alloc:.0f} MiB  reserved={rsrv:.0f} MiB  peak={peak:.0f} MiB")


if __name__ == "__main__":
    main()
