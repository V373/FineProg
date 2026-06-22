# policy_training 项目架构说明

本文档面向需要快速理解 `/home/user/zhangzk/projects/fineprog/policy_training` 的上游 LLM。内容聚焦源码结构、训练/评估数据流、配置契约和扩展边界；`checkpoint/`、`outputs/`、`wandb/`、`__pycache__/`、`.pytest_cache/`、HDF5 数据和视频文件视为运行产物或数据资产，不作为核心源码展开。

## 1. 项目定位

`policy_training` 是 `fineprog` 下的离线策略训练子项目。它以 robomimic 风格 HDF5 数据集为输入，使用 SB3 风格的自定义 IQL（Implicit Q-Learning）实现训练连续控制策略，并支持：

- 单次离线训练：`train_policy.py`
- 批量 IQL 训练启动器：`iql_scale_train_sb3.py`
- 手动 checkpoint 评估：`evaluate_policy.py`
- 训练过程中周期性 robosuite rollout 评估
- 自描述 v1 checkpoint：保存训练配置、环境元数据、观测形状、obs flatten 切片、权重和优化器

当前源码中真正接入算法工厂的是 IQL；`algos/online_rl/` 和 `datasets/metaworld/` 基本是占位目录。

## 2. 顶层目录结构

```text
policy_training/
  train_policy.py                 # 单次离线训练入口
  evaluate_policy.py              # 手动评估入口
  iql_scale_train_sb3.py           # 批量训练 launcher
  configs/
    iql.yaml                       # 默认训练配置
    evaluate_policy.yaml           # 默认评估配置
  algos/
    __init__.py                    # build_algo 工厂，当前仅支持 iql
    offline_rl/
      base_offline_rl.py           # 离线 RL 抽象基类和通用 learn_offline 循环
      iql.py                       # IQL 算法实现
    online_rl/                     # 占位
  models/
    feature_extractors.py          # flat_range 与 resnet18conv 特征提取器
    model_utils.py                 # FrozenPretrainedResNet18Conv、SpatialSoftmax
    policies.py                    # SB3 SAC 派生 actor / critic / policy
    value_critic.py                # IQL 的 V(s) 网络
  datasets/
    robomimic/
      replay_buffer.py             # robomimic HDF5 replay buffer
    metaworld/                     # 占位
  envs/
    __init__.py                    # 将 vendored robomimic runtime 加到 sys.path
    robomimic.py                   # robomimic / robosuite env adapter
    robomimic_runtime/             # vendored robomimic runtime，评估和 ResNet18Conv 依赖它
  utils/
    config.py                      # YAML config 加载、默认值合并、路径解析
    logger.py                      # CLI、seed、device、run naming、W&B logger
    train_utils.py                 # checkpoint metadata 构造
    checkpoints.py                 # checkpoint v1 schema 校验
    eval_utils.py                  # checkpoint 加载、obs adapter、rollout、评估输出
  tests/
    test_eval_output_paths.py      # eval 输出路径和 CLI override 单测
    test_in_training_rollout_eval.py # 训练中 rollout eval / 保存调度单测
  checkpoint/                      # 训练 checkpoint 产物
  outputs/                         # eval JSON/MP4、scale launcher 日志等产物
  wandb/                           # W&B 本地运行产物
```

## 3. 运行入口

### 3.1 `train_policy.py`

单次训练主入口。核心顺序：

1. `parse_policy_train_args()` 读取 `--config`、`--device`、`--smoke`、`--resume`。
2. `PolicyTrainingConfig.load()` 加载 YAML，合并 eval 默认值，解析相对路径。
3. 设置 seed 和 device。
4. `derive_run_metadata()` 从数据集路径 / HDF5 attr / YAML 推导 env、task、mask、reward type、seed、timestamp。
5. `resolve_save_dir()` 生成 checkpoint 输出目录。
6. 初始化 `WandBLogger`。
7. 构造 `RobomimicReplayBuffer`，读取 HDF5 到内存，生成 `observation_space`、`action_space`、`obs_slices`。
8. 如果 `iql.features_extractor_type == "resnet18conv"`，把 replay buffer 的 `obs_slices` 注入 `cfg.iql.features_extractor_kwargs.obs_slices`。
9. `build_algo()` 创建 IQL 算法实例。
10. `build_checkpoint_metadata()` 从 HDF5 和 replay buffer 构造自描述 checkpoint 元数据。
11. 如果 `eval.enabled`，创建 `TrainingRolloutEvaluator`。
12. 如果传入 `--resume`，加载 checkpoint 的 modules 和 optimizers。
13. 调用 `algo.learn_offline(...)` 执行训练、日志、保存和可选周期性 rollout 评估。

常用命令：

```bash
cd /home/user/zhangzk/projects/fineprog/policy_training
python train_policy.py --config configs/iql.yaml
python train_policy.py --config configs/iql.yaml --smoke
python train_policy.py --config configs/iql.yaml --device cpu --resume checkpoint/.../final.pt
```

### 3.2 `iql_scale_train_sb3.py`

批量训练 launcher。它不直接训练模型，而是为多组 mask / dataset / seed 生成临时 YAML，并以 subprocess 启动 `train_policy.py`。

默认设置：

- 数据目录：`datasets/robomimic/can/mh/reward_labeled/resnet18feats`
- dataset stems：
  - `image_2view_v15_reward_labeled_original`
  - `image_2view_v15_reward_labeled_PBRS`
- masks：
  - `IQL_expert`
  - `IQL_expert_worse`
- seeds：`[1, 2, 3, 4, 5]`
- 最大并发：`2`
- 每个临时配置强制使用 `features_extractor_type: resnet18conv`，obs keys 固定为 low-dim 三项加两路 ResNet18Conv 特征。
- 日志写入 `outputs/scale_train_logs/iql_scale_train_sb3_{timestamp}/`。
- 支持 `--strict-dataset-check`、`--smoke`、`--device`、`--max-parallel`。

### 3.3 `evaluate_policy.py`

手动评估 checkpoint 的入口。核心顺序：

1. 加载 `configs/evaluate_policy.yaml` 并应用 CLI override。
2. `load_checkpoint_for_eval()` 加载并校验 strict schema v1 checkpoint。
3. 重建 IQL 算法实例、加载 modules、设为 eval mode。
4. 从 checkpoint 取训练配置、env metadata、shape metadata、obs slices 和可选归一化统计。
5. 构造 `ObservationAdapter`：把环境返回的 dict obs 转成训练时的 flat obs。
6. 构造 `SB3IQLRolloutPolicy`：包装 IQL actor，必要时反归一化 action。
7. `create_robomimic_env()` 创建 robosuite 环境。
8. 执行 N 个 rollout，输出 summary、per-rollout stats、JSON 和可选 MP4。

常用命令：

```bash
cd /home/user/zhangzk/projects/fineprog/policy_training
python evaluate_policy.py --config configs/evaluate_policy.yaml
python evaluate_policy.py --agent checkpoint/.../final.pt --n_rollouts 20 --horizon 400 --no_video
```

## 4. 配置系统

### 4.1 训练配置 `configs/iql.yaml`

主要字段：

- `algo_name`: 当前为 `iql`。
- `iql`: 算法和网络超参。
  - `features_extractor_type`: `flat_range` 或 `resnet18conv`。
  - `features_extractor_kwargs`: `resnet18conv` 使用；包含 `low_dim_keys` 和 `visual_specs`。
  - `learning_rate`, `tau`, `gamma`, `n_critics`, `n_critics_to_sample`
  - `policy_extraction`: `awr` 或 `ddpg`
  - `advantage_temp`, `expectile`, `clip_score`, `ddpg_bc_weight`
  - `pi_net_arch`, `qf_net_arch`, layer norm 开关
- `train`:
  - `batch_size`, `n_steps`, `log_every`, `save_every`
  - `save_dir_root`: checkpoint 根目录，默认配置为 `checkpoint`
  - `save_dir`: 可选硬覆盖输出目录
- `eval`: 训练中 rollout eval 配置。
  - `enabled`, `every_n_steps`, `warmstart_steps`, `n_rollouts`, `horizon`
  - `stochastic`, `terminate_on_success`
  - `video` 和 `output`
- `dataset`:
  - `h5_path`, `obs_keys`, `filter_key`
  - `action_keys`: `null` 时使用 demo 内 `actions`
  - `strict_next_obs`, `normalize_obs`, `normalize_actions`
- `device`, `seed`
- `wandb.enabled`, `wandb.project`

`PolicyTrainingConfig.load()` 只内置了 eval 默认值；其他字段主要依赖 YAML 显式给出。它会把相对路径解析到 `policy_training/` 根目录下，并注入 `project_root` 和 `config_path`。

### 4.2 评估配置 `configs/evaluate_policy.yaml`

主要字段：

- `agent`: checkpoint 路径。
- `device`: `auto` / `cuda` / `cpu`。
- `seed`
- `policy.stochastic`
- `rollout.n_rollouts`, `rollout.horizon`
- `env.name_override`
- `video.enabled`, `video.path`, `video.skip`, `video.fps`, frame size, camera names
- `output.dir`, `output.json_path`

CLI override 优先级高于 YAML。`video.path` 非空会强制开启视频；`--no_video` 会禁用视频。

## 5. 数据契约

### 5.1 HDF5 输入结构

`RobomimicReplayBuffer` 期望 robomimic 风格 HDF5：

```text
file.hdf5
  data/
    attrs["env_args"]             # JSON 字符串，用于重建评估环境
    demo_0/
      attrs["num_samples"]
      obs/
        <obs_key>                 # shape: (T, ...)
      next_obs/
        <obs_key>                 # shape: (T, ...)
      actions                     # shape: (T, action_dim)，除非 dataset.action_keys 指定多键 action
      rewards                     # shape: (T,)
      dones or terminals          # shape: (T,)
      states                      # 可存在，训练 replay buffer 不直接使用
    demo_1/
      ...
  mask/
    <filter_key>                  # demo id 列表，用于选择子集
```

默认 `configs/iql.yaml` 使用的数据是 `can/mh/reward_labeled/resnet18feats` 下的预计算 ResNet18Conv 特征 HDF5。抽样到的默认数据结构包含：

- 顶层：`data`, `mask`
- `mask` 里有 `IQL_expert`, `IQL_expert_worse`, `IQL_expert_half` 等 split
- demo 内有 `actions`, `dones`, `next_obs`, `obs`, `original_rewards`, `rewards`, `states`
- action shape 为 `(T, 7)`
- 默认 obs keys：
  - `robot0_eef_pos`: `(T, 3)`
  - `robot0_eef_quat`: `(T, 4)`
  - `robot0_gripper_qpos`: `(T, 2)`
  - `agentview_image`: `(T, 512, 3, 3)`，这里不是原始 RGB，而是预计算 ResNet18Conv feature map
  - `robot0_eye_in_hand_image`: `(T, 512, 3, 3)`

因此默认 flat observation dim 为 `3 + 4 + 2 + 512*3*3 + 512*3*3 = 9225`。经 `resnet18convFeaturesExtractor` 后，默认特征维度为 `9 + 64 + 64 = 137`。

### 5.2 Replay Buffer 行为

`datasets/robomimic/replay_buffer.py` 的 `RobomimicReplayBuffer`：

- 按 `filter_key` 从 `mask/{filter_key}` 选择 demos；未指定则用全部 demo。
- 按 `dataset.obs_keys` 顺序读取 `obs` 和 `next_obs`。
- rank 1 obs reshape 为 `(T, 1)`；rank > 2 obs flatten 为 `(T, -1)`。
- 第一条 demo 决定 `obs_slices`: `{obs_key: slice(start, stop)}`。
- 默认 `strict_next_obs=True`，缺失 `next_obs` 会报错。
- `action_keys=None` 时读取 `demo/actions`；否则拼接多个指定 action key。
- `normalize_obs=True` 时对整个 flat observation 计算 mean/std 并同时作用于 obs 和 next_obs。
- `normalize_actions=True` 时对 action 计算 mean/std 并作用于 replay buffer action。
- `sample(batch_size)` 返回 `ReplayBatch(observations, actions, next_observations, rewards, dones)`，全部是 torch tensor，放在初始化时指定的 device 上。

## 6. 算法与模型

### 6.1 算法工厂

`algos/__init__.py` 暴露 `build_algo(algo_name, observation_space, action_space, cfg, device)`。当前只支持：

- `iql` -> `algos/offline_rl/iql.py::IQL`

新增算法需要在该工厂中注册。

### 6.2 `OfflineRLBase`

`algos/offline_rl/base_offline_rl.py` 提供离线 RL 抽象基类：

- 子类必须实现 `_setup_model()`、`train_step()`、module/optimizer state dict 的保存和加载。
- `save(save_dir, tag)` 写 strict schema checkpoint；保存前调用 `validate_checkpoint_payload()`。
- `load(checkpoint_path)` 恢复 `global_step`、modules 和 optimizers。
- `learn_offline(...)` 是通用训练循环：
  - 每步从 replay buffer sample batch。
  - 调用 `train_step()`。
  - 增加 `global_step`。
  - 按 `rollout_evaluator.eval_cfg` 触发训练中 eval。
  - 按 `log_every` 写日志。
  - 按 `save_every` 保存 `step_{global_step}.pt`。
  - 结束保存 `final.pt`。

### 6.3 IQL 实现

`algos/offline_rl/iql.py::IQL` 是当前核心算法。

模型组件：

- `CustomMlpPolicy`
  - `actor`: squashed diagonal Gaussian actor，来自 SB3 SAC policy 派生实现。
  - `critic`: Q ensemble，数量由 `n_critics` 控制。
  - `critic_target`: target Q ensemble。
- `ValueCritic`
  - 单独的 V(s) 网络，使用 critic 的 feature extractor deepcopy。

训练步骤：

1. critic / value 更新：
   - `q_preds = critic(obs, actions)`
   - 从 `critic_target(obs, actions)` 随机抽取 `n_critics_to_sample` 个 Q，取 min 作为 expectile value 目标的一部分。
   - `target_q_values = rewards + (1 - dones) * gamma * V(next_obs)`
   - Q loss: MSE(`q_preds`, `target_q_values`)
   - V loss: expectile regression，误差为 `V(obs) - min_target_Q(obs, action)`
   - 对 critic 和 V 网络做 grad norm clip。
2. target network 更新：
   - 每 `target_update_interval` 步用 `polyak_update(..., tau)` 更新 `critic_target`。
3. actor 更新：
   - `policy_extraction == "awr"`：使用 advantage weighted regression，`exp((Q - V) * advantage_temp)` 后按 `clip_score` 截断，loss 为负加权 log prob。
   - `policy_extraction == "ddpg"`：使用 Q(pi) 加行为克隆 log prob 项，权重由 `ddpg_bc_weight / average_abs_Q` 缩放。

记录的核心指标包括 actor loss、Q loss、V loss、平均 Q、平均 reward、平均 V、Q target、actor log prob，以及 AWR 分支的 average advantage / advantage weight。

### 6.4 特征提取器

`models/feature_extractors.py`：

- `FlatRangeFeaturesExtractor`
  - 输入 flat obs `(B, D)`。
  - 根据 `dim_ranges` 切分连续区间。
  - 每段过一个 `nn.Linear(in_dim, out_dim)`。
  - 拼接成最终 features。
- `resnet18convFeaturesExtractor`
  - 面向预计算 ResNet18Conv feature map + low-dim obs。
  - low-dim keys 直接从 flat obs slice 中取出并 passthrough。
  - visual keys 从 flat obs slice reshape 回 `(B, C, H, W)`，默认 `(B, 512, 3, 3)`。
  - 每个 visual key 走 `SpatialSoftmax(input_shape=[C,H,W], num_kp=32)`，再 flatten 和 `Linear(num_kp*2, feature_dimension)`。
  - 默认两路视觉各输出 64 维。
  - 不包含 ResNet18Conv backbone 参数；训练阶段要求 HDF5 已存预计算 feature map。

### 6.5 神经网络工具

`models/model_utils.py`：

- `SpatialSoftmax`: 自包含实现，行为对齐 robomimic，输入 `[B,C,H,W]`，输出 `[B,num_kp,2]` keypoints。
- `FrozenPretrainedResNet18Conv`: 包装 vendored robomimic 的 `ResNet18Conv(pretrained=True, freeze=True, imagenet_norm=True)`。评估时如果环境给的是 RGB 图像，它会把图像转为与训练 HDF5 一致的 ResNet18Conv feature map。

`models/policies.py`：

- `create_mlp(...)`: 支持可选 `LayerNorm`。
- `CustomActor`: SB3 SAC Actor 派生，支持 policy MLP layer norm。
- `CustomContinuousCritic`: 多 Q 网络 critic，支持 critic MLP layer norm，并允许按 `critic_indices` 只计算部分 critic。
- `CustomSACPolicy` / `CustomMlpPolicy`: 组装 actor、critic、critic_target。

`models/value_critic.py`：

- `ValueCritic`: IQL 的 V(s) 网络；可训练自己的 feature extractor 参数。

## 7. 评估架构

### 7.1 robomimic / robosuite 环境封装

`envs/__init__.py` 会把 `envs/robomimic_runtime` 插入 `sys.path`，从而让 vendored robomimic 既可通过 `envs.robomimic_runtime.robomimic` 访问，也可在部分 vendored 文件里通过顶层 `robomimic` import 访问。

`envs/robomimic.py`：

- `load_env_metadata_from_dataset(dataset_path)` 从 HDF5 读取 env metadata，当前仅支持 `EnvType.ROBOSUITE_TYPE`。
- `create_robomimic_env(...)` 初始化 robomimic ObsUtils modality specs：
  - low-dim keys -> `obs.low_dim`
  - visual keys -> `obs.rgb`
  - 然后调用 vendored `EnvUtils.create_env_from_metadata(...)` 创建 robosuite env。

注意：评估依赖外部 robosuite / robosuite_models 等环境包；vendored runtime 不是完整环境模拟器。

### 7.2 checkpoint 加载与 obs adapter

`utils/eval_utils.py` 的关键类和函数：

- `load_checkpoint_for_eval(checkpoint_path, device_setting)`
  - `torch.load` checkpoint。
  - `validate_checkpoint_payload()` 校验 v1 schema。
  - 从 checkpoint config 和 shape metadata 重建 obs/action space。
  - 对 `resnet18conv` 注入 checkpoint 中的 `obs_slices`。
  - `build_algo()` 重建算法并加载 modules。
  - 设置 policy / actor / critic / V 网络为 eval mode。
- `ObservationAdapter`
  - 输入环境 dict obs。
  - 按 `cfg.dataset.obs_keys` 顺序拼接。
  - low-dim obs reshape 为 flat。
  - visual obs 如果已是期望 feature shape，直接 flatten；如果是 RGB，则用 `FrozenPretrainedResNet18Conv` 提特征。
  - 拼接后校验 flat dim 等于 checkpoint `shape_metadata["observation_dim"]`。
  - 如 checkpoint 有 obs normalization stats，则应用训练时同一套归一化。
- `SB3IQLRolloutPolicy`
  - 调用 IQL actor 产生 action。
  - `stochastic=False` 时使用 deterministic action。
  - 如 checkpoint 有 action normalization stats，则输出前反归一化。

### 7.3 rollout 与输出

- `rollout(...)`：
  - `env.reset()` 后用 `env.get_state()` / `env.reset_to()` 固定初始状态。
  - 每步调用 policy、`env.step(action)`、读取 reward/done/success。
  - success 从 `env.is_success()` 中取 `task` 字段。
  - `terminate_on_success=True` 时成功即提前结束。
  - 可用 `imageio_ffmpeg.write_frames` 写 MP4。
- `_summary_from_rollouts(...)` 聚合平均 Return、Horizon、Success_Rate、Num_Success、Num_Rollouts。
- 手动评估默认输出目录：

```text
outputs/eval/{env_name}/{task_name}/{split_name}/{algo_mask_name}/{seed_label}/{run_id}/{checkpoint_tag}/
  eval_step_{global_step}_{evalseed}_{nrollouts}_{horizon}.json
  eval_step_{global_step}_{evalseed}_{nrollouts}_{horizon}.mp4  # video.enabled 时
```

- 训练中 eval 由 `TrainingRolloutEvaluator` 管理：
  - 在 `learn_offline()` 内按 `eval.every_n_steps` 和 `eval.warmstart_steps` 触发。
  - 返回 `eval/return`, `eval/horizon`, `eval/success_rate`, `eval/num_success`, `eval/num_rollouts`, `eval/time_minutes`。
  - 可把少量 rollout video 记录到 W&B。

## 8. Checkpoint schema

`utils/checkpoints.py` 定义 strict v1 schema：

```python
CHECKPOINT_SCHEMA_VERSION = 1
REQUIRED_CHECKPOINT_FIELDS = (
    "checkpoint_schema_version",
    "global_step",
    "algo_name",
    "config",
    "env_metadata",
    "shape_metadata",
    "obs_slices",
    "modules",
    "optimizers",
)
```

`OfflineRLBase.save()` 会组合：

- 通用训练状态：
  - `global_step`
  - `modules`
  - `optimizers`
- `build_checkpoint_metadata()` 生成的元数据：
  - `checkpoint_schema_version`
  - `algo_name`
  - `config`: 训练 config 的 plain dict
  - `env_metadata`: HDF5 `data.attrs["env_args"]` JSON
  - `shape_metadata`
    - `action_dim`, `action_shape`
    - `observation_dim`
    - `obs_shapes`
    - `low_dim_obs_keys`, `visual_obs_keys`
    - `use_image_obs`
    - `first_demo_id`
  - `obs_slices`: `{key: [start, stop]}`
  - `normalization_stats`: 可选，包括 obs/actions

IQL 当前保存的 modules：

- `policy`: 包含 actor、critic、critic_target 等 SB3 policy state dict
- `v_net`: IQL value network state dict

IQL 当前保存的 optimizers：

- `actor_optimizer`
- `critic_optimizer`
- `v_optimizer`

评估入口不支持 legacy checkpoint；缺少上述字段或 schema version 不等于 1 会直接报错。

## 9. 输出命名与 run metadata

`utils/logger.py` 负责 run metadata：

- env / task 推导优先级：
  1. 从 dataset path 的 `datasets/{env}/{task}/...` 推导。
  2. 从 HDF5 `data.attrs["env_args"]` 推导。
  3. 从 YAML `env_name` / `task_name` fallback。
- reward type 从 HDF5 文件名中 `_reward_labeled_{variant}` 推导，例如 `PBRS` -> `pbrs`。
- mask name 从 `dataset.filter_key` 推导；如果以 `algo_name + "_"` 开头，会去掉前缀。
- `algo_mask_name = {ALGO}__{filter_key}`，例如 `IQL__IQL_expert_worse`。

当前 `resolve_save_dir()` 生成 checkpoint 目录：

```text
checkpoint/{env_name}/{task_name}/[{scale_run}/]{algo_mask_name}/{reward_type}/{run_timestamp}/{seed_label}/ckpt/
```

若 `train.save_dir` 显式指定，则完全跳过自动命名。

W&B 命名：

- group: `{env}/{task}/{algo_mask}/{reward_type}`，scale run 追加 `/SCALE`
- name: `{env}-{task}-[SCALE-]{algo_mask}-{reward_type}-{seed_label}`

## 10. 测试覆盖

`tests/test_eval_output_paths.py`：

- 评估输出 metadata 推导。
- 兼容旧/新 checkpoint path 布局。
- 自动 video/json path。
- video/json path override。
- `--no_video` 与 legacy `video.path` 行为。
- CLI override 对 eval config 的影响。

`tests/test_in_training_rollout_eval.py`：

- `PolicyTrainingConfig` 的 eval 默认值。
- `TrainingRolloutEvaluator.run()` 的指标聚合。
- `OfflineRLBase.learn_offline()` 中 eval trigger schedule 和 checkpoint save schedule。

可运行：

```bash
cd /home/user/zhangzk/projects/fineprog/policy_training
pytest tests
```

## 11. 重要依赖

`policy_training` 未单独提供 requirements 文件；依赖主要来自上层 `fineprog/requirements.txt` 和源码 imports。

核心依赖：

- `torch`, `torchvision`
- `numpy`, `h5py`, `pyyaml`
- `gymnasium`
- `stable-baselines3`
- `tqdm`
- `wandb`
- `imageio-ffmpeg`
- `robosuite`, `robosuite_models` 等 robomimic / robosuite 运行环境依赖

注意：上层 `requirements.txt` 未显式列出 `stable-baselines3` 和 `gymnasium`，但 `policy_training` 源码直接 import 它们。

## 12. 扩展和维护注意事项

- 源码使用顶层 import，如 `from algos import build_algo`、`from utils...`，默认假设运行 cwd 是 `policy_training/` 或该目录已在 `sys.path`。
- 训练阶段的 `resnet18conv` 分支不在线计算 ResNet feature；它要求 HDF5 已存与 `visual_specs.input_shape` 匹配的预计算 feature map。
- 评估阶段可以处理环境返回的原始 RGB，因为 `ObservationAdapter` 会用冻结 ResNet18Conv 转成 feature map。
- `obs_slices` 是连接 replay buffer、feature extractor、checkpoint 和 eval adapter 的关键契约。修改 `dataset.obs_keys` 或 HDF5 shape 时必须保证 `visual_specs` 和 `obs_slices` 一致。
- `normalize_obs=True` 当前按整个 flat observation 计算一组 mean/std，而不是逐 obs key 统计。
- `create_robomimic_env()` 当前只支持 robosuite 类型 env metadata。
- `checkpoint_schema_version` 是严格契约；修改 checkpoint 字段时需要同步 `validate_checkpoint_payload()`、`build_checkpoint_metadata()` 和 `load_checkpoint_for_eval()`。
- 新增算法时需要同时考虑：算法工厂注册、保存/加载模块名、评估侧是否能用现有 `SB3IQLRolloutPolicy` 或需要新 rollout policy wrapper。
- `envs/robomimic_runtime/` 是 vendored runtime。除非要修复 robomimic 兼容性，普通训练逻辑应优先改 `policy_training` 自有模块。

## 13. 给上游 LLM 的最短阅读路径

若上下文预算有限，优先阅读：

1. `train_policy.py`
2. `configs/iql.yaml`
3. `datasets/robomimic/replay_buffer.py`
4. `algos/offline_rl/iql.py`
5. `models/feature_extractors.py`
6. `utils/train_utils.py`
7. `utils/eval_utils.py`
8. `evaluate_policy.py`

这 8 个文件覆盖了从 HDF5 数据、flat obs 契约、IQL 更新、checkpoint 元数据到 rollout 评估的主链路。
