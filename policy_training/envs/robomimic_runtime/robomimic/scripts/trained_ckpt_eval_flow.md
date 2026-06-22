# trained ckpt eval 工程流程说明

本文档说明 `run_trained_agent.py` 如何完成一次 trained checkpoint evaluation。重点不是逐行解释代码，而是说明每个大步骤为什么要这么做、里面的小模块各自负责什么、以及这些设计对 eval 结果有什么影响。

## 0. 总体心智模型

一次 trained ckpt eval 可以理解成四件事：

1. 读入 checkpoint，恢复训练时的 policy。
2. 根据 checkpoint 里保存的环境信息，重新创建一个匹配的 eval env。
3. 让 policy 在 env 里跑若干条 episode。
4. 统计成功率、回报、步数，并可选保存视频或 rollout dataset。

这里的关键点是：ckpt 不只是模型权重。它还保存了训练 config、obs/action shape、env metadata、normalization stats 等信息。eval 必须尽量复用这些信息，否则 policy 输入输出格式、环境 observation、frame stack、action normalization 等很容易和训练时不一致，评估结果就不可信。

## 1. CLI 参数：告诉脚本这次 eval 想怎么跑

`run_trained_agent.py` 首先解析命令行参数。CLI 参数解决的是“这次评估的运行设置”问题，而不是“模型是什么结构”问题。

常用参数含义：

- `--agent`: 要评估的 checkpoint 路径，必填。
- `--n_rollouts`: 跑多少条 episode，默认 `27`。
- `--horizon`: 每条 episode 最多跑多少步，默认 `400`。
- `--env`: 可选，用来覆盖 ckpt 中记录的 env name。
- `--render`: 是否打开屏幕渲染。
- `--video_path`: 是否保存 rollout 视频。
- `--video_skip`: 视频每隔多少 env step 采一帧，默认 `5`。
- `--camera_names`: 渲染视频或屏幕时使用哪些 camera，默认 `agentview`。
- `--dataset_path`: 是否把 rollout 轨迹保存成 hdf5 dataset。
- `--dataset_obs`: 保存 dataset 时是否也保存 obs / next_obs。
- `--seed`: 是否固定 numpy / torch 随机种子。

为什么这些参数放在 CLI，而不是完全用 ckpt 里的 config：

- eval 经常需要临时调整，比如多跑几条 episode、换一个 horizon、保存视频、换 camera。
- 这些调整不改变 policy 本身，只改变本次评估方式。
- ckpt 仍然负责提供模型结构、环境 metadata、obs/action 格式等核心信息。

需要注意一个实现细节：`--horizon` 默认是 `400`。代码里虽然写了 `args.horizon is None` 时才从 ckpt config 读取 `config.experiment.rollout.horizon`，但由于 argparse 默认给了 `400`，所以正常不传 `--horizon` 时也会使用 `400`，不会读取 ckpt 里的 horizon。

## 2. 选择 device：决定模型推理跑在 CPU 还是 GPU

脚本会通过 `TorchUtils.get_torch_device(try_to_use_cuda=True)` 选择 torch device。

这个步骤的意义很直接：

- 如果机器有可用 CUDA，就把模型放到 GPU 上推理，速度更快。
- 如果没有 CUDA，就回退到 CPU，保证脚本仍然能运行。

device 只影响 policy 推理的位置，不改变 env 的运行方式。robosuite / gym 环境本身通常仍在 CPU / simulator 侧执行。

## 3. policy_from_checkpoint：把 ckpt 还原成可 rollout 的 policy

入口模块：`FileUtils.policy_from_checkpoint(...)`

这个步骤回答的问题是：只有一个 `.pth` 文件，如何恢复成一个可以接收 observation、输出 action 的 policy？

### 3.1 读取 checkpoint

相关模块：`load_dict_from_checkpoint(...)`、`maybe_dict_from_checkpoint(...)`

意义：从磁盘读取 `.pth`，得到一个 Python dict。这个 dict 里面不只有权重，还有 config、env metadata、shape metadata、normalization stats 等。

为什么要这样做：后面 policy 恢复和 env 创建都会用到同一个 ckpt dict。集中读取一次，可以避免重复读文件，也能保证 policy 和 env 使用的是同一份 metadata。

### 3.2 还原 config

相关模块：`config_from_checkpoint(...)`

意义：从 `ckpt["config"]` 里还原训练时使用的 config。config 决定了算法类型、obs keys、frame stack、action keys、action normalization、网络结构等。

为什么 eval 需要训练 config：policy 的网络结构、输入 observation 的处理方式、action 后处理方式都必须和训练时一致。如果训练时用了 frame stack，eval 也必须套同样的 wrapper；如果 action 使用特殊格式，比如 `rot_6d`，eval 输出时也需要按 config 转回环境能执行的格式。

简单说，config 是恢复 policy 行为的说明书。

### 3.3 初始化 ObsUtils

相关模块：`ObsUtils.initialize_obs_utils_with_config(config)`

意义：告诉 robomimic 哪些 observation key 是 low-dim，哪些是 rgb，哪些是 depth。

为什么这一步重要：图像 observation 通常需要转 channel、归一化、变成网络期望的格式；depth observation 也可能有特殊处理规则。如果 obs modality 判断错了，policy 看到的输入格式就会和训练时不一致。

### 3.4 读取 shape_metadata

相关字段：`ckpt["shape_metadata"]`

常用内容：

- `all_shapes`: 每个 obs key 的 shape。
- `ac_dim`: action 维度。
- `all_obs_keys`: policy 使用哪些 observation keys。
- `use_images`: 是否使用 rgb image obs。
- `use_depths`: 是否使用 depth obs。

意义：一方面用来重建模型输入层和输出维度，另一方面用来决定 eval env 是否需要每步返回 camera image / depth。

为什么 env 创建也需要 shape metadata：如果 policy 训练时用 image obs，eval env 必须开启 camera obs；如果 policy 训练时不用 image obs，就不必每步渲染图像，能节省大量开销。

### 3.5 恢复 normalization stats

相关字段：`obs_normalization_stats`、`action_normalization_stats`

意义：obs stats 用来把 eval 时的 observation 按训练集统计量归一化；action stats 用来把 policy 输出从训练时的归一化空间转换回环境 action 空间。

为什么不能省：如果训练时 normalize 了 obs，eval 不 normalize，policy 输入分布会变。如果训练时 normalize 了 action，eval 不 unnormalize，环境收到的动作尺度会错。这类错误通常不会报错，但会让 policy 表现明显变差。

### 3.6 通过 algo_factory 重建模型

相关模块：`algo_factory(...)`、`model.deserialize(...)`、`model.set_eval()`

意义：`algo_factory` 根据 algo name 和 config 创建正确的算法对象，比如 BC、IQL、Diffusion Policy 等；`deserialize` 把 ckpt 里的权重加载进去；`set_eval` 关闭训练态行为，比如 dropout / batchnorm 的训练更新等。

为什么不用直接保存一个裸模型：robomimic 的 policy 不只是一个 torch module。不同算法有不同网络、辅助模块和推理逻辑。通过 `algo_factory + deserialize` 可以统一恢复各种算法。

### 3.7 包装成 RolloutPolicy

相关模块：`RolloutPolicy`

意义：把训练用的 Algo 对象包装成 rollout 用 policy。对外暴露简单接口：输入 env observation，输出 env action。

`RolloutPolicy` 主要负责：

- episode 开始时 reset policy 内部状态。
- 把 numpy obs 转成 torch tensor。
- 自动加 batch 维。
- 搬到正确 device。
- 做 obs normalization。
- 处理 rgb / depth observation。
- 调用模型生成 action。
- 把 action 从 tensor 转回 numpy。
- 必要时做 action unnormalization 和 rotation format conversion。

它的价值是把“模型推理前后的格式处理”集中封装起来。rollout loop 不需要关心 policy 是 transformer、RNN、diffusion，还是普通 MLP。

## 4. env_from_checkpoint：创建和 ckpt 匹配的 eval env

入口模块：`FileUtils.env_from_checkpoint(...)`

这个步骤回答的问题是：policy 要在哪个环境里评估？这个环境应该怎么配置？

### 4.1 使用 env_metadata

相关字段：`ckpt["env_metadata"]`

通常包含：

- `env_name`: 环境名。
- `type`: 环境类型，比如 robosuite / gym / ig_momart。
- `env_kwargs`: 创建环境所需参数。
- `env_version`: 环境版本，可选。

意义：保证 eval env 尽量和训练数据 / 训练 rollout 使用的 env 一致，避免手写 env 参数时漏掉 controller、robot、camera、reward shaping 等设置。

为什么 env metadata 必须从 ckpt 来：同名任务在不同 kwargs 下可能行为不同。controller、机器人型号、相机配置、reward shaping 都会影响 observation 和 success。如果 eval env 和训练时不一致，评估结果可能没有可比性。

### 4.2 env_name override

相关参数：`--env`

意义：默认使用 ckpt 中保存的 `env_name`；如果传了 `--env`，则用 CLI 的 env name 覆盖 ckpt 中的 env name。

为什么保留这个能力：有时想测试同一个 policy 在相似任务或变体任务上的泛化能力。这只覆盖 env name，其他 env kwargs 仍来自 ckpt metadata。

使用时需要谨慎：如果新 env 的 observation / action / success 定义和原 env 不兼容，可能直接报错或得到无意义结果。

### 4.3 根据 env type 选择环境类

相关模块：`EnvUtils.get_env_class(...)`、`EnvUtils.create_env_from_metadata(...)`

env type 到 class 的映射：

- `ROBOSUITE_TYPE` -> `EnvRobosuite`
- `GYM_TYPE` -> `EnvGym`
- `IG_MOMART_TYPE` -> `EnvGibsonMOMART`

意义：robomimic 用统一接口包装不同后端环境。rollout loop 只需要调用 `reset`、`step`、`render`、`get_state`、`is_success`，不需要在主脚本里区分 robosuite / gym / iGibson 的细节。

### 4.4 决定是否开启 image / depth obs

来源：`shape_meta["use_images"]`、`shape_meta["use_depths"]`

意义：如果 policy 使用 image obs，env 每步必须返回 camera image；如果 policy 使用 depth obs，env 每步必须返回 depth；如果 policy 不用图像，就关闭这些昂贵功能，提高 eval 速度。

为什么这个决定来自 shape metadata，而不是 CLI：policy 输入需求是训练时确定的。eval 时不能随意少给一个 obs key，否则模型无法正常推理。

### 4.5 render 和 render_offscreen

来源：`render = args.render`，`render_offscreen = args.video_path is not None`

意义：`render=True` 用于屏幕可视化；`render_offscreen=True` 用于保存视频。如果 policy 本身使用 image obs，也会在环境内部强制开启 offscreen renderer。

为什么屏幕 render 和视频不能同时开：一个是交互式窗口，一个是离屏写视频，混在一起容易引入渲染后端冲突和性能问题，所以脚本明确禁止二者同时使用。

### 4.6 包 env wrapper

相关模块：`EnvUtils.wrap_env_from_config(...)`

目前主要处理 `FrameStackWrapper`。

意义：如果训练时用了 `config.train.frame_stack > 1`，eval 时也要让 env 返回堆叠后的 observation。对 temporal policy 来说，这一步很关键。

为什么 wrapper 不直接写在 env 里：frame stack 是训练配置，不是环境本身的物理属性。同一个 env 可以被不同 policy 用不同 wrapper 评估。

## 5. robosuite env 创建时的关键设置

如果环境类型是 robosuite，实际会创建 `EnvRobosuite`。这个 wrapper 的意义是把 robosuite 原生接口适配成 robomimic 的统一 `EnvBase` 接口。

关键设置及理由：

- `has_renderer=render`: 用户要看屏幕画面时才打开，避免无意义开销。
- `has_offscreen_renderer=render_offscreen or use_image_obs`: 保存视频或获取 image obs 时必须打开。
- `ignore_done=True`: robosuite 任务通常按固定 horizon rollout，是否成功由 `is_success()` 判断。
- `use_object_obs=True`: robomimic 常用 object low-dim obs，默认保留。
- `use_camera_obs=use_image_obs`: 只有 policy 需要 image obs 时才让 env 每步返回图像。
- `camera_depths=use_depth_obs`: 只有 policy 需要 depth obs 时才打开。

`EnvRobosuite` 还负责：

- 把 robosuite 原始 observation 整理成 robomimic policy 期望的 obs dict。
- 处理 image 上下翻转。
- 提供 `get_state()` 和 `reset_to()`，用于保存 / 恢复 simulator state。
- 提供 `is_success()`，统一返回至少包含 `task` 的成功字典。
- 提供 `serialize()`，用于把 rollout dataset 的 env metadata 写回 hdf5。

## 6. seed：让随机性尽量可复现

如果传入 `--seed`，脚本会设置 numpy seed 和 torch seed。

意义：尽量让 policy 推理中的随机性、torch 随机操作等可复现，便于比较不同 ckpt 或不同代码修改后的结果。

限制：这不一定能完全固定 simulator 的所有随机性。如果环境内部、物理引擎或渲染后端还有额外随机源，可能还需要环境级 seed 支持。

## 7. video writer：把 rollout 渲染成视频

如果提供 `--video_path`，脚本会创建 ffmpeg video writer。

这个模块的意义：eval 不只是输出数字，也能人工检查 policy 到底在做什么。对 debug 很有用，比如看失败是因为抓取偏了、相机 obs 错了、动作尺度错了，还是任务成功判断异常。

当前脚本的设计：

- 每个 camera 渲染成 `512x512`。
- 多个 camera 横向拼接。
- `fps=20`。
- `video_skip` 控制每隔多少 env step 写一帧。

为什么要有 `video_skip`：每步都写视频会很慢，文件也很大。rollout 通常有几百步，隔几步采样足够观察行为。

为什么使用 `imageio_ffmpeg.write_frames(...)`：脚本注释说明，原先 imageio 自动插件选择在某些环境会出错。直接用 imageio_ffmpeg 可以绕开插件自动选择问题，把 RGB frame 流式送给 ffmpeg。

## 8. dataset writer：把 policy rollout 保存成新的 hdf5

如果提供 `--dataset_path`，脚本会把 rollout 轨迹写成 hdf5。

这个模块的意义：可以把 trained policy 生成的数据保存下来。后续可以做数据分析、回放、再训练、DAgger 风格流程，或状态转 observation 处理。

默认保存：actions、states、rewards、dones。

可选保存：obs、next_obs。

为什么默认不保存 obs：image obs / depth obs 体积很大。保存 states 更轻量。对 robosuite 这类 simulator，可以之后用 `dataset_states_to_obs.py` 根据 states 重新提取 obs。

为什么要保存 env metadata：hdf5 的 `env_args` 记录了如何重新创建环境。后续回放或从 states 重新生成 obs 时，需要知道原来的 env 配置。

## 9. 单条 rollout：policy 和 env 真正交互的地方

入口函数：`rollout(...)`

它的职责是跑一条 episode，并返回统计信息 `stats` 和轨迹数据 `traj`。

### 9.1 episode 开始：reset policy 和 env

开始时会依次做：

1. `policy.start_episode()`
2. `env.reset()`
3. `env.get_state()`
4. `env.reset_to(state_dict)`

每一步的理由：

- `policy.start_episode()`: 清空 policy 的 episode 内部状态，比如 RNN hidden state、历史缓存等。
- `env.reset()`: 让环境进入一个新的初始状态。
- `env.get_state()`: 取出当前 simulator state，用于记录初始状态。
- `env.reset_to(state_dict)`: 对 robosuite 来说，这是为了让状态回放 / action playback 更稳定一致。

`reset_to` 这一步看起来重复，但它的意义是把“初始状态”显式固定下来。这样后续保存 states 或做 deterministic playback 时更可靠。

### 9.2 每一步：obs -> action -> env.step

核心过程：当前 obs 进入 `RolloutPolicy`，被预处理后送进模型，模型输出 action，action 后处理后交给 `env.step(action)`，环境返回 next_obs、reward、done、info。

这个过程的意义：policy 负责决策，env 负责执行 action 和推进仿真，rollout loop 负责把两者接起来并记录结果。

每一步都会更新：累计 reward、当前任务是否 success、可选 video frame、action / reward / done / state，以及可选 obs / next_obs。

### 9.3 为什么每步都保存 state

保存 simulator state 的意义：states 是环境的低维、可回放表示，比 image obs 更省空间，可以用于后续 dataset 回放或重新提取 obs。

对 robosuite 来说，state dict 通常包括 MuJoCo model XML、simulator flattened state，新版本里还可能有 episode metadata。

### 9.4 什么时候结束一条 episode

结束条件是：`done`、`success`，或者达到 `horizon`。

意义：

- `done`: 环境认为 episode 结束。
- `success`: 任务已经成功，没有必要继续跑。
- `horizon`: 防止 policy 永远跑下去。

对 robosuite 来说，由于通常设置了 `ignore_done=True`，`done` 往往不会触发，主要是 success 或 horizon 结束。

需要注意：独立 eval 脚本没有暴露 `terminate_on_success` 参数，而是固定 success 后提前停止。

### 9.5 rollout 异常处理

如果仿真中出现环境定义的 rollout exception，脚本会打印 warning，而不是直接让整个 eval 崩掉。

意义：有些坏 policy 可能让机器人动作过大、仿真不稳定。单条 rollout 出错不应该直接中断所有评估，这样至少还能得到其他 rollout 的统计结果。

## 10. 统计指标：把多条 rollout 汇总成 eval 结果

每条 rollout 返回：

- `Return`: 本条 episode 的累计 reward。
- `Horizon`: 本条 episode 实际执行步数。
- `Success_Rate`: 本条 episode 是否成功，成功为 `1.0`，失败为 `0.0`。

所有 rollout 完成后，脚本计算：

- 平均 Return。
- 平均 Horizon。
- 平均 Success_Rate。
- `Num_Success`: 成功 episode 数。

为什么这样统计：单条 rollout 可能受初始状态、随机性、仿真扰动影响。多条 rollout 的平均成功率更能代表 policy 性能。Return 可以看 reward 表现，但在很多 imitation learning 任务里，Success_Rate 通常更直观。Horizon 可以辅助判断 policy 是快速成功、拖到最后失败，还是很早异常结束。

## 11. 和训练期 rollout 的关系

训练脚本 `train.py` 里也会做 eval rollout，调用的是 `TrainUtils.rollout_with_stats(...)`。

独立脚本 `run_trained_agent.py` 和训练期 rollout 的目的不同：

- 训练期 rollout 是训练过程的一部分，用来定期监控模型、保存 best ckpt、写 tensorboard / wandb。
- 独立 eval 是训练完成后，拿某个 ckpt 单独评估、保存视频或保存 policy 生成的数据。

主要差异：

- 独立 eval 使用自己的 `rollout(...)`。
- 独立 eval 可以保存 hdf5 rollout dataset。
- 独立 eval 每步记录 simulator state。
- 独立 eval 默认 `n_rollouts=27`，训练期 base config 默认 `rollout.n=50`。
- 独立 eval 固定 success 后提前停止。
- 独立 eval 的 `--horizon` 默认是 `400`，正常不传时不会读取 ckpt config 的 horizon。

## 12. 端到端流程总结

完整流程可以概括为：

```text
CLI args
  -> 读取 ckpt
  -> 恢复 config / obs utils / shape metadata
  -> 重建 algo model 并加载权重
  -> 包装成 RolloutPolicy
  -> 从 ckpt env_metadata 创建 eval env
  -> 按 config 套 env wrapper
  -> 可选 seed / video writer / dataset writer
  -> 跑 n_rollouts 条 episode
  -> 每条 episode 内执行 obs -> action -> env.step
  -> success / done / horizon 时结束
  -> 汇总 Return / Horizon / Success_Rate / Num_Success
  -> 可选保存 video 和 hdf5 dataset
```

一句话总结：`run_trained_agent.py` 的核心思想是“用 ckpt 恢复训练时的 policy 和环境契约，再用 CLI 控制这次评估怎么跑”。ckpt 保证输入输出格式、环境配置和训练时一致；CLI 提供评估时需要的灵活性，比如跑多少条、保存不保存视频、是否导出 dataset。
