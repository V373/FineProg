# fineprog 工程架构

**项目定位**：位于 `/home/user/zhangzk/projects/fineprog/` 的机器人操作进度表示与策略学习工程。根目录负责基于 TCC 的视频时序表示训练、嵌入提取、进度评估及奖励数据构建；`policy_training/` 负责基于 Robomimic 数据的离线 IQL、在线 SAC、策略评估与实验分析。

---

## 一、目录结构总览

```
fineprog/
├── train_encoder.py              # TCC 编码器训练入口
├── extract_embeddings.py         # 编码器嵌入提取入口
├── evaluate_encoder.py           # 评估任务统一入口
├── models/                       # ResNet50 Conv4c + 时序嵌入器
├── algos/
│   ├── loss/                     # TCC / InfoNCE / Triplet / Composite 损失
│   └── eval_task/
│       ├── base_task.py          # BaseTask + build_task 工厂
│       └── tcc_eval_tasks/       # 对齐、分类、进度、激活图等评估任务
├── dataset_preparation/          # 视频 H5 构建与 phase 标签写入
├── configs_v2/
│   ├── project.yaml              # 根目录与工件命名模板
│   ├── train.yaml / extract.yaml / data_process.yaml
│   ├── loss/ / eval/ / visualize/
│   └── registry/                 # datasets.yaml / runs.yaml
├── scripts/
│   ├── tsne/                     # t-SNE 可视化
│   ├── robomimic/                # 数据、奖励与 IQL 实验辅助脚本
│   ├── compute_mean_embedding_path.py
│   ├── backfill_latent_distance.py
│   └── run_online_sac_reward_sweep.py
├── policy_training/              # 策略学习子系统
│   ├── train_policy.py           # 离线 IQL / 在线 SAC 统一训练入口
│   ├── evaluate_policy.py        # 串行或并行 rollout 评估
│   ├── algos/                    # OfflineRLBase、IQL、OnlineRLBase、OnlineSAC
│   ├── models/                   # actor / critic / V 网络与特征提取器
│   ├── datasets/robomimic/       # 离线与在线 replay buffer
│   ├── reward_model/             # TCC Expert Projection 在线奖励
│   ├── envs/                     # Robomimic / Robosuite 环境适配
│   ├── configs/                  # iql / online_sac / evaluate_policy
│   ├── utils/                    # checkpoint、日志、评估与配置工具
│   └── tests/                    # 策略训练与评估回归测试
├── third_party/robomimic/        # Robomimic fork（Git submodule）
├── datasets/                     # 根编码器流程的数据与嵌入工件
├── checkpoint/ / outputs/ / wandb/  # 根编码器流程运行产物
├── tests/                        # Gaussian progress 单元测试
└── utils/                        # ConfigV2、registry 与训练中评估
```

---

## 二、总体工作流

```
[进度表示学习]
原始视频 / Robomimic HDF5
  → dataset_preparation/ → datasets/processed/*.h5
  → train_encoder.py → checkpoint/*/encoder_epoch*.pt
  → extract_embeddings.py → datasets/embeddings/**/*.h5
  → evaluate_encoder.py / scripts/tsne/ → 进度指标与 H5 / PNG / MP4 工件

[策略与奖励学习]
Expert Projection / Gaussian Progress 等进度结果
  → scripts/robomimic/make_robomimic_dense_reward.py
  → Robomimic original / dense / PBRS HDF5
  → policy_training/train_policy.py
      ├── IQL：离线 replay buffer 训练
      └── Online SAC：环境采样 + sparse / dense / PBRS 奖励训练
  → policy_training/evaluate_policy.py → rollout JSON / MP4 / success 指标
```

**人工标注辅助流程**：
```
raw/*.mp4
    ↓  mp4vid_to_png.py  --task <task_folder> [--fps N] [--size 224]
    ↓  指定 VIDEO_NAMES 仅提取需要标注的视频
datasets/raw_img/{video_stem}/frame_{idx:06d}.png
    ↓  人工逐帧查看，记录关键事件帧号
datasets/phase_labels/pouring_train56_phase_labels.csv
    ↓  add_phase_labels.py  --embd_h5 ...  --keyframes_csv ...
datasets/embeddings/{run_name}/{dataset_stem}-embd-labeled.h5
```

**当前入口状态补充**：
- `train_encoder.py`、`extract_embeddings.py`、`evaluate_encoder.py` 当前统一通过 `ConfigV2` 读取 `configs_v2/`。
- 旧版 `configs/` 与 `current_run.yaml` 已移除，主流程不再保留 v1 fallback。
- `dataset_preparation/mp4vid_to_h5data.py`、`train_encoder.py`、`extract_embeddings.py`、`scripts/compute_mean_embedding_path.py` 均支持通过 `--register` 调用 `RegistryV2`，把数据集 / run / embedding 工件追加登记到 `configs_v2/registry/`。
- `train_encoder.py` 新增可选 backbone feature cache 分支，适用于 `train_base in {only_bn, frozen}` 且 `extract_backbone_cache=true` 的训练路径。
- `train_encoder.py` 还支持 `in_training_eval` 配置块：在 checkpoint 保存后复用一份临时 embedding 执行训练中评估，当前落地任务为 `latent_distance_heatmap` 与 `kendalls_tau`。
- `evaluate_encoder.py` 暴露 `run_eval_task()` 供 `utils/in_training_eval.py` 和批处理脚本复用，因此部分评估既可命令行运行，也可被程序化调用。
- `policy_training/train_policy.py` 通过 `algo_name` 选择 `iql` 或 `online_sac`；两者共用模型、checkpoint、W&B 和 rollout 评估基础设施。
- `policy_training/evaluate_policy.py` 支持多 seed、串行/多进程 rollout 及自动组织 JSON / MP4 输出。

---

## 三、模型组件（`models/`）

### `TCCEncoder`（encoder.py）
完整编码器，串联骨干网络与时序嵌入模块：

```
输入: [B, clip_len, context_size, 3, 224, 224]
         ↓ 展平为 [B*clip_len*context_size, 3, 224, 224]
ResNet50Conv4cBackbone
         ↓ [B*clip_len*context_size, 1024, 14, 14]
         ↓ 重组为 [B, clip_len, context_size, 1024, 14, 14]
TCCTemporalEmbedder
         ↓
输出: [B, clip_len, 128]
```

### `ResNet50Conv4cBackbone`（backbone.py）
- 使用 ResNet50 的 `conv1 → layer1 → layer2 → layer3(Conv4c)` 阶段
- 输出特征图：`[N, 1024, 14, 14]`
- 支持三种训练模式：`frozen` / `only_bn`（仅 BN 参数可训练）/ `train_all`

### `TCCTemporalEmbedder`（temporal_embedder.py）
- 输入：`[B*clip_len, 1024, context_size, 14, 14]`
- 两层 3D Conv（1024→512→512）
- 全局 3D MaxPool → 两层 FC（512→512）→ 线性投影至 128 维
- 输出：`[B, clip_len, 128]`

---

## 四、算法组件（`algos/`）

### 损失函数层级

| 层级 | 文件 | 职责 |
|------|------|------|
| 接口层 | `encoder_loss.py` | `BaseEncoderLoss` 抽象基类；`build_loss()` 工厂函数 |
| 对比损失 | `contrastive/loss_temporal_infonce.py` | 单视频时序阈值 InfoNCE 损失 |
| 对比损失 | `contrastive/loss_temporal_triplet.py` | 单视频时序 Triplet Margin 损失 |
| 复合损失 | `composite/loss_composite.py` | 多个子损失的加权和 |
| 统一入口 | `tcc/loss_tcc.py` | `TCCLoss`：读配置、分发到对齐实现 |
| 数学实现 | `tcc/deterministic_alignment.py` | 所有时步参与 cycle-consistency 计算 |
| 数学实现 | `tcc/stochastic_alignment.py` | 随机采样部分 cycle（内存高效） |

**接口规范**：
- 输入：`embeddings [B, clip_len, D]` + `batch dict`（含 `target_steps`, `seq_len`）
- 输出：`{"loss": scalar tensor, "metrics": dict}`
- `build_loss()` 当前支持 `tcc` / `temporal_infonce` / `temporal_triplet` / `composite`（另保留 `temporal_contrastive_infonce` 别名）
- `configs_v2/train.yaml` 通过 `loss_name` + `loss_config` 选择训练时启用的损失实现；对应 YAML 位于 `configs_v2/loss/`

### 评估任务

| 文件 | 职责 |
|------|------|
| `base_task.py` | `BaseTask` 抽象基类；`build_task()` 工厂函数 |
| `tcc_eval_tasks/task_kendall.py` | Kendall's Tau：对所有视频对做最近邻对齐，度量时序保序性 |
| `tcc_eval_tasks/task_phase_classification.py` | 冻结嵌入 + 线性 SVM phase 分类 |
| `tcc_eval_tasks/task_expert_projection.py` | 将 non-expert 轨迹软投影到 expert 轨迹 |
| `tcc_eval_tasks/task_latent_distance_heatmap.py` | 计算单条或全量轨迹的帧间距离热图 |
| `tcc_eval_tasks/task_gaussian_progress_fitting.py` | 从 expert 轨迹离线拟合分段高斯进度模型 |
| `tcc_eval_tasks/task_gaussian_progress_pred.py` | 用高斯模型推断 non-expert 连续进度 |
| `tcc_eval_tasks/task_activation_map.py` | 重跑 encoder 并导出 backbone / temporal 激活图 |

**当前工厂支持的任务名**：`kendalls_tau` / `classification` / `expert_projection` / `latent_distance_heatmap` / `gaussian_progress_fitting` / `gaussian_progress_pred` / `activation_map`

---

## 四-B、评估系统详细框架（`evaluate_encoder.py` + `algos/eval_task/`）

### 总体架构

评估当前以 **V2 配置 + 离线 embedding 工件** 为主：`evaluate_encoder.py` 从 `configs_v2/eval/{task}.yaml` 读取任务配置，再由 `ConfigV2` 把 registry ref 解析为具体路径。除 `activation_map` 会加载 checkpoint 并重跑 encoder 外，其余任务直接消费一个或多个 H5 工件。

```
configs_v2/
    project.yaml + registry/*.yaml   → dataset / run / embedding 别名解析
    eval/{task_name}.yaml            → 任务超参数与输入 ref
          ↓
evaluate_encoder.py  main(--task)
    ① ConfigV2.load_eval(task_name)
    ② resolve *_embedding_ref / *_dataset_ref → absolute paths
    ③ 按任务加载 embedding / Gaussian model / processed dataset / checkpoint
    ④ build_task(task_name)
    ⑤ task.configure(resolved) / task.evaluate(...)
          ↓
algos/eval_task/base_task.py
  build_task(task_name) → BaseTask 子类实例
          ↓
algos/eval_task/tcc_eval_tasks/*.py
          ↓
输出：标量指标 + 任务特定工件（PNG / H5 / MP4）
```

### V2 路径解析层级（`ConfigV2.load_eval`）

```
1. configs_v2/eval/{task}.yaml  :: 显式 *_h5_path / output_dir 覆盖
2. *_embedding_ref / *_dataset_ref  →  registry/runs.yaml / registry/datasets.yaml
3. project.yaml  :: dirs + templates 负责派生 checkpoint / embedding / output 的绝对路径
```

评估主路径已统一为 `configs_v2/eval/*.yaml` + registry 引用解析，不再依赖旧版配置桥接文件。

### `BaseTask` 接口规范

```python
class BaseTask(abc.ABC):
    def evaluate(self, embeddings_dataset: dict) -> dict:
        # 必须返回:
        # {
        #   "task_name":    str,
        #   "metric_name":  str,
        #   "metric_value": float,
        # }
```

除 `evaluate()` 外，`BaseTask` 还提供可选的 `configure(config: dict)` 钩子。H5 路径驱动的任务通常保存完整 resolved config 后自行读取输入；`classification` 则由入口加载 train / val embedding 后调用任务。

`embeddings_dataset` 字段：

| 字段 | 类型 | 说明 |
|------|------|------|
| `video_id` | `list[str]` | 视频标识符（H5 key） |
| `embeddings` | `list[np.ndarray]` | 每个元素形状 `[T_i, 128]` |
| `target_steps` | `list[np.ndarray]` | 原始帧索引，形状 `[T_i]` |
| `seq_len` | `list[int]` | 视频原始总帧数（用于截断） |
| `action_id` | `list[int]` | 动作类别 ID |
| `phase_labels` | `list[np.ndarray | None]` | phase 标签（若输入 H5 含标注） |
| `keyframe_labels` | `list[np.ndarray | None]` | keyframe 标签（若输入 H5 含标注） |
| `labeled` | `list[bool]` | 该视频是否带人工标注 |

### 已实现任务

#### Kendall's Tau（`task_kendall.py`）

**目的**：度量 TCC 嵌入空间对视频时序结构的保留程度。

**算法步骤**（`_compute_kendalls_tau`）：

```
输入: embs_list  (N 个视频的嵌入序列)
      stride     (下采样间隔，默认 1，当前配置 5)
      distance   (距离度量，默认 sqeuclidean)

对所有有序对 (i, j)，i ≠ j：
  query_feats     = embs_list[i][::stride]     # [T_q, 128]
  candidate_feats = embs_list[j][::stride]     # [T_c, 128]
  dists           = cdist(query_feats, candidate_feats, metric=distance)  # [T_q, T_c]
  nns             = argmin(dists, axis=1)       # 每个 query 帧在 candidate 中的最近邻索引
  tau_ij          = kendalltau(arange(T_q), nns).correlation

mean_tau = mean(所有非对角非 NaN 的 tau_ij)
```

**返回值**：

```python
{
  "task_name":    "kendalls_tau",
  "metric_name":  "kendalls_tau",
  "metric_value": float,   # ∈ [-1, 1]，越接近 1 表示时序保序越好
}
```

**附加输出**：
- 终端打印 N×N pairwise tau 矩阵（含每行均值）
- 保存热图 PNG：`outputs/kendall_heatmap/{run_name}/{h5_stem}/kendall_heatmap_{timestamp}.png`

**主要超参数**（`configs_v2/eval/kendalls_tau.yaml`）：

| 参数 | 当前值 | 含义 |
|------|--------|------|
| `kendall_stride` | 5 | 序列下采样步长，减小内存与计算量 |
| `kendall_distance` | sqeuclidean | 最近邻距离度量 |

#### Phase 分类（`task_phase_classification.py` / `task_name=classification`）

用冻结 encoder 嵌入训练线性 SVM，在带标注的验证视频上评估 phase 分类准确率，并输出混淆矩阵热图至 `outputs/confusion_matrix/`。

当 `gen_tsne_phase_label: true` 时，还会额外生成 `embd_tsne_phase_label.h5`，供 phase t-SNE 可视化脚本直接读取。

#### Expert Projection（`task_expert_projection.py` / `task_name=expert_projection`）

将一条 expert 轨迹（通常是 `mean_path` 变体）作为参照，对 non-expert 嵌入序列执行 soft nearest-neighbour 投影。

- 输入：expert mean-path H5 + non-expert embedding H5 + 非专家数据集 registry 信息
- 输出：`outputs/expert_projection/{expert_stem}/{nonexpert_stem}/expert_projection-{timestamp}.h5`
- 返回指标：`global_mean_hard_nn_distance`
- 可选附加工件：对齐曲线 PNG、t-SNE 可视化、同步原视频 MP4

#### Latent Distance Heatmap（`task_latent_distance_heatmap.py` / `task_name=latent_distance_heatmap`）

对单条或全部嵌入轨迹计算帧间 `T×T` 欧氏距离矩阵，可选输出热图、anchor-distance 曲线，并基于原始 L2 距离计算 VOC（Velocity Ordering Consistency, Spearman）指标。

- 输入：单个 embedding H5 + `selected_video_index`（整数或 `"all"`）
- 主要配置：`plot_mode=heatmap|anchor_distance_curves|both`、`normalize_embeddings`、`convert_to_similarity`、`similarity_tau`
- 输出目录：`outputs/latent_distance_heatmap/{run_name}/{h5_stem}/...`
- 返回指标：`voc_spearman`；在 `"all"` 模式下还会返回 valid video 数与 skipped video IDs
- 备注：`convert_to_similarity=true` 只影响可视化矩阵/相似度统计，VOC 始终基于原始 L2 距离计算

#### Gaussian Progress（`task_gaussian_progress_fitting.py` / `task_gaussian_progress_pred.py`）

两阶段进度建模流程：

- `gaussian_progress_fitting`：按归一化时序进度把多条 expert embedding 分箱，为每个 bin 拟合均值和 full covariance；支持 independent、shared 及加权混合 covariance，输出模型 H5 和可选 t-SNE 诊断图。
- `gaussian_progress_pred`：只读取已保存模型和 non-expert embedding，计算各进度 bin 的高斯后验，输出逐帧 progress mean / variance、entropy、MAP bin 与 Mahalanobis 诊断。
- 默认输出分别位于 `outputs/gaussian_progress_fitting/` 与 `outputs/gaussian_progress_pred/`。

#### Activation Map（`task_activation_map.py` / `task_name=activation_map`）

从 processed video H5 读取完整轨迹，加载 encoder checkpoint，并通过 forward hook 获取 ResNet50 Conv4c 与第二层 Conv3D 激活。输出逐帧 overlay PNG、两路 MP4 和原始 heatmap H5；该任务是当前评估系统中唯一需要重跑 encoder 的任务。

### 待实现任务（`build_task` 工厂已预留）

| 任务名 | 文件（待创建） | 说明 |
|--------|--------------|------|
| `event_completion` | `tcc_eval_tasks/task_event_completion.py` | 预测动作完成百分比 |
| `few_shot_classification` | `tcc_eval_tasks/task_few_shot.py` | 少样本动作识别 |

---

## 四-C、平均嵌入路径（`scripts/compute_mean_embedding_path.py`）

对嵌入 H5 中所有视频做时间归一化后插值到统一长度 K，取算术平均得到 `[K, 128]` 的平均路径。输出新 H5 文件（单条 `video_id="mean"` 记录，含路径及分散度 / 累积距离诊断），并保存诊断图至 `outputs/mean_path/`。

---

## 四-D、t-SNE 嵌入可视化（`scripts/tsne/visualize_embeddings_tsne*.py`）

多个脚本当前默认读取 `configs_v2/visualize/*.yaml`（由 `ConfigV2.load_visualize()` 合并 `base.yaml` 与 per-flow YAML），无需重跑 encoder：

| 脚本 | 用途 |
|------|------|
| `visualize_embeddings_tsne.py` | 单组嵌入 t-SNE，每帧按时序进度着色 |
| `visualize_embeddings_tsne_2groups.py` | 双组（如训练 vs 验证）拼接后联合降维，共享 2D 空间对比 |
| `visualize_embeddings_tsne_phase.py` | 读取 `embd_tsne_phase_label.h5`，按 phase 标签和标签来源着色 |
| `visualize_embeddings_tsne_gap_analysis.py` | t-SNE 散点与原始视频联动，输出 gap-analysis MP4 |

**路径解析**（由 `ConfigV2.load_visualize()` 完成）：
```
1. per-flow YAML :: 显式 embedding_h5_path / output_dir / val_labeled_h5_path
2. embedding_ref / embedding_ref_group2 / val_labeled_embedding_ref
3. registry/*.yaml + project.yaml  →  自动派生 embedding 路径与输出目录
```

**主要超参数**（`configs_v2/visualize/base.yaml` + per-flow YAML）：

| 参数 | 说明 |
|------|------|
| `embedding_ref` | 第一组 embedding registry 别名 |
| `embedding_ref_group2` | 第二组 embedding registry 别名（仅双组 / gap-analysis 使用） |
| `val_labeled_embedding_ref` | 带标注验证集 embedding 别名（phase t-SNE 使用） |
| `max_frames_per_video` | 每个视频最多采样帧数（默认 300） |
| `use_pca_before_tsne` | 是否先 PCA 降至 `pca_dim` 维再做 t-SNE |
| `standardize` | 是否对嵌入做 Z-score 标准化 |

**输出**：PNG / MP4 工件统一保存至 `outputs/tsne/{run_name}/...`

---

## 四-E、辅助脚本分层

为避免目录树把 `scripts/` 展开成超长列表，实际脚本可按功能理解为下列几组：

| 类别 | 脚本 | 作用 |
|------|------|------|
| 嵌入分析 | `compute_mean_embedding_path.py`、`tsne/visualize_embeddings_tsne*.py`、`backfill_latent_distance.py` | 平均路径、t-SNE 与历史 latent-distance 工件维护 |
| Robomimic 数据 / 奖励 | `robomimic/make_robomimic_video.py`、`robomimic/make_robomimic_dense_reward.py`、`robomimic/policy_data_preproc/extract_image_features.py` | 视频导出、progress→dense/PBRS reward、ResNet18 特征预计算 |
| 策略实验 | `robomimic/iql/*.py`、`run_online_sac_reward_sweep.py` | IQL 规模训练 / 基准 / 曲线与 Online SAC 奖励 sweep |
| 编码器实验 | `train_square_triplet_weight_sweep.py` | Square 任务 composite loss 权重 sweep |
| 测试 / 辅助 | `tests/test_gaussian_progress*.py`、`_smoke_test_in_training_eval.py`、`tmp_rerun_expert_projection_visuals.py`、`wait_for_pid_then_train.py` | Gaussian progress 回归、训练中评估烟测与任务串行启动 |

---

## 五、数据层（`dataset_preparation/`）

### `H5VideoDataset`（h5vid_dataset.py）
- 从 HDF5 文件读取视频帧序列
- 为每个目标时步构建**因果时序上下文窗口**（不含未来帧）
- 支持两种采样策略：
  - `offset_uniform`：从 `[random_offset, seq_len-1]` 均匀采样
  - `stride`：随机偏移 + 固定步长
- `sample_all=True` 模式：导出所有帧（供 `extract_embeddings.py` 使用）

**输出 batch 结构**：
```python
{
  "frames":       [B, clip_len, context_size, 3, 224, 224],
  "target_steps": [B, clip_len],   # 目标帧的时步索引
  "seq_len":      [B],             # 视频总帧数
  "action_id":    [B],             # 动作类别 ID
  "video_id":     [B]              # 视频 ID（字符串）
}
```

### `mp4vid_to_png.py`
将 MP4 视频逐帧提取为 PNG，专门用于人工标注关键帧。

```
python dataset_preparation/mp4vid_to_png.py <task_folder> [--fps N] [--size 224]
```

- 在脚本顶部的 `VIDEO_NAMES` 列表中指定要提取的视频（留空则处理全部）
- 输出路径：`datasets/raw_img/{video_stem}/frame_{idx:06d}.png`
- 参数 `--fps`：下采样帧率（默认不下采样，提取所有帧）
- 参数 `--size`：输出图像边长（默认 224px）

### `add_phase_labels.py`
读取人工标注的关键帧 CSV，为嵌入 H5 文件生成并写入 `phase_labels` 和 `keyframe_labels`，输出带 `-labeled` 后缀的新 H5 文件（不修改原始文件）。

```
python dataset_preparation/add_phase_labels.py \
    --embd_h5  datasets/embeddings/.../pouring-2vid-embd.h5 \
    --keyframes_csv  "datasets/phase_labels/pouring_train56_phase_labels.csv"
```

**功能说明**：
- 自动缩放：若 CSV 中 `last_key_frame_idx ≠ max(target_steps)`，按比例缩放所有关键帧索引后取整
- 支持两种 CSV 格式：新格式（`name, id, key_frame_idx`）和旧格式（`video_id, key_frame_idx`）
- H5 查找顺序：先按 `id`（零填充数字 ID），再按 `name`（视频文件名）
- 对 H5 中所有视频写入 `attrs["labeled"]`（有 CSV 条目为 `True`，否则为 `False`）
- 输出文件：`<同目录>/<stem>-labeled.h5`

---

## 六、配置系统（`configs_v2/`）

### `configs_v2/`（当前主流程）

| 文件 / 目录 | 控制内容 |
|------|---------|
| `project.yaml` | 项目目录布局与 checkpoint / embedding 命名模板 |
| `registry/datasets.yaml` | dataset alias → `processed_h5`、`raw_dir`、`robomimic_hdf5`、`mask_key`、`phase_labels_csv` |
| `registry/runs.yaml` | run alias → `run_name`、`checkpoint_epoch`；embedding alias → `run_ref`、`dataset_ref`、`variant` |
| `data_process.yaml` | `mp4vid_to_h5data.py` 的处理阶段配置 |
| `train.yaml` | `train_encoder.py` 的训练阶段配置；含采样参数、backbone cache、`loss_name` / `loss_config`、`in_training_eval` |
| `extract.yaml` | `extract_embeddings.py` 的提取阶段配置；解析 `checkpoint_ref` / `extract_dataset` |
| `loss/*.yaml` | `loss_tcc.yaml` / `loss_temporal_infonce.yaml` / `loss_temporal_triplet.yaml` / `loss_composite_*.yaml` 等损失超参数 |
| `eval/*.yaml` | Kendall、classification、expert projection、latent distance、Gaussian progress、activation map 配置 |
| `visualize/base.yaml` + `visualize/*.yaml` | t-SNE / phase / gap-analysis / mean-path 等可视化配置 |

**运行时角色**：
- `utils/config_v2.py` 提供 `load_data_process()` / `load_train()` / `load_extract()` / `load_eval()` / `load_visualize()`，负责把 ref 解析成绝对路径。
- `utils/in_training_eval.py` 在训练中复用 `evaluate_encoder.run_eval_task()` 执行 checkpoint 后评估，并负责 wandb 标量/图像日志。
- `utils/registry_v2.py` 负责 append-only 注册表写入；`utils/registry_scan.py` 负责磁盘工件与 registry 的自动对账。
- `policy_training/configs/` 是独立的策略配置域，不经 ConfigV2；由 `PolicyTrainingConfig` 读取 `iql.yaml`、`online_sac.yaml` 和 `evaluate_policy.yaml`。

---

## 七、关键数据格式

**训练集 H5 格式**（`datasets/processed/`）：
```
/videos/<video_id>/
    frames      [T, H, W, C]  uint8
    attrs: seq_len, action_id
```

**嵌入 H5 格式**（`datasets/embeddings/`，由 `extract_embeddings.py` 生成）：
```
/videos/<video_id>/
    embeddings      [T_out, 128]  float32
    target_steps    [T_out]       int64
    attrs: seq_len, action_id
```

**平均路径 H5 格式**（`*-embd-mean_path.h5`，由 `compute_mean_embedding_path.py` 生成）：
```
/videos/mean/
    embeddings      [K, 128]   float32   # K = round(mean(T_i))，时间归一化插值后均值
    target_steps    [K]        int64     # 0, 1, ..., K-1
    attrs: seq_len=K, action_id=-1
```

**带标注的嵌入 H5 格式**（`datasets/embeddings/`，由 `add_phase_labels.py` 生成，文件名以 `-labeled.h5` 结尾）：
```
/videos/<video_id>/
    embeddings       [T_out, 128]  float32   # 不变
    target_steps     [T_out]       int64     # 不变
    phase_labels     [T_out]       int64     # 仅 labeled=True 的视频有此字段
    keyframe_labels  [T_out]       int64     # 仅 labeled=True 的视频有此字段
    attrs:
        seq_len      int
        action_id    int
        labeled      bool   # True = 已人工标注；False = 无标注
```

`phase_labels` 取值规则：
- 排序后关键帧为 `e0 < e1 < ... < eK`，共产生 `K` 个 phase（0 到 K-1）
- `target_steps[t] ∈ [e_i, e_{i+1})` → `phase_labels[t] = i`（最后区间右侧闭合）
- 超出 `[e0, eK]` 范围 → `-1`

`keyframe_labels` 取值规则：
- `target_steps[t] == e_i` → `keyframe_labels[t] = i`
- 非关键帧 → `-1`

使用带标注数据的示例：
```python
with h5py.File("pouring-2vid-embd-labeled.h5") as f:
    for vid_id, grp in f["videos"].items():
        if grp.attrs.get("labeled", False):
            ph = grp["phase_labels"][:]      # [T_out] int64
            kf = grp["keyframe_labels"][:]   # [T_out] int64
```

**t-SNE phase-label H5 格式**（`datasets/embeddings/{run_name}/embd_tsne_phase_label.h5`，由 `evaluate_encoder.py` 分类任务 `gen_tsne_phase_label: true` 时生成）：
```
/videos/<video_id>/          # train 视频在前；val 视频在后，ID 冲突时键名加 "val_" 前缀
    embeddings   [T, 128]  float32
    phase_labels [T]       int64   # train labeled → 原始标注；其余 → SVM 预测值
    attrs:
        data_type        str   # "train" 或 "val"
        is_ground_truth  bool  # True = 来自人工标注；False = SVM 预测
```

**Gaussian progress 模型与预测 H5**：
```
# fitting 输出
/model/
    bin_progress_values       [K]
    bin_means                 [K, D]
    bin_final_covariances     [K, D, D]
    bin_log_determinants      [K]
    bin_counts                [K]

# prediction 输出
/nonexperts/<video_id>/
    target_steps, progress_label, progress_variance
    posterior_entropy, map_bin, min_mahalanobis_sq
    posterior                [T, K]   # save_posterior=true 时
```

---

## 八、实验跟踪

- 编码器训练使用 W&B 记录损失、checkpoint 后评估指标与可选图像；日志目录为根目录 `wandb/`。
- 编码器 Run 命名规则：`{LOSS_TAG}-{dataset}-{backbone}-{train_base}-{timestamp}`。
- 策略训练使用独立项目与目录 `policy_training/wandb/`，run metadata 由 env、task、algorithm、mask、reward type、seed 和 timestamp 推导。
- 两条训练线分别把权重写到根目录 `checkpoint/` 与 `policy_training/checkpoint/`，不共享 checkpoint schema。

---

## 九、datasets 目录详细结构

```
datasets/
├── raw/                          # 原始视频文件（MP4/MOV），按数据集子集组织
│   ├── pouring/                  # 小子集（2个动作，仅 view_0），共 2 个 MP4
│   ├── pouring_all/              # 原始完整集（多动作，双视角），共 70 个 MP4，35 动作 × 2 视角
│   ├── pouring_all_val/          # 原始验证集，共 14 个 MOV，7 动作 × 2 视角
│   ├── pouring_train56/          # 新划分训练集，共 56 个 MP4，28 动作 × 2 视角
│   ├── pouring_val14/            # 新划分验证集，共 14 个 MP4，7 动作 × 2 视角
│   ├── robomimic_can_ph/         # Robomimic Can PH 任务视频
│   ├── robomimic_lift_ph/        # Robomimic Lift PH 任务视频
│   ├── robomimic_square_ph/      # Robomimic Square PH 任务视频
│   ├── robomimic_transport_ph/   # Robomimic Transport PH 任务视频
│   └── robomimic_{can,lift,square,transport}_mh/  # Mixed-Human 任务视频
│
├── raw_img/                      # mp4vid_to_png.py 提取的帧图像（供人工标注用）
│   └── {video_stem}/             # 每个视频一个子目录
│       └── frame_{idx:06d}.png   # 逐帧 PNG 图像
│
├── phase_labels/                 # 人工标注的关键帧索引（CSV，按 train/val 分拆）
│   ├── pouring_train56_phase_labels.csv
│   └── pouring_val14_phase_labels.csv
│
├── processed/                    # mp4vid_to_h5data.py 转换后的 H5 数据集
│   ├── pouring-2vid.h5           # 2 个视频的小型调试集
│   ├── pouring-4vid.h5           # 4 个视频的小型调试集
│   ├── pouring_all_training-70vid.h5   # 70 个视频的完整集
│   ├── pouring_all_val-14vid.h5        # 14 个视频验证集（MOV 来源）
│   ├── pouring_train56-56vid_fullframe.h5  # 56 个视频的新划分训练集
│   ├── pouring_val14-14vid_fullframe.h5    # 14 个视频的新划分验证集
│   ├── robomimic_{task}_ph-{N}vid_train.h5  # Robomimic 任务训练集
│   ├── robomimic_{task}_ph-{N}vid_valid.h5  # Robomimic 任务验证集
│   └── robomimic_{task}_mh-{N}vid_{quality}.h5  # Mixed-Human 子集
│
└── embeddings/                   # extract_embeddings.py 输出的嵌入 H5 文件
    ├── pouring-2vid-embd.h5      # 早期调试用的 2 视频嵌入
    └── {run_name}/               # 每次 run 独立子目录
        ├── {dataset_stem}-embd.h5              # 原始嵌入（无标注）
        ├── {dataset_stem}-embd-labeled.h5      # 带 phase/keyframe 标注的嵌入副本
        ├── {dataset_stem}-embd-mean_path.h5    # 平均嵌入路径（由 compute_mean_embedding_path.py 生成）
        └── embd_tsne_phase_label.h5            # 分类评估生成，含 train+val 全量嵌入与 phase 标签
```

### 命名规范

**原始视频文件名格式**：
```
{action_name}_real_view_{view_id}.mp4    # MP4 格式（下划线分隔 view id）
{action_name}_real_view{view_id}.mov     # MOV 格式（无下划线）
```
- `action_name`：描述倒水动作的语义名，如 `milk_to_white`、`pom_to_clear99`
- `view_id`：摄像机视角编号（`0` 或 `1`），同一动作通常有两个视角

**H5 文件命名格式**：
```
{dataset_folder_name}-{N}vid.h5          # processed/
{dataset_folder_name}-{N}vid-embd.h5     # embeddings/
```

### 数据处理流水线

```
raw/*.mp4 / *.mov
    ↓  mp4vid_to_h5data.py
    ↓  参数: IMAGE_SIZE=224, TARGET_FPS=10, COMPRESSION=None, CHUNK_LEN=8
    ↓  操作: 提取帧 → resize 到 224×224 → 按 action_name 分配 action_id
processed/{dataset}-{N}vid.h5
    ↓  H5VideoDataset (h5vid_dataset.py)
    ↓  操作: 随机采样 clip_len 帧 + 构建因果上下文窗口
    ↓  (DataLoader, batch_size=2)
train_encoder.py → TCCEncoder → build_loss(loss_name) → checkpoint/
    ↓  extract_embeddings.py (sample_all=True, batch_size=1)
embeddings/{run_name}/{dataset}-embd.h5
    ↓  add_phase_labels.py  --embd_h5 ...  --keyframes_csv ...
embeddings/{run_name}/{dataset}-embd-labeled.h5   （含 phase/keyframe 标注）
    ↓  evaluate_encoder.py / scripts/tsne/visualize_embeddings_tsne*.py
Kendall / Classification / Expert Projection / Gaussian Progress /
Latent Distance Heatmap(VOC) / Activation Map / t-SNE
```

人工标注辅助流程与 H5 内部 schema 已在上文第二节、第七节说明，这里不再重复展开。

### 当前数据集规模

| 子集 | 文件 | 视频数 | 动作数 | 格式 | 用途 |
|------|------|--------|--------|------|------|
| pouring (小) | `pouring-2vid.h5` | 2 | 1 | MP4 | 调试 |
| pouring (小) | `pouring-4vid.h5` | 4 | 2 | MP4 | 调试 |
| pouring_all | `pouring_all_training-70vid.h5` | 70 | 35 | MP4 | 原始完整集 |
| pouring_all_val | `pouring_all_val-14vid.h5` | 14 | 7 | MOV | 原始验证集 |
| pouring_train56 | `pouring_train56-56vid_fullframe.h5` | 56 | 28 | MP4 | 新划分训练集 |
| pouring_val14 | `pouring_val14-14vid_fullframe.h5` | 14 | 7 | MP4 | 新划分验证集 |
| robomimic_square_ph | `robomimic_square_ph-{N}vid_{split}.h5` | 36/90/180 (train), 4/10/20 (valid) | — | MP4 | Robomimic Square PH 任务 |
| robomimic_can_ph | `robomimic_can_ph-{N}vid_{split}.h5` | 36/180 (train), 4/20 (valid) | — | MP4 | Robomimic Can PH 任务 |
| robomimic_lift_ph | `robomimic_lift_ph-{N}vid_{split}.h5` | 36/180 (train), 4/20 (valid) | — | MP4 | Robomimic Lift PH 任务 |
| robomimic_transport_ph | `robomimic_transport_ph-{N}vid_{split}.h5` | 36 (train), 4 (valid) | — | MP4 | Robomimic Transport PH 任务 |
| robomimic MH | `robomimic_{task}_mh-{N}vid_{quality}.h5` | Can 100/300、Lift 100、Square 100、Transport 50 | — | MP4 | okay / worse / all 质量子集 |

### 当前人工标注状态

标注文件：`datasets/phase_labels/`（按 train/val 分拆为两个 CSV）

| 视频名 | 数字 ID | 标注关键帧数 | Phase 数 |
|--------|---------|------------|----------|
| clearsoda_to_white_real_view_0 | 000003 | 6 | 5 |
| clearsoda_to_white_real_view_1 | 000004 | 6 | 5 |
| milk_to_white_real_view_0 | 000019 | 6 | 5 |
| milk_to_white_real_view_1 | 000020 | 6 | 5 |
| pom_to_clear_real_view_0 | 000023 | 6 | 5 |
| pom_to_clear_real_view_1 | 000024 | 6 | 5 |

CSV 列说明：
- `name`：视频文件名（去扩展名），与 `processed H5` 中的 `video_id` 对应
- `id`：零填充六位数字 ID，与 `pouring_train56-56vid_fullframe.h5` 中按字母序分配的 key 对应
- `key_frame_idx`：该关键事件帧在原始视频中的帧编号（10fps 下采样后）

---

## 十、策略训练子系统（`policy_training/`）

`policy_training` 与根目录的 encoder 训练相互独立，但通过进度标签、mean-path H5 和 encoder checkpoint 对接。更细的离线 IQL 数据契约与 checkpoint 字段见 `policy_training/PROJECT_ARCHITECTURE.md`。

### 训练与算法

```
Robomimic HDF5
  ├── IQL
  │     RobomimicReplayBuffer → IQL.train_step()
  │     → actor + Q ensemble + target Q + V(s)
  └── Online SAC
        Robomimic env → OnlineDataCollector → OnlineReplayBuffer
        → OnlineSAC.train_step()
        → sparse_done / dense / PBRS reward
```

- `train_policy.py` 是统一入口，`algos.build_algo()` 当前注册 `iql` 与 `online_sac`。
- IQL 使用 expectile V 回归和 AWR（或 DDPG+BC）actor 更新；可在训练中执行 rollout eval 与 dataset value eval。
- Online SAC 复用自定义 actor / critic 与特征提取器，按环境步采样并训练；当前仅支持 `action_chunk_size=1`，offline warmstart 与 mixed buffer 仍为预留项。
- `models/feature_extractors.py` 支持普通 flat range，以及“low-dim + 预计算 ResNet18 feature map + SpatialSoftmax”路径。

### TCC 在线奖励桥接

`reward_model/tcc_expert_proj_reward.py::TCCExpertProjectionDenseRewardProvider` 加载根目录的 TCC encoder checkpoint 与 expert mean-path H5，对在线 RGB observation 编码并软投影到 expert 进度轴。Online SAC 可选择：

- `sparse_done`：环境稀疏成功奖励；
- `dense`：稀疏奖励加当前 progress；
- `pbrs`：`sparse + gamma * progress_next - progress_current`。

离线数据侧由 `scripts/robomimic/make_robomimic_dense_reward.py` 把 Expert Projection progress 写成 original / dense / PBRS 三类 Robomimic HDF5；`scripts/robomimic/policy_data_preproc/extract_image_features.py` 再按需预计算冻结 ResNet18Conv 特征。

### 策略评估与 checkpoint

- `evaluate_policy.py` 从 self-describing checkpoint 重建算法、观测适配器和 Robosuite 环境，支持多 seed、串行或多进程 rollout。
- `TrainingRolloutEvaluator` 供 IQL 与 Online SAC 在训练中周期评估；并行 worker 可独立使用 CPU 或 CUDA。
- 评估汇总 Return、Horizon、Success Rate，输出 JSON，并可保存或上传 rollout MP4、progress 曲线。
- checkpoint schema v1 保存 config、环境/shape metadata、obs slices、modules、optimizers 与可选 normalization stats；离线与在线算法共用该 schema。

---

## 十一、Robomimic 依赖边界

- `third_party/robomimic/` 是 Git submodule，保留完整 Robomimic fork、算法模板与工具，主要用于上游参考和数据流程。
- `policy_training/envs/robomimic_runtime/` 是策略子系统使用的 vendored runtime，提供环境、观测网络与工具依赖。
- 项目实际调用的 IQL / Online SAC 实现在 `policy_training/algos/`；不直接复用 `third_party/robomimic/robomimic/algo/` 中的训练实现。
