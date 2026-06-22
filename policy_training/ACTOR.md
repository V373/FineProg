# IQL Actor Architecture and Tensor Shapes

This note summarizes the current IQL actor under
`/home/user/zhangzk/projects/fineprog/policy_training`, with emphasis on how
state / observation tensors become action predictions.

## Scope

- Training entry: `train_policy.py`
- Default config: `configs/iql.yaml`
- Algorithm: `algos/offline_rl/iql.py`
- Actor / policy modules: `models/policies.py`
- Feature extractors: `models/feature_extractors.py`
- Spatial softmax: `models/model_utils.py`
- Replay buffer flattening: `datasets/robomimic/replay_buffer.py`
- Eval-time observation adapter: `utils/eval_utils.py`

The active config uses `features_extractor_type: resnet18conv`. In training,
the image observation keys are not raw RGB images; they are precomputed
ResNet18Conv feature maps stored in the HDF5 file.

## Current Data Shapes

Configured dataset:

```text
datasets/robomimic/can/mh/reward_labeled/resnet18feats/image_2view_v15_reward_labeled_PBRS_resnet18feats.hdf5
filter_key: IQL_expert_worse
```

For the verified first selected demo (`demo_0`), `actions` has shape
`(118, 7)`, so:

\[
d_a = 7.
\]

The observation keys are flattened and concatenated in this order:

| key | per-step shape | flat dim | flat slice |
|---|---:|---:|---:|
| `robot0_eef_pos` | `(3,)` | `3` | `[0:3]` |
| `robot0_eef_quat` | `(4,)` | `4` | `[3:7]` |
| `robot0_gripper_qpos` | `(2,)` | `2` | `[7:9]` |
| `agentview_image` | `(512, 3, 3)` | `4608` | `[9:4617]` |
| `robot0_eye_in_hand_image` | `(512, 3, 3)` | `4608` | `[4617:9225]` |

Thus one flat observation is:

\[
o_t =
\left[
p_t,\ q_t,\ g_t,\ \operatorname{vec}(F_t^{agent}),\
\operatorname{vec}(F_t^{eye})
\right]
\in \mathbb{R}^{9225},
\]

where:

\[
p_t \in \mathbb{R}^{3},\quad
q_t \in \mathbb{R}^{4},\quad
g_t \in \mathbb{R}^{2},
\]

\[
F_t^{agent}, F_t^{eye} \in \mathbb{R}^{512 \times 3 \times 3}.
\]

The default training batch size is `512`, so sampled actor inputs have shape:

\[
O \in \mathbb{R}^{512 \times 9225},\quad
A \in \mathbb{R}^{512 \times 7}.
\]

Current config has:

```yaml
normalize_obs: false
normalize_actions: false
```

Therefore the actor consumes the flat observations as loaded from the HDF5
buffer, and predicted actions are not unnormalized after inference.

## Observation Flattening

`RobomimicReplayBuffer` reads each key from `data/{demo}/obs/{key}` and
`data/{demo}/next_obs/{key}`.

If an observation has rank greater than 2, it is flattened per time step:

\[
X \in \mathbb{R}^{T \times C \times H \times W}
\rightarrow
\operatorname{flat}(X) \in \mathbb{R}^{T \times (C H W)}.
\]

All configured obs keys are concatenated along feature dimension:

\[
O =
\operatorname{concat}
\left[
O^{eef\_pos}, O^{eef\_quat}, O^{gripper},
O^{agent}, O^{eye}
\right]
\in \mathbb{R}^{T \times 9225}.
\]

The resulting `observation_space` is:

\[
\mathcal{O} = \mathbb{R}^{9225}.
\]

The action is read from `data/{demo}/actions` because `action_keys: null`.
The resulting `action_space` has shape:

\[
\mathcal{A} = \mathbb{R}^{7}.
\]

## Actor Feature Extractor

The actor uses `resnet18convFeaturesExtractor`. It receives the flat
observation:

\[
x \in \mathbb{R}^{B \times 9225}.
\]

It slices the low-dimensional state components:

\[
x_{low} =
\left[
x_{0:3}, x_{3:7}, x_{7:9}
\right]
\in \mathbb{R}^{B \times 9}.
\]

For each visual key, it restores the flattened feature map:

\[
x_{agent}[:, 9:4617]
\rightarrow
F^{agent} \in \mathbb{R}^{B \times 512 \times 3 \times 3},
\]

\[
x_{eye}[:, 4617:9225]
\rightarrow
F^{eye} \in \mathbb{R}^{B \times 512 \times 3 \times 3}.
\]

Each camera branch applies:

\[
F^{cam}
\xrightarrow{\operatorname{Conv2d}(512,32,1)}
Z^{cam} \in \mathbb{R}^{B \times 32 \times 3 \times 3}.
\]

For each keypoint channel \(k\), SpatialSoftmax computes an attention
distribution over the \(3 \times 3\) spatial grid:

\[
\alpha_{k,u,v}
=
\frac{\exp(Z_{k,u,v}/\tau)}
{\sum_{u',v'} \exp(Z_{k,u',v'}/\tau)},
\quad \tau = 1.0.
\]

It then outputs expected grid coordinates:

\[
\kappa_k =
\left[
\sum_{u,v} \alpha_{k,u,v} X_{u,v},
\sum_{u,v} \alpha_{k,u,v} Y_{u,v}
\right]
\in \mathbb{R}^{2}.
\]

For `num_kp = 32`:

\[
\operatorname{SpatialSoftmax}(Z^{cam})
\in \mathbb{R}^{B \times 32 \times 2}.
\]

Then:

\[
\mathbb{R}^{B \times 32 \times 2}
\xrightarrow{\operatorname{Flatten}}
\mathbb{R}^{B \times 64}
\xrightarrow{\operatorname{Linear}(64,64)}
v^{cam} \in \mathbb{R}^{B \times 64}.
\]

There are two camera branches, so the final feature vector is:

\[
\phi(x)
=
\left[
x_{low},\ v^{agent},\ v^{eye}
\right]
\in
\mathbb{R}^{B \times (9 + 64 + 64)}
=
\mathbb{R}^{B \times 137}.
\]

Thus the actor MLP input dimension is:

\[
d_{\phi} = 137.
\]

## Actor MLP

The policy architecture in `configs/iql.yaml` is:

```yaml
pi_net_arch: [512, 400, 256]
policy_layer_norm: true
```

The actor uses ReLU activations and LayerNorm after every hidden linear layer.
For a single observation:

\[
z_0 = \phi(o_t) \in \mathbb{R}^{137}.
\]

Layer 1:

\[
h_1 =
\operatorname{ReLU}
\left(
\operatorname{LayerNorm}
\left(
W_1 z_0 + b_1
\right)
\right),
\quad
W_1 \in \mathbb{R}^{512 \times 137},
\quad
h_1 \in \mathbb{R}^{512}.
\]

Layer 2:

\[
h_2 =
\operatorname{ReLU}
\left(
\operatorname{LayerNorm}
\left(
W_2 h_1 + b_2
\right)
\right),
\quad
W_2 \in \mathbb{R}^{400 \times 512},
\quad
h_2 \in \mathbb{R}^{400}.
\]

Layer 3:

\[
h_3 =
\operatorname{ReLU}
\left(
\operatorname{LayerNorm}
\left(
W_3 h_2 + b_3
\right)
\right),
\quad
W_3 \in \mathbb{R}^{256 \times 400},
\quad
h_3 \in \mathbb{R}^{256}.
\]

The actor has two output heads:

\[
\mu_t = W_{\mu} h_3 + b_{\mu}
\in \mathbb{R}^{7},
\quad
W_{\mu} \in \mathbb{R}^{7 \times 256}.
\]

\[
\ell_t = W_{\sigma} h_3 + b_{\sigma}
\in \mathbb{R}^{7},
\quad
W_{\sigma} \in \mathbb{R}^{7 \times 256}.
\]

SB3 clamps the log standard deviation:

\[
\log \sigma_t
=
\operatorname{clip}(\ell_t, -20, 2)
\in \mathbb{R}^{7}.
\]

The actor parameterizes a squashed diagonal Gaussian:

\[
z_t \sim
\mathcal{N}
\left(
\mu_t,\ \operatorname{diag}(\sigma_t^2)
\right),
\quad
a_t = \tanh(z_t).
\]

So:

\[
a_t \in [-1, 1]^7.
\]

## Deterministic and Stochastic Inference

During evaluation, `SB3IQLRolloutPolicy.__call__` runs:

```python
mean_actions, log_std, _ = actor.get_action_dist_params(obs_tensor)
actions = actor.action_dist.actions_from_params(
    mean_actions,
    log_std,
    deterministic=not stochastic,
)
```

Default eval config has `stochastic: false`, so inference uses the mode of the
squashed Gaussian:

\[
\hat{a}_t = \tanh(\mu_t).
\]

If stochastic evaluation is enabled:

\[
\epsilon_t \sim \mathcal{N}(0, I),
\quad
z_t = \mu_t + \sigma_t \odot \epsilon_t,
\quad
\hat{a}_t = \tanh(z_t).
\]

Because `normalize_actions: false`, the action sent to the environment is:

\[
a_t^{env} = \hat{a}_t.
\]

If action normalization were enabled, eval would instead apply:

\[
a_t^{env} =
\hat{a}_t \odot \operatorname{scale}_a + \operatorname{offset}_a.
\]

## Eval-Time Raw Env Observation Path

Training uses precomputed ResNet18Conv feature maps from HDF5. Evaluation may
receive raw env observations. `ObservationAdapter` converts the env obs dict
back into the checkpoint's flat observation layout.

For low-dimensional keys:

\[
o^{key} \rightarrow \operatorname{reshape}(-1).
\]

For visual keys:

- If the env value already has shape `(512, 3, 3)`, it is flattened directly.
- If the env value is a raw image, it is passed through
  `FrozenPretrainedResNet18Conv`, producing the expected `(512, 3, 3)` feature
  map, then flattened.

The adapter then concatenates keys in `dataset.obs_keys` order and checks:

\[
\dim(\operatorname{flat\_obs}) = 9225.
\]

The actor always receives:

\[
o_t \in \mathbb{R}^{9225},
\]

regardless of whether the visual features came from HDF5 training data or
eval-time image encoding.

## Actor Training Objective in Current IQL

The active policy extraction mode is:

```yaml
policy_extraction: awr
advantage_temp: 5.0
clip_score: 100.0
```

IQL computes:

\[
A(s,a) =
\min_j Q_{\bar{\theta}_j}(s,a) - V_{\psi}(s).
\]

The AWR weight is:

\[
w(s,a) =
\operatorname{clip}
\left(
\exp(5.0 \cdot A(s,a)),\ 0,\ 100
\right).
\]

The actor loss is:

\[
\mathcal{L}_{actor}
=
-
\mathbb{E}_{(s,a)\sim D}
\left[
w(s,a)\log \pi_{\theta}(a|s)
\right].
\]

This trains the actor to increase likelihood of dataset actions that have high
estimated advantage under the IQL critic and value networks.

## End-to-End Formula

For deterministic rollout, the complete actor inference pipeline is:

\[
o_t =
\left[
p_t,\ q_t,\ g_t,\ \operatorname{vec}(F_t^{agent}),\
\operatorname{vec}(F_t^{eye})
\right]
\in \mathbb{R}^{9225}.
\]

\[
\phi(o_t)
=
\left[
p_t,\ q_t,\ g_t,\ 
P^{agent}(F_t^{agent}),\
P^{eye}(F_t^{eye})
\right]
\in \mathbb{R}^{137},
\]

where each visual projector \(P^{cam}\) is:

\[
P^{cam}
=
\operatorname{Linear}_{64 \to 64}
\circ
\operatorname{Flatten}
\circ
\operatorname{SpatialSoftmax}
\circ
\operatorname{Conv2d}_{512 \to 32}^{1 \times 1}.
\]

The actor MLP is:

\[
h_3 =
f_{\theta}^{MLP}(\phi(o_t))
\in \mathbb{R}^{256}.
\]

\[
\mu_t =
W_{\mu} h_3 + b_{\mu}
\in \mathbb{R}^{7}.
\]

\[
\hat{a}_t =
\tanh(\mu_t)
\in [-1,1]^7.
\]

With current normalization settings:

\[
a_t^{env} = \hat{a}_t.
\]

## Source Anchors

- `configs/iql.yaml`: actor config, obs keys, visual specs, normalization flags.
- `train_policy.py`: injects replay-buffer `obs_slices` into
  `features_extractor_kwargs` for `resnet18conv`.
- `datasets/robomimic/replay_buffer.py`: reads HDF5, flattens multi-dimensional
  obs, computes slices, builds observation and action spaces.
- `models/feature_extractors.py`: `resnet18convFeaturesExtractor`, low-dim
  passthrough, visual SpatialSoftmax branches, final 137-dim feature vector.
- `models/model_utils.py`: `SpatialSoftmax` implementation.
- `models/policies.py`: `CustomActor`, MLP, Gaussian mean and log-std heads.
- `algos/offline_rl/iql.py`: IQL actor training loss.
- `utils/eval_utils.py`: eval-time observation adapter and rollout policy.
