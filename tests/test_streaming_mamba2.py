"""Memory glue tests on CPU, plus official Mamba2 CUDA forward/backward checks."""

import copy
import importlib.util
import shutil
import sys
from pathlib import Path
from types import SimpleNamespace

import h5py
import numpy as np
import pytest
import torch
from torch import nn
from torch.nn import functional as F
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from fineprog.models import encoder as encoder_module
from fineprog.models.encoder import TCCEncoder
from fineprog.dataset_preparation.h5vid_dataset import H5VideoDataset, FeatureCacheDataset, collate_episodes
from fineprog.utils import encoder_checkpoint as checkpoints


class TinyBackbone(nn.Module):
    def __init__(self, pretrained=False):
        super().__init__()
        self.conv = nn.Conv2d(3, 8, 1)
        self.bn = nn.BatchNorm2d(8)

    def set_train_base_mode(self, mode):
        self.requires_grad_(mode == "train_all")
        if mode == "only_bn":
            self.bn.requires_grad_(True)
        self.train(mode != "frozen")

    def forward(self, x):
        return self.bn(self.conv(F.adaptive_avg_pool2d(x, 2)))


class TinyTemporal(nn.Module):
    def __init__(self, in_channels, hidden_channels, embed_dim, debug=False):
        super().__init__()
        self.proj = nn.Linear(8, embed_dim)

    def forward(self, x):
        return self.proj(x.mean(dim=(2, 4, 5)))


class CausalTestBlock(nn.Module):
    """Tests wrapper state/gather semantics, not Mamba numerics (covered on CUDA)."""
    def __init__(self, d_model, **kwargs):
        super().__init__()
        self.proj = nn.Linear(d_model, d_model, bias=False)
        self.lengths = []

    def forward(self, x):
        self.lengths.append(x.shape[1])
        return self.proj(x).cumsum(1)

    def allocate_inference_cache(self, batch_size, max_seqlen):
        state = self.proj.weight.new_zeros(batch_size, self.proj.in_features)
        return state.clone(), state.clone()

    def step(self, x, conv, ssm):
        conv.copy_(self.proj(x[:, 0]))
        ssm.add_(conv)
        return ssm.unsqueeze(1), conv, ssm


@pytest.fixture
def tiny_local(monkeypatch):
    monkeypatch.setattr(encoder_module, "ResNet50Conv4cBackbone", TinyBackbone)
    monkeypatch.setattr(encoder_module, "TCCTemporalEmbedder", TinyTemporal)
    torch.manual_seed(7)
    old_threads = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(old_threads)


@pytest.fixture
def cpu_memory(tiny_local, monkeypatch):
    monkeypatch.setitem(sys.modules, "mamba_ssm", SimpleNamespace(Mamba2=CausalTestBlock))
    monkeypatch.setattr(checkpoints, "version", lambda name: "test")


@pytest.fixture
def training_module(monkeypatch):
    # Import-time constants must not resolve the developer's active dataset.
    # Training smoke tests install their own fully resolved config below.
    with monkeypatch.context() as import_patch:
        import_patch.syspath_prepend(str(Path(__file__).resolve().parents[1]))
        from utils.config_v2 import ConfigV2
        import_patch.setattr(ConfigV2, "load_train",
                             lambda self: self._load(self._root / "train.yaml"))
        import train_encoder
    return train_encoder


def make_encoder(memory_config=None, **kwargs):
    return TCCEncoder(pretrained=False, clip_len=3, context_stride=2,
                      memory_config={"enabled": True, "visual_chunk_size": 2,
                                     **(memory_config or {})}, **kwargs)


def make_dual_encoder(memory_config=None, **kwargs):
    return make_encoder(embedding_normalization="l2",
                        memory_config={"dual_branch": True, **(memory_config or {})},
                        **kwargs)


DUAL_INDEPENDENT_HISTORY = {
    "long_history_frames": 4, "short_history_frames": 4,
    "long_history_stride": 20, "short_history_stride": 5,
}


def test_dual_history_windows_normalization_and_shared_modules(cpu_memory):
    encoder = make_dual_encoder(embedding_dim=4, memory_config={
        "long_history_frames": 3, "short_history_frames": 2, "history_stride": 2})
    memory = encoder.memory
    raw = torch.arange(1, 21, dtype=torch.float32).reshape(1, 5, 4)
    captured = {"long": [], "short": [], "block": []}
    hooks = [module.register_forward_pre_hook(
        lambda module, args, key=key: captured[key].append(args[0].detach().clone()))
        for key, module in (("long", memory.long_proj), ("short", memory.short_proj),
                            ("block", memory.block))]
    with torch.autocast("cpu", dtype=torch.bfloat16):
        result = memory(raw)
    for hook in hooks:
        hook.remove()
    local = F.normalize(raw, dim=-1)
    long_indices = torch.tensor([[0, 0, 0], [0, 0, 1], [0, 0, 2], [0, 1, 3], [0, 2, 4]])
    short_indices = torch.tensor([[0, 0], [0, 1], [0, 2], [1, 3], [2, 4]])
    for branch, indices in (("long", long_indices), ("short", short_indices)):
        torch.testing.assert_close(captured[branch][0], local[:, indices].flatten(2))
        assert captured[branch][0].dtype == torch.float32
    assert result.shape == (1, 5, 2, 4) and result.dtype == torch.float32
    assert len(captured["block"]) == 2
    for tokens in captured["block"]:
        assert tokens.dtype == torch.float32
        torch.testing.assert_close(tokens.square().mean(-1), torch.ones(1, 5), rtol=4e-3, atol=0)
    assert sum(isinstance(module, CausalTestBlock) for module in memory.modules()) == 1
    assert memory.long_proj.weight.data_ptr() != memory.short_proj.weight.data_ptr()
    assert not list(memory.pre_norm.parameters())
    assert not any("alpha" in name for name, _ in memory.named_parameters())
    torch.testing.assert_close(memory(raw * torch.arange(1, 6)[None, :, None]), result)


def test_dual_independent_stride_windows_and_first_frame_padding(cpu_memory):
    memory = make_dual_encoder(embedding_dim=4,
                               memory_config=DUAL_INDEPENDENT_HISTORY).memory
    raw = torch.randn(2, 65, 4)
    captured = {"long": [], "short": []}
    hooks = [projection.register_forward_pre_hook(
        lambda module, args, branch=branch: captured[branch].append(args[0].detach().clone()))
        for branch, projection in (("long", memory.long_proj), ("short", memory.short_proj))]
    with torch.no_grad(), torch.autocast("cpu", dtype=torch.bfloat16):
        memory(raw)
    times = [0, 1, 5, 15, 20, 40, 60, 64]
    expected_indices = {
        "long": [[0, 0, 0, 0], [0, 0, 0, 1], [0, 0, 0, 5], [0, 0, 0, 15],
                 [0, 0, 0, 20], [0, 0, 20, 40], [0, 20, 40, 60], [4, 24, 44, 64]],
        "short": [[0, 0, 0, 0], [0, 0, 0, 1], [0, 0, 0, 5], [0, 5, 10, 15],
                  [5, 10, 15, 20], [25, 30, 35, 40], [45, 50, 55, 60], [49, 54, 59, 64]],
    }
    local = F.normalize(raw, dim=-1)
    full_windows = {branch: windows.pop() for branch, windows in captured.items()}
    for branch, indices in expected_indices.items():
        torch.testing.assert_close(full_windows[branch][:, times],
                                   local[:, torch.tensor(indices)].flatten(2))
        assert full_windows[branch].dtype == torch.float32
    state = memory.init_stream_state(2)
    with torch.no_grad(), torch.autocast("cpu", dtype=torch.bfloat16):
        for t in range(raw.shape[1]):
            memory.step(raw[:, t], state)
            state["steps"].add_(1)
    for hook in hooks:
        hook.remove()
    for branch, windows in captured.items():
        torch.testing.assert_close(torch.stack(windows, dim=1), full_windows[branch])


@pytest.mark.parametrize("settings,expected_span", [
    pytest.param(DUAL_INDEPENDENT_HISTORY, 60, id="hf4-long20-short5"),
    pytest.param({"long_history_frames": 2, "short_history_frames": 4,
                  "long_history_stride": 5, "short_history_stride": 20},
                 60, id="short-has-largest-span"),
    pytest.param({"long_history_frames": 1, "short_history_frames": 1,
                  "long_history_stride": 20, "short_history_stride": 5},
                 0, id="one-frame-zero-history"),
])
@pytest.mark.parametrize("branch_output_l2", [False, True])
def test_dual_independent_stride_raw_stream_and_selective_reset(
        cpu_memory, settings, expected_span, branch_output_l2):
    memory = make_dual_encoder(
        embedding_dim=4, memory_config={**settings, "branch_output_l2": branch_output_l2}).memory.eval()
    raw = torch.randn(2, 65, 4)
    with torch.no_grad(), torch.autocast("cpu", dtype=torch.bfloat16):
        expected = memory(raw)
        state = memory.init_stream_state(2)
        assert memory.history_span == expected_span
        assert state["local_history"].shape == (2, expected_span, 4)
        outputs = []
        for t in range(raw.shape[1]):
            outputs.append(memory.step(raw[:, t], state))
            state["steps"].add_(1)
        actual = torch.stack(outputs, dim=1)
        torch.testing.assert_close(actual, expected)
        torch.testing.assert_close(memory.fuse(raw, actual), memory.fuse(raw, expected))
        if expected_span:
            torch.testing.assert_close(state["local_history"],
                                       F.normalize(raw[:, -expected_span:], dim=-1))
        before = {key: value.clone() for key, value in state.items()}
        TCCEncoder.reset_stream_state(state, [True, False])
        for key in state:
            assert state[key][0].count_nonzero() == 0
            torch.testing.assert_close(state[key][1], before[key][1])
        next_raw = torch.randn(2, 4)
        continued = memory.step(next_raw, before)
        restarted = memory.step(next_raw[:1], memory.init_stream_state(1))
        actual = memory.step(next_raw, state)
        torch.testing.assert_close(actual[:1], restarted)
        torch.testing.assert_close(actual[1:], continued[1:])


@pytest.mark.parametrize("settings,long_stride,short_stride", [
    pytest.param({}, 20, 20, id="legacy-default"),
    pytest.param({"history_stride": 7}, 7, 7, id="legacy-shared"),
    pytest.param({"long_history_stride": 5}, 5, 20, id="long-override-default"),
    pytest.param({"short_history_stride": 5}, 20, 5, id="short-override-default"),
    pytest.param({"history_stride": 7, "long_history_stride": 20},
                 20, 7, id="long-override-shared"),
    pytest.param({"history_stride": 7, "short_history_stride": 5},
                 7, 5, id="short-override-shared"),
])
def test_dual_stride_overrides_fall_back_independently(cpu_memory, settings, long_stride, short_stride):
    encoder = make_dual_encoder(embedding_dim=4, memory_config=settings)
    memory = encoder.memory
    assert memory.long_history_stride == encoder.memory_config["long_history_stride"] == long_stride
    assert memory.short_history_stride == encoder.memory_config["short_history_stride"] == short_stride
    explicit = make_dual_encoder(embedding_dim=4, memory_config={
        **settings, "long_history_stride": long_stride, "short_history_stride": short_stride}).memory
    explicit.load_state_dict(memory.state_dict())
    raw = torch.randn(1, 65, 4)
    torch.testing.assert_close(memory(raw), explicit(raw), rtol=0, atol=0)


@pytest.mark.parametrize("settings,legacy", [
    pytest.param(DUAL_INDEPENDENT_HISTORY, False, id="independent-stride-checkpoint"),
    pytest.param({}, True, id="legacy-default-stride-checkpoint"),
    pytest.param({"history_stride": 7}, True, id="legacy-shared-stride-checkpoint"),
])
def test_dual_stride_checkpoint_metadata_and_legacy_loading(cpu_memory, tmp_path, settings, legacy):
    encoder = make_dual_encoder(embedding_dim=4, memory_config=settings).eval()
    payload = checkpoints.checkpoint_dict(encoder)
    saved_memory = payload["encoder_config"]["memory_config"]
    if legacy:
        saved_memory.pop("long_history_stride")
        saved_memory.pop("short_history_stride")
        if "history_stride" not in settings:
            saved_memory.pop("history_stride")
    else:
        assert saved_memory["long_history_stride"] == 20
        assert saved_memory["short_history_stride"] == 5
    path = tmp_path / "dual_stride.pt"
    torch.save(payload, path)
    restored = checkpoints.build_encoder_from_checkpoint(path, "cpu").eval()
    for name in ("long_history_stride", "short_history_stride"):
        assert restored.memory_config[name] == encoder.memory_config[name]
    raw = [torch.randn(length, 4) for length in (65, 61)]
    with torch.no_grad():
        torch.testing.assert_close(encoder.forward_sequence(raw, [65, 61]),
                                   restored.forward_sequence(raw, [65, 61]), rtol=0, atol=0)


@pytest.mark.parametrize("branch_output_l2", [False, True])
def test_dual_branch_l2_controls_gate_and_weighted_sum(cpu_memory, branch_output_l2):
    encoder = make_dual_encoder(embedding_dim=4,
                               memory_config={"branch_output_l2": branch_output_l2})
    memory = encoder.memory
    with torch.no_grad():
        memory.proj.weight.copy_(torch.eye(4))
        memory.proj.bias.zero_()
    raw = torch.randn(1, 2, 4)
    y = torch.tensor([[[[3., 4., 0., 0.], [0., 0., 0., 2.]],
                       [[0., 12., 0., 0.], [0., 0., 6., 8.]]]])
    branches = F.normalize(y, dim=-1) if branch_output_l2 else y
    gate_inputs = []
    hook = memory.gate.register_forward_pre_hook(
        lambda module, args: gate_inputs.append(args[0].detach().clone()))
    with torch.autocast("cpu", dtype=torch.bfloat16):
        initial = memory.fuse(raw, y)
    assert initial.dtype == torch.float32
    torch.testing.assert_close(gate_inputs[-1], branches.flatten(-2))
    torch.testing.assert_close(memory.gate(branches.flatten(-2)), torch.zeros(1, 2, 2))
    torch.testing.assert_close(initial, branches.mean(-2))
    # Exercise unequal, input-dependent weights after the zero-initialized gate.
    with torch.no_grad():
        memory.gate[-1].weight.copy_(torch.arange(8).reshape(2, 4) / 10)
        memory.gate[-1].bias.copy_(torch.tensor([0.2, -0.3]))
    logits = memory.gate(branches.flatten(-2))
    expected = (logits.softmax(-1).unsqueeze(-1) * branches).sum(-2)
    with torch.autocast("cpu", dtype=torch.bfloat16):
        actual = memory.fuse(raw, y)
    torch.testing.assert_close(gate_inputs[-1], branches.flatten(-2))
    torch.testing.assert_close(actual, expected)
    torch.testing.assert_close(memory.fuse(raw + 1000, y), actual)
    torch.testing.assert_close(encoder._apply_embedding_normalization(actual).norm(dim=-1),
                               torch.ones(1, 2))
    hook.remove()


@pytest.mark.parametrize("train_base,train_embedding", [("frozen", False), ("only_bn", True)])
@pytest.mark.parametrize("branch_output_l2", [False, True])
def test_dual_weight_logging_full_frames_padding_and_gradients(
        cpu_memory, training_module, train_base, train_embedding, branch_output_l2):
    encoder = make_dual_encoder(
        embedding_dim=4, train_base=train_base, train_embedding=train_embedding,
        memory_config={"branch_output_l2": branch_output_l2})
    with torch.no_grad():
        encoder.memory.gate[-1].weight.normal_(std=0.1)
        encoder.memory.gate[-1].bias.copy_(torch.tensor([0.2, -0.3]))
    raw = [torch.randn(n, 4, requires_grad=True) for n in (7, 3)]
    captured = []
    hook = encoder.memory.gate.register_forward_hook(
        lambda module, args, logits: captured.append(logits.detach().softmax(-1)))
    with torch.no_grad():
        encoder.forward_episode(raw, [7, 3])
    hook.remove()
    real_weights = ([captured[0][0, :7], captured[0][1, :3]]
                    if train_base == "frozen" else captured)
    expected_sum = torch.cat(real_weights).sum(0)

    targets = torch.tensor([[5, 6], [1, 2]])
    params = tuple(raw) + tuple(encoder.memory.parameters())
    target = torch.randn(2, 2, 4)
    plain = encoder.forward_episode(raw, [7, 3], targets)
    plain_grads = torch.autograd.grad((plain - target).square().mean(), params)
    stats = {}
    logged = encoder.forward_episode(raw, [7, 3], targets, memory_norm_stats=stats)
    logged_grads = torch.autograd.grad((logged - target).square().mean(), params)
    torch.testing.assert_close(logged, plain, rtol=0, atol=0)
    for actual, expected in zip(logged_grads, plain_grads):
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    assert stats["num_frames"] == 10  # Full episodes, not the four loss targets.
    torch.testing.assert_close(stats["branch_weight_sum"], expected_sum)
    assert all(not value.requires_grad for value in stats.values()
               if isinstance(value, torch.Tensor))

    # Accumulation must weight each real frame equally across different batches.
    encoder.forward_episode([raw[1]], [3], targets[1:], memory_norm_stats=stats)
    assert stats["num_frames"] == 13
    torch.testing.assert_close(stats["branch_weight_sum"], expected_sum + real_weights[1].sum(0))
    metrics = training_module._memory_log(encoder, stats)
    assert set(metrics) == {"memory/input_norm_mean", "memory/pre_normalization_norm_mean",
                            "memory/long_weight_mean", "memory/short_weight_mean"}
    torch.testing.assert_close(
        torch.tensor([metrics["memory/long_weight_mean"], metrics["memory/short_weight_mean"]]),
        stats["branch_weight_sum"] / 13)
    assert metrics["memory/long_weight_mean"] + metrics["memory/short_weight_mean"] == pytest.approx(1)
    assert training_module._memory_log(encoder) == {}
    assert training_module._memory_log(encoder, {"num_frames": 0}) == {}


@pytest.mark.parametrize("branch_output_l2", [False, True])
def test_dual_causality_target_gather_and_padding(cpu_memory, branch_output_l2):
    encoder = make_dual_encoder(memory_config={
        "long_history_frames": 3, "short_history_frames": 2, "history_stride": 2,
        "branch_output_l2": branch_output_l2})
    raw = torch.randn(2, 9, 128, requires_grad=True)
    indices = torch.tensor([[2, 4, 4], [1, 2, 2]])
    result = encoder.forward_episode(raw, [7, 3], indices)
    changed = raw.detach().clone()
    changed[0, 5:] = 1000
    changed[1, 3:] = 1000
    torch.testing.assert_close(result, encoder.forward_episode(changed, [7, 3], indices))
    torch.testing.assert_close(result[:, 1], result[:, 2])
    (result * torch.linspace(-1, 1, 128)).sum().backward()
    assert raw.grad[0, 0].abs().sum() > 0
    assert raw.grad[0, 5:].count_nonzero() == raw.grad[1, 3:].count_nonzero() == 0


@pytest.mark.parametrize("branch_output_l2", [False, True])
@pytest.mark.parametrize("long_frames,short_frames,stride", [(3, 2, 2), (8, 2, 20)])
def test_dual_stream_matches_episode_and_selective_reset(
        cpu_memory, branch_output_l2, long_frames, short_frames, stride):
    encoder = make_dual_encoder(memory_config={
        "long_history_frames": long_frames, "short_history_frames": short_frames,
        "history_stride": stride, "branch_output_l2": branch_output_l2}).eval()
    frames = torch.rand(2, 9, 3, 224, 224)
    with torch.no_grad():
        raw = [encoder.encode_local_episode(x) for x in frames]
        expected = encoder.forward_sequence(raw, [9, 9])
        state = encoder.init_stream_state(2)
        assert state["local_history"].shape == (2, (long_frames - 1) * stride, 128)
        for kind in ("conv", "ssm"):
            assert state[f"{kind}_long"].data_ptr() != state[f"{kind}_short"].data_ptr()
        with torch.autocast("cpu", dtype=torch.bfloat16):
            actual = torch.stack([encoder.encode_step(frames[:, t], state)[0]
                                  for t in range(9)], dim=1)
        torch.testing.assert_close(actual, expected)
        history_span = state["local_history"].shape[1]
        if history_span < 9:
            torch.testing.assert_close(state["local_history"],
                                       F.normalize(torch.stack(raw)[:, -history_span:], dim=-1))
        before = {key: value.clone() for key, value in state.items()}
        encoder.get_stream_embedding(state)
        for key in state:
            torch.testing.assert_close(state[key], before[key])
        encoder.reset_stream_state(state, [True, False])
        for key in state:
            assert state[key][0].count_nonzero() == 0
            torch.testing.assert_close(state[key][1], before[key][1])
        fresh = encoder.init_stream_state(1)
        actual, _ = encoder.encode_step(frames[:, 0], state)
        expected, _ = encoder.encode_step(frames[:1, 0], fresh)
        torch.testing.assert_close(actual[:1], expected)


@pytest.mark.parametrize("train_base,train_embedding", [("frozen", False), ("only_bn", True)])
@pytest.mark.parametrize("branch_output_l2", [False, True])
@pytest.mark.parametrize("history_settings", [
    pytest.param({}, id="legacy-hf8-hf2-hs20"),
    pytest.param(DUAL_INDEPENDENT_HISTORY, id="hf4-long20-short5"),
])
def test_dual_two_step_training_norm_logging_and_checkpoint(
        cpu_memory, training_module, tmp_path, train_base, train_embedding, branch_output_l2,
        history_settings):
    _memory_log = training_module._memory_log

    encoder = make_dual_encoder(train_base=train_base, train_embedding=train_embedding,
                               memory_config={"branch_output_l2": branch_output_l2,
                                              "freeze_bn_stats": False, **history_settings}).train()
    assert encoder.memory_config["long_history_frames"] == history_settings.get("long_history_frames", 8)
    assert encoder.memory_config["short_history_frames"] == history_settings.get("short_history_frames", 2)
    assert encoder.memory_config["history_stride"] == 20
    assert encoder.memory_config["long_history_stride"] == history_settings.get("long_history_stride", 20)
    assert encoder.memory_config["short_history_stride"] == history_settings.get("short_history_stride", 20)
    local_before = {key: value.clone() for key, value in encoder.state_dict().items()
                    if not key.startswith("memory.")}
    frames = torch.rand(6, 3, 224, 224)
    target = F.normalize(torch.randn(1, 2, 128), dim=-1)
    optimizer = torch.optim.Adam([p for p in encoder.parameters() if p.requires_grad], lr=1e-3)
    hidden_before = encoder.memory.gate[0].weight.detach().clone()
    for step in range(2):
        optimizer.zero_grad()
        raw = encoder.encode_local_episode(frames)
        assert raw.requires_grad == train_embedding
        if raw.requires_grad:
            raw.retain_grad()
        stats = {}
        output = encoder.forward_episode([raw], [6], torch.tensor([[2, 5]]),
                                         memory_norm_stats=stats)
        (-(output * target).sum(-1).mean()).backward()
        modules = [encoder.memory.long_proj, encoder.memory.short_proj,
                   encoder.memory.block, encoder.memory.proj, encoder.memory.gate[-1]]
        if step == 1:
            modules.append(encoder.memory.gate[0])
        if train_embedding:
            modules += [encoder.backbone.bn, encoder.temporal_embedder]
            assert torch.isfinite(raw.grad).all() and raw.grad.abs().sum() > 0
        for module in modules:
            grads = [p.grad for p in module.parameters() if p.requires_grad]
            assert grads and all(g is not None and torch.isfinite(g).all() for g in grads)
            assert sum(g.abs().sum() for g in grads) > 0
        assert all(p.grad is None for p in encoder.backbone.conv.parameters())
        if step == 0:
            assert encoder.memory.gate[0].weight.grad.count_nonzero() == 0
        optimizer.step()
        if step == 0:
            torch.testing.assert_close(encoder.memory.gate[0].weight, hidden_before)
        metrics = _memory_log(encoder, stats)
        assert "memory/alpha" not in metrics and stats["num_frames"] == 6
        for key in ("memory/input_norm_mean", "memory/pre_normalization_norm_mean"):
            assert np.isfinite(metrics[key]) and metrics[key] > 0
        assert 0 <= metrics["memory/long_weight_mean"] <= 1
        assert 0 <= metrics["memory/short_weight_mean"] <= 1
        assert metrics["memory/long_weight_mean"] + metrics["memory/short_weight_mean"] == pytest.approx(1)
        if step == 0:
            assert metrics["memory/long_weight_mean"] == metrics["memory/short_weight_mean"] == 0.5
    assert not torch.equal(encoder.memory.gate[0].weight, hidden_before)
    if train_base == "frozen":
        for key, value in local_before.items():
            torch.testing.assert_close(value, encoder.state_dict()[key])
    else:
        assert encoder.backbone.bn.training
        assert not torch.equal(encoder.backbone.bn.weight, local_before["backbone.bn.weight"])
    encoder.eval()
    path = tmp_path / "dual.pt"
    torch.save(checkpoints.checkpoint_dict(encoder), path)
    restored = checkpoints.build_encoder_from_checkpoint(path, "cpu").eval()
    assert restored.memory_config["dual_branch"] is True
    assert restored.memory_config["branch_output_l2"] is branch_output_l2
    for name in ("long_history_stride", "short_history_stride"):
        assert restored.memory_config[name] == encoder.memory_config[name]
    with torch.no_grad():
        torch.testing.assert_close(encoder.forward_sequence([raw.detach()], [6]),
                                   restored.forward_sequence([raw.detach()], [6]))


@pytest.mark.parametrize("history_frames,history_stride", [(1, 1), (3, 1), (3, 2)])
@pytest.mark.parametrize("train_base,train_embedding", [("frozen", False), ("only_bn", True)])
@pytest.mark.parametrize("normalize_input", [False, True])
def test_full_history_gather_causality_and_padding(
        cpu_memory, history_frames, history_stride, train_base, train_embedding, normalize_input):
    encoder = make_encoder(train_base=train_base, train_embedding=train_embedding,
                           memory_config={"normalize_input": normalize_input,
                                          "history_frames": history_frames,
                                          "history_stride": history_stride})
    raw = torch.randn(2, 9, 128, requires_grad=True)
    indices = torch.tensor([[2, 4, 4], [1, 2, 2]])
    result = encoder.forward_episode(raw, [7, 3], indices)
    changed = raw.detach().clone()
    changed[0, 5:] = 1000
    changed[1, 3:] = 1000
    torch.testing.assert_close(result, encoder.forward_episode(changed, [7, 3], indices))
    torch.testing.assert_close(result[:, 1], result[:, 2])
    (result * torch.linspace(-1, 1, 128)).sum().backward()
    assert raw.grad[0, 0].abs().sum() > 0  # unsampled history receives gradients
    assert raw.grad[0, 5:].count_nonzero() == 0
    assert raw.grad[1, 3:].count_nonzero() == 0


@pytest.mark.parametrize("stride", [1, 3])
def test_single_frame_history_matches_legacy(cpu_memory, tmp_path, stride):
    encoder = make_encoder()
    explicit = make_encoder(memory_config={"history_frames": 1, "history_stride": stride})
    explicit.load_state_dict(encoder.state_dict(), strict=True)
    assert isinstance(explicit.memory.history_proj, nn.Identity)
    assert not any("history_proj." in key for key in encoder.state_dict())
    raw = torch.randn(2, 5, 128)
    torch.testing.assert_close(encoder.memory(raw), encoder.memory.block(raw), rtol=0, atol=0)
    torch.testing.assert_close(encoder.forward_sequence(raw, [5, 3]),
                               explicit.forward_sequence(raw, [5, 3]), rtol=0, atol=0)
    legacy = checkpoints.checkpoint_dict(encoder)
    legacy["encoder_config"]["memory_config"].pop("history_frames")
    legacy["encoder_config"]["memory_config"].pop("history_stride")
    legacy["encoder_config"]["memory_config"].pop("normalize_input", None)
    path = tmp_path / "legacy_memory.pt"
    torch.save(legacy, path)
    restored = checkpoints.build_encoder_from_checkpoint(path, "cpu")
    assert restored.memory_config["history_frames"] == restored.memory_config["history_stride"] == 1
    assert restored.memory.normalize_input is False
    torch.testing.assert_close(encoder.forward_sequence(raw, [5, 3]),
                               restored.forward_sequence(raw, [5, 3]), rtol=0, atol=0)


@pytest.mark.parametrize("name", ["history_frames", "history_stride"])
@pytest.mark.parametrize("value", [0, -1, True, 1.5, "2", None])
def test_history_settings_require_positive_integers(cpu_memory, name, value):
    with pytest.raises(ValueError, match=rf"memory\.{name} must be a positive integer"):
        make_encoder(memory_config={name: value})


@pytest.mark.parametrize("name", ["long_history_stride", "short_history_stride"])
@pytest.mark.parametrize("value", [0, -1, True, 1.5, "2", None])
def test_dual_history_strides_require_positive_integers(cpu_memory, name, value):
    with pytest.raises(ValueError, match=rf"memory\.{name} must be a positive integer"):
        make_dual_encoder(memory_config={name: value})


@pytest.mark.parametrize("stride,expected", [
    (1, [[1, 2, 1, 2, 1, 2], [1, 2, 1, 2, 3, 4], [1, 2, 3, 4, 5, 6],
         [3, 4, 5, 6, 7, 8], [5, 6, 7, 8, 9, 10]]),
    (2, [[1, 2, 1, 2, 1, 2], [1, 2, 1, 2, 3, 4], [1, 2, 1, 2, 5, 6],
         [1, 2, 3, 4, 7, 8], [1, 2, 5, 6, 9, 10]]),
])
def test_history_window_order_padding_and_fp32_projection(cpu_memory, stride, expected):
    encoder = make_encoder(embedding_dim=2,
                           memory_config={"history_frames": 3, "history_stride": stride})
    raw = torch.arange(1, 11, dtype=torch.float32).reshape(1, 5, 2)
    captured = []
    hook = encoder.memory.history_proj.register_forward_pre_hook(
        lambda module, args: captured.append(args[0].detach().clone()))
    with torch.autocast("cpu", dtype=torch.bfloat16):
        output = encoder.memory(raw)
    hook.remove()
    assert captured[0].dtype == output.dtype == torch.float32
    assert output.shape == raw.shape
    torch.testing.assert_close(captured[0], torch.tensor([expected], dtype=torch.float32))
    assert encoder.memory(torch.ones(1, 1, 2)).shape == (1, 1, 2)


@pytest.mark.parametrize("history_frames,history_stride",
                         [(1, 1)] + [(h, s) for h in (2, 4, 8) for s in (5, 10, 20)])
def test_normalized_memory_input_contract(cpu_memory, history_frames, history_stride):
    """Each frame is normalized before mixing; the learned mix is normalized again."""
    encoder = make_encoder(embedding_normalization="l2", memory_config={
        "use_residual": False, "normalize_input": True,
        "history_frames": history_frames, "history_stride": history_stride})
    raw = 1000 * torch.randn(2, 11, 128)
    original = raw.clone()
    projected_inputs, mixer_inputs = [], []
    hooks = [
        encoder.memory.history_proj.register_forward_pre_hook(
            lambda module, args: projected_inputs.append(args[0].detach().clone())),
        encoder.memory.block.register_forward_pre_hook(
            lambda module, args: mixer_inputs.append(args[0].detach().clone())),
    ]
    with torch.autocast("cpu", dtype=torch.bfloat16):
        output = encoder._apply_embedding_normalization(encoder.memory.fuse(raw, encoder.memory(raw)))
    torch.testing.assert_close(raw, original, rtol=0, atol=0)
    offsets = torch.arange(history_frames - 1, -1, -1) * history_stride
    indices = (torch.arange(11)[:, None] - offsets).clamp_min(0)
    expected = F.normalize(raw[:, indices], dim=-1).flatten(start_dim=2)
    torch.testing.assert_close(projected_inputs[0], expected)
    assert projected_inputs[0].dtype == mixer_inputs[0].dtype == torch.float32
    torch.testing.assert_close(mixer_inputs[0].square().mean(-1),
                               torch.ones(2, 11), rtol=4e-3, atol=0)
    torch.testing.assert_close(output.norm(dim=-1), torch.ones(2, 11))

    # Independent positive per-frame rescaling must disappear before history fusion.
    scales = torch.logspace(-2, 3, 22).reshape(2, 11, 1)
    output = encoder.memory(raw)
    rescaled = encoder.memory(raw * scales)
    torch.testing.assert_close(output, rescaled, rtol=2e-5, atol=2e-6)
    if history_frames > 1:
        with torch.no_grad():
            encoder.memory.history_proj.weight.mul_(1e4)
            encoder.memory.history_proj.bias.mul_(1e4)
        encoder.memory(raw)
        # Projection gain cannot pass directly into the mixer; epsilon causes a
        # small difference from the unamplified RMS, but directions are preserved.
        torch.testing.assert_close(mixer_inputs[-1].square().mean(-1),
                                   torch.ones(2, 11), rtol=1e-5, atol=0)
        torch.testing.assert_close(F.normalize(mixer_inputs[-1], dim=-1),
                                   F.normalize(mixer_inputs[0], dim=-1))
    zero_output = encoder.memory(torch.zeros(2, 1, 128))
    assert zero_output.shape == (2, 1, 128) and torch.isfinite(zero_output).all()
    assert torch.isfinite(mixer_inputs[-1]).all()
    for hook in hooks:
        hook.remove()
    # Fixed normalization adds no checkpoint weights.
    legacy = make_encoder(memory_config={"history_frames": history_frames,
                                          "history_stride": history_stride})
    assert encoder.state_dict().keys() == legacy.state_dict().keys()


def test_normalized_warm_start_preserves_local_gradients(cpu_memory):
    encoder = make_encoder(train_base="only_bn", train_embedding=True,
                           embedding_normalization="l2", memory_config={
                               "use_residual": False, "normalize_input": True,
                               "history_frames": 4, "history_stride": 5}).train()
    frames = torch.rand(6, 3, 224, 224)
    raw = encoder.encode_local_episode(frames)
    raw.retain_grad()
    output = encoder.forward_sequence([raw], [6])[0]
    target = F.normalize(torch.randn_like(output), dim=-1)
    (-(output * target).sum(-1).mean()).backward()
    assert torch.isfinite(raw.grad).all() and raw.grad.abs().sum() > 0
    for module in (encoder.backbone.bn, encoder.temporal_embedder,
                   encoder.memory.history_proj, encoder.memory.block, encoder.memory.proj):
        grads = [p.grad for p in module.parameters() if p.requires_grad]
        assert grads and all(g is not None and torch.isfinite(g).all() for g in grads)
        assert sum(g.abs().sum() for g in grads) > 0
    assert all(p.grad is None for p in encoder.backbone.conv.parameters())
    assert not encoder.backbone.bn.training


@pytest.mark.parametrize("train_base,train_embedding", [("frozen", False), ("only_bn", True)])
def test_memory_norms_cover_real_frames_and_preserve_gradients(
        cpu_memory, training_module, train_base, train_embedding):
    _memory_log = training_module._memory_log

    encoder = make_encoder(train_base=train_base, train_embedding=train_embedding,
                           embedding_normalization="l2")
    with torch.no_grad():
        encoder.memory.block.proj.weight.copy_(torch.eye(128))
        encoder.memory.proj.weight.copy_(torch.eye(128))
        encoder.memory.proj.bias.zero_()
        encoder.memory.alpha.fill_(0.5)
    plain = copy.deepcopy(encoder)
    raw = [F.pad(torch.tensor(values, dtype=torch.float32), (0, 126)).requires_grad_()
           for values in ([[3, 4], [0, 12], [5, 12], [10000, 10000]],
                          [[8, 15], [7, 24], [20000, 20000]])]
    plain_raw = [x.detach().clone().requires_grad_() for x in raw]
    indices = torch.tensor([[0, 0], [0, 0]])
    stats = {}
    actual = encoder.forward_episode(raw, [3, 2], indices, memory_norm_stats=stats)
    expected = plain.forward_episode(plain_raw, [3, 2], indices)
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    # Norms cover all five real frames, even though the loss samples only frame 0.
    # Large tail observations and the batched forward's right padding are excluded.
    fused = torch.tensor([[4.5, 6], [1.5, 20], [9, 26], [12, 22.5], [14.5, 43.5]])
    assert stats["num_frames"] == 5
    metrics = _memory_log(encoder, stats)
    assert metrics["memory/input_norm_mean"] == pytest.approx(14.4)
    assert metrics["memory/pre_normalization_norm_mean"] == pytest.approx(fused.norm(dim=-1).mean().item())
    assert all(not value.requires_grad and value.grad_fn is None
               for value in stats.values() if isinstance(value, torch.Tensor))

    weights = torch.linspace(-1, 1, 128)
    (actual * weights).sum().backward()
    (expected * weights).sum().backward()
    for p, q in zip(encoder.memory.parameters(), plain.memory.parameters()):
        torch.testing.assert_close(p.grad, q.grad, rtol=0, atol=0)
    for p, q in zip(raw, plain_raw):
        torch.testing.assert_close(p.grad, q.grad, rtol=0, atol=0)

    # Accumulation across unequal batches is weighted by real frame count.
    encoder.forward_episode([raw[0][:1]], [1], torch.tensor([[0]]), memory_norm_stats=stats)
    assert stats["num_frames"] == 6
    metrics = _memory_log(encoder, stats)
    assert metrics["memory/input_norm_mean"] == pytest.approx(77 / 6)
    assert metrics["memory/pre_normalization_norm_mean"] == pytest.approx((fused.norm(dim=-1).sum().item() + 7.5) / 6)
    assert set(_memory_log(encoder)) == {"memory/alpha"}


@pytest.mark.parametrize("history_frames,history_stride", [(1, 1), (3, 1), (3, 2)])
@pytest.mark.parametrize("use_residual", [True, False])
@pytest.mark.parametrize("normalization", ["none", "l2"])
@pytest.mark.parametrize("normalize_input", [False, True])
def test_stream_matches_episode_and_selective_reset(
        cpu_memory, normalization, use_residual, history_frames, history_stride, normalize_input):
    encoder = make_encoder(embedding_normalization=normalization,
                           memory_config={"use_residual": use_residual,
                                          "normalize_input": normalize_input,
                                          "history_frames": history_frames,
                                          "history_stride": history_stride}).eval()
    frames = torch.rand(2, 7, 3, 224, 224)
    with torch.no_grad():
        raw = [encoder.encode_local_episode(x) for x in frames]
        expected = encoder.forward_sequence(raw, [7, 7])
        state = encoder.init_stream_state(2)
        if history_frames > 1:
            assert state["raw_history"].shape == (2, (history_frames - 1) * history_stride, 128)
        else:
            assert "raw_history" not in state
        streamed = torch.stack([encoder.encode_step(frames[:, t], state)[0] for t in range(7)], dim=1)
    torch.testing.assert_close(streamed, expected)
    before = {k: v.clone() for k, v in state.items()}
    encoder.get_stream_embedding(state)
    for k in state:
        torch.testing.assert_close(state[k], before[k])
    encoder.reset_stream_state(state, [True, False])
    for k in state:
        assert state[k][0].count_nonzero() == 0
        torch.testing.assert_close(state[k][1], before[k][1])
    fresh = encoder.init_stream_state(1)
    actual, _ = encoder.encode_step(frames[:, 0], state)
    expected, _ = encoder.encode_step(frames[:1, 0], fresh)
    torch.testing.assert_close(actual[:1], expected)
    if use_residual:
        encoder.memory.alpha.data.zero_()
        local = encoder._apply_embedding_normalization(raw[0])
        torch.testing.assert_close(encoder.forward_sequence([raw[0]], [7])[0], local)


@pytest.mark.parametrize("history_frames,history_stride", [(1, 1), (3, 2)])
@pytest.mark.parametrize("normalize_input", [False, True])
def test_no_residual_frozen_training_and_checkpoint(
        cpu_memory, tmp_path, history_frames, history_stride, normalize_input):
    encoder = make_encoder(train_base="frozen", train_embedding=False,
                           embedding_normalization="l2",
                           memory_config={"use_residual": False, "alpha_init": 0.01,
                                          "normalize_input": normalize_input,
                                          "history_frames": history_frames,
                                          "history_stride": history_stride}).train()
    assert encoder.memory.alpha.item() == 1.0
    assert not encoder.memory.alpha.requires_grad
    local_before = {k: v.clone() for k, v in encoder.state_dict().items()
                    if not k.startswith("memory.")}
    raw = encoder.encode_local_episode(torch.rand(4, 3, 224, 224))
    assert not raw.requires_grad
    y = encoder.memory(raw.unsqueeze(0))[0]
    fused = encoder.memory.fuse(raw, y)
    torch.testing.assert_close(fused, encoder.memory.proj(y))
    # Holding the Mamba output fixed, local embeddings have no skip path.
    torch.testing.assert_close(fused, encoder.memory.fuse(raw + 100, y))
    actual = encoder.forward_sequence([raw], [4])[0]
    torch.testing.assert_close(actual, F.normalize(fused, dim=-1))
    optimizer = torch.optim.Adam([p for p in encoder.parameters() if p.requires_grad], lr=1e-3)
    (actual * torch.linspace(-1, 1, 128)).sum().backward()
    assert encoder.memory.alpha.grad is None
    modules = [encoder.memory.block, encoder.memory.proj]
    if history_frames > 1:
        modules.append(encoder.memory.history_proj)
    for module in modules:
        assert all(p.grad is not None and torch.isfinite(p.grad).all()
                   for p in module.parameters())
        assert sum(p.grad.abs().sum() for p in module.parameters()) > 0
    optimizer.step()
    assert encoder.memory.alpha.item() == 1.0
    for key, value in local_before.items():
        torch.testing.assert_close(value, encoder.state_dict()[key])

    path = tmp_path / "no_residual.pt"
    torch.save(checkpoints.checkpoint_dict(encoder), path)
    restored = checkpoints.build_encoder_from_checkpoint(path, "cpu").eval().train()
    assert restored.memory_config["use_residual"] is False
    assert restored.memory_config["normalize_input"] is normalize_input
    assert restored.memory_config["history_frames"] == history_frames
    assert restored.memory_config["history_stride"] == history_stride
    assert not restored.memory.use_residual and not restored.memory.alpha.requires_grad
    assert restored.memory.alpha.item() == 1.0
    torch.testing.assert_close(restored.forward_sequence([raw], [4]),
                               encoder.forward_sequence([raw], [4]))


def test_joint_checkpointing_gradients_and_bn(cpu_memory):
    encoder = make_encoder(train_base="train_all", train_embedding=True).train()
    plain = copy.deepcopy(encoder)
    plain.memory_config["checkpoint_visual"] = False
    frames = torch.rand(6, 3, 224, 224, requires_grad=True)
    stats = encoder.backbone.bn.running_mean.clone()
    losses = []
    for model in (encoder, plain):
        e = model.encode_local_episode(frames)
        z = model.forward_episode([e], [6], torch.tensor([[5]]))
        loss = z.square().sum()
        loss.backward()
        losses.append(loss)
    torch.testing.assert_close(*losses)
    for p, q in zip(encoder.parameters(), plain.parameters()):
        assert p.grad is not None and torch.isfinite(p.grad).all()
        torch.testing.assert_close(p.grad, q.grad)
    assert frames.grad[0].abs().sum() > 0
    torch.testing.assert_close(encoder.backbone.bn.running_mean, stats)
    encoder.eval().train()
    assert not encoder.backbone.bn.training and encoder.backbone.bn.weight.requires_grad


def test_only_bn_recompute_preserves_stats_and_gradients(cpu_memory):
    encoder = make_encoder(train_base="only_bn", train_embedding=True,
                           memory_config={"freeze_bn_stats": False}).train()
    encoder.backbone.bn.num_batches_tracked.fill_(60000)
    plain = copy.deepcopy(encoder)
    plain.memory_config["checkpoint_visual"] = False
    buffer_refs = (encoder.backbone.bn.running_mean, encoder.backbone.bn.running_var,
                   encoder.backbone.bn.num_batches_tracked)
    frames = [torch.rand(n, 3, 224, 224) for n in (6, 5)]
    optimizers = [torch.optim.Adam([p for p in m.parameters() if p.requires_grad], lr=1e-3)
                  for m in (encoder, plain)]
    for step in range(2):
        outputs = []
        for model, optimizer in zip((encoder, plain), optimizers):
            optimizer.zero_grad()
            raw = [model.encode_local_episode(x) for x in frames]
            z = model.forward_episode(raw, [6, 5], torch.tensor([[2, 5], [2, 4]]))
            z.square().sum().backward()
            outputs.append(z)
            assert model.backbone.bn.num_batches_tracked.item() == 60000 + 6 * (step + 1)
        torch.testing.assert_close(*outputs)
        for p, q in zip(encoder.parameters(), plain.parameters()):
            if p.requires_grad:
                assert p.grad is not None and torch.isfinite(p.grad).all()
                torch.testing.assert_close(p.grad, q.grad)
            else:
                assert p.grad is None and q.grad is None
        for optimizer in optimizers:
            optimizer.step()
        for key, value in encoder.state_dict().items():
            torch.testing.assert_close(value, plain.state_dict()[key])
    for actual, original in zip((encoder.backbone.bn.running_mean, encoder.backbone.bn.running_var,
                                 encoder.backbone.bn.num_batches_tracked), buffer_refs):
        assert actual is original
    assert encoder.backbone.bn.running_mean.abs().sum() > 0
    encoder.eval().train()
    assert encoder.backbone.bn.training and encoder.backbone.bn.weight.requires_grad
    assert not encoder.backbone.conv.weight.requires_grad


@pytest.mark.parametrize("freeze_bn_stats", [True, False])
def test_frozen_local_and_only_bn_modes(cpu_memory, freeze_bn_stats):
    encoder = make_encoder(train_base="frozen", train_embedding=False,
                           memory_config={"freeze_bn_stats": freeze_bn_stats}).train()
    local_before = {k: v.clone() for k, v in encoder.state_dict().items() if not k.startswith("memory.")}
    raw = encoder.encode_local_episode(torch.rand(4, 3, 224, 224))
    assert not raw.requires_grad
    optimizer = torch.optim.Adam([p for p in encoder.parameters() if p.requires_grad], lr=1e-3)
    encoder.forward_episode([raw], [4], torch.tensor([[3]])).square().sum().backward()
    optimizer.step()
    for key, value in local_before.items():
        torch.testing.assert_close(value, encoder.state_dict()[key])
    assert encoder.memory.alpha.grad is not None
    encoder._train_base, encoder._train_embedding = "only_bn", True
    encoder.eval().train()
    assert not encoder.backbone.conv.weight.requires_grad
    assert encoder.backbone.bn.weight.requires_grad
    assert encoder.backbone.bn.training == (not freeze_bn_stats)
    assert encoder.temporal_embedder.proj.weight.requires_grad


def test_checkpoint_restores_architecture_and_legacy_warm_start(cpu_memory, tmp_path):
    encoder = make_encoder(embedding_normalization="l2", train_base="train_all").eval()
    path = tmp_path / "memory.pt"
    torch.save(checkpoints.checkpoint_dict(encoder), path)
    restored = checkpoints.build_encoder_from_checkpoint(path, "cpu", {"embedding_dim": 16, "context_stride": 99}).eval()
    raw = [torch.randn(7, 128)]
    torch.testing.assert_close(encoder.forward_sequence(raw, [7]), restored.forward_sequence(raw, [7]))
    assert restored.context_stride == 2

    local = TCCEncoder(pretrained=False, context_stride=3, embedding_normalization="l2")
    local_path, config_path = tmp_path / "local.pt", tmp_path / "local.yaml"
    torch.save({"model_state_dict": local.state_dict(), "embedding_normalization": "l2"}, local_path)
    config_path.write_text(yaml.safe_dump({"context_size": 2, "context_stride": 3, "embedding_dim": 128}))
    trained, effective = checkpoints.build_training_encoder({
        "train_base": "frozen", "train_embedding": False, "context_stride": 99,
        "memory": {"enabled": True, "local_checkpoint_path": str(local_path), "local_config_path": str(config_path)},
    }, "cpu")
    assert effective["context_stride"] == 3 and trained.embedding_normalization == "l2"
    assert not any(p.requires_grad for p in trained.temporal_embedder.parameters())


@pytest.mark.parametrize("dual_branch", [False, True])
def test_full_dataset_cache_and_output_stride(cpu_memory, tmp_path, dual_branch):
    from fineprog.extract_embeddings import extract_embeddings
    path = tmp_path / "videos.h5"
    with h5py.File(path, "w") as f:
        for i, length in enumerate((7, 3)):
            f.create_dataset(f"videos/{i}/frames", data=np.random.default_rng(i).integers(0, 256, (length, 224, 224, 3), dtype=np.uint8))
    dataset = H5VideoDataset(str(path), clip_len=5, context_stride=2, sample_all=True,
                             sample_all_stride=3, full_episode=True, transport_frames_as_uint8=True)
    assert len(dataset[0]["frames_full"]) == 7
    assert dataset[0]["target_steps"].tolist() == [0, 3, 6]
    encoder = make_encoder(memory_config={"dual_branch": dual_branch}).eval()
    loader = torch.utils.data.DataLoader(dataset, batch_size=1, collate_fn=collate_episodes)
    sparse = extract_embeddings(encoder, loader, torch.device("cpu"))
    dataset.sample_all_stride = 1
    dense = extract_embeddings(encoder, loader, torch.device("cpu"))
    np.testing.assert_allclose(sparse[0]["embeddings"], dense[0]["embeddings"][[0, 3, 6]], atol=1e-6)
    cache = {str(i): torch.randn(n, 128) for i, n in enumerate((7, 3))}
    cached = FeatureCacheDataset(cache, dataset, input_key="raw_embeddings")
    batch = collate_episodes([cached[0], cached[1]])
    assert [len(e) for e in batch["raw_embeddings"]] == [7, 3]
    assert batch["seq_len"].tolist() == [7, 3]
    assert batch["target_steps"].shape == (2, 5)


@pytest.mark.parametrize("cache_kind,dual_branch,branch_output_l2,history_settings", [
    ("joint", False, True, {}), ("raw", False, True, {}),
    ("backbone", False, True, {}), ("only_bn", False, True, {}),
    ("raw", True, True, {}), ("raw", True, False, {}),
    ("only_bn", True, True, {}), ("only_bn", True, False, {}),
    pytest.param("raw", True, True, DUAL_INDEPENDENT_HISTORY, id="raw-hf4-long20-short5"),
])
def test_training_eval_and_wandb(
        cpu_memory, training_module, monkeypatch, tmp_path, cache_kind, dual_branch, branch_output_l2,
        history_settings):
    """Exercise both existing loops, real loss and all three real eval tasks."""
    training = training_module
    from fineprog.utils.config_v2 import ConfigV2

    class HookBackbone(TinyBackbone):
        def forward(self, x):
            return self.bn(self.conv(F.adaptive_avg_pool2d(x, 14))).repeat(1, 128, 1, 1)

    class HookTemporal(nn.Module):
        def __init__(self, in_channels, hidden_channels, embed_dim, debug=False):
            super().__init__()
            self.conv1 = nn.Conv3d(1024, 4, 1)
            self.conv2 = nn.Conv3d(4, 4, 1)
            self.proj = nn.Linear(4, embed_dim)

        def forward(self, x):
            b, t, c, channels, h, w = x.shape
            x = x.permute(0, 1, 3, 2, 4, 5).reshape(b*t, channels, c, h, w)
            x = self.conv2(self.conv1(x)).mean((2, 3, 4))
            return self.proj(x).reshape(b, t, -1)

    monkeypatch.setattr(encoder_module, "ResNet50Conv4cBackbone", HookBackbone)
    monkeypatch.setattr(encoder_module, "TCCTemporalEmbedder", HookTemporal)
    built = []
    def build_training_encoder(*args, **kwargs):
        result = checkpoints.build_training_encoder(*args, **kwargs)
        built.append(result[0])
        return result
    monkeypatch.setattr(training, "build_training_encoder", build_training_encoder)
    monkeypatch.setattr(training, "checkpoint_dict", checkpoints.checkpoint_dict)
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    root = Path(__file__).resolve().parents[1]
    configs = tmp_path / "configs"
    shutil.copytree(root / "configs_v2", configs)
    h5_path = tmp_path / "short.h5"
    with h5py.File(h5_path, "w") as f:
        for i, length in enumerate((4, 6)):
            f.create_dataset(f"videos/{i}/frames", data=np.random.default_rng(i).integers(0, 256, (length, 224, 224, 3), dtype=np.uint8))
    (configs / "registry/datasets.yaml").write_text(yaml.safe_dump({"datasets": {"short": {"processed_h5": str(h5_path)}}}))
    config = yaml.safe_load((configs / "train.yaml").read_text())
    config.update(train_dataset="short", pretrained=False, batch_size=2, clip_len=3,
                  context_stride=2, num_workers=0, train_base="train_all" if cache_kind == "joint" else "frozen",
                  train_embedding=cache_kind != "raw", extract_backbone_cache=cache_kind == "backbone")
    config["memory"]["enabled"] = True
    if dual_branch:
        config["embedding_normalization"] = "l2"
        config["memory"].update(dual_branch=True, long_history_frames=3,
                                short_history_frames=2, history_stride=2,
                                branch_output_l2=branch_output_l2)
        config["memory"].update(history_settings)
    if cache_kind == "only_bn":
        config["train_base"] = "only_bn"
        config["memory"]["freeze_bn_stats"] = False
    config["in_training_eval"]["eval_dataset_ref"] = "short"
    config["in_training_eval"]["kendalls_tau"]["kendall_stride"] = 1
    if cache_kind == "raw":
        local = TCCEncoder(pretrained=False, context_stride=2,
                           embedding_normalization="l2" if dual_branch else "none")
        local_path = tmp_path / "local.pt"
        torch.save(checkpoints.checkpoint_dict(local), local_path)
        config["memory"]["local_checkpoint_path"] = str(local_path)
    train_path = configs / "train.yaml"
    train_path.write_text(yaml.safe_dump(config))
    monkeypatch.setattr(training, "_TRAIN_V2", ConfigV2(configs).load_train())
    monkeypatch.setattr(training, "_V2_TRAIN_YAML", str(train_path))
    monkeypatch.setattr(training, "_V2_LOSS_YAML", str(configs / config["loss_config"]))
    logs, uploaded_configs, run_config = [], [], {}
    def init(**kwargs):
        run_config.update(kwargs["config"])
        monkeypatch.setattr(training.wandb, "run", SimpleNamespace(name="smoke"))
    monkeypatch.setattr(training.wandb, "init", init)
    monkeypatch.setattr(training.wandb, "log", lambda payload, **kwargs: logs.append((payload, kwargs)))
    monkeypatch.setattr(training.wandb, "save", lambda path, **kwargs: uploaded_configs.append(path))
    monkeypatch.setattr(training.wandb, "finish", lambda: None)
    monkeypatch.setattr(training.wandb, "Image", lambda path, **kwargs: str(path))
    monkeypatch.setattr(training.wandb, "Video", lambda path, **kwargs: str(path))
    training.train(num_epochs=1, batch_size=2, num_workers=0, log_every=1, checkpoint_every=1,
                   checkpoint_dir=str(tmp_path / "checkpoints"))
    checkpoint_path = tmp_path / "checkpoints/smoke/encoder_epoch000001.pt"
    restored = checkpoints.build_encoder_from_checkpoint(checkpoint_path, "cpu")
    restored.eval().train()
    assert restored.backbone.bn.training == (cache_kind == "only_bn")
    assert built[0].backbone.bn.training == (cache_kind == "only_bn")
    if cache_kind == "only_bn":
        assert built[0].backbone.bn.num_batches_tracked.item() == 2
    assert restored.backbone.conv.weight.requires_grad == (cache_kind == "joint")
    norm_logs = [p for p, _ in logs if "memory/input_norm_mean" in p]
    assert len(norm_logs) == 1
    if dual_branch:
        assert not any("memory/alpha" in p for p, _ in logs)
        assert restored.memory_config["dual_branch"] is True
        assert restored.memory_config["branch_output_l2"] is branch_output_l2
        for name, value in history_settings.items():
            assert restored.memory_config[name] == run_config["memory"][name] == value
        assert norm_logs[0]["memory/long_weight_mean"] == 0.5
        assert norm_logs[0]["memory/short_weight_mean"] == 0.5
    else:
        assert norm_logs[0]["memory/alpha"] == restored.memory.alpha.item()
        assert "memory/long_weight_mean" not in norm_logs[0]
        assert "memory/short_weight_mean" not in norm_logs[0]
    for key in ("memory/input_norm_mean", "memory/pre_normalization_norm_mean"):
        assert key in norm_logs[0] and np.isfinite(norm_logs[0][key])
        assert norm_logs[0][key] > 0
    assert run_config["memory"]["enabled"] and str(train_path) in uploaded_configs
    keys = set().union(*(p for p, _ in logs))
    for task in ("latent_distance_heatmap", "kendalls_tau", "activation_map"):
        assert any(key.startswith(f"eval/train/{task}/") for key in keys), (task, keys)
    assert len(list((tmp_path / "checkpoints").rglob("*.mp4"))) == 2
    assert not list((tmp_path / "checkpoints").rglob("tmp_eval_embd.h5"))


@pytest.mark.skipif(not torch.cuda.is_available() or importlib.util.find_spec("mamba_ssm") is None,
                    reason="Official Mamba2 and CUDA required")
@pytest.mark.parametrize("branch_output_l2", [False, True])
@pytest.mark.parametrize("history_settings", [
    pytest.param({}, id="legacy-hf8-hf2-hs20"),
    pytest.param(DUAL_INDEPENDENT_HISTORY, id="hf4-long20-short5"),
])
def test_official_cuda_dual_amp_sequence_step_two_step_backward(branch_output_l2, history_settings):
    from fineprog.models.streaming_mamba2 import DualStreamingMamba2

    torch.manual_seed(42)
    memory = DualStreamingMamba2(128, {"branch_output_l2": branch_output_l2,
                                     **history_settings}).cuda().train()
    raw = (1000 * torch.randn(2, 257, 128, device="cuda")).half().requires_grad_()
    target = F.normalize(torch.randn(2, 257, 128, device="cuda"), dim=-1)
    optimizer = torch.optim.Adam(memory.parameters(), lr=4e-4)
    scaler = torch.amp.GradScaler("cuda")
    hidden_before = memory.gate[0].weight.detach().clone()
    for iteration in range(2):
        optimizer.zero_grad()
        raw.grad = None
        stats = {}
        valid = torch.ones(2, 257, device="cuda", dtype=torch.bool)
        valid[1, -6:] = False
        with torch.autocast("cuda"):
            y = memory(raw)
            fused = memory.fuse(raw, y, weight_stats=stats, valid=valid)
            embeddings = F.normalize(fused, dim=-1)
            loss = -(embeddings * target).sum(-1).mean()
        torch.testing.assert_close(fused, memory.fuse(raw, y), rtol=0, atol=0)
        assert not stats["branch_weight_sum"].requires_grad
        torch.testing.assert_close(stats["branch_weight_sum"].sum(), valid.sum().float())
        if iteration == 0:
            torch.testing.assert_close(stats["branch_weight_sum"], torch.full((2,), 254., device="cuda"))
        assert y.shape == (2, 257, 2, 128)
        assert y.dtype == fused.dtype == torch.float32
        assert torch.isfinite(y).all() and torch.isfinite(loss)
        assert embeddings.var(dim=1).sum(-1).min() > 1e-4
        if iteration == 0:
            state = memory.init_stream_state(2)
            with torch.no_grad(), torch.autocast("cuda"):
                outputs = []
                for t in range(raw.shape[1]):
                    outputs.append(memory.step(raw[:, t], state))
                    state["steps"].add_(1)
                streamed = torch.stack(outputs, dim=1)
                torch.testing.assert_close(y, streamed, rtol=2e-3, atol=5e-4)
                torch.testing.assert_close(embeddings, F.normalize(memory.fuse(raw, streamed), dim=-1),
                                           rtol=2e-3, atol=5e-4)
                torch.testing.assert_close(state["local_history"],
                                           F.normalize(raw[:, -memory.history_span:].float(), dim=-1))
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        assert raw.grad is not None and torch.isfinite(raw.grad).all() and raw.grad.abs().sum() > 0
        modules = [memory.long_proj, memory.short_proj, memory.block, memory.proj, memory.gate[-1]]
        if iteration == 1:
            modules.append(memory.gate[0])
        before = []
        for module in modules:
            params = list(module.parameters())
            assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in params)
            assert sum(p.grad.abs().sum() for p in params) > 0
            before.append([p.detach().clone() for p in params])
        scaler.step(optimizer)
        scaler.update()
        for module, initial in zip(modules, before):
            assert any(not torch.equal(p, previous) for p, previous in zip(module.parameters(), initial))
    assert not torch.equal(memory.gate[0].weight, hidden_before)


@pytest.mark.skipif(not torch.cuda.is_available() or importlib.util.find_spec("mamba_ssm") is None,
                    reason="Official Mamba2 and CUDA required")
@pytest.mark.parametrize("history_frames,history_stride", [(1, 1), (4, 5), (8, 20)])
@pytest.mark.parametrize("input_dtype", [torch.float32, torch.float16])
def test_official_cuda_normalized_large_raw_amp_sequence_step_backward(
        history_frames, history_stride, input_dtype):
    from fineprog.models.streaming_mamba2 import StreamingMamba2
    torch.manual_seed(42)
    memory = StreamingMamba2(128, {
        "normalize_input": True, "use_residual": False,
        "history_frames": history_frames, "history_stride": history_stride}).cuda().train()
    raw = (1000 * torch.randn(2, 257, 128, device="cuda")).to(input_dtype).requires_grad_()
    target = F.normalize(torch.randn(2, 257, 128, device="cuda"), dim=-1)
    optimizer = torch.optim.Adam([p for p in memory.parameters() if p.requires_grad], lr=4e-4)
    scaler = torch.amp.GradScaler("cuda")
    with torch.autocast("cuda"):
        y = memory(raw)
        embeddings = F.normalize(memory.fuse(raw, y), dim=-1)
        # A direction-dependent objective remains informative after final L2.
        loss = -(embeddings * target).sum(-1).mean()
    assert y.dtype == torch.float32 and torch.isfinite(y).all()
    assert (y.count_nonzero(dim=-1) > 0).all()
    assert torch.isfinite(loss) and torch.isfinite(embeddings).all()
    assert embeddings.var(dim=1).sum(-1).min() > 1e-4
    conv, ssm = memory.allocate_cache(2)
    state = {"conv": conv, "ssm": ssm,
             "steps": torch.zeros(2, dtype=torch.long, device="cuda")}
    if history_frames > 1:
        state["raw_history"] = torch.zeros(2, memory.history_span, 128, device="cuda")
    with torch.no_grad(), torch.autocast("cuda"):
        outputs = []
        for t in range(raw.shape[1]):
            outputs.append(memory.step(raw[:, t], state))
            state["steps"].add_(1)
        # Parallel scan and recurrent stepping associate FP32 sums differently;
        # allow small absolute error near zero over the 257-token sequence.
        torch.testing.assert_close(y, torch.stack(outputs, dim=1), rtol=2e-3, atol=5e-4)
        if history_frames > 1:
            torch.testing.assert_close(state["raw_history"],
                                       raw[:, -memory.history_span:].float(), rtol=0, atol=0)
    scaler.scale(loss).backward()
    scaler.unscale_(optimizer)
    assert raw.grad is not None and torch.isfinite(raw.grad).all() and raw.grad.abs().sum() > 0
    modules = [memory.block, memory.proj]
    if history_frames > 1:
        modules.append(memory.history_proj)
    before = []
    for module in modules:
        params = list(module.parameters())
        assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in params)
        assert sum(p.grad.abs().sum() for p in params) > 0
        before.append([p.detach().clone() for p in params])
    assert memory.alpha.grad is None
    scaler.step(optimizer)
    scaler.update()
    for module, initial in zip(modules, before):
        assert any(not torch.equal(p, previous) for p, previous in zip(module.parameters(), initial))


@pytest.mark.skipif(not torch.cuda.is_available() or importlib.util.find_spec("mamba_ssm") is None,
                    reason="Official Mamba2 and CUDA required")
@pytest.mark.parametrize("length", [1, 3, 17, 257])
def test_official_cuda_sequence_step_and_backward(length):
    from fineprog.models.streaming_mamba2 import StreamingMamba2
    torch.manual_seed(7)
    memory = StreamingMamba2(128, {}).cuda().eval()
    x = torch.randn(2, length, 128, device="cuda", requires_grad=True)
    y = memory(x)
    conv, ssm = memory.allocate_cache(2)
    state = {"conv": conv, "ssm": ssm}
    with torch.no_grad():
        stepped = torch.stack([memory.step(x[:, t], state) for t in range(length)], dim=1)
    torch.testing.assert_close(y, stepped, rtol=2e-3, atol=2e-4)
    history_grad, = torch.autograd.grad(y[:, -1].square().mean(), x)
    assert torch.isfinite(history_grad).all() and history_grad[:, 0].abs().sum() > 0
    with torch.autocast("cuda"):
        out = memory.fuse(x, memory(x))
        loss = out[:, -1].square().mean()
    optimizer = torch.optim.Adam(memory.parameters(), lr=1e-4)
    scaler = torch.amp.GradScaler("cuda")
    scaler.scale(loss).backward()
    scaler.unscale_(optimizer)
    # FP16 can underflow after long natural decay. Check beyond the convolution
    # window (and across the scan boundary for T=257); FP32 checked frame 0 above.
    assert torch.isfinite(x.grad).all() and x.grad[:, max(0, length - 8)].abs().sum() > 0
    assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in memory.parameters())


@pytest.mark.skipif(not torch.cuda.is_available() or importlib.util.find_spec("mamba_ssm") is None,
                    reason="Official Mamba2 and CUDA required")
def test_official_cuda_history_sequence_step_and_backward():
    from fineprog.models.streaming_mamba2 import StreamingMamba2
    torch.manual_seed(7)
    memory = StreamingMamba2(128, {"history_frames": 3, "history_stride": 2}).cuda().eval()
    raw = torch.randn(2, 17, 128, device="cuda", requires_grad=True)
    with torch.autocast("cuda"):
        y = memory(raw)
        loss = memory.fuse(raw, y).square().mean()
    conv, ssm = memory.allocate_cache(2)
    state = {"conv": conv, "ssm": ssm,
             "raw_history": raw.new_zeros(2, memory.history_span, 128),
             "steps": torch.zeros(2, dtype=torch.long, device="cuda")}
    with torch.no_grad():
        outputs = []
        for t in range(raw.shape[1]):
            outputs.append(memory.step(raw[:, t], state))
            state["steps"].add_(1)
        torch.testing.assert_close(y, torch.stack(outputs, dim=1), rtol=2e-3, atol=2e-4)
    loss.backward()
    assert raw.grad is not None and torch.isfinite(raw.grad).all()
    assert raw.grad[:, 0].abs().sum() > 0
    assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in memory.parameters())
    assert memory.history_proj.weight.grad.abs().sum() > 0


@pytest.mark.skipif(not torch.cuda.is_available() or importlib.util.find_spec("mamba_ssm") is None,
                    reason="Official Mamba2 and CUDA required")
@pytest.mark.parametrize("input_dtype", [torch.float32, torch.float16])
def test_official_cuda_large_raw_amp_forward_backward(input_dtype):
    """Warm-started raw embeddings can have norms near 1e4 before final L2."""
    from fineprog.models.streaming_mamba2 import StreamingMamba2
    torch.manual_seed(42)
    memory = StreamingMamba2(128, {}).cuda().train()
    # Cover both FP32 frozen caches and FP16 visual outputs, crossing a scan chunk.
    raw = (1000 * torch.randn(2, 257, 128, device="cuda")).to(input_dtype).requires_grad_()
    target = F.normalize(torch.randn(2, 128, device="cuda"), dim=-1)
    optimizer = torch.optim.Adam(memory.parameters(), lr=4e-4)
    scaler = torch.amp.GradScaler("cuda")
    with torch.no_grad():
        expected = memory.block(raw.float())
    with torch.autocast("cuda"):
        y = memory(raw)
        embeddings = F.normalize(memory.fuse(raw, y), dim=-1)
        loss = (embeddings[:, -1] - target).square().mean()
    assert torch.isfinite(y).all() and torch.isfinite(loss)
    torch.testing.assert_close(y, expected, rtol=2e-3, atol=2e-4)
    scaler.scale(loss).backward()
    scaler.unscale_(optimizer)
    assert raw.grad is not None and torch.isfinite(raw.grad).all()
    assert raw.grad.abs().sum() > 0
    assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in memory.parameters())
    alpha_before = memory.alpha.detach().clone()
    scaler.step(optimizer)
    scaler.update()
    assert optimizer.state[memory.alpha]["step"].item() == 1
    assert not torch.equal(memory.alpha, alpha_before)


@pytest.mark.skipif(not torch.cuda.is_available() or importlib.util.find_spec("mamba_ssm") is None,
                    reason="Official Mamba2 and CUDA required")
@pytest.mark.parametrize("train_base,freeze_bn_stats", [("train_all", True), ("only_bn", False)])
def test_official_cuda_real_local_joint_and_stream(tmp_path, train_base, freeze_bn_stats):
    torch.manual_seed(7)
    encoder = make_encoder(train_base=train_base, train_embedding=True,
                           memory_config={"freeze_bn_stats": freeze_bn_stats}).cuda().train()
    frames = torch.rand(6, 3, 224, 224, requires_grad=True)
    optimizer = torch.optim.Adam(encoder.parameters(), lr=1e-4)
    scaler = torch.amp.GradScaler("cuda")
    with torch.autocast("cuda"):
        raw = encoder.encode_local_episode(frames)
        result = encoder.forward_episode([raw], [6], torch.tensor([[5]]))
        loss = result.square().mean()
    scaler.scale(loss).backward()
    scaler.unscale_(optimizer)
    # Frame 0 is outside the selected two-frame local context (frames 3 and 5).
    assert frames.grad[0].abs().sum() > 0 and torch.isfinite(frames.grad).all()
    for module in (encoder.backbone, encoder.temporal_embedder, encoder.memory):
        trainable = [p for p in module.parameters() if p.requires_grad]
        assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in trainable)
        assert sum(p.grad.abs().sum() for p in trainable) > 0
        assert all(p.grad is None for p in module.parameters() if not p.requires_grad)
    for module in encoder.backbone.modules():
        if isinstance(module, nn.modules.batchnorm._BatchNorm):
            assert module.num_batches_tracked.item() == (0 if freeze_bn_stats else 3)
    scaler.step(optimizer)
    scaler.update()
    encoder.eval()
    with torch.no_grad():
        raw = encoder.encode_local_episode(frames.detach())
        full = encoder.forward_sequence([raw], [6])
        state = encoder.init_stream_state(1)
        # encode_step must remain consistent even inside an outer AMP context.
        with torch.autocast("cuda"):
            streamed = torch.stack([encoder.encode_step(frame[None].cuda(), state)[0]
                                    for frame in frames.detach()], dim=1)
        torch.testing.assert_close(full, streamed, rtol=2e-3, atol=2e-4)
    path = tmp_path / "real_memory.pt"
    torch.save(checkpoints.checkpoint_dict(encoder), path)
    restored = checkpoints.build_encoder_from_checkpoint(path, "cuda").eval()
    with torch.no_grad():
        torch.testing.assert_close(restored.forward_sequence([raw], [6]), full)
