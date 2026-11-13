"""Cross-attention wrapper semantics, plus a short real-Mamba CUDA check."""

import copy
import importlib.util
from pathlib import Path
import sys
from types import SimpleNamespace

import pytest
import torch
from torch import nn
from torch.nn import functional as F

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from fineprog.models.streaming_mamba2 import CrossAttentionStreamingMamba2, StreamingMamba2


class TinyCausalBlock(nn.Module):
    """A recurrence for wrapper tests, not a replacement for Mamba validation."""
    def __init__(self, d_model, **kwargs):
        super().__init__()
        self.proj = nn.Linear(d_model, d_model, bias=False)
        self.forward_calls = 0
        self.step_calls = 0

    def forward(self, inputs):
        self.forward_calls += 1
        return self.proj(inputs).cumsum(dim=1)

    def allocate_inference_cache(self, batch_size, max_seqlen):
        state = self.proj.weight.new_zeros(batch_size, self.proj.in_features)
        return state.clone(), state.clone()

    def step(self, inputs, conv, ssm):
        self.step_calls += 1
        conv.copy_(self.proj(inputs[:, 0]))
        ssm.add_(conv)
        return ssm.unsqueeze(1), conv, ssm


@pytest.fixture
def cpu_mamba(monkeypatch):
    monkeypatch.setitem(sys.modules, "mamba_ssm", SimpleNamespace(Mamba2=TinyCausalBlock))
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    torch.manual_seed(17)
    yield
    torch.set_num_threads(previous)


def make_memory(**config):
    return CrossAttentionStreamingMamba2(4, {"normalize_input": True, **config})


def run_full(memory, raw, *, lengths=None, stats=None, diagnostics=None, history_mask=None):
    projected = memory(raw, lengths=lengths, stats=stats, diagnostics=diagnostics,
                       history_mask=history_mask)
    valid = (None if lengths is None else
             torch.arange(raw.shape[1], device=raw.device)[None] < torch.as_tensor(lengths, device=raw.device)[:, None])
    output = memory.fuse(raw, projected, weight_stats=stats, valid=valid, diagnostics=diagnostics)
    return projected, output


@pytest.mark.parametrize("attention_mode", ["learned", "uniform", "none"])
@pytest.mark.parametrize("branch_output_l2", [False, True])
def test_full_streaming_causality_and_empty_rows(cpu_mamba, attention_mode, branch_output_l2):
    memory = make_memory(attention_mode=attention_mode, branch_output_l2=branch_output_l2).eval()
    raw = torch.randn(2, 7, 4)
    diagnostics = {}
    with torch.no_grad():
        projected, output = run_full(memory, raw, diagnostics=diagnostics)
        state = memory.init_stream_state(2)
        streamed = []
        for t in range(raw.shape[1]):
            step_diagnostics = {}
            y = memory.step(raw[:, t], state, diagnostics=step_diagnostics)
            streamed.append(memory.fuse(raw[:, t], y, diagnostics=step_diagnostics))
            torch.testing.assert_close(step_diagnostics["history_lengths"], torch.full((2,), t))
            if t == 0:
                assert step_diagnostics["attention"].shape == (2, 0)
            state["steps"].add_(1)
        torch.testing.assert_close(output, torch.stack(streamed, dim=1), rtol=1e-5, atol=1e-6)
        changed = raw.clone()
        changed[:, 4:] += 100
        changed_projected, _ = run_full(memory, changed)
        torch.testing.assert_close(projected[:, :4], changed_projected[:, :4], rtol=0, atol=0)
    attention = diagnostics["attention"]
    assert torch.isfinite(output).all() and torch.isfinite(attention).all()
    assert not attention.masked_select(~diagnostics["attention_valid"]).any()
    expected_sum = torch.zeros(2, 7) if attention_mode == "none" else torch.tensor([[0., 1., 1., 1., 1., 1., 1.]]).expand(2, -1)
    torch.testing.assert_close(attention.sum(-1), expected_sum)
    assert state["keys"].shape[1] >= 7
    torch.testing.assert_close(state["values"][:, :7], memory._prepare_input(raw.unsqueeze(-2)))


def test_padding_mask_replay_and_statistics_use_real_full_episode(cpu_mamba):
    memory = make_memory()
    raw = torch.randn(2, 5, 4)
    lengths = [5, 1]
    stats, diagnostics = {}, {}
    projected, output = run_full(memory, raw, lengths=lengths, stats=stats, diagnostics=diagnostics)
    assert stats["num_frames"] == 6
    assert stats["num_history_frames"] == 4
    valid = torch.arange(5)[None] < torch.tensor(lengths)[:, None]
    history_valid = valid & (diagnostics["history_lengths"] > 0)
    for stat, diagnostic in (("current_component_norm_sum", "current_component_norm"),
                             ("history_component_norm_sum", "history_component_norm"),
                             ("history_injection_ratio_sum", "history_injection_ratio")):
        torch.testing.assert_close(stats[stat], diagnostics[diagnostic][history_valid].sum())
    assert all(not value.requires_grad and value.grad_fn is None and value.numel() <= 2
               for value in stats.values())
    assert not diagnostics["attention_valid"][1].any()
    assert not diagnostics["attention"][1].any()
    for b, length in enumerate(lengths):
        individual = memory(raw[b:b + 1, :length])
        torch.testing.assert_close(projected[b, :length], individual[0])
    history_mask = torch.ones(2, 5, dtype=torch.bool)
    history_mask[:, 1:3] = False
    replay_diagnostics = {}
    replay = memory(raw, lengths=lengths, history_mask=history_mask, diagnostics=replay_diagnostics)
    torch.testing.assert_close(projected[..., 1, :], replay[..., 1, :], rtol=0, atol=0)
    torch.testing.assert_close(projected[:, :2], replay[:, :2], rtol=0, atol=0)
    assert not replay_diagnostics["attention"][..., 1:3].any()
    assert not torch.equal(projected[0, 2:, 0], replay[0, 2:, 0])
    empty_stats, empty_diagnostics = {}, {}
    run_full(memory, raw[:, :1], stats=empty_stats, diagnostics=empty_diagnostics)
    assert empty_stats["num_frames"] == 2 and empty_stats["num_history_frames"] == 0
    assert empty_stats["current_component_norm_sum"] == 0
    assert (empty_diagnostics["current_component_norm"] > 0).all()


def test_nonfinite_padding_is_not_an_observation(cpu_mamba):
    memory = make_memory()
    raw = torch.randn(2, 6, 4)
    poisoned = raw.clone()
    poisoned[1, 2:4] = torch.nan
    poisoned[1, 4:] = torch.inf
    expected, _ = run_full(memory, raw, lengths=[6, 2])
    stats, diagnostics = {}, {}
    actual, output = run_full(memory, poisoned, lengths=[6, 2], stats=stats, diagnostics=diagnostics)
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    assert torch.isfinite(output).all()
    assert all(torch.isfinite(value).all() for value in stats.values())
    assert torch.isfinite(diagnostics["attention"]).all()


@pytest.mark.parametrize("mode,weights", [("learned", [0.1, 0.9]), ("fixed", [0.3, 0.7]),
                                         ("short_only", [0., 1.]), ("long_only", [1., 0.])])
def test_effective_fusion_weights_and_exact_short_source(cpu_mamba, mode, weights):
    memory = make_memory(fusion_mode=mode, fixed_fusion_weights=[0.3, 0.7])
    raw = torch.randn(2, 5, 4)
    stats, diagnostics = {}, {}
    projected, output = run_full(memory, raw, stats=stats, diagnostics=diagnostics)
    torch.testing.assert_close(stats["branch_weight_sum"], torch.tensor(weights) * 10)
    torch.testing.assert_close(diagnostics["branch_weights"], torch.tensor(weights).expand(2, 5, 2))
    if mode == "short_only":
        source = StreamingMamba2(4, {"use_residual": False, "normalize_input": True})
        source.load_state_dict({name: value for name, value in memory.state_dict().items()
                                if name in source.state_dict()}, strict=True)
        expected = source.fuse(raw, source(raw))
        torch.testing.assert_close(output, expected, rtol=0, atol=0)
    else:
        expected = (F.normalize(projected, dim=-1) * torch.tensor(weights).view(1, 1, 2, 1)).sum(-2)
        torch.testing.assert_close(output, expected)


def test_diagnostics_do_not_repeat_modules_or_change_outputs_gradients(cpu_mamba):
    memory = make_memory()
    raw = torch.randn(2, 7, 4)
    reference = copy.deepcopy(memory)
    probe = torch.randn(2, 7, 4)
    _, expected = run_full(reference, raw)
    (expected * probe).sum().backward()
    calls = []
    hook = memory.proj.register_forward_hook(lambda *args: calls.append(1))
    stats, diagnostics = {}, {}
    _, output = run_full(memory, raw, stats=stats, diagnostics=diagnostics)
    hook.remove()
    (output * probe).sum().backward()
    torch.testing.assert_close(output, expected, rtol=0, atol=0)
    assert memory.block.forward_calls == 2 and len(calls) == 2
    for (name, param), (_, other) in zip(memory.named_parameters(), reference.named_parameters()):
        if param.requires_grad:
            assert param.grad is not None and torch.isfinite(param.grad).all(), name
            torch.testing.assert_close(param.grad, other.grad, rtol=0, atol=0)
        else:
            assert param.grad is None, name
    assert memory.current_proj.weight.grad.abs().sum() > 0
    assert memory.retrieval_proj.weight.grad.abs().sum() > 0
    assert memory.query_proj.weight.grad.abs().sum() > 0
    assert all(not value.requires_grad for value in diagnostics.values())


def test_partial_reset_and_single_state_advance(cpu_mamba):
    memory = make_memory().eval()
    raw = torch.randn(2, 7, 4)
    state, plain_state = memory.init_stream_state(2), memory.init_stream_state(2)
    with torch.no_grad():
        for t in range(7):
            if t == 3:
                for stream in (state, plain_state):
                    for value in stream.values():
                        value[0] = 0
            diag, stats = {}, {}
            result = memory.step(raw[:, t], state, stats=stats, diagnostics=diag)
            plain = memory.step(raw[:, t], plain_state)
            memory.fuse(raw[:, t], result, weight_stats=stats, diagnostics=diag)
            torch.testing.assert_close(result, plain, rtol=0, atol=0)
            for key in state:
                torch.testing.assert_close(state[key], plain_state[key], rtol=0, atol=0)
            if t == 3:
                assert diag["history_lengths"].tolist() == [0, 3]
                assert stats["num_history_frames"] == 1
                fresh = memory(raw[:1, t:t + 1])
                torch.testing.assert_close(result[0], fresh[0, 0])
        torch.testing.assert_close(state["kv_lengths"], torch.tensor([4, 7]))
    assert state["conv_long"].data_ptr() != state["conv_short"].data_ptr()
    assert state["ssm_long"].data_ptr() != state["ssm_short"].data_ptr()
    assert memory.block.step_calls == 28  # Two branches, two independent runs, seven steps.


@pytest.mark.parametrize("attention_mode", ["uniform", "none"])
def test_disabled_gradient_paths_and_none_statistics(cpu_mamba, attention_mode):
    memory = make_memory(attention_mode=attention_mode, fusion_mode="fixed")
    raw = torch.randn(2, 4, 4)
    stats = {}
    _, output = run_full(memory, raw, stats=stats)
    (output * torch.randn_like(output)).sum().backward()
    assert not memory.query_proj.weight.requires_grad and not memory.key_proj.weight.requires_grad
    assert not any(p.requires_grad for p in memory.gate.parameters())
    assert memory.current_proj.weight.grad is not None
    assert stats["num_history_frames"] == 6
    if attention_mode == "none":
        assert not memory.retrieval_proj.weight.requires_grad
        assert stats["history_component_norm_sum"] == 0
        assert stats["history_injection_ratio_sum"] == 0


def test_validation_and_state_dict_roundtrip(cpu_mamba):
    for invalid in ({"history_frames": 2}, {"use_residual": True}, {"attention_dim": 0},
                    {"attention_mode": "topk"}, {"fusion_mode": "mean"},
                    {"fixed_fusion_weights": [0.2, 0.2]}, {"branch_output_l2": "true"}):
        with pytest.raises(ValueError):
            make_memory(**invalid)
    memory = make_memory(attention_dim=None)
    assert memory.attention_dim == 4
    restored = CrossAttentionStreamingMamba2(4, memory.config)
    restored.load_state_dict(memory.state_dict(), strict=True)
    raw = torch.randn(2, 3, 4)
    torch.testing.assert_close(restored(raw), memory(raw), rtol=0, atol=0)
    with pytest.raises(ValueError):
        memory(raw, lengths=[3, 0])
    with pytest.raises(ValueError):
        memory(raw, history_mask=torch.ones(2, 3))


@pytest.mark.skipif(not torch.cuda.is_available() or importlib.util.find_spec("mamba_ssm") is None,
                    reason="Official Mamba2 and CUDA required")
def test_official_cuda_sequence_stream_and_frozen_memory_backward():
    torch.manual_seed(42)
    memory = CrossAttentionStreamingMamba2(128, {"normalize_input": True}).cuda().eval()
    raw = (1000 * torch.randn(2, 17, 128, device="cuda")).half()
    stats, diagnostics = {}, {}
    with torch.autocast("cuda"):
        projected, output = run_full(memory, raw, stats=stats, diagnostics=diagnostics)
        loss = (F.normalize(output, dim=-1) * torch.randn_like(output)).sum()
    loss.backward()
    for name, parameter in memory.named_parameters():
        if parameter.requires_grad:
            assert parameter.grad is not None and torch.isfinite(parameter.grad).all(), name
        else:
            assert parameter.grad is None, name
    assert memory.current_proj.weight.grad.abs().sum() > 0
    assert memory.retrieval_proj.weight.grad.abs().sum() > 0
    with torch.no_grad(), torch.autocast("cuda"):
        state = memory.init_stream_state(2)
        streamed = torch.stack([memory.step(raw[:, t], state) for t in range(raw.shape[1])], dim=1)
        torch.testing.assert_close(projected, streamed, rtol=2e-3, atol=5e-4)
        torch.testing.assert_close(output, memory.fuse(raw, streamed), rtol=2e-3, atol=5e-4)
    assert projected.dtype == output.dtype == torch.float32
    assert torch.isfinite(output).all() and stats["num_frames"] == 34


@pytest.mark.skipif(not torch.cuda.is_available() or importlib.util.find_spec("mamba_ssm") is None,
                    reason="Official Mamba2 and CUDA required")
def test_official_cuda_singleton_scan_remains_differentiable():
    memory = CrossAttentionStreamingMamba2(128, {"normalize_input": True}).cuda()
    raw = torch.randn(1, 1, 128, device="cuda")
    stats = {}
    _, output = run_full(memory, raw, stats=stats)
    (F.normalize(output, dim=-1) * torch.randn_like(output)).sum().backward()
    assert torch.isfinite(memory.current_proj.weight.grad).all()
    assert memory.current_proj.weight.grad.abs().sum() > 0
    for name, parameter in memory.named_parameters():
        if parameter.requires_grad:
            assert parameter.grad is not None and torch.isfinite(parameter.grad).all(), name
    assert stats["num_history_frames"] == 0


@pytest.mark.skipif(not torch.cuda.is_available() or importlib.util.find_spec("mamba_ssm") is None,
                    reason="Official Mamba2 and CUDA required")
@pytest.mark.parametrize("length", [1, 3, 257])
def test_official_cuda_short_and_cross_chunk_lengths_with_padding(length):
    torch.manual_seed(43)
    memory = CrossAttentionStreamingMamba2(128, {"normalize_input": True}).cuda().eval()
    raw = (1000 * torch.randn(2, length, 128, device="cuda")).half()
    short_length = max(1, length // 2)
    lengths = [length, short_length]
    poisoned = raw.clone()
    poisoned[1, short_length:] = torch.nan
    diagnostics = {}
    with torch.no_grad(), torch.autocast("cuda"):
        projected, output = run_full(memory, poisoned, lengths=lengths, diagnostics=diagnostics)
        clean_projected, _ = run_full(memory, raw, lengths=lengths)
        torch.testing.assert_close(projected, clean_projected, rtol=0, atol=0)
        state = memory.init_stream_state(2)
        streamed = torch.stack([memory.step(raw[:, t], state) for t in range(length)], dim=1)
        streamed_output = memory.fuse(raw, streamed)
        valid = torch.arange(length, device="cuda")[None] < torch.tensor(lengths, device="cuda")[:, None]
        torch.testing.assert_close(projected[valid], streamed[valid], rtol=2e-3, atol=5e-4)
        normalized = F.normalize(output, dim=-1)
        streamed_normalized = F.normalize(streamed_output, dim=-1)
        torch.testing.assert_close(normalized[valid], streamed_normalized[valid], rtol=2e-3, atol=5e-4)
        for row, row_length in enumerate(lengths):
            unpadded = memory(raw[row:row + 1, :row_length])
            torch.testing.assert_close(projected[row, :row_length], unpadded[0], rtol=2e-3, atol=5e-4)
        error = (projected[valid] - streamed[valid]).abs().max().item()
        normalized_error = (normalized[valid] - streamed_normalized[valid]).abs().max().item()
    assert torch.isfinite(projected).all() and torch.isfinite(output).all()
    assert not diagnostics["attention"][:, 0].any()
    assert not diagnostics["history_lengths"][:, 0].any()
    torch.testing.assert_close(normalized[valid].norm(dim=-1), torch.ones(int(valid.sum()), device="cuda"))
    print(f"CUDA length={length}: projected_max_abs={error:.9g}, normalized_max_abs={normalized_error:.9g}")
