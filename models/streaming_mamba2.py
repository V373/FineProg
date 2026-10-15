"""Mamba2 with optional causal history fusion and residual output adaptation."""

import torch
from torch import nn
from torch.nn import functional as F


class StreamingMamba2(nn.Module):
    def __init__(self, embedding_dim, config):
        super().__init__()
        self.history_frames = config.get("history_frames", 1)
        self.history_stride = config.get("history_stride", 1)
        for name in ("history_frames", "history_stride"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value < 1:
                raise ValueError(f"memory.{name} must be a positive integer.")
        self.history_span = (self.history_frames - 1) * self.history_stride
        self.normalize_input = config.get("normalize_input", False)
        if not isinstance(self.normalize_input, bool):
            raise ValueError("memory.normalize_input must be a boolean.")
        try:
            from mamba_ssm import Mamba2
        except ImportError as exc:
            raise ImportError(
                "Memory requires mamba-ssm and causal-conv1d; see requirements-mamba.txt."
            ) from exc
        self.config = {
            "d_model": embedding_dim, "d_state": 64, "d_conv": 4,
            "expand": 2, "headdim": 64, "ngroups": 1, "chunk_size": 256,
        }
        self.config.update({k: config[k] for k in self.config if k in config})
        if self.config["d_model"] != embedding_dim:
            raise ValueError("memory.d_model must equal embedding_dim (no input bridge).")
        self.block = Mamba2(**self.config, layer_idx=0)
        self.proj = nn.Linear(embedding_dim, embedding_dim)
        self.use_residual = config.get("use_residual", True)
        alpha_init = float(config.get("alpha_init", 1e-3)) if self.use_residual else 1.0
        # Keep the alpha state-dict/logging field in both modes.
        self.alpha = nn.Parameter(torch.tensor(alpha_init), requires_grad=self.use_residual)
        self.history_proj = (nn.Linear(self.history_frames * embedding_dim, embedding_dim)
                             if self.history_frames > 1 else nn.Identity())
        self.pre_norm = (nn.RMSNorm(embedding_dim, eps=1e-5, elementwise_affine=False)
                         if self.normalize_input else nn.Identity())
        # Wrapper settings belong in checkpoints, not the official Mamba2 kwargs.
        self.config.update(history_frames=self.history_frames, history_stride=self.history_stride,
                           normalize_input=self.normalize_input)

    def _prepare_input(self, window):
        """Raw [..., history_frames, D] -> mixer tokens; caches stay unnormalized."""
        if self.normalize_input:
            window = F.normalize(window, p=2, dim=-1, eps=1e-12)
        return self.pre_norm(self.history_proj(window.flatten(start_dim=-2)))

    def forward(self, raw):
        # Raw local embeddings can be large enough to overflow the FP16 SSM scan.
        with torch.autocast(device_type=raw.device.type, enabled=False):
            raw = raw.float()
            if self.history_frames > 1:
                offsets = torch.arange(self.history_frames - 1, -1, -1, device=raw.device)
                indices = (torch.arange(raw.shape[1], device=raw.device)[:, None]
                           - offsets * self.history_stride).clamp_min(0)
                window = raw[:, indices]
            else:
                window = raw.unsqueeze(-2)
            return self.block(self._prepare_input(window))

    def fuse(self, raw, memory_output):
        output = self.proj(memory_output).float()
        return raw.float() + self.alpha * output if self.use_residual else output

    def allocate_cache(self, batch_size):
        # Preserve the official transposed conv layout and weight-dependent dtype.
        return self.block.allocate_inference_cache(batch_size, max_seqlen=1)

    def init_stream_state(self, batch_size):
        conv, ssm = self.allocate_cache(batch_size)
        state = {"conv": conv, "ssm": ssm,
                 "steps": torch.zeros(batch_size, dtype=torch.long, device=conv.device)}
        if self.history_frames > 1:
            state["raw_history"] = torch.zeros(
                batch_size, self.history_span, self.proj.in_features, device=conv.device)
        return state

    def step(self, raw, state):
        with torch.autocast(device_type=raw.device.type, enabled=False):
            raw = raw.float()
            if self.history_frames > 1:
                # Fresh/reset rows repeat their first local embedding on the left.
                history = torch.where((state["steps"] == 0)[:, None, None],
                                      raw[:, None, :], state["raw_history"])
                window = torch.cat((history, raw[:, None, :]), dim=1)
                state["raw_history"] = window[:, 1:].contiguous()
                window = window[:, ::self.history_stride]
            else:
                window = raw.unsqueeze(-2)
            tokens = self._prepare_input(window)
            y, state["conv"], state["ssm"] = self.block.step(
                tokens.unsqueeze(1), state["conv"], state["ssm"]
            )
        return y[:, 0]


class DualStreamingMamba2(nn.Module):
    """Two history scales with shared weights and independent recurrent states.

    forward/step return Mamba outputs with a penultimate [long, short] axis.
    fuse projects and mixes those outputs; the encoder owns final normalization.
    """

    def __init__(self, embedding_dim, config):
        super().__init__()
        from mamba_ssm import Mamba2

        self.long_history_frames = config.get("long_history_frames", 8)
        self.short_history_frames = config.get("short_history_frames", 2)
        # Legacy configurations use one stride for both branches.
        self.history_stride = config.get("history_stride", 20)
        self.long_history_stride = config.get("long_history_stride", self.history_stride)
        self.short_history_stride = config.get("short_history_stride", self.history_stride)
        for name in ("long_history_frames", "short_history_frames",
                     "long_history_stride", "short_history_stride"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value < 1:
                raise ValueError(f"memory.{name} must be a positive integer.")
        self.history_span = max(
            (self.long_history_frames - 1) * self.long_history_stride,
            (self.short_history_frames - 1) * self.short_history_stride,
        )
        self.branch_output_l2 = config.get("branch_output_l2", True)
        self.config = {
            "d_model": embedding_dim, "d_state": 64, "d_conv": 4,
            "expand": 2, "headdim": 64, "ngroups": 1, "chunk_size": 256,
        }
        self.config.update({k: config[k] for k in self.config if k in config})
        if self.config["d_model"] != embedding_dim:
            raise ValueError("memory.d_model must equal embedding_dim (no input bridge).")
        self.block = Mamba2(**self.config, layer_idx=0)
        self.proj = nn.Linear(embedding_dim, embedding_dim)
        self.long_proj = nn.Linear(self.long_history_frames * embedding_dim, embedding_dim)
        self.short_proj = nn.Linear(self.short_history_frames * embedding_dim, embedding_dim)
        self.pre_norm = nn.RMSNorm(embedding_dim, eps=1e-5, elementwise_affine=False)
        self.gate = nn.Sequential(
            nn.Linear(2 * embedding_dim, embedding_dim), nn.SiLU(), nn.Linear(embedding_dim, 2))
        nn.init.zeros_(self.gate[-1].weight)
        nn.init.zeros_(self.gate[-1].bias)
        self.config.update(
            dual_branch=True, long_history_frames=self.long_history_frames,
            short_history_frames=self.short_history_frames, history_stride=self.history_stride,
            long_history_stride=self.long_history_stride,
            short_history_stride=self.short_history_stride,
            branch_output_l2=self.branch_output_l2, normalize_input=True, use_residual=False)

    def _prepare_input(self, window, projection):
        # The shared local embeddings have already been L2-normalized.
        return self.pre_norm(projection(window.flatten(start_dim=-2)))

    def forward(self, raw):
        with torch.autocast(device_type=raw.device.type, enabled=False):
            local = F.normalize(raw.float(), p=2, dim=-1, eps=1e-12)
            times = torch.arange(local.shape[1], device=local.device)[:, None]
            outputs = []
            for frames, stride, projection in (
                    (self.long_history_frames, self.long_history_stride, self.long_proj),
                    (self.short_history_frames, self.short_history_stride, self.short_proj)):
                offsets = torch.arange(frames - 1, -1, -1, device=local.device)
                indices = (times - offsets * stride).clamp_min(0)
                outputs.append(self.block(self._prepare_input(local[:, indices], projection)))
            return torch.stack(outputs, dim=-2)

    def fuse(self, raw, memory_output, *, weight_stats=None, valid=None):
        with torch.autocast(device_type=memory_output.device.type, enabled=False):
            branches = self.proj(memory_output.float())
            if self.branch_output_l2:
                branches = F.normalize(branches, p=2, dim=-1, eps=1e-12)
            weights = self.gate(branches.flatten(start_dim=-2)).softmax(dim=-1)
            if weight_stats is not None:
                logged_weights = weights.detach()
                if valid is not None:
                    logged_weights = logged_weights.masked_fill(~valid[..., None], 0)
                weight_sum = logged_weights.reshape(-1, 2).sum(dim=0)
                weight_stats["branch_weight_sum"] = (
                    weight_stats.get("branch_weight_sum", 0) + weight_sum)
            return (weights.unsqueeze(-1) * branches).sum(dim=-2)

    def init_stream_state(self, batch_size):
        device = self.proj.weight.device
        state = {
            "steps": torch.zeros(batch_size, dtype=torch.long, device=device),
            "local_history": torch.zeros(
                batch_size, self.history_span, self.proj.in_features, device=device),
        }
        for branch in ("long", "short"):
            state[f"conv_{branch}"], state[f"ssm_{branch}"] = (
                self.block.allocate_inference_cache(batch_size, max_seqlen=1))
        return state

    def step(self, raw, state):
        with torch.autocast(device_type=raw.device.type, enabled=False):
            local = F.normalize(raw.float(), p=2, dim=-1, eps=1e-12)
            history = torch.where((state["steps"] == 0)[:, None, None],
                                  local[:, None, :], state["local_history"])
            window = torch.cat((history, local[:, None, :]), dim=1)
            state["local_history"] = window[:, 1:].contiguous()
            outputs = []
            for branch, frames, stride, projection in (
                    ("long", self.long_history_frames, self.long_history_stride, self.long_proj),
                    ("short", self.short_history_frames, self.short_history_stride, self.short_proj)):
                start = self.history_span - (frames - 1) * stride
                tokens = self._prepare_input(window[:, start::stride], projection)
                y, state[f"conv_{branch}"], state[f"ssm_{branch}"] = self.block.step(
                    tokens.unsqueeze(1), state[f"conv_{branch}"], state[f"ssm_{branch}"])
                outputs.append(y[:, 0])
            return torch.stack(outputs, dim=-2)
