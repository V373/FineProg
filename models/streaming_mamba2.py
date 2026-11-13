"""Mamba2 with optional causal history fusion and residual output adaptation."""

import math

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


class CrossAttentionStreamingMamba2(StreamingMamba2):
    """Short-guided, past-only retrieval with shared Mamba/P and separate states.

    Unlike the legacy wrappers, forward/step return *projected* [long, short]
    tensors. fuse never applies P again. The encoder owns final normalization.
    Streaming steps are inference-only and leave the encoder's ``steps`` counter
    to its caller; ``kv_lengths`` tracks the independently reset history banks.
    """

    def __init__(self, embedding_dim, config):
        config = dict(config)
        if config.get("history_frames", 1) != 1 or config.get("use_residual", False):
            raise ValueError("Cross-attention requires a single-input, non-residual source memory.")
        config["use_residual"] = False
        super().__init__(embedding_dim, config)
        attention_dim = config.get("attention_dim")
        self.attention_dim = min(32, embedding_dim) if attention_dim is None else attention_dim
        if (isinstance(self.attention_dim, bool) or not isinstance(self.attention_dim, int)
                or self.attention_dim < 1):
            raise ValueError("memory.attention_dim must be a positive integer or null.")
        self.attention_mode = config.get("attention_mode", "learned")
        self.fusion_mode = config.get("fusion_mode", "learned")
        if self.attention_mode not in ("learned", "uniform", "none"):
            raise ValueError("memory.attention_mode must be learned, uniform or none.")
        if self.fusion_mode not in ("learned", "fixed", "short_only", "long_only"):
            raise ValueError("memory.fusion_mode must be learned, fixed, short_only or long_only.")
        self.branch_output_l2 = config.get("branch_output_l2", True)
        self.freeze_pretrained_memory = config.get("freeze_pretrained_memory", True)
        for name in ("branch_output_l2", "freeze_pretrained_memory"):
            if not isinstance(getattr(self, name), bool):
                raise ValueError(f"memory.{name} must be a boolean.")
        weights = config.get("fixed_fusion_weights", [0.1, 0.9])
        if (not isinstance(weights, (list, tuple)) or len(weights) != 2
                or any(isinstance(w, bool) or not isinstance(w, (int, float))
                       or not math.isfinite(w) or w < 0 for w in weights)
                or not math.isclose(sum(weights), 1.0, rel_tol=0, abs_tol=1e-6)):
            raise ValueError("memory.fixed_fusion_weights must be two nonnegative weights summing to one.")
        self.fixed_fusion_weights = tuple(float(w) for w in weights)
        self.attention_norm = nn.RMSNorm(embedding_dim, eps=1e-5, elementwise_affine=False)
        self.query_proj = nn.Linear(embedding_dim, self.attention_dim, bias=False)
        self.key_proj = nn.Linear(embedding_dim, self.attention_dim, bias=False)
        self.current_proj = nn.Linear(embedding_dim, embedding_dim, bias=False)
        self.retrieval_proj = nn.Linear(embedding_dim, embedding_dim, bias=False)
        self.gate = nn.Sequential(
            nn.Linear(2 * embedding_dim, embedding_dim), nn.SiLU(), nn.Linear(embedding_dim, 2))
        with torch.no_grad():
            self.current_proj.weight.copy_(torch.eye(embedding_dim))
            self.retrieval_proj.weight.copy_(1e-3 * torch.eye(embedding_dim))
            self.gate[-1].weight.zero_()
            self.gate[-1].bias.copy_(torch.tensor([math.log(0.1), math.log(0.9)]))
        self.config.update(
            mode="cross_attention", dual_branch=True, use_residual=False,
            attention_dim=self.attention_dim, attention_mode=self.attention_mode,
            fusion_mode=self.fusion_mode, fixed_fusion_weights=list(self.fixed_fusion_weights),
            branch_output_l2=self.branch_output_l2,
            freeze_pretrained_memory=self.freeze_pretrained_memory)
        self.configure_trainability()

    def train(self, mode=True):
        super().train(mode)
        self.configure_trainability()
        return self

    def configure_trainability(self):
        for module in (self.block, self.proj, self.history_proj, self.pre_norm):
            module.requires_grad_(not self.freeze_pretrained_memory)
            module.train(self.training and not self.freeze_pretrained_memory)
        self.alpha.requires_grad_(False)
        long_active = self.fusion_mode != "short_only" and not (
            self.fusion_mode == "fixed" and self.fixed_fusion_weights[0] == 0)
        self.query_proj.requires_grad_(long_active and self.attention_mode == "learned")
        self.key_proj.requires_grad_(long_active and self.attention_mode == "learned")
        self.current_proj.requires_grad_(long_active)
        self.retrieval_proj.requires_grad_(long_active and self.attention_mode != "none")
        self.gate.requires_grad_(self.fusion_mode == "learned")

    def _branch_normalize(self, projected):
        return F.normalize(projected, dim=-1, eps=1e-12) if self.branch_output_l2 else projected

    def _scan(self, tokens):
        # mamba-ssm 2.3.2's fused convolution may retain an unaligned stride
        # when both batch and time are singleton (contiguous() is a no-op).
        # A right-padding token fixes the layout without changing the causal
        # first output or breaking the gradient through a frozen Mamba block.
        if tokens.is_cuda and tokens.shape[:2] == (1, 1):
            return self.block(F.pad(tokens, (0, 0, 0, 1)))[:, :1]
        return self.block(tokens)

    def _read_history(self, query, keys, values, allowed):
        counts = allowed.sum(-1)
        if keys.shape[1] == 0:
            attention = query.new_zeros(allowed.shape)
            readout = values.new_zeros((*query.shape[:-1], values.shape[-1]))
        elif self.attention_mode == "none":
            attention = query.new_zeros(allowed.shape)
            readout = values.new_zeros((*query.shape[:-1], values.shape[-1]))
        elif self.attention_mode == "uniform":
            attention = allowed.to(query.dtype) / counts.clamp_min(1).unsqueeze(-1)
            readout = attention @ values
        else:
            logits = (query @ keys.transpose(-1, -2)) / math.sqrt(self.attention_dim)
            logits = logits.masked_fill(~allowed, -torch.inf)
            # Empty rows must never enter softmax as an all-minus-infinity row.
            logits = torch.where((counts > 0).unsqueeze(-1), logits, torch.zeros_like(logits))
            attention = logits.softmax(-1).masked_fill(~allowed, 0)
            readout = attention @ values
        return readout, attention, counts

    @staticmethod
    def _add_stat(stats, name, value):
        value = value.detach()
        stats[name] = stats.get(name, 0) + value

    def _collect_read(self, current, history, counts, valid, attention, allowed,
                      stats, diagnostics):
        if stats is None and diagnostics is None:
            return
        with torch.no_grad():
            current_norm = current.detach().float().norm(dim=-1)
            history_norm = history.detach().float().norm(dim=-1)
            ratio = history_norm / (current_norm + 1e-12)
            if stats is not None:
                history_valid = valid & (counts > 0)
                for name, values in (
                        ("current_component_norm_sum", current_norm),
                        ("history_component_norm_sum", history_norm),
                        ("history_injection_ratio_sum", ratio)):
                    self._add_stat(stats, name, values.masked_fill(~history_valid, 0).sum())
                self._add_stat(stats, "num_history_frames", history_valid.sum())
            if diagnostics is not None:
                diagnostics.update(
                    attention=attention.detach(), attention_valid=allowed.detach(),
                    history_lengths=counts.detach(),
                    current_component_norm=current_norm.masked_fill(~valid, 0),
                    history_component_norm=history_norm.masked_fill(~valid, 0),
                    history_injection_ratio=ratio.masked_fill(~valid, 0))

    def forward(self, raw, *, lengths=None, stats=None, diagnostics=None, history_mask=None):
        if raw.ndim != 3 or raw.shape[-1] != self.proj.in_features or raw.shape[1] < 1:
            raise ValueError("Cross-attention inputs must have shape [B,T,D], T >= 1.")
        batch, time, _ = raw.shape
        lengths = (torch.full((batch,), time, device=raw.device, dtype=torch.long)
                   if lengths is None else torch.as_tensor(lengths, device=raw.device))
        if (lengths.shape != (batch,) or lengths.dtype == torch.bool or lengths.is_floating_point()
                or bool(((lengths < 1) | (lengths > time)).any())):
            raise ValueError("lengths must contain one real episode length in [1,T] per batch row.")
        valid = torch.arange(time, device=raw.device)[None, :] < lengths[:, None]
        if history_mask is not None:
            history_mask = torch.as_tensor(history_mask, device=raw.device)
            if history_mask.shape != valid.shape or history_mask.dtype != torch.bool:
                raise ValueError("history_mask must be a boolean [B,T] allowed-key mask.")
        allowed_keys = valid if history_mask is None else valid & history_mask
        allowed = (valid[:, :, None] & allowed_keys[:, None, :]
                   & torch.ones(time, time, device=raw.device, dtype=torch.bool).tril(-1))
        with torch.autocast(device_type=raw.device.type, enabled=False):
            # Mask before phi/scan: a zero attention weight cannot suppress a
            # NaN value (0 * NaN is NaN), and padding is not an observation.
            local = raw.float().masked_fill(~valid.unsqueeze(-1), 0)
            tokens = self._prepare_input(local.unsqueeze(-2))
            short = self.proj(self._scan(tokens)).float()
            query = self.query_proj(self.attention_norm(self._branch_normalize(short)))
            keys = self.key_proj(self.attention_norm(tokens))
            readout, attention, counts = self._read_history(query, keys, tokens, allowed)
            current, history = self.current_proj(tokens), self.retrieval_proj(readout)
            long = self.proj(self._scan(current + history)).float()
            self._collect_read(current, history, counts, valid, attention, allowed, stats, diagnostics)
            return torch.stack((long, short), dim=-2)

    def fuse(self, raw, projected, *, weight_stats=None, valid=None, diagnostics=None):
        with torch.autocast(device_type=projected.device.type, enabled=False):
            projected = projected.float()
            branches = self._branch_normalize(projected)
            if self.fusion_mode == "learned":
                weights = self.gate(branches.flatten(start_dim=-2)).softmax(-1)
                output = (weights.unsqueeze(-1) * branches).sum(-2)
            else:
                fixed = ((0., 1.) if self.fusion_mode == "short_only" else
                         (1., 0.) if self.fusion_mode == "long_only" else self.fixed_fusion_weights)
                weights = projected.new_tensor(fixed).expand(*projected.shape[:-2], 2)
                # Preserve exactly the source's pre-final-normalization path.
                output = (projected[..., 1, :] if self.fusion_mode == "short_only" else
                          (weights.unsqueeze(-1) * branches).sum(-2))
            if weight_stats is not None or diagnostics is not None:
                with torch.no_grad():
                    valid = (torch.ones(output.shape[:-1], dtype=torch.bool, device=output.device)
                             if valid is None else valid.to(device=output.device, dtype=torch.bool))
                    difference = 1 - F.cosine_similarity(
                        branches[..., 0, :].detach(), branches[..., 1, :].detach(), dim=-1, eps=1e-12)
                    logged_weights = weights.detach().masked_fill(~valid.unsqueeze(-1), 0)
                    if weight_stats is not None:
                        self._add_stat(weight_stats, "branch_weight_sum", logged_weights.reshape(-1, 2).sum(0))
                        for name, values in (
                                ("input_norm_sum", raw.detach().float().norm(dim=-1)),
                                ("pre_normalization_norm_sum", output.detach().norm(dim=-1)),
                                ("branch_cosine_distance_sum", difference)):
                            self._add_stat(weight_stats, name, values.masked_fill(~valid, 0).sum())
                        self._add_stat(weight_stats, "num_frames", valid.sum())
                    if diagnostics is not None:
                        diagnostics.update(branch_weights=logged_weights,
                                           branch_difference=difference.masked_fill(~valid, 0))
            return output

    def init_stream_state(self, batch_size):
        device = self.proj.weight.device
        state = {"steps": torch.zeros(batch_size, dtype=torch.long, device=device),
                 "kv_lengths": torch.zeros(batch_size, dtype=torch.long, device=device),
                 "keys": self.proj.weight.new_zeros(batch_size, 0, self.attention_dim),
                 "values": self.proj.weight.new_zeros(batch_size, 0, self.proj.in_features)}
        for branch in ("long", "short"):
            state[f"conv_{branch}"], state[f"ssm_{branch}"] = self.allocate_cache(batch_size)
        return state

    def step(self, raw, state, *, stats=None, diagnostics=None):
        with torch.autocast(device_type=raw.device.type, enabled=False):
            tokens = self._prepare_input(raw.float().unsqueeze(-2))
            short, state["conv_short"], state["ssm_short"] = self.block.step(
                tokens.unsqueeze(1), state["conv_short"], state["ssm_short"])
            short = self.proj(short[:, 0]).float()
            query = self.query_proj(self.attention_norm(self._branch_normalize(short))).unsqueeze(1)
            lengths = state["kv_lengths"]
            capacity = state["keys"].shape[1]
            allowed = (torch.arange(capacity, device=raw.device)[None, :] < lengths[:, None]).unsqueeze(1)
            readout, attention, counts = self._read_history(query, state["keys"], state["values"], allowed)
            current = self.current_proj(tokens)
            history = self.retrieval_proj(readout[:, 0])
            long, state["conv_long"], state["ssm_long"] = self.block.step(
                (current + history).unsqueeze(1), state["conv_long"], state["ssm_long"])
            long = self.proj(long[:, 0]).float()
            self._collect_read(current, history, counts[:, 0],
                               torch.ones(len(raw), dtype=torch.bool, device=raw.device),
                               attention[:, 0], allowed[:, 0], stats, diagnostics)
            # Append only after retrieval. Capacity growth never discards history.
            required = int(lengths.max().item()) + 1
            if required > capacity:
                new_capacity = max(required, max(1, capacity * 2))
                for name in ("keys", "values"):
                    old = state[name]
                    expanded = old.new_zeros(len(raw), new_capacity, old.shape[-1])
                    expanded[:, :capacity] = old
                    state[name] = expanded
            rows = torch.arange(len(raw), device=raw.device)
            state["keys"][rows, lengths] = self.key_proj(self.attention_norm(tokens))
            state["values"][rows, lengths] = tokens
            lengths.add_(1)
            return torch.stack((long, short), dim=-2)
