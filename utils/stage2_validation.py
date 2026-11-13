"""Fixed-reference, full-episode semantic validation during stage-two training."""

import json
from pathlib import Path

import numpy as np
import torch


def _summarize(labels, probabilities, nll):
    valid = np.isfinite(nll)
    r, g = valid & (labels == 0), valid & (labels == 2)

    def mean(values, mask):
        return float(values[mask].mean()) if mask.any() else None

    positive = probabilities[np.arange(len(labels)), labels.clip(0, 2)]
    return {
        "set_loss": mean(nll, valid), "r_loss": mean(nll, r), "g_loss": mean(nll, g),
        "positive_mass": mean(positive, valid),
        "r_positive_mass": mean(positive, r), "g_positive_mass": mean(positive, g),
        "r_count": int(r.sum()), "g_count": int(g.sum()),
        "valid_queries": int(valid.sum()),
        "set_accuracy": mean((probabilities.argmax(1) == labels).astype(float), valid),
    }


class Stage2Validator:
    """Reuse current embeddings for losses and per-frame probability plots.

    The reference indices are sampled once with a private RNG. All validation
    frames are encoded, including M, and only valid R/G frames enter the loss.
    """

    def __init__(self, expert_dataset, validation_dataset, *, refs_per_class=4,
                 seed=42, tau=0.128):
        if refs_per_class < 1 or not np.isfinite(tau) or tau <= 0:
            raise ValueError("Reference count and tau must be positive")
        self.expert_dataset = expert_dataset
        self.validation_dataset = validation_dataset
        self.tau = tau
        rng = np.random.default_rng(seed)
        self.reference_steps = {}
        for vid in expert_dataset.video_ids:
            labels = expert_dataset.labels[vid]
            selected = []
            for category in range(3):
                choices = np.flatnonzero(labels == category)
                if len(choices):
                    selected.extend(rng.choice(choices, refs_per_class,
                                               replace=len(choices) < refs_per_class))
            if selected:
                self.reference_steps[vid] = np.asarray(selected, dtype=np.int64)
        if not self.reference_steps:
            raise ValueError("Validation needs at least one labeled expert reference")

    @staticmethod
    def _encode(encoder, dataset, vid, steps=None):
        pixels = dataset._frames_cache[vid]["frames"]
        if steps is None:
            steps = np.arange(len(pixels), dtype=np.int64)
        device = next(encoder.parameters()).device
        if encoder.memory is not None:
            frames = torch.from_numpy(pixels).permute(0, 3, 1, 2)
            raw = encoder.encode_local_episode(frames)
            targets = torch.from_numpy(steps)[None]
            return encoder.forward_episode([raw], [len(pixels)], targets)[0]
        chunks = []
        offsets = np.arange(encoder.context_size - 1, -1, -1) * encoder.context_stride
        for start in range(0, len(steps), 20):
            indices = np.maximum(steps[start:start + 20, None] - offsets[None], 0)
            frames = torch.from_numpy(pixels[indices]).permute(0, 1, 4, 2, 3)[None].to(device)
            chunks.append(encoder(frames.float().div(255))[0])
        return torch.cat(chunks)

    def _query_results(self, query, labels, references, ref_labels):
        probabilities, losses = [], []
        for start in range(0, len(query), 64):
            z = query[start:start + 64].float()
            target = torch.as_tensor(labels[start:start + 64], device=z.device)
            logits = -(z[:, None] - references[None]).square().sum(-1) / self.tau
            log_all = torch.logsumexp(logits, dim=1)
            log_mass = torch.stack([
                torch.logsumexp(logits[:, ref_labels == category], dim=1) - log_all
                for category in range(3)
            ], dim=1)
            same = target[:, None] == ref_labels[None]
            valid = ((target == 0) | (target == 2)) & same.any(1) & (~same).any(1)
            nll = torch.full_like(log_all, torch.nan)
            nll[valid] = -log_mass[valid, target[valid]]
            probabilities.append(log_mass.exp().cpu().numpy())
            losses.append(nll.cpu().numpy())
        return np.concatenate(probabilities), np.concatenate(losses)

    @torch.no_grad()
    def evaluate(self, encoder, output_dir, epoch):
        was_training = encoder.training
        tf32 = torch.backends.cudnn.allow_tf32, torch.backends.cuda.matmul.allow_tf32
        bn_before = {k: v.clone() for k, v in encoder.named_buffers()
                     if "running_" in k or k.endswith("num_batches_tracked")}
        encoder.eval()
        torch.backends.cudnn.allow_tf32 = False
        torch.backends.cuda.matmul.allow_tf32 = False
        try:
            device = next(encoder.parameters()).device
            with torch.autocast(device_type=device.type, enabled=False):
                refs, ref_labels, ref_steps, ref_vids = [], [], [], []
                for vid, steps in self.reference_steps.items():
                    refs.append(self._encode(encoder, self.expert_dataset, vid, steps))
                    ref_labels.extend(self.expert_dataset.labels[vid][steps])
                    ref_steps.extend(steps)
                    ref_vids.extend([vid] * len(steps))
                references = torch.cat(refs).float()
                categories = torch.as_tensor(ref_labels, dtype=torch.long, device=device)
                arrays = {"ref_steps": np.asarray(ref_steps, dtype=np.int64),
                          "ref_labels": np.asarray(ref_labels, dtype=np.int64),
                          "ref_video_ids": np.asarray(ref_vids)}
                per_video, all_labels, all_probs, all_nll = {}, [], [], []
                for vid in self.validation_dataset.video_ids:
                    query = self._encode(encoder, self.validation_dataset, vid)
                    labels = self.validation_dataset.labels[vid]
                    probs, nll = self._query_results(query, labels, references, categories)
                    if not np.isfinite(probs).all():
                        raise FloatingPointError(f"Nonfinite validation probabilities: {vid}")
                    per_video[vid] = {**_summarize(labels, probs, nll), "num_frames": len(labels)}
                    arrays.update({f"{vid}_labels": labels, f"{vid}_probs": probs,
                                   f"{vid}_set_nll": nll})
                    all_labels.append(labels)
                    all_probs.append(probs)
                    all_nll.append(nll)
            result = {"epoch": int(epoch),
                      **_summarize(np.concatenate(all_labels), np.concatenate(all_probs),
                                   np.concatenate(all_nll)),
                      "per_video": per_video, "reference_count": len(ref_labels),
                      "reference_video_ids": list(self.reference_steps),
                      "validation_video_ids": list(self.validation_dataset.video_ids),
                      "cudnn_allow_tf32": False, "matmul_allow_tf32": False,
                      "bn_buffers_unchanged": True}
            for name, value in encoder.named_buffers():
                if name in bn_before and not torch.equal(value, bn_before[name]):
                    raise AssertionError(f"Validation changed BN buffer {name}")
            output = Path(output_dir)
            output.mkdir(parents=True, exist_ok=True)
            np.savez_compressed(output / "frames.npz", **arrays)
            (output / "summary.json").write_text(json.dumps(result, indent=2, allow_nan=False))
            return result
        finally:
            torch.backends.cudnn.allow_tf32, torch.backends.cuda.matmul.allow_tf32 = tf32
            encoder.train(was_training)
            if was_training:
                encoder.configure_trainability()
