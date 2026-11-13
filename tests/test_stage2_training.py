"""Stage-two checkpoint migration and differentiable single-frame training."""

import copy
import csv
import json
from pathlib import Path

import h5py
import numpy as np
import pytest
import torch
import yaml

from test_streaming_mamba2 import cpu_memory, tiny_local
from fineprog.models.encoder import TCCEncoder
from fineprog.utils import encoder_checkpoint as checkpoints


def source_checkpoint(tmp_path, memory):
    model = TCCEncoder(
        pretrained=False, clip_len=3, context_size=2, context_stride=20,
        embedding_dim=4, embedding_normalization="l2", single_frame_context=True,
        train_base="frozen" if memory else "only_bn", train_embedding=not memory,
        memory_config={"enabled": memory, "history_frames": 1,
                       "use_residual": False, "normalize_input": True,
                       "freeze_bn_stats": True, "cache_local_embeddings": True,
                       "visual_chunk_size": 2, "checkpoint_visual": True},
    ).eval()
    model.backbone.bn.running_mean.fill_(0.2)
    model.backbone.bn.running_var.fill_(0.7)
    model.backbone.bn.num_batches_tracked.fill_(19)
    path = tmp_path / "source.pt"
    torch.save(checkpoints.checkpoint_dict(model), path)
    return model, path


@pytest.mark.parametrize("memory", [False, True])
def test_stage2_load_changes_policy_not_weights(cpu_memory, tmp_path, memory):
    from fineprog.train_stage2 import build_stage2_encoder

    source, path = source_checkpoint(tmp_path, memory)
    model = build_stage2_encoder(path, "cpu")
    assert model.single_frame_context is True
    assert (model.context_size, model.context_stride, model.embedding_dim) == (2, 20, 4)
    assert model.embedding_normalization == "l2"
    for name, value in source.state_dict().items():
        torch.testing.assert_close(value, model.state_dict()[name], rtol=0, atol=0)
    assert model._train_base == "only_bn" and model._train_embedding is True
    assert not model.backbone.conv.weight.requires_grad
    assert model.backbone.bn.weight.requires_grad and model.backbone.bn.bias.requires_grad
    assert all(p.requires_grad for p in model.temporal_embedder.parameters())
    if memory:
        assert model.memory.history_frames == 1 and not model.memory.use_residual
        assert model.memory_config["freeze_bn_stats"] is False
        assert model.memory_config["cache_local_embeddings"] is False
        assert model.memory_config["visual_chunk_size"] == 20
        assert all(p.requires_grad for p in model.memory.block.parameters())
        assert all(p.requires_grad for p in model.memory.proj.parameters())
        assert not model.memory.alpha.requires_grad
    model.eval()
    assert not model.backbone.bn.training
    model.train()
    model.configure_trainability()
    assert model.backbone.bn.training
    assert model.backbone.bn.num_batches_tracked.item() == 19


def test_stage2_local_dynamic_targets_and_current_pixels(cpu_memory, tmp_path):
    from fineprog.train_stage2 import build_stage2_encoder, encode_batch

    _, path = source_checkpoint(tmp_path, False)
    model = build_stage2_encoder(path, "cpu").train()
    frames = torch.rand(2, 5, 2, 3, 224, 224, requires_grad=True)
    observed = []
    hook = model.backbone.register_forward_pre_hook(
        lambda module, args: observed.append(args[0].detach().clone()))
    output = encode_batch(model, {"frames": frames}, "cpu")
    hook.remove()
    assert output.shape == (2, 5, 4) and model.clip_len == 3
    expected = frames[:, :, -1:].expand_as(frames).reshape(-1, 3, 224, 224)
    torch.testing.assert_close(observed[0], expected, rtol=0, atol=0)
    (output * torch.randn_like(output)).sum().backward()
    assert frames.grad[:, :, 0].count_nonzero() == 0
    assert frames.grad[:, :, -1].abs().sum() > 0
    assert model.backbone.conv.weight.grad is None
    assert model.backbone.bn.weight.grad.abs().sum() > 0


def test_stage2_hf1_history_crosses_visual_chunks(cpu_memory, tmp_path):
    from fineprog.train_stage2 import build_stage2_encoder, encode_batch

    _, path = source_checkpoint(tmp_path, True)
    model = build_stage2_encoder(path, "cpu").train()
    frames = torch.rand(21, 3, 224, 224, requires_grad=True)
    batch = {"frames_full": [frames], "seq_len": torch.tensor([21]),
             "target_steps": torch.tensor([[20]])}
    output = encode_batch(model, batch, "cpu")
    assert model.memory.block.lengths == [21]
    assert model.backbone.bn.num_batches_tracked.item() == 21
    (output * torch.randn_like(output)).sum().backward()
    assert model.backbone.bn.num_batches_tracked.item() == 21
    assert frames.grad[0].abs().sum() > 0
    assert torch.isfinite(frames.grad).all()


def training_batch(memory, expert):
    # Replay uses the first three targets. References retain all three classes.
    target_steps = torch.tensor([[0, 1, 2, 0, 1, 2], [0, 1, 2, 0, 1, 2]])
    labels = torch.tensor([[0, 1, 2, 0, 1, 2], [0, 1, 2, 0, 1, 2]])
    if not expert:
        target_steps = target_steps[:, :3]
        labels = labels[:, :3]
    batch = {"target_steps": target_steps, "labels": labels,
             "replay_count": 3 if expert else 0, "seq_len": torch.tensor([3, 3]),
             "video_id": ["000000", "000001"]}
    frames = [torch.rand(3, 3, 224, 224) for _ in range(2)]
    if memory:
        batch["frames_full"] = frames
    else:
        contexts = []
        for video, steps in zip(frames, target_steps):
            indices = torch.stack(((steps - 20).clamp_min(0), steps), dim=-1)
            contexts.append(video[indices])
        batch["frames"] = torch.stack(contexts)
    return batch


class ReplayLoss(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.shapes = []

    def forward(self, embeddings, batch):
        self.shapes.append((tuple(embeddings.shape), tuple(batch["target_steps"].shape)))
        return {"loss": embeddings.sum() * 0, "metrics": {}}


@pytest.mark.parametrize("memory", [False, True])
def test_stage2_set_loss_updates_and_checkpoint_roundtrip(cpu_memory, tmp_path, memory):
    from fineprog.train_stage2 import build_stage2_encoder, compute_stage2_loss, encode_batch

    _, path = source_checkpoint(tmp_path, memory)
    model = build_stage2_encoder(path, "cpu").train()
    expert, query = training_batch(memory, True), training_batch(memory, False)
    before = copy.deepcopy(model.state_dict())
    loss_module = ReplayLoss()
    optimizer = torch.optim.Adam([p for p in model.parameters() if p.requires_grad], lr=1e-3)
    for _ in range(2):
        optimizer.zero_grad(set_to_none=True)
        out = compute_stage2_loss(model, expert, query, loss_module, "cpu")
        assert torch.isfinite(out["loss"]) and out["loss"].item() > 0
        bn_count = model.backbone.bn.num_batches_tracked.item()
        out["loss"].backward()
        assert model.backbone.bn.num_batches_tracked.item() == bn_count
        for module in (model.backbone, model.temporal_embedder, model.memory):
            if module is None:
                continue
            grads = [p.grad for p in module.parameters() if p.requires_grad]
            assert all(g is not None and torch.isfinite(g).all() for g in grads)
            assert sum(g.abs().sum() for g in grads) > 0
        assert model.backbone.conv.weight.grad is None
        optimizer.step()
    assert loss_module.shapes == [((2, 3, 4), (2, 3))] * 2
    torch.testing.assert_close(model.backbone.conv.weight, before["backbone.conv.weight"], rtol=0, atol=0)
    assert not torch.equal(model.backbone.bn.weight, before["backbone.bn.weight"])
    assert not torch.equal(model.temporal_embedder.proj.weight, before["temporal_embedder.proj.weight"])
    assert model.backbone.bn.num_batches_tracked.item() > 19
    if memory:
        assert not torch.equal(model.memory.proj.weight, before["memory.proj.weight"])
    model.eval()
    with torch.no_grad():
        expected = encode_batch(model, query, "cpu")
    final_path = tmp_path / "stage2.pt"
    torch.save(checkpoints.checkpoint_dict(model), final_path)
    restored = checkpoints.build_encoder_from_checkpoint(final_path, "cpu").eval()
    assert restored.single_frame_context is True
    assert restored._train_base == "only_bn" and restored._train_embedding is True
    if memory:
        assert restored.memory_config["freeze_bn_stats"] is False
    with torch.no_grad():
        torch.testing.assert_close(encode_batch(restored, query, "cpu"), expected, rtol=0, atol=0)


def run_fixture(tmp_path, monkeypatch, memory):
    from fineprog import train_stage2

    _, source = source_checkpoint(tmp_path, memory)
    rng = np.random.default_rng(7)
    paths = {}
    for role, count in (("expert", 4), ("query", 3)):
        h5_path, labels_path = tmp_path / f"{role}.h5", tmp_path / f"{role}.csv"
        with h5py.File(h5_path, "w") as handle, labels_path.open("w", newline="") as file:
            writer = csv.DictWriter(file, fieldnames=["id", "frame_idx", "label", "label_id"])
            writer.writeheader()
            videos = handle.create_group("videos")
            for index in range(1, count + 1):
                video_id = f"{index:06d}"
                frames = rng.integers(0, 256, (5, 224, 224, 3), dtype=np.uint8)
                videos.create_group(video_id).create_dataset("frames", data=frames)
                for step, label in enumerate(("R", "R", "M", "G", "G")):
                    writer.writerow({"id": video_id, "frame_idx": step, "label": label,
                                     "label_id": {"R": 0, "M": 1, "G": 2}[label]})
        paths[role] = (h5_path, labels_path)
    loss_dir = tmp_path / "source_loss"
    loss_dir.mkdir()
    loss_path = loss_dir / "composite.yaml"
    loss_path.write_text("components: []\n")

    class Resolver:
        def load_train(self, config_path):
            config = yaml.safe_load(Path(config_path).read_text())
            config["h5_path"] = str(paths["expert"][0])
            return config

        def resolve_dataset(self, name):
            return {"processed_h5_path": str(paths["query"][0])}

    monkeypatch.setattr(train_stage2, "ConfigV2", Resolver)
    monkeypatch.setattr(train_stage2, "build_loss", lambda *args, **kwargs: ReplayLoss())
    config = {
        "train_dataset": "expert", "nonexpert_dataset": "query",
        "checkpoint_path": str(source), "expert_labels_csv": str(paths["expert"][1]),
        "nonexpert_labels_csv": str(paths["query"][1]), "loss_config": str(loss_path),
        "output_dir": str(tmp_path / "run"), "nonexpert_video_ids": ["000001", "000002"],
        "expert_batch_size": 2, "nonexpert_batch_size": 1, "refs_per_class": 1,
        "query_steps": 3, "random_offset": 0, "learning_rate": 1e-3,
        "set_weight": 0.1, "set_tau": 0.128, "seed": 42, "device": "cpu",
        "cpu_threads": 1,
    }
    return train_stage2, config


@pytest.mark.parametrize("memory", [False, True])
@pytest.mark.parametrize("epoch_mode", [False, True])
def test_stage2_run_epochs_checkpoints_and_legacy_steps(cpu_memory, tmp_path, monkeypatch,
                                                      memory, epoch_mode):
    train_stage2, config = run_fixture(tmp_path, monkeypatch, memory)
    evaluations, validated_ids = [], []

    class Validator:
        def __init__(self, expert_dataset, validation_dataset, **kwargs):
            assert expert_dataset.video_ids == ["000001", "000002", "000003", "000004"]
            validated_ids.extend(validation_dataset.video_ids)

        def evaluate(self, encoder, output_dir, epoch):
            assert encoder.training
            evaluations.append(epoch)
            result = {"epoch": epoch, "set_loss": 1 / (epoch + 1),
                      "r_loss": 1 / (epoch + 1), "g_loss": 1 / (epoch + 1),
                      "positive_mass": 0.5, "valid_queries": 4,
                      "validation_video_ids": list(validated_ids)}
            output_dir = Path(output_dir)
            output_dir.mkdir(parents=True, exist_ok=True)
            (output_dir / "summary.json").write_text(json.dumps(result))
            return result

    monkeypatch.setattr(train_stage2, "Stage2Validator", Validator)
    if epoch_mode:
        config.update(num_epochs=2, checkpoint_every_epochs=1,
                      validation_video_ids=["000003"])
        expected_steps = 4  # Four expert episodes / batch two, repeated twice.
    else:
        config["num_steps"] = 3
        expected_steps = 3
    config_path = tmp_path / "config.yaml"
    config_path.write_text(yaml.safe_dump(config))
    output = train_stage2.run(config_path)
    records = list(csv.DictReader((output / "metrics.csv").open()))
    assert len(records) == expected_steps
    assert [int(row["step"]) for row in records] == list(range(1, expected_steps + 1))
    assert all(set(row["query_video_ids"].split(",")) <= {"000001", "000002"} for row in records)
    saved_config = yaml.safe_load((output / "resolved_config.yaml").read_text())
    assert saved_config["num_steps"] == expected_steps
    if epoch_mode:
        assert saved_config["steps_per_epoch"] == 2
        assert validated_ids == ["000003"]
        assert evaluations == [0, 1, 2]
        history = json.loads((output / "validation_history.json").read_text())
        assert [item["epoch"] for item in history] == [0, 1, 2]
        assert history[0]["checkpoint"] == config["checkpoint_path"]
        summary = json.loads((output / "summary.json").read_text())
        assert summary["best_validation"]["epoch"] == 2
        assert summary["best_validation"]["checkpoint"] == str(output / "encoder_epoch000002.pt")
        assert not (output / "encoder_epoch000000.pt").exists()
        for epoch in (1, 2):
            path = output / f"encoder_epoch{epoch:06d}.pt"
            payload = torch.load(path, map_location="cpu", weights_only=True)
            assert payload["training_provenance"]["step"] == epoch * 2
            assert payload["encoder_config"]["single_frame_context"] is True
    else:
        assert evaluations == []
        assert (output / "encoder_step000003.pt").is_file()
        assert not list(output.glob("encoder_epoch*.pt"))


def test_stage2_run_rejects_overlapping_validation(cpu_memory, tmp_path, monkeypatch):
    train_stage2, config = run_fixture(tmp_path, monkeypatch, False)
    config.update(num_epochs=2, checkpoint_every_epochs=1,
                  validation_video_ids=["000002"])
    config_path = tmp_path / "config.yaml"
    config_path.write_text(yaml.safe_dump(config))
    with pytest.raises(ValueError, match="(?i)overlap|disjoint|validation"):
        train_stage2.run(config_path)
    assert not (Path(config["output_dir"]) / "metrics.csv").exists()
