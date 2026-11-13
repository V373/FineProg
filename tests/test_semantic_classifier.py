"""Upper R/M/G semantic classifier: structure, partial fine-tuning, data, inference, short run."""

import csv
import json
from pathlib import Path
import sys

import h5py
import numpy as np
import pytest
import torch
import torch.nn.functional as F
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from fineprog import train_semantic
from fineprog.dataset_preparation.semantic_frame_dataset import SemanticFrameDataset
from fineprog.models.semantic_classifier import SemanticClassifier, load_semantic_classifier


def is_frozen(name):
    return name.startswith("features.") and int(name.split(".")[1]) < 6


def calibrated_model(size=224, l2_normalize=True):
    """Random weights whose BN statistics match them, standing in for pretrained weights.

    With the default statistics (mean 0, variance 1) a randomly initialised
    EfficientNet-B0 attenuates its input to ~1e-13 by the last stage, so neither
    the outputs nor the backbone gradients would depend on the frames.
    """
    model = SemanticClassifier(pretrained=False, l2_normalize=l2_normalize)
    for module in model.features.modules():
        if isinstance(module, torch.nn.BatchNorm2d):
            module.momentum = 1.0  # Adopt the statistics of the calibration batch.
            module.train()
    with torch.no_grad():
        model(torch.randint(0, 256, (8, 3, size, size), dtype=torch.uint8))
    return model.eval()


def write_data(directory, sequences, size=32):
    """H5 + label CSV. Pixel value = 10 * video + frame + 100 * channel; None = unlabelled row."""
    directory.mkdir(parents=True, exist_ok=True)
    h5_path, csv_path = directory / "videos.h5", directory / "labels.csv"
    with h5py.File(h5_path, "w") as file, csv_path.open("w", newline="") as labels_file:
        writer = csv.writer(labels_file)
        writer.writerow(["name", "id", "frame_idx", "label", "label_id"])
        videos = file.create_group("videos")
        for vid, labels in sequences.items():
            values = 10 * int(vid) + np.arange(len(labels))[:, None] + 100 * np.arange(3)
            pixels = np.broadcast_to(values[:, None, None, :], (len(labels), size, size, 3))
            videos.create_group(vid).create_dataset("frames", data=pixels.astype(np.uint8))
            for step, label in enumerate(labels):
                writer.writerow([vid, vid, step] + (["", ""] if label is None else ["RMG"[label], label]))
    return h5_path, csv_path


def test_dimensions_unit_latent_and_parameter_counts():
    torch.manual_seed(0)
    model = calibrated_model()
    frames = torch.randint(0, 256, (2, 3, 224, 224), dtype=torch.uint8)
    with torch.no_grad():
        normalized = (frames.float() / 255 - model.mean) / model.std
        assert model.features(normalized).shape == (2, 1280, 7, 7)
        latent, logits = model.embed(frames), model(frames)
        torch.testing.assert_close(model(frames.float() / 255), logits)
    assert latent.shape == (2, 32) and logits.shape == (2, 3)
    torch.testing.assert_close(latent.norm(dim=1), torch.ones(2))

    model.l2_normalize = False
    with torch.no_grad():
        raw = model.mlp(model.features(normalized).mean((2, 3)))
        torch.testing.assert_close(model.embed(frames), raw)
        torch.testing.assert_close(model(frames), model.head(raw))
    assert not torch.allclose(raw.norm(dim=1), torch.ones(2))
    torch.testing.assert_close(F.normalize(raw, dim=1), latent)

    def count(parameters):
        return sum(parameter.numel() for parameter in parameters)

    assert count(model.features.parameters()) == 4_007_548
    assert count(model.parameters()) == 4_175_743
    assert count(p for p in model.parameters() if p.requires_grad) == 3_323_935


def test_one_step_updates_only_the_tail_and_keeps_every_bn_statistic():
    torch.manual_seed(0)
    model = calibrated_model().train()
    assert model.training and not any(module.training for module in model.features.modules())
    for name, parameter in model.named_parameters():
        assert parameter.requires_grad == (not is_frozen(name)), name

    before = {name: value.clone() for name, value in model.state_dict().items()}
    optimizer = torch.optim.Adam(model.param_groups(1e-5, 1e-4))
    frames = torch.randint(0, 256, (3, 3, 224, 224), dtype=torch.uint8)
    F.cross_entropy(model(frames), torch.tensor([0, 1, 2])).backward()
    optimizer.step()

    parameters = dict(model.named_parameters())
    assert any(name.endswith("running_mean") for name in before)
    for name, value in model.state_dict().items():
        # Buffers (BN running mean / var / count, input mean / std) are not parameters.
        trainable = name in parameters and not is_frozen(name)
        assert (not torch.equal(value, before[name])) == trainable, name


def test_param_groups_cover_exactly_the_trainable_parameters():
    model = SemanticClassifier(pretrained=False)
    backbone, head = model.param_groups(1e-5, 1e-4)
    assert (backbone["lr"], head["lr"]) == (1e-5, 1e-4)
    names = {id(parameter): name for name, parameter in model.named_parameters()}
    grouped = [names[id(p)] for group in (backbone, head) for p in group["params"]]
    assert sorted(grouped) == sorted(n for n, p in model.named_parameters() if p.requires_grad)
    assert all(names[id(p)].startswith("features.") for p in backbone["params"])
    assert not any(names[id(p)].startswith("features.") for p in head["params"])


def test_dataset_keeps_only_labelled_frames_and_balances_classes(tmp_path):
    dataset = SemanticFrameDataset(*write_data(tmp_path, {
        "000001": [0, 0, None, 1, 2], "000002": [None, 0, 1, 1, 1, 2, 2]}))
    assert len(dataset) == 10
    assert dataset.video_ids.tolist() == ["000001"] * 4 + ["000002"] * 6
    assert dataset.frame_idx.tolist() == [0, 1, 3, 4, 1, 2, 3, 4, 5, 6]
    assert dataset.labels.tolist() == [0, 0, 1, 2, 0, 1, 1, 1, 2, 2]
    for index in range(len(dataset)):
        image, label = dataset[index]
        base = 10 * int(dataset.video_ids[index]) + int(dataset.frame_idx[index])
        assert image.dtype == torch.uint8 and image.shape == (3, 32, 32)
        assert [channel.unique().tolist() for channel in image] == [[base], [base + 100], [base + 200]]
        assert label == dataset.labels[index]

    sampler = dataset.balanced_sampler(6000)
    mass = np.bincount(dataset.labels, weights=sampler.weights.numpy(), minlength=3)
    np.testing.assert_allclose(mass, mass[0])
    torch.manual_seed(0)
    drawn = np.bincount(dataset.labels[list(sampler)], minlength=3)
    assert drawn.sum() == 6000
    np.testing.assert_allclose(drawn / 6000, 1 / 3, atol=0.03)


def test_predict_is_independent_per_frame():
    torch.manual_seed(0)
    model = calibrated_model(64)
    levels = torch.tensor([0, 60, 120, 180, 240], dtype=torch.uint8).view(5, 1, 1, 1)
    frames = levels + torch.randint(0, 16, (5, 3, 64, 64), dtype=torch.uint8)
    probabilities, labels = model.predict(frames)
    assert probabilities.shape == (5, 3) and labels.shape == (5,)
    torch.testing.assert_close(probabilities.sum(1), torch.ones(5))
    assert torch.equal(labels, probabilities.argmax(1))
    # The frames must give different outputs, or the checks below would be vacuous.
    assert (probabilities[0] - probabilities[4]).abs().max() > 1e-3

    order = torch.tensor([3, 0, 4, 1, 2])
    shuffled = model.predict(frames[order], batch_size=2)[0]
    torch.testing.assert_close(shuffled, probabilities[order], atol=1e-5, rtol=0)
    torch.testing.assert_close(model.train().predict(frames)[0], probabilities, atol=1e-5, rtol=0)


def test_classification_metrics_orientation_and_undefined_ratios():
    metrics = train_semantic.classification_metrics(
        np.array([0, 0, 0, 1, 2, 2]), np.array([0, 2, 2, 1, 0, 2]))
    assert metrics["confusion_matrix"] == [[1, 0, 2], [0, 1, 0], [1, 0, 1]]
    assert metrics["accuracy"] == pytest.approx(3 / 6)
    assert metrics["precision"] == pytest.approx({"R": 1 / 2, "M": 1.0, "G": 1 / 3})
    assert metrics["recall"] == pytest.approx({"R": 1 / 3, "M": 1.0, "G": 1 / 2})
    assert (metrics["R_as_G"], metrics["G_as_R"], metrics["num_frames"]) == (2, 1, 6)

    never_g = train_semantic.classification_metrics(np.array([0, 2]), np.array([0, 0]))
    assert never_g["precision"] == {"R": 0.5, "M": None, "G": None}
    assert never_g["recall"] == {"R": 1.0, "M": None, "G": 0.0}


@pytest.mark.parametrize("l2_normalize", [True, False])
def test_run_writes_artifacts_and_checkpoint_reproduces_predictions(tmp_path, monkeypatch, l2_normalize):
    train = write_data(tmp_path / "train", {"000001": [0, 0, 1, 1, 2, 2], "000002": [0, 1, 2, 2, 1, 0]})
    val = write_data(tmp_path / "val", {"000001": [0, None, 1, 2, 2]})

    class Resolver:
        def load_train(self, config_path):
            return {**yaml.safe_load(Path(config_path).read_text()), "h5_path": str(train[0])}

        def resolve_dataset(self, name):
            assert name == "val"
            return {"processed_h5_path": str(val[0])}

    trained = []

    def build(pretrained, l2_normalize):
        assert pretrained is False
        trained.append(calibrated_model(32, l2_normalize))
        return trained[0]

    monkeypatch.setattr(train_semantic, "ConfigV2", Resolver)
    monkeypatch.setattr(train_semantic, "SemanticClassifier", build)
    config = tmp_path / "config.yaml"
    config.write_text(yaml.safe_dump({
        "train_dataset": "train", "val_dataset": "val",
        "train_labels_csv": str(train[1]), "val_labels_csv": str(val[1]),
        "pretrained": False, "l2_normalize": l2_normalize, "output_dir": str(tmp_path / "run"),
        "batch_size": 4,
        "num_steps": 3, "eval_every": 2, "backbone_lr": 1e-5, "head_lr": 1e-4,
        "seed": 42, "device": "cpu"}))
    output = train_semantic.run(config)

    assert output == tmp_path / "run" and (output / "resolved_config.yaml").exists()
    steps = list(csv.DictReader((output / "metrics.csv").open()))
    assert [row["step"] for row in steps] == ["1", "2", "3"]
    assert all(np.isfinite(float(row["loss"])) for row in steps)
    history = json.loads((output / "val_metrics.json").read_text())
    assert [item["step"] for item in history] == [2, 3]
    assert all(item["num_frames"] == np.sum(item["confusion_matrix"]) == 4 for item in history)

    rows = list(csv.DictReader((output / "val_predictions.csv").open()))
    assert [(row["video_id"], row["frame_idx"], row["label"]) for row in rows] == [
        ("000001", "0", "R"), ("000001", "2", "M"), ("000001", "3", "G"), ("000001", "4", "G")]
    assert history[-1]["accuracy"] == np.mean([row["label"] == row["prediction"] for row in rows])

    model = load_semantic_classifier(output / "semantic_classifier.pt")
    assert not model.training and model.l2_normalize is l2_normalize
    reloaded, expected = model.state_dict(), trained[0].state_dict()
    assert reloaded.keys() == expected.keys()
    for name, value in expected.items():
        assert torch.equal(reloaded[name], value), name
    frames = torch.from_numpy(SemanticFrameDataset(*val).frames).permute(0, 3, 1, 2)
    probabilities, labels = model.predict(frames)
    assert ["RMG"[label] for label in labels] == [row["prediction"] for row in rows]
    saved = torch.tensor([[float(row[f"p_{name}"]) for name in "RMG"] for row in rows])
    torch.testing.assert_close(probabilities, saved, atol=1e-5, rtol=0)

    with pytest.raises(FileExistsError):
        train_semantic.run(config)
