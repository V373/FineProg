import csv
import json

import h5py
import numpy as np
import pytest
import torch

from fineprog import models, training
from fineprog.data import VideoDataset, read_labels, read_videos, sample_steps
from test_models import tiny_models


@pytest.fixture
def task(tmp_path):
    rng = np.random.default_rng(42)
    data, labels = {}, {}
    for split, count in (("train", 4), ("valid", 2), ("fail", 4)):
        data[split] = tmp_path / f"{split}.h5"
        labels[split] = tmp_path / f"{split}.csv"
        with h5py.File(data[split], "w") as file, labels[split].open("w", newline="") as stream:
            writer = csv.writer(stream)
            writer.writerow(["name", "id", "frame_idx", "label", "label_id"])
            for i in range(count):
                vid = f"{i + 1:06d}"
                file.create_dataset(f"videos/{vid}/frames", data=rng.integers(0, 256, (6, 6, 6, 3), dtype=np.uint8))
                writer.writerows([vid, vid, step, "RMG"[step // 2], step // 2] for step in range(6))
    local = tmp_path / "initial_local.pt"
    return {"name": "tiny", "data": data, "labels": labels, "weights": {"lower": local},
            "stage2": {"query_split": "fail", "train_ids": ["000001", "000002"],
                       "valid_ids": ["000003", "000004"]}}


def test_lower_two_steps_checkpoint_updates_and_reload(tiny_models, task, tmp_path):
    cfg = {"pretrained_path": "unused", "batch_size": 2, "clip_len": 4,
           "num_epochs": 2, "temperature": .1}
    path = training.train_lower(task, cfg, tmp_path / "lower", max_steps=2)
    payload = torch.load(path, weights_only=True)
    loaded = models.load_local(path)
    assert payload["step"] == 2
    assert loaded.backbone.bn1.num_batches_tracked == 2
    assert all(torch.isfinite(value).all() for value in loaded.state_dict().values())
    assert (path.parent / "loss.csv").is_file()


def test_upper_and_teacher_labels_cover_unlabelled_h5_frames(tiny_models, task, tmp_path):
    path = training.train_upper(task, {"pretrained_path": None, "batch_size": 3, "num_steps": 2,
                                      "eval_every": 1}, tmp_path / "upper", max_steps=2)
    assert len(json.loads((path.parent / "validation.json").read_text())) == 2
    assert models.load_semantic(path).l2_normalize is False
    del task["labels"]["fail"]
    generated = training.predict_labels(task, path, "fail", tmp_path / "labels")
    videos = read_videos(task["data"]["fail"])
    labels = read_labels(generated, {vid: len(x) for vid, x in videos.items()})
    assert all((values >= 0).all() for values in labels.values())
    with generated.open() as stream:
        assert len(list(csv.DictReader(stream))) == 24


def test_stage2_fixed_human_validation_and_pseudo_training_labels(tiny_models, task, tmp_path):
    initial = models.LocalEncoder().eval()
    torch.save({"model_state_dict": initial.state_dict()}, task["weights"]["lower"])
    pseudo = tmp_path / "pseudo.csv"
    with task["labels"]["fail"].open() as stream, pseudo.open("w", newline="") as out:
        reader = csv.DictReader(stream)
        writer = csv.DictWriter(out, fieldnames=reader.fieldnames)
        writer.writeheader()
        for row in reader:
            # Deliberately wrong held-out pseudo labels must not enter validation.
            if row["id"] in task["stage2"]["valid_ids"]:
                row.update(label="R", label_id="0")
            writer.writerow(row)
    cfg = {"query_labels": pseudo, "clip_len": 4, "query_steps": 4,
           "refs_per_class": 1, "expert_batch_size": 2, "query_batch_size": 1,
           "num_epochs": 1, "temperature": .1, "validation_chunk_size": 3}
    path = training.train_stage2(task, cfg, tmp_path / "stage2", max_steps=2)
    history = json.loads((path.parent / "validation.json").read_text())
    assert history[0]["g_loss"] is not None
    assert history[-1]["valid_queries"] == 8
    loaded = models.load_local(path)
    assert loaded.backbone.bn1.num_batches_tracked == 4
    torch.testing.assert_close(initial.backbone.conv1.weight, loaded.backbone.conv1.weight, rtol=0, atol=0)
    assert not torch.equal(initial.temporal_embedder.proj.weight, loaded.temporal_embedder.proj.weight)
    assert torch.load(path, weights_only=True)["step"] == 2


def test_stage2_rejects_overlap_before_creating_output(task, tmp_path):
    task["stage2"]["valid_ids"] = ["000001"]
    with pytest.raises(ValueError, match="disjoint"):
        training.train_stage2(task, {}, tmp_path / "overlap")
    assert not (tmp_path / "overlap").exists()


def test_missing_reference_class_preserves_original_placeholder_sampling(task, tmp_path):
    sidecar = tmp_path / "missing_g.csv"
    with task["labels"]["train"].open() as stream, sidecar.open("w", newline="") as out:
        reader = csv.DictReader(stream)
        writer = csv.DictWriter(out, fieldnames=reader.fieldnames)
        writer.writeheader()
        for row in reader:
            if row["label"] == "G":
                row.update(label="", label_id="")
            writer.writerow(row)
    dataset = VideoDataset(task["data"]["train"], 3, labels_path=sidecar, role="expert", refs_per_class=2)
    np.random.seed(42)
    replay = sample_steps(6, 3)
    references = [*np.random.choice([0, 1], 2, replace=False),
                  *np.random.choice([2, 3], 2, replace=False),
                  *np.random.choice([0, 1, 2, 3], 2)]
    expected_next_random = np.random.rand()
    np.random.seed(42)
    sample = dataset[0]
    np.testing.assert_array_equal(sample["target_steps"], np.concatenate([replay, references]))
    assert sample["labels"][-2:].tolist() == [-1, -1]
    assert np.random.rand() == expected_next_random
