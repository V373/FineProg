"""Stage2 labels, episode isolation, and replay/reference sampling contracts."""

import csv
from pathlib import Path
import sys

import h5py
import numpy as np
import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from fineprog.dataset_preparation.h5vid_dataset import H5VideoDataset
from fineprog.dataset_preparation.stage2_dataset import (
    Stage2VideoDataset, collate_stage2, read_labels,
)


CONFIG = {"clip_len": 4, "context_size": 2, "context_stride": 3,
          "single_frame_context": True, "memory_config": {"enabled": False}}


def write_csv(path, rows):
    with path.open("w", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(["name", "id", "frame_idx", "label", "label_id"])
        writer.writerows(rows)
    return path


def files(tmp_path, sequences=None):
    sequences = sequences or {"000001": [0] * 3 + [1] * 4 + [2] * 5,
                              "000002": [0] * 9 + [2]}
    path = tmp_path / "videos.h5"
    rows = []
    with h5py.File(path, "w") as file:
        videos = file.create_group("videos")
        for vid, labels in sequences.items():
            pixels = np.broadcast_to(np.arange(len(labels), dtype=np.uint8)[:, None, None, None],
                                     (len(labels), 8, 8, 3)).copy()
            videos.create_group(vid).create_dataset("frames", data=pixels)
            rows.extend([[vid, vid, t, "RMG"[label], label] for t, label in enumerate(labels)])
    return path, write_csv(tmp_path / "labels.csv", rows)


def test_expert_replay_unchanged_and_all_reference_classes_present(tmp_path):
    h5, csv_path = files(tmp_path)
    original = H5VideoDataset(str(h5), clip_len=4, context_size=2, context_stride=3,
                              random_offset=1, transport_frames_as_uint8=True)
    stage2 = Stage2VideoDataset(h5, csv_path, encoder_config=CONFIG, role="expert", refs_per_class=2)
    np.random.seed(42)
    before = original[0]
    np.random.seed(42)
    after = stage2[0]
    torch.testing.assert_close(after["target_steps"][:4], before["target_steps"].long())
    torch.testing.assert_close(after["frames"][:4], before["frames"])
    assert after["replay_count"] == 4
    assert after["target_steps"].dtype == after["labels"].dtype == torch.int64
    assert after["frames"].dtype == torch.uint8
    assert after["labels"][4:].tolist() == [0, 0, 1, 1, 2, 2]
    np.testing.assert_array_equal(stage2.labels["000001"][after["target_steps"]], after["labels"])


def test_query_goal_coverage_and_subset_pixels_only(tmp_path, monkeypatch):
    h5, csv_path = files(tmp_path)
    dataset = Stage2VideoDataset(h5, csv_path, encoder_config=CONFIG, role="query",
                                video_ids=["000002"], query_steps=4)
    monkeypatch.setattr(dataset, "_sample_offset_uniform", lambda _: np.array([1, 2, 3, 4]))
    item = dataset[0]
    assert dataset.video_ids == list(dataset._frames_cache) == ["000002"]
    assert set(dataset.labels) == {"000001", "000002"}
    assert item["replay_count"] == 0
    assert item["target_steps"].tolist() == [1, 2, 3, 9]
    assert item["labels"].tolist() == [0, 0, 0, 2]
    assert item["frames"][-1, -1, 0, 0, 0] == 9


def test_full_episode_keeps_all_frames_and_ragged_collation(tmp_path):
    h5, csv_path = files(tmp_path)
    config = {**CONFIG, "memory_config": {"enabled": True}}
    dataset = Stage2VideoDataset(h5, csv_path, encoder_config=config, role="query", query_steps=4)
    batch = collate_stage2([dataset[0], dataset[1]])
    assert "frames" not in batch
    assert [len(x) for x in batch["frames_full"]] == [12, 10]
    for frames in batch["frames_full"]:
        torch.testing.assert_close(frames[:, 0, 0, 0], torch.arange(len(frames), dtype=torch.uint8))
    assert batch["seq_len"].tolist() == [12, 10]
    assert batch["target_steps"].shape == batch["labels"].shape == (2, 4)
    assert batch["replay_count"] == 0


def test_missing_reference_class_is_masked_not_relabelled(tmp_path):
    h5, csv_path = files(tmp_path, {"000001": [0] * 8})
    dataset = Stage2VideoDataset(h5, csv_path, encoder_config=CONFIG, role="expert", refs_per_class=2)
    item = dataset[0]
    assert item["labels"][4:].tolist() == [0, 0, -1, -1, -1, -1]
    assert int(item["target_steps"].max()) < 8
    batch = collate_stage2([item, item])
    assert batch["frames"].shape == (2, 10, 2, 3, 8, 8)


def test_missing_and_unknown_semantics_are_unsupervised(tmp_path):
    csv_path = write_csv(tmp_path / "labels.csv", [
        ["demo", "000001", 0, "R", 0],
        ["demo", "000001", 1, "unknown", 9],
        ["demo", "000001", 3, "", 1],
    ])
    labels = read_labels(csv_path, {"000001": 5})
    assert labels["000001"].tolist() == [0, -1, -1, -1, -1]


@pytest.mark.parametrize("extra,match", [
    (["demo", "000001", 0, "R", 0], "Duplicate"),
    (["demo", "000001", 3, "R", 0], "out of range"),
    (["demo", "000001", -1, "R", 0], "out of range"),
    (["demo", "1", 1, "R", 0], "Unknown video_id"),
    (["demo", "000001", 1, "M", 0], "Inconsistent"),
    (["demo", "000001", 1, "R", "invalid"], "Inconsistent"),
])
def test_corrupt_label_coordinates_or_ids_fail(tmp_path, extra, match):
    csv_path = write_csv(tmp_path / "labels.csv", [["demo", "000001", 0, "R", 0], extra])
    with pytest.raises(ValueError, match=match):
        read_labels(csv_path, {"000001": 3})


def test_revised_pick_mango_labels_align_and_keep_reviewed_boundaries():
    project = Path(__file__).resolve().parents[1]
    h5_path = project / "datasets/processed/pick_mango_vid-18vid_fail.h5"
    csv_path = project / "datasets/phase_labels/pick_mango_vid-18vid_fail_rmg_labels.csv"
    if not h5_path.exists() or not csv_path.exists():
        pytest.skip("Local Pick Mango dataset is not available")
    with h5py.File(h5_path, "r") as file:
        lengths = {vid: len(group["frames"]) for vid, group in file["videos"].items()}
    labels = read_labels(csv_path, lengths)
    flat = np.concatenate(list(labels.values()))
    assert np.bincount(flat).tolist() == [2617, 1370, 26]
    assert labels["000007"][104:106].tolist() == [1, 0]
    assert labels["000014"][149:151].tolist() == [0, 1]
    assert labels["000014"][185:187].tolist() == [1, 0]
    assert labels["000016"][180:182].tolist() == [0, 1]
    assert labels["000016"][235:237].tolist() == [1, 0]
    assert np.all(labels["000002"][76:88] == 0)
    assert np.all(labels["000014"][210:225] == 0)
    assert np.all(labels["000016"][358:375] == 2)
    assert np.all(labels["000017"][269:278] == 2)
