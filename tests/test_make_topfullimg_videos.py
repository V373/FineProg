import csv
import importlib.util
import random
import shutil
import sys
from pathlib import Path

import cv2
import h5py
import numpy as np
import pytest


SCRIPT_PATH = (
    Path(__file__).resolve().parents[1]
    / "datasets"
    / "raw"
    / "fruits_expert"
    / "make_topfullimg_videos.py"
)
SPEC = importlib.util.spec_from_file_location("make_topfullimg_videos", SCRIPT_PATH)
assert SPEC is not None and SPEC.loader is not None
converter = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = converter
SPEC.loader.exec_module(converter)


def _write_episode(root: Path, name: str) -> Path:
    episode = root / name
    frame_dir = episode / converter.FRAME_DIR_NAME
    frame_dir.mkdir(parents=True)

    first = np.zeros((6, 6, 3), dtype=np.uint8)
    first[..., 0] = 10
    first[..., 1] = 20
    first[..., 2] = 30
    second = first.copy()
    second[2:4, 2:4] = [40, 50, 60]
    assert cv2.imwrite(str(frame_dir / "0.jpg"), first)
    assert cv2.imwrite(str(frame_dir / "1.jpg"), second)
    shutil.copyfile(frame_dir / "0.jpg", frame_dir / "2.jpg")
    return episode


def test_default_split_is_reproducible_and_only_splits_train_40():
    episodes = [Path(f"20260909{index:06d}") for index in range(76)]

    split = converter.build_split_plan(
        episodes,
        train_count=40,
        valid_count=4,
        calibration_count=18,
        fail_count=18,
        split_seed=42,
    )

    shuffled = episodes[:58]
    random.Random(42).shuffle(shuffled)
    assert split["train"] == shuffled[:36]
    assert split["valid"] == shuffled[36:40]
    assert split["calibration"] == shuffled[40:]
    assert split["fail"] == episodes[58:]
    assert len({episode for items in split.values() for episode in items}) == 76


@pytest.mark.parametrize(
    "counts, message",
    [
        ((40, 4, 18, 17), "must equal"),
        ((4, 4, 36, 36), "smaller than"),
        ((40, 0, 18, 18), "positive"),
    ],
)
def test_invalid_split_counts_are_rejected(counts, message):
    train_count, valid_count, calibration_count, fail_count = counts
    episodes = [Path(str(index)) for index in range(76)]
    with pytest.raises(ValueError, match=message):
        converter.build_split_plan(
            episodes,
            train_count=train_count,
            valid_count=valid_count,
            calibration_count=calibration_count,
            fail_count=fail_count,
            split_seed=42,
        )


def test_frame_index_overlay_is_one_based_and_does_not_mutate_input():
    source = np.zeros((64, 64, 3), dtype=np.uint8)

    first = converter.add_frame_index_overlay(source, 0)
    second = converter.add_frame_index_overlay(source, 1)

    assert not source.any()
    assert first.any()
    assert second.any()
    assert not np.array_equal(first, second)


def test_overlay_flag_is_parsed(monkeypatch):
    monkeypatch.setattr(sys, "argv", ["make_topfullimg_videos.py", "--overlay-frame-index"])

    args = converter.parse_args()

    assert args.overlay_frame_index is True


def test_mp4_and_direct_h5_share_crop_rgb_and_duplicate_filter(tmp_path, monkeypatch):
    raw_root = tmp_path / "collect_data"
    episodes = [_write_episode(raw_root, f"timestamp_{index}") for index in range(4)]
    video_root = tmp_path / "videos"
    video_root.mkdir()
    h5_root = tmp_path / "processed"

    monkeypatch.setattr(converter, "INPUT_ROOT", raw_root)
    monkeypatch.setattr(converter, "OUTPUT_ROOT", video_root)
    monkeypatch.setattr(converter, "ENABLE_CROP", True)
    monkeypatch.setattr(converter, "ENABLE_DUPLICATE_FRAME_FILTER", True)
    monkeypatch.setattr(converter, "CROP_SIZE", (4, 4))
    monkeypatch.setattr(converter, "CROP_OFFSET_H", 1)
    monkeypatch.setattr(converter, "CROP_OFFSET_W", 1)
    monkeypatch.setattr(converter, "H5_IMAGE_SIZE", 4)

    captured_frames = []

    class FakeWriter:
        def __init__(self, path):
            self.path = Path(path)

        def append_data(self, frame):
            captured_frames.append(frame.copy())

        def close(self):
            self.path.touch()

    monkeypatch.setattr(
        converter.imageio,
        "get_writer",
        lambda path, **kwargs: FakeWriter(path),
    )
    status, written, duplicates_removed = converter.convert_timestamp(episodes[0], 0)
    assert (status, written, duplicates_removed) == ("converted", 2, 1)

    split_plan = converter.build_split_plan(
        episodes,
        train_count=2,
        valid_count=1,
        calibration_count=1,
        fail_count=1,
        split_seed=42,
    )
    outputs = converter.write_h5_splits(
        episodes,
        split_plan,
        h5_output_dir=h5_root,
        h5_prefix="mango",
        split_seed=42,
        overwrite=False,
    )

    assert [path.name for path in outputs] == [
        "mango-1vid_train.h5",
        "mango-1vid_valid.h5",
        "mango-1vid_calibration.h5",
        "mango-1vid_fail.h5",
    ]
    source_by_episode = {episode.name: episode for episode in episodes}
    for output in outputs:
        with h5py.File(output, "r") as h5file:
            assert list(h5file.keys()) == ["videos"]
            video = h5file["videos/000001"]
            assert video["frames"].shape == (2, 4, 4, 3)
            assert video["frames"].dtype == np.uint8
            assert video["frames"].compression is None
            episode = source_by_episode[video.attrs["source_episode"]]
            expected, source_count, removed = converter.load_processed_frames(episode)
            np.testing.assert_array_equal(video["frames"][:], expected)
            source_bgr = cv2.imread(
                str(episode / converter.FRAME_DIR_NAME / "0.jpg"),
                cv2.IMREAD_COLOR,
            )[1:5, 1:5]
            source_rgb = cv2.cvtColor(source_bgr, cv2.COLOR_BGR2RGB)
            np.testing.assert_array_equal(video["frames"][0], source_rgb)
            assert source_count == 3
            assert removed == 1
            assert Path(video.attrs["source_frame_dir"]) == (
                episode / converter.FRAME_DIR_NAME
            )
            assert Path(video.attrs["path"]).parent == video_root

    np.testing.assert_array_equal(
        captured_frames, converter.load_processed_frames(episodes[0])[0]
    )
    mapping = h5_root / "idx_mapping" / "mango-1vid_train_idx_mapping.csv"
    with mapping.open(newline="") as csvfile:
        rows = list(csv.DictReader(csvfile))
    assert len(rows) == 1
    assert rows[0]["source_episode"] in source_by_episode
    assert rows[0]["processed_h5_name"] == "mango-1vid_train.h5"

    with pytest.raises(FileExistsError, match="--overwrite-h5"):
        converter.write_h5_splits(
            episodes,
            split_plan,
            h5_output_dir=h5_root,
            h5_prefix="mango",
            split_seed=42,
            overwrite=False,
        )
