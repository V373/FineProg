import csv
from pathlib import Path
import sys

import h5py
import numpy as np
import pytest


_PROJECTS_ROOT = Path(__file__).resolve().parents[2]
if str(_PROJECTS_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECTS_ROOT))

from fineprog.dataset_preparation import mp4vid_to_h5data as converter_module
from fineprog.dataset_preparation.mp4vid_to_h5data import MP4ToH5Converter


def _make_demo_files(raw_dir: Path, demo_ids: range) -> None:
    raw_dir.mkdir(parents=True)
    for demo_id in demo_ids:
        (raw_dir / f"demo_{demo_id}.mp4").touch()


def test_demo_id_range_uses_numeric_order(tmp_path):
    raw_dir = tmp_path / "raw" / "numeric_order"
    _make_demo_files(raw_dir, range(11))
    converter = MP4ToH5Converter(
        raw_root=str(raw_dir),
        output_dir=str(tmp_path / "processed"),
        demo_id_range=(1, 10),
        split="train",
    )

    converter.scan_mp4_files()
    selected = converter._select_demo_id_range(1, 10)

    assert [path.name for path in selected] == [
        f"demo_{demo_id}.mp4" for demo_id in range(1, 11)
    ]


@pytest.mark.parametrize(
    ("start_id", "end_id", "split", "expected_name"),
    [
        (0, 35, "train", "fruit_expert_videos-36vid_train.h5"),
        (36, 39, "valid", "fruit_expert_videos-4vid_valid.h5"),
        (36, 55, "valid", "fruit_expert_videos-20vid_valid.h5"),
    ],
)
def test_range_mode_writes_expected_h5_and_mapping(
    tmp_path,
    monkeypatch,
    start_id,
    end_id,
    split,
    expected_name,
):
    raw_dir = tmp_path / "raw" / "fruit_expert_videos"
    _make_demo_files(raw_dir, range(start_id, end_id + 1))
    output_dir = tmp_path / "processed"
    mapping_dir = output_dir / "idx_mapping"
    monkeypatch.setattr(converter_module, "IDX_MAPPING_DIR", mapping_dir)
    monkeypatch.setattr(
        MP4ToH5Converter,
        "get_video_info",
        lambda self, path: (20, 1, 224, 224),
    )
    monkeypatch.setattr(
        MP4ToH5Converter,
        "extract_frames",
        lambda self, path, stride: np.zeros((1, 224, 224, 3), dtype=np.uint8),
    )
    converter = MP4ToH5Converter(
        raw_root=str(raw_dir),
        output_dir=str(output_dir),
        demo_id_range=(start_id, end_id),
        split=split,
    )

    converter.run()

    output_path = output_dir / expected_name
    expected_count = end_id - start_id + 1
    with h5py.File(output_path, "r") as h5_file:
        video_ids = sorted(h5_file["videos"].keys())
        assert video_ids == [
            f"{local_id:06d}" for local_id in range(1, expected_count + 1)
        ]
        first = h5_file["videos"][video_ids[0]]
        last = h5_file["videos"][video_ids[-1]]
        assert first.attrs["action_name"] == f"demo_{start_id}"
        assert last.attrs["action_name"] == f"demo_{end_id}"
        assert Path(first.attrs["path"]).name == f"demo_{start_id}.mp4"
        assert Path(last.attrs["path"]).name == f"demo_{end_id}.mp4"
        assert first["frames"].shape == (1, 224, 224, 3)
        assert first["frames"].dtype == np.uint8
        assert first.attrs["fps"] == 20
        assert first.attrs["num_frames"] == 1

    mapping_path = mapping_dir / f"{output_path.stem}_idx_mapping.csv"
    with mapping_path.open(newline="") as csv_file:
        rows = list(csv.DictReader(csv_file))
    assert len(rows) == expected_count
    assert rows[0]["video_idx"] == "000001"
    assert rows[0]["source_video_name"] == f"demo_{start_id}.mp4"
    assert rows[-1]["video_idx"] == f"{expected_count:06d}"
    assert rows[-1]["source_video_name"] == f"demo_{end_id}.mp4"


def test_demo_id_range_rejects_missing_id(tmp_path):
    raw_dir = tmp_path / "raw" / "missing"
    _make_demo_files(raw_dir, range(2))
    converter = MP4ToH5Converter(
        raw_root=str(raw_dir),
        output_dir=str(tmp_path / "processed"),
        demo_id_range=(0, 2),
        split="train",
    )
    converter.scan_mp4_files()

    with pytest.raises(ValueError, match="Missing requested demo videos: demo_2"):
        converter._select_demo_id_range(0, 2)


def test_demo_id_range_rejects_duplicate_id(tmp_path):
    raw_dir = tmp_path / "raw" / "duplicate"
    raw_dir.mkdir(parents=True)
    (raw_dir / "demo_1.mp4").touch()
    (raw_dir / "demo_1.mov").touch()
    converter = MP4ToH5Converter(
        raw_root=str(raw_dir),
        output_dir=str(tmp_path / "processed"),
        demo_id_range=(1, 1),
        split="valid",
    )
    converter.scan_mp4_files()

    with pytest.raises(ValueError, match="Duplicate demo ID 1"):
        converter._select_demo_id_range(1, 1)


@pytest.mark.parametrize(
    ("demo_id_range", "split", "message"),
    [
        ((0, 1), None, "provided together"),
        (None, "train", "provided together"),
        ((-1, 1), "train", "non-negative"),
        ((2, 1), "train", "start must be <= end"),
    ],
)
def test_demo_id_range_rejects_invalid_options(
    tmp_path,
    demo_id_range,
    split,
    message,
):
    with pytest.raises(ValueError, match=message):
        MP4ToH5Converter(
            raw_root=str(tmp_path / "raw"),
            output_dir=str(tmp_path / "processed"),
            demo_id_range=demo_id_range,
            split=split,
        )


def test_range_mode_fails_without_finalizing_partial_h5(tmp_path, monkeypatch):
    raw_dir = tmp_path / "raw" / "strict"
    _make_demo_files(raw_dir, range(2))
    output_dir = tmp_path / "processed"
    monkeypatch.setattr(converter_module, "IDX_MAPPING_DIR", output_dir / "mapping")
    monkeypatch.setattr(
        MP4ToH5Converter,
        "get_video_info",
        lambda self, path: (20, 1, 224, 224),
    )

    def _extract_frames(self, path, stride):
        if path.stem == "demo_1":
            return np.empty((0, 224, 224, 3), dtype=np.uint8)
        return np.zeros((1, 224, 224, 3), dtype=np.uint8)

    monkeypatch.setattr(MP4ToH5Converter, "extract_frames", _extract_frames)
    converter = MP4ToH5Converter(
        raw_root=str(raw_dir),
        output_dir=str(output_dir),
        demo_id_range=(0, 1),
        split="train",
    )

    with pytest.raises(ValueError, match="No frames extracted from demo_1.mp4"):
        converter.run()

    assert not (output_dir / "strict_tmp.h5").exists()
    assert not (output_dir / "strict-1vid_train.h5").exists()
    assert not (output_dir / "strict-2vid_train.h5").exists()
