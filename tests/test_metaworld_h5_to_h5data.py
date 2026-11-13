import json
from pathlib import Path
import sys

import h5py
import numpy as np
import pytest


_PROJECTS_ROOT = Path(__file__).resolve().parents[2]
if str(_PROJECTS_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECTS_ROOT))

from fineprog.dataset_preparation.h5vid_dataset import H5VideoDataset
from fineprog.dataset_preparation.metaworld_h5_to_h5data import (
    convert_metaworld_h5,
)


def _write_source(
    path: Path,
    trajectory_ids,
    *,
    resolution=(224, 224),
    frame_count=3,
    task_name="assembly-v2",
    fps=12,
) -> None:
    height, width = resolution
    with h5py.File(path, "w") as source:
        source.attrs["env_metadata"] = json.dumps(
            {"task_name": task_name, "fps": fps}
        )
        source.attrs["total"] = len(trajectory_ids)
        for trajectory_id in trajectory_ids:
            trajectory = source.create_group(f"traj_{trajectory_id}")
            frames = np.empty(
                (frame_count, height, width, 3),
                dtype=np.uint8,
            )
            # MetaWorld observations are BGR. Include the frame index so tests
            # can verify that conversion skips the initial observation.
            frame_indices = np.arange(frame_count, dtype=np.uint8)[:, None, None]
            frames[..., 0] = (trajectory_id + frame_indices) % 251
            frames[..., 1] = (trajectory_id + 20 + frame_indices) % 251
            frames[..., 2] = (trajectory_id + 40 + frame_indices) % 251
            trajectory.create_dataset("observations", data=frames)
            infos = trajectory.create_group("infos")
            # Include failed trajectories to prove conversion is unfiltered.
            infos.create_dataset(
                "goal",
                data=np.full(frame_count, trajectory_id % 2 == 0, dtype=np.bool_),
            )


def _action_names(path: Path) -> list[str]:
    with h5py.File(path, "r") as output:
        return [
            output["videos"][video_id].attrs["action_name"]
            for video_id in sorted(output["videos"].keys())
        ]


def test_single_output_is_trainable_and_keeps_numeric_order(tmp_path):
    source_path = tmp_path / "source.hdf5"
    _write_source(source_path, [10, 2, 0])
    output_dir = tmp_path / "processed"

    outputs = convert_metaworld_h5(source_path, output_dir=output_dir)

    assert outputs == [output_dir / "assembly-v2-3vid.h5"]
    output_path = outputs[0]
    assert _action_names(output_path) == ["traj_0", "traj_2", "traj_10"]

    with h5py.File(output_path, "r") as output:
        assert list(output.keys()) == ["videos"]
        videos = output["videos"]
        assert sorted(videos.keys()) == ["000001", "000002", "000003"]
        first = videos["000001"]
        frames = first["frames"]
        assert frames.shape == (2, 224, 224, 3)
        assert frames.dtype == np.uint8
        assert frames.compression is None
        assert frames.chunks == (2, 224, 224, 3)
        np.testing.assert_array_equal(frames[0, 0, 0], [41, 21, 1])
        np.testing.assert_array_equal(frames[-1, 0, 0], [42, 22, 2])
        assert first.attrs["action_id"] == 0
        assert first.attrs["fps"] == 12
        assert first.attrs["num_frames"] == 2
        assert Path(first.attrs["path"]) == (
            source_path.parent / "video" / source_path.stem / "traj_0.mp4"
        )

    dataset = H5VideoDataset(
        h5_path=str(output_path),
        clip_len=2,
        context_size=1,
        context_stride=1,
    )
    sample = dataset[0]
    assert sample["frames"].shape == (2, 1, 3, 224, 224)
    assert sample["action_id"] == 0


def test_split_40_into_36_train_and_4_valid(tmp_path):
    source_path = tmp_path / "forty.hdf5"
    _write_source(source_path, range(40), frame_count=2)
    output_dir = tmp_path / "processed"

    outputs = convert_metaworld_h5(
        source_path,
        output_dir=output_dir,
        valid_count=4,
    )

    assert outputs == [
        output_dir / "assembly-v2-36vid_train.h5",
        output_dir / "assembly-v2-4vid_valid.h5",
    ]
    train_names = _action_names(outputs[0])
    valid_names = _action_names(outputs[1])
    assert train_names == [f"traj_{index}" for index in range(36)]
    assert valid_names == [f"traj_{index}" for index in range(36, 40)]
    assert set(train_names).isdisjoint(valid_names)
    assert len(train_names) + len(valid_names) == 40


def test_split_allows_independent_custom_names(tmp_path):
    source_path = tmp_path / "source.hdf5"
    _write_source(source_path, range(5))
    output_dir = tmp_path / "processed"

    outputs = convert_metaworld_h5(
        source_path,
        output_dir=output_dir,
        valid_count=1,
        train_output_name="custom_train",
    )

    assert outputs == [
        output_dir / "custom_train.h5",
        output_dir / "assembly-v2-1vid_valid.h5",
    ]


def test_non_224_input_is_resized_and_channel_flipped(tmp_path):
    source_path = tmp_path / "small.hdf5"
    _write_source(source_path, [0], resolution=(8, 12), frame_count=2)

    output_path = convert_metaworld_h5(
        source_path,
        output_dir=tmp_path / "processed",
        output_name="resized",
    )[0]

    with h5py.File(output_path, "r") as output:
        frame = output["videos/000001/frames"][0]
        assert frame.shape == (224, 224, 3)
        np.testing.assert_array_equal(frame[100, 100], [41, 21, 1])


@pytest.mark.parametrize("valid_count", [0, 3, 4])
def test_invalid_valid_count_is_rejected(tmp_path, valid_count):
    source_path = tmp_path / "source.hdf5"
    _write_source(source_path, range(3))

    with pytest.raises(ValueError, match="valid_count must satisfy"):
        convert_metaworld_h5(
            source_path,
            output_dir=tmp_path / "processed",
            valid_count=valid_count,
        )


def test_mode_conflicts_and_existing_output_are_rejected(tmp_path):
    source_path = tmp_path / "source.hdf5"
    _write_source(source_path, range(3))
    output_dir = tmp_path / "processed"

    with pytest.raises(ValueError, match="cannot be combined"):
        convert_metaworld_h5(
            source_path,
            output_dir=output_dir,
            output_name="full",
            valid_count=1,
        )

    output_path = convert_metaworld_h5(
        source_path,
        output_dir=output_dir,
        output_name="full",
    )[0]
    with pytest.raises(FileExistsError, match="--overwrite"):
        convert_metaworld_h5(
            source_path,
            output_dir=output_dir,
            output_name=output_path.name,
        )


def test_missing_observations_removes_temporary_output(tmp_path):
    source_path = tmp_path / "bad.hdf5"
    with h5py.File(source_path, "w") as source:
        source.attrs["env_metadata"] = json.dumps({"task_name": "bad-v2"})
        source.create_group("traj_0")
    output_dir = tmp_path / "processed"

    with pytest.raises(KeyError, match="missing observations"):
        convert_metaworld_h5(source_path, output_dir=output_dir)

    assert not (output_dir / "bad-v2-1vid.h5").exists()
    assert list(output_dir.glob("*.tmp")) == []
