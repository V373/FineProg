import importlib.util
import json
import os
from pathlib import Path
import pickle
import sys

import cv2
import h5py
import numpy as np
import pytest


SCRIPT_PATH = (
    Path(__file__).resolve().parents[1]
    / "datasets"
    / "raw"
    / "drawer_place"
    / "restore_train_data_to_collect_data.py"
)
SPEC = importlib.util.spec_from_file_location("restore_drawer_place_collect_data", SCRIPT_PATH)
assert SPEC is not None and SPEC.loader is not None
converter = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = converter
SPEC.loader.exec_module(converter)


def _jpeg_payload(camera_index: int, frame_index: int) -> bytes:
    image = np.empty((12, 16, 3), dtype=np.uint8)
    image[..., 0] = 20 + camera_index * 30
    image[..., 1] = 40 + frame_index * 20
    image[..., 2] = 80 + camera_index + frame_index
    ok, encoded = cv2.imencode(".jpg", image)
    assert ok
    return encoded.tobytes()


def _write_source_episode(path: Path, frame_count: int) -> None:
    with h5py.File(path, "w") as source:
        source.attrs["sim"] = False
        source.attrs["custom_annotation"] = "preserved by source.hdf5"
        source.create_dataset(
            "action",
            data=(np.arange(frame_count * 14, dtype=np.float32).reshape(frame_count, 14) + 0.5),
        )
        source.create_dataset("auto_stage_labels", data=np.arange(frame_count, dtype=np.int64))

        observations = source.create_group("observations")
        observations.create_dataset(
            "qpos", data=np.arange(frame_count * 14, dtype=np.float32).reshape(frame_count, 14)
        )
        observations.create_dataset(
            "qvel",
            data=(np.arange(frame_count * 14, dtype=np.float32).reshape(frame_count, 14) + 100),
        )
        observations.create_dataset(
            "ee_pos_quat",
            data=(np.arange(frame_count * 14, dtype=np.float32).reshape(frame_count, 14) + 200),
        )
        observations.create_dataset(
            "gripper_position",
            data=(np.arange(frame_count * 2, dtype=np.float32).reshape(frame_count, 2) + 300),
        )
        observations.create_dataset(
            "eef_pose",
            data=(np.arange(frame_count * 12, dtype=np.float32).reshape(frame_count, 12) + 400),
        )

        images = observations.create_group("images")
        payloads = {
            spec.source_name: [_jpeg_payload(camera_index, frame_index) for frame_index in range(frame_count)]
            for camera_index, spec in enumerate(converter.CAMERAS)
        }
        max_length = max(len(payload) for items in payloads.values() for payload in items) + 11
        compress_len = np.zeros((len(converter.CAMERAS), frame_count), dtype=np.int32)
        for spec in converter.CAMERAS:
            dataset = images.create_dataset(
                spec.source_name,
                shape=(frame_count, max_length),
                dtype=np.uint8,
            )
            for frame_index, payload in enumerate(payloads[spec.source_name]):
                dataset[frame_index, : len(payload)] = np.frombuffer(payload, dtype=np.uint8)
                compress_len[spec.compress_len_row, frame_index] = len(payload)
        source.create_dataset("compress_len", data=compress_len)


def _expected_observation(source: h5py.File, frame_index: int) -> dict[str, np.ndarray]:
    return {
        output_key: np.asarray(source[source_path][frame_index])
        for output_key, source_path in converter.OBSERVATION_FIELDS
    }


def test_restore_collect_data_preserves_jpegs_observations_and_source_hdf5(tmp_path):
    input_dir = tmp_path / "train_data"
    input_dir.mkdir()
    first_source = input_dir / "episode_init_10.hdf5"
    second_source = input_dir / "episode_init_2.hdf5"
    _write_source_episode(first_source, frame_count=2)
    _write_source_episode(second_source, frame_count=3)

    output_dir = tmp_path / "collect_data"
    output_dir.mkdir()
    summary = converter.restore_collect_data(input_dir, output_dir)

    assert summary.output_dir == output_dir
    assert (summary.episode_count, summary.frame_count, summary.jpeg_count) == (2, 5, 20)
    assert sorted(path.name for path in output_dir.iterdir() if path.is_dir()) == [
        "episode_init_10",
        "episode_init_2",
    ]

    manifest = json.loads((output_dir / converter.MANIFEST_NAME).read_text())
    assert manifest["format_version"] == converter.FORMAT_VERSION
    assert [record["source_file"] for record in manifest["episodes"]] == [
        "episode_init_2.hdf5",
        "episode_init_10.hdf5",
    ]

    for source_path in (first_source, second_source):
        episode_dir = output_dir / source_path.stem
        archive_path = episode_dir / converter.SOURCE_LINK_NAME
        assert os.stat(source_path).st_ino == os.stat(archive_path).st_ino
        assert os.stat(source_path).st_dev == os.stat(archive_path).st_dev
        assert converter.sha256_file(source_path) == converter.sha256_file(archive_path)

        with h5py.File(source_path, "r") as source, h5py.File(archive_path, "r") as archive:
            assert archive.attrs["custom_annotation"] == "preserved by source.hdf5"
            frame_count = source["action"].shape[0]
            for frame_index in range(frame_count):
                for spec in converter.CAMERAS:
                    length = int(source["compress_len"][spec.compress_len_row, frame_index])
                    expected = source[f"observations/images/{spec.source_name}"][frame_index, :length].tobytes()
                    assert (episode_dir / spec.output_dir / f"{frame_index}.jpg").read_bytes() == expected

                with (episode_dir / "observation" / f"{frame_index}.pkl").open("rb") as observation_file:
                    restored = pickle.load(observation_file)
                expected_observation = _expected_observation(source, frame_index)
                assert list(restored) == list(expected_observation)
                for key, expected in expected_observation.items():
                    assert restored[key].dtype == expected.dtype
                    assert restored[key].shape == expected.shape
                    assert restored[key].tobytes() == expected.tobytes()


def test_restore_refuses_nonempty_output_without_touching_it(tmp_path):
    input_dir = tmp_path / "train_data"
    input_dir.mkdir()
    _write_source_episode(input_dir / "episode_init_0.hdf5", frame_count=1)
    output_dir = tmp_path / "collect_data"
    output_dir.mkdir()
    sentinel = output_dir / "keep-me.txt"
    sentinel.write_text("user data")

    with pytest.raises(FileExistsError, match="not empty"):
        converter.restore_collect_data(input_dir, output_dir)

    assert sentinel.read_text() == "user data"
    assert list(tmp_path.glob(".collect_data.restore-*.tmp")) == []


def test_restore_rejects_invalid_jpeg_length_and_cleans_staging_dir(tmp_path):
    input_dir = tmp_path / "train_data"
    input_dir.mkdir()
    source_path = input_dir / "episode_init_0.hdf5"
    _write_source_episode(source_path, frame_count=1)
    with h5py.File(source_path, "r+") as source:
        source["compress_len"][0, 0] = source["observations/images/top"].shape[1] + 1

    output_dir = tmp_path / "collect_data"
    output_dir.mkdir()
    with pytest.raises(converter.DatasetFormatError, match="invalid JPEG length"):
        converter.restore_collect_data(input_dir, output_dir)

    assert list(output_dir.iterdir()) == []
    assert list(tmp_path.glob(".collect_data.restore-*.tmp")) == []
