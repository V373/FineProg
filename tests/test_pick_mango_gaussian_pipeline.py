from pathlib import Path

import h5py
import numpy as np
import pytest
import yaml

from scripts import run_pick_mango_gaussian_pipeline as pipeline


def _write_embedding_h5(path: Path, lengths: list[int]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with h5py.File(path, "w") as h5_file:
        h5_file.attrs["embedding_dim"] = pipeline.INPUT_EMBEDDING_DIM
        h5_file.attrs["embedding_normalization"] = "l2"
        videos = h5_file.create_group("videos")
        for index, length in enumerate(lengths):
            video = videos.create_group(f"{index:06d}")
            embeddings = np.zeros(
                (length, pipeline.INPUT_EMBEDDING_DIM),
                dtype=np.float32,
            )
            embeddings[:, 0] = 1.0
            video.create_dataset("embeddings", data=embeddings)
            video.create_dataset("target_steps", data=np.arange(length))


def test_variant_and_split_contracts() -> None:
    assert [variant.slug for variant in pipeline.VARIANTS] == [
        "filtered",
        "unfiltered",
    ]
    assert [variant.dataset_prefix for variant in pipeline.VARIANTS] == [
        "pick_mango_filtered_vid",
        "pick_mango_vid",
    ]
    assert [(split.name, split.expected_video_count) for split in pipeline.SPLITS] == [
        ("train", 36),
        ("valid4", 4),
        ("calibration18", 18),
        ("fail18", 18),
    ]
    assert [split.name for split in pipeline.SPLITS if split.is_prediction_query] == [
        "valid4",
        "fail18",
    ]


@pytest.mark.parametrize(
    ("real_points", "expected_per_bin", "expected_synthetic"),
    [
        (3960, 198, 3960),
        (4404, 220, 4400),
        (31, 2, 40),
    ],
)
def test_gaussian_samples_track_real_point_count(
    real_points: int,
    expected_per_bin: int,
    expected_synthetic: int,
) -> None:
    per_bin = pipeline._gaussian_samples_per_bin(
        real_points,
        pipeline.NUM_PROGRESS_BINS,
    )
    assert per_bin == expected_per_bin
    assert pipeline.NUM_PROGRESS_BINS * per_bin == expected_synthetic
    assert abs(expected_synthetic - real_points) <= pipeline.NUM_PROGRESS_BINS // 2


def test_prepare_extract_config_uses_mango_training_shape(tmp_path: Path) -> None:
    config_root = tmp_path / "configs"
    config_root.mkdir()
    extract_path = config_root / "extract.yaml"
    extract_path.write_text(
        yaml.safe_dump({"checkpoint_ref": "old", "num_workers": 4}),
        encoding="utf-8",
    )
    variant = pipeline.VARIANTS[0]
    split = pipeline.SPLITS[0]
    output_path = tmp_path / "train-embd.h5"

    pipeline._prepare_extract_config(
        config_root,
        variant,
        split,
        output_path,
        num_workers=2,
    )

    config = yaml.safe_load(extract_path.read_text(encoding="utf-8"))
    assert config["extract_dataset"] == "pick_mango_filtered_vid_36vid_train"
    assert config["checkpoint_path"] == str(variant.checkpoint_path)
    assert config["embedding_dim"] == 128
    assert config["clip_len"] == 20
    assert config["context_size"] == 2
    assert config["context_stride"] == 16
    assert config["sample_all_stride"] == 1
    assert config["embedding_save_path"] == str(output_path)
    assert config["num_workers"] == 2
    assert "checkpoint_ref" not in config


def test_prepare_fitting_config_sets_dynamic_tsne_samples(tmp_path: Path) -> None:
    config_root = tmp_path / "configs"
    eval_root = config_root / "eval"
    eval_root.mkdir(parents=True)
    config_path = eval_root / "gaussian_progress_fitting.yaml"
    config_path.write_text(
        yaml.safe_dump({"tsne_viz": {"random_seed": 42}}),
        encoding="utf-8",
    )
    expert_path = tmp_path / "expert.h5"
    _write_embedding_h5(expert_path, [7, 11, 13])

    real_points, per_bin = pipeline._prepare_fitting_config(
        config_root,
        expert_path,
        tmp_path / "outputs",
    )

    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    assert real_points == 31
    assert per_bin == 2
    assert config["num_bins"] == 20
    assert config["pca_dim"] == 32
    assert config["tsne_viz"]["gaussian_samples_per_bin"] == 2
    assert config["tsne_viz"]["random_seed"] == 42


def test_run_variant_uses_calibration_and_twelve_predictions(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    checkpoint_path = tmp_path / "encoder_epoch020000.pt"
    checkpoint_path.touch()
    monkeypatch.setattr(
        pipeline.MangoVariant,
        "checkpoint_path",
        property(lambda _self: checkpoint_path),
    )
    monkeypatch.setattr(pipeline, "_validate_processed_h5", lambda *args, **kwargs: None)

    extracted: list[str] = []

    def fake_extract(_config_root, _variant, split, **_kwargs):
        extracted.append(split.name)
        return tmp_path / f"{split.name}.h5"

    monkeypatch.setattr(pipeline, "_extract_embedding", fake_extract)
    model_path = tmp_path / "gaussian_progress_model.h5"
    monkeypatch.setattr(
        pipeline,
        "_fit_gaussian_model",
        lambda *_args, **_kwargs: model_path,
    )
    predictions: list[tuple[str, str, float, str]] = []

    def fake_predict(
        _config_root,
        _variant,
        query_split,
        *,
        calibration_path,
        threshold,
        threshold_tag,
        **_kwargs,
    ):
        predictions.append(
            (query_split.name, calibration_path.name, threshold, threshold_tag)
        )
        return tmp_path / "prediction.h5"

    monkeypatch.setattr(pipeline, "_predict_one", fake_predict)

    pipeline._run_variant(
        pipeline.VARIANTS[0],
        tmp_path / "configs",
        dry_run=False,
        gpu=None,
        num_workers=None,
    )

    assert extracted == ["train", "valid4", "calibration18", "fail18"]
    assert len(predictions) == 12
    assert {query for query, _, _, _ in predictions} == {"valid4", "fail18"}
    assert {calibration for _, calibration, _, _ in predictions} == {
        "calibration18.h5"
    }
    assert {threshold for _, _, threshold, _ in predictions} == {
        threshold for threshold, _ in pipeline.PREDICTION_THRESHOLDS
    }
