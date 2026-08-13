from pathlib import Path
import sys

import h5py
import numpy as np
import pytest
import torch


_PROJECTS_ROOT = Path(__file__).resolve().parents[2]
if str(_PROJECTS_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECTS_ROOT))

from fineprog.extract_embeddings import load_checkpoint, save_embeddings_h5
from fineprog.utils.embedding_normalization import read_embedding_normalization


def _model() -> torch.nn.Module:
    model = torch.nn.Linear(2, 2)
    model.embedding_normalization = "none"
    return model


def _results(embeddings: np.ndarray, video_id: str = "000001") -> list[dict]:
    return [
        {
            "video_id": video_id,
            "embeddings": embeddings,
            "target_steps": np.arange(len(embeddings), dtype=np.int64),
            "seq_len": len(embeddings),
            "action_id": 0,
        }
    ]


def test_new_checkpoint_restores_l2_mode(tmp_path):
    source = _model()
    path = tmp_path / "new.pt"
    torch.save(
        {
            "model_state_dict": source.state_dict(),
            "embedding_normalization": "l2",
        },
        path,
    )

    target = _model()
    mode = load_checkpoint(target, str(path), torch.device("cpu"))

    assert mode == "l2"
    assert target.embedding_normalization == "l2"


def test_legacy_checkpoint_defaults_to_none(tmp_path):
    source = _model()
    path = tmp_path / "legacy.pt"
    torch.save(source.state_dict(), path)

    target = _model()
    target.embedding_normalization = "l2"
    with pytest.warns(UserWarning, match="legacy checkpoint"):
        mode = load_checkpoint(target, str(path), torch.device("cpu"))

    assert mode == "none"
    assert target.embedding_normalization == "none"


def test_l2_embedding_h5_writes_metadata(tmp_path):
    path = tmp_path / "l2.h5"
    embeddings = np.array([[1.0, 0.0], [0.0, 1.0]], dtype=np.float32)

    save_embeddings_h5(_results(embeddings), str(path), "l2")

    with h5py.File(path, "r") as h5_file:
        assert h5_file.attrs["embedding_normalization"] == "l2"
        assert h5_file.attrs["embedding_dim"] == 2
        assert read_embedding_normalization(h5_file, str(path)) == "l2"


@pytest.mark.parametrize("embedding_dim", [32, 64])
def test_embedding_h5_writes_actual_dimension(tmp_path, embedding_dim):
    path = tmp_path / f"embeddings-{embedding_dim}.h5"
    embeddings = np.zeros((2, embedding_dim), dtype=np.float32)

    save_embeddings_h5(_results(embeddings), str(path), "none")

    with h5py.File(path, "r") as h5_file:
        assert h5_file.attrs["embedding_dim"] == embedding_dim
        assert h5_file["videos"]["000001"]["embeddings"].shape == (
            2,
            embedding_dim,
        )


def test_embedding_h5_rejects_inconsistent_dimensions(tmp_path):
    path = tmp_path / "inconsistent.h5"
    results = (
        _results(np.zeros((2, 32), dtype=np.float32), "000001")
        + _results(np.zeros((2, 64), dtype=np.float32), "000002")
    )

    with pytest.raises(ValueError, match="inconsistent embedding dimensions"):
        save_embeddings_h5(results, str(path), "none")
    assert not path.exists()


def test_embedding_h5_rejects_empty_results(tmp_path):
    path = tmp_path / "empty.h5"

    with pytest.raises(ValueError, match="at least one video"):
        save_embeddings_h5([], str(path), "none")
    assert not path.exists()


def test_l2_embedding_h5_rejects_non_unit_rows(tmp_path):
    path = tmp_path / "invalid.h5"
    embeddings = np.array([[2.0, 0.0], [0.0, 1.0]], dtype=np.float32)

    with pytest.raises(ValueError, match="not unit norm"):
        save_embeddings_h5(_results(embeddings), str(path), "l2")


def test_legacy_embedding_h5_defaults_to_none(tmp_path):
    path = tmp_path / "legacy.h5"
    with h5py.File(path, "w") as h5_file:
        h5_file.create_group("videos")

    with h5py.File(path, "r") as h5_file:
        with pytest.warns(UserWarning, match="legacy H5"):
            mode = read_embedding_normalization(h5_file, str(path))

    assert mode == "none"
