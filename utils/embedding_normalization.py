"""Minimal metadata and numerical checks for embedding normalization."""

import warnings

import numpy as np


VALID_EMBEDDING_NORMALIZATIONS = ("none", "l2")


def validate_embedding_normalization(value, context: str) -> str:
    """Return a validated ``none``/``l2`` normalization mode."""
    if isinstance(value, bytes):
        value = value.decode("utf-8")
    if not isinstance(value, str) or value not in VALID_EMBEDDING_NORMALIZATIONS:
        raise ValueError(
            f"{context}: embedding_normalization must be one of "
            f"{VALID_EMBEDDING_NORMALIZATIONS}, got {value!r}."
        )
    return value


def read_embedding_normalization(h5_file, file_path: str) -> str:
    """Read root normalization metadata; missing metadata means legacy raw."""
    if "embedding_normalization" not in h5_file.attrs:
        warnings.warn(
            f"{file_path}: legacy H5 has no embedding_normalization "
            "metadata; treating it as 'none'.",
            UserWarning,
            stacklevel=2,
        )
        return "none"
    return validate_embedding_normalization(
        h5_file.attrs["embedding_normalization"],
        str(file_path),
    )


def validate_embeddings_for_normalization(
    embeddings: np.ndarray,
    normalization: str,
    context: str,
) -> None:
    """Validate finiteness and, for ``l2``, row-wise unit norms."""
    normalization = validate_embedding_normalization(normalization, context)
    array = np.asarray(embeddings)
    if array.ndim != 2:
        raise ValueError(
            f"{context}: embeddings must have shape [T, D], got {array.shape}."
        )
    if not np.issubdtype(array.dtype, np.number):
        raise ValueError(f"{context}: embeddings must be numeric.")
    if not np.isfinite(array).all():
        raise ValueError(f"{context}: embeddings contain NaN or Inf.")

    if normalization == "l2":
        norms = np.linalg.norm(array.astype(np.float64, copy=False), axis=1)
        valid = np.isclose(norms, 1.0, rtol=1.0e-5, atol=1.0e-5)
        if not np.all(valid):
            raise ValueError(
                f"{context}: embedding_normalization='l2' but "
                f"{int((~valid).sum())}/{len(norms)} rows are not unit norm; "
                f"norm range=[{float(norms.min()):.6g}, {float(norms.max()):.6g}]."
            )
