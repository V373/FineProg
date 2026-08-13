"""Report ranks of per-bin covariance matrices in a Gaussian progress H5 file.

The prediction H5 stores the fitted covariance matrices under
``model/bin_final_covariances``.  This script also reports the rank of their
elementwise arithmetic mean.  If the prediction file points to its source
Gaussian model, the stored pooled ``model/shared_covariance`` is reported too.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import h5py
import numpy as np


DEFAULT_H5_PATH = Path(
    "/home/user/zhangzk/projects/fineprog/outputs/gaussian_progress_pred/"
    "gaussian_progress_model-covariance_mode-independent-20260730-194319-pca16/"
    "robomimic_can_mh-100vid_worse-embd/20260730-194821/"
    "gaussian_progress_pred-20260730-194821.h5"
)
DEFAULT_DATASET = "model/bin_final_covariances"
DEFAULT_TOLERANCE = 1e-5


def _rank_details(
    matrix: np.ndarray, tolerance: float | None
) -> tuple[int, float, float]:
    """Return matrix rank, effective SVD tolerance, and minimum singular value."""
    singular_values = np.linalg.svd(matrix, compute_uv=False)
    effective_tolerance = tolerance
    if effective_tolerance is None:
        effective_tolerance = (
            singular_values.max()
            * max(matrix.shape)
            * np.finfo(singular_values.dtype).eps
        )
    rank = int(np.linalg.matrix_rank(matrix, tol=effective_tolerance))
    return rank, float(effective_tolerance), float(singular_values.min())


def _read_project_shared_covariance(
    input_path: Path,
) -> tuple[Path, np.ndarray] | None:
    """Read the project's stored pooled covariance locally or through a link."""
    with h5py.File(input_path, "r") as input_file:
        shared_path = "model/shared_covariance"
        if shared_path in input_file:
            shared_covariance = np.asarray(
                input_file[shared_path][:], dtype=np.float64
            )
            return input_path, shared_covariance
        source_value = input_file.attrs.get("gaussian_model_h5_path")

    if source_value is None:
        return None
    if isinstance(source_value, bytes):
        source_value = source_value.decode("utf-8")

    source_path = Path(str(source_value)).expanduser()
    if not source_path.is_absolute():
        source_path = input_path.parent / source_path
    if not source_path.is_file():
        return None

    with h5py.File(source_path, "r") as source_file:
        shared_path = "model/shared_covariance"
        if shared_path not in source_file:
            return None
        shared_covariance = np.asarray(source_file[shared_path][:], dtype=np.float64)
    return source_path, shared_covariance


def report_ranks(
    input_path: Path, dataset_path: str, tolerance: float | None
) -> None:
    """Load covariance matrices and print all requested ranks."""
    with h5py.File(input_path, "r") as input_file:
        if dataset_path not in input_file:
            raise KeyError(f"Dataset {dataset_path!r} not found in {input_path}")
        covariances = np.asarray(input_file[dataset_path][:], dtype=np.float64)

    if (
        covariances.ndim != 3
        or covariances.shape[1] != covariances.shape[2]
    ):
        raise ValueError(
            "Expected covariance shape (num_bins, dim, dim), "
            f"got {covariances.shape}"
        )

    print(f"H5: {input_path}")
    print(f"dataset: {dataset_path}")
    print(f"shape: {covariances.shape}")
    tolerance_label = (
        "NumPy default SVD tolerance"
        if tolerance is None
        else f"absolute SVD tolerance={tolerance:.12e}"
    )
    print(f"\nPer-bin ranks ({tolerance_label}):")
    ranks = []
    for bin_index, covariance in enumerate(covariances):
        rank, effective_tolerance, minimum_singular_value = _rank_details(
            covariance, tolerance
        )
        ranks.append(rank)
        print(
            f"  bin {bin_index:02d}: rank={rank:2d}, "
            f"min_singular={minimum_singular_value:.12e}, "
            f"tol={effective_tolerance:.12e}"
        )

    # Scaling by the number of bins does not change rank, so sum and mean have
    # identical ranks.  The mean is the natural equally weighted merge.
    mean_shared_covariance = covariances.mean(axis=0)
    mean_rank, mean_tolerance, mean_minimum_singular_value = _rank_details(
        mean_shared_covariance, tolerance
    )

    print(f"\nPer-bin rank list: {ranks}")
    print(
        "Equal-weight shared covariance (elementwise mean): "
        f"rank={mean_rank}, min_singular={mean_minimum_singular_value:.12e}, "
        f"tol={mean_tolerance:.12e}"
    )

    project_shared = _read_project_shared_covariance(input_path)
    if project_shared is not None:
        source_path, shared_covariance = project_shared
        shared_rank, shared_tolerance, shared_minimum_singular_value = (
            _rank_details(shared_covariance, tolerance)
        )
        print(f"Linked Gaussian model: {source_path}")
        print(
            "Project pooled shared covariance (stored in source model): "
            f"rank={shared_rank}, "
            f"min_singular={shared_minimum_singular_value:.12e}, "
            f"tol={shared_tolerance:.12e}"
        )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "h5_path",
        nargs="?",
        type=Path,
        default=DEFAULT_H5_PATH,
        help=f"Input H5 path (default: {DEFAULT_H5_PATH})",
    )
    parser.add_argument(
        "--dataset",
        default=DEFAULT_DATASET,
        help=f"Covariance dataset path (default: {DEFAULT_DATASET})",
    )
    parser.add_argument(
        "--tol",
        type=float,
        default=DEFAULT_TOLERANCE,
        help=f"Absolute singular-value tolerance (default: {DEFAULT_TOLERANCE:g})",
    )
    args = parser.parse_args()
    if args.tol is not None and args.tol < 0.0:
        parser.error("--tol must be non-negative")
    report_ranks(args.h5_path.expanduser().resolve(), args.dataset, args.tol)


if __name__ == "__main__":
    main()
