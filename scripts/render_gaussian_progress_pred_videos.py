"""Render three-panel videos from one Gaussian progress prediction H5."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

_PROJECTS_ROOT = Path(__file__).resolve().parents[2]
if str(_PROJECTS_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECTS_ROOT))

from fineprog.algos.eval_task.tcc_eval_tasks.task_gaussian_progress_pred import (
    _save_prediction_videos,
)


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Render raw-video, progress, and confidence MP4s for one Gaussian "
            "progress prediction H5."
        )
    )
    parser.add_argument(
        "--prediction_h5_path",
        required=True,
        help="Path to one gaussian_progress_pred-*.h5 file.",
    )
    args = parser.parse_args()
    _save_prediction_videos(Path(args.prediction_h5_path))


if __name__ == "__main__":
    main()
