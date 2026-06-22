from __future__ import annotations

from pathlib import Path
from typing import Sequence

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def save_progress_curve(
    progress: Sequence[float] | np.ndarray,
    output_path: str | Path,
    *,
    title: str = "TCC progress",
    subtitle: str | None = None,
) -> str:
    values = np.asarray(progress, dtype=np.float32).reshape(-1)
    if values.size == 0:
        raise ValueError("progress must contain at least one value.")

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    x = np.linspace(0.0, 1.0, values.size, dtype=np.float32)
    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.plot(x, values, color="#1f77b4", linewidth=1.8)
    ax.scatter(x, values, color="#1f77b4", s=12, alpha=0.55)
    ax.set_xlabel("video frame (normalised)")
    ax.set_ylabel("progress")
    ax.set_xlim(0.0, 1.0)
    ax.set_ylim(-0.02, 1.02)
    ax.grid(True, alpha=0.25)
    if subtitle:
        ax.set_title(f"{title}\n{subtitle}", fontsize=11)
    else:
        ax.set_title(title, fontsize=11)
    fig.tight_layout()
    fig.savefig(output_path, dpi=120)
    plt.close(fig)
    return str(output_path)
