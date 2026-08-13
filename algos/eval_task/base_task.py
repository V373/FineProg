"""Base class for embeddings-based downstream evaluation tasks.

Inspired by google-research/tcc/evaluation/task.py, but stripped down to
support only the embeddings-dataset evaluation path (no iterator/encoder).
"""

import abc


class BaseTask(abc.ABC):
    """Abstract base for all embeddings-based evaluation tasks."""

    def __init__(self, task_name: str, downstream_task: bool = True):
        self.task_name = task_name
        # All tasks in this codebase operate on pre-extracted embeddings.
        self.downstream_task = downstream_task

    def configure(self, config: dict) -> None:
        """Apply a resolved V2 config dict to this task instance.

        V2-driven tasks (e.g. ExpertProjectionTask, LatentDistanceHeatmapTask)
        override this method to store their configuration.  Tasks that receive
        their configuration through constructor arguments or direct attribute
        assignment (e.g. KendallsTauTask) may leave this as a no-op.

        Args:
            config: Fully resolved config dict produced by
                ConfigV2.load_eval(task_name).  Keys are task-specific.
        """

    @abc.abstractmethod
    def evaluate(self, embeddings_dataset) -> dict:
        """Evaluate on a pre-extracted embeddings dataset.

        V2 H5-driven tasks (e.g. ExpertProjectionTask, LatentDistanceHeatmapTask)
        read their own H5 paths from self.config set via configure() and ignore
        the embeddings_dataset argument (pass None from evaluate.py).

        Args:
            embeddings_dataset: List or dict of records produced by
                extract_embeddings.py, or None for H5-driven tasks.

        Returns:
            dict with keys:
                "task_name"    (str)
                "metric_name"  (str)
                "metric_value" (float)

        Raises:
            NotImplementedError: If the subclass has not implemented this.
        """
        raise NotImplementedError(
            f"{self.__class__.__name__} must implement evaluate()."
        )


# ---------------------------------------------------------------------------
# Task builder
# ---------------------------------------------------------------------------

_SUPPORTED_TASKS = {
    "kendalls_tau",
    "classification",
    "expert_projection",
    "gaussian_progress_fitting",
    "gaussian_progress_pred",
    "latent_distance_heatmap",
    "activation_map",
    # "event_completion",
    # "few_shot_classification",
}


def build_task(task_name: str, config_path: str = None, **kwargs) -> BaseTask:
    """Factory that returns a concrete BaseTask instance.

    Deferred imports are used inside each branch to avoid circular dependencies.

    Args:
        task_name:   Name of the evaluation task.
        config_path: Optional path to a YAML/JSON config file for the task.

    Returns:
        An instance of the requested BaseTask subclass.

    Raises:
        ValueError: If task_name is not recognised.
    """
    if task_name == "kendalls_tau":
        from fineprog.algos.eval_task.tcc_eval_tasks.task_kendall import KendallsTauTask  # noqa: PLC0415
        return KendallsTauTask(config_path=config_path)

    if task_name == "classification":
        from fineprog.algos.eval_task.tcc_eval_tasks.task_phase_classification import PhaseClassificationTask  # noqa: PLC0415
        svm_c                = float(kwargs.get("svm_c",                1.0))
        max_iter             = int(kwargs.get("max_iter",             10000))
        output_dir           = kwargs.get("output_dir",                None)
        gen_tsne_phase_label = bool(kwargs.get("gen_tsne_phase_label", False))
        return PhaseClassificationTask(
            svm_c=svm_c, max_iter=max_iter, output_dir=output_dir,
            gen_tsne_phase_label=gen_tsne_phase_label,
        )

    elif task_name == "expert_projection":
        from fineprog.algos.eval_task.tcc_eval_tasks.task_expert_projection import ExpertProjectionTask  # noqa: PLC0415
        return ExpertProjectionTask()

    elif task_name == "gaussian_progress_fitting":
        from fineprog.algos.eval_task.tcc_eval_tasks.task_gaussian_progress_fitting import GaussianProgressFittingTask  # noqa: PLC0415
        return GaussianProgressFittingTask()

    elif task_name == "gaussian_progress_pred":
        from fineprog.algos.eval_task.tcc_eval_tasks.task_gaussian_progress_pred import GaussianProgressPredTask  # noqa: PLC0415
        return GaussianProgressPredTask()

    elif task_name == "latent_distance_heatmap":
        from fineprog.algos.eval_task.tcc_eval_tasks.task_latent_distance_heatmap import LatentDistanceHeatmapTask  # noqa: PLC0415
        return LatentDistanceHeatmapTask()

    elif task_name == "activation_map":
        from fineprog.algos.eval_task.tcc_eval_tasks.task_activation_map import ActivationMapTask  # noqa: PLC0415
        return ActivationMapTask()

    # ---- future tasks ----
    # elif task_name == "event_completion":
    # elif task_name == "few_shot_classification":

    raise ValueError(
        f"Unknown task '{task_name}'. Supported tasks: {sorted(_SUPPORTED_TASKS)}"
    )


# ---------------------------------------------------------------------------
# Minimal sanity check
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    task = build_task("kendalls_tau")
    print(f"task_name      : {task.task_name}")
    print(f"downstream_task: {task.downstream_task}")
