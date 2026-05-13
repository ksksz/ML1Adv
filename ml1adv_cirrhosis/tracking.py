"""Optional ClearML tracking helpers."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

try:
    from clearml import Dataset, Task
except Exception:  # pragma: no cover - optional integration
    Dataset = None
    Task = None


def clearml_enabled() -> bool:
    """Return True when ClearML integration is enabled via environment variable."""
    return os.getenv("ENABLE_CLEARML", "0") == "1" and Task is not None


def start_task() -> Any | None:
    """Start a ClearML task if integration is enabled."""
    if not clearml_enabled():
        return None
    project_name = os.getenv("CLEARML_PROJECT", "ML1Adv")
    task_name = os.getenv("CLEARML_TASK", "Cirrhosis Survival Training")
    return Task.init(project_name=project_name, task_name=task_name)


def log_training_context(
    task: Any | None,
    params: dict[str, Any],
    dataset_path: str,
) -> None:
    """Log hyperparameters and optionally upload dataset metadata."""
    if task is None:
        return
    task.connect(params, name="catboost_params")
    task.connect({"dataset_path": dataset_path}, name="data")
    if Dataset is not None:
        project_name = os.getenv("CLEARML_PROJECT", "ML1Adv")
        task_name = os.getenv("CLEARML_TASK", "Cirrhosis Survival Training")
        dataset = Dataset.create(
            dataset_project=project_name,
            dataset_name=f"{task_name} dataset snapshot",
        )
        dataset.add_files(dataset_path)
        dataset.finalize(auto_upload=True)


def log_training_results(
    task: Any | None,
    baseline_score: float,
    catboost_score: float,
    model_path: Path,
    summary_path: Path,
    optuna_path: Path | None = None,
) -> None:
    """Log metrics and artifacts to ClearML."""
    if task is None:
        return
    logger = task.get_logger()
    logger.report_scalar("metrics", "baseline_log_loss", value=baseline_score, iteration=0)
    logger.report_scalar("metrics", "catboost_cv_log_loss", value=catboost_score, iteration=0)
    task.upload_artifact("trained_model", artifact_object=str(model_path))
    task.upload_artifact("training_summary", artifact_object=str(summary_path))
    if optuna_path is not None and optuna_path.exists():
        task.upload_artifact("optuna_study", artifact_object=str(optuna_path))


def close_task(task: Any | None) -> None:
    """Close the ClearML task when it is active."""
    if task is not None:
        task.close()
