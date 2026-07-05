from __future__ import annotations

from dataclasses import dataclass
from collections.abc import Callable, Iterable
from typing import Any

import torch
from sklearn.metrics import accuracy_score, f1_score

from continual_experiment_manager import ContinualExperimentManager


@dataclass(frozen=True)
class TaskDefinition:
    """
    Describes one sequential learning task.

    The task factories must return fresh IterableDataset objects,
    because the datasets are consumed during embedding.
    """

    name: str
    train_factory: Callable[[], Iterable]
    test_factory: Callable[[], Iterable]
    label_transform: Callable[[str], str]


def run_continual_experiment(
    task_definitions: list[TaskDefinition],
    feature_pipeline: Any,
    strategy: Any,
    strategy_config: dict,
    device: torch.device,
):
    """
    Shared continual-learning experiment loop.

    This function does not care whether `strategy` is:
    - Multiclass PNN with replay
    - Binary task-aware PNN
    - SVM
    - Transformer classifier
    - confidence-router model

    Every strategy must implement:
        initialize(...)
        train_stage(...)
        predict_labels(...)
    """

    if not task_definitions:
        raise ValueError(
            "task_definitions cannot be empty."
        )

    experiment_manager = ContinualExperimentManager(
        num_tasks=len(task_definitions),
        use_intervention_override=strategy_config.get(
            "use_intervention_override",
            True,
        ),
    )

    strategy_state = None

    # Stores transformed test data for all tasks seen so far.
    test_cache = []

    # Useful for saving results later as JSON / CSV.
    all_step_metrics = []

    for task_index, task in enumerate(task_definitions):
        print("\n" + "=" * 72)
        print(f"STEP {task_index}: {task.name}")
        print("=" * 72)

        # ---------------------------------------------------------
        # 1. Shared training feature preparation
        # ---------------------------------------------------------
        raw_train, train_labels = feature_pipeline.embed_split(
            dataset=task.train_factory(),
            task_name=f"{task.name}/train",
            label_transform=task.label_transform,
        )

        if len(raw_train) == 0 or not train_labels:
            raise RuntimeError(
                f"No training samples were produced for "
                f"task '{task.name}'."
            )

        train_labels = [
            str(label)
            for label in train_labels
        ]

        # Fit RobustScaler + UMAP only once, on Task 0.
        if task_index == 0:
            X_train = (
                feature_pipeline.fit_transform_first_task(
                    raw_train
                )
            )
        else:
            X_train = feature_pipeline.transform(
                raw_train
            )

        # ---------------------------------------------------------
        # 2. Classifier-specific training
        # ---------------------------------------------------------
        stage = {
            "task_index": task_index,
            "task_name": task.name,
            "X_train": X_train,
            "y_train": train_labels,
            "device": device,
        }

        if strategy_state is None:
            strategy_state = strategy.initialize(
                feature_dim=X_train.shape[1],
                config=strategy_config,
                device=device,
            )

        strategy_state, train_info = strategy.train_stage(
            state=strategy_state,
            stage=stage,
        )

        # ---------------------------------------------------------
        # 3. Shared test feature preparation
        # ---------------------------------------------------------
        raw_test, test_labels = feature_pipeline.embed_split(
            dataset=task.test_factory(),
            task_name=f"{task.name}/test",
            label_transform=task.label_transform,
        )

        if len(raw_test) == 0 or not test_labels:
            raise RuntimeError(
                f"No test samples were produced for "
                f"task '{task.name}'."
            )

        test_labels = [
            str(label)
            for label in test_labels
        ]

        X_test = feature_pipeline.transform(
            raw_test
        )

        test_cache.append(
            {
                "task_index": task_index,
                "task_name": task.name,
                "X_test": X_test,
                "y_test": test_labels,
            }
        )

        # ---------------------------------------------------------
        # 4. Shared evaluation over all tasks seen so far
        # ---------------------------------------------------------
        print(
            f"Evaluating {len(test_cache)} "
            "test task(s)..."
        )

        step_metrics = []

        for evaluated_task_index, cached_task in enumerate(
            test_cache
        ):
            predictions = strategy.predict_labels(
                state=strategy_state,
                features=cached_task["X_test"],
                evaluation_context={
                    "task_index": cached_task["task_index"],
                    "task_name": cached_task["task_name"],
                },
            )

            if len(predictions) != len(cached_task["y_test"]):
                raise RuntimeError(
                    f"Prediction count mismatch for "
                    f"'{cached_task['task_name']}': "
                    f"expected {len(cached_task['y_test'])}, "
                    f"got {len(predictions)}."
                )

            accuracy = accuracy_score(
                cached_task["y_test"],
                predictions,
            )

            macro_f1 = f1_score(
                cached_task["y_test"],
                predictions,
                average="macro",
                zero_division=0,
            )

            print(
                f"  Task {evaluated_task_index} "
                f"— {cached_task['task_name']}: "
                f"accuracy={accuracy:.4f}, "
                f"macro-F1={macro_f1:.4f}"
            )

            experiment_manager.record_accuracy(
                task_index=evaluated_task_index,
                accuracy=accuracy,
            )

            step_metrics.append(
                {
                    "evaluated_task_index": evaluated_task_index,
                    "task_name": cached_task["task_name"],
                    "accuracy": accuracy,
                    "macro_f1": macro_f1,
                }
            )

        all_step_metrics.append(
            {
                "learning_step": task_index,
                "learning_task_name": task.name,
                "train_info": train_info,
                "evaluation_metrics": step_metrics,
            }
        )

        experiment_manager.advance_to_next_task()

    experiment_manager.summary()

    return {
        "strategy_state": strategy_state,
        "experiment_manager": experiment_manager,
        "all_step_metrics": all_step_metrics,
    }