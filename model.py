"""CLI entrypoint and production-ready model artifact for the homework."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import pandas as pd
from catboost import CatBoostClassifier

from ml1adv_cirrhosis.config import (
    CLASS_NAMES,
    DATA_DIR,
    ID_COLUMN,
    METADATA_FILE,
    MODEL_DIR,
    MODEL_FILE,
    OPTUNA_STUDY_FILE,
    RESULTS_FILE,
    TRAINING_SUMMARY_FILE,
)
from ml1adv_cirrhosis.logging_utils import get_logger
from ml1adv_cirrhosis.pipeline import (
    TrainingArtifacts,
    baseline_cross_val_log_loss,
    catboost_cross_val_log_loss,
    detect_categorical_columns,
    get_default_catboost_params,
    load_dataset,
    prepare_catboost_features,
    run_optuna_study,
    split_features_target,
)


class My_Classifier_Model:
    """Train and serve the cirrhosis survival classifier."""

    def __init__(self) -> None:
        self.logger = get_logger()
        DATA_DIR.mkdir(parents=True, exist_ok=True)
        MODEL_DIR.mkdir(parents=True, exist_ok=True)

    def train(
        self,
        dataset: str,
        optuna_trials: int = 0,
        iterations: int | None = None,
        learning_rate: float | None = None,
        depth: int | None = None,
    ) -> dict[str, Any]:
        """Train the model, save artifacts, and log results."""
        try:
            self.logger.info("Starting training with dataset: %s", dataset)
            dataframe = load_dataset(dataset)
            features, target = split_features_target(dataframe)
            categorical_columns = detect_categorical_columns(features)
            prepared_features = prepare_catboost_features(features, categorical_columns)
            categorical_indices = [
                prepared_features.columns.get_loc(column) for column in categorical_columns
            ]

            baseline_score = baseline_cross_val_log_loss(features, target)
            if optuna_trials > 0:
                self.logger.info("Running Optuna with %d trials", optuna_trials)
                _, catboost_params, optuna_trials_summary = run_optuna_study(
                    features=features,
                    target=target,
                    n_trials=optuna_trials,
                )
                OPTUNA_STUDY_FILE.write_text(
                    json.dumps(optuna_trials_summary, ensure_ascii=False, indent=2),
                    encoding="utf-8",
                )
            else:
                catboost_params = get_default_catboost_params()
            if iterations is not None:
                catboost_params["iterations"] = iterations
            if learning_rate is not None:
                catboost_params["learning_rate"] = learning_rate
            if depth is not None:
                catboost_params["depth"] = depth

            catboost_score = catboost_cross_val_log_loss(
                features=features,
                target=target,
                params=catboost_params,
            )

            final_model = CatBoostClassifier(**catboost_params)
            final_model.fit(
                prepared_features,
                target,
                cat_features=categorical_indices,
            )
            final_model.save_model(MODEL_FILE)

            training_artifacts = TrainingArtifacts(
                baseline_log_loss=baseline_score,
                catboost_cv_log_loss=catboost_score,
                best_params=catboost_params,
                cat_columns=categorical_columns,
                feature_columns=prepared_features.columns.tolist(),
            )

            METADATA_FILE.write_text(
                json.dumps(
                    {
                        "class_names": CLASS_NAMES,
                        "cat_columns": categorical_columns,
                        "feature_columns": prepared_features.columns.tolist(),
                    },
                    ensure_ascii=False,
                    indent=2,
                ),
                encoding="utf-8",
            )
            TRAINING_SUMMARY_FILE.write_text(
                training_artifacts.to_json(),
                encoding="utf-8",
            )

            self.logger.info(
                "Training finished. Baseline log loss: %.6f | CatBoost CV log loss: %.6f",
                baseline_score,
                catboost_score,
            )
            return json.loads(training_artifacts.to_json())
        except Exception as exc:
            self.logger.exception("Training failed: %s", exc)
            raise

    def predict(self, dataset: str) -> Path:
        """Load the saved model and generate a submission file."""
        try:
            self.logger.info("Starting inference with dataset: %s", dataset)
            if not MODEL_FILE.exists() or not METADATA_FILE.exists():
                raise FileNotFoundError(
                    "Trained model artifacts were not found. Run the train command first."
                )

            metadata = json.loads(METADATA_FILE.read_text(encoding="utf-8"))
            dataframe = load_dataset(dataset)
            features = dataframe[metadata["feature_columns"]].copy()
            prepared_features = prepare_catboost_features(
                features, metadata["cat_columns"]
            )

            model = CatBoostClassifier()
            model.load_model(MODEL_FILE)
            predicted_probabilities = model.predict_proba(prepared_features)

            results = pd.DataFrame(predicted_probabilities, columns=model.classes_)
            results = results.reindex(columns=CLASS_NAMES, fill_value=0.0)
            results.columns = [f"Status_{class_name}" for class_name in results.columns]
            results.insert(0, ID_COLUMN, dataframe[ID_COLUMN].values)
            results.to_csv(RESULTS_FILE, index=False)

            self.logger.info("Inference finished. Results saved to %s", RESULTS_FILE)
            return RESULTS_FILE
        except Exception as exc:
            self.logger.exception("Prediction failed: %s", exc)
            raise


def build_parser() -> argparse.ArgumentParser:
    """Construct the CLI argument parser."""
    parser = argparse.ArgumentParser(description="Cirrhosis survival classifier CLI")
    subparsers = parser.add_subparsers(dest="command", required=True)

    train_parser = subparsers.add_parser("train", help="Train the model")
    train_parser.add_argument("--dataset", required=True, help="Path to train.csv")
    train_parser.add_argument("--optuna-trials", type=int, default=0)
    train_parser.add_argument("--iterations", type=int, default=None)
    train_parser.add_argument("--learning-rate", type=float, default=None)
    train_parser.add_argument("--depth", type=int, default=None)

    predict_parser = subparsers.add_parser("predict", help="Run inference")
    predict_parser.add_argument("--dataset", required=True, help="Path to test.csv")
    return parser


def main() -> None:
    """Run the command line interface."""
    args = build_parser().parse_args()
    model = My_Classifier_Model()

    if args.command == "train":
        summary = model.train(
            dataset=args.dataset,
            optuna_trials=args.optuna_trials,
            iterations=args.iterations,
            learning_rate=args.learning_rate,
            depth=args.depth,
        )
        print(json.dumps(summary, ensure_ascii=False, indent=2))
    elif args.command == "predict":
        output_path = model.predict(dataset=args.dataset)
        print(str(output_path))


if __name__ == "__main__":
    main()
