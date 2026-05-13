"""Reusable data preparation and training utilities."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from typing import Any

import numpy as np
import optuna
import pandas as pd
from catboost import CatBoostClassifier
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import log_loss
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from ml1adv_cirrhosis.config import CLASS_NAMES, TARGET_COLUMN


@dataclass
class TrainingArtifacts:
    """Summary of fitted training artifacts and metrics."""

    baseline_log_loss: float
    catboost_cv_log_loss: float
    best_params: dict[str, Any]
    cat_columns: list[str]
    feature_columns: list[str]

    def to_json(self) -> str:
        """Serialize artifacts to a JSON string."""
        return json.dumps(asdict(self), ensure_ascii=False, indent=2)


def load_dataset(dataset_path: str) -> pd.DataFrame:
    """Load dataset from a CSV file."""
    return pd.read_csv(dataset_path)


def split_features_target(dataframe: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    """Split training data into features and target."""
    features = dataframe.drop(columns=[TARGET_COLUMN]).copy()
    target = dataframe[TARGET_COLUMN].copy()
    return features, target


def detect_categorical_columns(features: pd.DataFrame) -> list[str]:
    """Detect categorical columns by dtype."""
    return features.select_dtypes(include="object").columns.tolist()


def prepare_catboost_features(
    features: pd.DataFrame, categorical_columns: list[str]
) -> pd.DataFrame:
    """Convert categorical missing values to strings for CatBoost."""
    prepared = features.copy()
    for column in categorical_columns:
        prepared[column] = prepared[column].fillna("missing").astype(str)
    return prepared


def build_baseline_pipeline(features: pd.DataFrame) -> Pipeline:
    """Create the baseline RandomForest pipeline."""
    categorical_columns = detect_categorical_columns(features)
    numeric_columns = [column for column in features.columns if column not in categorical_columns]

    preprocessor = ColumnTransformer(
        transformers=[
            (
                "numeric",
                Pipeline(
                    steps=[
                        ("imputer", SimpleImputer(strategy="median")),
                        ("scaler", StandardScaler()),
                    ]
                ),
                numeric_columns,
            ),
            (
                "categorical",
                Pipeline(
                    steps=[
                        ("imputer", SimpleImputer(strategy="most_frequent")),
                        ("encoder", OneHotEncoder(handle_unknown="ignore")),
                    ]
                ),
                categorical_columns,
            ),
        ]
    )

    return Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            (
                "model",
                RandomForestClassifier(
                    n_estimators=300,
                    max_depth=None,
                    class_weight="balanced_subsample",
                    random_state=42,
                    n_jobs=-1,
                ),
            ),
        ]
    )


def baseline_cross_val_log_loss(features: pd.DataFrame, target: pd.Series) -> float:
    """Evaluate the baseline model via cross-validation."""
    pipeline = build_baseline_pipeline(features)
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    predicted_probabilities = cross_val_predict(
        pipeline,
        features,
        target,
        cv=cv,
        method="predict_proba",
        n_jobs=1,
    )
    return float(log_loss(target, predicted_probabilities, labels=CLASS_NAMES))


def get_default_catboost_params() -> dict[str, Any]:
    """Return the tuned CatBoost parameters used for training."""
    return {
        "loss_function": "MultiClass",
        "eval_metric": "MultiClass",
        "iterations": 521,
        "depth": 7,
        "learning_rate": 0.055494458850482806,
        "l2_leaf_reg": 1.813367026107457,
        "random_strength": 0.8559635774451427,
        "bagging_temperature": 1.2029489348760298,
        "border_count": 206,
        "min_data_in_leaf": 34,
        "grow_policy": "Depthwise",
        "auto_class_weights": None,
        "random_seed": 42,
        "thread_count": -1,
        "verbose": False,
    }


def catboost_cross_val_log_loss(
    features: pd.DataFrame,
    target: pd.Series,
    params: dict[str, Any] | None = None,
) -> float:
    """Evaluate CatBoost with stratified cross-validation."""
    prepared_features = prepare_catboost_features(
        features, detect_categorical_columns(features)
    )
    categorical_columns = detect_categorical_columns(prepared_features)
    categorical_indices = [prepared_features.columns.get_loc(column) for column in categorical_columns]
    class_to_index = {class_name: index for index, class_name in enumerate(CLASS_NAMES)}
    predicted_probabilities = np.zeros((len(prepared_features), len(CLASS_NAMES)))
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    model_params = params or get_default_catboost_params()

    for train_index, valid_index in cv.split(prepared_features, target):
        fold_model = CatBoostClassifier(**model_params)
        fold_model.fit(
            prepared_features.iloc[train_index],
            target.iloc[train_index],
            cat_features=categorical_indices,
            eval_set=(
                prepared_features.iloc[valid_index],
                target.iloc[valid_index],
            ),
            use_best_model=True,
        )
        fold_predictions = fold_model.predict_proba(prepared_features.iloc[valid_index])
        ordered_probabilities = np.zeros((len(valid_index), len(CLASS_NAMES)))
        for class_index, class_name in enumerate(fold_model.classes_):
            ordered_probabilities[:, class_to_index[class_name]] = fold_predictions[:, class_index]
        predicted_probabilities[valid_index] = ordered_probabilities

    return float(log_loss(target, predicted_probabilities, labels=CLASS_NAMES))


def run_optuna_study(
    features: pd.DataFrame,
    target: pd.Series,
    n_trials: int = 12,
) -> tuple[float, dict[str, Any], list[dict[str, Any]]]:
    """Run Optuna search for CatBoost hyperparameters."""
    prepared_features = prepare_catboost_features(
        features, detect_categorical_columns(features)
    )
    categorical_columns = detect_categorical_columns(prepared_features)
    categorical_indices = [prepared_features.columns.get_loc(column) for column in categorical_columns]
    class_to_index = {class_name: index for index, class_name in enumerate(CLASS_NAMES)}
    cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)
    trials_summary: list[dict[str, Any]] = []

    def objective(trial: optuna.Trial) -> float:
        params = {
            "loss_function": "MultiClass",
            "eval_metric": "MultiClass",
            "iterations": trial.suggest_int("iterations", 300, 900),
            "depth": trial.suggest_int("depth", 4, 8),
            "learning_rate": trial.suggest_float("learning_rate", 0.02, 0.15, log=True),
            "l2_leaf_reg": trial.suggest_float("l2_leaf_reg", 1.0, 10.0, log=True),
            "random_strength": trial.suggest_float("random_strength", 0.1, 5.0, log=True),
            "bagging_temperature": trial.suggest_float("bagging_temperature", 0.0, 2.0),
            "border_count": trial.suggest_int("border_count", 32, 255),
            "min_data_in_leaf": trial.suggest_int("min_data_in_leaf", 1, 64),
            "grow_policy": trial.suggest_categorical(
                "grow_policy", ["SymmetricTree", "Depthwise"]
            ),
            "auto_class_weights": trial.suggest_categorical(
                "auto_class_weights", [None, "Balanced"]
            ),
            "random_seed": 42,
            "thread_count": -1,
            "verbose": False,
        }
        predicted_probabilities = np.zeros((len(prepared_features), len(CLASS_NAMES)))
        for train_index, valid_index in cv.split(prepared_features, target):
            fold_model = CatBoostClassifier(**params)
            fold_model.fit(
                prepared_features.iloc[train_index],
                target.iloc[train_index],
                cat_features=categorical_indices,
                eval_set=(
                    prepared_features.iloc[valid_index],
                    target.iloc[valid_index],
                ),
                use_best_model=True,
            )
            fold_predictions = fold_model.predict_proba(prepared_features.iloc[valid_index])
            ordered_probabilities = np.zeros((len(valid_index), len(CLASS_NAMES)))
            for class_index, class_name in enumerate(fold_model.classes_):
                ordered_probabilities[:, class_to_index[class_name]] = fold_predictions[:, class_index]
            predicted_probabilities[valid_index] = ordered_probabilities
        score = float(log_loss(target, predicted_probabilities, labels=CLASS_NAMES))
        trials_summary.append(
            {
                "trial": trial.number,
                "value": score,
                "params": {
                    key: value for key, value in params.items() if key not in {"verbose", "thread_count"}
                },
            }
        )
        return score

    study = optuna.create_study(direction="minimize")
    study.optimize(objective, n_trials=n_trials, show_progress_bar=False)

    best_params = {
        "loss_function": "MultiClass",
        "eval_metric": "MultiClass",
        **study.best_params,
        "random_seed": 42,
        "thread_count": -1,
        "verbose": False,
    }
    return float(study.best_value), best_params, trials_summary
