"""Project configuration constants."""

from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT_DIR / "data"
MODEL_DIR = ROOT_DIR / "model"
NOTEBOOKS_DIR = ROOT_DIR / "notebooks"
LOG_FILE = DATA_DIR / "log_file.log"
MODEL_FILE = MODEL_DIR / "catboost_model.cbm"
METADATA_FILE = MODEL_DIR / "metadata.json"
TRAINING_SUMMARY_FILE = MODEL_DIR / "training_summary.json"
OPTUNA_STUDY_FILE = MODEL_DIR / "optuna_study.json"
RESULTS_FILE = DATA_DIR / "results.csv"
TARGET_COLUMN = "Status"
ID_COLUMN = "id"
CLASS_NAMES = ["C", "CL", "D"]
