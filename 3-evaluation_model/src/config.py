"""Configuration for the evaluation model training pipeline."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[1]
OUTPUT_DIR = ROOT_DIR / "output"
OBJECTIVE_METRICS_CSV = ROOT_DIR.parent / "1-objective_metrics" / "output" / "metrics.csv"
SUBJECTIVE_SCORES_CSV = (
    ROOT_DIR.parent / "2-subjective_evaluation" / "output" / "evaluation_score.csv"
)

FEATURE_COLUMNS = ["TTC", "PET", "a_p", "a_l", "jerk", "yaw_rate", "task_time", "avg_delay", "IO", "impact"]
TARGET_COLUMN = "score"


@dataclass(frozen=True)
class TrainingConfig:
    metrics_path: Path = OBJECTIVE_METRICS_CSV
    scores_path: Path = SUBJECTIVE_SCORES_CSV
    output_dir: Path = OUTPUT_DIR
    seed: int = 42
    sample_random_state: int = 42
    shuffle_random_state: int = 42
    test_size: float = 0.15
    samples_per_score_bin: int = 600
    k_dim: int = 8
    v_dim: int = 8
    d_model: int = 16
    hidden_dims: tuple[int, int, int, int] = (20, 40, 20, 10)
    learning_rate: float = 0.005
    epochs: int = 122
    batch_size: int = 350
