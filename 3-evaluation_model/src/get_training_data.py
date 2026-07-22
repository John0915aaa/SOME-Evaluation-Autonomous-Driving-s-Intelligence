"""Load and align objective metrics with subjective evaluation scores."""

from __future__ import annotations

from pathlib import Path

import pandas as pd


def _read_csv(path: Path, description: str) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"{description} CSV not found: {path}")
    data = pd.read_csv(path)
    if data.empty:
        raise ValueError(f"{description} CSV is empty: {path}")
    if "index" not in data.columns:
        raise ValueError(f"{description} CSV is missing the required 'index' column: {path}")
    return data


def _normalize_and_validate_index(data: pd.DataFrame, description: str) -> pd.DataFrame:
    data = data.copy()
    numeric_index = pd.to_numeric(data["index"], errors="coerce")
    if numeric_index.isna().any():
        bad_rows = data.index[numeric_index.isna()].tolist()[:10]
        raise ValueError(f"{description} contains invalid index values at row(s): {bad_rows}")
    if (numeric_index % 1 != 0).any():
        bad_values = numeric_index[numeric_index % 1 != 0].tolist()[:10]
        raise ValueError(f"{description} contains non-integer index values: {bad_values}")
    data["index"] = numeric_index.astype("int64")

    duplicate_mask = data["index"].duplicated(keep=False)
    if duplicate_mask.any():
        duplicate_indexes = sorted(data.loc[duplicate_mask, "index"].unique().tolist())
        raise ValueError(
            f"{description} contains duplicate index values: {duplicate_indexes[:10]}"
        )
    return data


def _validate_numeric_columns(data: pd.DataFrame, columns: list[str], description: str) -> pd.DataFrame:
    data = data.copy()
    for column in columns:
        try:
            data[column] = pd.to_numeric(data[column], errors="raise")
        except (TypeError, ValueError) as exc:
            raise ValueError(
                f"{description} column {column!r} contains non-numeric values."
            ) from exc
    missing_counts = data[columns].isna().sum()
    missing_counts = missing_counts[missing_counts > 0]
    if not missing_counts.empty:
        details = ", ".join(f"{column}={count}" for column, count in missing_counts.items())
        raise ValueError(f"{description} contains missing numeric values: {details}")
    return data


def load_training_data(metrics_path: Path, scores_path: Path) -> pd.DataFrame:
    """Return metrics joined one-to-one with scores after strict index checks."""
    metrics = _normalize_and_validate_index(
        _read_csv(metrics_path, "Objective metrics"),
        "Objective metrics",
    )
    scores = _normalize_and_validate_index(
        _read_csv(scores_path, "Subjective scores"),
        "Subjective scores",
    )

    if "score" not in scores.columns:
        raise ValueError(
            f"Subjective scores CSV is missing the required 'score' column: {scores_path}"
        )
    if "score" in metrics.columns:
        raise ValueError(
            "Objective metrics CSV must not contain a 'score' column; labels must come "
            f"from {scores_path}."
        )

    metric_indexes = set(metrics["index"])
    score_indexes = set(scores["index"])
    metrics_only = sorted(metric_indexes - score_indexes)
    scores_only = sorted(score_indexes - metric_indexes)
    if metrics_only or scores_only:
        raise ValueError(
            "Index mismatch between objective metrics and subjective scores. "
            f"Only in metrics ({len(metrics_only)}): {metrics_only[:10]}; "
            f"only in scores ({len(scores_only)}): {scores_only[:10]}."
        )

    metric_columns = [column for column in metrics.columns if column != "index"]
    if not metric_columns:
        raise ValueError("Objective metrics CSV does not contain any metric columns.")
    metrics = _validate_numeric_columns(metrics, metric_columns, "Objective metrics")
    scores = _validate_numeric_columns(scores[["index", "score"]], ["score"], "Subjective scores")

    training_data = metrics.merge(
        scores,
        on="index",
        how="inner",
        sort=False,
        validate="one_to_one",
    )
    if len(training_data) != len(metrics):
        raise RuntimeError(
            f"Unexpected merge result: metrics={len(metrics)}, merged={len(training_data)}."
        )
    return training_data
