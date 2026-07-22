"""Scene-level normalized acceleration, jerk, and yaw-rate comfort metrics."""

from __future__ import annotations

import math

import numpy as np

from src.objective_metrics_config import OUTPUT_DIR
from src.waymo_metric_utils import InteractionContext, iter_contexts, metadata, state_from_array, write_csv

COMFORT_LINEAR_DECAY_SLOPE = 0.5
COMFORT_THRESHOLDS = {
    "a_p": (0.89, 2.12),
    "a_l": (0.40, 1.80),
    "jerk": (0.60, 1.00),
    "yaw_rate": (0.10, 0.37),
}
COMFORT_OUTPUT_FILES = {
    "a_p": "longitudinal_acceleration.csv",
    "a_l": "lateral_acceleration.csv",
    "jerk": "jerk.csv",
    "yaw_rate": "yaw_rate.csv",
}


def _instantaneous_comfort_value(
    metric_value: float,
    comfortable_threshold: float,
    uncomfortable_threshold: float,
    slope: float = COMFORT_LINEAR_DECAY_SLOPE,
) -> float:
    """Evaluate the continuous piecewise comfort function c(t)."""
    if not 0 <= comfortable_threshold < uncomfortable_threshold:
        raise ValueError(
            "Comfort thresholds must satisfy "
            f"0 <= m_c < m_u, got {comfortable_threshold}, {uncomfortable_threshold}."
        )
    if slope < 0:
        raise ValueError(f"Comfort linear decay slope must be non-negative, got {slope}.")

    metric_magnitude = abs(float(metric_value))
    if np.isnan(metric_magnitude):
        return np.nan
    threshold_range = uncomfortable_threshold - comfortable_threshold
    upper_value = 1.0 - slope * threshold_range
    if upper_value < 0:
        raise ValueError(
            f"Comfort slope {slope} makes c_u={upper_value} negative for "
            f"thresholds ({comfortable_threshold}, {uncomfortable_threshold})."
        )

    if metric_magnitude <= comfortable_threshold:
        return 1.0
    if metric_magnitude <= uncomfortable_threshold:
        return 1.0 - slope * (metric_magnitude - comfortable_threshold)
    return upper_value * math.exp(
        -(metric_magnitude - uncomfortable_threshold) / threshold_range
    )


def _standardized_comfort_score(
    metric_values: list[float],
    comfortable_threshold: float,
    uncomfortable_threshold: float,
    slope: float = COMFORT_LINEAR_DECAY_SLOPE,
) -> float:
    """Return S_c = 1 - mean(c(t)) for one interaction."""
    if not metric_values:
        return np.nan
    comfort_values = [
        _instantaneous_comfort_value(
            value,
            comfortable_threshold,
            uncomfortable_threshold,
            slope,
        )
        for value in metric_values
    ]
    if np.isnan(comfort_values).any():
        return np.nan
    return float(np.clip(np.mean(comfort_values), 0.0, 1.0))


def _comfort_state(context: InteractionContext, ego_index: int, time_index: int) -> tuple[float, float, float, float, float]:
    state = state_from_array(context.column_dict, context.states[ego_index, time_index, :])
    longitudinal_acc = state.ax * math.cos(state.h) + state.ay * math.sin(state.h)
    lateral_acc = -state.ax * math.sin(state.h) + state.ay * math.cos(state.h)
    longitudinal_speed = state.vx * math.cos(state.h) + state.vy * math.sin(state.h)
    jerk = _jerk_profile(context, ego_index, time_index)
    yaw_rate = lateral_acc / longitudinal_speed if longitudinal_speed >= 0.01 else 0
    return longitudinal_acc, lateral_acc, longitudinal_speed, jerk, yaw_rate


def _jerk_profile(context: InteractionContext, ego_index: int, time_index: int) -> float:
    """Return the planar jerk magnitude in m/s^3 at one timestep.

    Jerk is the time derivative of the world-frame acceleration vector. Interior
    samples use a central difference; the two endpoints use one-sided
    differences. A single-sample interaction has no measurable acceleration
    change and therefore receives a jerk value of zero.
    """
    sample_count = len(context.all_timesteps)
    if not 0 <= time_index < sample_count:
        raise IndexError(f"time_index={time_index} is outside the {sample_count}-sample interaction.")
    if context.dt <= 0:
        raise ValueError(f"Scene timestep must be positive, got dt={context.dt}.")
    if sample_count == 1:
        return 0.0

    if time_index == 0:
        previous_index, next_index, time_delta = 0, 1, context.dt
    elif time_index == sample_count - 1:
        previous_index, next_index, time_delta = sample_count - 2, sample_count - 1, context.dt
    else:
        previous_index, next_index, time_delta = time_index - 1, time_index + 1, 2 * context.dt

    previous_state = state_from_array(
        context.column_dict, context.states[ego_index, previous_index, :]
    )
    next_state = state_from_array(
        context.column_dict, context.states[ego_index, next_index, :]
    )
    jerk_x = (next_state.ax - previous_state.ax) / time_delta
    jerk_y = (next_state.ay - previous_state.ay) / time_delta
    return math.hypot(jerk_x, jerk_y)


def calculate_comfort_for_context(context: InteractionContext) -> dict[str, dict]:
    ego_index = context.all_agents.index(context.ego_id)
    base_row = metadata(context)
    metric_profiles = {metric_name: [] for metric_name in COMFORT_THRESHOLDS}

    for time_index, _ in enumerate(context.all_timesteps):
        ap, al, _, jerk, yr = _comfort_state(context, ego_index, time_index)
        metric_profiles["a_p"].append(ap)
        metric_profiles["a_l"].append(al)
        metric_profiles["jerk"].append(jerk)
        metric_profiles["yaw_rate"].append(yr)

    rows = {}
    for metric_name, metric_values in metric_profiles.items():
        comfortable_threshold, uncomfortable_threshold = COMFORT_THRESHOLDS[metric_name]
        row = dict(base_row)
        row.update({
            "comfortable_threshold": comfortable_threshold,
            "uncomfortable_threshold": uncomfortable_threshold,
            "linear_decay_slope": COMFORT_LINEAR_DECAY_SLOPE,
            "n_timesteps": len(metric_values),
            metric_name: _standardized_comfort_score(
                metric_values,
                comfortable_threshold,
                uncomfortable_threshold,
            ),
        })
        rows[metric_name] = row
    return rows


def calculate_comfort_metrics(target_id: int | None = None) -> None:
    metric_rows = {metric_name: [] for metric_name in COMFORT_THRESHOLDS}
    for context in iter_contexts(target_id):
        context_rows = calculate_comfort_for_context(context)
        for metric_name, row in context_rows.items():
            metric_rows[metric_name].append(row)

    output_dir = OUTPUT_DIR / "comfort"
    for metric_name, output_filename in COMFORT_OUTPUT_FILES.items():
        write_csv(metric_rows[metric_name], output_dir / output_filename)


def run(target_id: int | None = None) -> None:
    calculate_comfort_metrics(target_id)


if __name__ == "__main__":
    run()
