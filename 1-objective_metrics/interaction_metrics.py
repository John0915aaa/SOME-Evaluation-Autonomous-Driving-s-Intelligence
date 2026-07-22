"""Social interaction metric: Interaction Orientation (IO)."""

from __future__ import annotations

import math

import numpy as np
from shapely.geometry import Point
from shapely.ops import nearest_points

from src.objective_metrics_config import OUTPUT_DIR
from src.utils.visualize_utils import process_tracks_single
from src.waymo_metric_utils import (
    VEHICLE_LENGTH,
    VEHICLE_WIDTH,
    InteractionContext,
    iter_contexts,
    metadata,
    state_from_array,
    write_csv,
)

DELTA_T_SECONDS = 0.5
CONFLICT_TIME_SECONDS = 5


def _normalize_angle(angle: float) -> float:
    return (angle + math.pi) % (2 * math.pi) - math.pi


def _priority_value(context: InteractionContext) -> float:
    """Map the scene right-of-way label to p(t): AV=0, HV=1, equal=0.5."""
    priority_label = str(context.row.get("priority_label", "")).strip()
    normalized_label = priority_label.lower()
    if normalized_label == "equal":
        return 0.5
    if normalized_label == "ego":
        return 0.0
    if normalized_label in {"hv", "hv_id"}:
        return 1.0

    track_ids = [agent.strip() for agent in str(context.track_id).split(";")]
    if priority_label == context.ego_id:
        return 0.0
    if priority_label in track_ids:
        return 1.0
    raise ValueError(
        f"Invalid priority_label={priority_label!r} for interaction index={context.index}; "
        f"expected 'ego', 'equal', or one of track_id={track_ids}."
    )


def _standardized_interaction_score(io_values: list[float],priority_value: float,) -> float:

    if not np.isfinite(priority_value):
        return 1.0

    if not io_values:
        return 1.0

    timestep_scores = []

    for io_value in io_values:
        if not np.isfinite(io_value):
            continue

        timestep_score = 1.0 - (io_value - priority_value) ** 2

        if np.isfinite(timestep_score):
            timestep_scores.append(timestep_score)

    if not timestep_scores:
        return 1.0

    score = float(np.mean(timestep_scores))

    if not np.isfinite(score):
        return 1.0

    return float(np.clip(score, 0.0, 1.0))


def _conflict_state(context: InteractionContext, av_index: int, hv_index: int, time_index: int):
    future_index = time_index + int(DELTA_T_SECONDS / 0.1)
    if future_index >= len(context.all_timesteps):
        return None
    position_index = [context.column_dict["x"], context.column_dict["y"]]
    velocity_index = [context.column_dict["vx"], context.column_dict["vy"]]
    timerange = CONFLICT_TIME_SECONDS / 0.1
    av_track = process_tracks_single(
        CONFLICT_TIME_SECONDS, context.states, av_index, time_index, timerange, position_index, velocity_index
    )
    hv_track = process_tracks_single(
        CONFLICT_TIME_SECONDS, context.states, hv_index, time_index, timerange, position_index, velocity_index
    )
    if av_track is None or hv_track is None:
        return None
    av_line, av_speed = av_track["line"], av_track["velocity"]
    hv_line, hv_speed = hv_track["line"], hv_track["velocity"]
    intersection = av_line.intersection(hv_line)
    if intersection.is_empty:
        return None
    av_state = state_from_array(context.column_dict, context.states[av_index, time_index, :])
    _, conflict_point = nearest_points(Point(av_state.x, av_state.y), intersection)
    av_distance = av_line.project(conflict_point)
    hv_distance = hv_line.project(conflict_point)
    future_av = state_from_array(context.column_dict, context.states[av_index, future_index, :])
    av_future_distance = av_line.project(Point(future_av.x, future_av.y))
    return conflict_point, av_distance, hv_distance, av_speed, hv_speed, av_future_distance


def _s_norm(
    av_distance: float,
    hv_distance: float,
    av_speed: float,
    hv_speed: float,
    delta_t: float,
    av_future_distance: float,
    delta_theta: float,
) -> float:
    if hv_speed <= 1e-6:
        return 0.0
    if delta_theta > math.pi / 2:
        delta_theta = math.pi - delta_theta
        hv_min = hv_distance + VEHICLE_LENGTH + VEHICLE_WIDTH * (1 / math.sin(delta_theta) + 1 / math.tan(delta_theta)) / 2
        av_min = av_distance
        hv_max = hv_distance
        av_max = av_distance + VEHICLE_LENGTH + VEHICLE_WIDTH * (1 / math.sin(delta_theta) + 1 / math.tan(delta_theta)) / 2
    else:
        hv_min = hv_distance + VEHICLE_LENGTH + VEHICLE_WIDTH * math.tan(delta_theta) / 2
        av_min = av_distance
        hv_max = hv_distance
        av_max = av_distance + VEHICLE_LENGTH + VEHICLE_WIDTH * math.tan(delta_theta) / 2

    t_min = hv_min / hv_speed
    t_max = hv_max / hv_speed
    if t_min <= 1e-6 or t_max <= 1e-6:
        return 0.0
    a_min = 2 * (av_min - av_speed * t_min) / t_min**2
    a_max = 2 * (av_max - av_speed * t_max) / t_max**2
    if abs(a_min) <= 1e-6 or abs(a_max) <= 1e-6:
        return 0.0
    s_min = -((av_speed + a_min * delta_t) ** 2 - av_speed**2) / (2 * a_min)
    s_max = ((av_speed + a_max * delta_t) ** 2 - av_speed**2) / (2 * a_max)
    if s_max == s_min:
        return 0.0
    return float(np.clip((av_future_distance - s_min) / (s_max - s_min), 0, 1))


def _itsi(av_distance: float, hv_distance: float, av_speed: float, hv_speed: float) -> float:
    if av_speed <= 1e-6 or hv_speed <= 1e-6 or hv_distance <= 1e-6:
        return 0.0
    ttcp_av = (av_distance + VEHICLE_LENGTH) / av_speed
    ttcp_hv = (hv_distance + VEHICLE_LENGTH) / hv_speed
    delta_ttcp = ttcp_av - ttcp_hv
    if hv_distance >= hv_speed * ttcp_av / 2:
        cooperative_acc = 2 * (hv_distance - hv_speed * ttcp_av) / ttcp_av**2
    else:
        cooperative_acc = hv_speed**2 / (2 * hv_distance)
    delta_ttcp_norm = 1 - (1 / (1 + math.exp(-delta_ttcp)))
    cooperative_acc_norm = 1 - (1 / (1 + math.exp(-cooperative_acc)))
    values = np.array([delta_ttcp_norm, cooperative_acc_norm], dtype=float)
    exp_values = np.exp(values - np.max(values))
    weights = exp_values / np.sum(exp_values)
    return float(np.dot(weights, values))


def calculate_interaction_for_context(context: InteractionContext,) -> dict:
    av_index = context.all_agents.index(context.ego_id)
    hv_index = context.all_agents.index(context.key_agent)
    priority_value = _priority_value(context)

    io_values = []
    invalid_io_timesteps = 0
    row = metadata(context)

    for time_index, _ in enumerate(context.all_timesteps):
        conflict_state = _conflict_state(
            context,
            av_index,
            hv_index,
            time_index,
        )

        if conflict_state is None:
            continue

        (
            _,
            av_distance,
            hv_distance,
            av_speed,
            hv_speed,
            av_future_distance,
        ) = conflict_state

        input_values = (
            av_distance,
            hv_distance,
            av_speed,
            hv_speed,
            av_future_distance,
        )

        if not all(np.isfinite(value) for value in input_values):
            invalid_io_timesteps += 1
            continue

        av_state = state_from_array(
            context.column_dict,
            context.states[av_index, time_index, :],
        )
        hv_state = state_from_array(
            context.column_dict,
            context.states[hv_index, time_index, :],
        )

        if not np.isfinite(av_state.h) or not np.isfinite(hv_state.h):
            invalid_io_timesteps += 1
            continue

        delta_theta = abs(
            _normalize_angle(av_state.h - hv_state.h)
        )

        s_norm = _s_norm(
            av_distance,
            hv_distance,
            av_speed,
            hv_speed,
            DELTA_T_SECONDS,
            av_future_distance,
            delta_theta,
        )

        itsi = _itsi(
            av_distance,
            hv_distance,
            av_speed,
            hv_speed,
        )

        io_value = 1.0 - itsi * s_norm

        if np.isfinite(io_value):
            io_values.append(float(io_value))
        else:
            invalid_io_timesteps += 1

    interaction_score = _standardized_interaction_score(
        io_values,
        priority_value,
    )

    if not np.isfinite(interaction_score):
        interaction_score = 1.0

    row.update({
        "priority_label": context.row["priority_label"],
        "priority_value": priority_value,
        "n_valid_timesteps": len(io_values),
        "n_invalid_timesteps": invalid_io_timesteps,
        "IO": interaction_score,
    })

    return row


def calculate_interaction_metrics(target_id: int | None = None) -> None:
    rows = [calculate_interaction_for_context(context) for context in iter_contexts(target_id)]
    write_csv(rows, OUTPUT_DIR / "interaction" / "io.csv")


def run(target_id: int | None = None) -> None:
    calculate_interaction_metrics(target_id)


if __name__ == "__main__":
    run()
