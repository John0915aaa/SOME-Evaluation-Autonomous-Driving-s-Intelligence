"""Safety metrics: Time-to-Collision (TTC) and Post-Encroachment Time (PET)."""

from __future__ import annotations

import math

import numpy as np
import pandas as pd
from shapely.geometry import Point
from shapely.ops import nearest_points

from src.objective_metrics_config import OUTPUT_DIR
from src.two_dim_ttc import TTC
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

PET_LOOKAHEAD_FRAMES = 2
CONFLICT_TIME_SECONDS = 5
PET_DISTANCE_THRESHOLD = 15

TTC_THRESHOLD_SECONDS = 3.0
PET_THRESHOLD_SECONDS = 2.0
SAFETY_EXP_FACTOR = 0.5


def _standardized_safety_score(
    metric_values: list[float],
    threshold_seconds: float,
    exp_factor: float = SAFETY_EXP_FACTOR,
) -> float:
    """Return 1 - mean(exp(-exp_factor * m(t) / threshold))."""
    if threshold_seconds <= 0:
        raise ValueError(
            f"Safety threshold must be positive, got {threshold_seconds}."
        )
    if exp_factor <= 0:
        raise ValueError(
            f"Safety exponential factor must be positive, got {exp_factor}."
        )
    if not metric_values:
        return np.nan

    values = np.asarray(metric_values, dtype=float)
    if np.isnan(values).any():
        return np.nan

    values = np.maximum(values, 0.0)
    score = 1.0 - np.mean(
        np.exp(-exp_factor * values / threshold_seconds)
    )
    return float(np.clip(score, 0.0, 1.0))


def _ttc_between_pair(
    context: InteractionContext,
    ego_index: int,
    agent_index: int,
    time_index: int,
) -> float:
    ego = state_from_array(
        context.column_dict,
        context.states[ego_index, time_index, :],
    )
    other = state_from_array(
        context.column_dict,
        context.states[agent_index, time_index, :],
    )

    sample = {
        "x_i": ego.x,
        "y_i": ego.y,
        "vx_i": ego.vx,
        "vy_i": ego.vy,
        "hx_i": math.cos(ego.h),
        "hy_i": math.sin(ego.h),
        "length_i": VEHICLE_LENGTH,
        "width_i": VEHICLE_WIDTH,
        "x_j": other.x,
        "y_j": other.y,
        "vx_j": other.vx,
        "vy_j": other.vy,
        "hx_j": math.cos(other.h),
        "hy_j": math.sin(other.h),
        "length_j": VEHICLE_LENGTH,
        "width_j": VEHICLE_WIDTH,
    }
    return TTC(pd.DataFrame(sample, index=[0]), "values")


def calculate_ttc_for_context(context: InteractionContext) -> dict:
    ego_index = context.all_agents.index(context.ego_id)
    agent_index = context.all_agents.index(context.key_agent)

    ttc_values = [
        _ttc_between_pair(
            context,
            ego_index,
            agent_index,
            time_index,
        )
        for time_index, _ in enumerate(context.all_timesteps)
    ]

    row = metadata(context)
    row.update({
        "threshold_s": TTC_THRESHOLD_SECONDS,
        "exp_factor": SAFETY_EXP_FACTOR,
        "n_timesteps": len(ttc_values),
        "TTC": _standardized_safety_score(
            ttc_values,
            TTC_THRESHOLD_SECONDS,
            SAFETY_EXP_FACTOR,
        ),
    })
    return row


def calculate_ttc_metrics(target_id: int | None = None) -> None:
    rows = [
        calculate_ttc_for_context(context)
        for context in iter_contexts(target_id)
    ]
    write_csv(rows, OUTPUT_DIR / "safety" / "ttc_score.csv")


def _projected_conflict_state(
    context: InteractionContext,
    first_index: int,
    second_index: int,
    time_index: int,
):
    if time_index + PET_LOOKAHEAD_FRAMES >= len(context.all_timesteps):
        return None

    position_index = [
        context.column_dict["x"],
        context.column_dict["y"],
    ]
    velocity_index = [
        context.column_dict["vx"],
        context.column_dict["vy"],
    ]

    timerange = CONFLICT_TIME_SECONDS / context.dt

    first_track = process_tracks_single(
        CONFLICT_TIME_SECONDS,
        context.states,
        first_index,
        time_index,
        timerange,
        position_index,
        velocity_index,
    )
    second_track = process_tracks_single(
        CONFLICT_TIME_SECONDS,
        context.states,
        second_index,
        time_index,
        timerange,
        position_index,
        velocity_index,
    )

    if first_track is None or second_track is None:
        return None

    first_line = first_track["line"]
    first_speed = first_track["velocity"]
    second_line = second_track["line"]
    second_speed = second_track["velocity"]

    intersection = first_line.intersection(second_line)
    if intersection.is_empty:
        return None

    first_state = state_from_array(
        context.column_dict,
        context.states[first_index, time_index, :],
    )
    _, conflict_point = nearest_points(
        Point(first_state.x, first_state.y),
        intersection,
    )

    first_distance = first_line.project(conflict_point)
    second_distance = second_line.project(conflict_point)

    first_future = state_from_array(
        context.column_dict,
        context.states[
            first_index,
            time_index + PET_LOOKAHEAD_FRAMES,
            :,
        ],
    )
    second_future = state_from_array(
        context.column_dict,
        context.states[
            second_index,
            time_index + PET_LOOKAHEAD_FRAMES,
            :,
        ],
    )

    first_delta = first_line.project(
        Point(first_future.x, first_future.y)
    )
    second_delta = second_line.project(
        Point(second_future.x, second_future.y)
    )

    return (
        conflict_point,
        first_distance,
        second_distance,
        first_speed,
        second_speed,
        first_delta,
        second_delta,
    )


def _find_conflict_point(
    context: InteractionContext,
    ego_index: int,
    agent_index: int,
):
    for time_index, _ in enumerate(context.all_timesteps):
        conflict_state = _projected_conflict_state(
            context,
            ego_index,
            agent_index,
            time_index,
        )
        if conflict_state is not None:
            return conflict_state[0]
    return None


def _pet_profile(
    context: InteractionContext,
    ego_index: int,
    agent_index: int,
) -> tuple[float, dict[str, float]]:
    """Return the minimum PET and valid timestep-level PET values."""
    if context.path_relationship != "CP":
        return np.inf, {}

    if len(context.all_timesteps) < 2:
        return np.inf, {}

    conflict_point = _find_conflict_point(
        context,
        ego_index,
        agent_index,
    )
    if conflict_point is None or conflict_point.is_empty:
        return np.inf, {}

    conflict_x = float(conflict_point.x)
    conflict_y = float(conflict_point.y)
    if not np.isfinite([conflict_x, conflict_y]).all():
        return np.inf, {}

    start_index = None
    for time_index, _ in enumerate(context.all_timesteps):
        ego = state_from_array(
            context.column_dict,
            context.states[ego_index, time_index, :],
        )
        other = state_from_array(
            context.column_dict,
            context.states[agent_index, time_index, :],
        )

        if not np.isfinite(
            [ego.x, ego.y, other.x, other.y]
        ).all():
            continue

        ego_distance = math.hypot(
            conflict_x - ego.x,
            conflict_y - ego.y,
        )
        other_distance = math.hypot(
            conflict_x - other.x,
            conflict_y - other.y,
        )

        if (
            ego_distance <= PET_DISTANCE_THRESHOLD
            or other_distance <= PET_DISTANCE_THRESHOLD
        ):
            start_index = time_index
            break

    if (
        start_index is None
        or start_index >= len(context.all_timesteps) - 1
    ):
        return np.inf, {}

    previous_ego = state_from_array(
        context.column_dict,
        context.states[ego_index, start_index, :],
    )
    previous_other = state_from_array(
        context.column_dict,
        context.states[agent_index, start_index, :],
    )

    pet_values: list[float] = []
    timed_values: dict[str, float] = {}
    clearance_distance = VEHICLE_LENGTH / 2 + VEHICLE_WIDTH / 2

    for time_index in range(
        start_index + 1,
        len(context.all_timesteps),
    ):
        timestamp = context.all_timesteps[time_index]

        ego = state_from_array(
            context.column_dict,
            context.states[ego_index, time_index, :],
        )
        other = state_from_array(
            context.column_dict,
            context.states[agent_index, time_index, :],
        )

        state_values = [
            ego.x,
            ego.y,
            ego.vx,
            ego.vy,
            other.x,
            other.y,
            other.vx,
            other.vy,
        ]
        if not np.isfinite(state_values).all():
            previous_ego, previous_other = ego, other
            continue

        ego_dx = conflict_x - ego.x
        ego_dy = conflict_y - ego.y
        other_dx = conflict_x - other.x
        other_dy = conflict_y - other.y

        ego_distance = math.hypot(ego_dx, ego_dy)
        other_distance = math.hypot(other_dx, other_dy)

        ego_approach_speed = (
            (ego.vx * ego_dx + ego.vy * ego_dy) / ego_distance
            if ego_distance > 1e-6
            else 0.0
        )
        other_approach_speed = (
            (other.vx * other_dx + other.vy * other_dy)
            / other_distance
            if other_distance > 1e-6
            else 0.0
        )

        if (
            ego_approach_speed > 1e-6
            and other_approach_speed > 1e-6
        ):
            ego_arrive = max(
                ego_distance - clearance_distance,
                0.0,
            ) / ego_approach_speed
            ego_pass = (
                ego_distance + VEHICLE_WIDTH / 2
            ) / ego_approach_speed

            other_arrive = max(
                other_distance - clearance_distance,
                0.0,
            ) / other_approach_speed
            other_pass = (
                other_distance + VEHICLE_WIDTH / 2
            ) / other_approach_speed

            if ego_arrive >= other_arrive:
                pet = ego_arrive - other_pass
            else:
                pet = other_arrive - ego_pass

            pet = max(float(pet), 0.0)
            if np.isfinite(pet):
                pet = round(pet, 3)
                pet_values.append(pet)
                timed_values[f"PET, t={timestamp}"] = pet

        previous_ego_vector = np.array([
            conflict_x - previous_ego.x,
            conflict_y - previous_ego.y,
        ])
        current_ego_vector = np.array([ego_dx, ego_dy])

        previous_other_vector = np.array([
            conflict_x - previous_other.x,
            conflict_y - previous_other.y,
        ])
        current_other_vector = np.array([other_dx, other_dy])

        ego_crossed = (
            np.dot(previous_ego_vector, current_ego_vector) <= 0
        )
        other_crossed = (
            np.dot(previous_other_vector, current_other_vector) <= 0
        )

        previous_ego, previous_other = ego, other

        if ego_crossed or other_crossed:
            break

    if not pet_values:
        return np.inf, {}

    return float(min(pet_values)), timed_values


def calculate_pet_for_context(context: InteractionContext) -> dict:
    ego_index = context.all_agents.index(context.ego_id)
    agent_index = context.all_agents.index(context.key_agent)

    minimum_pet, pet_values = _pet_profile(
        context,
        ego_index,
        agent_index,
    )

    if context.path_relationship == "CP":
        pet_profile = [
            pet_values.get(f"PET, t={timestamp}", np.inf)
            for timestamp in context.all_timesteps
        ]
    else:
        pet_profile = [np.inf] * len(context.all_timesteps)

    row = metadata(context)
    row.update({
        "type": context.path_relationship,
        "threshold_s": PET_THRESHOLD_SECONDS,
        "exp_factor": SAFETY_EXP_FACTOR,
        "n_timesteps": len(pet_profile),
        "n_valid_timesteps": len(pet_values),
        "minimum_pet_s": minimum_pet,
        "PET": _standardized_safety_score(
            pet_profile,
            PET_THRESHOLD_SECONDS,
            SAFETY_EXP_FACTOR,
        ),
    })
    return row


def calculate_pet_metrics(target_id: int | None = None) -> None:
    rows = [
        calculate_pet_for_context(context)
        for context in iter_contexts(target_id)
    ]
    write_csv(rows, OUTPUT_DIR / "safety" / "pet_score.csv")


def run(target_id: int | None = None) -> None:
    calculate_ttc_metrics(target_id)
    calculate_pet_metrics(target_id)


if __name__ == "__main__":
    run()