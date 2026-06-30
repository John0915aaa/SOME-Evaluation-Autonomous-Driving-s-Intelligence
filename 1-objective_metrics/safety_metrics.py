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


def _ttc_between_pair(context: InteractionContext, ego_index: int, agent_index: int, time_index: int) -> float:
    ego = state_from_array(context.column_dict, context.states[ego_index, time_index, :])
    other = state_from_array(context.column_dict, context.states[agent_index, time_index, :])
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


def calculate_ttc_metrics(target_id: int | None = None) -> None:
    rows = []
    for context in iter_contexts(target_id):
        ego_index = context.all_agents.index(context.ego_id)
        agent_index = context.all_agents.index(context.key_agent)
        row = metadata(context)
        for time_index, timestamp in enumerate(context.all_timesteps):
            row[f"TTC, t={timestamp}"] = _ttc_between_pair(context, ego_index, agent_index, time_index)
        rows.append(row)
    write_csv(rows, OUTPUT_DIR / "safety" / "ttc_results.csv")


def _projected_conflict_state(context: InteractionContext, first_index: int, second_index: int, time_index: int):
    if time_index + PET_LOOKAHEAD_FRAMES >= len(context.all_timesteps):
        return None
    position_index = [context.column_dict["x"], context.column_dict["y"]]
    velocity_index = [context.column_dict["vx"], context.column_dict["vy"]]
    timerange = CONFLICT_TIME_SECONDS / 0.1
    first_track = process_tracks_single(
        CONFLICT_TIME_SECONDS, context.states, first_index, time_index, timerange, position_index, velocity_index
    )
    second_track = process_tracks_single(
        CONFLICT_TIME_SECONDS, context.states, second_index, time_index, timerange, position_index, velocity_index
    )
    if first_track is None or second_track is None:
        return None
    first_line, first_speed = first_track["line"], first_track["velocity"]
    second_line, second_speed = second_track["line"], second_track["velocity"]
    intersection = first_line.intersection(second_line)
    if intersection.is_empty:
        return None
    first_state = state_from_array(context.column_dict, context.states[first_index, time_index, :])
    _, conflict_point = nearest_points(Point(first_state.x, first_state.y), intersection)
    first_distance = first_line.project(conflict_point)
    second_distance = second_line.project(conflict_point)
    first_future = state_from_array(
        context.column_dict, context.states[first_index, time_index + PET_LOOKAHEAD_FRAMES, :]
    )
    second_future = state_from_array(
        context.column_dict, context.states[second_index, time_index + PET_LOOKAHEAD_FRAMES, :]
    )
    first_delta = first_line.project(Point(first_future.x, first_future.y))
    second_delta = second_line.project(Point(second_future.x, second_future.y))
    return conflict_point, first_distance, second_distance, first_speed, second_speed, first_delta, second_delta


def _find_conflict_point(context: InteractionContext, ego_index: int, agent_index: int):
    for time_index, _ in enumerate(context.all_timesteps):
        conflict_state = _projected_conflict_state(context, ego_index, agent_index, time_index)
        if conflict_state is not None:
            return conflict_state[0]
    return None


def _pet_profile(context: InteractionContext, ego_index: int, agent_index: int) -> tuple[float, dict]:
    if context.path_relationship != "CP":
        return np.inf, {}
    conflict_point = _find_conflict_point(context, ego_index, agent_index)
    if conflict_point is None:
        return np.inf, {}

    start_timestamp = None
    start_ego_distance = None
    for time_index, timestamp in enumerate(context.all_timesteps):
        ego = state_from_array(context.column_dict, context.states[ego_index, time_index, :])
        other = state_from_array(context.column_dict, context.states[agent_index, time_index, :])
        ego_distance = conflict_point.distance(Point(ego.x, ego.y))
        other_distance = conflict_point.distance(Point(other.x, other.y))
        if ego_distance <= PET_DISTANCE_THRESHOLD or other_distance <= PET_DISTANCE_THRESHOLD:
            start_timestamp = timestamp
            start_ego_distance = ego_distance
            break
    if start_timestamp is None:
        return np.inf, {}

    pet_values: list[float] = []
    timed_values: dict = {}
    previous_ego = state_from_array(context.column_dict, context.states[ego_index, 0, :])
    previous_other = state_from_array(context.column_dict, context.states[agent_index, 0, :])
    end_timestamp = None

    for time_index, timestamp in enumerate(context.all_timesteps):
        if timestamp <= start_timestamp:
            continue
        ego = state_from_array(context.column_dict, context.states[ego_index, time_index, :])
        other = state_from_array(context.column_dict, context.states[agent_index, time_index, :])
        ego_distance = start_ego_distance
        other_distance = conflict_point.distance(Point(other.x, other.y))
        ego_speed = ego.vx * math.cos(ego.h) + ego.vy * math.sin(ego.h)
        other_speed = other.vx * math.cos(other.h) + other.vy * math.sin(other.h)
        if abs(ego_speed) < 1e-6 or abs(other_speed) < 1e-6:
            continue

        ego_pass = (ego_distance + VEHICLE_WIDTH / 2) / ego_speed
        ego_arrive = (ego_distance - VEHICLE_LENGTH / 2 - VEHICLE_WIDTH / 2) / ego_speed
        other_pass = (other_distance + VEHICLE_WIDTH / 2) / other_speed
        other_arrive = (other_distance - VEHICLE_LENGTH / 2 - VEHICLE_WIDTH / 2) / other_speed
        pet = ego_pass - other_arrive if ego_pass > other_arrive else other_pass - ego_arrive
        pet = round(float(pet), 3)
        pet_values.append(pet)
        timed_values[f"PET, t={timestamp}"] = pet

        ego_before = np.array([conflict_point.x - previous_ego.x, conflict_point.y - previous_ego.y])
        ego_after = np.array([conflict_point.x - ego.x, conflict_point.y - ego.y])
        other_before = np.array([conflict_point.x - previous_other.x, conflict_point.y - previous_other.y])
        other_after = np.array([conflict_point.x - other.x, conflict_point.y - other.y])
        if np.dot(ego_before, ego_after) < 0 or np.dot(other_before, other_after) < 0:
            end_timestamp = timestamp
            break

    if not pet_values:
        return np.inf, {}
    if end_timestamp is not None:
        timed_values = {
            key: value for key, value in timed_values.items()
            if int(key.split("t=")[1]) >= start_timestamp and int(key.split("t=")[1]) < end_timestamp
        }
    return min(pet_values), timed_values


def calculate_pet_metrics(target_id: int | None = None) -> None:
    rows = []
    for context in iter_contexts(target_id):
        ego_index = context.all_agents.index(context.ego_id)
        agent_index = context.all_agents.index(context.key_agent)
        min_pet, pet_values = _pet_profile(context, ego_index, agent_index)
        row = metadata(context)
        row["type"] = context.path_relationship
        row["min_PET"] = min_pet
        row.update(pet_values)
        rows.append(row)
    write_csv(rows, OUTPUT_DIR / "safety" / "pet_results.csv")


def run(target_id: int | None = None) -> None:
    calculate_ttc_metrics(target_id)
    calculate_pet_metrics(target_id)


if __name__ == "__main__":
    run()
