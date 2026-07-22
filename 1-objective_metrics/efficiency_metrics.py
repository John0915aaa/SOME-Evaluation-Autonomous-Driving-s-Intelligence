"""Efficiency metrics: task-time ratio and average-delay metric."""

from __future__ import annotations

import math

import numpy as np

from src.objective_metrics_config import OUTPUT_DIR
from src.waymo_metric_utils import InteractionContext, iter_contexts, speed, state_from_array, write_csv

MIN_VALID_SURROUNDING_SPEED = 0.5


def _ego_path_length(context: InteractionContext, ego_index: int) -> float:
    first_state = context.states[ego_index, 0, :]
    previous_x = first_state[context.column_dict["x"]]
    previous_y = first_state[context.column_dict["y"]]
    distance = 0.0
    for time_index, _ in enumerate(context.all_timesteps):
        state = context.states[ego_index, time_index, :]
        current_x = state[context.column_dict["x"]]
        current_y = state[context.column_dict["y"]]
        distance += math.hypot(current_x - previous_x, current_y - previous_y)
        previous_x, previous_y = current_x, current_y
    return distance


def _background_vehicle_speeds(context: InteractionContext) -> list[float]:
    speeds = []
    vehicles = [
        agent
        for agent in context.scene.agents
        if agent.type == 1 and agent.name != context.ego_id
    ]
    for vehicle in vehicles:
        vehicle_index = context.all_agents.index(vehicle.name)
        for time_index, _ in enumerate(context.all_timesteps):
            state = state_from_array(context.column_dict, context.states[vehicle_index, time_index, :])
            speeds.append(speed(state))
    return speeds


def calculate_task_time_for_context(context: InteractionContext) -> dict:
    ego_index = context.all_agents.index(context.ego_id)
    path_length = _ego_path_length(context, ego_index)
    background_speeds = _background_vehicle_speeds(context)
    max_speed = max(background_speeds) if background_speeds else np.nan
    task_time_frames = len(context.all_timesteps) + 1
    task_time_score = (
        np.nan
        if not max_speed or np.isnan(max_speed)
        else min(path_length / (task_time_frames * max_speed * context.dt), 1)
    )
    return {
        "index": context.index,
        "task_time_frames": task_time_frames,
        "task_time_seconds": task_time_frames * context.dt,
        "v_max": max_speed,
        "task_path": path_length,
        "task_time": task_time_score,
    }


def calculate_task_time_metrics(target_id: int | None = None) -> None:
    rows = [calculate_task_time_for_context(context) for context in iter_contexts(target_id)]
    write_csv(rows, OUTPUT_DIR / "efficiency" / "task_time.csv")


def _agent_mean_speed(
    context: InteractionContext,
    agent_index: int,
    *,
    require_continuously_moving: bool = False,
) -> float:
    speeds = []
    for time_index, _ in enumerate(context.all_timesteps):
        state = state_from_array(context.column_dict, context.states[agent_index, time_index, :])
        agent_speed = speed(state)
        if not np.isfinite(agent_speed):
            return np.nan
        if require_continuously_moving and agent_speed <= MIN_VALID_SURROUNDING_SPEED:
            return np.nan
        speeds.append(agent_speed)
    return float(np.mean(speeds)) if speeds else np.nan


def _normalized_avg_delay(ego_mean_speed: float, min_mean_speed: float, max_mean_speed: float) -> float:
    """Calculate aD = clip((v_ego - v_min) / (v_max - v_min), 0, 1)."""
    if any(np.isnan(value) for value in (ego_mean_speed, min_mean_speed, max_mean_speed)):
        return np.nan
    if math.isclose(max_mean_speed, min_mean_speed):
        # Min-max normalization is undefined when all surrounding vehicles have
        # the same mean speed. Preserve the limiting 0/1 interpretation.
        return 1.0 if ego_mean_speed > max_mean_speed else 0.0
    return float(np.clip(
        (ego_mean_speed - min_mean_speed) / (max_mean_speed - min_mean_speed),
        0.0,
        1.0,
    ))


def calculate_avg_delay_for_context(context: InteractionContext) -> dict:
    ego_index = context.all_agents.index(context.ego_id)
    ego_mean_speed = _agent_mean_speed(context, ego_index)

    mean_speeds = []
    vehicles = [
        agent
        for agent in context.scene.agents
        if agent.type == 1 and agent.name != context.ego_id
    ]

    for vehicle in vehicles:
        vehicle_index = context.all_agents.index(vehicle.name)
        mean_speed = _agent_mean_speed(
            context,
            vehicle_index,
            require_continuously_moving=True,
        )

        if np.isfinite(mean_speed):
            mean_speeds.append(mean_speed)

    if mean_speeds:
        min_mean_speed = min(mean_speeds)
        max_mean_speed = max(mean_speeds)
        avg_delay = _normalized_avg_delay(
            ego_mean_speed,
            min_mean_speed,
            max_mean_speed,
        )
    else:
        min_mean_speed = np.nan
        max_mean_speed = np.nan
        avg_delay = 1.0

    return {
        "index": context.index,
        "ego_mean_speed": ego_mean_speed,
        "min_mean_speed": min_mean_speed,
        "max_mean_speed": max_mean_speed,
        "avg_delay": avg_delay,
    }

def calculate_avg_delay_metrics(target_id: int | None = None) -> None:
    rows = [calculate_avg_delay_for_context(context) for context in iter_contexts(target_id)]
    write_csv(rows, OUTPUT_DIR / "efficiency" / "avg_delay.csv")


def run(target_id: int | None = None) -> None:
    calculate_task_time_metrics(target_id)
    calculate_avg_delay_metrics(target_id)


if __name__ == "__main__":
    run()
