"""Efficiency metrics: task-time ratio and background-vehicle mean speed."""

from __future__ import annotations

import math

import numpy as np

from src.objective_metrics_config import OUTPUT_DIR
from src.waymo_metric_utils import InteractionContext, iter_contexts, speed, state_from_array, write_csv


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
    vehicles = [agent for agent in context.scene.agents if agent.type == 1 and agent.name != "ego"]
    for vehicle in vehicles:
        vehicle_index = context.all_agents.index(vehicle.name)
        for time_index, _ in enumerate(context.all_timesteps):
            state = state_from_array(context.column_dict, context.states[vehicle_index, time_index, :])
            speeds.append(speed(state))
    return speeds


def calculate_task_time_metrics(target_id: int | None = None) -> None:
    rows = []
    for context in iter_contexts(target_id):
        ego_index = context.all_agents.index(context.ego_id)
        path_length = _ego_path_length(context, ego_index)
        background_speeds = _background_vehicle_speeds(context)
        max_speed = max(background_speeds) if background_speeds else np.nan
        task_time = len(context.all_timesteps) + 1
        ratio = np.nan if not max_speed or np.isnan(max_speed) else min(path_length / (task_time * max_speed * context.dt), 1)
        rows.append({
            "index": context.index,
            "task_time": task_time,
            "v_max": max_speed,
            "task_path": path_length,
            "ratio": ratio,
        })
    write_csv(rows, OUTPUT_DIR / "efficiency" / "task_time.csv")


def calculate_hv_mean_speed_metrics(target_id: int | None = None) -> None:
    rows = []
    for context in iter_contexts(target_id):
        mean_speeds = []
        vehicles = [agent for agent in context.scene.agents if agent.type == 1 and agent.name != "ego"]
        for vehicle in vehicles:
            vehicle_index = context.all_agents.index(vehicle.name)
            speeds = []
            for time_index, _ in enumerate(context.all_timesteps):
                state = state_from_array(context.column_dict, context.states[vehicle_index, time_index, :])
                vehicle_speed = speed(state)
                if vehicle_speed <= 0.5:
                    speeds = []
                    break
                speeds.append(vehicle_speed)
            if speeds:
                mean_speeds.append(sum(speeds) / len(speeds))
        rows.append({
            "index": context.index,
            "min_mean_speed": min(mean_speeds) if mean_speeds else np.nan,
            "max_mean_speed": max(mean_speeds) if mean_speeds else np.nan,
        })
    write_csv(rows, OUTPUT_DIR / "efficiency" / "background_vehicle_mean_speed.csv")


def run(target_id: int | None = None) -> None:
    calculate_task_time_metrics(target_id)
    calculate_hv_mean_speed_metrics(target_id)


if __name__ == "__main__":
    run()
