"""Surrounding-traffic impact metric: background-vehicle speed change."""

from __future__ import annotations

import math

import numpy as np

from src.objective_metrics_config import OUTPUT_DIR
from src.waymo_metric_utils import iter_contexts, state_from_array, write_csv

MIN_VALID_SPEED = 0.5


def _mean_speed(context, vehicle_index: int, frame_indices: range) -> float:
    speeds = []
    for time_index in frame_indices:
        state = state_from_array(context.column_dict, context.states[vehicle_index, time_index, :])
        vehicle_speed = math.hypot(state.vx, state.vy)
        if vehicle_speed > MIN_VALID_SPEED:
            speeds.append(vehicle_speed)
    return sum(speeds) / len(speeds) if speeds else np.nan


def calculate_impact_metrics(target_id: int | None = None) -> None:
    rows = []
    for context in iter_contexts(target_id):
        vehicles = [agent for agent in context.scene.agents if agent.type == 1 and agent.name != context.ego_id]
        row = {
            "index": context.index,
            "start_time": context.start_time,
            "type": context.ego_type,
            "ego": context.ego_id,
        }
        frame_range = range(len(context.all_timesteps))
        split_frame = int((context.start_time - context.start * context.dt) / context.dt) if context.start_time > 0 else 0
        split_frame = max(0, min(split_frame, max(len(frame_range) - 1, 0)))

        for vehicle_number, vehicle in enumerate(vehicles, start=1):
            vehicle_index = context.all_agents.index(vehicle.name)
            if context.start_time == 0:
                ratio = 0.0
            else:
                mean_before = _mean_speed(context, vehicle_index, range(0, split_frame))
                mean_after = _mean_speed(context, vehicle_index, range(split_frame, len(context.all_timesteps)))
                ratio = np.nan if np.isnan(mean_before) or mean_before == 0 else max(0, (mean_before - mean_after) / mean_before)
            row[f"HV{vehicle_number}"] = ratio
        rows.append(row)
    write_csv(rows, OUTPUT_DIR / "impact" / "background_vehicle_speed_change.csv")


def run(target_id: int | None = None) -> None:
    calculate_impact_metrics(target_id)


if __name__ == "__main__":
    run()
