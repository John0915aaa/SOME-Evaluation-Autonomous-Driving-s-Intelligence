"""Surrounding-traffic impact metric based on background-vehicle speed change."""

from __future__ import annotations

import math

import numpy as np

from src.objective_metrics_config import OUTPUT_DIR
from src.waymo_metric_utils import (
    InteractionContext,
    iter_contexts,
    state_from_array,
    write_csv,
)

MIN_VALID_SPEED = 0.5


def _mean_speed(
    context: InteractionContext,
    vehicle_index: int,
    frame_indices: range,
) -> float:
    speeds = []

    for time_index in frame_indices:
        state = state_from_array(
            context.column_dict,
            context.states[vehicle_index, time_index, :],
        )
        vehicle_speed = math.hypot(state.vx, state.vy)

        if np.isfinite(vehicle_speed) and vehicle_speed > MIN_VALID_SPEED:
            speeds.append(vehicle_speed)

    return float(np.mean(speeds)) if speeds else np.nan


def _aggregate_impact(speed_change_ratios: list[float]) -> float:
    """Return Impact = 1 - mean(max((v_before - v_after) / v_before, 0))."""
    valid_ratios = [
        ratio for ratio in speed_change_ratios
        if np.isfinite(ratio)
    ]

    # 没有背景车或没有任何有效背景车时，认为未影响周围交通。
    if not valid_ratios:
        return 1.0

    impact = 1.0 - float(np.mean(valid_ratios))
    return float(np.clip(impact, 0.0, 1.0))


def calculate_impact_for_context(
    context: InteractionContext,
) -> dict:
    vehicles = [
        agent
        for agent in context.scene.agents
        if agent.type == 1 and agent.name != context.ego_id
    ]

    if not vehicles:
        return {
            "index": context.index,
            "start_time": context.start_time,
            "type": context.ego_type,
            "ego": context.ego_id,
            "total_background_vehicles": 0,
            "n_background_vehicles": 0,
            "impact": 1.0,
        }

    sample_count = len(context.all_timesteps)

    if sample_count == 0:
        return {
            "index": context.index,
            "start_time": context.start_time,
            "type": context.ego_type,
            "ego": context.ego_id,
            "total_background_vehicles": len(vehicles),
            "n_background_vehicles": 0,
            "impact": 1.0,
        }

    split_frame = (
        int(
            (context.start_time - context.start * context.dt)
            / context.dt
        )
        if context.start_time > 0
        else 0
    )
    split_frame = max(0, min(split_frame, sample_count - 1))

    speed_change_ratios = []

    for vehicle in vehicles:
        vehicle_index = context.all_agents.index(vehicle.name)

        if context.start_time == 0:
            speed_change_ratios.append(0.0)
            continue

        mean_before = _mean_speed(
            context,
            vehicle_index,
            range(0, split_frame),
        )
        mean_after = _mean_speed(
            context,
            vehicle_index,
            range(split_frame, sample_count),
        )

        if (
            not np.isfinite(mean_before)
            or not np.isfinite(mean_after)
            or mean_before <= 0
        ):
            continue

        ratio = (mean_before - mean_after) / mean_before
        ratio = float(np.clip(ratio, 0.0, 1.0))
        speed_change_ratios.append(ratio)

    impact = _aggregate_impact(speed_change_ratios)

    # 最终保险，确保输出不出现 NaN 或 Inf。
    if not np.isfinite(impact):
        impact = 1.0

    return {
        "index": context.index,
        "start_time": context.start_time,
        "type": context.ego_type,
        "ego": context.ego_id,
        "total_background_vehicles": len(vehicles),
        "n_background_vehicles": len(speed_change_ratios),
        "impact": impact,
    }


def calculate_impact_metrics(
    target_id: int | None = None,
) -> None:
    rows = [
        calculate_impact_for_context(context)
        for context in iter_contexts(target_id)
    ]
    write_csv(rows, OUTPUT_DIR / "impact" / "impact.csv")


def run(target_id: int | None = None) -> None:
    calculate_impact_metrics(target_id)


if __name__ == "__main__":
    run()