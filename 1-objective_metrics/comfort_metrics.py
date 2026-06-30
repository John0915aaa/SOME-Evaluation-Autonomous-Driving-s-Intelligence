"""Comfort metrics: acceleration, velocity, and yaw-rate profiles."""

from __future__ import annotations

import math

from src.objective_metrics_config import OUTPUT_DIR
from src.waymo_metric_utils import InteractionContext, iter_contexts, metadata, state_from_array, write_csv


def _comfort_state(context: InteractionContext, ego_index: int, time_index: int) -> tuple[float, float, float, float, float]:
    state = state_from_array(context.column_dict, context.states[ego_index, time_index, :])
    longitudinal_acc = state.ax * math.cos(state.h) + state.ay * math.sin(state.h)
    lateral_acc = -state.ax * math.sin(state.h) + state.ay * math.cos(state.h)
    longitudinal_speed = state.vx * math.cos(state.h) + state.vy * math.sin(state.h)
    jerk = -state.vx * math.sin(state.h) + state.vy * math.cos(state.h)
    yaw_rate = lateral_acc / longitudinal_speed if longitudinal_speed >= 0.01 else 0
    return longitudinal_acc, lateral_acc, longitudinal_speed, jerk, yaw_rate


def calculate_comfort_metrics(target_id: int | None = None) -> None:
    longitudinal_acc_rows = []
    lateral_acc_rows = []
    longitudinal_speed_rows = []
    jerk_rows = []
    yaw_rate_rows = []

    for context in iter_contexts(target_id):
        ego_index = context.all_agents.index(context.ego_id)
        base_row = metadata(context)
        longitudinal_acc = dict(base_row)
        lateral_acc = dict(base_row)
        longitudinal_speed = dict(base_row)
        jerk = dict(base_row)
        yaw_rate = dict(base_row)

        for time_index, timestamp in enumerate(context.all_timesteps):
            ap, al, vp, vl, yr = _comfort_state(context, ego_index, time_index)
            longitudinal_acc[f"a_p, t={timestamp}"] = ap
            lateral_acc[f"a_l, t={timestamp}"] = al
            longitudinal_speed[f"v_p, t={timestamp}"] = vp
            jerk[f"jerk, t={timestamp}"] = vl
            yaw_rate[f"yaw_rate, t={timestamp}"] = yr

        longitudinal_acc_rows.append(longitudinal_acc)
        lateral_acc_rows.append(lateral_acc)
        longitudinal_speed_rows.append(longitudinal_speed)
        jerk_rows.append(jerk)
        yaw_rate_rows.append(yaw_rate)

    output_dir = OUTPUT_DIR / "comfort"
    write_csv(longitudinal_acc_rows, output_dir / "longitudinal_acceleration.csv")
    write_csv(lateral_acc_rows, output_dir / "lateral_acceleration.csv")
    write_csv(longitudinal_speed_rows, output_dir / "longitudinal_speed.csv")
    write_csv(jerk_rows, output_dir / "jerk.csv")
    write_csv(yaw_rate_rows, output_dir / "yaw_rate.csv")


def run(target_id: int | None = None) -> None:
    calculate_comfort_metrics(target_id)


if __name__ == "__main__":
    run()
