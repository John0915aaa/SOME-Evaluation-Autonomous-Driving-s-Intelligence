"""Shared utilities for DLMM prompt generation."""

from __future__ import annotations

import math
import os
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from shapely.geometry import Point
from shapely.ops import nearest_points
from trajdata import UnifiedDataset

try:
    from .TwoDimTTC import TTC
    from .utils.trajdata_utils import DataFrameCache, get_agent_states
    from .utils.visualize_utils import get_map_and_kdtrees, process_tracks_single
except ImportError:
    from TwoDimTTC import TTC
    from utils.trajdata_utils import DataFrameCache, get_agent_states
    from utils.visualize_utils import get_map_and_kdtrees, process_tracks_single

__all__ = [
    "AgentState",
    "DataFrameCache",
    "FOLDER_CACHE_MAP",
    "calculate_ttc_with_one_agent_in_currenttime",
    "considertime",
    "get_agent_state",
    "get_agent_states",
    "get_collision_point_with_dis_and_speed",
    "get_dataset",
    "get_map_and_kdtrees",
    "length",
    "load_csv_by_index",
    "load_csv_with_multiple_hvs",
    "process_tracks_single",
    "prompt_output_path",
    "rad_to_chinese_direction",
    "read_interaction_index",
    "width",
]

ROOT_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT_DIR.parent / "data"
PROMPT_DIR = ROOT_DIR / "prompt"
LOCAL_LIGHTWEIGHT_CACHE_ROOT = DATA_DIR / "waymo_lightweight_cache"

considertime = 5
length = 4.5
width = 1.8

def _cache_path(folder: str, env_var: str) -> str:
    specific_path = os.environ.get(env_var)
    if specific_path:
        return specific_path

    cache_root = os.environ.get("WAYMO_CACHE_ROOT")
    if cache_root:
        return str(Path(cache_root).expanduser() / folder)

    return str(LOCAL_LIGHTWEIGHT_CACHE_ROOT / folder)


FOLDER_CACHE_MAP = {
    "waymo_0-299": _cache_path("waymo_0-299", "WAYMO_CACHE_0_299"),
    "waymo_300-499": _cache_path("waymo_300-499", "WAYMO_CACHE_300_499"),
    "waymo_500-799": _cache_path("waymo_500-799", "WAYMO_CACHE_500_799"),
    "waymo_800-999": _cache_path("waymo_800-999", "WAYMO_CACHE_800_999"),
}


@dataclass
class AgentState:
    x: float
    y: float
    z: float
    vx: float
    vy: float
    ax: float
    ay: float
    h: float


def read_interaction_index(file_name: str) -> pd.DataFrame:
    return pd.read_csv(DATA_DIR / file_name)


def prompt_output_path(file_name: str) -> Path:
    PROMPT_DIR.mkdir(parents=True, exist_ok=True)
    return PROMPT_DIR / file_name


def get_agent_state(column_dict: dict, agent_state) -> AgentState:
    return AgentState(
        x=agent_state[column_dict["x"]],
        y=agent_state[column_dict["y"]],
        z=agent_state[column_dict["z"]],
        vx=agent_state[column_dict["vx"]],
        vy=agent_state[column_dict["vy"]],
        ax=agent_state[column_dict["ax"]],
        ay=agent_state[column_dict["ay"]],
        h=agent_state[column_dict["heading"]],
    )


def get_dataset(desired_data: str, cache_location: str) -> UnifiedDataset:
    try:
        return UnifiedDataset(
            desired_data=[desired_data],
            standardize_data=False,
            rebuild_cache=False,
            rebuild_maps=False,
            centric="scene",
            verbose=True,
            cache_location=cache_location,
            num_workers=os.cpu_count(),
            incl_vector_map=True,
            data_dirs={desired_data: " "},
        )
    except NameError as exc:
        if "WaymoDataset" in str(exc):
            raise RuntimeError(
                "trajdata Waymo support is not available in this Python environment. "
                "Install the Waymo extras/dependencies used by trajdata, then rerun "
                f"with the lightweight cache at {cache_location}."
            ) from exc
        raise


def calculate_ttc_with_one_agent_in_currenttime(column_dict, agents_states, ego_index, agent_index, time_index):
    ego = get_agent_state(column_dict, agents_states[ego_index, time_index, :])
    agent = get_agent_state(column_dict, agents_states[agent_index, time_index, :])
    data = {
        "x_i": ego.x,
        "y_i": ego.y,
        "vx_i": ego.vx,
        "vy_i": ego.vy,
        "hx_i": math.cos(ego.h),
        "hy_i": math.sin(ego.h),
        "length_i": length,
        "width_i": width,
        "x_j": agent.x,
        "y_j": agent.y,
        "vx_j": agent.vx,
        "vy_j": agent.vy,
        "hx_j": math.cos(agent.h),
        "hy_j": math.sin(agent.h),
        "length_j": length,
        "width_j": width,
    }
    return TTC(pd.DataFrame(data, index=[0]), "values")


def get_collision_point_with_dis_and_speed(column_dict, agents_states, s_agent_index, l_agent_index, time_index):
    position_index = [column_dict["x"], column_dict["y"]]
    velocity_index = [column_dict["vx"], column_dict["vy"]]
    timerange = considertime / 0.1
    s_track = process_tracks_single(
        considertime, agents_states, s_agent_index, time_index, timerange, position_index, velocity_index
    )
    l_track = process_tracks_single(
        considertime, agents_states, l_agent_index, time_index, timerange, position_index, velocity_index
    )
    if s_track is None or l_track is None:
        return None, None, None, None, None, None, None

    s_line, s_speed = s_track["line"], s_track["velocity"]
    l_line, l_speed = l_track["line"], l_track["velocity"]
    intersection = s_line.intersection(l_line)
    if intersection.is_empty:
        return None, None, None, None, None, None, None

    s_state = get_agent_state(column_dict, agents_states[s_agent_index, time_index, :])
    _, collision_point = nearest_points(Point(s_state.x, s_state.y), intersection)
    dis_s = s_line.project(collision_point)
    dis_l = l_line.project(collision_point)

    future_s = get_agent_state(column_dict, agents_states[s_agent_index, time_index + 2, :])
    future_l = get_agent_state(column_dict, agents_states[l_agent_index, time_index + 2, :])
    delta_s = s_line.project(Point(future_s.x, future_s.y))
    delta_l = l_line.project(Point(future_l.x, future_l.y))
    return collision_point, dis_s, dis_l, s_speed, l_speed, delta_s, delta_l


def rad_to_chinese_direction(rad: float) -> str | None:
    deg = np.degrees(rad) % 360

    def check_exact(target: float) -> bool:
        return abs(deg - target) <= 1 or abs(deg - target + 360) <= 1

    if check_exact(0) or check_exact(360):
        return "正东"
    if check_exact(90):
        return "正北"
    if check_exact(180):
        return "正西"
    if check_exact(270):
        return "正南"
    if 0 < deg < 90:
        return f"东偏北{round(deg)}度"
    if 90 < deg < 180:
        return f"北偏西{round(deg - 90)}度"
    if 180 < deg < 270:
        return f"西偏南{round(deg - 180)}度"
    if 270 < deg < 360:
        return f"南偏东{round(deg - 270)}度"
    return None


def load_csv_by_index(index, data_dir="base_data/waymo_train"):
    index_str = str(index) + "_"
    for filename in os.listdir(data_dir):
        if filename.startswith(index_str) and filename.endswith(".csv"):
            return pd.read_csv(os.path.join(data_dir, filename)), filename
    return None, None


def load_csv_with_multiple_hvs(filepath):
    import re

    df = pd.read_csv(filepath)
    hv_columns = [col for col in df.columns if re.match(r"HV\d+_x", col)]
    hv_ids = sorted(set(col.split("_")[0] for col in hv_columns))
    return df, hv_ids
