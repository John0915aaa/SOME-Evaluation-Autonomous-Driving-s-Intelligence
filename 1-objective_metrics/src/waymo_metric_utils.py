"""Shared Waymo/InterHub loading utilities for objective metrics."""

from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator

import pandas as pd
from trajdata import UnifiedDataset

from src.objective_metrics_config import FOLDER_CACHE_MAP, INDEX_CSV
from src.utils.trajdata_utils import DataFrameCache, get_agent_states
from src.utils.visualize_utils import get_map_and_kdtrees

VEHICLE_LENGTH = 4.5
VEHICLE_WIDTH = 1.8
START_EXTENSION_SECONDS = 1.0
END_EXTENSION_SECONDS = 2.5


@dataclass(frozen=True)
class AgentState:
    x: float
    y: float
    z: float
    vx: float
    vy: float
    ax: float
    ay: float
    h: float


@dataclass
class InteractionContext:
    row: pd.Series
    index: int
    dataset_name: str
    folder: str
    raw_scene_id: int
    track_id: str
    interact_ids: list[str]
    key_agents: list[str]
    ego_id: str
    key_agent: str
    ego_type: str
    path_relationship: str
    start: int
    end: int
    start_time: float
    dt: float
    dataset: UnifiedDataset
    scene: object
    all_agents: list[str]
    all_timesteps: range
    states: object
    column_dict: dict


def read_index(index_csv: Path = INDEX_CSV) -> pd.DataFrame:
    return pd.read_csv(index_csv)


def get_cache_location(folder: str) -> str:
    cache_location = FOLDER_CACHE_MAP.get(folder, "")
    if not cache_location:
        raise RuntimeError(
            f"Cache path for {folder!r} is not configured. Set the corresponding "
            "WAYMO_CACHE_* environment variable before running metrics."
        )
    return cache_location


def get_dataset(dataset_name: str, cache_location: str) -> UnifiedDataset:
    try:
        return UnifiedDataset(
            desired_data=[dataset_name],
            standardize_data=False,
            rebuild_cache=False,
            rebuild_maps=False,
            centric="scene",
            verbose=True,
            cache_location=cache_location,
            num_workers=1,
            incl_vector_map=True,
            data_dirs={dataset_name: " "},
        )
    except NameError as exc:
        if "WaymoDataset" in str(exc):
            raise RuntimeError(
                "trajdata Waymo support is not available in this Python environment. "
                "Install the Waymo extras/dependencies used by trajdata, then rerun "
                f"with the lightweight cache at {cache_location}."
            ) from exc
        raise


def state_from_array(column_dict: dict, state) -> AgentState:
    return AgentState(
        x=state[column_dict["x"]],
        y=state[column_dict["y"]],
        z=state[column_dict["z"]],
        vx=state[column_dict["vx"]],
        vy=state[column_dict["vy"]],
        ax=state[column_dict["ax"]],
        ay=state[column_dict["ay"]],
        h=state[column_dict["heading"]],
    )


def speed(state: AgentState) -> float:
    return math.hypot(state.vx, state.vy)


def select_ego_and_key_agent(row: pd.Series, interact_ids: list[str]) -> tuple[str, str]:
    key_agents = str(row["key_agents"]).split(";")
    if row["ego_type"] == "AV_1":
        ego_id = next(agent for agent in interact_ids if "ego" in agent)
        key_agent = next(agent for agent in key_agents if agent != "ego")
    else:
        key_agent = key_agents[0]
        ego_id = key_agents[1]
    return ego_id, key_agent


def load_context(row: pd.Series) -> InteractionContext:
    dataset_name = row["dataset"]
    folder = row["folder"]
    raw_scene_id = int(row["scenario_idx"])
    start = int(row["start"])
    end = int(row["end"])
    track_id = row["track_id"]
    interact_ids = str(track_id).split(";")
    key_agents = str(row["key_agents"]).split(";")
    ego_id, key_agent = select_ego_and_key_agent(row, interact_ids)

    dataset = get_dataset(dataset_name, get_cache_location(folder))
    raw_id_to_scene_index = {
        scene.raw_data_idx: scene_index for scene_index, scene in enumerate(dataset.scenes())
    }
    scene = dataset.get_scene(raw_id_to_scene_index[raw_scene_id])
    dt = scene.dt
    agents = {agent.name: agent for agent in scene.agents}
    all_agents = list(agents.keys())

    first_timestep = min(agents[agent].first_timestep for agent in interact_ids)
    last_timestep = max(agents[agent].last_timestep for agent in interact_ids)
    interaction_start = max(first_timestep, int(start - START_EXTENSION_SECONDS / dt))
    interaction_end = min(last_timestep, int(end + END_EXTENSION_SECONDS / dt))
    all_timesteps = range(interaction_start, interaction_end)

    vector_map, lane_kd_tree = get_map_and_kdtrees(dataset, scene)
    scene_cache = DataFrameCache(cache_path=dataset.cache_path, scene=scene)
    column_dict = scene_cache.column_dict
    states, _ = get_agent_states(
        interact_ids,
        all_agents,
        vector_map,
        lane_kd_tree,
        scene_cache,
        scene,
        column_dict,
        all_timesteps,
    )

    return InteractionContext(
        row=row,
        index=int(row["index"]),
        dataset_name=dataset_name,
        folder=folder,
        raw_scene_id=raw_scene_id,
        track_id=track_id,
        interact_ids=interact_ids,
        key_agents=key_agents,
        ego_id=ego_id,
        key_agent=key_agent,
        ego_type=row["ego_type"],
        path_relationship=row.get("path_relationship", ""),
        start=start,
        end=end,
        start_time=float(row.get("start_time", 0)),
        dt=dt,
        dataset=dataset,
        scene=scene,
        all_agents=all_agents,
        all_timesteps=all_timesteps,
        states=states,
        column_dict=column_dict,
    )


def iter_contexts(target_id: int | None = None) -> Iterator[InteractionContext]:
    for _, row in read_index().iterrows():
        index = int(row["index"])
        if target_id is not None and index != target_id:
            continue
        yield load_context(row)


def metadata(context: InteractionContext) -> dict:
    return {
        "index": context.index,
        "dataset": context.dataset_name,
        "folder": context.folder,
        "scenario_idx": context.raw_scene_id,
        "track_id": context.track_id,
        "ego_id": context.ego_id,
        "timerange": context.all_timesteps,
    }


def write_csv(rows: list[dict], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(output_path, index=False)
    print(f"Saved {output_path}")
