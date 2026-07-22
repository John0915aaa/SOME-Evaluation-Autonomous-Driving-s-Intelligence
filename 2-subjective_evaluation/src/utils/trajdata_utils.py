import numpy as np
from pathlib import Path

from trajdata.data_structures import Scene
from trajdata.caching import EnvCache
from trajdata import VectorMap
from trajdata.data_structures import AgentType
from trajdata.caching.df_cache import DataFrameCache


def load_random_scene(cache_path: Path, env_name: str, scene_dt: float) -> Scene:
    env_cache = EnvCache(cache_path)
    scenes_list = env_cache.load_env_scenes_list(env_name)
    random_scene_name = scenes_list[np.random.randint(0, len(scenes_list))].name
    print(scenes_list)
    print(random_scene_name)

    return env_cache.load_scene(env_name, random_scene_name, scene_dt)


def print_lane_connections(vector_map: VectorMap, lane_id: str):
    # Get the specific lane object
    lane = vector_map.get_road_lane(lane_id)

    # Print upstream lanes
    print("Previous Lanes:")
    for prev_lane_id in lane.prev_lanes:
        print(f"  - {prev_lane_id}")

    # Print downstream lanes
    print("Next Lanes:")
    for next_lane_id in lane.next_lanes:
        print(f"  - {next_lane_id}")

    # Print left adjacent lanes
    print("Adjacent Lanes Left:")
    for left_lane_id in lane.adj_lanes_left:
        print(f"  - {left_lane_id}")

    # Print right adjacent lanes
    print("Adjacent Lanes Right:")
    for right_lane_id in lane.adj_lanes_right:
        print(f"  - {right_lane_id}")


def current_lane_id(
    lane_kd_tree, query_point, distance_threshold=3, heading_threshold=20
):  # m, angle   # Use appropriate distance and heading thresholds for querying
    # distance_threshold = 3(m) 定义了查询点与道路之间的最大允许距离。只有那些与查询点距离小于该阈值的道路会被认为是相关的
    # heading_threshold = 20(度) 表示车辆的朝向与道路朝向之间的最大允许角度偏差。朝向的单位是弧度，因此该值会在函数内部被转换为弧度。这个阈值用于限制只有与查询点朝向相似的道路会被选中
    heading_threshold = np.pi / heading_threshold  # Heading threshold in radians
    # print(f"heading_threshold = {heading_threshold}")

    # Get possible lane indices
    lane_indices = lane_kd_tree.current_lane_inds(
        xyzh=query_point, distance_threshold=distance_threshold, heading_threshold=heading_threshold
    )
    return lane_indices


# 返回所有符合条件的道路索引

# 使用get_agent_states函数来提取车辆的position信息


def get_agent_states(interact_ids, all_agents, vec_map, lane_kd_tree, sc, desired_scene, column_dict, all_timesteps):
    """
    Retrieves the states and lane information for each agent in the given scene.

    Args:
        interact_ids (list): List of agent IDs to focus on for interaction analysis.
        all_agents (list): List of all agents present in the scene.     agent列表（所有的agent）
        vec_map (VectorMap): The vector map of the environment.         场景的地图信息
        lane_kd_tree: KD-tree for lanes used for proximity searches.
        sc (DataFrameCache): Cache object for accessing scene data.
        desired_scene: Scene object containing details about the scene.
        column_dict (dict): Dictionary mapping column names to their indices in raw state data.
        all_timesteps (list): List of all timesteps available in the scene.

    Returns:
        tuple: A tuple containing:
            - agent_states (np.ndarray): An array with state information for each agent across timesteps.
            - agent_lane_ids (dict): A dictionary with lane IDs assigned to each agent for each timestep.
        该函数返回一个tuple, 包括：
            -车辆的状态信息：一个包含车辆状态信息的Array
            -车辆道路信息：一个包含道路ID的字典
    """
    # Initialize the states array for all agents (dimensions: num_agents x num_timesteps x 8 state variables)
    # 初始化一个所有agents状态的3维-矩阵：维度是  agents个数 x timesteps数量 x 8个状态变量
    agent_states = np.zeros((len(all_agents), desired_scene.length_timesteps, 8))

    # Initialize a dictionary to hold lane IDs for each agent at each timestep
    # 初始化一个字典 agent_lane_ids，用于存储每个 agent 在每个时间步（timestep）所在的车道（lane）信息
    agent_lane_ids = {agent.name: [0] * len(all_timesteps) for agent in desired_scene.agents}

    # Iterate through each agent in the scene
    # 遍历这个场景中的所有agent
    # 这里的agents是场景中所有的agents，无论有没有交互
    # print(f"desired_scene.agents = {desired_scene.agents}")
    for agent in desired_scene.agents:
        current_lane = None  # 当前道路

        # Get indices for state variables (x, y, z, heading) from column_dict
        x_index = column_dict["x"]  # x_index = 0
        y_index = column_dict["y"]  # y_index = 1
        z_index = column_dict["z"]  # z_index = 2
        heading_index = column_dict["heading"]
        # 这是所有状态在column_dict中的索引，x:0 y:1 z:2 ...
        num = 0
        # Iterate through each timestep for the agent
        # 遍历当前agent的所有时间范围
        for t in range(agent.first_timestep, agent.last_timestep + 1):
            # print(f"agent = {agent}")
            # print(f"t = {t}")
            # Retrieve the raw state of the agent at the given timestep
            # 提取当前t时刻下agent的state：raw_state
            raw_state = sc.get_raw_state(agent_id=agent.name, scene_ts=t)

            # 一个包含车辆当前位置和朝向信息的 NumPy 数组。这个 query_point 用于后续的计算，通常是用来在 KD 树中查找与车辆位置最接近的道路（lane）信息
            query_point = np.array(
                [raw_state[x_index], raw_state[y_index], raw_state[z_index], raw_state[heading_index]]
            )
            # query_point = [agent.x.t, agent.y.t agent.z.t agent.h.t]
            # print(f"query_point = {query_point}")

            # Find lane indices using KD-tree
            # 根据当前时刻下agent的query_point，在lane_kd_tree下找到当前agent所在道路的id：lane_indices
            lane_indices = current_lane_id(lane_kd_tree, query_point)  # KD-tree for lanes used for proximity searches.
            # 返回所有符合条件的道路索引，这个道路索引是当前agent可能所在的道路的索引
            # print(f"lane_indices = {lane_indices}")
            # 这个索引可能不止一个

            lane_indices = [vec_map.lanes[i].id for i in lane_indices]
            # 将 lane_indices 中的每个元素映射成实际的道路 ID
            # print(f"lane_indices_id = {lane_indices}")

            # Determine the most appropriate lane for the agent
            # 获得与当前agent最接近的道路（就是当前道路）

            # 按条件获取当前agent所在道路ID

            # 如果当前不止一个道路，选择最合适的道路
            if len(lane_indices) > 1:
                query_point = np.array([raw_state[x_index], raw_state[y_index], raw_state[z_index]])
                closest = lane_kd_tree.closest_polyline_ind(query_point)
                closest_lane_id = vec_map.lanes[int(closest)].id
                # 筛选一个距离当前query_point最近的道路id
                if closest_lane_id in lane_indices:
                    chosen_lane = closest_lane_id
                else:
                    chosen_lane = next((lan for lan in lane_indices if lan == current_lane), lane_indices[0])
                    # 查找 lane_indices 中是否有与 current_lane 相同的车道。如果有，选择 current_lane，
                    # 否则就选择 lane_indices 列表中的第一个车道作为 chosen_lane

            # 如果当前没有发现道路，选择一条最近的道路
            elif len(lane_indices) == 0:
                # If no lanes are found, choose the closest lane
                query_point = np.array([raw_state[x_index], raw_state[y_index], raw_state[z_index]])
                closest = lane_kd_tree.closest_polyline_ind(query_point)
                chosen_lane = vec_map.lanes[int(closest)].id
                num = num + 1
            # 如果当前只有一个道路，那就是目标道路
            else:
                # If only one lane is found, select it
                chosen_lane = lane_indices[0]

            current_lane = chosen_lane
            # 这里，current_lane是一个道路id，表示当前agent所在道路的id

            # try的工作原理：
            # 本循环体：for t in range(agent.first_timestep, agent.last_timestep + 1):
            #   会遍历所有agent的起始时间和结束时间
            #   但是all_timesteps只包含了那些存在交互的agent运行的起始-结束时间范围
            #   因此，当那些不在交互时间范围内的agent尝试将其state写入agent_states时，就会被跳过，因为其运行时间并不在这个all_timesteps时间范围内
            #   但是这样只能过滤掉一部分的agent，并不能过滤所有不在交互范围内的agent
            try:
                # Update agent states with raw state data
                agent_index = all_agents.index(agent.name)  # 选择当前agent的index
                timestep_index = all_timesteps.index(t)  # 选择当前时间的index
                agent_states[agent_index, timestep_index, :] = raw_state
                # print(f"all_agents = {all_agents}")
                # print(f"agent_name = {agent.name}")
                # print(f"timestep_index = {timestep_index}")
                # 当前agent的状态存储到：当前agent_index, 当前timestep_index下，time_step是时间索引，而非真正的时间值
            except Exception as e:
                # print(f"Error processing agent {agent.name} at timestep {t}: {e}")
                continue

            # Update lane ID for the agent at the given timestep
            agent_lane_ids[agent.name][timestep_index] = chosen_lane
            # print(f"agent.name = {agent.name}, timestep_index = {timestep_index}, chosen_lane = {chosen_lane}")
            # 选出当前timestep_index下，对应agent的道路id，并存入字典agent_lane_ids当中

    return agent_states, agent_lane_ids


# 后续可以根据ego的id来直接从agent_lane_ids字典中读取all_timesteps中的道路ids
