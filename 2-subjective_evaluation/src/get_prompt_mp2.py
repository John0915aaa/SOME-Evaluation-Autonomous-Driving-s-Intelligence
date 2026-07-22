"""Generate prompts for merging scenarios where HV cuts in ahead of AV."""

import math

import numpy as np
import pandas as pd

from dlmm_prompt_utils import (
    DataFrameCache,
    FOLDER_CACHE_MAP,
    calculate_ttc_with_one_agent_in_currenttime,
    get_agent_states,
    get_dataset,
    get_map_and_kdtrees,
    prompt_output_path,
    rad_to_chinese_direction,
    read_interaction_index,
)

starting_extension_time = 1
ending_extension_time = 2.5

PROMPT_CASE = "merging_hv_front"


def calculate_indicator(index_df: pd.DataFrame) -> pd.DataFrame:
    datas = []

    for rank, (idx, row) in enumerate(index_df.iterrows(), start=1):
        desired_data = row["dataset"]
        folder = row["folder"]
        raw_scene_id = int(row["scenario_idx"])
        start = int(row["start"])
        end = int(row["end"])
        track_id = row["track_id"]
        key_agents = row["key_agents"].split(";")
        interact_ids = track_id.split(";")
        index = row["index"]

        lane_change_type = row["lane_change_type"]
        lane_change_time_index = int(row["lane_change_time_index"])
        lane_change_end_time_index = int(row["lane_change_end_time_index"])
        type_flag_init = row["type_flag_init"]

        if lane_change_type != "HV_front":
            continue

        if lane_change_end_time_index == -1:
            continue

        if row["path_relationship"] != "MP":
            continue

        for agent in key_agents:
            if agent != "ego":
                key_agent = agent

        flag2 = False
        for agent in key_agents:
            if agent == "ego":
                flag2 = True
        if flag2 == False:
            continue

        for agent in key_agents:
            if agent != "ego":
                key_agent = agent

        ego_id = next((agent for agent in interact_ids if "ego" in agent), None)

        cache_location = FOLDER_CACHE_MAP.get(folder)
        if cache_location is None:
            print(f"Unknown folder: {folder}, skipping.")
            continue

        dataset = get_dataset(desired_data, cache_location)

        id_rawid = {desired_scene.raw_data_idx: idx for idx, desired_scene in enumerate(dataset.scenes())}
        desired_scene = dataset.get_scene(id_rawid[raw_scene_id])

        dt = desired_scene.dt
        agents = {agent.name: agent for agent in desired_scene.agents}
        all_agents = list(agents.keys())

        first, last = 99999, 0
        for agent in interact_ids:
            first = min(first, agents[agent].first_timestep)
            last = max(last, agents[agent].last_timestep)
        interaction_start = max(first, int(start - starting_extension_time / dt))
        interaction_end = min(last, int(end + ending_extension_time / dt))
        all_timesteps = range(interaction_start, interaction_end)

        vec_map, lane_kd_tree = get_map_and_kdtrees(dataset, desired_scene)
        scene_cache = DataFrameCache(cache_path=dataset.cache_path, scene=desired_scene)
        column_dict = scene_cache.column_dict

        agents_states, _ = get_agent_states(
            interact_ids, all_agents, vec_map, lane_kd_tree, scene_cache, desired_scene, column_dict, all_timesteps
        )

        ego_index = all_agents.index(ego_id)
        agent_index = all_agents.index(key_agent)

        ego_init_state = agents_states[ego_index, 0, :]
        AV_init_x, AV_init_y, AV_init_h = (
            ego_init_state[column_dict["x"]],
            ego_init_state[column_dict["y"]],
            ego_init_state[column_dict["heading"]],
        )
        AV_init_v = ego_init_state[column_dict["vx"]] * math.cos(AV_init_h) + ego_init_state[
            column_dict["vy"]
        ] * math.sin(AV_init_h)
        AV_init_a = ego_init_state[column_dict["ax"]] * math.cos(AV_init_h) + ego_init_state[
            column_dict["ay"]
        ] * math.sin(AV_init_h)
        AV_init_direction = rad_to_chinese_direction(AV_init_h)

        agent_init_state = agents_states[agent_index, 0, :]
        HV_init_x, HV_init_y, HV_init_h = (
            agent_init_state[column_dict["x"]],
            agent_init_state[column_dict["y"]],
            agent_init_state[column_dict["heading"]],
        )
        HV_init_v = agent_init_state[column_dict["vx"]] * math.cos(HV_init_h) + agent_init_state[
            column_dict["vy"]
        ] * math.sin(HV_init_h)
        HV_init_a = agent_init_state[column_dict["ax"]] * math.cos(HV_init_h) + agent_init_state[
            column_dict["ay"]
        ] * math.sin(HV_init_h)
        HV_init_direction = rad_to_chinese_direction(HV_init_h)

        dis_init = math.sqrt((AV_init_x - HV_init_x) ** 2 + (AV_init_y - HV_init_y) ** 2)

        prompt = f"""-----场景index: {index}. 当前交互场景为: {type_flag_init} 在该过程中, 我们需要关注HV变道过程中ego与HV行为交互中的智能度表现-----\n初始时刻, ego的位置为({AV_init_x:.3f}, {AV_init_y:.3f}), 以{AV_init_v:.3f} m/s的速度、{AV_init_a:.3f} m/s^2的加速度向{AV_init_direction}方向行驶; HV的位置为({HV_init_x:.3f}, {HV_init_y:.3f}), 以{HV_init_v:.3f} m/s的速度、{HV_init_a:.3f} m/s^2的加速度向{HV_init_direction}方向行驶; 两车相距(车辆中心点相对距离){dis_init:.2f} m."""

        ego_state = agents_states[ego_index, lane_change_time_index, :]
        AV_x, AV_y = ego_state[column_dict["x"]], ego_state[column_dict["y"]]
        AV_h = ego_state[column_dict["heading"]]
        AV_v = ego_state[column_dict["vx"]] * math.cos(AV_h) + ego_state[column_dict["vy"]] * math.sin(AV_h)
        AV_a = ego_state[column_dict["ax"]] * math.cos(AV_h) + ego_state[column_dict["ay"]] * math.sin(AV_h)
        AV_direction = rad_to_chinese_direction(AV_h)

        agent_state = agents_states[agent_index, lane_change_time_index, :]
        HV_x, HV_y = agent_state[column_dict["x"]], agent_state[column_dict["y"]]
        HV_h = agent_state[column_dict["heading"]]
        HV_v = agent_state[column_dict["vx"]] * math.cos(HV_h) + agent_state[column_dict["vy"]] * math.sin(HV_h)
        HV_a = agent_state[column_dict["ax"]] * math.cos(HV_h) + agent_state[column_dict["ay"]] * math.sin(HV_h)
        HV_direction = rad_to_chinese_direction(HV_h)

        dis = math.sqrt((AV_x - HV_x) ** 2 + (AV_y - HV_y) ** 2)

        ttc_value = calculate_ttc_with_one_agent_in_currenttime(
            column_dict, agents_states, ego_index, agent_index, lane_change_time_index
        )
        prompt += f"""\n第{lane_change_time_index}秒时, HV的部分车身跨过其当前所在车道, 预示其即将进行变道行为. 该时刻, HV位置为({HV_x:.3f}, {HV_y:.3f}), 以{HV_v:.3f} m/s的速度, 朝着{HV_direction}方向进行变道操作; ego当前时刻速度为{AV_v:.3f} m/s, 加速度为{AV_a:.3f} m/s^2, 朝着{AV_direction}方向行驶, 此时, 两车相距{dis:.2f} m, 根据AV与HV当前状态可以计算AV与HV的TTC为{ttc_value:.3f}"""

        if type_flag_init == "ego与HV在同一车道上, ego在HV前方直行, HV在ego后方向左侧车道变道.":
            prompt += f"""\n第{lane_change_end_time_index}秒时, HV全部车身均位于左侧车道, 预示着其完成变道操作. 该HV进行变道的持续时间为{lane_change_end_time_index - lane_change_time_index}秒. """
        elif type_flag_init == "ego与HV在同一车道上, ego在HV前方直行, HV在ego后方向右侧车道变道.":
            prompt += f"""\n第{lane_change_end_time_index}秒时, HV全部车身均位于右侧车道, 预示着其完成变道操作. 该HV进行变道的持续时间为{lane_change_end_time_index - lane_change_time_index}秒. """
        else:
            prompt += f"""\n第{lane_change_end_time_index}秒时, HV全部车身均位于ego所在车道, 预示着其完成变道操作. 该HV进行变道的持续时间为{lane_change_end_time_index - lane_change_time_index}秒. """

        AV_vs = []
        HV_vs = []
        AV_as = []
        HV_as = []
        AV_dire = []
        HV_dire = []
        ttc_values = []
        ttc_flag = False

        a_times_1 = []
        a_times_2 = []
        a_times_3 = []
        for time_index, timestamp in enumerate(all_timesteps):
            if time_index <= lane_change_time_index:
                continue

            if time_index > lane_change_end_time_index:
                break
            ego_state = agents_states[ego_index, time_index, :]
            AV_x, AV_y = ego_state[column_dict["x"]], ego_state[column_dict["y"]]
            AV_h = ego_state[column_dict["heading"]]
            AV_v = ego_state[column_dict["vx"]] * math.cos(AV_h) + ego_state[column_dict["vy"]] * math.sin(AV_h)
            AV_a = ego_state[column_dict["ax"]] * math.cos(AV_h) + ego_state[column_dict["ay"]] * math.sin(AV_h)
            AV_direction = rad_to_chinese_direction(AV_h)

            if (AV_a < -1.5 and AV_a > -3.0) or (AV_a > 1.5 and AV_a < 3.0):
                a_times_1.append(time_index)
            elif (AV_a < -3.0 and AV_a > -5.0) or (AV_a > 3.0 and AV_a < 5.0):
                a_times_2.append(time_index)
            elif (AV_a < -5.0) or (AV_a > 5.0):
                a_times_3.append(time_index)

            AV_vs.append(float(round(AV_v, 2)))
            AV_as.append(float(round(AV_a, 2)))
            AV_dire.append(AV_direction)

            agent_state = agents_states[agent_index, time_index, :]
            HV_x, HV_y = agent_state[column_dict["x"]], agent_state[column_dict["y"]]
            HV_h = agent_state[column_dict["heading"]]
            HV_v = agent_state[column_dict["vx"]] * math.cos(HV_h) + agent_state[column_dict["vy"]] * math.sin(HV_h)
            HV_a = agent_state[column_dict["ax"]] * math.cos(HV_h) + agent_state[column_dict["ay"]] * math.sin(HV_h)
            HV_direction = rad_to_chinese_direction(HV_h)

            HV_vs.append(float(round(HV_v, 2)))
            HV_as.append(float(round(HV_a, 2)))
            HV_dire.append(HV_direction)

        for time_index, timestamp in enumerate(all_timesteps):
            if time_index > lane_change_end_time_index:
                break
            ttc_value = calculate_ttc_with_one_agent_in_currenttime(
                column_dict, agents_states, ego_index, agent_index, time_index
            )
            if ttc_value != np.inf:
                ttc_flag = True
            ttc_values.append(float(round(ttc_value, 3)))

        if ttc_flag == False:
            ttc = np.inf
        else:
            ttc = min(ttc_values)

        prompt += f"""从HV进行变道行为开始之后:\n -ego的速度变化序列为: {AV_vs} m/s;\n -ego的加速度变化序列为: {AV_as} m/s^2;\n -ego的行驶方向变化序列为: {AV_dire};"""
        prompt += f"""\n从0时刻到HV变道结束这一过程中, ego与前车HV的TTC值(该TTC值的计算考虑了车身边缘碰撞)变化序列为{ttc_values} s"""

        i = index

        AV_v_lats = []
        AV_a_lats = []
        AV_yawrates = []
        for time_index, timestamp in enumerate(all_timesteps):
            if time_index > lane_change_end_time_index:
                break
            ego_state = agents_states[ego_index, time_index, :]
            AV_x, AV_y = ego_state[column_dict["x"]], ego_state[column_dict["y"]]
            AV_h = ego_state[column_dict["heading"]]
            AV_v_lat = -ego_state[column_dict["vx"]] * math.sin(AV_h) + ego_state[column_dict["vy"]] * math.cos(AV_h)
            AV_a_lat = -ego_state[column_dict["ax"]] * math.sin(AV_h) + ego_state[column_dict["ay"]] * math.cos(AV_h)
            AV_v_lat = float(round((AV_v_lat), 3))
            AV_a_lat = float(round((AV_a_lat), 3))

            AV_yawrate = (
                ego_state[column_dict["vx"]] * ego_state[column_dict["ay"]]
                - ego_state[column_dict["vy"]] * ego_state[column_dict["ax"]]
            ) / (ego_state[column_dict["vx"]] ** 2 + ego_state[column_dict["vy"]] ** 2)
            AV_yawrate = float(round((AV_yawrate), 3))

            AV_v_lats.append(AV_v_lat)
            AV_a_lats.append(AV_a_lat)
            AV_yawrates.append(AV_yawrate)

        prompt += f"""\n从0时刻到HV变道结束这一过程中, 为考虑AV的舒适性我们列出了:\n -ego的横向(垂直于车头方向)速度序列为: {AV_v_lats} m/s;\n -ego的横向加速度序列为: {AV_a_lats} m/s^2;\n -ego的横摆角速度为: {AV_yawrates} rad/s"""

        data = {"indexs": i, "prompt": prompt, "ttc_min": ttc}

        datas.append(data)

    df = pd.DataFrame(datas)

    return df


if __name__ == "__main__":
    full_index_df = read_interaction_index("dlmm_interaction_index.csv")
    case_index_df = full_index_df[full_index_df["prompt_case"] == PROMPT_CASE]
    calculate_indicator(case_index_df).to_csv(prompt_output_path("prompts.csv"), index=False)
