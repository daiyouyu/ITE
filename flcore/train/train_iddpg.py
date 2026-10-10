# -*- coding: utf-8 -*-
import time
import numpy as np
import os
import random
import torch
from datetime import datetime as dt
from flcore.train.train_common import (
    default_presets, load_series_split, build_envs,
    infer_dims, list_by_agents, flatten_obs, flatten_actions
)
from flcore.algorithm.IDDPG import IDDPG
from flcore.Env.multi_env import MultiBatteryCoordinator
from flcore.utils.print_epreward import format_episode_info


def _random_state(env):
    """记录分支重放所需的随机状态，包括探索动作空间与环境内部生成器。"""
    return {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch": torch.get_rng_state(),
        "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
        "spaces": {a: env.action_spaces[a].np_random.bit_generator.state for a in env.agents},
        "environments": {a: env.envs[a]._rng.bit_generator.state for a in env.agents},
    }


def _restore_random_state(state, env):
    """恢复共同阶段终点的随机状态，保证两分支从相同轨迹起步。"""
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch"])
    if state["cuda"] is not None:
        torch.cuda.set_rng_state_all(state["cuda"])
    for agent in env.agents:
        env.action_spaces[agent].np_random.bit_generator.state = state["spaces"][agent]
        env.envs[agent]._rng.bit_generator.state = state["environments"][agent]


def train_iddpg_lost_contact(
    episodes=1000, train=365, fed_method="DSFA", branch_episode=200,
    lost_agent="agent_0", disconnect_schedule=((200, 400), (600, 800)),
):
    """先完成共同训练，再从同一检查点运行正常与断联分支。

    Args:
        episodes: 总 episode 数，必须大于共同阶段终点。
        train: 训练数据天数。
        fed_method: DSFA 或 FedAvg。
        branch_episode: 分支开始的 episode 索引。
        lost_agent: 通信失联的园区名称，园区本地训练仍继续。
        disconnect_schedule: 左闭右开失联区间，按 episode 索引计。

    Returns:
        分支名称到逐 episode 训练奖励数组的映射。
    """
    if fed_method not in ("DSFA", "FedAvg"):
        raise ValueError("断联对比实验仅支持 DSFA 和 FedAvg")
    if not 0 < branch_episode < episodes:
        raise ValueError("branch_episode 必须处于训练范围内")
    if any(start < branch_episode or end <= start for start, end in disconnect_schedule):
        raise ValueError("失联区间必须从分支阶段开始，且终点大于起点")
    presets = default_presets()
    train_series, _, _, train_idx, _ = load_series_split(
        train_days=train, test_days=0
    )
    if not train_idx:
        raise ValueError("训练数据为空；请增加 train_days")
    env = MultiBatteryCoordinator(train_series, **presets.env_kwargs)
    obs_dims, action_dims, max_actions, agents = infer_dims(env)
    if lost_agent not in agents:
        raise ValueError(f"未知失联园区: {lost_agent}")
    lost_index = agents.index(lost_agent)
    settings = presets.algo_kwargs

    def new_model():
        return IDDPG(obs_dims, action_dims, max_actions, **settings)

    model = new_model()
    model_root = os.path.join("model_pth", "lost_contact", fed_method)
    result_root = os.path.join("result", "lost_contact", fed_method)
    os.makedirs(model_root, exist_ok=True)
    os.makedirs(result_root, exist_ok=True)
    checkpoint_path = os.path.join(model_root, f"common_ep{branch_episode - 1}_checkpoint.pt")

    def run_episode(ep: int, disconnected: bool) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """运行一个 episode 并记录训练结果。

        Args:
            ep: 当前 episode 索引，仅用于进度输出。
            disconnected: 失联园区是否在本轮退出联邦通信。

        Returns:
            依次返回各园区训练奖励、训练成本、在线掩码和逐次聚合权重。
            聚合未发生或失败的位置填 NaN；失联园区仍参与环境交互及本地更新。
        """
        online = [i for i in range(len(agents)) if not (disconnected and i == lost_index)]
        if disconnected:
            # 失联期每轮清空其描述符；重连后首次聚合仅使用该轮最近更新。
            model.proto_history[lost_index].clear()
        obs, _ = env.reset()
        ep_rew = np.zeros(len(agents), dtype=np.float64)
        ep_cost = np.zeros(len(agents), dtype=np.float64)
        max_aggregations = (max(1, len(train_idx)) - 1) // 24
        weights = np.full((max_aggregations, len(agents), len(agents)), np.nan)
        aggregation_count = 0
        steps = 0
        horizon = max(1, len(train_idx))
        for t in range(horizon):
            obs_list = list_by_agents(obs, agents)
            if t < presets.noise_warmup_steps:
                actions = [env.action_spaces[a].sample() for a in agents]
            else:
                actions = model.select_actions(obs_list, noise_scale=0.3 * (1 - t / horizon))
            next_obs, reward, term, trunc, info = env.step(dict(zip(agents, actions)))
            next_obs_list = list_by_agents(next_obs, agents)
            reward_list = list_by_agents(reward, agents)
            done = [bool(term[a]) or bool(trunc[a]) for a in agents]
            model.replay.add(flatten_obs(obs_list), flatten_actions(actions), reward_list,
                             flatten_obs(next_obs_list), done)
            if t > 0 and t % 3 == 0:
                model.update(online_agents=online)
            if t > 0 and t % 24 == 0:
                weight = model.Fed_Aggergate(method=fed_method, online_agents=online)
                if weight is not None:
                    weights[aggregation_count] = weight
                aggregation_count += 1
            obs = next_obs
            ep_rew += reward_list
            ep_cost += [info[a]["total_cost"] for a in agents]
            steps += 1
            if all(done):
                break
        online_mask = np.zeros(len(agents), dtype=bool)
        online_mask[online] = True
        print(f"{fed_method} {'disconnected' if disconnected else 'normal'} "
              f"ep={ep} online={[agents[i] for i in online]}")
        return ep_rew * 24 / max(steps, 1), ep_cost, online_mask, weights

    common = {key: [] for key in ("train_rewards", "train_costs", "online", "weights")}
    for ep in range(branch_episode):
        reward, cost, online, weight = run_episode(ep, False)
        for key, value in zip(common, (reward, cost, online, weight)):
            common[key].append(value)

    checkpoint = {
        "episode": branch_episode - 1,
        "algorithm": model.checkpoint_state(),
        "random": _random_state(env),
        "results": common,
        "settings": {"episodes": episodes, "branch_episode": branch_episode,
                     "lost_agent": lost_agent, "disconnect_schedule": disconnect_schedule,
                     "fed_method": fed_method},
    }
    torch.save(checkpoint, checkpoint_path)
    np.savez(os.path.join(result_root, f"common_ep0_{branch_episode - 1}.npz"),
             **{key: np.asarray(value) for key, value in common.items()},
             episodes=np.arange(branch_episode), agents=np.asarray(agents))

    results = {}
    for branch in ("normal", "disconnected"):
        # 每个分支都从文件中的共同阶段终点重建模型及随机状态。
        saved = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        model = new_model()
        model.load_checkpoint_state(saved["algorithm"])
        _restore_random_state(saved["random"], env)
        records = {key: list(value) for key, value in saved["results"].items()}
        for ep in range(branch_episode, episodes):
            disconnected = branch == "disconnected" and any(
                start <= ep < end for start, end in disconnect_schedule
            )
            reward, cost, online, weight = run_episode(ep, disconnected)
            for key, value in zip(records, (reward, cost, online, weight)):
                records[key].append(value)
        branch_dir = os.path.join(model_root, branch)
        model.save_to_directory(branch_dir)
        np.savez(os.path.join(result_root, f"{branch}_ep0_{episodes - 1}.npz"),
                 **{key: np.asarray(value) for key, value in records.items()},
                 lost_agent_train_costs=np.asarray(records["train_costs"])[:, lost_index],
                 episodes=np.arange(episodes), agents=np.asarray(agents),
                 branch_episode=branch_episode,
                 disconnect_schedule=np.asarray(disconnect_schedule, dtype=np.int64))
        np.save(os.path.join(result_root, f"{branch}_{lost_agent}_train_costs.npy"),
                np.asarray(records["train_costs"])[:, lost_index])
        results[branch] = np.asarray(records["train_rewards"])
    env.close()
    return results

# The function signature is updated to accept fed_method
def train_iddpg(episodes=1000, train=7, test=1, Federated=True, fed_method='DSFA'):
    # --- Setup ---
    presets = default_presets()
    train_series, test_series, T, train_idx, test_idx = load_series_split(
        path1="./data/IES_data/G_demand.csv",
        path2="./data/IES_data/H_demand.csv",
        train_days=train,
        test_days=test
    )
    env, test_env = build_envs(train_series, test_series, presets.env_kwargs)
    obs_dims, action_dims, max_actions, agents = infer_dims(env)
    
    # Initialize the algorithm
    iddpg = IDDPG(
        obs_dims, action_dims, max_actions,
        gamma=presets.algo_kwargs["gamma"], tau=presets.algo_kwargs["tau"],
        batch_size=presets.algo_kwargs["batch_size"], buffer_size=presets.algo_kwargs["buffer_size"],
        lr_actor=presets.algo_kwargs["lr_actor"], lr_critic=presets.algo_kwargs["lr_critic"]
    )

    rewards, test_rewards = [], []
    # This list will store the average weight matrix for each episode
    federation_weights_history = []

    # --- Training Loop ---
    for ep in range(episodes):
        start_time = time.time()
        obs, _ = env.reset()
        ep_rew = np.zeros(len(agents), dtype=np.float32)
        
        # List to store weights for the current episode
        episode_weights = []
        
        # (Your existing ep_info dictionary setup remains unchanged)
        ep_info = {a: {k: 0.0 for k in [
            "G_demand_MWH", "p_bat_MWh", "market_buy_MWh", "market_sell_MWh", 
            "newpower_MWh", "e_grid_buy_MWh", "P_boiler_e_MWh", "P_CHP_e_MWh",
            "h_demand_MWH", "h_grid_buy_MWh", "P_CHP_h_MWh", "P_HB_h_MWh",
            "soc_cost", "boiler_cost", "CHP_cost", "HB_cost", "market_cost"
        ]} for a in range(len(agents))}

        horizon = max(1, len(train_idx))
        for t in range(horizon):
            obs_list = list_by_agents(obs, agents)
            
            # Action selection
            if t < presets.noise_warmup_steps:
                actions_list = [env.action_spaces[a].sample() for a in agents]
            else:
                noise_scale = 0.3 * (1 - t / horizon)
                actions_list = iddpg.select_actions(obs_list, noise_scale=noise_scale)

            action_dict = {a: actions_list[i] for i, a in enumerate(agents)}
            next_obs, rew_dict, term_dict, trunc_dict, info_dict = env.step(action_dict)

            # Store transition in replay buffer
            next_obs_list = list_by_agents(next_obs, agents)
            rew_list = list_by_agents(rew_dict, agents)
            done_list = [bool(term_dict[a]) or bool(trunc_dict[a]) for a in agents]
            joint_obs, joint_act = flatten_obs(obs_list), flatten_actions(actions_list)
            joint_next_obs = flatten_obs(next_obs_list)
            iddpg.replay.add(joint_obs, joint_act, rew_list, joint_next_obs, done_list)

            # Update networks
            if t > 0 and t % 3 == 0:
                iddpg.update()

            # --- Federated Aggregation Step ---
            # I_fed is 24, as in your original code
            if Federated and t > 0 and t % 24 == 0:
                # Call aggregation and potentially get weights back
                agg_weights = iddpg.Fed_Aggergate(method=fed_method)
                
                # If DSFA was used, weights are returned and we record them
                if agg_weights is not None: # If weights are returned, store them for this episode
                    episode_weights.append(agg_weights)

            obs = next_obs
            ep_rew += np.array(rew_list, dtype=np.float32)
            
            # (Your existing info dictionary update logic remains unchanged)
            for idx, a in enumerate(agents):
                info = info_dict[a]
                ep_info[idx]["G_demand_MWH"] += info.get("G_demand", 0.0)
                ep_info[idx]["market_buy_MWh"] += info.get("market_buy_MWh", 0.0)
                # ... and so on for all other keys
            
            if all(done_list):
                break

        # --- End of Episode ---
        # Calculate and store the average weights for this episode
        if episode_weights:
            avg_episode_weights = np.mean(episode_weights, axis=0)
            federation_weights_history.append(avg_episode_weights)

        rewards.append((ep_rew / max(1, t)) * 24)
        ep_time = (time.time() - start_time) / 60
        print(f"Fed:{Federated}_iddpg ({fed_method})\n"
              f"Episode {ep+1}/{episodes} | Time: {ep_time:.2f} min | Est. Rem: {ep_time * (episodes - ep - 1):.2f} min")
        print(format_episode_info(ep, (ep_rew / max(1, t)) * 24, ep_info[0]))

    # --- After Training ---
    env.close()
    # Save the model
    model_prefix = f"iddpg_{fed_method}"
    iddpg.save(prefix=model_prefix, Fed=Federated)

    # --- Save Federation Weights ---
    if federation_weights_history:
        # Create a directory for today's results if it doesn't exist
        now_date = dt.now().strftime('%Y%m%d')
        save_dir = os.path.join('result', now_date)
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        
        # Define a unique file name for the weights
        timestamp = dt.now().strftime('%H%M%S')
        weights_filename = f'fed_weights_{fed_method}_{timestamp}.npy'
        weights_filepath = os.path.join(save_dir, weights_filename)
        
        # Convert list of matrices to a 3D numpy array and save
        weights_array = np.array(federation_weights_history)
        np.save(weights_filepath, weights_array)
        print(f"\nFederation weights history saved to: {weights_filepath}")
        print(f"Shape of saved weights: {weights_array.shape}")

    return rewards, test_rewards
