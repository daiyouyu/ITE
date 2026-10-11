# -*- coding: utf-8 -*-
import time
import gc
import io
import numpy as np
import os
import random
import torch
from concurrent.futures import ProcessPoolExecutor
from datetime import datetime as dt
from multiprocessing import get_context, shared_memory
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


def _run_lost_contact_episode(
    model: IDDPG,
    env: MultiBatteryCoordinator,
    agents: list[str],
    horizon: int,
    noise_warmup_steps: int,
    episode: int,
    fed_method: str,
    lost_index: int,
    disconnected: bool,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """运行一个 episode，返回训练奖励、成本、在线掩码和逐次聚合权重。

    失联园区继续本地交互与更新；仅从联邦通信集合中移除。
    权重数组中未发生聚合的位置填 NaN。
    """
    online = [i for i in range(len(agents)) if not (disconnected and i == lost_index)]
    if disconnected:
        # 避免旧描述符跨越失联期；重连后从当前 episode 的更新重新积累。
        model.proto_history[lost_index].clear()
    obs, _ = env.reset()
    episode_reward = np.zeros(len(agents), dtype=np.float64)
    episode_cost = np.zeros(len(agents), dtype=np.float64)
    weights = np.full(((horizon - 1) // 24, len(agents), len(agents)), np.nan)
    aggregation_count = 0
    steps = 0
    for t in range(horizon):
        obs_list = list_by_agents(obs, agents)
        if t < noise_warmup_steps:
            actions = [env.action_spaces[agent].sample() for agent in agents]
        else:
            actions = model.select_actions(obs_list, noise_scale=0.3 * (1 - t / horizon))
        next_obs, reward, term, trunc, info = env.step(dict(zip(agents, actions)))
        next_obs_list = list_by_agents(next_obs, agents)
        reward_list = list_by_agents(reward, agents)
        done = [bool(term[agent]) or bool(trunc[agent]) for agent in agents]
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
        episode_reward += reward_list
        episode_cost += [info[agent]["total_cost"] for agent in agents]
        steps += 1
        if all(done):
            break
    online_mask = np.zeros(len(agents), dtype=bool)
    online_mask[online] = True
    print(f"{fed_method} {'disconnected' if disconnected else 'normal'} "
          f"ep={episode} online={[agents[i] for i in online]}", flush=True)
    return episode_reward * 24 / max(steps, 1), episode_cost, online_mask, weights


def _train_lost_contact_branch(
    branch: str,
    shared_name: str,
    shared_size: int,
    train_days: int,
    cpu_threads: int,
) -> tuple[str, np.ndarray]:
    """子进程从同一内存快照恢复状态，独立续训并保存本分支结果。"""
    torch.set_num_threads(cpu_threads)
    shared = shared_memory.SharedMemory(name=shared_name)
    try:
        snapshot = torch.load(
            io.BytesIO(shared.buf[:shared_size].tobytes()),
            map_location="cpu",
            weights_only=False,
        )
    finally:
        shared.close()

    settings = snapshot["settings"]
    fed_method = settings["fed_method"]
    episodes = settings["episodes"]
    branch_episode = settings["branch_episode"]
    lost_agent = settings["lost_agent"]
    disconnect_schedule = settings["disconnect_schedule"]
    presets = default_presets()
    train_series, _, _, train_idx, _ = load_series_split(train_days=train_days, test_days=0)
    env = MultiBatteryCoordinator(train_series, **presets.env_kwargs)
    try:
        obs_dims, action_dims, max_actions, agents = infer_dims(env)
        if agents != snapshot["agents"] or len(train_idx) != snapshot["horizon"]:
            raise ValueError("分支训练数据或园区顺序与共同阶段不一致")
        lost_index = agents.index(lost_agent)
        model = IDDPG(obs_dims, action_dims, max_actions, **presets.algo_kwargs)
        model.load_checkpoint_state(snapshot["algorithm"])
        _restore_random_state(snapshot["random"], env)
        records = {key: list(value) for key, value in snapshot["results"].items()}
        del snapshot

        for episode in range(branch_episode, episodes):
            disconnected = branch == "disconnected" and any(
                start <= episode < end for start, end in disconnect_schedule
            )
            values = _run_lost_contact_episode(
                model, env, agents, len(train_idx), presets.noise_warmup_steps,
                episode, fed_method, lost_index, disconnected,
            )
            for key, value in zip(records, values):
                records[key].append(value)

        output_root = settings["output_root"]
        model_root = os.path.join(output_root, "model_pth", "lost_contact", fed_method)
        result_root = os.path.join(output_root, "result", "lost_contact", fed_method)
        model.save_to_directory(os.path.join(model_root, branch))
        train_costs = np.asarray(records["train_costs"])
        np.savez(os.path.join(result_root, f"{branch}_ep0_{episodes - 1}.npz"),
                 **{key: np.asarray(value) for key, value in records.items()},
                 lost_agent_train_costs=train_costs[:, lost_index],
                 episodes=np.arange(episodes), agents=np.asarray(agents),
                 branch_episode=branch_episode,
                 disconnect_schedule=np.asarray(disconnect_schedule, dtype=np.int64))
        np.save(os.path.join(result_root, f"{branch}_{lost_agent}_train_costs.npy"),
                train_costs[:, lost_index])
        return branch, np.asarray(records["train_rewards"])
    finally:
        env.close()


def train_iddpg_lost_contact(
    episodes: int = 1000,
    train: int = 365,
    fed_method: str = "DSFA",
    branch_episode: int = 200,
    lost_agent: str = "agent_0",
    disconnect_schedule: tuple[tuple[int, int], ...] = ((200, 400), (600, 800)),
    branch_cpu_threads: int | None = None,
    output_root: str = ".",
) -> dict[str, np.ndarray]:
    """共同训练后经内存快照并行启动正常与断联分支，不写磁盘检查点。

    Args:
        episodes: 总 episode 数，必须大于共同阶段终点。
        train: 训练数据天数。
        fed_method: DSFA 或 FedAvg。
        branch_episode: 分支开始的 episode 索引。
        lost_agent: 通信失联园区；失联期间仍本地训练。
        disconnect_schedule: 左闭右开的失联 episode 区间。
        branch_cpu_threads: 每个分支进程可用的 PyTorch CPU 线程数。
        output_root: 模型和结果的根目录，默认使用当前工作目录。

    Returns:
        正常、断联分支到逐 episode 训练奖励数组的映射。
    """
    if fed_method not in ("DSFA", "FedAvg"):
        raise ValueError("断联对比实验仅支持 DSFA 和 FedAvg")
    if not 0 < branch_episode < episodes:
        raise ValueError("branch_episode 必须处于训练范围内")
    if any(start < branch_episode or end <= start for start, end in disconnect_schedule):
        raise ValueError("失联区间必须从分支阶段开始，且终点大于起点")
    if branch_cpu_threads is None:
        branch_cpu_threads = max(1, (os.cpu_count() or 1) // 2)
    if branch_cpu_threads < 1:
        raise ValueError("branch_cpu_threads 必须大于或等于 1")

    presets = default_presets()
    train_series, _, _, train_idx, _ = load_series_split(train_days=train, test_days=0)
    if not train_idx:
        raise ValueError("训练数据为空；请增加 train_days")
    env = MultiBatteryCoordinator(train_series, **presets.env_kwargs)
    try:
        obs_dims, action_dims, max_actions, agents = infer_dims(env)
        if lost_agent not in agents:
            raise ValueError(f"未知失联园区: {lost_agent}")
        lost_index = agents.index(lost_agent)
        model = IDDPG(obs_dims, action_dims, max_actions, **presets.algo_kwargs)
        common = {key: [] for key in ("train_rewards", "train_costs", "online", "weights")}
        for episode in range(branch_episode):
            values = _run_lost_contact_episode(
                model, env, agents, len(train_idx), presets.noise_warmup_steps,
                episode, fed_method, lost_index, False,
            )
            for key, value in zip(common, values):
                common[key].append(value)

        snapshot = {
            "algorithm": model.checkpoint_state(),
            "random": _random_state(env),
            "results": common,
            "agents": agents,
            "horizon": len(train_idx),
            "settings": {"episodes": episodes, "branch_episode": branch_episode,
                         "lost_agent": lost_agent, "disconnect_schedule": disconnect_schedule,
                         "fed_method": fed_method,
                         "output_root": os.path.abspath(output_root)},
        }
    finally:
        env.close()

    result_root = os.path.join(output_root, "result", "lost_contact", fed_method)
    os.makedirs(result_root, exist_ok=True)
    np.savez(os.path.join(result_root, f"common_ep0_{branch_episode - 1}.npz"),
             **{key: np.asarray(value) for key, value in common.items()},
             episodes=np.arange(branch_episode), agents=np.asarray(agents))

    # Windows 进程传递大型 replay 时可能触及管道消息大小限制，改用共享内存。
    serialized = io.BytesIO()
    torch.save(snapshot, serialized)
    payload = serialized.getvalue()
    shared = shared_memory.SharedMemory(create=True, size=len(payload))
    try:
        shared.buf[:len(payload)] = payload
        shared_size = len(payload)
        del serialized, payload, snapshot, model
        # 父进程不再训练；释放共同阶段模型占用的 GPU 缓存供两个分支使用。
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        # Linux 默认 fork 会继承共同训练阶段的 CUDA 状态，子进程无法再次初始化。
        with ProcessPoolExecutor(max_workers=2, mp_context=get_context("spawn")) as executor:
            futures = [
                executor.submit(
                    _train_lost_contact_branch, branch, shared.name, shared_size,
                    train, branch_cpu_threads,
                )
                for branch in ("normal", "disconnected")
            ]
            return dict(future.result() for future in futures)
    finally:
        shared.close()
        shared.unlink()

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
