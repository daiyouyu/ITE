# -*- coding: utf-8 -*-
import time
from pathlib import Path
from typing import Any

import numpy as np

from flcore.algorithm.DDPG import DDPGAgent
from flcore.train.train_common import (
    build_envs,
    default_presets,
    flatten_actions,
    flatten_obs,
    infer_dims,
    list_by_agents,
    load_series_split,
)
from flcore.utils.print_epreward import format_episode_info


EPISODE_INFO_KEYS = (
    "G_demand_MWH",
    "p_bat_MWh",
    "market_buy_MWh",
    "market_sell_MWh",
    "newpower_MWh",
    "e_grid_buy_MWh",
    "P_boiler_e_MWh",
    "P_CHP_e_MWh",
    "h_demand_MWH",
    "h_grid_buy_MWh",
    "P_CHP_h_MWh",
    "P_HB_h_MWh",
    "soc_cost",
    "boiler_cost",
    "CHP_cost",
    "HB_cost",
    "market_cost",
)


def _split_joint_action(
    joint_action: np.ndarray,
    agents: list[str],
    action_dims: list[int],
    action_spaces: dict[str, Any],
) -> dict[str, np.ndarray]:
    """
    将整体智能体输出的联合动作按固定 agent 顺序拆回环境动作字典。

    动作会按各子环境的上下界再次裁剪，避免探索噪声产生越界值。
    """
    action = np.asarray(joint_action, dtype=np.float32).reshape(-1)
    expected_dim = sum(action_dims)
    if action.size != expected_dim:
        raise ValueError(
            f"联合动作维度不匹配: expected={expected_dim}, actual={action.size}"
        )

    action_dict: dict[str, np.ndarray] = {}
    start = 0
    for agent_id, action_dim in zip(agents, action_dims):
        end = start + action_dim
        action_space = action_spaces[agent_id]
        local_action = np.clip(
            action[start:end],
            action_space.low,
            action_space.high,
        ).astype(np.float32)
        action_dict[agent_id] = local_action
        start = end
    return action_dict


def _create_episode_info(agents: list[str]) -> dict[str, dict[str, float]]:
    """创建各单体的回合统计容器。"""
    return {
        agent_id: {key: 0.0 for key in EPISODE_INFO_KEYS}
        for agent_id in agents
    }


def _update_episode_info(
    episode_info: dict[str, dict[str, float]],
    info_dict: dict[str, dict[str, Any]],
    agents: list[str],
) -> None:
    """累加环境信息，字段口径与 IDDPG、MADDPG 训练保持一致。"""
    for agent_id in agents:
        info = info_dict[agent_id]
        agent_info = episode_info[agent_id]
        agent_info["G_demand_MWH"] += info.get("G_demand", 0.0)
        agent_info["p_bat_MWh"] += info.get("p_bat", 0.0)
        agent_info["market_buy_MWh"] += info.get("market_buy_MWh", 0.0)
        agent_info["market_sell_MWh"] += info.get("market_sell_MWh", 0.0)
        agent_info["newpower_MWh"] += info.get("newpower_gen", 0.0)
        agent_info["e_grid_buy_MWh"] += info.get("grid_buy_MWh", 0.0)
        agent_info["P_boiler_e_MWh"] += info.get("P_boiler_e", 0.0)
        agent_info["P_CHP_e_MWh"] += info.get("P_CHP_e", 0.0)
        agent_info["h_demand_MWH"] += info.get("H_demand", 0.0)
        agent_info["h_grid_buy_MWh"] += info.get("h_grid_buy", 0.0)
        agent_info["P_CHP_h_MWh"] += info.get("P_CHP_h", 0.0)
        agent_info["P_HB_h_MWh"] += info.get("P_HB_h", 0.0)
        agent_info["soc_cost"] += info.get("soc_cost", 0.0)
        agent_info["boiler_cost"] += info.get("boiler_cost", 0.0)
        agent_info["CHP_cost"] += info.get("CHP_cost", 0.0)
        agent_info["HB_cost"] += info.get("HB_cost", 0.0)
        agent_info["market_cost"] += info.get("market_cashflow", 0.0)


def _sum_episode_info(
    episode_info: dict[str, dict[str, float]],
) -> dict[str, float]:
    """汇总所有单体的统计量，用于展示整体训练结果。"""
    return {
        key: sum(agent_info[key] for agent_info in episode_info.values())
        for key in EPISODE_INFO_KEYS
    }


def train_ddpg(
    episodes: int = 1000,
    train: int = 7,
    test: int = 1,
    max_steps: int | None = None,
) -> tuple[list[np.ndarray], list[np.ndarray]]:
    """
    使用一个 DDPG 智能体对 ``MultiBatteryCoordinator`` 中的所有单体进行整体训练。

    每个园区仅用自身观测调用共享主干及专属头部生成动作。
    中心 Critic 用奖励总和训练，仅更新共享主干；园区 Critic 用自身奖励
    训练，仅更新对应头部。中心 Critic 读取联合观测和动作，园区 Critic 仅读取本地输入。

    Args:
        episodes: 训练回合数。
        train: 传给公共数据切分逻辑的训练天数。
        test: 传给公共数据切分逻辑的测试天数。
        max_steps: 每回合最大步数；为 None 时使用完整训练序列。

    Returns:
        每回合各单体的日均奖励记录，以及预留的测试奖励记录。
    """
    if episodes <= 0:
        raise ValueError("episodes 必须大于 0")
    if max_steps is not None and max_steps <= 0:
        raise ValueError("max_steps 必须大于 0 或为 None")

    presets = default_presets()
    train_series, test_series, _, train_idx, _ = load_series_split(
        path1="./data/IES_data/G_demand.csv",
        path2="./data/IES_data/H_demand.csv",
        train_days=train,
        test_days=test,
    )
    env, test_env = build_envs(train_series, test_series, presets.env_kwargs)
    obs_dims, action_dims, _, agents = infer_dims(env)

    state_dim = sum(obs_dims)
    action_dim = sum(action_dims)
    agent = DDPGAgent(
        state_dim=state_dim,
        action_dim=action_dim,
        max_action=1.0,
        gamma=presets.algo_kwargs["gamma"],
        tau=presets.algo_kwargs["tau"],
        batch_size=presets.algo_kwargs["batch_size"],
        buffer_size=presets.algo_kwargs["buffer_size"],
        lr_actor=presets.algo_kwargs["lr_actor"],
        lr_critic=presets.algo_kwargs["lr_critic"],
        obs_dims=obs_dims,
        action_dims=action_dims,
    )

    rewards: list[np.ndarray] = []
    test_rewards: list[np.ndarray] = []
    total_env_steps = 0
    full_horizon = max(1, len(train_idx))
    horizon = full_horizon if max_steps is None else min(full_horizon, max_steps)

    print(
        "DDPG 整体训练初始化完成: "
        f"agents={len(agents)}, state_dim={state_dim}, action_dim={action_dim}, "
        f"critics=1 个中心（全局输入）+ {agent.n_agents} 个园区（本地输入）, "
        "actor=本地观测→共享主干→专属头部；中心 Q 更新主干，园区 Q 更新头部"
    )

    try:
        for episode in range(episodes):
            start_time = time.time()
            obs_dict, _ = env.reset()
            state = flatten_obs(list_by_agents(obs_dict, agents))
            episode_rewards = np.zeros(len(agents), dtype=np.float32)
            episode_info = _create_episode_info(agents)
            completed_steps = 0

            for step in range(horizon):
                if total_env_steps < presets.noise_warmup_steps:
                    action_list = [env.action_spaces[a].sample() for a in agents]
                    joint_action = flatten_actions(action_list)
                else:
                    # 噪声随全部训练进度衰减，避免每个新回合重新回到高噪声。
                    progress = (episode * horizon + step) / max(1, episodes * horizon - 1)
                    noise_scale = 0.3 * (1.0 - progress)
                    joint_action = flatten_actions([
                        agent.select_local_action(
                            obs_dict[agent_id], i, explore=True, noise_scale=noise_scale,
                        )
                        for i, agent_id in enumerate(agents)
                    ])

                action_dict = _split_joint_action(
                    joint_action,
                    agents,
                    action_dims,
                    env.action_spaces,
                )
                next_obs_dict, reward_dict, term_dict, trunc_dict, info_dict = env.step(
                    action_dict
                )

                next_state = flatten_obs(list_by_agents(next_obs_dict, agents))
                reward_list = np.asarray(
                    list_by_agents(reward_dict, agents),
                    dtype=np.float32,
                )
                done_list = [
                    bool(term_dict[a]) or bool(trunc_dict[a])
                    for a in agents
                ]
                system_done = all(done_list)

                agent.add_to_replay_buffer(
                    state,
                    joint_action,
                    reward_list,
                    next_state,
                    system_done,
                )
                agent.train()

                state = next_state
                obs_dict = next_obs_dict
                episode_rewards += reward_list
                _update_episode_info(episode_info, info_dict, agents)
                completed_steps += 1
                total_env_steps += 1

                if system_done:
                    break

            daily_rewards = episode_rewards / max(1, completed_steps) * 24
            rewards.append(daily_rewards)
            episode_minutes = (time.time() - start_time) / 60
            remaining_minutes = episode_minutes * (episodes - episode - 1)
            print(
                "DDPG（整体训练）\n"
                f"当前轮次时间: {episode_minutes:.3f} 分钟, "
                f"预计剩余时间: {remaining_minutes:.3f} 分钟"
            )
            print(
                format_episode_info(
                    episode,
                    daily_rewards,
                    _sum_episode_info(episode_info),
                )
            )

        agent.save("./model_pth/ddpg")
        result_dir = Path("./result/CentralizedActor")
        result_dir.mkdir(parents=True, exist_ok=True)
        result_path = result_dir / "result_arrays.npz"
        # 与主入口保持相同的键和维度（园区数 × 回合数），便于复用绘图工具。
        np.savez(result_path, DDPG=np.asarray(rewards).T)
        print(f"DDPG 训练结果已保存到 {result_path}")
    finally:
        env.close()
        test_env.close()

    return rewards, test_rewards
