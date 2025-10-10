from flcore.train.train_common import (
    default_presets, load_series_split, build_envs,
    infer_dims, list_by_agents, flatten_obs, flatten_actions
)
from flcore.algorithm.MADDPG import MADDPG
from flcore.utils.print_epreward import format_episode_info
import numpy as np
import time

def train_maddpg(episodes=1000,train=7,test=1):
    # --- 统一使用公共预设 ---
    presets = default_presets()
    train_series, test_series, T, train_idx, test_idx = load_series_split(
        "./data/GridSet_no_pred.csv", train_days=train, test_days=test
    )
    env, test_env = build_envs(train_series, test_series, presets.env_kwargs)

    # 维度探测（公共函数）
    obs_dims, action_dims, max_actions, agents = infer_dims(env)
    maddpg = MADDPG(
        obs_dims, action_dims, max_actions,
        **presets.algo_kwargs
    )

    rewards_record, rewards, test_rewards = [], [], []

    for ep in range(episodes):
        start_time = time.time()
        obs, _ = env.reset()
        ep_rew = np.zeros(len(agents), dtype=np.float32)

        ep_info = {
            a: {
                "G_demand_MWH": 0.0, "p_bat_MWh": 0.0,
                "market_buy_MWh": 0.0, "market_sell_MWh": 0.0, "grid_buy_MWh": 0.0,
                "newpower_gen_MWh": 0.0, "bioler_gen_MWh": 0.0,
                "soc_cost": 0.0, "boiler_cost": 0.0, "p_grid_buy": 0.0,
                "elec_cost": 0.0, "total_cost": 0.0
            } for a in range(len(agents))
        }

        horizon = max(1, test_idx)
        for t in range(horizon):
            obs_list = list_by_agents(obs, agents)

            # 统一探索策略：前若干步强探索，后续降噪
            if t < presets.noise_warmup_steps:
                actions_list = [env.action_spaces[a].sample() for a in agents]
            else:
                noise_scale = 0.3 * (1 - t / horizon)
                actions_list = maddpg.select_actions(obs_list, noise_scale=noise_scale)

            action_dict = {a: actions_list[i] for i, a in enumerate(agents)}
            next_obs, rew_dict, term_dict, trunc_dict, info_dict = env.step(action_dict)

            next_obs_list = list_by_agents(next_obs, agents)
            rew_list = list_by_agents(rew_dict, agents)
            done_list = [bool(term_dict[a]) or bool(trunc_dict[a]) for a in agents]

            joint_obs = flatten_obs(obs_list)
            joint_actions = flatten_actions(actions_list)
            joint_next_obs = flatten_obs(next_obs_list)

            maddpg.replay.add(joint_obs, joint_actions, rew_list, joint_next_obs, done_list)
            if t % 3 == 0:
                maddpg.update()
            #if t % 24 == 0:
                #maddpg.Fed_Aggergate()
            obs = next_obs
            ep_rew += np.array(rew_list, dtype=np.float32)

            for idx, a in enumerate(agents):
                info = info_dict[a]
                ep_info[idx]["G_demand_MWH"] += info.get("G_demand", 0.0)
                ep_info[idx]["market_buy_MWh"] += info.get("market_buy_MWh", 0.0)
                ep_info[idx]["market_sell_MWh"] += info.get("market_sell_MWh", 0.0)
                ep_info[idx]["newpower_gen_MWh"] += info.get("newpower_gen", 0.0)
                ep_info[idx]["bioler_gen_MWh"] += info.get("bioler_gen", 0.0)
                ep_info[idx]["grid_buy_MWh"] += info.get("grid_buy_MWh", 0.0)
                ep_info[idx]["p_bat_MWh"] += info.get("p_bat", 0.0)
                ep_info[idx]["p_grid_buy"] += info.get("p_grid_buy", 0.0)
                ep_info[idx]["soc_cost"] += info.get("soc_cost", 0.0)
                ep_info[idx]["boiler_cost"] += info.get("boiler_cost", 0.0)
                ep_info[idx]["elec_cost"] += info.get("elec_cost", 0.0)
                ep_info[idx]["total_cost"] += info.get("total_cost", 0.0)

            if all(done_list):
                break

        # 与原实现保持一致的“按小时归一后*24”的口径
        rewards.append((ep_rew / max(1, t)) * 24)
        ep_time = (time.time() - start_time) / 60
        print(f"当前轮次时间: {ep_time:.3f}分钟,预计剩余时间：{ep_time * (horizon-t):.3f}")
        print(format_episode_info(ep, (ep_rew / max(1, t)) * 24, ep_info[0]))

    env.close()
    maddpg.save("maddpg")
    return rewards, test_rewards
