from flcore.Env.multi_env import MultiBatteryCoordinator  # 或 MultiBatteryCoordinator（若采用“子环境+协调器”）
from data.load_data import load_power_data
from flcore.algorithm.MADDPG import MADDPG
from flcore.utils.print_epreward import format_episode_info
import numpy as np

# ----------------------------
# Utilities
# ----------------------------
def flatten_obs(obs_list):
    return np.concatenate([np.asarray(o, dtype=np.float32).ravel() for o in obs_list], axis=0)

def flatten_actions(action_list):
    return np.concatenate([np.asarray(a, dtype=np.float32).ravel() for a in action_list], axis=0)

def list_by_agents(d: dict, agents: list[str]):
    """按固定顺序把 {aid: x} -> [x_i]"""
    return [d[a] for a in agents]

# ----------------------------
# Training loop
# ----------------------------
def train_maddpg(episodes=2000, max_steps=50, render=False):
    # 1) 构建环境（支持并行多智能体：reset 返回 (obs_dict, info)，step 接受 action_dict）
    data = load_power_data("./data/GridSet_no_pred.csv", price_mode="mean")

    # 假设数据是 dict[str, np.ndarray]，每个字段按时间序列存储
    T = len(data[0]["P"])  # 总时长
    days = T // 24  # 小时转天数（前提：dt_hours=1）
    train_days = 7 * 3
    test_days = 7 * 1

    train_idx = train_days * 24
    test_idx = (train_days + test_days) * 24

    # --- 前 5 天训练集 ---
    train_series = [
        {k: v[:train_idx] for k, v in d.items()} for d in data
    ]
    # --- 后 2 天测试集 ---
    test_series = [
        {k: v[train_idx:test_idx] for k, v in d.items()} for d in data
    ]

    env = MultiBatteryCoordinator(
        train_series,
        n_agents=4,
        dt_hours=1.0,
        E_bat_MWh=10000.0,
        P_bat_max_MW=5000.0,
        eta_ch=0.95, eta_dis=0.95,
        soc_min=0.1, soc_max=0.9, soc_init=0.1,
        deg_cost_per_MW=0.1,
        obs_norm=True,
    )

    test_env = MultiBatteryCoordinator(
        test_series,
        n_agents=4,
        dt_hours=1.0,
        E_bat_MWh=10000.0,
        P_bat_max_MW=5000.0,
        eta_ch=0.95, eta_dis=0.95,
        soc_min=0.1, soc_max=0.9, soc_init=0.1,
        deg_cost_per_MW=0.1,
        obs_norm=True,
    )

    # 2) reset（Gymnasium: obs, info）
    obs, _ = env.reset()
    agents = env.agents  # 保证一个固定顺序
    print("agents:", agents)
    print("obs type:", type(obs))

    # 3) 维度探测
    sample_obs_list = list_by_agents(obs, agents)
    obs_dims = [int(np.asarray(o).size) for o in sample_obs_list]
    action_spaces = [env.action_spaces[a] for a in agents]
    action_dims = [int(space.shape[0]) for space in action_spaces]
    max_actions = [float(space.high[0]) for space in action_spaces]

    maddpg = MADDPG(
        obs_dims, action_dims, max_actions,
        lr_actor=1e-3, lr_critic=1e-3, gamma=0.95, tau=0.01,
        batch_size=256, buffer_size=200000
    )

    rewards_record = []
    rewards = []
    test_rewards = []

    for ep in range(episodes):
        obs, _ = env.reset()
        ep_rewards = np.zeros(len(agents), dtype=np.float32)

        # 统计信息初始化（给 format_episode_info 用）
        ep_info = {
            a: {
                "p_bat": 0.0, "p_grid_buy": 0.0,
                "G_demand": 0.0, "newpower_gen": 0.0, "grid_price": 0.0, "bioler_gen": 0.0,
                "soc": 0.0,
                "cost_grid": 0.0, "cost_deg": 0.0, "cost_gen": 0.0,
                "market_buy_MWh": 0.0,
                "market_sell_MWh": 0.0,
                "grid_buy_MWh": 0.0,
                "elec_cost": 0.0,
                "total_cost": 0.0
            }
            for a in range(len(agents))
        }

        for step in range(max_steps):
            # 4) 观测 -> 动作（按固定顺序）
            obs_list = list_by_agents(obs, agents)
            actions_list = maddpg.select_actions(obs_list, noise_scale=0.02)
            # 构造 action 字典（一次性 step）
            action_dict = {a: actions_list[i] for i, a in enumerate(agents)}

            # 5) 一次性推进环境一步（重要：不要对每个 agent 分别调用 env.step）
            next_obs, rew_dict, term_dict, trunc_dict, info_dict = env.step(action_dict)

            # （可选）打印：仅在 ep%50==49 且 step<48 且打印第一个 agent
            if (ep + 1) % 50 == 0 and step < 48:
                a0 = agents[0]
                i0 = info_dict[a0]
                print(
                    f"step:{step},"
                    f"电力负荷: {i0['G_demand']:.3f},\n"
                    f"新能源:{i0['newpower_gen']:.3f},"
                    f"电池：{i0['p_bat']:.3f},"
                    f"锅炉:{i0['bioler_gen']:.3f},"
                    f"锅炉耗费：{i0['cost_gen']:.3f},\n"
                    f"电网：{i0['net_power_MW']:.3f},"
                    f"电网耗费{i0['elec_cost']:.3f},"
                )

            # 6) 组织经验（把 dict -> list，合成 joint）
            next_obs_list = list_by_agents(next_obs, agents)
            rew_list = list_by_agents(rew_dict, agents)
            # term 和 trunc 合并为 done（per-agent）
            done_list = [
                bool(term_dict[a]) or bool(trunc_dict[a])
                for a in agents
            ]

            joint_obs = flatten_obs(obs_list)
            joint_actions = flatten_actions(actions_list)
            joint_next_obs = flatten_obs(next_obs_list)

            maddpg.replay.add(joint_obs, joint_actions, rew_list, joint_next_obs, done_list)
            maddpg.update()   # 如果你有联邦聚合，放在合适频次调用

            # 累计奖励/信息
            obs = next_obs
            ep_rewards += np.array(rew_list, dtype=np.float32)

            for idx, a in enumerate(agents):
                info = info_dict[a]
                ep_info[idx]["p_bat"]      += info.get("p_bat", 0.0)
                ep_info[idx]["p_grid_buy"]     += info.get("p_grid_buy", 0.0)
                ep_info[idx]["G_demand"]   += info.get("G_demand", 0.0)
                ep_info[idx]["newpower_gen"] += info.get("newpower_gen", 0.0)
                ep_info[idx]["grid_price"] += info.get("grid_price", 0.0)
                ep_info[idx]["bioler_gen"] += info.get("bioler_gen", 0.0)
                ep_info[idx]["soc"]        += info.get("soc", 0.0)
                ep_info[idx]["cost_grid"]  += info.get("cost_grid", 0.0)
                ep_info[idx]["cost_deg"]   += info.get("cost_deg", 0.0)
                ep_info[idx]["cost_gen"]   += info.get("cost_gen", 0.0)
                ep_info[idx]["market_buy_MWh"] += info.get("market_buy_MWh", 0.0)
                ep_info[idx]["market_sell_MWh"] += info.get("market_sell_MWh", 0.0)
                ep_info[idx]["grid_buy_MWh"] += info.get("grid_buy_MWh", 0.0)
                ep_info[idx]["elec_cost"] += info.get("elec_cost", 0.0)
                ep_info[idx]["total_cost"] += info.get("total_cost", 0.0)

            # 如果所有 agent 都 done 就提前结束该 episode
            if all(done_list):
                break

        #maddpg.Fed_Aggergate()
        ## ===== 测试（评估，不写入重放缓存 & 不加噪声）=====
        #test_obs, _ = test_env.reset()
        #test_ep_rewards = np.zeros(len(agents), dtype=np.float32)

        # for tstep in range(max_steps):
        #     test_obs_list = list_by_agents(test_obs, agents)
        #     test_actions = maddpg.select_actions(test_obs_list, noise_scale=0.0)  # 测试不加噪声
        #     test_action_dict = {a: test_actions[i] for i, a in enumerate(agents)}
        #
        #     test_next_obs, test_rew_dict, test_term_dict, test_trunc_dict, _ = test_env.step(test_action_dict)
        #
        #     test_ep_rewards += np.array(list_by_agents(test_rew_dict, agents), dtype=np.float32)
        #     test_obs = test_next_obs
        #
        #     if all(bool(test_term_dict[a]) or bool(test_trunc_dict[a]) for a in agents):
        #         break
        rewards_record.append(ep_rewards.sum())
        rewards.append(ep_rewards)
        #test_rewards.append(test_ep_rewards)
        print(format_episode_info(ep, ep_rewards, ep_info[1]))
        # lines = []
        # for test_ep in test_ep_rewards:
        #     lines.append(f"test_Reward: {test_ep:.3f},")
        # print('\n'.join(lines))

    env.close()
    return rewards,test_rewards
