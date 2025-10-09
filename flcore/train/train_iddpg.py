# -*- coding: utf-8 -*-
import numpy as np
import time
from flcore.Env.multi_env import MultiBatteryCoordinator
from data.load_data import load_power_data
from flcore.algorithm.IDDPG import IDDPG
from flcore.utils.print_epreward import format_episode_info

def _flatten(xs):
    return np.concatenate([np.asarray(x, dtype=np.float32).ravel() for x in xs], axis=0)

def _by_agents(d: dict, agents: list[str]):
    return [d[a] for a in agents]

def train_iddpg(episodes=1000,train = 31,test = 1,
                gamma=0.99, tau=0.01, batch_size=256, buffer_size=200000,
                noise_warmup_steps=24):
    # === 数据切分（与 train_maddpg 同风格）===
    data = load_power_data("./data/GridSet_no_pred.csv", price_mode="mean")
    T = len(data[0]["P"])

    train_idx = train * 24
    test_idx = (train + test) * 24

    train_series = [{k: v[:train_idx] for k, v in d.items()} for d in data]
    test_series  = [{k: v[train_idx:test_idx] for k, v in d.items()} for d in data]

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

    obs, _ = env.reset()
    agents = env.agents
    print("agents:", agents)
    print("obs type:", type(obs))

    obs_dims = [int(np.asarray(o).size) for o in _by_agents(obs, agents)]
    action_dims = [int(env.action_spaces[a].shape[0]) for a in agents]
    max_actions = [float(env.action_spaces[a].high[0]) for a in agents]

    iddpg = IDDPG(
        obs_dims, action_dims, max_actions,
        lr_actor=1e-3, lr_critic=1e-3,
        gamma=gamma, tau=tau,
        batch_size=batch_size, buffer_size=buffer_size
    )

    rewards = []
    test_rewards = []

    for ep in range(episodes):
        start_time = time.time()
        obs, _ = env.reset()
        ep_rew = np.zeros(len(agents), dtype=np.float32)

        # 统计（打印友好）
        ep_info = {
            a: {"p_bat":0.0,"p_grid_buy":0.0,"G_demand":0.0,"newpower_gen":0.0,"grid_price":0.0,"bioler_gen":0.0,
                "soc":0.0,"cost_deg":0.0,"cost_gen":0.0,
                "market_buy_MWh":0.0,"market_sell_MWh":0.0,"grid_buy_MWh":0.0,"elec_cost":0.0,"total_cost":0.0}
            for a in range(len(agents))
        }

        for t in range(test_idx):
            obs_list = _by_agents(obs, agents)
            # 前若干步探索更强（或直接 sample），后续减噪
            if t < noise_warmup_steps:
                actions_list = [env.action_spaces[a].sample() for a in agents]
            else:
                noise_scale = (1-t/test_idx) * 0.3
                actions_list = iddpg.select_actions(obs_list, noise_scale=noise_scale)

            action_dict = {a: actions_list[i] for i, a in enumerate(agents)}
            next_obs, rew_dict, term_dict, trunc_dict, info_dict = env.step(action_dict)
            next_obs_list = _by_agents(next_obs, agents)
            rew_list = _by_agents(rew_dict, agents)
            done_list = [bool(term_dict[a]) or bool(trunc_dict[a]) for a in agents]

            joint_obs = _flatten(obs_list)
            joint_act = _flatten(actions_list)
            joint_next_obs = _flatten(next_obs_list)

            iddpg.replay.add(joint_obs, joint_act, rew_list, joint_next_obs, done_list)
            if t % 3 == 0:
                iddpg.update()               # 按需在若干步后再触发
            #if t % 10 == 0:
                #iddpg.Fed_Aggergate()   # 你想要“每 N 次 update 联邦一次”就放外面按频次调
            obs = next_obs
            ep_rew += np.array(rew_list, dtype=np.float32)

            # 累计打印用 info
            for idx, a in enumerate(agents):
                info = info_dict[a]
                ep_info[idx]["p_bat"] += info.get("p_bat", 0.0)
                ep_info[idx]["p_grid_buy"] += info.get("p_grid_buy", 0.0)
                ep_info[idx]["G_demand"] += info.get("G_demand", 0.0)
                ep_info[idx]["newpower_gen"] += info.get("newpower_gen", 0.0)
                ep_info[idx]["grid_price"] += info.get("grid_price", 0.0)
                ep_info[idx]["bioler_gen"] += info.get("bioler_gen", 0.0)
                ep_info[idx]["soc"] += info.get("soc", 0.0)
                ep_info[idx]["cost_deg"] += info.get("cost_deg", 0.0)
                ep_info[idx]["cost_gen"] += info.get("cost_gen", 0.0)
                ep_info[idx]["market_buy_MWh"] += info.get("market_buy_MWh", 0.0)
                ep_info[idx]["market_sell_MWh"] += info.get("market_sell_MWh", 0.0)
                ep_info[idx]["grid_buy_MWh"] += info.get("grid_buy_MWh", 0.0)
                ep_info[idx]["elec_cost"] += info.get("elec_cost", 0.0)
                ep_info[idx]["total_cost"] += info.get("total_cost", 0.0)

            if all(done_list):
                break

        rewards.append((ep_rew / t) * 24)
        end_time = time.time()
        ep_time = (end_time - start_time) / 60
        print(f"当前轮次时间: {ep_time:.3f}分钟，预计剩余时间{ep_time * (episodes - ep):.3f}分钟")
        print(format_episode_info(ep, (ep_rew / t) * 24, ep_info[0]))

        # ===== 可选：测试评估（不入缓存、不加噪声）=====
        # test_obs, _ = test_env.reset()
        # test_ep_rew = np.zeros(len(agents), dtype=np.float32)
        # for _ in range(max_steps):
        #     al = iddpg.select_actions(_by_agents(test_obs, agents), noise_scale=0.0)
        #     nd = {a: al[i] for i, a in enumerate(agents)}
        #     test_next_obs, test_rew_dict, test_term_dict, test_trunc_dict, _ = test_env.step(nd)
        #     test_ep_rew += np.array(_by_agents(test_rew_dict, agents), dtype=np.float32)
        #     test_obs = test_next_obs
        #     if all(bool(test_term_dict[a]) or bool(test_trunc_dict[a]) for a in agents):
        #         break
        # test_rewards.append(test_ep_rew)

    env.close()
    return rewards, test_rewards
