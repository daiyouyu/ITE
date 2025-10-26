# -*- coding: utf-8 -*-
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import List, Tuple, Dict, Any
import numpy as np

from data.load_data import load_power_data
from data.load_data import load_ITE_data
from flcore.Env.multi_env import MultiBatteryCoordinator


# ---------- 公用小工具 ----------
def flatten_obs(obs_list: List[np.ndarray]) -> np.ndarray:
    return np.concatenate([np.asarray(o, dtype=np.float32).ravel() for o in obs_list], axis=0)


def flatten_actions(action_list: List[np.ndarray]) -> np.ndarray:
    return np.concatenate([np.asarray(a, dtype=np.float32).ravel() for a in action_list], axis=0)


def list_by_agents(d: Dict[str, Any], agents: List[str]):
    return [d[a] for a in agents]


# ---------- 预设集 ----------
@dataclass
class Presets:
    # 数据切分（天）
    train_days: int = 7
    test_days: int = 1
    # 环境共享参数（多处复用）
    env_kwargs: dict = None
    # 算法共享超参
    algo_kwargs: dict = None
    # 训练细节
    noise_warmup_steps: int = 24


def default_presets() -> Presets:
    """
    可选 profile:
      - 'fast_debug': 2/1 天（接近你原 train_maddpg 的快速试跑）
      - 'weekly'   : 7/1 天（默认）
      - 'monthly'  : 31/1 天（接近你原 train_iddpg 的设置）
    """

    env_kwargs = dict(
        n_agents=4,
        dt_hours=1.0,
        deg_cost_per_MW=1,
        obs_norm=True,
        # === 新增：逐 agent 覆盖 ===
        per_agent_kwargs={
            "agent_0": {"E_bat_MWh": 10.0, "P_bat_max_MW": 6.0, "deg_cost_per_MW": 1.0,
                        "CHP_a": 0.76, "CHP_b": 0.4275, "CHP_c": 0.114,
                        "CHP_d": 271.6, "CHP_e": 203.7, "CHP_f": 75,
                        "Fbmax": 2, "cf": 612,
                        "P_HB_e_h": 15, "P_HB_e_l": 0,
                        },

            "agent_1": {"E_bat_MWh": 20.0, "P_bat_max_MW": 8.0, "deg_cost_per_MW": 1.5,
                        "CHP_a": 0.6, "CHP_b": 0.5, "CHP_c": 0.108,
                        "CHP_d": 229.2, "CHP_e": 171.9, "CHP_f": 75,
                        "Fbmax": 1, "cf": 650,
                        "P_HB_e_h": 5, "P_HB_e_l": 1,
                        },

            "agent_2": {"E_bat_MWh": 30.0, "P_bat_max_MW": 12.0, "deg_cost_per_MW": 0.8,
                        "CHP_a": 0.76, "CHP_b": 0.4275, "CHP_c": 0.114,
                        "CHP_d": 271.6, "CHP_e": 203.7, "CHP_f": 75,
                        "Fbmax": 2, "cf": 612,
                        "P_HB_e_h": 15, "P_HB_e_l": 0,
                        },

            "agent_3": {"E_bat_MWh": 15.0, "P_bat_max_MW": 5.0, "deg_cost_per_MW": 2.0,
                        "CHP_a": 0.76, "CHP_b": 0.4275, "CHP_c": 0.114,
                        "CHP_d": 271.6, "CHP_e": 203.7, "CHP_f": 75,
                        "Fbmax": 2, "cf": 612,
                        "P_HB_e_h": 15, "P_HB_e_l": 0,
                        },
        },
    )

    algo_kwargs = dict(
        lr_actor=1e-3, lr_critic=1e-3,
        gamma=0.95, tau=0.01,
        batch_size=256, buffer_size=200_000
    )

    return Presets(
        env_kwargs=env_kwargs,
        algo_kwargs=algo_kwargs,
        noise_warmup_steps=24
    )


# ---------- 数据与环境 ----------
def load_series_split(path1="./data/IES_data/G_demand.csv",
                      path2="./data/IES_data/H_demand.csv",
                      train_days=7, test_days=1):
    data = load_ITE_data(path1, path2)
    T = len(data[0]["P"])
    days = (train_days + test_days) * 24
    start_day = datetime(2019, 1, 1)
    print(start_day.weekday())
    hour_weekdays = []
    for hour in range(days):
        current_time = start_day + timedelta(hours=hour)
        hour_weekdays.append(current_time.weekday())

    # data = load_power_data(path)

    train_idx = [i for i, wd in enumerate(hour_weekdays) if wd in {1, 2, 3, 4, 5}]
    # train_idx = train_days * 24
    # test_idx = (train_days + test_days) * 24
    test_idx = [i for i, wd in enumerate(hour_weekdays) if wd in {0, 6}]

    train_series = []
    test_series = []
    for d in data:
        train_data = {k: [v[i] for i in train_idx] for k, v in d.items()}
        train_series.append(train_data)
        test_data = {k: [v[i] for i in test_idx] for k, v in d.items()}
        test_series.append(test_data)
    # test_series = [{k: v[train_idx:test_idx] for k, v in d.items()} for d in data]

    return train_series, test_series, T, train_idx, test_idx


def build_envs(train_series, test_series, env_kwargs):
    train_env = MultiBatteryCoordinator(train_series, **env_kwargs)
    test_env = MultiBatteryCoordinator(test_series, **env_kwargs)
    return train_env, test_env


def infer_dims(env) -> Tuple[List[int], List[int], List[float], List[str]]:
    obs_dict, _ = env.reset()
    agents = env.agents
    sample_obs_list = list_by_agents(obs_dict, agents)
    obs_dims = [int(np.asarray(o).size) for o in sample_obs_list]
    action_spaces = [env.action_spaces[a] for a in agents]
    action_dims = [int(space.shape[0]) for space in action_spaces]
    max_actions = [float(space.high[0]) for space in action_spaces]
    return obs_dims, action_dims, max_actions, agents
