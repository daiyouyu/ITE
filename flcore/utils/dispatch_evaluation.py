"""从保存的 Actor 权重评估一天的调度成本。"""

import json
from pathlib import Path
from typing import Literal


def evaluate_training_day(model_directory: Path, train_start_date: str | None = None, *,
                          compact_actions: bool = False, federated: bool = True,
                          algorithm: Literal["IDDPG", "MADDPG"] = "IDDPG") -> dict:
    """在指定日期的 24 小时内无噪声评估模型，默认沿用通信实验的训练日期。

    Args:
        model_directory: 当前实验保存的 Actor 权重目录。
        compact_actions: 按与训练相同的设备配置建立异构动作环境。
        federated: 指定加载联邦模型或独立训练模型。
        algorithm: 指定评估时重建的模型类型。

    Returns:
        包含评估日索引、各园区总成本与碳税成本，以及系统汇总成本的字典。

    Raises:
        ValueError: 数据不足以覆盖完整的评估日或动作配置不一致时抛出。
    """
    import torch
    from flcore.Env.multi_env import MultiBatteryCoordinator
    from flcore.train.train_common import (default_presets, infer_dims,
                                           list_by_agents, load_series_split)

    eval_series, _, _, train_indices, _ = load_series_split(
        train_days=1, test_days=0, train_start_date=train_start_date)
    if len(train_indices) != 24:
        raise ValueError(f'评估数据不足 24 小时: {len(train_indices)}')
    presets = default_presets(compact_actions=compact_actions)
    if compact_actions:
        configuration = json.loads((model_directory.parent / "actions.json").read_text(encoding="utf-8"))
        presets.env_kwargs["observation_scales"] = configuration["observation_scales"]
    env = MultiBatteryCoordinator(eval_series, **presets.env_kwargs)
    try:
        obs_dims, action_dims, max_actions, agents = infer_dims(env)
        if compact_actions and any(
            tuple(configuration["actions"][agent]) != env.envs[agent].action_names for agent in agents
        ):
            raise ValueError("评估环境的动作清单与已保存的模型不一致")
        if algorithm == "MADDPG":
            from flcore.algorithm.MADDPG import MADDPG

            model = MADDPG(obs_dims, action_dims, max_actions,
                           shared_obs_indices=[-2], **presets.algo_kwargs)
        else:
            from flcore.algorithm.IDDPG import IDDPG

            model = IDDPG(obs_dims, action_dims, max_actions, **presets.algo_kwargs)
        for index in range(len(agents)):
            weights_path = model_directory / f'{federated}_actor_{index}.pth'
            model.actors[index].load_state_dict(
                torch.load(weights_path, map_location='cpu', weights_only=True)
            )
        obs, _ = env.reset()
        park_costs = {agent: 0.0 for agent in agents}
        park_carbon_costs = {agent: 0.0 for agent in agents}
        steps = 0
        for _ in range(24):
            actions = model.select_actions(list_by_agents(obs, agents), noise_scale=0.0)
            obs, _, terminated, truncated, info = env.step(
                {agent: actions[index] for index, agent in enumerate(agents)}
            )
            for agent in agents:
                park_costs[agent] += float(info[agent]['total_cost'])
                park_carbon_costs[agent] += float(info[agent]['co2_cost'])
            steps += 1
            if all(bool(terminated[agent]) or bool(truncated[agent]) for agent in agents):
                break
        return {
            'evaluation_start_hour': train_indices[0],
            'evaluation_date': str(eval_series[0]['datetime'][0])[:10],
            'evaluation_hours': steps,
            'park_costs_cny': park_costs,
            'park_carbon_costs_cny': park_carbon_costs,
            'system_cost_cny': float(sum(park_costs.values())),
            'system_carbon_cost_cny': float(sum(park_carbon_costs.values())),
        }
    finally:
        env.close()
