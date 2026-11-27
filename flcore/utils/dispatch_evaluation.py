"""从保存的 Actor 权重评估一天的调度成本。"""

import json
from pathlib import Path
from typing import TYPE_CHECKING, Literal

from flcore.Env.multi_env import MultiBatteryCoordinator
from flcore.train.train_common import Presets, default_presets, infer_dims

if TYPE_CHECKING:
    from flcore.algorithm.IDDPG import IDDPG
    from flcore.algorithm.MADDPG import MADDPG


STANDARD_ALGORITHMS = ("IDDPG", "MADDPG", "FedAvg", "DSFA")
HETEROGENEOUS_ALGORITHMS = ("IDDPG", "MADDPG", "BackboneFedAvg", "HeterogeneousDSFA")
FEDERATED_ALGORITHMS = {"FedAvg", "DSFA", "BackboneFedAvg", "HeterogeneousDSFA"}


def resolve_evaluation_run(
    run_dir: Path, seed: int | None = None,
) -> tuple[Path, dict, tuple[str, ...]]:
    """解析批次配置，返回模型所在目录、训练配置和该实验的算法列表。

    异构批次可直接指定 seed 目录；批次只配置一个 seed 时自动选择，
    配置多个 seed 时需明确指定，避免评估到其他模型。
    """
    run_dir = Path(run_dir)
    direct_seed = (
        run_dir.name.startswith("seed_") and run_dir.name[5:].isdigit()
        and (run_dir.parent / "config.json").is_file()
    )
    batch_dir = run_dir.parent if direct_seed else run_dir
    config = json.loads((batch_dir / "config.json").read_text(encoding="utf-8"))
    if "seeds" not in config:
        if seed is not None:
            raise ValueError("--seed 仅用于异构实验批次")
        return run_dir, config, STANDARD_ALGORITHMS

    seeds = list(dict.fromkeys(config["seeds"]))
    if direct_seed:
        directory_seed = int(run_dir.name[5:])
        if seed is not None and seed != directory_seed:
            raise ValueError(f"--seed {seed} 与目录 {run_dir} 不一致")
        seed = directory_seed
    elif seed is None and len(seeds) == 1:
        seed = seeds[0]
    if seed not in seeds:
        raise ValueError(f"请通过 --seed 选择批次中的种子，可用值：{seeds}")

    seed_dir = batch_dir / f"seed_{seed}"
    if not seed_dir.is_dir():
        raise FileNotFoundError(f"未找到种子模型目录：{seed_dir}")
    return seed_dir, config, HETEROGENEOUS_ALGORITHMS


def evaluation_presets(
    model_directory: Path, *, compact_actions: bool,
) -> tuple[Presets, dict[str, list[str]] | None]:
    """恢复模型的评估环境预设，异构模型沿用训练时保存的观测尺度和动作清单。"""
    presets = default_presets(compact_actions=compact_actions)
    if not compact_actions:
        return presets, None
    configuration = json.loads(
        (model_directory.parent / "actions.json").read_text(encoding="utf-8")
    )
    presets.env_kwargs["observation_scales"] = configuration["observation_scales"]
    return presets, configuration["actions"]


def load_actor_model(
    model_directory: Path, env: MultiBatteryCoordinator, *, algorithm: Literal["IDDPG", "MADDPG"],
    federated: bool, algo_kwargs: dict, action_config: dict[str, list[str]] | None,
) -> "IDDPG | MADDPG":
    """根据评估环境重建推理模型，核对动作顺序后加载各园区的 Actor 权重。"""
    import torch
    from flcore.algorithm.IDDPG import IDDPG
    from flcore.algorithm.MADDPG import MADDPG

    obs_dims, action_dims, max_actions, agents = infer_dims(env)
    if action_config is not None and any(
        tuple(action_config.get(agent, ())) != env.envs[agent].action_names for agent in agents
    ):
        raise ValueError(f"评估环境的动作清单与已保存的模型不一致：{model_directory}")
    if algorithm == "MADDPG":
        model = MADDPG(obs_dims, action_dims, max_actions,
                       shared_obs_indices=[-2], **algo_kwargs)
    else:
        model = IDDPG(obs_dims, action_dims, max_actions, **algo_kwargs)
    for index, actor in enumerate(model.actors):
        weights_path = model_directory / f"{federated}_actor_{index}.pth"
        actor.load_state_dict(torch.load(weights_path, map_location="cpu", weights_only=True))
        actor.eval()
    return model


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
    from flcore.train.train_common import list_by_agents, load_series_split

    eval_series, _, _, train_indices, _ = load_series_split(
        train_days=1, test_days=0, train_start_date=train_start_date)
    if len(train_indices) != 24:
        raise ValueError(f'评估数据不足 24 小时: {len(train_indices)}')
    presets, action_config = evaluation_presets(model_directory, compact_actions=compact_actions)
    env = MultiBatteryCoordinator(eval_series, **presets.env_kwargs)
    try:
        model = load_actor_model(model_directory, env, algorithm=algorithm, federated=federated,
                                 algo_kwargs=presets.algo_kwargs, action_config=action_config)
        obs, _ = env.reset()
        agents = env.agents
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
