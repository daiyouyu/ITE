"""运行两阶段通信的 K=0、1、2、3 对照实验，并保存画图所需的结果。"""
import argparse
import csv
import json
import os
import random
import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path


def evaluate_next_day(model_directory: Path, train_days: int) -> dict:
    """在训练区间之后的 24 小时无噪声评估模型。

    Args:
        model_directory: 当前实验保存的 Actor 权重目录。
        train_days: 训练所用的连续天数，用于确定评估日的起点。

    Returns:
        包含评估日索引、各园区总成本与碳税成本，以及系统汇总成本的字典。

    Raises:
        ValueError: 数据不足以覆盖完整的下一天时抛出。
    """
    import numpy as np
    import torch
    from data.load_data import load_ITE_data
    from flcore.Env.multi_env import MultiBatteryCoordinator
    from flcore.train.train_common import default_presets, infer_dims, list_by_agents
    from flcore.algorithm.IDDPG import IDDPG

    series = load_ITE_data('./data/IES_data/G_demand.csv', './data/IES_data/H_demand.csv')
    start = train_days * 24
    end = start + 24
    if any(len(park['P']) < end for park in series):
        raise ValueError(f'评估数据不足 24 小时: start={start}, end={end}')
    eval_series = [{key: values[start:end] for key, values in park.items()} for park in series]
    presets = default_presets()
    env = MultiBatteryCoordinator(eval_series, **presets.env_kwargs)
    try:
        obs_dims, action_dims, max_actions, agents = infer_dims(env)
        model = IDDPG(obs_dims, action_dims, max_actions, **presets.algo_kwargs)
        for index in range(len(agents)):
            weights_path = model_directory / f'True_actor_{index}.pth'
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
            'evaluation_start_hour': start,
            'evaluation_hours': steps,
            'park_costs_cny': park_costs,
            'park_carbon_costs_cny': park_carbon_costs,
            'system_cost_cny': float(sum(park_costs.values())),
            'system_carbon_cost_cny': float(sum(park_carbon_costs.values())),
        }
    finally:
        env.close()


def run_experiment(strategy: str, neighbors: int, seed: int, epochs: int,
                   train_days: int, output: str) -> dict:
    """在独立进程训练一组配置，并保存通信与次日评估结果。

    Args:
        strategy: 邻居选择策略。
        neighbors: 最多接收的其他园区 Actor 数量，不含自身。
        seed: 本组训练的随机种子。
        epochs: 训练轮数。
        train_days: 训练数据天数，默认实验为一天。
        output: 本批实验的输出目录。

    Returns:
        包含通信字节、末段训练奖励和次日成本的画图汇总数据。
    """
    import numpy as np
    import torch
    torch.set_num_threads(1)
    from flcore.train.train_iddpg import train_iddpg
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    directory = Path(output) / f'{strategy}_k{neighbors}_seed{seed}'
    directory.mkdir(parents=True, exist_ok=False)
    # 分进程日志避免多组输出交错，异常仍通过 future 传给主进程。
    from contextlib import redirect_stdout, redirect_stderr
    with (directory / 'training.log').open('w', encoding='utf-8') as log:
        with redirect_stdout(log), redirect_stderr(log):
            train_iddpg(epochs, train_days, 0, True, 'DSFA',
                        communication=dict(strategy=strategy, neighbors=neighbors, seed=seed),
                        output_root=str(directory))
    records = json.loads((directory / 'communication.json').read_text(encoding='utf-8'))['rounds']
    with np.load(directory / 'rewards.npz') as reward_file:
        rewards = reward_file['DSFA']
    evaluation = evaluate_next_day(directory / 'models', train_days)
    (directory / 'evaluation.json').write_text(
        json.dumps(evaluation, indent=2, ensure_ascii=False), encoding='utf-8'
    )
    return dict(
        strategy=strategy, neighbors=neighbors, seed=seed, epochs=epochs,
        train_days=train_days, rounds=len(records),
        descriptor_bytes=sum(r['descriptor_bytes'] for r in records),
        request_bytes=sum(r['request_bytes'] for r in records),
        actor_bytes=sum(r['actor_bytes'] for r in records),
        total_bytes=sum(r['total_bytes'] for r in records),
        final_100_episode_reward_per_park=rewards[:, -min(100, rewards.shape[1]):].mean(axis=1).tolist(),
        **evaluation, output=str(directory),
    )


def main() -> None:
    """默认最近邻策略下四组 K 值并行运行，训练数据为一天。"""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--strategy', nargs='+', choices=['nearest', 'farthest', 'random'],
                        default=['nearest'])
    parser.add_argument('--neighbors', nargs='+', type=int, choices=[0, 1, 2, 3], default=[0, 1, 2, 3])
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--seeds', nargs='+', type=int,
                        help='可选：一次运行多个随机种子；指定后覆盖 --seed')
    parser.add_argument('--epochs', type=int, default=1000)
    parser.add_argument('--train-days', type=int, default=1)
    parser.add_argument('--workers', type=int, default=4)
    args = parser.parse_args()
    if min(args.epochs, args.train_days, args.workers) <= 0:
        parser.error('epochs、train-days 和 workers 必须为正')
    os.chdir(Path(__file__).resolve().parent)
    seeds = list(dict.fromkeys(args.seeds if args.seeds is not None else [args.seed]))
    jobs = [(s, k, seed) for seed in seeds
            for s in dict.fromkeys(args.strategy)
            for k in dict.fromkeys(args.neighbors)]
    directory = Path('result/communication') / datetime.now().strftime('%Y%m%d_%H%M%S_%f')
    directory.mkdir(parents=True)
    (directory / 'config.json').write_text(json.dumps(vars(args), indent=2), encoding='utf-8')
    results = []
    workers = min(args.workers, len(jobs), os.cpu_count() or 1)
    print(f'并行进程: {workers}；数据: {args.train_days} 天；结果: {directory}', flush=True)
    with ProcessPoolExecutor(max_workers=workers, mp_context=mp.get_context('spawn')) as pool:
        futures = [pool.submit(run_experiment, s, k, seed, args.epochs,
                               args.train_days, str(directory)) for s, k, seed in jobs]
        for future in as_completed(futures):
            result = future.result()
            results.append(result)
            (directory / 'summary.json').write_text(json.dumps(results, indent=2), encoding='utf-8')
            with (directory / 'plot_data.csv').open('w', newline='', encoding='utf-8') as stream:
                fieldnames = ['strategy', 'neighbors', 'seed', 'epochs', 'train_days',
                              'rounds', 'descriptor_bytes', 'request_bytes', 'actor_bytes',
                              'total_bytes', 'evaluation_start_hour', 'evaluation_hours',
                              'system_cost_cny', 'system_carbon_cost_cny',
                              *[f'agent_{index}_final_100_episode_reward' for index in range(4)],
                              *[f'agent_{index}_cost_cny' for index in range(4)],
                              *[f'agent_{index}_carbon_cost_cny' for index in range(4)]]
                writer = csv.DictWriter(stream, fieldnames=fieldnames)
                writer.writeheader()
                for item in sorted(results, key=lambda row: (row['strategy'], row['neighbors'], row['seed'])):
                    row = {key: item.get(key) for key in fieldnames}
                    for index in range(4):
                        row[f'agent_{index}_final_100_episode_reward'] = (
                            item['final_100_episode_reward_per_park'][index]
                        )
                        row[f'agent_{index}_cost_cny'] = item['park_costs_cny'][f'agent_{index}']
                        row[f'agent_{index}_carbon_cost_cny'] = item['park_carbon_costs_cny'][f'agent_{index}']
                    writer.writerow(row)
            print(f"完成 {result['strategy']} K={result['neighbors']}，通信轮数={result['rounds']}", flush=True)


if __name__ == '__main__':
    main()
