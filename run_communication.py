"""并行运行最近邻、最远邻、随机选择与 K=2、3 的六组通信实验。"""
import argparse
import json
import os
import random
import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path


def run_experiment(strategy: str, neighbors: int, seed: int, epochs: int,
                   train_days: int, output: str) -> dict:
    """在独立进程训练一组配置，固定随机种子并返回通信统计。"""
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
    # 分进程日志避免六组输出交错，异常仍通过 future 传给主进程。
    from contextlib import redirect_stdout, redirect_stderr
    with (directory / 'training.log').open('w', encoding='utf-8') as log:
        with redirect_stdout(log), redirect_stderr(log):
            train_iddpg(epochs, train_days, 0, True, 'DSFA',
                        communication=dict(strategy=strategy, neighbors=neighbors, seed=seed),
                        output_root=str(directory))
    records = json.loads((directory / 'communication.json').read_text(encoding='utf-8'))['rounds']
    return dict(strategy=strategy, neighbors=neighbors, seed=seed,
                rounds=len(records), total_bytes=sum(r['total_bytes'] for r in records),
                output=str(directory))


def main() -> None:
    """默认六组并行、一天数据；支持缩小组合及调整训练轮数。"""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--strategy', nargs='+', choices=['nearest', 'farthest', 'random'],
                        default=['nearest', 'farthest', 'random'])
    parser.add_argument('--neighbors', nargs='+', type=int, choices=[2, 3], default=[2, 3])
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--epochs', type=int, default=1000)
    parser.add_argument('--train-days', type=int, default=1)
    parser.add_argument('--workers', type=int, default=6)
    args = parser.parse_args()
    if min(args.epochs, args.train_days, args.workers) <= 0:
        parser.error('epochs、train-days 和 workers 必须为正')
    os.chdir(Path(__file__).resolve().parent)
    jobs = [(s, k) for s in dict.fromkeys(args.strategy) for k in dict.fromkeys(args.neighbors)]
    directory = Path('result/communication') / datetime.now().strftime('%Y%m%d_%H%M%S_%f')
    directory.mkdir(parents=True)
    (directory / 'config.json').write_text(json.dumps(vars(args), indent=2), encoding='utf-8')
    results = []
    workers = min(args.workers, len(jobs), os.cpu_count() or 1)
    print(f'并行进程: {workers}；数据: {args.train_days} 天；结果: {directory}', flush=True)
    with ProcessPoolExecutor(max_workers=workers, mp_context=mp.get_context('spawn')) as pool:
        futures = [pool.submit(run_experiment, s, k, args.seed, args.epochs,
                               args.train_days, str(directory)) for s, k in jobs]
        for future in as_completed(futures):
            result = future.result()
            results.append(result)
            (directory / 'summary.json').write_text(json.dumps(results, indent=2), encoding='utf-8')
            print(f"完成 {result['strategy']} K={result['neighbors']}，通信轮数={result['rounds']}", flush=True)


if __name__ == '__main__':
    main()
