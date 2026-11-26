"""在同一套异构园区上对照 IDDPG、MADDPG、主干等权聚合和 DSFA。"""

import argparse
import csv
import json
import os
import random
from contextlib import redirect_stdout
from datetime import date, datetime, timedelta
from pathlib import Path


def run_experiment(args: argparse.Namespace, output: Path) -> None:
    """对每个种子训练四组模型，在留出日评估，并保存模型、日志和汇总指标。"""
    import numpy as np
    import torch

    from flcore.train.train_iddpg import train_iddpg
    from flcore.train.train_maddpg import train_maddpg
    from run_communication import evaluate_training_day

    torch.set_num_threads(1)
    groups = (
        ("IDDPG", False, "DSFA"),
        ("MADDPG", False, None),
        ("BackboneFedAvg", True, "FedAvg"),
        ("HeterogeneousDSFA", True, "DSFA"),
    )
    rows = []
    for seed in dict.fromkeys(args.seeds):
        for name, federated, method in groups:
            random.seed(seed)
            np.random.seed(seed)
            torch.manual_seed(seed)
            directory = output / f"seed_{seed}" / name
            directory.mkdir(parents=True)
            communication = None
            if name == "HeterogeneousDSFA":
                communication = dict(strategy="nearest", neighbors=args.neighbors, seed=seed)
            with (directory / "training.log").open("w", encoding="utf-8") as log, redirect_stdout(log):
                if name == "MADDPG":
                    train_maddpg(
                        args.epochs, args.train_days, 0, Federated=False,
                        compact_actions=True, experiment_dir=str(directory),
                        train_start_date=args.train_date, action_seed=seed,
                    )
                else:
                    train_iddpg(
                        args.epochs, args.train_days, 0, federated, method,
                        compact_actions=True,
                        communication=communication, experiment_dir=str(directory),
                        train_start_date=args.train_date, action_seed=seed,
                    )
            evaluation = evaluate_training_day(
                directory / "models", args.eval_date, compact_actions=True, federated=federated,
                algorithm="MADDPG" if name == "MADDPG" else "IDDPG",
            )
            (directory / "evaluation.json").write_text(
                json.dumps(evaluation, ensure_ascii=False, indent=2), encoding="utf-8",
            )
            with np.load(directory / "rewards.npz") as archive:
                rewards = archive["rewards"]
            row = {
                "seed": seed,
                "method": name,
                "evaluation_date": evaluation["evaluation_date"],
                "system_cost_cny": evaluation["system_cost_cny"],
                "system_carbon_cost_cny": evaluation["system_carbon_cost_cny"],
                "final_100_episode_mean_system_reward": float(rewards[:, -100:].sum(axis=0).mean()),
            }
            for agent, cost in evaluation["park_costs_cny"].items():
                row[f"{agent}_cost_cny"] = cost
                row[f"{agent}_carbon_cost_cny"] = evaluation["park_carbon_costs_cny"][agent]
            rows.append(row)
            with (output / "summary.csv").open("w", encoding="utf-8", newline="") as stream:
                writer = csv.DictWriter(stream, fieldnames=list(row))
                writer.writeheader()
                writer.writerows(rows)
            print(f"seed={seed} {name} 留出日系统成本 {row['system_cost_cny']:.2f} CNY", flush=True)


def main() -> None:
    """读取实验参数，检查训练和评估日期互斥，并建立时间戳结果目录。"""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--epochs", type=int, default=8000, help="训练回合数")
    parser.add_argument("--train-days", type=int, default=1, help="连续训练数据天数")
    parser.add_argument("--train-date", default="2019-06-30", help="训练起始日期 YYYY-MM-DD")
    parser.add_argument("--eval-date", default="2019-07-01", help="留出评估日期 YYYY-MM-DD")
    parser.add_argument("--seeds", nargs="+", type=int, default=[0], help="四组共用的随机种子")
    parser.add_argument("--neighbors", type=int, choices=range(4), default=3,
                        help="DSFA 最多选择的邻居数，不含自身")
    parser.add_argument("--output-root", type=Path, default=Path("result/heterogeneous_dsfa"))
    args = parser.parse_args()
    if min(args.epochs, args.train_days) <= 0:
        parser.error("epochs 和 train-days 必须为正")
    if any(not 0 <= seed <= 2**32 - 1 for seed in args.seeds):
        parser.error("seeds 必须在 0 至 2^32 - 1 之间")
    try:
        train_start = date.fromisoformat(args.train_date)
        eval_day = date.fromisoformat(args.eval_date)
    except ValueError:
        parser.error("日期必须为 YYYY-MM-DD")
    if train_start <= eval_day < train_start + timedelta(days=args.train_days):
        parser.error("评估日必须在训练日期范围之外")
    args.train_date, args.eval_date = train_start.isoformat(), eval_day.isoformat()
    output_root = args.output_root.resolve()
    os.chdir(Path(__file__).resolve().parent)
    output = output_root / datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    output.mkdir(parents=True)
    (output / "config.json").write_text(
        json.dumps(vars(args), indent=2, default=str), encoding="utf-8",
    )
    print(f"实验结果目录 {output}", flush=True)
    run_experiment(args, output)


if __name__ == "__main__":
    main()
