"""运行 4.2.1 的 IDDPG、MADDPG、FedAvg、DSFA 对比实验。"""

import argparse
import json
import random
from datetime import datetime
from pathlib import Path

import numpy as np
import torch

from flcore.train.train_iddpg import train_iddpg
from flcore.train.train_maddpg import train_maddpg


ALGORITHMS = ("IDDPG", "MADDPG", "FedAvg", "DSFA")


def main() -> None:
    """按指定天数和轮数训练对比算法，分别保存模型与奖励。"""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", choices=ALGORITHMS, action="append",
                        help="指定算法；可重复传入，默认运行全部四种")
    parser.add_argument("--epochs", type=int, default=1000, help="每种算法的训练轮数")
    parser.add_argument("--train-days", type=int, default=1, help="训练数据天数，默认一天")
    parser.add_argument("--seed", type=int, default=0, help="随机种子")
    parser.add_argument("--output", type=Path,
                        help="输出目录；默认按时间生成 result/4.2.1 下的新目录")
    args = parser.parse_args()
    if args.epochs < 1 or args.train_days < 1:
        parser.error("--epochs 和 --train-days 必须为正整数")

    algorithms = list(dict.fromkeys(args.run or ALGORITHMS))
    output = args.output or Path("result/4.2.1") / datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    output.mkdir(parents=True, exist_ok=False)
    (output / "config.json").write_text(
        json.dumps({"algorithms": algorithms, "epochs": args.epochs,
                    "train_days": args.train_days, "seed": args.seed},
                   ensure_ascii=False, indent=2), encoding="utf-8")

    for algorithm in algorithms:
        # 每个算法从同一随机种子开始，避免运行顺序改变初始状态。
        random.seed(args.seed)
        np.random.seed(args.seed)
        torch.manual_seed(args.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(args.seed)
        run_dir = output / algorithm
        print(f"开始训练 {algorithm}，输出目录：{run_dir}", flush=True)
        if algorithm == "IDDPG":
            train_iddpg(args.epochs, args.train_days, 0, False,
                        experiment_dir=str(run_dir))
        elif algorithm == "MADDPG":
            train_maddpg(args.epochs, args.train_days, 0, False,
                         experiment_dir=str(run_dir))
        elif algorithm == "FedAvg":
            train_iddpg(args.epochs, args.train_days, 0, True, "FedAvg",
                        experiment_dir=str(run_dir))
        else:
            train_iddpg(args.epochs, args.train_days, 0, True, "DSFA",
                        experiment_dir=str(run_dir), missing_devices_experiment=False)
    print(f"4.2.1 训练完成：{output}", flush=True)


if __name__ == "__main__":
    main()
