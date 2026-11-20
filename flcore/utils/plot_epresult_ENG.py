"""绘制 run_421.py 输出的四种算法收敛曲线。"""

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import ScalarFormatter


ALGORITHMS = ("MADDPG", "DSFA", "IDDPG", "FedAvg")
LINE_STYLES = {
    "MADDPG": ("maddpg", "#1f77b4", "-"),
    "DSFA": ("OURS", "#ff7f0e", "--"),
    "IDDPG": ("iddpg", "#2ca02c", "-."),
    "FedAvg": ("FedAvg", "#d62728", ":"),
}


def load_daily_cost(run_dir: Path, algorithm: str) -> np.ndarray:
    """读取逐轮园区奖励，返回系统每日成本等价值（CNY）。

    Args:
        run_dir: run_421.py 生成的批次目录。
        algorithm: 算法名称，对应目录下的 rewards.npz。

    Returns:
        每轮的系统成本等价值。环境奖励为成本的负值并除以 1000，
        训练函数已将每轮奖励换算为 24 小时口径。

    Raises:
        ValueError: 奖励数组不是园区 × episode 的非空二维结构时抛出。
    """
    reward_path = run_dir / algorithm / "rewards.npz"
    with np.load(reward_path) as result:
        rewards = np.asarray(result["rewards"], dtype=np.float64)
    if rewards.ndim != 2 or rewards.shape[0] == 0 or rewards.shape[1] == 0:
        raise ValueError(f"奖励数组应为园区 × episode 的非空二维数组：{reward_path}")
    return -rewards.sum(axis=0) * 1000.0


def moving_average(values: np.ndarray, window: int) -> np.ndarray:
    """计算不丢失前几个 episode 的滑动平均；window=1 时返回原曲线。"""
    cumulative = np.concatenate(([0.0], np.cumsum(values)))
    indices = np.arange(1, len(values) + 1)
    starts = np.maximum(0, indices - window)
    return (cumulative[indices] - cumulative[starts]) / (indices - starts)


def draw_result(run_dir: Path, output: Path, window: int) -> None:
    """从同一实验批次绘制四种算法的逐轮成本收敛曲线并保存 PNG。"""
    fig, ax = plt.subplots(figsize=(10, 6), dpi=200)
    try:
        for algorithm in ALGORITHMS:
            cost = load_daily_cost(run_dir, algorithm)
            episodes = np.arange(1, len(cost) + 1)
            label, color, line_style = LINE_STYLES[algorithm]
            if window > 1:
                ax.plot(episodes, cost, color=color, linestyle=line_style,
                        alpha=0.15, linewidth=0.7)
            ax.plot(episodes, moving_average(cost, window), label=label,
                    color=color, linestyle=line_style, linewidth=2.3)

        ax.set_xlim(0, 1000)
        ax.set_xticks([0, 300, 600, 1000])
        ax.set_xlabel("Episode", fontsize=20)
        ax.set_ylabel("Equivalent Daily System Cost (CNY)", fontsize=20)
        formatter = ScalarFormatter(useMathText=True)
        formatter.set_scientific(True)
        formatter.set_powerlimits((0, 0))
        ax.yaxis.set_major_formatter(formatter)
        ax.tick_params(axis="both", labelsize=16)
        ax.yaxis.get_offset_text().set_fontsize(16)
        ax.grid(True, linestyle="--", alpha=0.35)
        ax.legend(frameon=True, fontsize=17)
        fig.tight_layout()
        output.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output, dpi=300, bbox_inches="tight")
        print(f"Saved plot -> {output}")
    finally:
        plt.close(fig)


def main() -> None:
    """解析批次目录、输出路径与平滑窗口，然后生成英文收敛图。"""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_dir", type=Path, help="run_421.py 生成的批次目录")
    parser.add_argument("--output", type=Path,
                        help="PNG 输出路径，默认保存到批次目录下的 convergence_ENG.png")
    parser.add_argument("--window", type=int, default=20,
                        help="滑动平均窗口，默认 20；设为 1 绘制原始曲线")
    args = parser.parse_args()
    if args.window < 1:
        parser.error("--window 必须大于或等于 1")
    draw_result(args.run_dir, args.output or args.run_dir / "convergence_ENG.png",
                args.window)


if __name__ == "__main__":
    main()
