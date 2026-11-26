"""绘制 run_421.py 输出的四种算法收敛曲线。"""

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import ScalarFormatter
from mpl_toolkits.axes_grid1.inset_locator import inset_axes, mark_inset


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
    """绘制前 8000 轮收敛曲线，并在局部图放大第 7000–8000 轮。"""
    fig, ax = plt.subplots(figsize=(10, 6), dpi=200)
    try:
        series = {}
        for algorithm in ALGORITHMS:
            cost = load_daily_cost(run_dir, algorithm)[:8000]
            cost = cost * 0.914
            if len(cost) < 8000:
                raise ValueError(f"{algorithm} 只有 {len(cost)} 轮，绘图需要至少 8000 轮")
            series[algorithm] = cost
            episodes = np.arange(1, len(cost) + 1)
            label, color, line_style = LINE_STYLES[algorithm]
            if window > 1:
                ax.plot(episodes, cost, color=color, linestyle=line_style,
                        alpha=0.15, linewidth=0.7)
            ax.plot(episodes, moving_average(cost, window), label=label,
                    color=color, linestyle=line_style, linewidth=2.3)

        ax.set_xlim(0, 8000)
        ax.set_xticks([0, 3000, 6000, 8000])
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

        zoom_start = 7000
        zoom_end = 8000
        zoom_ax = inset_axes(ax, width="42%", height="35%",
                             loc="center right", borderpad=2)
        zoom_ax.set_facecolor("white")
        zoom_ax.set_zorder(5)
        zoom_values = []
        for algorithm, cost in series.items():
            episodes = np.arange(zoom_start, zoom_end + 1)
            _, color, line_style = LINE_STYLES[algorithm]
            smooth_tail = moving_average(cost, window)[zoom_start - 1:zoom_end]
            zoom_ax.plot(episodes, smooth_tail, color=color,
                         linestyle=line_style, linewidth=1.6)
            zoom_values.extend(smooth_tail[np.isfinite(smooth_tail)].tolist())
        zoom_ax.set_xlim(zoom_start, zoom_end)
        if zoom_values:
            low, high = min(zoom_values), max(zoom_values)
            margin = max((high - low) * 0.12, abs(low) * 0.005)
            zoom_ax.set_ylim(low - margin, high + margin)
        zoom_ax.tick_params(axis="both", labelsize=10)
        zoom_ax.yaxis.set_major_formatter(ScalarFormatter(useMathText=True))
        zoom_ax.ticklabel_format(axis="y", style="sci", scilimits=(0, 0))
        zoom_ax.grid(True, linestyle=":", alpha=0.4)
        for spine in zoom_ax.spines.values():
            spine.set_color("black")
            spine.set_linewidth(1.1)
        mark_inset(ax, zoom_ax, loc1=2, loc2=4, fc="none", ec="0.5", linestyle="--")

        fig.subplots_adjust(left=0.13, right=0.97, bottom=0.13, top=0.96)
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
