"""比较联邦聚合权重的逐轮变化与训练后期调整幅度。"""

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.colors import Normalize, LinearSegmentedColormap
import numpy as np

OUTPUT_RESULT_PATH = Path(r"D:\ITE\result\fed_weights")
DEFAULT_DSFA = Path(r"D:\ITE\result\20260404\fed_weights_DSFA_194147.npy")
DEFAULT_ALLDSFA = Path(r"D:\ITE\result\20261005\fed_weights_AllDSFA_004613.npy")


def load_ies4_weights(file_path: Path) -> np.ndarray:
    """读取完整矩阵序列，返回 IES4 对四个模型来源的权重（episode，4）。

    文件须为至少两个 episode 的 4×4 有限数值矩阵序列。
    行索引为接收方，列索引为模型来源；不接受仅保存对角线的数据。
    """
    data = np.load(file_path, allow_pickle=False)
    if data.ndim != 3 or data.shape[1:] != (4, 4) or data.shape[0] < 2:
        raise ValueError(f"需要完整的 (episode, 4, 4) 权重矩阵: {file_path}, shape={data.shape}")
    data = np.asarray(data, dtype=float)
    if not np.isfinite(data).all():
        raise ValueError(f"权重包含非有限值: {file_path}")
    return data[:, 3, :]


def plot_comparison(
    dsfa: np.ndarray,
    all_dsfa: np.ndarray,
    output_dir: Path,
    late_fraction: float = 0.5,
) -> plt.Figure:
    """生成上下排列的变化热图与后期点图，仅保存 PNG。

    热图展示 IES4 的 W41、W44；点图展示 W41、W42、W43、W44。
    热图取每 20 个 episode 的原始权重的极差（最大值减最小值），尾窗按实际样本计算。
    两种方法使用相同的完整线性色阶；下方指标由原始差分计算。
    """
    if dsfa.shape != all_dsfa.shape or dsfa.ndim != 2 or dsfa.shape[1] != 4:
        raise ValueError("两组数据形状必须一致，且包含 IES4 的四个来源权重")
    if not 0 < late_fraction <= 1:
        raise ValueError("late_fraction 必须在 (0, 1] 内")
    rounds = dsfa.shape[0]
    start = int(rounds * (1 - late_fraction))
    if rounds - start < 2:
        raise ValueError("后期区间至少需要两个轮次")
    changes = [np.diff(values, axis=0) for values in (dsfa, all_dsfa)]
    metrics = np.array([np.abs(delta[start:]).mean(axis=0) for delta in changes])
    # 热图直接计算原始权重的窗口极差，不对逐轮差分求极差。
    window_size = 20
    window_starts = np.arange(0, rounds, window_size)
    raw_heat = np.array([dsfa[:, 0], all_dsfa[:, 0],
                         dsfa[:, 3], all_dsfa[:, 3]])
    heat_data = np.column_stack([
        np.ptp(raw_heat[:, left:left + window_size], axis=1)
        for left in window_starts
    ])
    edges = np.append(window_starts - 0.5, rounds - 0.5)
    norm = Normalize(vmin=0, vmax=max(float(heat_data.max()), 0.25))
    cmap = LinearSegmentedColormap.from_list(
        "adjustment_magnitude", ["#f7fbff", "#c6dbef", "#6baed6", "#2171b5", "#08306b"])
    colors = ["#2479ad", "#d87522"]
    labels = ["DSFA", "AllDSFA"]
    with plt.rc_context({"font.family": "DejaVu Sans", "font.size": 12,
                         "axes.spines.top": False, "axes.spines.right": False}):
        fig = plt.figure(figsize=(11, 11), facecolor="white")
        ax_heat = fig.add_axes([0.19, 0.59, 0.68, 0.34])
        # 同一来源权重的两种方法相邻；明确窗口边界，最后一个窗口不拉伸。
        heat = ax_heat.pcolormesh(edges, np.arange(5) - 0.5, heat_data,
                                 cmap=cmap, norm=norm, shading="flat",
                                 edgecolors="white", linewidth=0.35)
        ax_heat.set_ylim(3.5, -0.5)
        ax_heat.set_xlim(-0.5, rounds - 0.5)
        ax_heat.set_yticks(range(4), [r"$W_{41}$ · DSFA", r"$W_{41}$ · AllDSFA",
                                    r"$W_{44}$ · DSFA", r"$W_{44}$ · AllDSFA"], fontsize=15)
        ax_heat.axhline(1.5, color="white", linewidth=3)
        ax_heat.axvline(start - 0.5, color="#303030", linestyle="--", linewidth=1.3)
        ax_heat.set_title("(a) Weight variation range", loc="left", fontsize=15, fontweight="bold", pad=14)
        ax_heat.set_xlabel("Episode", fontsize=16, labelpad=10)
        ax_heat.tick_params(axis="x", labelsize=15)
        ax_heat.grid(False)
        cax = fig.add_axes([0.90, 0.59, 0.022, 0.34])
        cb = fig.colorbar(heat, cax=cax, ticks=[0.00, 0.15, 0.25])
        cb.ax.set_yticklabels(["0.00", "0.15", "0.25"])
        cb.ax.tick_params(labelsize=16)
        cb.ax.set_title(r"$\Delta W_{4i}$", fontsize=18, pad=12)
        ax = fig.add_axes([0.19, 0.10, 0.68, 0.35])
        ax.set_title("(b) Late-training adjustment", loc="left", fontsize=15, fontweight="bold", pad=15)
        ax.axhspan(2.63, 3.37, color="#f2f3f5", zorder=0)
        for i in range(4):
            ax.plot(metrics[:, i], [i, i], color="#b6bdc5", linewidth=2.5, zorder=1)
        for method in range(2):
            ax.scatter(metrics[method], range(4), s=180, color=colors[method],
                       marker="o" if method == 0 else "D", label=labels[method], zorder=3,
                       edgecolors="white", linewidths=1)
            for i, value in enumerate(metrics[method]):
                ax.annotate(f"{value:.4f}", (value, i), xytext=(0, 16 if method == 0 else -25),
                            textcoords="offset points", ha="center", fontsize=15, color=colors[method])
        ax.set_yticks(range(4), [rf"$W_{{4{i + 1}}}$" for i in range(4)], fontsize=16)
        ax.get_yticklabels()[3].set_fontweight("bold")
        ax.set_ylim(3.65, -0.85)
        ax.set_xlim(0, max(float(metrics.max()) * 1.2, 1e-5))
        ax.set_xlabel(r"Mean $|W_{4i}^{(t)}-W_{4i}^{(t-1)}|$", fontsize=16, labelpad=10)
        ax.set_xticks(np.linspace(0, ax.get_xlim()[1], 4))
        ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda value, _: f"{value:.3f}"))
        ax.tick_params(axis="x", labelsize=15)
        ax.grid(axis="x", color="#e8e8e8", linewidth=0.8)
        ax.set_axisbelow(True)
        ax.legend(loc="upper right", bbox_to_anchor=(1, 1.0), ncol=2,
                  frameon=False, fontsize=14, markerscale=1.15,
                  handletextpad=0.6, columnspacing=1.3)
        output_dir.mkdir(parents=True, exist_ok=True)
        output_path = output_dir / "fed_weights_DSFA_comparison_ENG.png"
        fig.savefig(output_path, dpi=600, facecolor="white")
        print(f"已保存 PNG: {output_path}; 后期区间: {start}-{rounds - 1}")
        print(f"后期平均调整幅度（行：DSFA / AllDSFA；列：W41 / W42 / W43 / W44）:\n{metrics}")
        return fig


def main() -> None:
    """解析输入文件、输出目录及后期比例；默认保存图片，--show 可交互查看。"""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dsfa", type=Path, default=DEFAULT_DSFA)
    parser.add_argument("--all-dsfa", type=Path, default=DEFAULT_ALLDSFA)
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_RESULT_PATH)
    parser.add_argument("--late-fraction", type=float, default=0.5)
    parser.add_argument("--show", action="store_true")
    args = parser.parse_args()
    fig = plot_comparison(load_ies4_weights(args.dsfa), load_ies4_weights(args.all_dsfa),
                          args.output_dir, args.late_fraction)
    if args.show:
        plt.show()
    plt.close(fig)


if __name__ == "__main__":
    main()


