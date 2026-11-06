"""绘制园区通信失联实验的训练奖励、成本和聚合权重曲线。"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib import font_manager
from matplotlib.ticker import FuncFormatter


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_RESULT_DIR = PROJECT_ROOT / "result" / "lost_contact"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "result" / "lost_contact" / "figure"
REWARD_SCALE = 100
ALGORITHM_COLORS = {"DSFA": "#1768A6", "FedAvg": "#D47A28"}
BRANCH_STYLES = {"normal": "-", "disconnected": "--"}
BRANCH_LABELS = {"normal": "正常", "disconnected": "断联"}
REWARD_STYLES = {
    ("DSFA", "normal"): {"color": "#1D4ED8", "linestyle": "-", "marker": None},
    ("DSFA", "disconnected"): {"color": "#008F7A", "linestyle": (0, (6, 2)), "marker": "o"},
    ("FedAvg", "normal"): {"color": "#C06000", "linestyle": "-", "marker": None},
    ("FedAvg", "disconnected"): {"color": "#7C3AED", "linestyle": (0, (4, 2, 1, 2)), "marker": "^"},
}


def scientific_reward_formatter(value: float, _position: int) -> str:
    """按既有英文奖励图的格式，将纵轴刻度显示为科学计数法。"""
    if value == 0:
        return "0"
    exponent = int(np.floor(np.log10(abs(value))))
    coefficient = value / (10 ** exponent)
    return rf"{coefficient:.2f}$\times 10^{{{exponent}}}$"


def configure_chinese_font() -> None:
    """优先选用系统中文字体，避免图例和坐标轴文字显示为方框。"""
    for font_path in (
        Path(r"C:\Windows\Fonts\msyh.ttc"),
        Path(r"C:\Windows\Fonts\simhei.ttf"),
        Path("/System/Library/Fonts/PingFang.ttc"),
        Path("/usr/share/fonts/opentype/noto-cjk/NotoSansCJK-Regular.ttc"),
    ):
        if font_path.is_file():
            font_manager.fontManager.addfont(str(font_path))
            plt.rcParams["font.sans-serif"] = [
                font_manager.FontProperties(fname=str(font_path)).get_name(),
                "DejaVu Sans",
            ]
            break
    plt.rcParams["axes.unicode_minus"] = False


def load_results(result_dir: Path) -> dict[tuple[str, str], dict[str, np.ndarray]]:
    """读取各算法的两条分支，并校验绘图所需字段和训练范围。

    Args:
        result_dir: 包含 DSFA、FedAvg 子目录的断联实验结果目录。

    Returns:
        以 ``(算法, 分支)`` 为键的结果字典；未运行的算法会跳过。

    Raises:
        FileNotFoundError: 找不到任何完整的算法分支结果。
        ValueError: 结果数组形状、episode 或园区顺序不一致。
    """
    results: dict[tuple[str, str], dict[str, np.ndarray]] = {}
    for algorithm in ALGORITHM_COLORS:
        algorithm_dir = result_dir / algorithm
        branch_files = {}
        for branch in BRANCH_STYLES:
            candidates = sorted(algorithm_dir.glob(f"{branch}_ep0_*.npz"))
            if candidates:
                branch_files[branch] = max(
                    candidates,
                    key=lambda path: int(path.stem.rsplit("_", 1)[-1]),
                )
        if len(branch_files) != len(BRANCH_STYLES):
            if branch_files:
                raise FileNotFoundError(f"{algorithm_dir} 缺少正常或断联分支结果")
            continue

        for branch, path in branch_files.items():
            with np.load(path, allow_pickle=False) as data:
                required = {"train_rewards", "train_costs", "weights", "online", "episodes", "agents"}
                missing = required.difference(data.files)
                if missing:
                    raise ValueError(f"{path} 缺少字段: {sorted(missing)}")
                record = {key: data[key].copy() for key in required}
                record["branch_episode"] = (
                    np.asarray(data["branch_episode"]).copy()
                    if "branch_episode" in data.files else np.asarray(200)
                )
                record["disconnect_schedule"] = (
                    data["disconnect_schedule"].copy()
                    if "disconnect_schedule" in data.files
                    else np.asarray(((200, 400), (600, 800)))
                )

            episodes = record["episodes"]
            agents = record["agents"]
            expected_shape = (len(episodes), len(agents))
            if not np.array_equal(episodes, np.arange(len(episodes))):
                raise ValueError(f"{path} 的 episodes 必须从 0 连续递增")
            for key in ("train_rewards", "train_costs", "online"):
                if record[key].shape != expected_shape:
                    raise ValueError(f"{path} 的 {key} 形状错误: {record[key].shape}")
            if record["weights"].ndim != 4 or record["weights"].shape[:1] != (len(episodes),) \
                    or record["weights"].shape[2:] != (len(agents), len(agents)):
                raise ValueError(f"{path} 的 weights 形状错误: {record['weights'].shape}")
            results[(algorithm, branch)] = record

        normal = results[(algorithm, "normal")]
        disconnected = results[(algorithm, "disconnected")]
        if not np.array_equal(normal["episodes"], disconnected["episodes"]) \
                or not np.array_equal(normal["agents"], disconnected["agents"]):
            raise ValueError(f"{algorithm} 的两条分支 episode 或园区顺序不一致")

    if not results:
        raise FileNotFoundError(f"{result_dir} 中没有完整的正常/断联结果")
    reference = next(iter(results.values()))
    for (algorithm, branch), record in results.items():
        if not np.array_equal(record["episodes"], reference["episodes"]) \
                or not np.array_equal(record["agents"], reference["agents"]):
            raise ValueError(f"{algorithm} {branch} 与其他分支的 episode 或园区顺序不一致")
    if len(reference["agents"]) != 4:
        raise ValueError("当前四园区图要求结果中恰有四个园区")
    return results


def rolling_mean(values: np.ndarray, window: int) -> np.ndarray:
    """计算保留原始长度的滑动均值，窗口开头只使用已有 episode。"""
    values = np.asarray(values, dtype=float)
    if window <= 1:
        return values.copy()
    valid = np.isfinite(values)
    sums = np.cumsum(np.where(valid, values, 0.0))
    counts = np.cumsum(valid.astype(np.int64))
    start = np.maximum(np.arange(len(values)) - window, -1)
    previous_sums = np.where(start >= 0, sums[np.maximum(start, 0)], 0.0)
    previous_counts = np.where(start >= 0, counts[np.maximum(start, 0)], 0)
    totals = sums - previous_sums
    observations = counts - previous_counts
    return np.divide(totals, observations, out=np.full_like(totals, np.nan), where=observations > 0)


def branch_rolling_mean(
    values: np.ndarray,
    window: int,
    branch: str,
    record: dict[str, np.ndarray],
) -> np.ndarray:
    """断联分支在阶段边界重新计算滑动均值，避免跨阶段混合数据。"""
    if branch == "normal":
        return rolling_mean(values, window)
    boundaries = {0, len(values), int(record["branch_episode"])}
    for start, end in record["disconnect_schedule"]:
        boundaries.update((int(start), int(end)))
    boundaries = sorted(index for index in boundaries if 0 <= index <= len(values))
    smoothed = np.full(len(values), np.nan)
    for start, end in zip(boundaries, boundaries[1:]):
        smoothed[start:end] = rolling_mean(values[start:end], window)
    return smoothed


def decorate_axis(ax: plt.Axes, record: dict[str, np.ndarray]) -> None:
    """标记共同训练终点与两段失联区间，统一折线图样式。"""
    branch_episode = int(record["branch_episode"])
    ax.axvline(branch_episode, color="#68737D", linewidth=1.0, linestyle=":")
    for start, end in record["disconnect_schedule"]:
        ax.axvspan(int(start), int(end), color="#8093A5", alpha=0.10, linewidth=0)
    ax.set_xlim(int(record["episodes"][0]), int(record["episodes"][-1]))
    ax.grid(True, color="#D8DEE5", linewidth=0.65, alpha=0.8)
    ax.spines[["top", "right"]].set_visible(False)


def plot_rewards(
    results: dict[tuple[str, str], dict[str, np.ndarray]],
    output_dir: Path,
    window: int,
    *,
    focus_start: int | None = None,
) -> Path:
    """用独立颜色和线型绘制各园区奖励，放大视图按原始值调整纵轴。"""
    first = next(iter(results.values()))
    agents = first["agents"]
    fig, axes = plt.subplots(2, 2, figsize=(16, 10), sharex=True)
    for index, agent in enumerate(agents):
        ax = axes.flat[index]
        decorate_axis(ax, first)
        visible_values = []
        for (algorithm, branch), record in results.items():
            x = record["episodes"]
            y = record["train_rewards"][:, index] * REWARD_SCALE
            style = REWARD_STYLES[(algorithm, branch)]
            # 两个分支在共同训练阶段完全相同，只绘制正常分支以免重叠遮挡。
            plot_mask = x >= int(record["branch_episode"]) if branch == "disconnected" else np.ones_like(x, dtype=bool)
            plot_x = x[plot_mask]
            plot_y = y[plot_mask]
            if window > 1:
                ax.plot(plot_x, plot_y, color=style["color"], linewidth=0.5, alpha=0.12)
            plotted_values = branch_rolling_mean(y, window, branch, record)[plot_mask]
            ax.plot(plot_x, plotted_values, color=style["color"],
                    linestyle=style["linestyle"], marker=style["marker"],
                    markevery=35, markersize=3, markerfacecolor="white",
                    markeredgewidth=0.9, linewidth=1.25,
                    label=f"{algorithm} · {BRANCH_LABELS[branch]}")
            if focus_start is not None:
                visible_values.append(y[x >= focus_start])
        if focus_start is not None:
            ax.set_xlim(focus_start, int(first["episodes"][-1]))
            finite_values = np.concatenate(visible_values)
            finite_values = finite_values[np.isfinite(finite_values)]
            if finite_values.size:
                lower, upper = float(finite_values.min()), float(finite_values.max())
                padding = max((upper - lower) * 0.08, 0.5 * REWARD_SCALE)
                ax.set_ylim(lower - padding, upper + padding)
        ax.yaxis.set_major_formatter(FuncFormatter(scientific_reward_formatter))
        ax.set_title(f"园区 {index + 1} ({agent})")
        ax.set_ylabel("训练奖励（×100）/ 24 小时")
        ax.set_xlabel("Episode")
    for ax in axes.flat[len(agents):]:
        ax.set_visible(False)
    handles, labels = axes.flat[0].get_legend_handles_labels()
    fig.subplots_adjust(top=0.86, bottom=0.08, hspace=0.26, wspace=0.15)
    fig.legend(handles, labels, loc="upper center", ncol=len(labels),
               bbox_to_anchor=(0.5, 0.94), frameon=False)
    value_label = "逐 episode 原始值" if window == 1 else f"{window} episode 滑动均值"
    view_label = "，分支阶段放大" if focus_start is not None else ""
    fig.suptitle(f"正常与断联训练奖励对比（{value_label}{view_label}）", y=0.99, fontsize=16)
    filename = "train_rewards_detail.png" if focus_start is not None else "train_rewards.png"
    output_path = output_dir / filename
    fig.savefig(output_path, dpi=220)
    plt.close(fig)
    return output_path


def plot_reward_transitions(
    results: dict[tuple[str, str], dict[str, np.ndarray]],
    output_dir: Path,
    radius: int = 25,
) -> Path:
    """放大首次重连和第二次失联前后的逐轮奖励，不对数据进行平滑。"""
    first = next(iter(results.values()))
    schedule = first["disconnect_schedule"]
    if len(schedule) < 2:
        raise ValueError("局部放大图需要两段失联区间")
    transitions = (
        (int(schedule[0][1]), "First reconnection"),
        (int(schedule[1][0]), "Second disconnection"),
    )
    episodes = first["episodes"]
    selected_agent_indices = (0, 2)
    branch_labels = {"normal": "Normal", "disconnected": "Disconnected"}
    fig, axes = plt.subplots(len(selected_agent_indices), 2, figsize=(12, 6.8))
    for row, agent_index in enumerate(selected_agent_indices):
        for column, (center, phase_name) in enumerate(transitions):
            ax = axes[row, column]
            start = max(int(episodes[0]), center - radius)
            end = min(int(episodes[-1]), center + radius)
            decorate_axis(ax, first)
            ax.axvline(center, color="#475569", linestyle=":", linewidth=1.2)
            ax.set_xlim(start, end)
            visible_values = []
            for (algorithm, branch), record in results.items():
                values = record["train_rewards"][:, agent_index] * REWARD_SCALE
                mask = (episodes >= start) & (episodes <= end)
                style = REWARD_STYLES[(algorithm, branch)]
                plotted_values = values[mask]
                visible_values.append(plotted_values)
                ax.plot(episodes[mask], plotted_values,
                        color=style["color"], linestyle=style["linestyle"],
                        marker=style["marker"], markevery=5, markersize=3.5,
                        markerfacecolor="white", markeredgewidth=0.9,
                        linewidth=1.4, label=f"{algorithm} · {branch_labels[branch]}")
            finite_values = np.concatenate(visible_values)
            finite_values = finite_values[np.isfinite(finite_values)]
            if finite_values.size:
                lower, upper = float(finite_values.min()), float(finite_values.max())
                padding = max((upper - lower) * 0.08, 0.3 * REWARD_SCALE)
                ax.set_ylim(lower - padding, upper + padding)
            ax.yaxis.set_major_formatter(FuncFormatter(scientific_reward_formatter))
            ax.set_title(f"IES {agent_index + 1}: {phase_name} (episode {center})", fontsize=10)
            ax.set_ylabel("Training reward per 24 h", fontsize=9)
            ax.set_xlabel("Episode", fontsize=9)
            ax.tick_params(axis="both", labelsize=8)
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.subplots_adjust(left=0.12, right=0.97, top=0.80, bottom=0.10,
                        hspace=0.58, wspace=0.38)
    fig.legend(handles, labels, loc="upper center", ncol=len(labels),
               bbox_to_anchor=(0.5, 0.91), frameon=False, fontsize=9)
    fig.suptitle("Training Reward Around Participation Changes", y=0.98, fontsize=12)
    output_path = output_dir / "train_rewards_transitions.png"
    fig.savefig(output_path, dpi=220)
    plt.close(fig)
    return output_path


def plot_reward_gap(
    results: dict[tuple[str, str], dict[str, np.ndarray]],
    output_dir: Path,
) -> Path:
    """绘制断联与正常分支的逐轮奖励差值，正值表示断联分支奖励更高。"""
    first = next(iter(results.values()))
    agents = first["agents"]
    episodes = first["episodes"]
    fig, axes = plt.subplots(2, 2, figsize=(16, 10), sharex=True)
    for index in range(len(agents)):
        ax = axes.flat[index]
        decorate_axis(ax, first)
        ax.set_xlim(200, int(episodes[-1]))
        ax.axhline(0, color="#475569", linewidth=0.9, zorder=1)
        for algorithm, color in ALGORITHM_COLORS.items():
            if (algorithm, "normal") not in results:
                continue
            normal = results[(algorithm, "normal")]["train_rewards"][:, index]
            disconnected = results[(algorithm, "disconnected")]["train_rewards"][:, index]
            ax.plot(episodes, (disconnected - normal) * REWARD_SCALE,
                    color=color, linewidth=0.9,
                    alpha=0.85, label=algorithm, zorder=2)
        ax.yaxis.set_major_formatter(FuncFormatter(scientific_reward_formatter))
        ax.set_title(f"IES {index + 1}")
        ax.set_ylabel("Reward difference (disconnected − normal)")
        ax.set_xlabel("Episode")
    handles, labels = axes.flat[0].get_legend_handles_labels()
    fig.subplots_adjust(left=0.10, right=0.97, top=0.84, bottom=0.09,
                        hspace=0.42, wspace=0.40)
    fig.legend(handles, labels, loc="upper center", ncol=len(labels),
               bbox_to_anchor=(0.5, 0.94), frameon=False)
    fig.suptitle("Effect of Disconnection on Training Reward", y=0.99, fontsize=16)
    output_path = output_dir / "train_reward_gap.png"
    fig.savefig(output_path, dpi=220)
    plt.close(fig)
    return output_path


def plot_lost_agent_costs(
    results: dict[tuple[str, str], dict[str, np.ndarray]],
    output_dir: Path,
    window: int,
) -> Path:
    """绘制 agent_0 的训练成本，显示其失联期仍参与本地训练的结果。"""
    first = next(iter(results.values()))
    agent_indices = np.flatnonzero(first["agents"] == "agent_0")
    if len(agent_indices) != 1:
        raise ValueError("结果中必须恰有一个 agent_0")
    index = int(agent_indices[0])
    fig, ax = plt.subplots(figsize=(15, 5.5), layout="constrained")
    decorate_axis(ax, first)
    for (algorithm, branch), record in results.items():
        y = record["train_costs"][:, index]
        ax.plot(record["episodes"], branch_rolling_mean(y, window, branch, record),
                color=ALGORITHM_COLORS[algorithm], linestyle=BRANCH_STYLES[branch],
                linewidth=1.8, label=f"{algorithm} · {BRANCH_LABELS[branch]}")
    ax.set_title(f"园区 1 训练成本（{window} episode 滑动均值）")
    ax.set_xlabel("Episode")
    ax.set_ylabel("训练总成本")
    ax.legend(ncol=2, frameon=False)
    output_path = output_dir / "agent_0_train_costs.png"
    fig.savefig(output_path, dpi=220)
    plt.close(fig)
    return output_path


def plot_lost_agent_weights(
    results: dict[tuple[str, str], dict[str, np.ndarray]],
    output_dir: Path,
    window: int,
) -> Path:
    """绘制在线接收园区对 agent_0 Actor 的平均聚合权重。"""
    first = next(iter(results.values()))
    agent_indices = np.flatnonzero(first["agents"] == "agent_0")
    if len(agent_indices) != 1:
        raise ValueError("结果中必须恰有一个 agent_0")
    lost_index = int(agent_indices[0])
    recipients = [i for i in range(len(first["agents"])) if i != lost_index]
    fig, ax = plt.subplots(figsize=(15, 5.5), layout="constrained")
    decorate_axis(ax, first)
    for (algorithm, branch), record in results.items():
        # 聚合时刻权重形状为 episode × 聚合次数 × 接收方 × 参数提供方。
        # NaN 代表该时刻未发生聚合；失联时 agent_0 的列为零。
        contribution = record["weights"][:, :, recipients, lost_index]
        finite = np.isfinite(contribution)
        totals = np.where(finite, contribution, 0.0).sum(axis=(1, 2))
        counts = finite.sum(axis=(1, 2))
        episode_weight = np.divide(
            totals, counts, out=np.full(len(counts), np.nan), where=counts > 0,
        )
        ax.plot(record["episodes"], branch_rolling_mean(episode_weight, window, branch, record),
                color=ALGORITHM_COLORS[algorithm], linestyle=BRANCH_STYLES[branch],
                linewidth=1.8, label=f"{algorithm} · {BRANCH_LABELS[branch]}")
    ax.set_title(f"agent_0 对其他园区的聚合权重贡献（{window} episode 滑动均值）")
    ax.set_xlabel("Episode")
    ax.set_ylabel("平均权重")
    ax.set_ylim(bottom=0)
    ax.legend(ncol=2, frameon=False)
    output_path = output_dir / "agent_0_aggregation_weights.png"
    fig.savefig(output_path, dpi=220)
    plt.close(fig)
    return output_path


def main() -> None:
    """从命令行加载断联实验结果并保存三张折线图。"""
    parser = argparse.ArgumentParser(description="绘制园区断联实验结果")
    parser.add_argument("--result-dir", type=Path, default=DEFAULT_RESULT_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--window", type=int, default=1,
                        help="滑动均值窗口大小；设为 1 绘制原始曲线")
    parser.add_argument("--transition-radius", type=int, default=25,
                        help="400 和 600 附近局部放大图的左右 episode 数")
    args = parser.parse_args()
    if args.window < 1:
        parser.error("--window 必须大于或等于 1")
    if args.transition_radius < 1:
        parser.error("--transition-radius 必须大于或等于 1")
    results = load_results(args.result_dir)
    output_dir = args.output_dir or args.result_dir / "figure"
    output_dir.mkdir(parents=True, exist_ok=True)
    configure_chinese_font()
    print(f"已保存: {plot_rewards(results, output_dir, args.window)}")
    focus_start = max(int(next(iter(results.values()))["branch_episode"]) - 20, 0)
    print(f"已保存: {plot_rewards(results, output_dir, 1, focus_start=focus_start)}")
    print(f"已保存: {plot_reward_transitions(results, output_dir, args.transition_radius)}")
    print(f"已保存: {plot_reward_gap(results, output_dir)}")
    for plot_function in (plot_lost_agent_costs, plot_lost_agent_weights):
        print(f"已保存: {plot_function(results, output_dir, args.window)}")


if __name__ == "__main__":
    main()
