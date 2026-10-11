"""绘制园区通信失联实验的训练奖励、成本和聚合权重曲线。"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib import font_manager


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_RESULT_DIR = PROJECT_ROOT / "result" / "lost_contact"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "result" / "lost_contact" / "figure"
ALGORITHM_COLORS = {"DSFA": "#1768A6", "FedAvg": "#D47A28"}
BRANCH_STYLES = {"normal": "-", "disconnected": "--"}
BRANCH_LABELS = {"normal": "正常", "disconnected": "断联"}


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
) -> Path:
    """为每个园区绘制四条分支奖励曲线，颜色区分算法、线型区分分支。"""
    first = next(iter(results.values()))
    agents = first["agents"]
    fig, axes = plt.subplots(2, 2, figsize=(16, 10), sharex=True)
    for index, agent in enumerate(agents):
        ax = axes.flat[index]
        decorate_axis(ax, first)
        for (algorithm, branch), record in results.items():
            x = record["episodes"]
            y = record["train_rewards"][:, index]
            color = ALGORITHM_COLORS[algorithm]
            if window > 1:
                ax.plot(x, y, color=color, linewidth=0.55, alpha=0.11)
            ax.plot(x, branch_rolling_mean(y, window, branch, record), color=color,
                    linestyle=BRANCH_STYLES[branch], linewidth=1.8,
                    label=f"{algorithm} · {BRANCH_LABELS[branch]}")
        ax.set_title(f"园区 {index + 1} ({agent})")
        ax.set_ylabel("训练奖励 / 24 小时")
        ax.set_xlabel("Episode")
    for ax in axes.flat[len(agents):]:
        ax.set_visible(False)
    handles, labels = axes.flat[0].get_legend_handles_labels()
    fig.subplots_adjust(top=0.86, bottom=0.08, hspace=0.26, wspace=0.15)
    fig.legend(handles, labels, loc="upper center", ncol=len(labels),
               bbox_to_anchor=(0.5, 0.94), frameon=False)
    fig.suptitle(f"正常与断联训练奖励对比（{window} episode 滑动均值）", y=0.99, fontsize=16)
    output_path = output_dir / "train_rewards.png"
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
    parser.add_argument("--window", type=int, default=25,
                        help="滑动均值窗口大小；设为 1 绘制原始曲线")
    args = parser.parse_args()
    if args.window < 1:
        parser.error("--window 必须大于或等于 1")
    results = load_results(args.result_dir)
    output_dir = args.output_dir or args.result_dir / "figure"
    output_dir.mkdir(parents=True, exist_ok=True)
    configure_chinese_font()
    for plot_function in (plot_rewards, plot_lost_agent_costs, plot_lost_agent_weights):
        print(f"已保存: {plot_function(results, output_dir, args.window)}")


if __name__ == "__main__":
    main()
