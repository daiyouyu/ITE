"""绘制四个园区在不同训练算法下的奖励/成本收敛曲线。"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Mapping

import matplotlib.pyplot as plt
import numpy as np
from matplotlib import font_manager
from matplotlib.axes import Axes
from mpl_toolkits.axes_grid1.inset_locator import inset_axes, mark_inset


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_PRIMARY_RESULT = PROJECT_ROOT / "result" / "20260903" / "result_arrays.npz"
DEFAULT_BASELINE_RESULT = PROJECT_ROOT / "result" / "20260329" / "result_arrays.npz"
DEFAULT_OUTPUT_PATH = PROJECT_ROOT / "result" / "figure" / "plot_epresult_for_four.png"
PARK_COUNT = 4

ALGORITHM_STYLES = {
    "maddpg": ("MADDPG", "-", "#1f77b4"),
#    "DSFA": ("OURS", "--", "#ff7f0e"),
    "iddpg": ("IDDPG", "-.", "#2ca02c"),
#    "FedAvg": ("FedAvg", ":", "#d62728"),
    "DDPG": ("DDPG", ":", "#d62728"),
}


def configure_chinese_font() -> None:
    """配置可用的中文字体，避免园区标题和坐标名称显示为方框。"""

    preferred_fonts = [
        Path(r"C:\Windows\Fonts\msyh.ttc"),
        Path(r"C:\Windows\Fonts\simhei.ttf"),
        Path("/System/Library/Fonts/PingFang.ttc"),
        Path("/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc"),
    ]
    for font_path in preferred_fonts:
        if not font_path.exists():
            continue
        try:
            font_manager.fontManager.addfont(str(font_path))
            font_name = font_manager.FontProperties(fname=str(font_path)).get_name()
            plt.rcParams["font.sans-serif"] = [font_name, "DejaVu Sans"]
            break
        except (OSError, RuntimeError, ValueError):
            continue

    plt.rcParams["axes.unicode_minus"] = False


def normalize_park_rewards(
    rewards: np.ndarray,
    algorithm_name: str,
    park_count: int = PARK_COUNT,
) -> np.ndarray:
    """将单个算法的奖励数据统一转换为 ``(园区, Episode)``。

    Args:
        rewards: 奖励数组，支持 ``(园区, Episode)`` 或
            ``(Episode, 园区)`` 两种排列。
        algorithm_name: 算法名称，仅用于生成清晰的异常信息。
        park_count: 期望的园区数量。

    Returns:
        类型为 float、形状为 ``(park_count, episode_count)`` 的二维数组。

    Raises:
        ValueError: 数组不是二维数据，或任一维均无法对应园区数量时抛出。
    """

    values = np.squeeze(np.asarray(rewards, dtype=float))
    if values.ndim != 2:
        raise ValueError(
            f"算法 {algorithm_name!r} 的结果应为二维数组，实际形状为 {values.shape}"
        )

    if values.shape[0] == park_count:
        return values
    if values.shape[1] == park_count:
        return values.T

    raise ValueError(
        f"算法 {algorithm_name!r} 的结果中未找到 {park_count} 个园区，"
        f"实际形状为 {values.shape}"
    )


def load_park_results(
    primary_result_path: Path,
    baseline_result_path: Path,
    park_count: int = PARK_COUNT,
) -> dict[str, np.ndarray]:
    """从两份 NPZ 文件读取各算法的逐园区训练结果。

    ``DSFA`` 和 ``FedAvg`` 沿用统一规划脚本，从 primary 文件读取；
    ``maddpg`` 和 ``iddpg`` 从 baseline 文件读取。
    """

    if not primary_result_path.is_file():
        raise FileNotFoundError(f"找不到联邦训练结果文件: {primary_result_path}")
    if not baseline_result_path.is_file():
        raise FileNotFoundError(f"找不到基线训练结果文件: {baseline_result_path}")

    with np.load(primary_result_path) as primary_result, np.load(
        baseline_result_path
    ) as baseline_result:
        sources = {
            "maddpg": baseline_result,
#            "DSFA": primary_result,
            "iddpg": baseline_result,
            "DDPG": primary_result,
        }
        park_results: dict[str, np.ndarray] = {}
        for algorithm_name, result_file in sources.items():
            if algorithm_name not in result_file.files:
                raise KeyError(
                    f"结果文件缺少算法 {algorithm_name!r}；"
                    f"现有键为 {result_file.files}"
                )
            park_results[algorithm_name] = normalize_park_rewards(
                result_file[algorithm_name],
                algorithm_name,
                park_count,
            ).copy()

    return park_results


def _add_zoom_inset(
    ax: Axes,
    park_series: Mapping[str, np.ndarray],
) -> None:
    """在子图中放大最后 15% 的 Episode，便于比较收敛后的细微差异。"""

    max_episode_count = max(len(series) for series in park_series.values())
    if max_episode_count < 5:
        return

    zoom_start_index = int(max_episode_count * 0.85)
    zoom_start_episode = zoom_start_index + 1
    zoom_values: list[float] = []
    zoom_ax = inset_axes(ax, width="42%", height="34%", loc="center right", borderpad=1.2)

    for algorithm_name, series in park_series.items():
        label, line_style, color = ALGORITHM_STYLES[algorithm_name]
        episodes = np.arange(1, len(series) + 1)
        zoom_ax.plot(
            episodes,
            series,
            label=label,
            linestyle=line_style,
            color=color,
            linewidth=1.2,
        )
        if len(series) > zoom_start_index:
            finite_values = series[zoom_start_index:][
                np.isfinite(series[zoom_start_index:])
            ]
            zoom_values.extend(finite_values.tolist())

    zoom_ax.set_xlim(zoom_start_episode, max_episode_count)
    if zoom_values:
        y_min = min(zoom_values)
        y_max = max(zoom_values)
        value_range = y_max - y_min
        margin = value_range * 0.1 if value_range > 0 else max(abs(y_min) * 0.02, 1.0)
        zoom_ax.set_ylim(y_min - margin, y_max + margin)

    zoom_ax.grid(True, linestyle=":", alpha=0.5)
    zoom_ax.tick_params(axis="both", labelsize=7)
    mark_inset(ax, zoom_ax, loc1=2, loc2=4, fc="none", ec="0.5", linestyle="--")


def print_final_stage_summary(
    park_results: Mapping[str, np.ndarray],
    final_ratio: float = 0.1,
) -> None:
    """打印各园区最后一段训练数据的均值，辅助比较最终训练表现。"""

    print("\n各园区末 10% Episode 平均值：")
    for park_index in range(PARK_COUNT):
        summary_parts = []
        for algorithm_name, values in park_results.items():
            label = ALGORITHM_STYLES[algorithm_name][0]
            series = values[park_index]
            final_count = max(1, int(len(series) * final_ratio))
            final_mean = float(np.nanmean(series[-final_count:]))
            summary_parts.append(f"{label}={final_mean:.4f}")
        print(f"园区 {park_index + 1}: " + ", ".join(summary_parts))


def draw_four_park_results(
    park_results: Mapping[str, np.ndarray],
    output_path: Path = DEFAULT_OUTPUT_PATH,
    max_episodes: int | None = 1000,
    value_scale: float = 100.0,
    show: bool = True,
) -> None:
    """绘制四个园区的 2×2 算法训练对比图并保存。

    Args:
        park_results: 算法名到 ``(4, Episode)`` 数组的映射。
        output_path: 图片保存路径。
        max_episodes: 最多绘制的 Episode 数；传入 ``None`` 表示全部绘制。
        value_scale: 奖励/成本数值缩放倍数，与统一规划脚本默认保持为 100。
        show: 保存后是否调用 ``plt.show()`` 显示图像。
    """

    missing_algorithms = set(ALGORITHM_STYLES) - set(park_results)
    if missing_algorithms:
        raise ValueError(f"缺少算法结果: {sorted(missing_algorithms)}")
    if max_episodes is not None and max_episodes <= 0:
        raise ValueError("max_episodes 必须大于 0，或设置为 None")

    prepared_results: dict[str, np.ndarray] = {}
    for algorithm_name in ALGORITHM_STYLES:
        values = normalize_park_rewards(park_results[algorithm_name], algorithm_name)
        episode_slice = slice(None, max_episodes)
        prepared_results[algorithm_name] = values[:, episode_slice] * value_scale

    configure_chinese_font()
    fig, axes = plt.subplots(2, 2, figsize=(15, 10), dpi=150)
    flat_axes = axes.ravel()

    for park_index, ax in enumerate(flat_axes):
        current_park_series: dict[str, np.ndarray] = {}
        for algorithm_name, values in prepared_results.items():
            label, line_style, color = ALGORITHM_STYLES[algorithm_name]
            series = values[park_index]
            current_park_series[algorithm_name] = series
            episodes = np.arange(1, len(series) + 1)
            ax.plot(
                episodes,
                series,
                label=label,
                linestyle=line_style,
                color=color,
                linewidth=1.3,
                alpha=0.9,
            )

        ax.set_title(f"园区 {park_index + 1}", fontsize=14, fontweight="bold")
        ax.set_xlabel("Episode", fontsize=11)
        ax.set_ylabel("日平均费用", fontsize=11)
        ax.grid(True, linestyle="--", alpha=0.35)
        ax.tick_params(axis="both", labelsize=9)
        ax.ticklabel_format(axis="y", style="sci", scilimits=(0, 0), useMathText=True)
        for spine in ax.spines.values():
            spine.set_visible(True)
            spine.set_color("black")
            spine.set_linewidth(0.8)

        _add_zoom_inset(ax, current_park_series)

    legend_handles, legend_labels = flat_axes[0].get_legend_handles_labels()
    fig.legend(
        legend_handles,
        legend_labels,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.97),
        ncol=len(ALGORITHM_STYLES),
        frameon=True,
        fontsize=11,
    )
    fig.suptitle("四个园区不同算法训练结果对比", fontsize=17, fontweight="bold", y=0.995)
    fig.tight_layout(rect=(0, 0, 1, 0.93))

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=600, bbox_inches="tight")
    print(f"图片已保存至: {output_path}")
    print_final_stage_summary(prepared_results)

    if show:
        plt.show()
    else:
        plt.close(fig)


def parse_args() -> argparse.Namespace:
    """解析绘图脚本的命令行参数。"""

    parser = argparse.ArgumentParser(description="绘制四个园区的算法训练结果对比图")
    parser.add_argument(
        "--primary-result",
        type=Path,
        default=DEFAULT_PRIMARY_RESULT,
        help="包含 DSFA 和 FedAvg 的 result_arrays.npz 路径",
    )
    parser.add_argument(
        "--baseline-result",
        type=Path,
        default=DEFAULT_BASELINE_RESULT,
        help="包含 maddpg 和 iddpg 的 result_arrays.npz 路径",
    )
    parser.add_argument(
        "--episodes",
        type=int,
        default=1000,
        help="最多绘制的 Episode 数，默认 1000",
    )
    parser.add_argument(
        "--scale",
        type=float,
        default=100.0,
        help="结果数值缩放倍数，默认 100，与统一规划脚本一致",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT_PATH,
        help="输出图片路径",
    )
    parser.add_argument(
        "--no-show",
        action="store_true",
        help="只保存图片，不弹出绘图窗口",
    )
    return parser.parse_args()


def main() -> None:
    """加载训练结果，应用统一规划脚本的缩放规则并绘制四园区图。"""

    args = parse_args()
    park_results = load_park_results(args.primary_result, args.baseline_result)

    # 与统一规划脚本保持一致：FedAvg 曲线使用原有的 1.015 修正系数。
    # park_results["FedAvg"] = park_results["FedAvg"] * 1.015
    draw_four_park_results(
        park_results,
        output_path=args.output,
        max_episodes=args.episodes,
        value_scale=args.scale,
        show=not args.no_show,
    )


if __name__ == "__main__":
    try:
        main()
    except (FileNotFoundError, KeyError, ValueError, OSError) as exc:
        raise SystemExit(f"无法绘制四园区训练结果: {exc}") from exc
