"""自动读取一个或多个 NPZ 结果文件，并绘制其中的全部奖励曲线。"""

from __future__ import annotations

import os
from collections.abc import Iterable, Mapping
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib import font_manager
from mpl_toolkits.axes_grid1.inset_locator import inset_axes, mark_inset


PROJECT_ROOT = Path(__file__).resolve().parents[2]
OUTPUT_PATH = PROJECT_ROOT / "result_plot.png"
MAX_EPISODES = 1000
VALUE_SCALE = 100.0

# 只需在这里写入需要读取的 NPZ 文件；文件中的全部键会被自动绘制。
rew = [
    PROJECT_ROOT / "result" / "20261005" / "result_arrays.npz",
    PROJECT_ROOT / "result" / "20260404" / "result_arrays.npz",
]


def configure_chinese_font() -> None:
    """配置可用的中文字体，并确保负号可以正常显示。"""

    preferred_fonts = [
        Path(r"C:\Windows\Fonts\msyh.ttc"),
        Path(r"C:\Windows\Fonts\simhei.ttf"),
        Path("/System/Library/Fonts/PingFang.ttc"),
        Path("/System/Library/Fonts/STHeiti Light.ttc"),
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


def prepare_reward_series(
    values: np.ndarray,
    key: str,
    max_episodes: int | None = MAX_EPISODES,
    value_scale: float = VALUE_SCALE,
) -> np.ndarray:
    """将 NPZ 中的一项数据转换为可绘制的一维 Episode 序列。

    一维数组会直接使用；二维及更高维数组会把最后一维视为 Episode，
    其余维度求和。这样可兼容当前 ``(园区, Episode)`` 的结果格式。

    Args:
        values: 从 NPZ 文件读取的数组。
        key: 数组对应的键，仅用于生成明确的错误信息。
        max_episodes: 最多保留的 Episode 数，``None`` 表示不截断。
        value_scale: 绘图前应用的数值缩放倍数。

    Returns:
        经过聚合、截断和缩放的一维浮点数组。

    Raises:
        ValueError: 数据为空、不是数值数组或 ``max_episodes`` 非法时抛出。
    """

    if max_episodes is not None and max_episodes <= 0:
        raise ValueError("max_episodes 必须大于 0，或设置为 None")

    try:
        reward_values = np.asarray(values, dtype=float)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"键 {key!r} 不是可绘制的数值数组") from exc

    reward_values = np.squeeze(reward_values)
    if reward_values.ndim == 0 or reward_values.size == 0:
        raise ValueError(f"键 {key!r} 不包含有效的 Episode 序列")
    if reward_values.ndim > 1:
        reward_values = reward_values.sum(axis=tuple(range(reward_values.ndim - 1)))

    episode_slice = slice(None, max_episodes)
    return reward_values[episode_slice] * value_scale


def load_reward_data(
    result_paths: Iterable[str | os.PathLike[str]],
    max_episodes: int | None = MAX_EPISODES,
    value_scale: float = VALUE_SCALE,
) -> dict[str, np.ndarray]:
    """自动读取所有 NPZ 文件中的全部键并生成绘图数据。

    当不同文件包含同名键时，标签会自动追加文件父目录名，例如
    ``DSFA (20261005)``，避免后读取的数据覆盖先读取的数据。

    Args:
        result_paths: 一个或多个 NPZ 文件路径。
        max_episodes: 每条曲线最多读取的 Episode 数。
        value_scale: 每条曲线统一使用的缩放倍数。

    Returns:
        图例名称到一维奖励序列的映射，顺序与路径及 NPZ 键顺序一致。
    """

    paths = [Path(path).expanduser() for path in result_paths]
    if not paths:
        raise ValueError("rew 至少需要包含一个 NPZ 文件路径")

    missing_paths = [str(path) for path in paths if not path.is_file()]
    if missing_paths:
        raise FileNotFoundError("找不到结果文件: " + ", ".join(missing_paths))

    key_counts: dict[str, int] = {}
    for path in paths:
        with np.load(path, allow_pickle=False) as result_file:
            for key in result_file.files:
                key_counts[key] = key_counts.get(key, 0) + 1

    reward_data: dict[str, np.ndarray] = {}
    for path in paths:
        with np.load(path, allow_pickle=False) as result_file:
            print(f"读取 {path}，键: {result_file.files}")
            for key in result_file.files:
                label = key if key_counts[key] == 1 else f"{key} ({path.parent.name})"
                # 父目录也重名时继续编号，保证每一条曲线都不会被覆盖。
                unique_label = label
                label_index = 2
                while unique_label in reward_data:
                    unique_label = f"{label} #{label_index}"
                    label_index += 1
                reward_data[unique_label] = prepare_reward_series(
                    result_file[key],
                    key,
                    max_episodes=max_episodes,
                    value_scale=value_scale,
                )

    if not reward_data:
        raise ValueError("指定的 NPZ 文件中没有可绘制的键")
    return reward_data


def draw_result(
    rewards_record: Mapping[str, np.ndarray],
    output_path: Path = OUTPUT_PATH,
    show: bool = True,
) -> None:
    """绘制全部奖励曲线、末段局部放大图，并保存图片。"""

    if not rewards_record:
        raise ValueError("rewards_record 不能为空")

    configure_chinese_font()
    fig, ax = plt.subplots(figsize=(10, 6))
    line_styles = ["-", "--", "-.", ":"]
    color_map = plt.get_cmap("tab10")

    for index, (label, reward) in enumerate(rewards_record.items()):
        style = line_styles[index % len(line_styles)]
        color = color_map(index % color_map.N)
        episodes = np.arange(1, len(reward) + 1)
        ax.plot(
            episodes,
            reward,
            label=label,
            linestyle=style,
            color=color,
            linewidth=1.5,
            alpha=0.9,
        )

    ax.set_title("结果 (Result)", fontsize=14)
    ax.set_xlabel("Episode", fontsize=12)
    ax.set_ylabel("日平均费用", fontsize=12)
    ax.grid(True, linestyle="--", alpha=0.4)
    ax.legend(loc="best", frameon=True, shadow=True)

    max_length = max(len(reward) for reward in rewards_record.values())
    if max_length >= 5:
        zoom_start = int(max_length * 0.85)
        zoom_ax = inset_axes(
            ax,
            width="40%",
            height="30%",
            loc="center right",
            borderpad=2,
        )
        zoom_values: list[float] = []
        for index, reward in enumerate(rewards_record.values()):
            style = line_styles[index % len(line_styles)]
            color = color_map(index % color_map.N)
            episodes = np.arange(1, len(reward) + 1)
            zoom_ax.plot(episodes, reward, linestyle=style, color=color, linewidth=2)
            if len(reward) > zoom_start:
                finite_values = reward[zoom_start:][np.isfinite(reward[zoom_start:])]
                zoom_values.extend(finite_values.tolist())

        zoom_ax.set_xlim(zoom_start + 1, max_length)
        if zoom_values:
            y_min, y_max = min(zoom_values), max(zoom_values)
            value_range = y_max - y_min
            margin = (
                value_range * 0.1
                if value_range > 0
                else max(abs(y_min) * 0.02, 1.0)
            )
            zoom_ax.set_ylim(y_min - margin, y_max + margin)
        zoom_ax.grid(True, linestyle=":", alpha=0.5)
        mark_inset(ax, zoom_ax, loc1=2, loc2=4, fc="none", ec="0.5", linestyle="--")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    # inset_axes 与 tight_layout 不兼容，手动留边可避免保存时产生布局警告。
    fig.subplots_adjust(left=0.1, right=0.96, bottom=0.11, top=0.92)
    fig.savefig(output_path, dpi=600, bbox_inches="tight")
    print(f"图片已保存至: {output_path}")

    if show:
        plt.show()
    else:
        plt.close(fig)


def main() -> None:
    """读取 ``rew`` 中的全部文件和键，然后生成结果图。"""
    rew = [
        'D:\\ITE\\result\\20260404\\result_arrays.npz',
        'D:\\ITE\\result\\20261005\\result_arrays.npz'
    ]
    reward_data = load_reward_data(rew)
    draw_result(reward_data)


if __name__ == "__main__":
    try:
        main()
    except (FileNotFoundError, OSError, ValueError) as exc:
        raise SystemExit(f"无法加载数据或绘图: {exc}") from exc
