"""绘制不同 K 值下四个园区的运行费用。"""

from __future__ import annotations

import argparse
import csv
from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


DEFAULT_RESULT_DIR = Path("result/communication/20261119_201339_886057")
K_COLORS = ("#C9DEE8", "#8FB9CB", "#5593AC", "#23627F")


def load_results(csv_path: Path) -> list[dict[str, float | int | str]]:
    """读取并校验绘图数据。

    Args:
        csv_path: 实验入口生成的 plot_data.csv 路径。

    Returns:
        按 K 和随机种子排序的记录；各园区成本转换为千元。

    Raises:
        ValueError: 数据为空、关键字段缺失或评估时长不一致时抛出。
    """
    required = {"strategy", "neighbors", "seed", "evaluation_start_hour",
                "evaluation_hours",
                *(f"agent_{index}_cost_cny" for index in range(4))}
    with csv_path.open(newline="", encoding="utf-8-sig") as stream:
        reader = csv.DictReader(stream)
        missing = required.difference(reader.fieldnames or ())
        if missing:
            raise ValueError(f"绘图数据缺少字段: {sorted(missing)}")
        records = []
        for raw in reader:
            if raw["strategy"] != "nearest":
                raise ValueError("K 值图只比较 nearest 策略，请单独筛选数据")
            row: dict[str, float | int | str] = {
                "k": int(raw["neighbors"]),
                "seed": int(raw["seed"]),
                "evaluation_start_hour": int(raw["evaluation_start_hour"]),
                "evaluation_date": raw.get("evaluation_date") or "",
                "evaluation_hours": int(raw["evaluation_hours"]),
            }
            for index in range(4):
                row[f"ies_{index + 1}_cost_kcny"] = float(raw[f"agent_{index}_cost_cny"]) / 1_000
            if not all(np.isfinite(value) for value in row.values() if not isinstance(value, str)):
                raise ValueError(f"K={row['k']} 的绘图数据包含非有限数值")
            records.append(row)
    if not records:
        raise ValueError(f"绘图数据为空: {csv_path}")
    if {int(row["evaluation_hours"]) for row in records} != {24}:
        raise ValueError("成本比较要求每组都完成相同的 24 小时评估")
    if len({int(row["evaluation_start_hour"]) for row in records}) != 1:
        raise ValueError("同一张图片中的评估起点必须一致")
    if len({str(row["evaluation_date"]) for row in records}) != 1:
        raise ValueError("同一张图片中的评估日期必须一致")
    keys = [(int(row["k"]), int(row["seed"])) for row in records]
    if len(set(keys)) != len(keys):
        raise ValueError("同一个 K 和 seed 出现重复记录")
    return sorted(records, key=lambda row: (row["k"], row["seed"]))


def draw_park_costs(records: list[dict[str, float | int | str]], output_dir: Path) -> Path:
    """为每个园区绘制四组 K 值的横向柱状图，并导出 PNG。

    Args:
        records: load_results 返回的单一选择策略实验记录。
        output_dir: 图像保存目录。

    Returns:
        PNG 路径。单 seed 显示原值；多 seed 显示均值和标准差误差线。
    """
    grouped: dict[int, list[dict[str, float | int | str]]] = defaultdict(list)
    for row in records:
        grouped[int(row["k"])].append(row)
    k_values = sorted(grouped)
    if set(k_values) != {0, 1, 2, 3}:
        raise ValueError(f"需要 K=0、1、2、3 四组，实际为 {k_values}")

    plt.rcParams.update({
        "font.family": "DejaVu Sans",
        "font.size": 10,
    })
    fig, ax = plt.subplots(figsize=(8, 5.2), constrained_layout=True)
    group_positions = np.arange(4, dtype=float)
    bar_height = 0.18
    largest_value = 0.0
    for k in k_values:
        means = []
        deviations = []
        for index in range(4):
            values = np.asarray([row[f"ies_{index + 1}_cost_kcny"] for row in grouped[k]], dtype=float)
            means.append(float(values.mean()))
            deviations.append(float(values.std(ddof=1)) if len(values) > 1 else 0.0)
        largest_value = max(largest_value, *(mean + deviation for mean, deviation in zip(means, deviations)))
        positions = group_positions + (k - 1.5) * bar_height
        ax.barh(positions, means, height=bar_height * 0.88,
                xerr=deviations if any(deviations) else None,
                color=K_COLORS[k], edgecolor="white", linewidth=0.6,
                capsize=2.5, label=f"K = {k}", zorder=3)
        for position, mean in zip(positions, means):
            ax.text(mean + 2, position, f"{mean:.0f}", va="center", ha="left",
                    fontsize=8.5, color="#34434D")
    ax.set_yticks(group_positions, [f"IES {index}" for index in range(1, 5)])
    ax.invert_yaxis()
    ax.set_xlim(0, largest_value + 25)
    ax.set_xlabel("Operating cost (10³ CNY)")
    ax.set_ylabel("Park")
    ax.grid(axis="x", color="#E4E9EE", linewidth=0.8, zorder=0)
    ax.spines[["top", "right", "left"]].set_visible(False)
    ax.tick_params(axis="y", length=0, pad=10)
    ax.legend(ncol=4, loc="lower center", bbox_to_anchor=(0.5, 1.01), frameon=False)
    evaluation_start_hour = int(records[0]["evaluation_start_hour"])
    evaluation_date = str(records[0]["evaluation_date"])
    if evaluation_date:
        period = f"training day ({evaluation_date})" if evaluation_start_hour == 0 else f"evaluation day ({evaluation_date})"
    elif evaluation_start_hour == 0:
        period = "training day (hours 0–23)"
    else:
        period = f"evaluation hours {evaluation_start_hour}–{evaluation_start_hour + 23}"
    ax.set_title(f"Park operating cost on {period}",
                 loc="left", fontweight="bold", pad=44)

    output_dir.mkdir(parents=True, exist_ok=True)
    png_path = output_dir / "k_park_costs.png"
    fig.savefig(png_path, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return png_path


def main() -> None:
    """从指定批次的 CSV 生成园区费用横向柱状图。"""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("result_dir", nargs="?", type=Path, default=DEFAULT_RESULT_DIR)
    parser.add_argument("--output-dir", type=Path, help="默认保存到结果目录的 figures 子目录")
    args = parser.parse_args()
    result_dir = args.result_dir.resolve()
    output_dir = args.output_dir.resolve() if args.output_dir else result_dir / "figures"
    png_path = draw_park_costs(
        load_results(result_dir / "plot_data.csv"), output_dir
    )
    print(f"图像已保存: {png_path}")


if __name__ == "__main__":
    main()
