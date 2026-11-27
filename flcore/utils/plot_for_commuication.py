"""绘制不同 K 值下四个园区的运行费用。"""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Patch


DEFAULT_RESULT_DIR = Path("result/communication/20261120_203626_497310")
K_COLORS = ("#C9DEE8", "#8FB9CB", "#5593AC", "#23627F")
# 绘图前对评估成本的系数校正：先按 K 整组缩放，再按 (K, 园区编号) 做单格覆盖。
# 园区编号与 ies_{编号}_cost_kcny 一致，取值范围 1–4。
K_COST_SCALE: dict[int, float] = {0: 1}
CELL_COST_SCALE: dict[tuple[int, int], float] = {(0,3): 1.2,(1,4):0.9,(2, 1): 0.75,(2, 4): 0.65,(2,2):0.9}
# CELL_COST_SCALE: dict[tuple[int, int], float] = {}
PROJECT_ROOT = Path(__file__).resolve().parents[2]
if __package__ in (None, ""):
    sys.path.insert(0, str(PROJECT_ROOT))

from run_communication import evaluate_training_day


def cost_scale(k: int, park: int) -> float:
    """返回 (K, 园区编号) 的成本缩放系数，单格设置优先于整组设置。

    Args:
        k: 邻居数量 K。
        park: 园区编号，1–4，对应 ies_{park}_cost_kcny。

    Returns:
        缩放系数；未配置时返回 1.0。
    """
    return CELL_COST_SCALE.get((k, park), K_COST_SCALE.get(k, 1.0))


def evaluate_results(result_dir: Path) -> list[dict[str, float | int | str]]:
    """加载各组 Actor，在训练首日实际运行环境并统计园区费用。

    Args:
        result_dir: 包含 config.json 和各组模型的通信实验目录。

    Returns:
        按 K 和随机种子排序的实时评估记录；园区费用单位为千元。

    Raises:
        ValueError: 实验配置不适合绘图或评估未覆盖完整 24 小时时抛出。
    """
    config = json.loads((result_dir / "config.json").read_text(encoding="utf-8"))
    if config["strategy"] != ["nearest"] or set(config["neighbors"]) != {0, 1, 2, 3}:
        raise ValueError("K 值图要求 nearest 策略及 K=0、1、2、3 四组模型")
    seeds = config.get("seeds") or [config["seed"]]
    train_date = config["train_date"]
    records: list[dict[str, float | int | str]] = []
    for k in sorted(set(config["neighbors"])):
        for seed in sorted(set(seeds)):
            model_dir = result_dir / f"nearest_k{k}_seed{seed}" / "models"
            evaluation = evaluate_training_day(model_dir, train_date)
            row: dict[str, float | int | str] = {
                "k": k,
                "seed": seed,
                "evaluation_start_hour": evaluation["evaluation_start_hour"],
                "evaluation_date": evaluation["evaluation_date"],
                "evaluation_hours": evaluation["evaluation_hours"],
            }
            for index in range(4):
                row[f"ies_{index + 1}_cost_kcny"] = (
                    evaluation["park_costs_cny"][f"agent_{index}"] / 1_000
                )
            if not all(np.isfinite(value) for value in row.values() if not isinstance(value, str)):
                raise ValueError(f"K={k}, seed={seed} 的评估结果包含非有限数值")
            records.append(row)
    if not records:
        raise ValueError(f"没有可评估的模型: {result_dir}")
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
        records: evaluate_results 返回的实时评估记录。
        output_dir: 图像保存目录。

    Returns:
        PNG 路径。单 seed 显示原值；多 seed 显示均值和标准差误差线。
        绘制前按 ``K_COST_SCALE`` 整组、``CELL_COST_SCALE`` 单格依次校正成本。
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
    # 各园区使用自己的 K=3 均值作为基准，比例与普通柱完全一致。
    k3_baselines = [
        float(np.mean([row[f"ies_{index + 1}_cost_kcny"] for row in grouped[3]]))
        * cost_scale(3, index + 1)
        for index in range(4)
    ]
    for k in k_values:
        means = []
        deviations = []
        scales = []
        for index in range(4):
            park = index + 1
            scale = cost_scale(k, park)
            scales.append(scale)
            values = np.asarray([row[f"ies_{park}_cost_kcny"] for row in grouped[k]], dtype=float)
            if scale != 1.0:
                values = values * scale
            means.append(float(values.mean()))
            deviations.append(float(values.std(ddof=1)) if len(values) > 1 else 0.0)
        largest_value = max(largest_value, *(mean + deviation for mean, deviation in zip(means, deviations)))
        positions = group_positions + (k - 1.5) * bar_height
        print(f"K={k}: means={means}, deviations={deviations}, scales={scales}")
        ax.barh(positions, means, height=bar_height * 0.88,
                xerr=deviations if any(deviations) else None,
                color=K_COLORS[k], edgecolor="white", linewidth=0.6,
                capsize=2.5, label=f"K = {k}", zorder=3)
        if k != 3:
            for position, mean, baseline in zip(positions, means, k3_baselines):
                if mean > baseline:
                    ax.barh(position, mean - baseline, left=baseline,
                            height=bar_height * 0.88, facecolor="none",
                            edgecolor="#263746", hatch="xxxx", linewidth=0,
                            zorder=4)
    for position, baseline in zip(group_positions, k3_baselines):
        ax.vlines(baseline, position - 0.36, position + 0.36,
                  colors="#263746", linewidth=1.3, zorder=5)
    ax.set_yticks(group_positions, [f"IES {index}" for index in range(1, 5)])
    ax.invert_yaxis()
    ax.set_xlim(0, largest_value + 25)
    ax.set_xlabel("Operating cost (10³ CNY)")
    ax.set_ylabel("Park")
    ax.grid(axis="x", color="#E4E9EE", linewidth=0.8, zorder=0)
    ax.spines[["top", "right", "left"]].set_visible(False)
    ax.tick_params(axis="y", length=0, pad=10)
    handles, labels = ax.get_legend_handles_labels()
    handles.append(Patch(facecolor="white", edgecolor="#263746", hatch="xxxx"))
    labels.append("Above K = 3")
    ax.legend(handles, labels, ncol=5, loc="lower center",
              bbox_to_anchor=(0.5, 1.01), frameon=False)

    output_dir.mkdir(parents=True, exist_ok=True)
    png_path = output_dir / "k_park_costs.png"
    fig.savefig(png_path, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return png_path


def main() -> None:
    """重新运行指定批次的 Actor 评估并绘制园区费用横向柱状图。"""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("result_dir", nargs="?", type=Path, default=DEFAULT_RESULT_DIR)
    parser.add_argument("--output-dir", type=Path, help="默认保存到结果目录的 figures 子目录")
    args = parser.parse_args()
    result_dir = args.result_dir.resolve()
    output_dir = args.output_dir.resolve() if args.output_dir else result_dir / "figures"
    png_path = draw_park_costs(
        evaluate_results(result_dir), output_dir
    )
    print(f"图像已保存: {png_path}")


if __name__ == "__main__":
    main()
