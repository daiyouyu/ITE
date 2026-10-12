import os
import re
import traceback
import matplotlib
# matplotlib.use("TkAgg")

import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
from matplotlib import font_manager


# ============================================================
# Matplotlib 中文配置
# ============================================================

plt.style.use("seaborn-v0_8-whitegrid")

plt.rcParams["axes.unicode_minus"] = False

FONT_LIST = [
    "Microsoft YaHei",
    "SimHei",
    "PingFang SC",
    "WenQuanYi Micro Hei",
    "Noto Sans CJK SC",
]

installed_fonts = [
    f.name for f in font_manager.fontManager.ttflist
]

selected_font = None

for font in FONT_LIST:
    if font in installed_fonts:
        selected_font = font
        break

if selected_font:
    plt.rcParams["font.family"] = "sans-serif"
    plt.rcParams["font.sans-serif"] = [selected_font]
    print(
        f"[*] Matplotlib 中文配置成功，"
        f"使用字体: {selected_font}"
    )
else:
    print(
        "[!] 未检测到常见中文字体，"
        "图表中文可能显示为方块。"
    )


def simple_moving_average(data, window_size=20):
    """
    简单移动平均。
    """

    if len(data) < window_size:
        return data, np.arange(len(data))

    smoothed = np.convolve(
        data,
        np.ones(window_size) / window_size,
        mode="valid",
    )

    x_axis = (
        np.arange(len(smoothed))
        + (window_size - 1) // 2
    )

    return smoothed, x_axis


def mirror_weights(
    weights: np.ndarray,
    mirror_index: int,
    dim1_index: int,
):
    """
    在内存中修改指定智能体的权重，不保存文件。

    数据格式：
        weights.shape = (N, 4, 4)

    镜像基准：
        c = weights[mirror_index, dim1_index, :]

    公式：
        x_new = 1 - |c - x_old|

    修改范围：
        weights[0:mirror_index, dim1_index, :]

    mirror_index 以及之后的数据保持不变。
    """

    if weights.ndim != 3:
        raise ValueError(
            f"要求数据为 3D，实际 shape={weights.shape}"
        )

    if not 0 <= mirror_index < weights.shape[0]:
        raise ValueError(
            f"mirror_index={mirror_index} 超出范围，"
            f"有效范围为 0 ~ {weights.shape[0] - 1}"
        )

    if not 0 <= dim1_index < weights.shape[1]:
        raise ValueError(
            f"dim1_index={dim1_index} 超出范围，"
            f"有效范围为 0 ~ {weights.shape[1] - 1}"
        )

    # 保留原始数据
    modified_weights = weights.copy()

    # --------------------------------------------------------
    # mirror_index 对应的 4 个权重作为 c
    #
    # 例如：
    # c = weights[200, 3, :]
    #
    # shape = (4,)
    # --------------------------------------------------------

    c = weights[
        mirror_index,
        dim1_index,
        :
    ].copy()

    # --------------------------------------------------------
    # x_new = 1 - |c - x_old|
    #
    # 只修改 mirror_index 之前的数据
    # --------------------------------------------------------

    modified_weights[
        :mirror_index,
        dim1_index,
        :
    ] = (
        weights[
            :mirror_index,
            dim1_index,
            :
        ] * 0.8

    )

    return modified_weights, c


def print_weight_info(
    original_weights,
    modified_weights,
    mirror_index,
    dim1_index,
    c,
):
    """
    打印修改前后的权重信息。
    """

    print("\n" + "=" * 60)
    print("数据基本信息")
    print("=" * 60)

    print(
        f"原始 shape: {original_weights.shape}"
    )

    print(
        f"mirror_index: {mirror_index}"
    )

    print(
        f"dim1_index: {dim1_index}"
    )

    print(
        f"c = weights[{mirror_index}, "
        f"{dim1_index}, :]："
    )

    print(c)

    # --------------------------------------------------------
    # 查看镜像点附近的数据
    # --------------------------------------------------------

    start_index = max(
        0,
        mirror_index - 5,
    )

    end_index = min(
        original_weights.shape[0],
        mirror_index + 6,
    )

    print("\n" + "=" * 60)
    print(
        f"镜像点附近数据 "
        f"[{start_index}:{end_index}]"
    )
    print("=" * 60)

    for i in range(
        start_index,
        end_index,
    ):
        print(
            f"\nEpisode {i}"
        )

        print(
            "原始：",
            original_weights[
                i,
                dim1_index,
                :
            ],
        )

        print(
            "修改：",
            modified_weights[
                i,
                dim1_index,
                :
            ],
        )

    # --------------------------------------------------------
    # 查看前几个数据
    # --------------------------------------------------------

    print("\n" + "=" * 60)
    print("前 10 个数据对比")
    print("=" * 60)

    for i in range(
        min(10, mirror_index)
    ):
        print(
            f"{i:4d}: "
            f"{original_weights[i, dim1_index, :]} "
            f"-> "
            f"{modified_weights[i, dim1_index, :]}"
        )


def plot_weights(
    original_weights,
    modified_weights,
    mirror_index,
    dim1_index,
):
    """
    绘制修改前后权重曲线。

    因为数据 shape=(N, 4, 4)，
    固定 dim1_index 后：

        weights[:, dim1_index, :]

    shape 为：

        (N, 4)

    因此会绘制 4 条权重曲线。
    """

    original_data = original_weights[
        :,
        dim1_index,
        :
    ]

    modified_data = modified_weights[
        :,
        dim1_index,
        :
    ]

    print("\n绘图数据 shape：")
    print(
        f"original_data: {original_data.shape}"
    )
    print(
        f"modified_data: {modified_data.shape}"
    )

    # ========================================================
    # 图1：修改后的完整曲线
    # ========================================================

    fig, ax = plt.subplots(
        figsize=(14, 8)
    )

    for i in range(
        modified_data.shape[1]
    ):
        raw_data = modified_data[:, i]

        # 原始数据淡线
        ax.plot(
            raw_data,
            alpha=0.25,
            linewidth=1.0,
        )

        # 平滑数据
        smoothed_data, smoothed_x = (
            simple_moving_average(
                raw_data,
                window_size=20,
            )
        )

        ax.plot(
            smoothed_x,
            smoothed_data,
            linewidth=2.2,
            label=f"Weight {i + 1}",
        )

    # 标记 mirror_index
    ax.axvline(
        x=mirror_index,
        linestyle="--",
        linewidth=2,
        label=f"mirror index = {mirror_index}",
    )

    ax.set_title(
        f"修改后的联邦权重 "
        f"(Agent {dim1_index + 1})",
        fontsize=16,
        fontweight="bold",
        pad=20,
    )

    ax.set_xlabel(
        "Episode",
        fontsize=12,
    )

    ax.set_ylabel(
        "权重值",
        fontsize=12,
    )

    ax.legend(
        frameon=True,
        shadow=True,
        loc="best",
    )

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    fig.tight_layout()

    # ========================================================
    # 图2：修改前后直接对比
    # ========================================================

    fig2, ax2 = plt.subplots(
        figsize=(14, 8)
    )

    for i in range(
        original_data.shape[1]
    ):
        # 修改前
        original_smooth, original_x = (
            simple_moving_average(
                original_data[:, i],
                window_size=20,
            )
        )

        ax2.plot(
            original_x,
            original_smooth,
            linestyle="--",
            alpha=0.65,
            linewidth=1.5,
            label=f"原始 Weight {i + 1}",
        )

        # 修改后
        modified_smooth, modified_x = (
            simple_moving_average(
                modified_data[:, i],
                window_size=20,
            )
        )

        ax2.plot(
            modified_x,
            modified_smooth,
            linewidth=2.2,
            label=f"修改 Weight {i + 1}",
        )

    ax2.axvline(
        x=mirror_index,
        linestyle="--",
        linewidth=2,
        label=f"mirror index = {mirror_index}",
    )

    ax2.set_title(
        f"联邦权重修改前后对比 "
        f"(Agent {dim1_index + 1})",
        fontsize=16,
        fontweight="bold",
        pad=20,
    )

    ax2.set_xlabel(
        "Episode",
        fontsize=12,
    )

    ax2.set_ylabel(
        "权重值",
        fontsize=12,
    )

    ax2.legend(
        frameon=True,
        shadow=True,
        loc="best",
        ncol=2,
    )

    ax2.spines["top"].set_visible(False)
    ax2.spines["right"].set_visible(False)

    fig2.tight_layout()
    fig.savefig("test.png", dpi=600, bbox_inches="tight")

    # 只显示，不保存
    plt.show()


def inspect_and_modify(
    file_path: str,
    mirror_index: int,
    dim1_index: int,
):
    """
    读取 -> 修改 -> 打印 -> 画图。

    不保存修改后的 npy 文件。
    """

    print(
        f"--- 开始检查文件: "
        f"{os.path.basename(file_path)} ---"
    )

    if not os.path.exists(file_path):
        print(
            f"错误: 文件未找到 '{file_path}'"
        )
        return

    try:

        # ====================================================
        # 读取数据
        # ====================================================

        weights = np.load(
            file_path,
            allow_pickle=True,
        )

        print(
            f"文件路径: {file_path}"
        )

        print(
            f"数据类型: {type(weights)}"
        )

        print(
            f"数据 shape: {weights.shape}"
        )

        print(
            f"数据 dtype: {weights.dtype}"
        )

        if weights.ndim != 3:
            raise ValueError(
                f"当前脚本要求权重 shape 为 3D，"
                f"实际为 {weights.shape}"
            )

        # ====================================================
        # 修改
        # ====================================================

        modified_weights, c = mirror_weights(
            weights=weights,
            mirror_index=mirror_index,
            dim1_index=dim1_index,
        )
        save_path =  Path(file_path)
        new_name = re.sub(
            r"_(\d+)\.npy$",
            r"_modify_\1.npy",
            save_path.name
        )

        output_path = save_path.with_name(
            new_name
        )

        np.save(
            output_path,
            modified_weights
        )
        # ====================================================
        # 打印
        # ====================================================

        print_weight_info(
            original_weights=weights,
            modified_weights=modified_weights,
            mirror_index=mirror_index,
            dim1_index=dim1_index,
            c=c,
        )

        # ====================================================
        # 画图
        # ====================================================

        plot_weights(
            original_weights=weights,
            modified_weights=modified_weights,
            mirror_index=mirror_index,
            dim1_index=dim1_index,
        )

        print("\n注意：")
        print(
            "本次修改只存在于内存中的 "
            "modified_weights，"
            "原始 npy 文件没有发生任何修改。"
        )

    except Exception as e:

        print(
            f"读取或处理文件时发生错误: {e}"
        )

        traceback.print_exc()

    finally:

        print(
            "--- 检查结束 ---"
        )


def main():

    target_file = (
        r"D:\\ITE\\result\\20261005"
        r"\\fed_weights_AllDSFA_004613.npy"
    )

    inspect_and_modify(
        file_path=target_file,

        # 使用 weights[200, 3, :]
        # 作为 c
        mirror_index=200,

        # shape[1] 中索引 3
        # 即第 4 个智能体
        dim1_index=3,
    )


if __name__ == "__main__":
    main()