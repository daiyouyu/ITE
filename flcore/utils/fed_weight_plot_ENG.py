import os
import traceback

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.axes import Axes
from matplotlib import font_manager

plt.style.use('seaborn-v0_8-whitegrid')
# ===================== 真正有效的中文配置 =====================
plt.rcParams['axes.unicode_minus'] = False  # 正常显示负号

# 按优先级排列的中文字体列表（把 Windows 常见字体放前面）
FONT_LIST = [
    "Microsoft YaHei",  # 微软雅黑 (Windows)
    "SimHei",  # 黑体 (Windows)
    "PingFang SC",  # 苹方 (macOS)
    "WenQuanYi Micro Hei",  # 文泉驿微米黑 (Linux)
    "Noto Sans CJK SC",  # 思源黑体
]
OUTPUT_RESULT_PATH = "D:\\ITE\\result\\fed_weights"
SMOOTHING_WINDOW = 20
HIGHLIGHT_IES_INDEX = 3

# 获取系统中所有真实安装的字体名称
installed_fonts = [f.name for f in font_manager.fontManager.ttflist]

# 寻找第一个存在的字体并设置
selected_font = None
for font in FONT_LIST:
    if font in installed_fonts:
        selected_font = font
        break

if selected_font:
    plt.rcParams['font.family'] = 'sans-serif'
    plt.rcParams['font.sans-serif'] = [selected_font]
    print(f"[*] Matplotlib 中文配置成功，使用字体: {selected_font}")
else:
    print("[!] 警告: 未在系统中检测到常见的中文字体，图表中文可能显示为方块。")


def simple_moving_average(
    data: np.ndarray,
    window_size: int = SMOOTHING_WINDOW,
) -> tuple[np.ndarray, np.ndarray]:
    """返回滑动均值及窗口中心坐标；不足一个窗口时保留原始数据。"""
    if window_size < 1:
        raise ValueError("平滑窗口必须为正整数")
    if len(data) < window_size:
        return data, np.arange(len(data))
    smoothed = np.convolve(data, np.ones(window_size) / window_size, mode='valid')
    x_axis = np.arange(len(smoothed)) + (window_size - 1) / 2
    return smoothed, x_axis


def calculate_fluctuation_band(
    values: np.ndarray,
    window_size: int = SMOOTHING_WINDOW,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """返回波动带的横坐标、平滑中心值及局部残差标准差。

    values 为一维原始权重，window_size 同时用于趋势平滑和残差统计。
    先减去对齐到原始轮次的平滑趋势，再计算完整窗口内的总体标准差。
    两端不外推；数据不足以构成完整残差窗口时返回三个空数组。
    非有限数据、非一维输入或窗口小于 2 时抛出 ValueError。
    """
    values = np.asarray(values, dtype=float)
    if values.ndim != 1 or not np.all(np.isfinite(values)):
        raise ValueError("波动带要求有限的一维权重数据")
    if window_size < 2:
        raise ValueError("波动带统计窗口必须大于或等于 2")
    if len(values) < window_size:
        return np.array([]), np.array([]), np.array([])
    trend, trend_x = simple_moving_average(values, window_size)
    # 偶数窗口的趋势位于半整数轮次，插值对齐后再相减，避免错位引入虚假波动。
    residual_x = np.arange(int(np.ceil(trend_x[0])), int(np.floor(trend_x[-1])) + 1)
    if len(residual_x) < window_size:
        return np.array([]), np.array([]), np.array([])
    residuals = values[residual_x] - np.interp(residual_x, trend_x, trend) + 1.5
    windows = np.lib.stride_tricks.sliding_window_view(residuals, window_size)
    standard_deviation = windows.std(axis=1, ddof=0) 
    band_x = residual_x[:len(standard_deviation)] + (window_size - 1) / 2
    band_center = np.interp(band_x, trend_x, trend)
    return band_x, band_center, standard_deviation


def inspect_npy_file(file_path: str, ax: Axes, title: str) -> bool:
    """
    检查权重文件，并在指定子图中绘制一维、二维或三维方阵的权重变化。

    Args:
        file_path: .npy 文件路径。
        ax: 用于绘图的子图坐标轴。
        title: 子图标题。

    Returns:
        成功绘图时返回 True；文件缺失、数据维度不支持或绘图失败时返回 False。
    """
    print(f"--- 开始检查文件: {os.path.basename(file_path)} ---")

    # 1. 检查文件是否存在
    if not os.path.exists(file_path):
        print(f"错误: 文件未找到 '{file_path}'")
        print("--- 检查结束 ---")
        return False

    try:
        # 2. 加载 .npy 文件
        # allow_pickle=True 对于加载包含对象（如字典）的数组是必需的
        data = np.load(file_path, allow_pickle=True)

        print(f"文件路径: {file_path}")
        print("-" * 40)

        # 3. 分析并打印文件内容信息
        print(f"加载后的数据类型: {type(data)}")

        # 检查是否为包含字典的 NumPy 数组（联邦学习权重常用格式）
        if data.ndim == 0 and data.item() and isinstance(data.item(), dict):
            print("文件内容: 这是一个包含模型权重的字典。")
            model_weights = data.item()
            print(f"字典中包含 {len(model_weights)} 个层/键。")
            for layer_name, weights in model_weights.items():
                if isinstance(weights, np.ndarray):
                    print(f"  - 层/键 '{layer_name}':")
                    print(f"    - 形状 (Shape): {weights.shape}")
                    print(f"    - 数据类型 (Dtype): {weights.dtype}")
                else:
                    print(f"  - 层/键 '{layer_name}': 类型为 {type(weights)}, 不是 NumPy 数组。")

        # 检查是否为常规的 NumPy 数组
        elif isinstance(data, np.ndarray):
            print("文件内容: 这是一个常规的 NumPy 数组。")
            print(f"  - 形状 (Shape): {data.shape}")
            print(f"  - 数据类型 (Dtype): {data.dtype}")
            # 打印数组的前几个元素作为预览
            print(f"  - 内容预览 (前5个元素): {data.flatten()[:5]}...")

            # 新增：对3D权重矩阵提取对角线元素
            plot_data = data
            if data.ndim == 3 and data.shape[1] == data.shape[2]:
                print(f"\n检测到3D权重矩阵 (shape: {data.shape})，提取对角线元素进行绘图。")
                # 提取每个 (M, M) 矩阵的对角线, 结果是 (N, M)
                plot_data = np.diagonal(data, axis1=1, axis2=2)
                print(f"  - 提取后用于绘图的数据形状 (Shape): {plot_data.shape}")

            # 使用 matplotlib 绘制折线图
            if plot_data.ndim in [1, 2]:
                print("\n正在生成折线图...")
                try:
                    # --- 美化改进 ---
                    # 定义一个好看的颜色循环
                    custom_colors_rgb_255 = [
                        (31, 119, 180),
                        (255, 127, 14),
                        (44, 160, 44),
                        (214, 39, 40)
                    ]
                    colors = [(r / 255., g / 255., b / 255.) for r, g, b in custom_colors_rgb_255]

                    if plot_data.ndim == 2:
                        num_lines = plot_data.shape[1]
                        for i in range(num_lines):
                            raw_data = plot_data[:, i]
                            current_color = colors[i % len(colors)]
                            highlighted = i == HIGHLIGHT_IES_INDEX
                            ax.plot(raw_data, color=current_color,
                                    alpha=0.3, linewidth=1.0)
                            smoothed_data, smoothed_x = simple_moving_average(raw_data)
                            if highlighted:
                                band_x, band_center, deviation = calculate_fluctuation_band(raw_data)
                                # 两图使用相同窗口和透明度，仅展示真实局部波动，不表示置信区间。
                                ax.fill_between(band_x, band_center - deviation, band_center + deviation,
                                                color=current_color, alpha=0.22,
                                                linewidth=0, zorder=3)
                            ax.plot(smoothed_x, smoothed_data, color=current_color, label=f'IES {i + 1}',
                                    linewidth=3.5 if highlighted else 2.5,
                                    zorder=5 if highlighted else 2)

                    else:
                        raw_data = plot_data
                        ax.plot(raw_data, color=colors[0], alpha=0.3, linewidth=1.0)
                        smoothed_data, smoothed_x = simple_moving_average(raw_data)
                        ax.plot(smoothed_x, smoothed_data, color=colors[0], label='IES 1', linewidth=2.5)

                    ax.set_title(title, fontsize=22, fontweight='bold', pad=18)
                    ax.set_xlabel("Episode", fontsize=20, labelpad=10)
                    ax.set_ylabel(r"$W_{ii}$",fontsize=20, labelpad=14)
                    ax.tick_params(axis='both', labelsize=17)
                    # 原下限 0.3 会截掉部分原始权重；两图统一范围以便直接比较。
                    ax.set_ylim(0.25, 0.8)
                    ax.set_xlim(0, max(len(plot_data) - 1, 1))
                    ax.spines['top'].set_visible(False)
                    ax.spines['right'].set_visible(False)
                    return True

                except Exception as plot_e:
                    print(f"绘制图形时发生错误: {plot_e}")
                    return False
            else:
                print(f"\n数组维度 ({data.ndim}) 暂不支持绘图。")

        else:
            print("文件内容: 这是一个未知的数据类型。")
            print("尝试直接打印内容...")
            print(data)

        print("-" * 40)
        return False

    except Exception as e:
        error_trace = traceback.format_exc()

        print(f"读取或解析文件时发生错误: {e}")
        print(error_trace)
        return False
    finally:
        print("--- 检查结束 ---")


if __name__ == "__main__":
    test_file = 'D:\\ITE\\result\\20260404\\fed_weights_DSFA_194147.npy'
    comparsion_file = 'D:\\ITE\\result\\20261005\\fed_weights_AllDSFA_004613.npy'

    fig, axes = plt.subplots(1, 2, figsize=(20, 8.5), sharey=True)
    test_ok = inspect_npy_file(test_file, axes[0], "(a) DSFA: Actor-only aggregation")
    comparison_ok = inspect_npy_file(
        comparsion_file, axes[1], "(b) AllDSFA: Actor–Critic aggregation"
    )

    if test_ok and comparison_ok:
        fig.subplots_adjust(left=0.09, right=0.98, top=0.88, bottom=0.28, wspace=0.22)
        handles, labels = axes[0].get_legend_handles_labels()
        fig.legend(handles, labels, loc='lower center', ncol=len(labels),
                   bbox_to_anchor=(0.5, 0.10), frameon=False, fontsize=24)
        fig.text(0.5, 0.095,
                 r"$W_{ii}$: contribution of IES $i$'s own model to its aggregated model; higher values mean more local-model retention.",
                 ha='center', va='center', fontsize=24)
        os.makedirs(OUTPUT_RESULT_PATH, exist_ok=True)
        output_path = os.path.join(OUTPUT_RESULT_PATH, "fed_weights_DSFA_comparison_ENG.png")
        fig.savefig(output_path, dpi=600, bbox_inches="tight")
        print(f"绘图完成，路径：{output_path}")
        plt.show()
    else:
        plt.close(fig)
        print("绘图失败：请检查两个权重文件及其数据维度。")
