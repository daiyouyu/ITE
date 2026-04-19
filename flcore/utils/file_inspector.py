import numpy as np
import os
import matplotlib.pyplot as plt
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


def inspect_npy_file(file_path: str):
    """
    加载并检查一个 .npy 文件，打印其类型、形状、数据类型和内容摘要。
    这个函数特别优化了对包含字典的 .npy 文件的检查（在联邦学习中很常见）。
    新增了对常规 NumPy 数组（1D或2D）的可视化绘图功能。

    Args:
        file_path (str): .npy 文件的路径。
    """
    print(f"--- 开始检查文件: {os.path.basename(file_path)} ---")

    # 1. 检查文件是否存在
    if not os.path.exists(file_path):
        print(f"错误: 文件未找到 '{file_path}'")
        print("--- 检查结束 ---")
        return

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
                    fig, ax = plt.subplots(figsize=(14, 8))

                    # 定义一个好看的颜色循环
                    custom_colors_rgb_255 = [
                        (31, 119, 180),
                        (255, 127, 14),
                        (44, 160, 44),
                        (214, 39, 40)
                    ]
                    colors = [(r / 255., g / 255., b / 255.) for r, g, b in custom_colors_rgb_255]

                    def simple_moving_average(data, window_size=20):
                        if len(data) < window_size:
                            return data, np.arange(len(data))
                        smoothed = np.convolve(data, np.ones(window_size) / window_size, mode='valid')
                        x_axis = np.arange(len(smoothed)) + (window_size - 1) // 2
                        return smoothed, x_axis

                    if plot_data.ndim == 2:
                        num_lines = plot_data.shape[1]
                        for i in range(num_lines):
                            raw_data = plot_data[:, i]
                            current_color = colors[i % len(colors)]
                            ax.plot(raw_data, color=current_color, alpha=0.3, linewidth=1.0)
                            smoothed_data, smoothed_x = simple_moving_average(raw_data)
                            ax.plot(smoothed_x, smoothed_data, color=current_color, label=f'Agent {i + 1}',
                                    linewidth=2.5)
                        ax.legend(title="智能体", frameon=True, shadow=True, loc='best', fontsize=10)

                    else:
                        raw_data = plot_data
                        ax.plot(raw_data, color=colors[0], alpha=0.3, linewidth=1.0)
                        smoothed_data, smoothed_x = simple_moving_average(raw_data)
                        ax.plot(smoothed_x, smoothed_data, color=colors[0], label='权重值 (平滑)', linewidth=2.5)
                        ax.legend(frameon=True, shadow=True)

                    ax.set_title(f'联邦权重历史', fontsize=16, fontweight='bold', pad=20)
                    ax.set_xlabel("Episode ", fontsize=12)
                    ax.set_ylabel("权重值", fontsize=12)
                    ax.spines['top'].set_visible(False)
                    ax.spines['right'].set_visible(False)
                    fig.tight_layout()
                    print("绘图完成，请查看弹出的窗口。")
                    plt.show()

                except Exception as plot_e:
                    print(f"绘制图形时发生错误: {plot_e}")
            else:
                print(f"\n数组维度 ({data.ndim}) 暂不支持绘图。")

        else:
            print("文件内容: 这是一个未知的数据类型。")
            print("尝试直接打印内容...")
            print(data)

        print("-" * 40)

    except Exception as e:
        print(f"读取或解析文件时发生错误: {e}")

    finally:
        print("--- 检查结束 ---")


if __name__ == "__main__":
    target_file = 'D:\\ITE\\result/20260404/fed_weights_DSFA_194147.npy'
    if os.path.exists(target_file):
        inspect_npy_file(target_file)
    else:
        print(f"错误: 示例文件未找到 '{target_file}'")
        print("请在脚本中修改 'target_file' 变量为你的 .npy 文件路径。")
