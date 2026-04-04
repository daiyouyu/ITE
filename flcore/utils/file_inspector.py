# /home/ubuntu/ITE/flcore/utils/file_inspector.py

import numpy as np
import os
import matplotlib.pyplot as plt
from matplotlib import font_manager

# ==== 中文字体配置 (确保中文标签正确显示) ====
_preferred_fonts = [
    r"C:\Windows\Fonts\msyh.ttc",
    r"C:\Windows\Fonts\simhei.ttf",
    r"/System/Library/Fonts/PingFang.ttc",
    r"/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc", # For Ubuntu
]
for _fp in _preferred_fonts:
    try:
        if os.path.exists(_fp):
            font_manager.fontManager.addfont(_fp)
            plt.rcParams['font.sans-serif'] = [os.path.splitext(os.path.basename(_fp))[0]] + plt.rcParams['font.sans-serif']
            plt.rcParams['axes.unicode_minus'] = False # 解决负号显示问题
            break
    except Exception:
        pass

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
                    plt.style.use('seaborn-v0_8-whitegrid')
                    fig, ax = plt.subplots(figsize=(14, 8))

                    # 定义一个好看的颜色循环
                    # 使用用户指定的颜色，并将其归一化到 [0, 1] 范围
                    custom_colors_rgb_255 = [
                        (31, 119, 180),  # 深蓝
                        (255, 127, 14),  # 橙色
                        (44, 160, 44),   # 绿色
                        (214, 39, 40)    # 红色
                    ]
                    colors = [(r / 255., g / 255., b / 255.) for r, g, b in custom_colors_rgb_255]

                    def simple_moving_average(data, window_size=20):
                        """计算移动平均值，并返回平滑后的数据和对应的x轴坐标"""
                        if len(data) < window_size:
                            return data, np.arange(len(data))
                        smoothed = np.convolve(data, np.ones(window_size)/window_size, mode='valid')
                        x_axis = np.arange(len(smoothed)) + (window_size - 1) // 2
                        return smoothed, x_axis
                    
                    # 如果是二维数组，每一列作为一条线
                    if plot_data.ndim == 2:
                        num_lines = plot_data.shape[1]
                        for i in range(num_lines):
                            raw_data = plot_data[:, i]
                            current_color = colors[i % len(colors)] # 循环使用这四种颜色
                            ax.plot(raw_data, color=current_color, alpha=0.3, linewidth=1.0) # 原始数据稍微透明
                            smoothed_data, smoothed_x = simple_moving_average(raw_data)
                            ax.plot(smoothed_x, smoothed_data, color=current_color, label=f'Agent {i+1}', linewidth=2.5) # 平滑曲线稍微粗一些
                        ax.legend(title="智能体", frameon=True, shadow=True, loc='best', fontsize=10)

                    # 如果是一维数组，直接绘制
                    else: # plot_data.ndim == 1 (如果只有一个Agent，或者数据本身就是1D)
                        raw_data = plot_data
                        ax.plot(raw_data, color=colors[0], alpha=0.3, linewidth=1.0)
                        smoothed_data, smoothed_x = simple_moving_average(raw_data)
                        ax.plot(smoothed_x, smoothed_data, color=colors[0], label='权重值 (平滑)', linewidth=2.5)
                        ax.legend(frameon=True, shadow=True)

                    # --- 统一设置美化选项 ---
                    ax.set_title(f'联邦权重历史: {os.path.basename(file_path)}', fontsize=16, fontweight='bold', pad=20)
                    ax.set_xlabel("Episode / 轮次", fontsize=12)
                    ax.set_ylabel("对角线权重值", fontsize=12)
                    ax.spines['top'].set_visible(False)
                    ax.spines['right'].set_visible(False)
                    fig.tight_layout()
                    print("绘图完成，请查看弹出的窗口。")
                    plt.show()

                except Exception as plot_e:
                    print(f"绘制图形时发生错误: {plot_e}")
            else:
                print(f"\n数组维度 ({data.ndim}) 暂不支持绘图。")
        
        # 其他未知类型
        else:
            print("文件内容: 这是一个未知的数据类型。")
            print("尝试直接打印内容...")
            print(data)

        print("-" * 40)
        # 如果你想查看完整的权重数据，可以取消下面这行代码的注释。
        # 注意：如果文件很大，这会打印大量数据。
        # print("完整内容:\n", data)

    except Exception as e:
        print(f"读取或解析文件时发生错误: {e}")
    
    finally:
        print("--- 检查结束 ---")



if __name__ == "__main__":
    # 你想要检查的 .npy 文件的路径
    # 这是一个示例路径，请根据你的实际文件路径进行修改
    target_file = '/home/ubuntu/ITE/result/20260403/fed_weights_DSFA_235608.npy'
    
    if os.path.exists(target_file):
        # 调用函数进行检查
        inspect_npy_file(target_file)
    else:
        print(f"错误: 示例文件未找到 '{target_file}'")
        print("请在脚本中修改 'target_file' 变量为你的 .npy 文件路径。")