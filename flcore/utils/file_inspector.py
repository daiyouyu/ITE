# /home/ubuntu/ITE/flcore/utils/file_inspector.py

import numpy as np
import os
import matplotlib.pyplot as plt

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

            # 新增：使用 matplotlib 绘制折线图
            # 只对1D或2D数组进行绘图
            if data.ndim in [1, 2]:
                print("\n正在生成折线图...")
                try:
                    plt.figure(figsize=(12, 7))
                    
                    # 如果是二维数组，每一列作为一条线
                    if data.ndim == 2:
                        num_lines = data.shape[1]
                        for i in range(num_lines):
                            plt.plot(data[:, i], label=f'Agent {i+1} 权重')
                        plt.title(f'联邦权重历史 ({os.path.basename(file_path)})')
                    # 如果是一维数组，直接绘制
                    else: # data.ndim == 1
                        plt.plot(data, label='权重值')
                        plt.title(f'联邦权重历史 ({os.path.basename(file_path)})')

                    plt.xlabel("联邦聚合轮次")
                    plt.ylabel("权重值")
                    plt.grid(True, linestyle='--', alpha=0.6)
                    plt.legend()
                    plt.tight_layout()
                    print("绘图完成，请查看弹出的窗口。")
                    plt.show()

                except Exception as plot_e:
                    print(f"绘制图形时发生错误: {plot_e}")
            else:
                print("\n数组维度 > 2，跳过绘图。")
        
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
    target_file = '/home/ubuntu/ITE/result/20260403/fed_weights_DSFA_151433.npy'
    
    if os.path.exists(target_file):
        # 调用函数进行检查
        inspect_npy_file(target_file)
    else:
        print(f"错误: 示例文件未找到 '{target_file}'")
        print("请在脚本中修改 'target_file' 变量为你的 .npy 文件路径。")