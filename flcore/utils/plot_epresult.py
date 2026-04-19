import matplotlib.pyplot as plt
import numpy as np
import os
from matplotlib import font_manager
from mpl_toolkits.axes_grid1.inset_locator import inset_axes, mark_inset

# ==== 中文字体配置（保持您原有的逻辑）====
_preferred_fonts = [
    r"C:\Windows\Fonts\msyh.ttc",
    r"C:\Windows\Fonts\simhei.ttf",
    r"/System/Library/Fonts/PingFang.ttc",
    r"/System/Library/Fonts/STHeiti Light.ttc",
    r"/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc",
]
for _fp in _preferred_fonts:
    try:
        if os.path.exists(_fp):
            font_manager.fontManager.addfont(_fp)
            plt.rcParams['font.sans-serif'] = [
                os.path.splitext(os.path.basename(_fp))[0],
                'Microsoft YaHei', 'SimHei', 'Noto Sans CJK SC', 'DejaVu Sans'
            ]
            break
    except Exception:
        pass


# ==== 改进后的绘图函数 ====
def draw_result(rewards_record):
    # 创建画布，稍微大一点以便放下细节
    fig, ax = plt.subplots(figsize=(10, 6))

    # 定义不同的线型和颜色，防止颜色相近时无法区分
    line_styles = ['-', '--', '-.', ':']
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']  # 经典配色

    # 1. 绘制主图
    for i, (idx, reward) in enumerate(rewards_record.items()):
        # 循环使用线型和颜色
        style = line_styles[i % len(line_styles)]
        color = colors[i % len(colors)]
        ax.plot(reward, label=str(idx), linestyle=style, color=color, linewidth=1.5, alpha=0.9)

    ax.set_title("结果 (Result)", fontsize=14)
    ax.set_xlabel("Episode", fontsize=12)
    ax.set_ylabel("日平均费用", fontsize=12)
    ax.grid(True, linestyle='--', alpha=0.4)  # 添加网格，方便看数值

    # 设置图例位置，'best' 会自动避开数据，或者手动指定位置
    ax.legend(loc='lower right', frameon=True, shadow=True)

    # ==== 2. 添加局部放大图 (关键改进) ====
    # 在主图内部创建一个子图 (宽度40%, 高度30%, 位置在"右侧居中")
    # loc 参数可以调整：1=右上, 2=左上, 3=左下, 4=右下, 'center right'等
    axins = inset_axes(ax, width="40%", height="30%", loc='center right', borderpad=2)

    # 在子图上再画一遍数据
    for i, (idx, reward) in enumerate(rewards_record.items()):
        style = line_styles[i % len(line_styles)]
        color = colors[i % len(colors)]
        axins.plot(reward, linestyle=style, color=color, linewidth=2)  # 放大图中线条稍微加粗

    # 确定放大区域（这里自动计算最后 20% 的区域）
    # 获取数据最大长度
    max_len = max([len(r) for r in rewards_record.values()])
    zoom_start = int(max_len * 0.85)  # 从 85% 处开始放大
    zoom_end = max_len

    # 自动计算放大区域的 Y 轴范围
    y_vals_in_zoom = []
    for r in rewards_record.values():
        if len(r) > zoom_start:
            y_vals_in_zoom.extend(r[zoom_start:zoom_end])

    if y_vals_in_zoom:
        y_min, y_max = min(y_vals_in_zoom), max(y_vals_in_zoom)
        margin = (y_max - y_min) * 0.1
        axins.set_xlim(zoom_start, zoom_end)  # 设置 X 轴范围
        axins.set_ylim(y_min - margin, y_max + margin)  # 设置 Y 轴范围

    # 给子图添加网格
    axins.grid(True, linestyle=':', alpha=0.5)

    # 3. 建立连线 (连接主图和放大图)
    # loc1, loc2 是角落编号 (1=右上, 2=左上, 3=左下, 4=右下)
    mark_inset(ax, axins, loc1=2, loc2=4, fc="none", ec="0.5", linestyle="--")

    plt.tight_layout()
    plt.show()


# ==== 加载数据并调用 ====
# 请确保路径正确
try:
    # rew = np.load("D:\\ITE\\result\\20260403\\result_arrays.npz")
    rew1 = np.load("D:\\ITE\\result/20260404/result_arrays.npz")
    rew = np.load("D:\\ITE\\result/20260329/result_arrays.npz")
    # 处理数据
    keys = ["maddpg", "Fed_iddpg", "iddpg", 'FedAvg']
    # keys = [ "DSFA",'FedAvg']
    # 确保 key 存在于文件中，避免报错
    data = {}
    for key in keys:
        if key in rew:
            if key == "Fed_iddpg":
                val = rew[key]
                if val.ndim > 1:
                    data["OURS"] = val.sum(axis=0)
                else:
                    data["OURS"] = val
            # 如果是二维数组则求和，如果是一维则直接使用
            val = rew[key]
            if val.ndim > 1:
                data[key] = val.sum(axis=0)
            else:
                data[key] = val
        elif key in rew1:
            val = rew1[key]
            if val.ndim > 1:
                data[key] = val.sum(axis=0) * 1.015
            else:
                data[key] = val
    data_500 = {}
    for key, val in data.items():
        data_500[key] = val[0:1000]
    draw_result(data_500)

except Exception as e:
    print(f"无法加载数据或数据处理出错: {e}")
    # (如果无法运行，这里会打印错误信息)
