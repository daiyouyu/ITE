import pandas as pd
import numpy as np
from load_data import load_ITE_data
import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib.dates import DateFormatter

mpl.rcParams["font.family"] = ["SimHei", "Microsoft YaHei"]
plt.rcParams['lines.linewidth'] = 3
path1 = "IES_data/G_demand.csv"
path2 = "IES_data/H_demand.csv"
data = load_ITE_data(path1, path2)
for d in data:
    del d['sin_h']
    del d['cos_h']
# 创建4个子图（对应4组数据），2行2列布局
fig, axes = plt.subplots(nrows=2, ncols=2, figsize=(16, 12), dpi=300)
axes = axes.flatten()  # 将2x2的axes转换为一维数组，方便遍历
indicators = [
    ('H_L', 'MWth'),
    ('G_L', 'MW'),
    ('R', 'MW'),
]
# 遍历4组数据
mouth = 6
for group in range(4):
    ax = axes[group]
    times = data[group]['datetime'][24 * 30 * mouth:24 * 30 * mouth + 24]  # 该组的时间序列
    # 遍历4个指标，在当前子图中绘制曲线
    for indicator, unit in indicators:
        values = data[group][indicator][24 * 30 * mouth:24 * 30 * mouth + 24]
        ax.plot(times, values, label=f'{indicator}（{unit}）')

    # 设置子图标题和坐标轴
    ax.set_title(f'第 {group + 1} 组数据指标变化')
    ax.set_xlabel('时间')
    ax.tick_params(axis='x', rotation=45)
    ax.grid(alpha=1)
    ax.legend()  # 每个子图单独显示图例

plt.tight_layout()
plt.show()
