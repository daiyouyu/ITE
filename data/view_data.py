from load_data import load_ITE_data
import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np

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


def view_HGR_data():
    # 遍历4组数据
    mouth = 1
    for group in range(4):
        ax = axes[group]
        times = data[group]['datetime'][24 * 30 * mouth:24 * 30 * mouth + 24]  # 该组的时间序列
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
    plt.savefig('./ITE_data.png', dpi=600)
    plt.show()


def view_P_data():
    # 创建新画布
    plt.figure(figsize=(14, 6), dpi=300)

    # 1. 设置夏季和冬季的时间索引
    s_month = 7  # 夏季
    w_month = 12  # 冬季

    s_start = 24 * 30 * s_month
    w_start = 24 * 30 * w_month

    # 2. 提取数据
    s_values = data[0]['P'][s_start: s_start + 24]
    w_values = data[0]['P'][w_start: w_start + 24]

    # ★关键步骤★：为了让阶梯图画满 0-24 小时，需要把最后一个价格重复一次
    # 变成 25 个数据点，对应 0, 1, ..., 24 这 25 个刻度
    s_values_plot = np.append(s_values, s_values[-1])
    w_values_plot = np.append(w_values, w_values[-1])

    # 3. 准备横坐标：0 到 24
    hours = np.arange(25)

    # 4. 绘图：使用 plt.step 替代 hlines
    # where='post' 表示：在点之后阶跃（即：0-1小时的值取 index=0 的价格）

    # 夏季：红色实线
    plt.step(hours, s_values_plot, where='post',
             label='夏季典型日电价 (7月)', color='red', linewidth=3)

    # 冬季：蓝色虚线
    plt.step(hours, w_values_plot, where='post',
             label='冬季典型日电价 (1月)', color='blue', linestyle='--', linewidth=3)

    # 5. 修饰图表
    plt.xlabel('小时 (h)', fontsize=12)
    plt.ylabel('价格 (元/MWh)', fontsize=12)

    # 设置刻度
    plt.xticks(range(25))  # 显示 0-24 所有刻度
    plt.xlim(0, 24)  # 严格限制 x 轴范围在 0 到 24

    # 网格线
    plt.grid(True, linestyle=':', alpha=0.5)
    plt.legend(loc='best')

    plt.tight_layout()
    plt.savefig('./ITE_data.png', dpi=600)
    plt.show()


if __name__ == '__main__':
    view_HGR_data()
