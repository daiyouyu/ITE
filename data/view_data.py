from load_data import load_ITE_data
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import matplotlib as mpl
import numpy as np

mpl.rcParams["font.family"] = ["SimHei", "Microsoft YaHei"]
plt.rcParams['lines.linewidth'] = 3
path1 = "IES_data/G_demand.csv"
path2 = "IES_data/H_demand.csv"
data = load_ITE_data(path1, path2)
for d in data:
    if 'sin_h' in d:
        del d['sin_h']
    if 'cos_h' in d:
        del d['cos_h']
# 创建4个子图（对应4组数据），2行2列布局
fig, axes = plt.subplots(nrows=2, ncols=2, figsize=(16, 12), dpi=300)
axes = axes.flatten()  # 将2x2的axes转换为一维数组，方便遍历
L_indicators = [
    ('H_L', '热力需求', 'MWth'),
    ('G_L', '电力需求', 'MW'),
    #    ('R', 'MW'),
]

R_indicators = [
    ('R_wind', '风力发电', 'MW'),
    ('R_solar', '太阳能发电', 'MW'),
]


def view_HGR_data():
    # 定义配色方案 (深海蓝, 活力橙)
    custom_colors = ['#0077B6', '#FF9F1C']
    mouth = 1

    for group in range(4):
        ax = axes[group]
        times = data[group]['datetime'][24 * 30 * mouth:24 * 30 * mouth + 24]  # 该组的时间序列

        for idx, (indicator, label, unit) in enumerate(L_indicators):
            values = data[group][indicator][24 * 30 * mouth:24 * 30 * mouth + 24]
            current_color = custom_colors[idx % len(custom_colors)]

            # 统一绘制折线 (设置 zorder=3 确保折线永远在最上层，不会被柱子挡住)
            ax.plot(times, values, label=f'{label}（{unit}）',
                    color=current_color,
                    linewidth=2.5, zorder=3)

            # 根据标签名称区分填充逻辑
            if '电' in label:
                # 电力需求：柱状填充
                # 关键：zorder=2 放在折线下方。
                # 宽度提醒：如果 times 是 datetime 对象，width=0.03 约等于 43 分钟的宽度，能产生好看的间隙。
                # 如果你的 times 只是普通的字符串，请把 width 改为 0.6 左右。
                ax.bar(times, values, color=current_color, alpha=0.35, width=0.03, zorder=2)
            else:
                # 热力需求：面积填充
                # zorder=1 放在最底层，透明度设低一点
                ax.fill_between(times, values, color=current_color, alpha=0.35, zorder=1)

        # 设置子图标题和间距
        ax.set_title(f'第 {group + 1} 组数据指标变化', fontsize=13, fontweight='bold', pad=12)
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%H:00'))
        # 优化坐标轴刻度 (因为只显示小时了，rotation=0 不旋转可能会更好看，你可以视情况保留 45)

        ax.set_xlabel('时间', fontsize=11, color='#333333')

        ax.tick_params(axis='x', rotation=0, colors='#555555', labelsize=10)
        ax.tick_params(axis='y', colors='#555555', labelsize=10)
        ax.set_ylim(0, 20)
        # 优化网格 (zorder=0 确保网格线在所有图表最下方)
        ax.grid(True, linestyle='--', alpha=0.4, color='#888888', zorder=0)

        # 优化边框
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['left'].set_color('#cccccc')
        ax.spines['bottom'].set_color('#cccccc')

        # 优化图例
        ax.legend(frameon=True, fancybox=True, framealpha=0.9, edgecolor='#eeeeee')

    plt.tight_layout()
    plt.savefig('./ITE_HGR_data.svg', dpi=600, bbox_inches='tight')
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
    plt.savefig('./ITE_P_data.png', dpi=600)
    plt.show()


def view_R_data():
    # 创建独立画布，避免影响其他图表。用于论文的图表应保证高清晰度
    fig_r, axes_r = plt.subplots(nrows=2, ncols=2, figsize=(16, 12), dpi=600)
    axes_r = axes_r.flatten()

    mouth = 6
    # 风能采用偏风属性的青绿色，太阳能采用太阳光的黄色
    color_wind = '#20B2AA'  # 浅海洋绿 / 青绿色 (LightSeaGreen)
    color_solar = '#FFD700'  # 金黄色 (Gold)

    for group in range(4):
        ax = axes_r[group]
        # 提取时间序列和对应的数据
        times = data[group]['datetime'][24 * 30 * mouth:24 * 30 * mouth + 24]
        wind_values = np.array(data[group]['R_wind'][24 * 30 * mouth:24 * 30 * mouth + 24])
        solar_values = np.array(data[group]['R_solar'][24 * 30 * mouth:24 * 30 * mouth + 24])

        # 绘制二者叠加的柱形图（注意bottom参数）
        ax.bar(times, wind_values, label='风力发电（MW）', color=color_wind, alpha=0.9, width=0.03, zorder=2)
        ax.bar(times, solar_values, bottom=wind_values, label='太阳能发电（MW）', color=color_solar, alpha=0.9,
               width=0.03, zorder=2)

        # 论文风格：图表标题、轴标签清晰，采用加粗字体
        ax.set_title(f'第 {group + 1} 组数据新能源发电出力', fontsize=14, fontweight='bold', pad=12)
        ax.set_xlabel('时间', fontsize=12, fontweight='bold', color='black')
        ax.set_ylabel('功率 (MW)', fontsize=12, fontweight='bold', color='black')

        # 优化坐标轴刻度，使其更易阅读
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%H:00'))
        # 优化坐标轴刻度 (因为只显示小时了，rotation=0 不旋转可能会更好看，你可以视情况保留 45)
        ax.tick_params(axis='x', rotation=0, colors='#555555', labelsize=10)
        ax.tick_params(axis='y', labelsize=11)

        # 统一y轴范围
        ax.set_ylim(0, 16)

        # 论文风格网格：只保留横向网格线以便于读数，将其放在最底层(zorder=0)
        ax.grid(True, axis='y', linestyle='--', alpha=0.6, color='#aaaaaa', zorder=0)

        # 优化边框：去除上右边框（常见学术规范），加粗左下边框
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['left'].set_color('black')
        ax.spines['left'].set_linewidth(1.5)
        ax.spines['bottom'].set_color('black')
        ax.spines['bottom'].set_linewidth(1.5)

        # 优化图例：去边框，位置右上角
        ax.legend(frameon=False, fontsize=12, loc='upper right')

    plt.tight_layout()
    # plt.savefig('./ITE_R_data.svg', dpi=600, bbox_inches='tight')
    plt.show()


if __name__ == '__main__':
    # view_HGR_data()
    # view_P_data()
    view_R_data()
