from load_data import load_ITE_data
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import matplotlib as mpl
import numpy as np

# ==== 修改为英文学术通用的字体设置 ====
mpl.rcParams["font.family"] = "sans-serif"
plt.rcParams['axes.unicode_minus'] = False
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

# 英文标签替换
L_indicators = [
    ('H_L', 'Heat Demand', 'MW'),
    ('G_L', 'Elec. Demand', 'MW'),
]

R_indicators = [
    ('R_wind', 'Wind Power', 'MW'),
    ('R_solar', 'Solar Power', 'MW'),
]


def view_HGR_data():
    import matplotlib.ticker as ticker

    # 1. 严格保留您原有的配色方案
    custom_colors = ['#0077B6', '#FF9F1C']
    mouth = 1

    # 2. 根据要求分配园区名称，并模仿参考图使用 (a)(b)(c)(d) 作为底部标题
    agent_names = [
        "(a) Residential Area Demand Profile",  # Agent 1: 居民区
        "(b) Renewable Energy Park Demand Profile",  # Agent 2: 新能源园区
        "(c) Residential Area Demand Profile",  # Agent 3: 居民区
        "(d) Industrial Park Demand Profile"  # Agent 4: 工业园区
    ]

    for group in range(4):
        ax = axes[group]
        times = data[group]['datetime'][24 * 30 * mouth:24 * 30 * mouth + 24]  # 该组的时间序列

        for idx, (indicator, label, unit) in enumerate(L_indicators):
            values = data[group][indicator][24 * 30 * mouth:24 * 30 * mouth + 24]
            current_color = custom_colors[idx % len(custom_colors)]

            # 采用参考图样式：实线 + 圆点标记 (marker='o')
            ax.plot(times, values, label=f'{label} ({unit})',
                    color=current_color, marker='o', markersize=5,
                    linewidth=2.0, zorder=3)

            # 采用参考图样式：统一使用面积填充 (取消原有的柱状图，统一为 fill_between)
            ax.fill_between(times, values, color=current_color, alpha=0.35, zorder=1)

        # ================= 模仿参考图的排版细节 =================

        # 清除原有顶部标题
        ax.set_title('')

        # Y轴标签
        ax.set_ylabel('Power (MW)', fontsize=18)

        # X轴标签与子图标题合并，放在图表下方（减小换行间距，从而减小上下留白）
        ax.set_xlabel(f'Hour (h)\n{agent_names[group]}', fontsize=18, labelpad=6)

        # 设置X轴刻度：每 2 小时一个刻度 (1:00, 3:00 ... 23:00)
        ax.xaxis.set_major_locator(mdates.HourLocator(byhour=range(1, 24, 3)))

        # 自定义格式化函数：去除小时前面的 0 (如 01:00 变成 1:00，还原参考图样式)
        def format_hour(x, pos):
            dt_obj = mdates.num2date(x)
            h = dt_obj.strftime('%H').lstrip('0')
            h = '0' if h == '' else h
            return f"{h}:00"

        ax.xaxis.set_major_formatter(ticker.FuncFormatter(format_hour))

        ax.tick_params(axis='x', rotation=0, labelsize=18)
        ax.tick_params(axis='y', labelsize=18)
        ax.set_ylim(bottom=0)

        # 网格线：全区域虚线网格，仿照参考图
        ax.grid(True, linestyle='--', alpha=0.7, color='#b0b0b0', zorder=0)

        # 边框：还原参考图的四周全黑封闭边框
        for spine in ax.spines.values():
            spine.set_visible(True)
            spine.set_color('black')
            spine.set_linewidth(1.0)

        # 图例：带边框，双列排布，放置在左上角
        ax.legend(loc='upper left', ncol=2, fontsize=14, frameon=True, edgecolor='#cccccc')

    # 减小 hspace 以紧凑排版
    plt.tight_layout()
    plt.subplots_adjust(hspace=0.2)
    plt.savefig('./figure/ITE_HGR_data_ENG.png', dpi=900, bbox_inches='tight')
    print("Saved plot -> figure/ITE_HGR_data_ENG.png")


def view_P_data():
    # 创建新画布，调整比例使其更贴近参考图的扁平学术风格
    plt.figure(figsize=(9, 4.5), dpi=300)

    # 1. 设置夏季和冬季的时间索引
    s_month = 7  # 夏季 (July)
    w_month = 12  # 冬季 (January)

    s_start = 24 * 30 * s_month
    w_start = 24 * 30 * w_month

    # 2. 提取数据
    s_values = data[0]['P'][s_start: s_start + 24]
    w_values = data[0]['P'][w_start: w_start + 24]

    # 阶梯图画满 0-24 小时
    s_values_plot = np.append(s_values, s_values[-1])
    w_values_plot = np.append(w_values, w_values[-1])

    # 3. 准备横坐标：0 到 24
    hours = np.arange(25)

    # 4. 绘图：使用 plt.step 并添加 marker，精确还原参考图样式
    # 红色实线 + 圆圈标记 (o)
    plt.step(hours, s_values_plot, where='post',
             label='Typical Summer Day (Jul)', color='#D32F2F', marker='o', markersize=6, linewidth=1.5)

    # 蓝紫色实线 + 方块标记 (s)
    plt.step(hours, w_values_plot, where='post',
             label='Typical Winter Day (Jan)', color='#00BFFF', marker='s', markersize=6, linewidth=1.5)

    # 5. 修饰图表
    plt.xlabel('Hour(h)', fontsize=12)
    # 标题间距与参考图保持一致
    plt.ylabel('Electricity price(CNY/MWh)', fontsize=12)

    # 设置X轴刻度：仿照参考图的奇数小时 1:00, 3:00... 23:00
    xticks_pos = np.arange(1, 24, 2)
    xtick_labels = [f"{h}:00" for h in xticks_pos]
    plt.xticks(xticks_pos, xtick_labels, fontsize=11)
    plt.yticks(fontsize=11)
    plt.xlim(0, 24)

    # 网格线：全区域虚线网格，仿照参考图
    plt.grid(True, linestyle='--', alpha=0.8, color='#b0b0b0')

    # 图例：带灰色边框，位于上方居中，双列排布
    plt.legend(loc='upper left', ncol=1, fontsize=9, frameon=True, edgecolor='#cccccc')

    # 边框：还原参考图的四周全黑边框（覆盖掉全局样式中的去除 top 和 right）
    ax = plt.gca()
    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_color('black')
        spine.set_linewidth(1.0)

    plt.tight_layout()
    plt.savefig('./figure/ITE_P_data_ENG.png', dpi=600, bbox_inches='tight')
    print("Saved plot -> figure/ITE_P_data_ENG.png")
    # plt.show()


def view_R_data():
    import matplotlib.ticker as ticker
    # 创建独立画布，避免影响其他图表。用于论文的图表应保证高清晰度
    fig_r, axes_r = plt.subplots(nrows=2, ncols=2, figsize=(16, 12), dpi=600)
    axes_r = axes_r.flatten()

    mouth = 6
    # 风能采用偏风属性的青绿色，太阳能采用太阳光的黄色
    color_wind = '#20B2AA'  # 浅海洋绿 / 青绿色 (LightSeaGreen)
    color_solar = '#FFD700'  # 金黄色 (Gold)

    agent_names = [
        "(a) Residential Area Renewable Generation",  # Agent 1
        "(b) Renewable Energy Park Renewable Generation",  # Agent 2
        "(c) Residential Area Renewable Generation",  # Agent 3
        "(d) Industrial Park Renewable Generation"  # Agent 4
    ]

    for group in range(4):
        ax = axes_r[group]
        # 提取时间序列和对应的数据
        times = data[group]['datetime'][24 * 30 * mouth:24 * 30 * mouth + 24]
        wind_values = np.array(data[group]['R_wind'][24 * 30 * mouth:24 * 30 * mouth + 24])
        solar_values = np.array(data[group]['R_solar'][24 * 30 * mouth:24 * 30 * mouth + 24])

        # 绘制二者叠加的柱形图（注意bottom参数）
        ax.bar(times, wind_values, label='Wind Power', color=color_wind, alpha=0.9, width=0.03, zorder=2)
        ax.bar(times, solar_values, bottom=wind_values, label='Solar Power', color=color_solar, alpha=0.9,
               width=0.03, zorder=2)

        # ================= 模仿 HGR 的排版细节 =================

        # 仿照 HGR 的布局风格，去掉原有标题
        ax.set_title('')

        # Y轴标签
        ax.set_ylabel('Power (MW)', fontsize=18)

        # X轴标签与子图标题合并，放在图表下方（减小换行间距，从而减小上下留白）
        ax.set_xlabel(f'Hour (h)\n{agent_names[group]}', fontsize=18, labelpad=6)

        # 优化坐标轴刻度，使其更易阅读：每 2 小时一个刻度 (1:00, 3:00 ... 23:00)
        ax.xaxis.set_major_locator(mdates.HourLocator(byhour=range(1, 24, 3)))

        def format_hour(x, pos):
            dt_obj = mdates.num2date(x)
            h = dt_obj.strftime('%H').lstrip('0')
            h = '0' if h == '' else h
            return f"{h}:00"

        ax.xaxis.set_major_formatter(ticker.FuncFormatter(format_hour))

        ax.tick_params(axis='x', rotation=0, labelsize=18)
        ax.tick_params(axis='y', labelsize=18)

        # 统一y轴范围
        ax.set_ylim(0, 16)

        # 网格线：全区域虚线网格，仿照 HGR
        ax.grid(True, linestyle='--', alpha=0.7, color='#b0b0b0', zorder=0)

        # 边框：四周全黑封闭边框
        for spine in ax.spines.values():
            spine.set_visible(True)
            spine.set_color('black')
            spine.set_linewidth(1.0)

        # 图例：带边框，双列排布，放置在左上角
        ax.legend(loc='upper left', ncol=1, fontsize=18, frameon=True, edgecolor='#cccccc')

    plt.tight_layout()
    plt.subplots_adjust(hspace=0.2)
    plt.savefig('./figure/ITE_R_data_ENG.png', dpi=900, bbox_inches='tight')
    print("Saved plot -> ./figure/ITE_R_data_ENG.png")
    # plt.show()


if __name__ == '__main__':
    view_HGR_data()
    # view_P_data()
    # view_R_data()
