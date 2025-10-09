import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.dates import DateFormatter

# 设置中文字体支持
plt.rcParams["font.family"] = ["SimHei", "WenQuanYi Micro Hei", "Heiti TC"]
plt.rcParams['figure.dpi'] = 300  # 全局分辨率设置
plt.rcParams['axes.grid'] = True
plt.rcParams['grid.alpha'] = 0.3

# 1. 数据加载与处理
csv_path = "D:\浏览器下载\Reinforcement-Learning-main-9.DDPG\9.DDPG\data\GridSet_no_pred.csv"  # 替换为实际文件路径
start = None  # 可设置起始日期，如"2023-01-01"
end = None    # 可设置结束日期，如"2023-01-31"

df = pd.read_csv(csv_path, parse_dates=['date'])
if start is not None:
    df = df[df['date'] >= pd.to_datetime(start)]
if end is not None:
    df = df[df['date'] <= pd.to_datetime(end)]
df = df.sort_values('date').reset_index(drop=True)

# 提取数据
load_cols = ['COAST','EAST','FWEST','NORTH','NCENT','SOUTH','SCENT','WEST']
L = df[load_cols].astype(float).values  # 负荷数据
lz_cols = ['LZ_AEN','LZ_CPS','LZ_HOUSTON','LZ_LCRA','LZ_NORTH','LZ_RAYBN','LZ_SOUTH','LZ_WEST']
P = df[lz_cols].astype(float).values  # 电价数据
new_col = ['WIND_ACTUAL_SYSTEM_WIDE','SOLAR_ACTUAL_SYSTEM_WIDE']
R = df[new_col].astype(float).values  # 新能源发电量
date_series = df['date'].values  # 时间序列
L = L[1000:1000+24,:]
P = P[1000:1000+24,:]
R = R[1000:1000+24,:]
date_series = date_series[1000:1000+24]

# 2. 绘制八个地区的电价图
plt.figure(figsize=(16, 12))
for i in range(8):
    plt.subplot(4, 2, i+1)  # 4行2列布局
    plt.plot(date_series, P[:, i], linewidth=1.2, alpha=0.8)
    plt.title(f'电价 - {lz_cols[i]}', fontsize=10)
    plt.ylabel('价格 ($/MWh)', fontsize=8)
    plt.xticks(rotation=45)
    plt.gca().xaxis.set_major_formatter(DateFormatter('%y-%m-%d'))  # 日期格式
    plt.tight_layout()

plt.suptitle('各地区电价趋势', fontsize=16, y=1.02)
plt.savefig('各地区电价趋势.pdf', dpi=600, bbox_inches='tight')
plt.close()


# 3. 绘制八个地区的电负荷图
plt.figure(figsize=(16, 12))
for i in range(8):
    plt.subplot(4, 2, i+1)  # 4行2列布局
    plt.plot(date_series, L[:, i], linewidth=1.2, alpha=0.8)
    plt.title(f'负荷 - {load_cols[i]}', fontsize=10)
    plt.ylabel('负荷 (MW)', fontsize=8)
    plt.xticks(rotation=45)
    plt.gca().xaxis.set_major_formatter(DateFormatter('%y-%m-%d'))  # 日期格式
    plt.tight_layout()

plt.suptitle('各地区电负荷趋势', fontsize=16, y=1.02)
plt.savefig('各地区电负荷趋势.pdf', dpi=600, bbox_inches='tight')
plt.close()


# 4. 绘制新能源发电量图

plt.figure(figsize=(12, 16))
for i in range(2):
    plt.subplot(2, 1, i+1)  # 4行2列布局
    plt.plot(date_series, R[:, i], linewidth=1.2, alpha=0.8)
    plt.title(f'新能源- {new_col[i]}', fontsize=10)
    plt.ylabel('发电 (MW)', fontsize=8)
    plt.xticks(rotation=45)
    plt.gca().xaxis.set_major_formatter(DateFormatter('%y-%m-%d'))  # 日期格式
    plt.tight_layout()

plt.suptitle('新能源发电趋势', fontsize=16, y=1.02)
plt.savefig('新能源发电量趋势.pdf', dpi=600, bbox_inches='tight')
plt.close()

print("三张图片已生成：各地区电价趋势.pdf、各地区电负荷趋势.pdf、新能源发电量趋势.pdf")
