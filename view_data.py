import numpy as np
import matplotlib.pyplot as plt

data=np.load("ori_rewards.npz")
data=data["rewards"]

# 提取三列数据
line1 = data[:, 0]  # 第一列数据
line2 = data[:, 1]  # 第二列数据
line3 = data[:, 2]  # 第三列数据

# 创建x轴坐标（数据点的索引）
x = np.arange(len(data))

# 创建图形和坐标轴
plt.figure(figsize=(10, 6))

# 绘制三条线，分别设置不同的颜色和标记
plt.plot(x, line1, label='Line 1', color='blue',linewidth=1, marker='o', markersize=2, linestyle='-')
#plt.plot(x, line2, label='Line 2', color='red', linewidth=1,marker='s', markersize=2, linestyle='-')
#plt.plot(x, line3, label='Line 3', color='green',linewidth=1, marker='^', markersize=2, linestyle='-')

# 添加标题和轴标签
plt.title('Three Lines from the Array', fontsize=14)
plt.xlabel('Index', fontsize=12)
plt.ylabel('Value', fontsize=12)

# 添加网格和图例
plt.grid(True, linestyle='--', alpha=0.7)
plt.legend(fontsize=12)

# 调整布局
plt.tight_layout()

# 显示图形
plt.show()

