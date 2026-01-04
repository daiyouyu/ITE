import matplotlib.pyplot as plt
import numpy as np
import os
from matplotlib import font_manager

# ==== 中文字体配置（解决 DejaVu Sans 缺少 CJK 的告警）====
# 优先尝试系统已安装字体；若存在则注册并设置为默认
_preferred_fonts = [
    r"C:\Windows\Fonts\msyh.ttc",  # 微软雅黑（Windows）
    r"C:\Windows\Fonts\simhei.ttf",  # 黑体（Windows）
    r"/System/Library/Fonts/PingFang.ttc",  # 苹方（macOS）
    r"/System/Library/Fonts/STHeiti Light.ttc",
    r"/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc",  # Noto CJK（Linux 常见）
]
for _fp in _preferred_fonts:
    try:
        if os.path.exists(_fp):
            font_manager.fontManager.addfont(_fp)
            # 设置 sans-serif 优先顺序（第一个找到的就是默认）
            plt.rcParams['font.sans-serif'] = [
                os.path.splitext(os.path.basename(_fp))[0],
                'Microsoft YaHei', 'SimHei', 'Noto Sans CJK SC', 'DejaVu Sans'
            ]
            break
    except Exception:
        pass


def draw_result(rewards_record):
    # plot
    for idx, reward in rewards_record.items():
        plt.plot(reward, label=str(idx))
    plt.title("结果")
    plt.xlabel("Episode")
    plt.ylabel("日平均费用")
    plt.legend()
    plt.show()


rew = np.load("D:\\ITE-main\\result\\result_arrays.npz")
keys = ["iddpg", "Fed_iddpg", "maddpg", "Fed_maddpg"]
data = {key: rew[key].sum(axis=0) for key in rew.keys()}
#data = {key: rew[key] for key in rew.keys()}
draw_result(data)
