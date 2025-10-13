import numpy as np
import matplotlib.pyplot as plt
import os
from matplotlib import font_manager

# ==== 中文字体配置（解决 DejaVu Sans 缺少 CJK 的告警）====
# 优先尝试系统已安装字体；若存在则注册并设置为默认
_preferred_fonts = [
    r"C:\Windows\Fonts\msyh.ttc",   # 微软雅黑（Windows）
    r"C:\Windows\Fonts\simhei.ttf", # 黑体（Windows）
    r"/System/Library/Fonts/PingFang.ttc",  # 苹方（macOS）
    r"/System/Library/Fonts/STHeiti Light.ttc",
    r"/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc", # Noto CJK（Linux 常见）
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

# 解决坐标轴负号显示为方块的问题
plt.rcParams['axes.unicode_minus'] = False
from typing import Dict, List, Tuple
import argparse

from flcore.Env.multi_env import MultiBatteryCoordinator
from data.load_data import load_power_data
from flcore.algorithm.IDDPG import IDDPG
from flcore.algorithm.MADDPG import MADDPG

# ----------------------------
# Helpers
# ----------------------------
def _flatten(xs: List[np.ndarray]) -> np.ndarray:
    return np.concatenate([np.asarray(x, dtype=np.float32).ravel() for x in xs], axis=0)

def _by_agents(d: Dict[str, np.ndarray], agents: List[str]) -> List[np.ndarray]:
    return [d[a] for a in agents]

# ----------------------------
# Evaluation & Plotting
# ----------------------------
def rollout_one_day_and_collect(env: MultiBatteryCoordinator,
                               model,
                               agents: List[str],
                               day_start_idx: int,
                               dt_hours: float = 1.0) -> Dict[str, np.ndarray]:
    """
    聚合版：返回全系统 24 小时的总需求与总供给分量。
    """
    obs, _ = env.reset()
    for _ in range(day_start_idx):
        zero_actions = {a: np.zeros(env.action_spaces[a].shape, dtype=np.float32) for a in agents}
        obs, _, term, trunc, _ = env.step(zero_actions)
        if all(bool(term[a]) or bool(trunc[a]) for a in agents):
            break

    hours = 48
    agg = {
        'demand': np.zeros(hours, dtype=np.float32),
        'renew': np.zeros(hours, dtype=np.float32),
        'bat_dis': np.zeros(hours, dtype=np.float32),
        'boiler': np.zeros(hours, dtype=np.float32),
        'market_buy': np.zeros(hours, dtype=np.float32),
        'grid_buy': np.zeros(hours, dtype=np.float32),
        'surplus_dump': np.zeros(hours, dtype=np.float32),
    }

    for h in range(hours):
        acts = model.select_actions(_by_agents(obs, agents), noise_scale=0.0)
        action_dict = {a: acts[i] for i, a in enumerate(agents)}
        next_obs, rew_dict, term_dict, trunc_dict, info_dict = env.step(action_dict)

        L = 0.0; R = 0.0; bat_dis = 0.0; boiler = 0.0
        m_buy_MWh = 0.0; grid_buy_MWh = 0.0; dump_MWh = 0.0
        for aid in agents:
            inf = info_dict[aid]
            L += float(inf.get('G_demand', 0.0))
            R += float(inf.get('newpower_gen', 0.0))
            p_bat = float(inf.get('p_bat', 0.0))
            bat_dis += max(0.0, p_bat)
            boiler += max(0.0, float(inf.get('bioler_gen', 0.0)))
            m_buy_MWh += float(inf.get('market_buy_MWh', 0.0))
            grid_buy_MWh += float(inf.get('grid_buy_MWh', 0.0))
            dump_MWh += float(inf.get('surplus_dump_MWh', 0.0))

        agg['demand'][h] = L
        agg['renew'][h] = R
        agg['bat_dis'][h] = bat_dis
        agg['boiler'][h] = boiler
        agg['market_buy'][h] = m_buy_MWh / max(1e-9, dt_hours)
        agg['grid_buy'][h] = grid_buy_MWh / max(1e-9, dt_hours)
        agg['surplus_dump'][h] = dump_MWh / max(1e-9, dt_hours)

        obs = next_obs
        if all(bool(term_dict[a]) or bool(trunc_dict[a]) for a in agents):
            for k in agg.keys():
                agg[k] = agg[k][:h+1]
            break

    return agg


def rollout_one_day_per_agent(env: MultiBatteryCoordinator,
                              model,
                              agents: List[str],
                              day_start_idx: int,
                              dt_hours: float = 1.0) -> Dict[str, Dict[str, np.ndarray]]:
    """
    分智能体版：为每个智能体分别收集其 24 小时的 需求与供给分量。

    返回：{
      agent_id: {
         'demand','renew','bat_dis','boiler','market_buy','grid_buy','surplus_dump' -> (24,)
      }
    }
    """
    obs, _ = env.reset()
    for _ in range(day_start_idx):
        zero_actions = {a: np.zeros(env.action_spaces[a].shape, dtype=np.float32) for a in agents}
        obs, _, term, trunc, _ = env.step(zero_actions)
        if all(bool(term[a]) or bool(trunc[a]) for a in agents):
            break

    hours = 24
    per = {a: {k: np.zeros(hours, dtype=np.float32) for k in [
        'demand','renew','bat_dis','boiler','market_buy','grid_buy','surplus_dump']}
        for a in agents}

    for h in range(hours):
        acts = model.select_actions(_by_agents(obs, agents), noise_scale=0.0)
        action_dict = {a: acts[i] for i, a in enumerate(agents)}
        next_obs, rew_dict, term_dict, trunc_dict, info_dict = env.step(action_dict)

        for aid in agents:
            inf = info_dict[aid]
            per[aid]['demand'][h] = float(inf.get('G_demand', 0.0))
            per[aid]['renew'][h] = float(inf.get('newpower_gen', 0.0))
            p_bat = float(inf.get('p_bat', 0.0))
            per[aid]['bat_dis'][h] =  p_bat
            per[aid]['boiler'][h] = max(0.0, float(inf.get('bioler_gen', 0.0)))
            per[aid]['market_buy'][h] = float(inf.get('market_buy_MWh', 0.0)) / max(1e-9, dt_hours)
            per[aid]['grid_buy'][h] = float(inf.get('grid_buy_MWh', 0.0)) / max(1e-9, dt_hours)
            per[aid]['surplus_dump'][h] = float(inf.get('surplus_dump_MWh', 0.0)) / max(1e-9, dt_hours)

        obs = next_obs
        if all(bool(term_dict[a]) or bool(trunc_dict[a]) for a in agents):
            for aid in agents:
                for k in per[aid].keys():
                    per[aid][k] = per[aid][k][:h+1]
            break

    return per


def plot_daily_stack(agg: Dict[str, np.ndarray],
                     title: str = "日内用电需求与各方向供给（MW）",
                     save_path: str = "daily_supply_stack.png") -> None:
    hours = len(agg['demand'])
    x = np.arange(hours)

    s1 = agg['renew']
    s2 = agg['bat_dis']
    s3 = agg['boiler']
    s4 = agg['market_buy']
    s5 = agg['grid_buy']

    fig, ax = plt.subplots(figsize=(14, 5))
    ax.bar(x, s1, label='可再生出力', width=0.8)
    ax.bar(x, s2, bottom=s1, label='电池放电', width=0.8)
    ax.bar(x, s3, bottom=s1 + s2, label='锅炉发电', width=0.8)
    ax.bar(x, s4, bottom=s1 + s2 + s3, label='内部购电', width=0.8)
    ax.bar(x, s5, bottom=s1 + s2 + s3 + s4, label='外网购电', width=0.8)
    ax.plot(x, agg['demand'], linestyle='--', linewidth=2.0, label='需求（L）')

    ax.set_xticks(x)
    ax.set_xticklabels([f"{h:02d}:00" for h in range(hours)])
    ax.set_ylabel('功率 / MW')
    ax.set_title(title)
    ax.legend(ncol=3, loc='upper right')
    ax.grid(axis='y', linestyle=':', alpha=0.6)
    plt.tight_layout()
    plt.savefig(save_path, dpi=200)
    plt.close(fig)


def plot_daily_stack_per_agent(per: Dict[str, Dict[str, np.ndarray]],
                               save_dir: str = ".",
                               filename_prefix: str = "daily_agent_") -> List[str]:
    """为每个智能体各出一张独立图片。"""
    os.makedirs(save_dir, exist_ok=True)
    saved = []
    for aid, dd in per.items():
        hours = len(dd['demand'])
        x = np.arange(hours)
        s1, s2, s3, s4, s5 = dd['renew'], dd['bat_dis'], dd['boiler'], dd['market_buy'], dd['grid_buy']

        fig, ax = plt.subplots(figsize=(14, 5))
        ax.bar(x, s1, label='可再生出力', width=0.8)
        ax.bar(x, s2, bottom=s1, label='电池放电', width=0.8)
        ax.bar(x, s3, bottom=s1 + s2, label='锅炉发电', width=0.8)
        ax.bar(x, s4, bottom=s1 + s2 + s3, label='内部购电', width=0.8)
        ax.bar(x, s5, bottom=s1 + s2 + s3 + s4, label='外网购电', width=0.8)
        ax.plot(x, dd['demand'], linestyle='--', linewidth=2.0, label='需求（L）')

        ax.set_xticks(x)
        ax.set_xticklabels([f"{h:02d}:00" for h in range(hours)])
        ax.set_ylabel('功率 / MW')
        ax.set_title(f"{aid}：日内需求与供给堆叠图（MW）")
        ax.grid(axis='y', linestyle=':', alpha=0.6)
        ax.legend(ncol=3, loc='upper right')
        plt.tight_layout()
        out = os.path.join(save_dir, f"{filename_prefix}{aid}.png")
        plt.savefig(out, dpi=200)
        plt.close(fig)
        saved.append(out)
    return saved


def plot_daily_stack_per_agent_grid(per: Dict[str, Dict[str, np.ndarray]],
                                    save_path: str = "daily_agents_grid.png",
                                    title: str = "各智能体日内需求与供给（MW）") -> str:
    """将所有智能体画在一张大图的四个子图（2x2）中。若智能体不是 4 个，会按需要排版。"""
    agent_ids = list(per.keys())
    n = len(agent_ids)
    # 计算网格行列（最多画到 2x2；>4 时自动扩展）
    import math
    cols = 2 if n > 1 else 1
    rows = math.ceil(n / cols)

    fig, axes = plt.subplots(rows, cols, figsize=(14, 6*rows), squeeze=False)
    fig.suptitle(title)

    # 统一图例元素名
    legend_labels = ['可再生出力','电池放电','锅炉发电','内部购电','外网购电','需求（L）']
    handles_sample = None

    for idx, aid in enumerate(agent_ids):
        r, c = divmod(idx, cols)
        ax = axes[r][c]
        dd = per[aid]
        hours = len(dd['demand'])
        x = np.arange(hours)
        s1, s2, s3, s4, s5 = dd['renew'], dd['bat_dis'], dd['boiler'], dd['market_buy'], dd['grid_buy']

        h1 = ax.bar(x, s1, width=0.8)
        h2 = ax.bar(x, s2, bottom=s1, width=0.8)
        h3 = ax.bar(x, s3, bottom=s1 + s2, width=0.8)
        h4 = ax.bar(x, s4, bottom=s1 + s2 + s3, width=0.8)
        h5 = ax.bar(x, s5, bottom=s1 + s2 + s3 + s4, width=0.8)
        l6, = ax.plot(x, dd['demand'], linestyle='--', linewidth=2.0)

        if handles_sample is None:
            handles_sample = [h1, h2, h3, h4, h5, l6]

        ax.set_xticks(x)
        ax.set_xticklabels([f"{h:02d}:00" for h in range(hours)])
        ax.set_ylabel('功率 / MW')
        ax.set_title(f"{aid}")
        ax.grid(axis='y', linestyle=':', alpha=0.6)

    # 删除空子图（当 n 不是 rows*cols 时）
    for k in range(n, rows*cols):
        r, c = divmod(k, cols)
        fig.delaxes(axes[r][c])

    # 放一个全局图例在底部
    if handles_sample is not None:
        fig.legend(handles_sample, legend_labels, loc='lower center', ncol=3)
        fig.subplots_adjust(bottom=0.08)

    fig.tight_layout(rect=[0, 0.05, 1, 0.96])
    plt.savefig(save_path, dpi=220)
    plt.close(fig)
    return save_path


# ----------------------------
# Main entry: evaluation + plot
# ----------------------------
def test_model_and_plot(algo:str="iddpg",
                        train_days: int = 31,
                        test_days: int = 1,
                        plot_day_offset: int = 0,
                        gamma: float = 0.99,
                        tau: float = 0.01,
                        batch_size: int = 256,
                        buffer_size: int = 200_000) -> Tuple[List[np.ndarray], List[np.ndarray]]:
    """
    1) 构建训练/测试环境
    2) 用已训练好的 model 策略在测试集上前向评估
    3) 从测试集选定的一天（plot_day_offset）绘制 24 小时堆叠柱图
    """
    # === 数据切分 ===
    data = load_power_data("./data/GridSet_no_pred.csv")
    T = len(data[0]["P"])  # 每个园区长度一致

    train_idx = train_days * 24
    test_idx = (train_days + test_days) * 24

    train_series = [{k: v[:train_idx] for k, v in d.items()} for d in data]
    test_series = [{k: v[train_idx:test_idx] for k, v in d.items()} for d in data]

    # === 环境 ===
    env = MultiBatteryCoordinator(
        train_series,
        n_agents=4,
        dt_hours=1.0,
        E_bat_MWh=10000.0,
        P_bat_max_MW=5000.0,
        eta_ch=0.95, eta_dis=0.95,
        soc_min=0.1, soc_max=0.9, soc_init=0.1,
        deg_cost_per_MW=0.1,
        obs_norm=True,
    )
    test_env = MultiBatteryCoordinator(
        test_series,
        n_agents=4,
        dt_hours=1.0,
        E_bat_MWh=10000.0,
        P_bat_max_MW=5000.0,
        eta_ch=0.95, eta_dis=0.95,
        soc_min=0.1, soc_max=0.9, soc_init=0.1,
        deg_cost_per_MW=0.1,
        obs_norm=True,
    )

    # === Agent ===
    obs, _ = test_env.reset()
    agents = test_env.agents
    obs_dims = [int(np.asarray(o).size) for o in _by_agents(obs, agents)]
    action_dims = [int(test_env.action_spaces[a].shape[0]) for a in agents]
    max_actions = [float(test_env.action_spaces[a].high[0]) for a in agents]
    if algo == "iddpg":
        model = IDDPG(
            obs_dims, action_dims, max_actions,
            lr_actor=1e-3, lr_critic=1e-3,
            gamma=gamma, tau=tau,
            batch_size=batch_size, buffer_size=buffer_size
        )
        # 假设本地已有训练权重
        
    elif algo == "maddpg":
        model = MADDPG(
            obs_dims, action_dims, max_actions,
            lr_actor=1e-3, lr_critic=1e-3,
            gamma=gamma, tau=tau,
            batch_size=batch_size, buffer_size=buffer_size
        )
    model.load(Fed=False)
    # === 先做一次完整测试集评估（可选）===
    test_rewards = []
    test_obs, _ = test_env.reset()
    ep_rew = np.zeros(len(agents), dtype=np.float32)
    for _ in range(test_idx - train_idx):
        al = model.select_actions(_by_agents(test_obs, agents), noise_scale=0.0)
        nd = {a: al[i] for i, a in enumerate(agents)}
        test_next_obs, test_rew_dict, test_term_dict, test_trunc_dict, _info = test_env.step(nd)
        ep_rew += np.array(_by_agents(test_rew_dict, agents), dtype=np.float32)
        test_obs = test_next_obs
        if all(bool(test_term_dict[a]) or bool(test_trunc_dict[a]) for a in agents):
            break
    test_rewards.append(ep_rew)

    # === 选取测试集中的第 plot_day_offset 天，绘图 ===
    # 该天在测试集内的起始索引（相对 test_env）
    day_start = plot_day_offset * 24
    agg = rollout_one_day_and_collect(test_env, model, agents, day_start_idx=day_start, dt_hours=1.0)
    plot_daily_stack(agg, title=f"测试集第 {plot_day_offset+1} 天：用电与供给堆叠图（MW）",
                     save_path=f"./result/{algo}/daily_supply_stack.png")

    # 分智能体：各出一张图
    per = rollout_one_day_per_agent(test_env, model, agents, day_start_idx=day_start, dt_hours=1.0)
    saved_files = plot_daily_stack_per_agent(per, save_dir=f"./result/{algo}", filename_prefix="daily_agent_")
    grid_path = plot_daily_stack_per_agent_grid(per, save_path=f"./result/{algo}/daily_agents_grid.png",
                                                title=f"测试集第 {plot_day_offset+1} 天：各智能体日内需求与供给（MW）")

    print("Saved plot -> daily_supply_stack.png")
    print(f"Saved plot -> {grid_path}")
    for p in saved_files:
        print(f"Saved plot -> {p}")
    return [], test_rewards


if __name__ == "__main__":
    # 创建参数解析器
    parser = argparse.ArgumentParser(description='测试模型并绘图的参数设置')

    # 添加参数
    parser.add_argument('--algo', type=str, default='maddpg',
                        help='算法名称，默认是maddpg')
    parser.add_argument('--train_days', type=int, default=31 * 12,
                        help='训练天数，默认是31*12')
    parser.add_argument('--test_days', type=int, default=4,
                        help='测试天数，默认是4')
    parser.add_argument('--plot_day_offset', type=int, default=1,
                        help='绘图偏移天数，默认是1')

    # 解析参数
    args = parser.parse_args()

    # 调用函数并传递参数
    _,re=test_model_and_plot(
        algo=args.algo,
        train_days=args.train_days,
        test_days=args.test_days,
        plot_day_offset=args.plot_day_offset
    )
    print(sum(re))