import numpy as np
import matplotlib.pyplot as plt
import os
from matplotlib import font_manager
from datetime import datetime as dt

# ==== Font Configuration ====
# Kept for compatibility, though we are using English now.
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

plt.rcParams['axes.unicode_minus'] = False
from typing import Dict, List, Tuple
import argparse

from flcore.Env.multi_env import MultiBatteryCoordinator
from flcore.train.train_common import (
    default_presets, load_series_split, build_envs,
    infer_dims, list_by_agents, flatten_obs, flatten_actions
)
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
        'R_wind': np.zeros(hours, dtype=np.float32),
        'R_solar': np.zeros(hours, dtype=np.float32),
        'bat_dis': np.zeros(hours, dtype=np.float32),
        'boiler': np.zeros(hours, dtype=np.float32),
        'P_CHP_e': np.zeros(hours, dtype=np.float32),
        'market_buy': np.zeros(hours, dtype=np.float32),
        'grid_buy': np.zeros(hours, dtype=np.float32),
        'surplus_dump': np.zeros(hours, dtype=np.float32),
    }

    for h in range(hours):
        acts = model.select_actions(_by_agents(obs, agents), noise_scale=0.0)
        action_dict = {a: acts[i] for i, a in enumerate(agents)}
        next_obs, rew_dict, term_dict, trunc_dict, info_dict = env.step(action_dict)

        L = 0.0;
        R = 0.0;
        bat_dis = 0.0;
        boiler = 0.0
        m_buy_MWh = 0.0;
        grid_buy_MWh = 0.0;
        dump_MWh = 0.0
        for aid in agents:
            inf = info_dict[aid]
            L += float(inf.get('G_demand', 0.0))
            R += float(inf.get('newpower_gen', 0.0))
            p_bat = float(inf.get('p_bat', 0.0))
            P_CHP_e = float(inf.get('P_CHP_e', 0.0))
            bat_dis += max(0.0, p_bat)
            boiler += max(0.0, float(inf.get('P_boiler_e', 0.0)))
            m_buy_MWh += float(inf.get('market_buy_MWh', 0.0))
            grid_buy_MWh += float(inf.get('grid_buy_MWh', 0.0))
            dump_MWh += float(inf.get('surplus_dump_MWh', 0.0))

        agg['demand'][h] = L
        agg['renew'][h] = R
        agg['bat_dis'][h] = bat_dis
        agg['boiler'][h] = boiler
        agg['P_CHP_e'][h] = P_CHP_e
        agg['market_buy'][h] = m_buy_MWh / max(1e-9, dt_hours)
        agg['grid_buy'][h] = grid_buy_MWh / max(1e-9, dt_hours)
        agg['surplus_dump'][h] = dump_MWh / max(1e-9, dt_hours)

        obs = next_obs
        if all(bool(term_dict[a]) or bool(trunc_dict[a]) for a in agents):
            for k in agg.keys():
                agg[k] = agg[k][:h + 1]
            break

    return agg


def rollout_one_day_per_agent(env: MultiBatteryCoordinator,
                              model,
                              agents: List[str],
                              day_start_idx: int,
                              dt_hours: float = 1.0) -> Dict[str, Dict[str, np.ndarray]]:
    obs, _ = env.reset()
    for _ in range(day_start_idx):
        zero_actions = {a: np.zeros(env.action_spaces[a].shape, dtype=np.float32) for a in agents}
        obs, _, term, trunc, _ = env.step(zero_actions)
        if all(bool(term[a]) or bool(trunc[a]) for a in agents):
            break

    hours = 24
    per = {a: {k: np.zeros(hours, dtype=np.float32) for k in [
        'G_demand', 'R_wind', 'R_solar', 'bat_dis', 'boiler', 'P_CHP_e', 'market_buy', 'grid_buy', 'surplus_dump',
        'H_demand', 'P_CHP_h', 'P_HB_h', 'h_grid_buy']}
           for a in agents}

    for h in range(hours):
        acts = model.select_actions(_by_agents(obs, agents), noise_scale=0.0)
        action_dict = {a: acts[i] for i, a in enumerate(agents)}
        next_obs, rew_dict, term_dict, trunc_dict, info_dict = env.step(action_dict)

        for aid in agents:
            inf = info_dict[aid]
            per[aid]['G_demand'][h] = float(inf.get('G_demand', 0.0))
            per[aid]['R_wind'][h] = float(inf.get('R_wind', 0.0))
            per[aid]['R_solar'][h] = float(inf.get('R_solar', 0.0))

            per[aid]['bat_dis'][h] = float(inf.get('p_bat', 0.0))
            per[aid]['boiler'][h] = max(0.0, float(inf.get('P_boiler_e', 0.0)))
            per[aid]['P_CHP_e'][h] = max(0.0, float(inf.get('P_CHP_e', 0.0)))
            per[aid]['market_buy'][h] = float(inf.get('market_buy_MWh', 0.0)) / max(1e-9, dt_hours)
            per[aid]['grid_buy'][h] = float(inf.get('grid_buy_MWh', 0.0)) / max(1e-9, dt_hours)
            per[aid]['surplus_dump'][h] = float(inf.get('surplus_dump_MWh', 0.0)) / max(1e-9, dt_hours)

            per[aid]['H_demand'][h] = float(inf.get('H_demand', 0.0))
            per[aid]['P_CHP_h'][h] = float(inf.get('P_CHP_h', 0.0))
            per[aid]['P_HB_h'][h] = float(inf.get('P_HB_h', 0.0))
            per[aid]['h_grid_buy'][h] = float(inf.get('h_grid_buy', 0.0))

        obs = next_obs
        if all(bool(term_dict[a]) or bool(trunc_dict[a]) for a in agents):
            for aid in agents:
                for k in per[aid].keys():
                    per[aid][k] = per[aid][k][:h + 1]
            break

    return per


# ==========================================
# Colors and Plotting Functions
# ==========================================

E_COLORS = {
    'R_wind': '#4DB6AC',
    'R_solar': '#FFEE58',
    'bat_dis': '#FFC107',
    'boiler': '#FF7043',
    'P_CHP_e': '#CE93D8',
    'market_buy': '#29B6F6',
    'grid_buy': '#E53935',
    'demand': '#333333'
}

H_COLORS = {
    'P_CHP_h': '#CE93D8',
    'P_HB_h': '#FF9800',
    'h_grid_buy': '#E53935',
    'demand': '#333333'
}


def _apply_modern_style(ax):
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_color('#cccccc')
    ax.spines['bottom'].set_color('#cccccc')
    ax.grid(axis='y', linestyle='--', alpha=0.4, color='#888888', zorder=0)
    ax.tick_params(colors='#555555', labelsize=10)
    ax.set_ylabel('Power (MW)', fontsize=11, color='#333333')


def plot_daily_stack(agg: Dict[str, np.ndarray],
                     title: str = "Intraday Electricity Demand and Supply Components (MW)",
                     save_path: str = "daily_supply_stack.png") -> None:
    hours = len(agg['demand'])
    x = np.arange(hours)

    fig, ax = plt.subplots(figsize=(14, 5))

    b1 = ax.bar(x, agg['R_wind'], label='Wind Power', color=E_COLORS['R_wind'], width=0.75, zorder=3)
    bottom = agg['R_wind'].copy()
    b2 = ax.bar(x, agg['R_solar'], bottom=bottom, label='Solar Power', color=E_COLORS['R_solar'], width=0.75, zorder=3)
    bottom += agg['R_solar']
    b3 = ax.bar(x, agg['bat_dis'], bottom=bottom, label='Battery Disch.', color=E_COLORS['bat_dis'], width=0.75,
                zorder=3)
    bottom += agg['bat_dis']
    b4 = ax.bar(x, agg['boiler'], bottom=bottom, label='Elec. Boiler', color=E_COLORS['boiler'], width=0.75, zorder=3)
    bottom += agg['boiler']
    b5 = ax.bar(x, agg['P_CHP_e'], bottom=bottom, label='CHP (Elec.)', color=E_COLORS['P_CHP_e'], width=0.75, zorder=3)
    bottom += agg['P_CHP_e']
    b6 = ax.bar(x, agg['market_buy'], bottom=bottom, label='Market Buy', color=E_COLORS['market_buy'], width=0.75,
                zorder=3)
    bottom += agg['market_buy']
    b7 = ax.bar(x, agg['grid_buy'], bottom=bottom, label='Grid Buy', color=E_COLORS['grid_buy'], width=0.75, zorder=3)

    ax.plot(x, agg['demand'], linestyle='--', linewidth=2.5, color=E_COLORS['demand'], label='Elec. Demand', zorder=4)

    step = max(1, hours // 8)
    ax.set_xticks(x[::step])
    ax.set_xticklabels([f"{h:02d}:00" for h in x][::step])
    ax.set_title(title, fontsize=14, fontweight='bold', pad=15)
    _apply_modern_style(ax)

    ax.legend(ncol=8, loc='upper center', bbox_to_anchor=(0.5, -0.15), frameon=False)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close(fig)


def plot_daily_stack_per_agent(per: Dict[str, Dict[str, np.ndarray]],
                               save_dir: str = ".",
                               filename_prefix: str = "daily_agent_") -> List[str]:
    os.makedirs(save_dir, exist_ok=True)
    saved = []
    for aid, dd in per.items():
        hours = len(dd['G_demand'])
        x = np.arange(hours)

        fig, ax = plt.subplots(figsize=(14, 5))

        ax.bar(x, dd['R_wind'], label='Wind Power', color=E_COLORS['R_wind'], width=0.75, zorder=3)
        bottom = dd['R_wind'].copy()
        ax.bar(x, dd['R_solar'], bottom=bottom, label='Solar Power', color=E_COLORS['R_solar'], width=0.75, zorder=3)
        bottom += dd['R_solar']
        ax.bar(x, dd['bat_dis'], bottom=bottom, label='Battery Disch.', color=E_COLORS['bat_dis'], width=0.75, zorder=3)
        bottom += dd['bat_dis']
        ax.bar(x, dd['boiler'], bottom=bottom, label='Elec. Boiler', color=E_COLORS['boiler'], width=0.75, zorder=3)
        bottom += dd['boiler']
        ax.bar(x, dd['P_CHP_e'], bottom=bottom, label='CHP (Elec.)', color=E_COLORS['P_CHP_e'], width=0.75, zorder=3)
        bottom += dd['P_CHP_e']
        ax.bar(x, dd['market_buy'], bottom=bottom, label='Market Buy', color=E_COLORS['market_buy'], width=0.75,
               zorder=3)
        bottom += dd['market_buy']
        ax.bar(x, dd['grid_buy'], bottom=bottom, label='Grid Buy', color=E_COLORS['grid_buy'], width=0.75, zorder=3)

        ax.plot(x, dd['G_demand'], linestyle='--', linewidth=2.5, color=E_COLORS['demand'], label='Elec. Demand',
                zorder=4)

        step = max(1, hours // 8)
        ax.set_xticks(x[::step])
        ax.set_xticklabels([f"{h:02d}:00" for h in x][::step])
        ax.set_title(f"{aid}: Intraday Demand and Supply Stack (MW)", fontsize=14, fontweight='bold', pad=15)
        _apply_modern_style(ax)

        ax.legend(ncol=8, loc='upper center', bbox_to_anchor=(0.5, -0.15), frameon=False)
        plt.tight_layout()

        out = os.path.join(save_dir, f"{filename_prefix}{aid}.png")
        plt.savefig(out, dpi=300, bbox_inches='tight')
        plt.close(fig)
        saved.append(out)
    return saved


def plot_daily_stack_per_agent_grid(per: Dict[str, Dict[str, np.ndarray]],
                                    save_path: str = "daily_agents_grid.png",
                                    title: str = "Intraday Electricity Demand and Supply for Each Agent (MW)") -> str:
    agent_ids = list(per.keys())
    n = len(agent_ids)
    import math
    cols = 2 if n > 1 else 1
    rows = math.ceil(n / cols)

    fig, axes = plt.subplots(rows, cols, figsize=(14, 5 * rows), squeeze=False)
    fig.suptitle(title, fontsize=16, fontweight='bold', y=0.98)

    legend_labels = ['Wind Power', 'Solar Power', 'Battery Disch.', 'Elec. Boiler', 'CHP (Elec.)', 'Market Buy',
                     'Grid Buy', 'Elec. Demand']
    handles_sample = None

    for idx, aid in enumerate(agent_ids):
        r, c = divmod(idx, cols)
        ax = axes[r][c]
        dd = per[aid]
        hours = len(dd['G_demand'])
        x = np.arange(hours)

        h1 = ax.bar(x, dd['R_wind'], color=E_COLORS['R_wind'], width=0.75, zorder=3)
        bottom = dd['R_wind'].copy()
        h2 = ax.bar(x, dd['R_solar'], bottom=bottom, color=E_COLORS['R_solar'], width=0.75, zorder=3)
        bottom += dd['R_solar']
        h3 = ax.bar(x, dd['bat_dis'], bottom=bottom, color=E_COLORS['bat_dis'], width=0.75, zorder=3)
        bottom += dd['bat_dis']
        h4 = ax.bar(x, dd['boiler'], bottom=bottom, color=E_COLORS['boiler'], width=0.75, zorder=3)
        bottom += dd['boiler']
        h5 = ax.bar(x, dd['P_CHP_e'], bottom=bottom, color=E_COLORS['P_CHP_e'], width=0.75, zorder=3)
        bottom += dd['P_CHP_e']
        h6 = ax.bar(x, dd['market_buy'], bottom=bottom, color=E_COLORS['market_buy'], width=0.75, zorder=3)
        bottom += dd['market_buy']
        h7 = ax.bar(x, dd['grid_buy'], bottom=bottom, color=E_COLORS['grid_buy'], width=0.75, zorder=3)

        l8, = ax.plot(x, dd['G_demand'], linestyle='--', linewidth=2.5, color=E_COLORS['demand'], zorder=4)

        if handles_sample is None:
            handles_sample = [h1, h2, h3, h4, h5, h6, h7, l8]

        step = max(1, hours // 8)
        ax.set_xticks(x[::step])
        ax.set_xticklabels([f"{h:02d}:00" for h in x][::step])
        ax.set_title(f"{aid}", fontsize=13, pad=10)
        _apply_modern_style(ax)

    for k in range(n, rows * cols):
        r, c = divmod(k, cols)
        fig.delaxes(axes[r][c])

    fig.subplots_adjust(bottom=0.12)
    if handles_sample is not None:
        fig.legend(handles_sample, legend_labels, loc='lower center', ncol=8, frameon=False, bbox_to_anchor=(0.5, 0.02))

    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close(fig)
    return save_path


def plot_daily_stack_per_agent_H_grid(per: Dict[str, Dict[str, np.ndarray]],
                                      save_path: str = "daily_agents_grid.png",
                                      title: str = "Intraday Thermal Demand and Supply for Each Agent (MW)") -> str:
    agent_ids = list(per.keys())
    n = len(agent_ids)
    import math
    cols = 2 if n > 1 else 1
    rows = math.ceil(n / cols)

    fig, axes = plt.subplots(rows, cols, figsize=(14, 5 * rows), squeeze=False)
    fig.suptitle(title, fontsize=16, fontweight='bold', y=0.98)

    legend_labels = ['CHP (Therm.)', 'Heat Pump', 'Heat Grid Buy', 'Heat Demand']
    handles_sample = None

    for idx, aid in enumerate(agent_ids):
        r, c = divmod(idx, cols)
        ax = axes[r][c]
        dd = per[aid]
        hours = len(dd['H_demand'])
        x = np.arange(hours)

        h1 = ax.bar(x, dd['P_CHP_h'], color=H_COLORS['P_CHP_h'], width=0.75, zorder=3)
        h2 = ax.bar(x, dd['P_HB_h'], bottom=dd['P_CHP_h'], color=H_COLORS['P_HB_h'], width=0.75, zorder=3)
        h3 = ax.bar(x, dd['h_grid_buy'], bottom=dd['P_CHP_h'] + dd['P_HB_h'], color=H_COLORS['h_grid_buy'], width=0.75,
                    zorder=3)
        l6, = ax.plot(x, dd['H_demand'], linestyle='--', linewidth=2.5, color=H_COLORS['demand'], zorder=4)

        if handles_sample is None:
            handles_sample = [h1, h2, h3, l6]

        step = max(1, hours // 8)
        ax.set_xticks(x[::step])
        ax.set_xticklabels([f"{h:02d}:00" for h in x][::step])
        ax.set_title(f"{aid} - Thermal System", fontsize=13, pad=10)
        _apply_modern_style(ax)

    for k in range(n, rows * cols):
        r, c = divmod(k, cols)
        fig.delaxes(axes[r][c])

    fig.subplots_adjust(bottom=0.12)
    if handles_sample is not None:
        fig.legend(handles_sample, legend_labels, loc='lower center', ncol=4, frameon=False, bbox_to_anchor=(0.5, 0.02))

    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close(fig)
    return save_path


# ----------------------------
# Main entry: evaluation + plot
# ----------------------------
def test_model_and_plot(algo: str = "iddpg",
                        Fed: bool = False,
                        train: int = 31,
                        test: int = 1,
                        plot_day_offset: int = 0,
                        gamma: float = 0.99,
                        tau: float = 0.01,
                        batch_size: int = 256,
                        buffer_size: int = 200_000) -> Tuple[List[np.ndarray], List[np.ndarray], Dict[str, float]]:
    presets = default_presets()
    train_series, test_series, T, train_idx, test_idx = load_series_split(
        path1="./data/IES_data/G_demand.csv",
        path2="./data/IES_data/H_demand.csv",
        train_days=train,
        test_days=test
    )
    env, test_env = build_envs(train_series, test_series, presets.env_kwargs)
    obs_dims, action_dims, max_actions, agents = infer_dims(env)

    obs, _ = test_env.reset()

    if algo == "iddpg":
        model = IDDPG(
            obs_dims, action_dims, max_actions,
            lr_actor=1e-3, lr_critic=1e-3,
            gamma=gamma, tau=tau,
            batch_size=batch_size, buffer_size=buffer_size
        )
    else:
        model = MADDPG(
            obs_dims, action_dims, max_actions,
            shared_obs_indices=[-2, -1],
            **presets.algo_kwargs
        )
    model.load(Fed=Fed)

    test_rewards = []
    test_obs, _ = test_env.reset()
    ep_rew = np.zeros(len(agents), dtype=np.float32)

    co2_costs = {a: 0.0 for a in agents}

    for _ in range(len(test_idx)):
        al = model.select_actions(_by_agents(test_obs, agents), noise_scale=0.0)
        nd = {a: al[i] for i, a in enumerate(agents)}
        test_next_obs, test_rew_dict, test_term_dict, test_trunc_dict, _info = test_env.step(nd)
        ep_rew += np.array(_by_agents(test_rew_dict, agents), dtype=np.float32)

        for a in agents:
            co2_costs[a] += _info[a].get("co2_cost", 0.0)

        test_obs = test_next_obs
        if all(bool(test_term_dict[a]) or bool(test_trunc_dict[a]) for a in agents):
            break
    test_rewards.append(ep_rew)

    day_start = plot_day_offset * 24
    agg = rollout_one_day_and_collect(test_env, model, agents, day_start_idx=day_start, dt_hours=1.0)
    now = dt.now().strftime("%Y%m%d")
    savefile = f"./result/{now}/{algo}"

    os.makedirs(savefile, exist_ok=True)

    plot_daily_stack(agg, title=f"Test Set Day {plot_day_offset + 1}: Electricity Demand and Supply Stack (MW)",
                     save_path=f"{savefile}/daily_supply_stack.png")

    per = rollout_one_day_per_agent(test_env, model, agents, day_start_idx=day_start, dt_hours=1.0)
    saved_files = plot_daily_stack_per_agent(per, save_dir=f"{savefile}", filename_prefix="daily_agent_")

    G_grid_path = plot_daily_stack_per_agent_grid(per, save_path=f"{savefile}/daily_agents_Fed{Fed}.png",
                                                  title=f"Test Set Day {plot_day_offset + 1}: Intraday Electricity Demand and Supply for Each Agent (MW)")

    H_grid_path = plot_daily_stack_per_agent_H_grid(per, save_path=f"{savefile}/H_daily_agents_Fed{Fed}.png",
                                                    title=f"Test Set Day {plot_day_offset + 1}: Intraday Thermal Demand and Supply for Each Agent (MW)")

    print("Saved plot -> daily_supply_stack.png")
    print(f"Saved plot -> {G_grid_path}")
    for p in saved_files:
        print(f"Saved plot -> {p}")
    return [], test_rewards, co2_costs


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Testing Model and Plotting Parameter Configurations')

    parser.add_argument('--train_days', type=int, default=30 * 11,
                        help='Training days, default is 31*12')
    parser.add_argument('--test_days', type=int, default=4,
                        help='Testing days, default is 4')
    parser.add_argument('--plot_day_offset', type=int, default=1,
                        help='Plot day offset, default is 1')

    args = parser.parse_args()

    param_combinations = [
        {'algo': 'iddpg', 'Fed': True},
        {'algo': 'iddpg', 'Fed': False},
        {'algo': 'maddpg', 'Fed': False},
        {'algo': 'FedAvg', 'Fed': True},
    ]

    for idx, params in enumerate(param_combinations, 1):
        print(f"\n===== Executing Parameter Set {idx} =====")
        print(f"Current Parameters: {params}")

        _, re, co2_costs = test_model_and_plot(
            algo=params['algo'],
            Fed=params['Fed'],
            train=args.train_days,
            test=args.test_days,
            plot_day_offset=args.plot_day_offset
        )

        total_reward = sum(re)
        print(f"Total Cumulative Reward: {total_reward}")
        print(f"====================================")
        print("CO2 Emission Costs for each Agent:")
        for a, cost in co2_costs.items():
            print(f"  Agent {a}: {cost:.2f}")
        print(f"  System Total: {sum(co2_costs.values()):.2f}")
        print(f"====================================\n")
