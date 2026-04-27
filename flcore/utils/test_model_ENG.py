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

    hours = 24
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

        L = 0.0
        R = 0.0
        bat_dis = 0.0
        boiler = 0.0
        P_CHP_e = 0.0
        m_buy_MWh = 0.0
        grid_buy_MWh = 0.0
        dump_MWh = 0.0
        for aid in agents:
            inf = info_dict[aid]
            L += float(inf.get('G_demand', 0.0))
            R += float(inf.get('newpower_gen', 0.0))
            p_bat = float(inf.get('p_bat', 0.0))
            bat_dis += p_bat
            P_CHP_e += max(0.0, float(inf.get('P_CHP_e', 0.0)))
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
    'grid_buy': "#AC1400",
    'market_buy': '#00B894',
    'R_solar': '#F6C445',
    'R_wind': '#2D9CDB',
    'P_CHP_e': '#34495E',
    'boiler': '#E67E22',
    'bat_dis': '#8E44AD',
    'demand': '#1F3A8A'
}

H_COLORS = {
    'P_CHP_h': '#5B8FF9',
    'P_HB_h': '#61DDAA',
    'h_grid_buy': '#F6BD16',
    'demand': '#C0392B'
}

AGENT_PROFILE_NAMES = [
    "Residential Area",
    "Renewable Energy Park",
    "Residential Area",
    "Industrial Park",
]

plt.rcParams.update({
    'font.family': 'sans-serif',
    'font.sans-serif': ['DejaVu Sans', 'Arial', 'Liberation Sans'],
    'axes.titlesize': 14,
    'axes.labelsize': 12,
    'legend.fontsize': 10,
})


def _agent_profile_name(agent_id: str, fallback_idx: int) -> str:
    try:
        idx = int(str(agent_id).split('_')[-1])
    except (ValueError, IndexError):
        idx = fallback_idx
    if 0 <= idx < len(AGENT_PROFILE_NAMES):
        return AGENT_PROFILE_NAMES[idx]
    return f"Agent {fallback_idx + 1}"


def _set_hour_ticks(ax, hours: int) -> None:
    tick_positions = np.arange(0, hours, 2)
    ax.set_xticks(tick_positions)
    ax.set_xticklabels([f"{h + 1}:00" for h in tick_positions])


def _apply_reference_style(ax, y_label: str = "Power (MW)") -> None:
    ax.set_facecolor('white')
    ax.grid(axis='y', linestyle=':', linewidth=0.9, alpha=0.95, color='#C3CBD8', zorder=0)
    ax.grid(axis='x', visible=False)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_color('#5B6473')
    ax.spines['bottom'].set_color('#5B6473')
    ax.spines['left'].set_linewidth(1.0)
    ax.spines['bottom'].set_linewidth(1.0)
    ax.tick_params(axis='both', labelsize=10, colors='#2F3A4B')
    ax.axhline(0.0, color='#8B95A5', linewidth=0.9, zorder=1)
    ax.set_ylabel(y_label)


def _stack_component_areas(ax,
                           x: np.ndarray,
                           components: List[Tuple[str, np.ndarray, str]]) -> Tuple[List, List[str]]:
    pos_base = np.zeros_like(x, dtype=np.float32)
    neg_base = np.zeros_like(x, dtype=np.float32)
    handles, labels = [], []

    for label, values, color in components:
        vals = np.asarray(values, dtype=np.float32)
        pos = np.clip(vals, 0.0, None)
        neg = np.clip(vals, None, 0.0)

        patch = None
        if np.any(pos):
            bars_pos = ax.bar(
                x, pos, bottom=pos_base, width=0.84, color=color, alpha=0.72,
                edgecolor='white', linewidth=0.25, zorder=2
            )
            pos_base += pos
            if len(bars_pos) > 0:
                patch = bars_pos[0]
        if np.any(neg):
            bars_neg = ax.bar(
                x, neg, bottom=neg_base, width=0.84, color=color, alpha=0.72,
                edgecolor='white', linewidth=0.25, zorder=2
            )
            neg_base += neg
            if patch is None and len(bars_neg) > 0:
                patch = bars_neg[0]

        if patch is not None:
            handles.append(patch)
            labels.append(label)

    return handles, labels


def _plot_electric_balance(ax,
                           series: Dict[str, np.ndarray],
                           demand_key: str = 'G_demand',
                           panel_caption: str = "",
                           show_legend: bool = True) -> Tuple[List, List[str]]:
    hours = len(series[demand_key])
    x = np.arange(hours)
    zeros = np.zeros(hours, dtype=np.float32)
    components = [
        ('Upper Power Grid', np.asarray(series.get('grid_buy', zeros), dtype=np.float32), E_COLORS['grid_buy']),
        ('P2P Electric Trading', np.asarray(series.get('market_buy', zeros), dtype=np.float32), E_COLORS['market_buy']),
        ('Solar Power', np.asarray(series.get('R_solar', zeros), dtype=np.float32), E_COLORS['R_solar']),
        ('Wind Power', np.asarray(series.get('R_wind', zeros), dtype=np.float32), E_COLORS['R_wind']),
        ('CHP Electric Output', np.asarray(series.get('P_CHP_e', zeros), dtype=np.float32), E_COLORS['P_CHP_e']),
        ('Electric Boiler Output', np.asarray(series.get('boiler', zeros), dtype=np.float32), E_COLORS['boiler']),
        ('Battery Storage (EES)', np.asarray(series.get('bat_dis', zeros), dtype=np.float32), E_COLORS['bat_dis']),
    ]

    comp_handles, comp_labels = _stack_component_areas(ax, x, components)
    demand_line, = ax.plot(
        x, np.asarray(series[demand_key], dtype=np.float32),
        color=E_COLORS['demand'], linewidth=2.1, marker='s', markersize=4.8,
        markerfacecolor='white', markeredgewidth=1.2,
        label='Elec. Demand', zorder=4
    )

    handles = [demand_line] + comp_handles
    labels = ['Elec. Demand'] + comp_labels

    _set_hour_ticks(ax, hours)
    _apply_reference_style(ax, y_label='Power (MW)')
    ax.set_xlabel("Hour (h)", labelpad=6)
    if panel_caption:
        ax.set_title(panel_caption, loc='left', fontsize=12, fontweight='medium', pad=6)

    y_values = [np.asarray(series[demand_key], dtype=np.float32)] + [np.asarray(v, dtype=np.float32) for _, v, _ in
                                                                     components]
    y_min = min(float(np.min(v)) for v in y_values)
    y_max = max(float(np.max(v)) for v in y_values)
    if y_max <= y_min:
        y_max = y_min + 1.0
    pad = 0.08 * (y_max - y_min)
    ax.set_ylim(y_min - pad, y_max + pad)

    if show_legend:
        ax.legend(handles, labels, loc='upper right', ncol=1, frameon=True, edgecolor='#D0D7E2')

    return handles, labels


def _plot_thermal_balance(ax,
                          series: Dict[str, np.ndarray],
                          panel_caption: str = "",
                          show_legend: bool = True) -> Tuple[List, List[str]]:
    hours = len(series['H_demand'])
    x = np.arange(hours)
    zeros = np.zeros(hours, dtype=np.float32)
    components = [
        ('CHP Thermal Output', np.asarray(series.get('P_CHP_h', zeros), dtype=np.float32), H_COLORS['P_CHP_h']),
        ('Heat Pump Output', np.asarray(series.get('P_HB_h', zeros), dtype=np.float32), H_COLORS['P_HB_h']),
        ('Heat Grid Purchase', np.asarray(series.get('h_grid_buy', zeros), dtype=np.float32), H_COLORS['h_grid_buy']),
    ]

    comp_handles, comp_labels = _stack_component_areas(ax, x, components)
    demand_line, = ax.plot(
        x, np.asarray(series['H_demand'], dtype=np.float32),
        color=H_COLORS['demand'], linewidth=2.1, marker='^', markersize=5.0,
        markerfacecolor='white', markeredgewidth=1.0,
        label='Heat Demand', zorder=4
    )

    handles = [demand_line] + comp_handles
    labels = ['Heat Demand'] + comp_labels

    _set_hour_ticks(ax, hours)
    _apply_reference_style(ax, y_label='Power (MW)')
    ax.set_xlabel("Hour (h)", labelpad=6)
    if panel_caption:
        ax.set_title(panel_caption, loc='left', fontsize=12, fontweight='medium', pad=6)

    y_values = [np.asarray(series['H_demand'], dtype=np.float32)] + [np.asarray(v, dtype=np.float32) for _, v, _ in
                                                                     components]
    y_min = min(float(np.min(v)) for v in y_values)
    y_max = max(float(np.max(v)) for v in y_values)
    if y_max <= y_min:
        y_max = y_min + 1.0
    pad = 0.08 * (y_max - y_min)
    ax.set_ylim(y_min - pad, y_max + pad)

    if show_legend:
        ax.legend(handles, labels, loc='upper right', ncol=1, frameon=True, edgecolor='#D0D7E2')

    return handles, labels


def plot_daily_stack(agg: Dict[str, np.ndarray],
                     title: str = "Diurnal Electrical Power Balance Components (MW)",
                     save_path: str = "daily_supply_stack.png") -> None:
    fig, ax = plt.subplots(figsize=(14, 5), dpi=300)
    _plot_electric_balance(ax, agg, demand_key='demand', panel_caption="", show_legend=True)
    ax.set_title(title, fontweight='semibold', pad=12)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close(fig)


def plot_daily_stack_per_agent(per: Dict[str, Dict[str, np.ndarray]],
                               save_dir: str = ".",
                               filename_prefix: str = "daily_agent_") -> List[str]:
    os.makedirs(save_dir, exist_ok=True)
    saved = []
    for idx, (aid, dd) in enumerate(per.items()):
        fig, ax = plt.subplots(figsize=(14, 5), dpi=300)
        area_name = _agent_profile_name(aid, idx)
        panel_caption = f"({chr(97 + idx)}) {area_name} Electrical Power Balance"
        _plot_electric_balance(ax, dd, demand_key='G_demand', panel_caption=panel_caption, show_legend=True)
        plt.tight_layout()

        out = os.path.join(save_dir, f"{filename_prefix}{aid}.png")
        plt.savefig(out, dpi=300, bbox_inches='tight')
        plt.close(fig)
        saved.append(out)
    return saved


def plot_daily_stack_per_agent_grid(per: Dict[str, Dict[str, np.ndarray]],
                                    save_path: str = "daily_agents_grid.png",
                                    title: str = "Diurnal Electrical Power Balance by Agent (MW)") -> str:
    agent_ids = list(per.keys())
    n = len(agent_ids)
    import math
    cols = 2 if n > 1 else 1
    rows = math.ceil(n / cols)

    fig, axes = plt.subplots(rows, cols, figsize=(14, 5.6 * rows), squeeze=False, dpi=300)
    fig.suptitle(title, fontsize=16, fontweight='semibold', y=0.985)

    handles_sample, labels_sample = None, None

    for idx, aid in enumerate(agent_ids):
        r, c = divmod(idx, cols)
        ax = axes[r][c]
        dd = per[aid]
        area_name = _agent_profile_name(aid, idx)
        panel_caption = f"({chr(97 + idx)}) {area_name} Electrical Power Balance"
        handles, labels = _plot_electric_balance(ax, dd, demand_key='G_demand', panel_caption=panel_caption,
                                                 show_legend=False)
        if handles_sample is None:
            handles_sample, labels_sample = handles, labels

    for k in range(n, rows * cols):
        r, c = divmod(k, cols)
        fig.delaxes(axes[r][c])

    fig.subplots_adjust(left=0.07, right=0.99, top=0.84, bottom=0.08, hspace=0.30, wspace=0.20)
    if handles_sample is not None:
        fig.legend(handles_sample, labels_sample, loc='upper center', ncol=4, frameon=False,
                   bbox_to_anchor=(0.5, 0.935))

    plt.savefig(save_path, dpi=1200, bbox_inches='tight')
    plt.close(fig)
    return save_path


def plot_daily_stack_per_agent_H_grid(per: Dict[str, Dict[str, np.ndarray]],
                                      save_path: str = "daily_agents_grid.png",
                                      title: str = "Diurnal Thermal Power Balance by Agent (MW)") -> str:
    agent_ids = list(per.keys())
    n = len(agent_ids)
    import math
    cols = 2 if n > 1 else 1
    rows = math.ceil(n / cols)

    fig, axes = plt.subplots(rows, cols, figsize=(14, 5.6 * rows), squeeze=False, dpi=300)
    fig.suptitle(title, fontsize=16, fontweight='semibold', y=0.985)

    handles_sample, labels_sample = None, None

    for idx, aid in enumerate(agent_ids):
        r, c = divmod(idx, cols)
        ax = axes[r][c]
        dd = per[aid]
        area_name = _agent_profile_name(aid, idx)
        panel_caption = f"({chr(97 + idx)}) {area_name} Thermal Power Balance"
        handles, labels = _plot_thermal_balance(ax, dd, panel_caption=panel_caption, show_legend=False)
        if handles_sample is None:
            handles_sample, labels_sample = handles, labels

    for k in range(n, rows * cols):
        r, c = divmod(k, cols)
        fig.delaxes(axes[r][c])

    fig.subplots_adjust(left=0.07, right=0.99, top=0.84, bottom=0.08, hspace=0.30, wspace=0.20)
    if handles_sample is not None:
        fig.legend(handles_sample, labels_sample, loc='upper center', ncol=4, frameon=False,
                   bbox_to_anchor=(0.5, 0.935))

    plt.savefig(save_path, dpi=1200, bbox_inches='tight')
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

    plot_daily_stack(agg, title=f"Test Day {plot_day_offset + 1}: Diurnal Electrical Power Balance (MW)",
                     save_path=f"{savefile}/daily_supply_stack.png")

    per = rollout_one_day_per_agent(test_env, model, agents, day_start_idx=day_start, dt_hours=1.0)
    saved_files = plot_daily_stack_per_agent(per, save_dir=f"{savefile}", filename_prefix="daily_agent_")

    G_grid_path = plot_daily_stack_per_agent_grid(per, save_path=f"{savefile}/daily_agents_Fed{Fed}.png",
                                                  title=f"Test Day {plot_day_offset + 1}: Diurnal Electrical Power Balance by Agent (MW)")

    H_grid_path = plot_daily_stack_per_agent_H_grid(per, save_path=f"{savefile}/H_daily_agents_Fed{Fed}.png",
                                                    title=f"Test Day {plot_day_offset + 1}: Diurnal Thermal Power Balance by Agent (MW)")

    print("Saved plot -> daily_supply_stack.png")
    print(f"Saved plot -> {G_grid_path}")
    for p in saved_files:
        print(f"Saved plot -> {p}")
    return [], test_rewards, co2_costs


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Testing Model and Plotting Parameter Configurations')

    parser.add_argument('--train_days', type=int, default=30 * 11,
                        help='Training days, default is 31*12')
    parser.add_argument('--test_days', type=int, default=20,
                        help='Testing days, default is 4')
    parser.add_argument('--plot_day_offset', type=int, default=50,
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
