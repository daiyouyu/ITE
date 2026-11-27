"""按论文的环境数据循环平移设置，评估训练首日的模型鲁棒性。"""

import argparse
import csv
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from flcore.Env.multi_env import MultiBatteryCoordinator
from flcore.train.train_common import default_presets, list_by_agents, load_series_split
from flcore.utils.dispatch_evaluation import (
    FEDERATED_ALGORITHMS, HETEROGENEOUS_ALGORITHMS, STANDARD_ALGORITHMS,
    evaluation_presets, load_actor_model, resolve_evaluation_run,
)


ROTATED_KEYS = ("G_L", "H_L", "R", "R_solar", "R_wind")


def rotate_environment(series: list[dict]) -> list[dict]:
    """只循环交换负荷和新能源序列，园区设备、价格与时间序列保持原位。

    第 i 个园区使用第 (i+1) 个园区的数据，最后一个使用第一个。
    返回新字典，不修改原始训练日数据。
    """
    if len(series) != 4:
        raise ValueError(f"论文环境轮转要求 4 个园区，实际为 {len(series)}")
    rotated = []
    for index, park in enumerate(series):
        source = series[(index + 1) % len(series)]
        new_park = dict(park)
        for key in ROTATED_KEYS:
            new_park[key] = source[key]
        rotated.append(new_park)
    return rotated


def load_model(model_dir: Path, algorithm: str, env: MultiBatteryCoordinator, *,
               action_config: dict[str, list[str]] | None = None):
    """为轮转评估加载指定算法的 Actor 权重。

    Args:
        model_dir: 当前算法的 models 目录。
        algorithm: 批次中保存模型的算法名称。
        env: 使用原始设备配置建立的评估环境。
        action_config: 异构模型保存的园区动作清单，旧模型可省略。

    Returns:
        可调用 select_actions 的模型，执行时不加入探索噪声。
    """
    return load_actor_model(
        model_dir, env, algorithm="MADDPG" if algorithm == "MADDPG" else "IDDPG",
        federated=algorithm in FEDERATED_ALGORITHMS, algo_kwargs=default_presets().algo_kwargs,
        action_config=action_config,
    )


def evaluate_day(model, series: list[dict], env_kwargs: dict) -> dict:
    """在训练首日无探索噪声运行 24 小时，汇总各园区费用与碳税。"""
    env = MultiBatteryCoordinator(series, **env_kwargs)
    try:
        obs, _ = env.reset()
        agents = env.agents
        costs = {agent: 0.0 for agent in agents}
        carbon_costs = {agent: 0.0 for agent in agents}
        steps = 0
        for _ in range(24):
            actions = model.select_actions(list_by_agents(obs, agents), noise_scale=0.0)
            obs, _, terminated, truncated, info = env.step(dict(zip(agents, actions)))
            for agent in agents:
                costs[agent] += float(info[agent]["total_cost"])
                carbon_costs[agent] += float(info[agent]["co2_cost"])
            steps += 1
            if all(terminated[agent] or truncated[agent] for agent in agents):
                break
        return {"hours": steps, "park_costs_cny": costs,
                "park_carbon_costs_cny": carbon_costs,
                "system_cost_cny": sum(costs.values()),
                "system_carbon_cost_cny": sum(carbon_costs.values())}
    finally:
        env.close()


def format_rotation_table(results: dict, algorithms: tuple[str, ...] = STANDARD_ALGORITHMS) -> str:
    """按论文表 2 的列顺序整理四个轮转园区、系统总值和增幅。"""
    header = (
        "| Algorithm | Agent 1 (Scene 2) | Agent 2 (Scene 3) | "
        "Agent 3 (Scene 4) | Agent 4 (Scene 1) | Total (CNY) |"
    )
    lines = ["# Dispatch costs under cyclic environment swaps", ""]
    for title, park_key, system_key, change_key in (
        ("Daily Scheduling Cost", "park_costs_cny", "system_cost_cny",
         "system_cost_change_percent"),
        ("Carbon Tax Cost", "park_carbon_costs_cny", "system_carbon_cost_cny",
         "system_carbon_cost_change_percent"),
    ):
        lines.extend([f"## {title}", "", header,
                      "| --- | ---: | ---: | ---: | ---: | ---: |"])
        for algorithm in algorithms:
            record = results[algorithm]
            rotated = record["rotated"]
            parks = [f"{rotated[park_key][f'agent_{index}']:,.2f}"
                     for index in range(4)]
            change = record[change_key]
            change_text = "" if change is None else f" ({change:+.2f}%)"
            label = f"{algorithm} (Ours)" if algorithm == algorithms[-1] else algorithm
            lines.append(
                f"| {label} | {' | '.join(parks)} | "
                f"{rotated[system_key]:,.2f}{change_text} |"
            )
        lines.append("")
    return "\n".join(lines)


def format_original_table(results: dict, algorithms: tuple[str, ...] = STANDARD_ALGORITHMS) -> str:
    """按论文原始环境表格的算法顺序输出总调度费用和碳税。"""
    ours_label = f"{algorithms[-1]} (Ours)"
    algorithm_width = max(15, len(ours_label))
    lines = ["Daily Scheduling Costs and Carbon Tax in the Original Environment (CNY)",
             f"{'Algorithm':<{algorithm_width}} {'Total Cost':>12} {'Carbon Tax':>16}"]
    for algorithm in (algorithms[-1], *algorithms[:-1]):
        original = results[algorithm]["original"]
        label = ours_label if algorithm == algorithms[-1] else algorithm
        lines.append(
            f"{label:<{algorithm_width}} {original['system_cost_cny']:>12,.2f} "
            f"{original['system_carbon_cost_cny']:>16,.2f}"
        )
    return "\n".join(lines)


def main() -> None:
    """加载同批次四个训练模型，对原始日和数据轮转日分别评估并保存结果。"""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_dir", type=Path,
                        help="4.2.1 或异构实验批次目录，也可指定异构批次下的 seed_N 目录")
    parser.add_argument("--seed", type=int, help="异构实验种子，批次只有一个种子时自动选择")
    parser.add_argument("--output", type=Path,
                        help="结果目录，默认使用模型所在批次或种子目录下的 environment_rotation")
    args = parser.parse_args()
    run_dir, config, algorithms = resolve_evaluation_run(args.run_dir, args.seed)
    output_dir = args.output or run_dir / "environment_rotation"
    if config["train_days"] != 1:
        raise ValueError("环境轮转测试目前只支持训练数据为一天的实验")
    # 读取训练配置中的日期，确保原始环境与轮转环境使用同一天。
    train_series, _, _, train_indices, _ = load_series_split(
        path1=str(PROJECT_ROOT / "data/IES_data/G_demand.csv"),
        path2=str(PROJECT_ROOT / "data/IES_data/H_demand.csv"),
        train_days=1,
        test_days=0,
        train_start_date=config.get("train_date"),
    )
    if len(train_indices) != 24:
        raise ValueError(f"训练首日应为 24 小时，实际为 {len(train_indices)}")
    rotated_series = rotate_environment(train_series)
    results = {}
    rows = []
    for algorithm in algorithms:
        model_dir = run_dir / algorithm / "models"
        presets, action_config = evaluation_presets(
            model_dir, compact_actions=algorithms == HETEROGENEOUS_ALGORITHMS,
        )
        reference_env = MultiBatteryCoordinator(train_series, **presets.env_kwargs)
        try:
            model = load_model(model_dir, algorithm, reference_env, action_config=action_config)
        finally:
            reference_env.close()
        original = evaluate_day(model, train_series, presets.env_kwargs)
        rotated = evaluate_day(model, rotated_series, presets.env_kwargs)
        results[algorithm] = {
            "original": original,
            "rotated": rotated,
            "system_cost_change_percent": (
                (rotated["system_cost_cny"] / original["system_cost_cny"] - 1) * 100
                if original["system_cost_cny"] else None
            ),
            "system_carbon_cost_change_percent": (
                (rotated["system_carbon_cost_cny"] /
                 original["system_carbon_cost_cny"] - 1) * 100
                if original["system_carbon_cost_cny"] else None
            ),
        }
        for scenario, values in (("original", original), ("rotated", rotated)):
            rows.append({"algorithm": algorithm, "scenario": scenario,
                         "system_cost_cny": values["system_cost_cny"],
                         "system_carbon_cost_cny": values["system_carbon_cost_cny"],
                         **{f"{agent}_cost_cny": cost
                            for agent, cost in values["park_costs_cny"].items()},
                         **{f"{agent}_carbon_cost_cny": cost
                            for agent, cost in values["park_carbon_costs_cny"].items()}})
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "results.json").write_text(
        json.dumps({"rotation": "agent_0<-agent_1<-agent_2<-agent_3<-agent_0",
                    "rotated_keys": ROTATED_KEYS, "results": results},
                   ensure_ascii=False, indent=2), encoding="utf-8")
    with (output_dir / "results.csv").open("w", newline="", encoding="utf-8-sig") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    table = format_rotation_table(results, algorithms)
    (output_dir / "table_2.md").write_text(table, encoding="utf-8")
    original_table = format_original_table(results, algorithms)
    (output_dir / "original_environment_table.txt").write_text(
        original_table + "\n", encoding="utf-8")
    print(original_table)
    print()
    print(table)
    print(f"结果已保存至：{output_dir}")


if __name__ == "__main__":
    main()
