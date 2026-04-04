import numpy as np
import os
import argparse
from datetime import datetime as dt

# 仅从并行工具中导入工作函数，主脚本将负责管理进程池
from flcore.utils.parallel_train import train_worker
import multiprocessing as mp

def main(args):
    """主训练函数，根据命令行参数执行训练任务。"""

    # 将用户友好的名称映射到 train_worker 所需的参数元组
    # 格式: (algo_name, episodes, train_days, federated, fed_method)
    task_map = {
        'DSFA': ('iddpg', args.epochs, args.train_days, True, 'DSFA'),
        'FedAvg': ('iddpg', args.epochs, args.train_days, True, 'FedAvg'),
        'IDDPG_solo': ('iddpg', args.epochs, args.train_days, False, 'DSFA'), # fed_method 在此为占位符
        'MADDPG_solo': ('maddpg', args.epochs, args.train_days, False, None),
        'Fed_MADDPG': ('maddpg', args.epochs, args.train_days, True, None)
    }

    # 根据 --run 参数构建要执行的任务列表
    tasks_to_run = []
    valid_run_names = []
    if not args.run:
        print("错误: 请至少使用 --run 参数指定一个要运行的训练任务。")
        print(f"可用任务: {', '.join(task_map.keys())}")
        return

    for run_name in args.run:
        if run_name in task_map:
            tasks_to_run.append(task_map[run_name])
            valid_run_names.append(run_name)
        else:
            print(f"警告: 未知的运行任务 '{run_name}'，将被忽略。")
    
    if not tasks_to_run:
        print("没有有效的任务可运行。程序退出。")
        return

    # 用于保存结果的字典
    results_to_save = {}

    # --- 根据模式选择训练方式 ---
    if args.mode == 'parallel':
        # 确定并行工作进程数
        num_workers = min(len(tasks_to_run), mp.cpu_count(), args.workers)
        print(f"使用并行训练模式，并行数: {num_workers}...")
        
        with mp.Pool(processes=num_workers) as pool:
            # starmap 会为 tasks_to_run 中的每个元组启动一个 train_worker 进程
            results_list = pool.starmap(train_worker, tasks_to_run)
        
        # 将结果映射回其任务名称
        for i, run_name in enumerate(valid_run_names):
            # train_worker 返回 (rewards, test_rewards)
            rewards = results_list[i][0]
            results_to_save[run_name] = np.array(rewards).T

    else: # 'serial' 模式
        print("使用串行训练模式...")
        from flcore.train.train_iddpg import train_iddpg
        from flcore.train.train_maddpg import train_maddpg

        for i, run_name in enumerate(valid_run_names):
            print(f"\n{'='*10} 开始运行任务: {run_name} {'='*10}")
            
            algo_name, episodes, train_days, federated, fed_method = tasks_to_run[i]

            if algo_name == 'iddpg':
                rewards, _ = train_iddpg(episodes, train_days, 0, federated, fed_method)
            else: # maddpg
                rewards, _ = train_maddpg(episodes, train_days, 0, federated)
            
            results_to_save[run_name] = np.array(rewards).T
            print(f"{'='*10} 任务 {run_name} 运行结束 {'='*10}\n")

    # --- 保存结果 ---
    if not results_to_save:
        print("没有训练结果可以保存。")
        return
        
    now = dt.now().strftime('%Y%m%d')
    save_dir = f'./result/{now}'
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    file_path = os.path.join(save_dir, 'result_arrays')
    # 使用字典解包来动态保存所有运行的结果
    np.savez(
        file_path,
        **results_to_save
    )
    print(f"\n所有任务完成！结果已保存到 {file_path}.npz")
    print("保存的键包括:", list(results_to_save.keys()))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="运行联邦强化学习训练任务")

    parser.add_argument(
        '-r', '--run',
        type=str,
        action='append', # 允许多次使用此参数
        help=f"指定要运行的训练任务。可用选项: {', '.join(['DSFA', 'FedAvg', 'IDDPG_solo', 'MADDPG_solo', 'Fed_MADDPG'])}。可多次指定，例如: --run DSFA --run FedAvg"
    )
    parser.add_argument(
        '-e', '--epochs',
        type=int,
        default=1000,
        help="训练的轮次 (episodes)。默认: 1000"
    )
    parser.add_argument(
        '-d', '--train_days',
        type=int,
        default=365,
        help="用于训练的数据天数。默认: 365"
    )
    parser.add_argument(
        '-m', '--mode',
        type=str,
        default='parallel',
        choices=['parallel', 'serial'],
        help="训练模式 (parallel 或 serial)。默认: parallel"
    )
    parser.add_argument(
        '-w', '--workers',
        type=int,
        default=4,
        help="并行模式下的最大工作进程数。默认: 4"
    )

    # 解析命令行参数
    args = parser.parse_args()

    # 调用主函数
    main(args)
