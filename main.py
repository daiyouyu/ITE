import numpy as np
import os
import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime as dt

# 仅从并行工具中导入工作函数，主脚本将负责管理进程池
from flcore.utils.parallel_train import train_worker, train_lost_contact_worker
from flcore.utils.memory_guard import GIB, safe_training_jobs
import multiprocessing as mp

def main(args):
    """主训练函数，根据命令行参数执行训练任务。"""

    # 将用户友好的名称映射到 train_worker 所需的参数元组
    # 格式: (algo_name, episodes, train_days, federated, fed_method)
    task_map = {
        'DDPG': ('ddpg', args.epochs, args.train_days, False, None),
        'DSFA': ('iddpg', args.epochs, args.train_days, True, 'DSFA'),
        'AllDSFA': ('iddpg', args.epochs, args.train_days, True, 'AllDSFA'),
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

    if args.lost_contact:
        # 断联实验独立保存全部分支结果，避免改写原有训练输出。
        unsupported = [name for name in valid_run_names if name not in ("DSFA", "FedAvg")]
        if unsupported:
            raise ValueError(f"断联对比实验仅支持 DSFA 和 FedAvg: {unsupported}")
        if len(set(valid_run_names)) != len(valid_run_names):
            raise ValueError("断联对比实验不能重复指定同一算法，以免覆盖同一目录")
        if args.workers < 1:
            raise ValueError("--workers 必须大于或等于 1")
        requested_jobs = min(getattr(args, "lost_contact_jobs", 2), args.workers)
        safe_jobs, available_memory = safe_training_jobs(requested_jobs)
        if safe_jobs < requested_jobs:
            available_gib = "未知" if available_memory is None else f"{available_memory / GIB:.1f} GiB"
            print(f"内存保护：可用内存 {available_gib}，断联训练并发数降为 {safe_jobs}")

        branch_episode = 200
        lost_agent = "agent_0"
        disconnect_schedule = ((200, 400), (600, 800))
        num_workers = min(len(valid_run_names), mp.cpu_count(), args.workers, safe_jobs)
        if args.mode == 'serial':
            num_workers = 1
        cpu_threads = max(1, mp.cpu_count() // num_workers)
        snapshots = {}
        try:
            # 两阶段屏障：全部算法先完成普通协作，之后才启动任一断联分支。
            for phase in ("normal", "disconnected"):
                print(f"断联实验阶段 {phase}: {', '.join(valid_run_names)}，并发数: {num_workers}")
                if args.mode == 'parallel' and num_workers > 1:
                    with ProcessPoolExecutor(
                        max_workers=num_workers,
                        mp_context=mp.get_context("spawn"),
                    ) as executor:
                        futures = {
                            executor.submit(
                                train_lost_contact_worker,
                                name, args.epochs, args.train_days, branch_episode,
                                lost_agent, disconnect_schedule, cpu_threads, phase,
                                snapshots.get(name), ".",
                            ): name for name in valid_run_names
                        }
                        errors = []
                        for future in as_completed(futures):
                            try:
                                name, snapshot_ref = future.result()
                                if snapshot_ref is not None:
                                    snapshots[name] = snapshot_ref
                            except Exception as exc:
                                errors.append(exc)
                        if errors:
                            raise errors[0]
                else:
                    from flcore.train.train_iddpg import train_iddpg

                    for name in valid_run_names:
                        result = train_iddpg(
                            args.epochs, args.train_days, 0, True, name,
                            lost_contact=True, lost_contact_phase=phase,
                            branch_episode=branch_episode, lost_agent=lost_agent,
                            disconnect_schedule=disconnect_schedule,
                            snapshot_ref=snapshots.get(name),
                        )
                        if phase == "normal":
                            snapshots[name] = result
            print(f"断联实验完成: {', '.join(valid_run_names)}")
        finally:
            snapshots.clear()
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
        from flcore.train.train_ddpg import train_ddpg

        for i, run_name in enumerate(valid_run_names):
            print(f"\n{'='*10} 开始运行任务: {run_name} {'='*10}")
            
            algo_name, episodes, train_days, federated, fed_method = tasks_to_run[i]

            if algo_name == 'ddpg':
                rewards, _ = train_ddpg(episodes, train_days, 0)
            elif algo_name == 'iddpg':
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
        help=f"指定要运行的训练任务。可用选项: {', '.join(['DDPG', 'DSFA', 'AllDSFA', 'FedAvg', 'IDDPG_solo', 'MADDPG_solo', 'Fed_MADDPG'])}。可多次指定，例如: --run DDPG --run DSFA"
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
    parser.add_argument(
        '--lost-contact',
        action='store_true',
        default=False,
        help="启用园区 1 两次失联、两次重连的共同训练及分支对比实验。",
    )
    parser.add_argument(
        '--lost-contact-jobs',
        type=int,
        default=2,
        help="断联实验同时运行的重训练进程上限，默认 2；内存不足时自动降低。",
    )

    # 解析命令行参数
    args = parser.parse_args()

    # 调用主函数
    main(args)
