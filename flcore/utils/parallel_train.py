import multiprocessing as mp
from functools import partial

def train_worker(algo_name, episodes, train_days, federated,fed_method):
    """单个训练进程"""
    if algo_name == 'ddpg':
        from flcore.train.train_ddpg import train_ddpg
        return train_ddpg(episodes, train_days, 0)
    elif algo_name == 'iddpg':
        from flcore.train.train_iddpg import train_iddpg
        return train_iddpg(episodes, train_days, 0, federated,fed_method)
    else:
        from flcore.train.train_maddpg import train_maddpg
        return train_maddpg(episodes, train_days, 0, federated)


def train_lost_contact_worker(
    fed_method: str,
    episodes: int,
    train_days: int,
    branch_episode: int,
    lost_agent: str,
    disconnect_schedule: tuple[tuple[int, int], ...],
    cpu_threads: int,
    output_root: str = ".",
) -> str:
    """在独立进程运行一个算法的断联对比实验。

    Args:
        fed_method: 当前进程负责的 DSFA 或 FedAvg。
        episodes: 总训练轮数。
        train_days: 训练数据天数。
        branch_episode: 共同阶段结束后的分支起点。
        lost_agent: 通信失联的园区。
        disconnect_schedule: 左闭右开的失联区间。
        cpu_threads: 此进程可使用的 PyTorch CPU 线程数。
        output_root: 模型与结果根目录，默认使用当前工作目录。

    Returns:
        已完成的算法名称；训练结果由训练函数写入算法专属目录。
    """
    import torch
    from flcore.train.train_iddpg import train_iddpg_lost_contact

    torch.set_num_threads(cpu_threads)
    train_iddpg_lost_contact(
        episodes=episodes,
        train=train_days,
        fed_method=fed_method,
        branch_episode=branch_episode,
        lost_agent=lost_agent,
        disconnect_schedule=disconnect_schedule,
        branch_cpu_threads=max(1, cpu_threads // 2),
        output_root=output_root,
    )
    return fed_method

def parallel_train_all( n_workers=4,
    tasks={
        ('iddpg', 500, 365, True,'DSFA'),
        ('iddpg', 500 , 365, True,'FedAvg')
        }
    ):
    """并行训练所有算法"""

    
    with mp.Pool(n_workers) as pool:
        results = pool.starmap(train_worker, tasks)
    
    return {
        'Fed_iddpg': results[0],
        'FedAvg_iddpg': results[1],
#        'maddpg': results[2],
#        'Fed_maddpg': results[3]
    }
