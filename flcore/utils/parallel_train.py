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
    phase: str,
    snapshot_ref: bytes | None = None,
    output_root: str = ".",
) -> tuple[str, bytes | None]:
    """在独立进程调用统一的 IDDPG 入口，执行指定的实验阶段。

    Args:
        fed_method: 当前进程负责的 DSFA 或 FedAvg。
        episodes: 总训练轮数。
        train_days: 训练数据天数。
        branch_episode: 共同阶段结束后的分支起点。
        lost_agent: 通信失联的园区。
        disconnect_schedule: 左闭右开的失联区间。
        cpu_threads: 此进程可使用的 PyTorch CPU 线程数。
        output_root: 模型与结果根目录，默认使用当前工作目录。
        phase: normal 或 disconnected。
        snapshot_ref: 正常阶段返回的内存状态引用。

    Returns:
        算法名称与正常阶段的内存状态；结果另存至算法目录。
    """
    import torch
    from flcore.train.train_iddpg import train_iddpg

    torch.set_num_threads(cpu_threads)
    result = train_iddpg(
        episodes, train_days, 0, True, fed_method,
        lost_contact=True,
        lost_contact_phase=phase,
        branch_episode=branch_episode,
        lost_agent=lost_agent,
        disconnect_schedule=disconnect_schedule,
        snapshot_ref=snapshot_ref,
        output_root=output_root,
    )
    return fed_method, result if phase == "normal" else None

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
