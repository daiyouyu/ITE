import multiprocessing as mp
from functools import partial

def train_worker(algo_name, episodes, train_days, federated):
    """单个训练进程"""
    if algo_name == 'iddpg':
        from flcore.train.train_iddpg import train_iddpg
        return train_iddpg(episodes, train_days, 0, federated)
    else:
        from flcore.train.train_maddpg import train_maddpg
        return train_maddpg(episodes, train_days, 0, federated)

def parallel_train_all(episodes=500, train_days=365, n_workers=4):
    """并行训练所有算法"""
    tasks = [
        ('iddpg', episodes, train_days, False),
        ('iddpg', episodes, train_days, True),
        ('maddpg', episodes, train_days, False),
        ('maddpg', episodes, train_days, True)
    ]
    
    with mp.Pool(n_workers) as pool:
        results = pool.starmap(train_worker, tasks)
    
    return {
        'iddpg': results[0],
        'Fed_iddpg': results[1],
        'maddpg': results[2],
        'Fed_maddpg': results[3]
    }