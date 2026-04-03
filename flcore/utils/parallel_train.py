import multiprocessing as mp
from functools import partial

def train_worker(algo_name, episodes, train_days, federated,fed_method):
    """单个训练进程"""
    if algo_name == 'iddpg':
        from flcore.train.train_iddpg import train_iddpg
        return train_iddpg(episodes, train_days, 0, federated,fed_method)
    else:
        from flcore.train.train_maddpg import train_maddpg
        return train_maddpg(episodes, train_days, 0, federated)

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