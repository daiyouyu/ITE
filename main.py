import numpy as np
import os,time
# 导入算法
from flcore.train.train_ddpg import train_ddpg
from flcore.train.train_iddpg import train_iddpg
from flcore.train.train_maddpg import train_maddpg

# save models
# maddpg.save(prefix="maddpg_simple_adv")


if __name__ == "__main__":
    # re1,test_re = train_maddpg(episodes=100,train = 7  , test = 1 ,Federated = True)
    iddpg, test_re = train_iddpg(episodes=1, train=365, test=0, Federated=False)
    Fed_iddpg, test_re = train_iddpg(episodes=1, train=365, test=0, Federated=True)
    maddpg, test_re = train_maddpg(episodes=1, train=365, test=0, Federated=False)
    Fed_maddpg, test_re = train_maddpg(episodes=1, train=365, test=0, Federated=True)

    re1 = np.array(iddpg).T
    re2 = np.array(Fed_iddpg).T
    re3 = np.array(maddpg).T
    re4 = np.array(Fed_maddpg).T
    iddpg = np.array(re1)
    Fed_iddpg = np.array(re2)
    maddpg = np.array(re3)

    Fed_maddpg = np.array(re4)
    now = time.time()
    now = time.localtime(now)
    now = time.strftime('%Y-%m-%d_%H:%M:%S', now)
    # 指定保存目录（例如当前目录下的 'data' 文件夹）
    save_dir = f'./result/{now}'

    # 检查目录是否存在，不存在则创建
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)  # 递归创建目录（包括父目录）

    # 拼接完整路径（目录 + 文件名）
    file_path = os.path.join(save_dir, f'result_arrays')

    # 保存到指定目录
    np.savez(file_path, iddpg=iddpg, Fed_iddpg=Fed_iddpg, maddpg=maddpg, Fed_maddpg=Fed_maddpg)
    # np.savez(file_path, maddpg=maddpg, Fed_maddpg=Fed_maddpg)
