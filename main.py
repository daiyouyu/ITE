# import numpy as np
# import os
# # 导入算法
# from flcore.train.train_ddpg import train_ddpg
# from flcore.train.train_iddpg import train_iddpg
# from flcore.train.train_maddpg import train_maddpg
# from datetime import datetime as dt

# if __name__ == "__main__":
#     # re1,test_re = train_maddpg(episodes=100,train = 7  , test = 1 ,Federated = True)
#     iddpg, test_re = train_iddpg(episodes=500, train=365, test=0, Federated=False)
#     Fed_iddpg, test_re = train_iddpg(episodes=500, train=365, test=0, Federated=True)
#     maddpg, test_re = train_maddpg(episodes=500, train=365, test=0, Federated=False)
#     Fed_maddpg, test_re = train_maddpg(episodes=500, train=365, test=0, Federated=True)

#     re1 = np.array(iddpg).T
#     re2 = np.array(Fed_iddpg).T
#     re3 = np.array(maddpg).T
#     re4 = np.array(Fed_maddpg).T
#     iddpg = np.array(re1)
#     Fed_iddpg = np.array(re2)
#     maddpg = np.array(re3)

#     Fed_maddpg = np.array(re4)
#     now = dt.now().strftime('%Y%m%d')
#     # 指定保存目录（例如当前目录下的 'data' 文件夹）
#     save_dir = f'./result/{now}'

#     # 检查目录是否存在，不存在则创建
#     if not os.path.exists(save_dir):
#         os.makedirs(save_dir)

#         # 拼接完整路径（目录 + 文件名）
#     file_path = os.path.join(save_dir, f'result_arrays')

#     # 保存到指定目录
#     np.savez(file_path, iddpg=iddpg, Fed_iddpg=Fed_iddpg, maddpg=maddpg, Fed_maddpg=Fed_maddpg)
#     # np.savez(file_path, maddpg=maddpg, Fed_maddpg=Fed_maddpg)

import numpy as np
import os
from datetime import datetime as dt
from flcore.utils.parallel_train import parallel_train_all

if __name__ == "__main__":
    # 并行训练（推荐）
    USE_PARALLEL = False
    EPOCHS = 1
    TRAIN_DAYS = 7

    if USE_PARALLEL:
        print("使用并行训练模式...")
        results = parallel_train_all(episodes=EPOCHS, train_days=TRAIN_DAYS, n_workers=4)

        iddpg = np.array(results['iddpg'][0]).T
        Fed_iddpg = np.array(results['Fed_iddpg'][0]).T
        maddpg = np.array(results['maddpg'][0]).T
        Fed_maddpg = np.array(results['Fed_maddpg'][0]).T
    else:
        # 原有串行训练
        from flcore.train.train_iddpg import train_iddpg
        from flcore.train.train_maddpg import train_maddpg

        iddpg, _ = train_iddpg(episodes=EPOCHS, train=TRAIN_DAYS, test=0, Federated=False)
        Fed_iddpg, _ = train_iddpg(episodes=EPOCHS, train=TRAIN_DAYS, test=0, Federated=True)
        maddpg, _ = train_maddpg(episodes=EPOCHS, train=TRAIN_DAYS, test=0, Federated=False)
        Fed_maddpg, _ = train_maddpg(episodes=EPOCHS, train=TRAIN_DAYS, test=0, Federated=True)

        iddpg = np.array(iddpg).T
        Fed_iddpg = np.array(Fed_iddpg).T
        maddpg = np.array(maddpg).T
        Fed_maddpg = np.array(Fed_maddpg).T

    # 保存结果
    now = dt.now().strftime('%Y%m%d')
    save_dir = f'./result/{now}'
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    file_path = os.path.join(save_dir, 'result_arrays')
    np.savez(file_path, iddpg=iddpg, Fed_iddpg=Fed_iddpg,
             maddpg=maddpg, Fed_maddpg=Fed_maddpg)
    print(f"结果已保存到 {file_path}.npz")
