import numpy as np
import os
from datetime import datetime as dt
from flcore.utils.parallel_train import parallel_train_all

if __name__ == "__main__":
    # 并行训练（推荐）
    USE_PARALLEL = True
    EPOCHS = 500
    TRAIN_DAYS = 365

    if USE_PARALLEL:
        print("使用并行训练模式...")
        results = parallel_train_all(episodes=EPOCHS, train_days=TRAIN_DAYS, n_workers=2)

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
    np.savez(
        file_path, 
        iddpg=iddpg,
        Fed_iddpg=Fed_iddpg,
        maddpg=maddpg, 
        Fed_maddpg=Fed_maddpg
        )
    print(f"结果已保存到 {file_path}.npz")
