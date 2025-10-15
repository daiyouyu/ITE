import matplotlib.pyplot as plt
import numpy as np
import os
#导入算法
from flcore.train.train_ddpg import train_ddpg
from flcore.train.train_iddpg import train_iddpg
from flcore.train.train_maddpg import train_maddpg

def draw_result(rewards_record):
    # plot
    for idx,reward in enumerate(rewards_record):
        plt.plot(reward,label=str(idx))
    plt.title("p_bat")
    plt.xlabel("Episode")
    plt.ylabel("SumReward")
    plt.legend()
    plt.show()

    # save models
    # maddpg.save(prefix="maddpg_simple_adv")



if __name__ == "__main__":
    #re1,test_re = train_maddpg(episodes=100,train = 7  , test = 1 ,Federated = True)
    re1, test_re = train_iddpg(episodes=50, train=2 , test=1, Federated=False)
    re2,test_re = train_iddpg(episodes=50,train = 2  , test = 1 ,Federated = True)
    re1 = np.array(re1)
    re1 =re1.T
    re2 = np.array(re2)
    re2 = re2.T
    draw_result(re1)
    draw_result(re2)

    re1_array = np.array(re1)
    re2_array = np.array(re2)
    # 指定保存目录（例如当前目录下的 'data' 文件夹）
    save_dir = './result'

    # 检查目录是否存在，不存在则创建
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)  # 递归创建目录（包括父目录）

    # 拼接完整路径（目录 + 文件名）
    file_path = os.path.join(save_dir, 'result_arrays')

    # 保存到指定目录
    np.savez(file_path, array1=re1_array, array2=re2_array)
