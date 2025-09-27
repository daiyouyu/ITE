import matplotlib.pyplot as plt
import numpy as np
#导入算法
from flcore.train.train_ddpg import train_ddpg
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
    #rewards_array = np.array(rewards)
    # 保存为.npz文件（压缩格式，支持存储多个数组）
    #np.savez('___ori_rewards.npz', rewards=rewards_array)


if __name__ == "__main__":
    #train_maddpg(episodes=5000, max_steps=100, render=False)
    re,test_re = train_maddpg(episodes=100, max_steps= 24 *7 )
    re = np.array(re)
    re =re.T
    #test_re = np.array(test_re)
    #test_re = test_re.T
    draw_result(re)
    #draw_result(test_re)
