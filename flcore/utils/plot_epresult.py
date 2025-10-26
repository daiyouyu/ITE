import matplotlib.pyplot as plt
import numpy as np
def draw_result(rewards_record):
    # plot
    for idx,reward in rewards_record.items():
        plt.plot(reward,label=str(idx))
    plt.title("p_bat")
    plt.xlabel("Episode")
    plt.ylabel("SumReward")
    plt.legend()
    plt.show()


rew = np.load("D:\\ITE\\result\\result_arrays.npz")
keys = ["iddpg", "Fed_iddpg", "maddpg", "Fed_maddpg"]
data = {key: rew[key].sum(axis=0) for key in rew.keys()}
draw_result(data)
