import matplotlib.pyplot as plt
import numpy as np
def draw_result(rewards_record):
    # plot
    for idx,reward in enumerate(rewards_record):
        plt.plot(reward,label=str(idx))
    plt.title("p_bat")
    plt.xlabel("Episode")
    plt.ylabel("SumReward")
    plt.legend()
    plt.show()

np.load()