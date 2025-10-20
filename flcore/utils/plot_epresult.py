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


rew = np.load("D:\\ITE\\result\\result_arrays.npz")
data1 = rew["array1"]
data2=rew["array2"]
data3=rew["array3"]
data4=rew["array4"]
data = [data1,data2,data3,data4]
for i in data:
    draw_result(i)