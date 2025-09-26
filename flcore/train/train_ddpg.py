from flcore.algorithm.DDPG import DDPGAgent
from flcore.Env.Env import BatteryArbEnv
from flcore.utils.print_epreward import format_episode_info
import gymnasium as gym
from data.load_data import load_power_data
import numpy as np
from gymnasium import spaces
from gymnasium.wrappers import RescaleAction
from gymnasium.spaces import utils as su


# ==== 1) 把 Dict 动作变成连续 Box(8,) 的包装器 ====
class DictToBoxActionWrapper(gym.ActionWrapper):
    """
    将 EnergyEnv 的 Dict 动作空间映射成一个连续 Box 向量:
    vector = [xwt, ashc, tes_rate, bfw1, bfw2, hs, ms, ls]
    - xwt: 连续 [0, xwt_max]，在 action() 中四舍五入成 int 传回原 env
    其余按原 env 的上下界拼接。
    """
    def __init__(self, env):
        super().__init__(env)

        # 原动作空间
        A = env.action_space
        assert isinstance(A, spaces.Dict), "期待原环境是 Dict 动作空间"
        # 取边界
        xwt_low, xwt_high = 0.0, float(env.xwt_max)  # 连续化
        ashc_low, ashc_high = A["ashc"].low.item(), A["ashc"].high.item()
        tes_low, tes_high = A["tes_rate"].low.item(), A["tes_rate"].high.item()
        bfw_low, bfw_high = A["bfw"].low.astype(float), A["bfw"].high.astype(float)  # (2,)
        im_low, im_high   = A["im"].low.astype(float),  A["im"].high.astype(float)   # (3,)

        low  = np.array([xwt_low, ashc_low, tes_low, bfw_low[0], bfw_low[1], im_low[0], im_low[1], im_low[2]], dtype=np.float32)
        high = np.array([xwt_high, ashc_high, tes_high, bfw_high[0], bfw_high[1], im_high[0], im_high[1], im_high[2]], dtype=np.float32)

        self._low  = low
        self._high = high
        self.action_space = spaces.Box(low=low, high=high, dtype=np.float32)

    def action(self, act_vec):
        # clip 并拆回 Dict
        a = np.clip(np.asarray(act_vec, dtype=np.float32), self._low, self._high)

        xwt      = int(np.round(float(a[0])))                 # 连续 → 四舍五入为整数
        ashc     = float(a[1])
        tes_rate = float(a[2])
        bfw      = np.array([float(a[3]), float(a[4])], dtype=np.float32)
        im       = np.array([float(a[5]), float(a[6]), float(a[7])], dtype=np.float32)

        return {"xwt": xwt, "ashc": np.array([ashc], dtype=np.float32),
                "tes_rate": np.array([tes_rate], dtype=np.float32),
                "bfw": bfw, "im": im}

    def reverse_action(self, action):
        # 仅用于需要把 Dict 回放成向量的场景（平常用不到）
        xwt = float(action["xwt"])
        ashc = float(np.asarray(action["ashc"]).reshape(-1)[0])
        tes  = float(np.asarray(action["tes_rate"]).reshape(-1)[0])
        bfw  = np.asarray(action["bfw"], dtype=float).reshape(-1)
        im   = np.asarray(action["im"], dtype=float).reshape(-1)
        return np.array([xwt, ashc, tes, bfw[0], bfw[1], im[0], im[1], im[2]], dtype=np.float32)

# ==== 2) 训练循环（DDPG 友好） ====
def train_ddpg(env_name=None, episodes=1000, max_steps=5000):
    # 构建环境
    data = load_power_data("./data/GridSet_no_pred.csv", price_mode="mean")

    env = BatteryArbEnv(
        data,
        dt_hours=1.0,
        E_bat_MWh=48.0,
        P_bat_max_MW=12.0,
        eta_ch=0.95, eta_dis=0.95,
        soc_min=0.1, soc_max=0.9, soc_init=0.5,
        deg_cost_per_MW=0.1,  # 例如每回合 1 个月
        obs_norm=True
    )
    #env = EnergyEnv()  # 如果你本来就直接用类
    #env = DictToBoxActionWrapper(env)        # 把 Dict→Box(8,)
    env = RescaleAction(env, -1.0, 1.0)      # 把动作统一映射到 [-1, 1]，Agent 输出 tanh 即可

    state_dim  = env.observation_space.shape[0]      # 已是 Box
    action_dim = env.action_space.shape[0]           # = 8
    # 由于已用 RescaleAction，max_action 可以设为 1.0（或者让 agent 自己 tanh）
    max_action = 1.0

    agent = DDPGAgent(state_dim, action_dim, max_action)

    rewards = []
    for ep in range(episodes):
        state, _ = env.reset()
        ep_reward = 0.0
        ep_info = {
            "p_bat": 0,
            "p_grid": 0,
            "G_demand": 0, "newpower_gen": 0, "grid_price": 0,"bioler_gen": 0,
            "soc": 0,
            "cost_grid": 0, "cost_deg": 0,
            "cost_gen": 0,
        }

        for step in range(max_steps):
            # 选择动作：前若干步可用随机探索
            if step < 4:

                action = env.action_space.sample()              # 已是 [-1,1] 范围
            else:
                action = agent.select_action(state, explore=True)  # 输出应在 [-1,1]

            # 交互
            next_state, reward, terminated, _, info = env.step(action)
            done = terminated
            if ep %50==0 and step < 48:
                print(f"step:{step},电力负荷: {info["G_demand"]:.3f},新能源:{info["newpower_gen"]:.3f},锅炉:{info["bioler_gen"]:.3f}，锅炉价格：{info["cost_gen"]:.3f},电池：{info["p_bat"]:.3f},电网：{info["p_grid"]:.3f},电网耗费：{info["G_demand"]:.3f}"),

                        # 存回放 & 训练
            agent.add_to_replay_buffer(state, action, reward, next_state, done)
            agent.train()

            state = next_state
            ep_reward += reward

            for key,value in ep_info.items():
                ep_info[key] += info[key]
            if done:
                break

        rewards.append(ep_reward)
        print(format_episode_info(ep, ep_reward, ep_info))

    env.close()
    return rewards

