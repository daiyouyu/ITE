
import torch
import torch.nn as nn
import torch.optim as optim
import random
import numpy as np
from collections import deque  # 导入双端队列，用于实现经验回放池
from pathlib import Path

#导入模型
from flcore.Model import Actor, Critic

# 定义经验回放池
class ReplayBuffer:
    def __init__(self, max_size):
        self.buffer = deque(maxlen=max_size)  # 初始化一个双端队列，设置最大容量

    def add(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))  # 将经验存入队列

    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)  # 随机采样一个小批量数据
        states, actions, rewards, next_states, dones = zip(*batch)  # 解压采样数据
        return (np.array(states), np.array(actions), np.array(rewards),
                np.array(next_states), np.array(dones))  # 返回 NumPy 数组格式的数据

    def size(self):
        return len(self.buffer)  # 返回经验池中当前存储的样本数量

class DDPGAgent:
    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        max_action: float = 1.0,
        gamma: float = 0.99,
        tau: float = 0.005,
        buffer_size: int = 100000,
        batch_size: int = 500,
        lr_actor: float = 1e-4,
        lr_critic: float = 1e-3,
        device: str | None = None,
    ):
        """初始化单智能体 DDPG，用于处理完整状态和联合动作。"""
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        self.actor = Actor(state_dim, action_dim, max_action=max_action).to(self.device)
        self.actor_target = Actor(state_dim, action_dim, max_action=max_action).to(self.device)
        self.actor_target.load_state_dict(self.actor.state_dict())
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=lr_actor)

        self.critic = Critic(state_dim, action_dim).to(self.device)
        self.critic_target = Critic(state_dim, action_dim).to(self.device)
        self.critic_target.load_state_dict(self.critic.state_dict())
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=lr_critic)

        self.max_action = float(max_action)
        self.expl_noise_std = 0.1
        self.gamma = gamma
        self.tau = tau
        self.replay_buffer = ReplayBuffer(buffer_size)
        self.batch_size = batch_size

    @torch.no_grad()
    def select_action(
        self,
        state: np.ndarray,
        explore: bool = True,
        noise_scale: float | None = None,
    ) -> np.ndarray:
        """
        根据完整状态生成联合动作。

        ``noise_scale`` 未指定时使用默认探索噪声；为 0 时输出确定性动作。
        """
        state_t = torch.as_tensor(state, dtype=torch.float32, device=self.device).unsqueeze(0)  # (1, S)
        # Actor 的输出层已经包含 Tanh，此处不能再次压缩，否则动作范围会退化到约 [-0.76, 0.76]。
        a = self.actor(state_t).cpu().numpy().flatten()
        if explore:
            actual_noise_scale = self.expl_noise_std if noise_scale is None else float(noise_scale)
            a = a + np.random.normal(0, actual_noise_scale, size=a.shape)
        return np.clip(a, -self.max_action, self.max_action).astype(np.float32)

    def train(self):
        if self.replay_buffer.size() < self.batch_size:
            return

        states, actions, rewards, next_states, dones = self.replay_buffer.sample(self.batch_size)
        # 统一 dtype & device
        states = torch.as_tensor(states, dtype=torch.float32, device=self.device)
        actions = torch.as_tensor(actions, dtype=torch.float32, device=self.device)
        rewards = torch.as_tensor(rewards, dtype=torch.float32, device=self.device).unsqueeze(1)  # (B,1)
        next_states = torch.as_tensor(next_states, dtype=torch.float32, device=self.device)
        dones = torch.as_tensor(dones, dtype=torch.float32, device=self.device).unsqueeze(1)      # (B,1)

        with torch.no_grad():
            next_actions = self.actor_target(next_states)
            target_q = self.critic_target(next_states, next_actions)
            target_q = rewards + (1.0 - dones) * self.gamma * target_q

        current_q = self.critic(states, actions)
        critic_loss = nn.MSELoss()(current_q, target_q)

        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.critic.parameters(), 1.0)  # 可选：稳定训练
        self.critic_optimizer.step()

        actor_loss = -self.critic(states, self.actor(states)).mean()

        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.actor.parameters(), 1.0)   # 可选
        self.actor_optimizer.step()

        # 软更新
        with torch.no_grad():
            for tp, p in zip(self.critic_target.parameters(), self.critic.parameters()):
                tp.data.mul_(1 - self.tau).add_(self.tau * p.data)
            for tp, p in zip(self.actor_target.parameters(), self.actor.parameters()):
                tp.data.mul_(1 - self.tau).add_(self.tau * p.data)

    def add_to_replay_buffer(self, state, action, reward, next_state, done):
        # 确保一维向量/标量，全是 numpy 基本类型，避免后面 stack 出现 object 数组
        state = np.asarray(state, dtype=np.float32).reshape(-1)
        action = np.asarray(action, dtype=np.float32).reshape(-1)
        next_state = np.asarray(next_state, dtype=np.float32).reshape(-1)
        reward = float(reward)
        done = float(done)
        self.replay_buffer.add(state, action, reward, next_state, done)

    def save(self, directory: str = "./model_pth/ddpg") -> None:
        """保存整体 Actor 和 Critic 参数。"""
        save_dir = Path(directory)
        save_dir.mkdir(parents=True, exist_ok=True)
        torch.save(self.actor.state_dict(), save_dir / "actor.pth")
        torch.save(self.critic.state_dict(), save_dir / "critic.pth")

    def load(self, directory: str = "./model_pth/ddpg") -> None:
        """加载整体 Actor 和 Critic 参数，并同步目标网络。"""
        save_dir = Path(directory)
        self.actor.load_state_dict(
            torch.load(save_dir / "actor.pth", map_location=self.device, weights_only=True)
        )
        self.critic.load_state_dict(
            torch.load(save_dir / "critic.pth", map_location=self.device, weights_only=True)
        )
        self.actor_target.load_state_dict(self.actor.state_dict())
        self.critic_target.load_state_dict(self.critic.state_dict())
