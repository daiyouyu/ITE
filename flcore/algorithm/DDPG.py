
import torch
import torch.nn as nn
import torch.optim as optim
import random
import numpy as np
from collections import deque  # 导入双端队列，用于实现经验回放池
#导入模型
from flcore.Model import Actor,Critic

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
    def __init__(self, state_dim, action_dim, max_action=1.0, gamma=0.99, tau=0.005,
                 buffer_size=100000, batch_size=500, device=None):
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        self.actor = Actor(state_dim, action_dim, max_action=1.0).to(self.device)
        self.actor_target = Actor(state_dim, action_dim, max_action=1.0).to(self.device)
        self.actor_target.load_state_dict(self.actor.state_dict())
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=1e-4)

        self.critic = Critic(state_dim, action_dim).to(self.device)
        self.critic_target = Critic(state_dim, action_dim).to(self.device)
        self.critic_target.load_state_dict(self.critic.state_dict())
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=1e-3)

        self.max_action = 1.0           # RescaleAction 已把环境动作缩放到 [-1,1]
        self.expl_noise_std = 0.1
        self.gamma = gamma
        self.tau = tau
        self.replay_buffer = ReplayBuffer(buffer_size)
        self.batch_size = batch_size

    @torch.no_grad()
    def select_action(self, state, explore=True):
        state_t = torch.as_tensor(state, dtype=torch.float32, device=self.device).unsqueeze(0)  # (1, S)
        a = self.actor(state_t)          # 假设 Actor 输出未做 tanh
        a = torch.tanh(a).cpu().numpy().flatten()  # 压到 [-1,1]
        if explore:
            a = a + np.random.normal(0, self.expl_noise_std, size=a.shape)
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
            next_actions = torch.tanh(self.actor_target(next_states))  # 目标动作也限幅到 [-1,1]
            target_q = self.critic_target(next_states, next_actions)
            target_q = rewards + (1.0 - dones) * self.gamma * target_q

        current_q = self.critic(states, actions)
        critic_loss = nn.MSELoss()(current_q, target_q)

        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.critic.parameters(), 1.0)  # 可选：稳定训练
        self.critic_optimizer.step()

        actor_loss = -self.critic(states, torch.tanh(self.actor(states))).mean()

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

