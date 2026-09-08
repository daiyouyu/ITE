
import torch
import torch.nn as nn
import torch.optim as optim
import random
import numpy as np
from collections import deque  # 导入双端队列，用于实现经验回放池
from pathlib import Path

#导入模型
from flcore.Model import Critic, StructuredCentralizedActor

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
        obs_dims: list[int] | None = None,
        action_dims: list[int] | None = None,
    ):
        """
        初始化采用结构化集中 Actor 的单智能体 DDPG。

        ``obs_dims`` 和 ``action_dims`` 描述各园区在联合向量中的切分方式。
        未提供时退化为单分支结构，以兼容原有的直接实例化调用。
        """
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        resolved_obs_dims = [state_dim] if obs_dims is None else list(obs_dims)
        resolved_action_dims = [action_dim] if action_dims is None else list(action_dims)
        if sum(resolved_obs_dims) != state_dim:
            raise ValueError(
                f"obs_dims 总和与 state_dim 不一致: {sum(resolved_obs_dims)} != {state_dim}"
            )
        if sum(resolved_action_dims) != action_dim:
            raise ValueError(
                "action_dims 总和与 action_dim 不一致: "
                f"{sum(resolved_action_dims)} != {action_dim}"
            )

        self.actor = StructuredCentralizedActor(
            resolved_obs_dims,
            resolved_action_dims,
            max_action=max_action,
        ).to(self.device)
        self.actor_target = StructuredCentralizedActor(
            resolved_obs_dims,
            resolved_action_dims,
            max_action=max_action,
        ).to(self.device)
        self.actor_target.load_state_dict(self.actor.state_dict())
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=lr_actor)

        self.n_agents = len(resolved_obs_dims)
        self.critics = nn.ModuleList(
            Critic(state_dim, action_dim) for _ in range(self.n_agents)
        ).to(self.device)
        self.critic_targets = nn.ModuleList(
            Critic(state_dim, action_dim) for _ in range(self.n_agents)
        ).to(self.device)
        self.critic_targets.load_state_dict(self.critics.state_dict())
        self.critic_optimizers = [
            optim.Adam(critic.parameters(), lr=lr_critic) for critic in self.critics
        ]

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

    def train(self) -> None:
        """独立拟合各园区回报，再汇总全局条件评价更新中心 Actor；样本不足时跳过。"""
        if self.replay_buffer.size() < self.batch_size:
            return

        states, actions, rewards, next_states, dones = self.replay_buffer.sample(self.batch_size)
        # 统一 dtype & device
        states = torch.as_tensor(states, dtype=torch.float32, device=self.device)
        actions = torch.as_tensor(actions, dtype=torch.float32, device=self.device)
        rewards = torch.as_tensor(rewards, dtype=torch.float32, device=self.device)
        rewards = rewards.reshape(self.batch_size, self.n_agents)
        next_states = torch.as_tensor(next_states, dtype=torch.float32, device=self.device)
        dones = torch.as_tensor(dones, dtype=torch.float32, device=self.device).unsqueeze(1)      # (B,1)

        with torch.no_grad():
            next_actions = self.actor_target(next_states)
            target_values = [
                rewards[:, i:i + 1] + (1.0 - dones) * self.gamma
                * target(next_states, next_actions)
                for i, target in enumerate(self.critic_targets)
            ]

        for critic, optimizer, target_q in zip(
            self.critics, self.critic_optimizers, target_values
        ):
            critic_loss = nn.MSELoss()(critic(states, actions), target_q)
            optimizer.zero_grad(set_to_none=True)
            critic_loss.backward()
            torch.nn.utils.clip_grad_norm_(critic.parameters(), 1.0)
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)

        # 仅冻结评价器参数，保留各 Q 对完整联合动作的梯度，学习跨园区影响。
        self.critics.requires_grad_(False)
        try:
            joint_actions = self.actor(states)
            actor_loss = -torch.cat([
                critic(states, joint_actions) for critic in self.critics
            ], dim=1).sum(dim=1).mean()
            self.actor_optimizer.zero_grad(set_to_none=True)
            actor_loss.backward()
            torch.nn.utils.clip_grad_norm_(self.actor.parameters(), 1.0)
            self.actor_optimizer.step()
        finally:
            self.critics.requires_grad_(True)

        # 软更新
        with torch.no_grad():
            for tp, p in zip(self.critic_targets.parameters(), self.critics.parameters()):
                tp.data.mul_(1 - self.tau).add_(self.tau * p.data)
            for tp, p in zip(self.actor_target.parameters(), self.actor.parameters()):
                tp.data.mul_(1 - self.tau).add_(self.tau * p.data)

    def add_to_replay_buffer(self, state, action, reward, next_state, done):
        """将联合 transition 写入回放池，并分别保留各环境的 reward。"""
        state = np.asarray(state, dtype=np.float32).reshape(-1)
        action = np.asarray(action, dtype=np.float32).reshape(-1)
        next_state = np.asarray(next_state, dtype=np.float32).reshape(-1)
        reward = np.asarray(reward, dtype=np.float32).reshape(-1)
        if reward.size != self.n_agents:
            raise ValueError(
                f"reward 必须按园区顺序提供: expected={self.n_agents}, actual={reward.size}"
            )
        done = float(done)
        self.replay_buffer.add(state, action, reward, next_state, done)

    def save(self, directory: str = "./model_pth/ddpg") -> None:
        """保存中心 Actor 和全部独立 Critic 参数。"""
        save_dir = Path(directory)
        save_dir.mkdir(parents=True, exist_ok=True)
        torch.save(self.actor.state_dict(), save_dir / "actor.pth")
        torch.save(self.critics.state_dict(), save_dir / "critics.pth")

    def load(self, directory: str = "./model_pth/ddpg") -> None:
        """加载相同园区配置的多 Critic 模型并同步目标网络；旧单 Critic 不可恢复。"""
        save_dir = Path(directory)
        if not (save_dir / "critics.pth").is_file():
            raise FileNotFoundError("缺少 critics.pth，旧单 Critic 检查点不能用于恢复多 Critic 训练")
        self.actor.load_state_dict(
            torch.load(save_dir / "actor.pth", map_location=self.device, weights_only=True)
        )
        self.critics.load_state_dict(
            torch.load(save_dir / "critics.pth", map_location=self.device, weights_only=True)
        )
        self.actor_target.load_state_dict(self.actor.state_dict())
        self.critic_targets.load_state_dict(self.critics.state_dict())
