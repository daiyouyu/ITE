
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
        初始化共享主干、多动作头及全局/园区两类 Critic 的 DDPG。

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
        self.backbone_optimizer = optim.Adam(self.actor.backbone.parameters(), lr=lr_actor)
        self.head_optimizers = [
            optim.Adam(head.parameters(), lr=lr_actor)
            for head in self.actor.action_heads
        ]

        self.n_agents = len(resolved_obs_dims)
        self.obs_dims = tuple(resolved_obs_dims)
        self.action_dims = tuple(resolved_action_dims)
        self.global_critic = Critic(state_dim, action_dim).to(self.device)
        self.global_critic_target = Critic(state_dim, action_dim).to(self.device)
        self.global_critic_target.load_state_dict(self.global_critic.state_dict())
        self.global_critic_optimizer = optim.Adam(self.global_critic.parameters(), lr=lr_critic)
        self.critics = nn.ModuleList(
            Critic(obs_dim, local_action_dim)
            for obs_dim, local_action_dim in zip(self.obs_dims, self.action_dims)
        ).to(self.device)
        self.critic_targets = nn.ModuleList(
            Critic(obs_dim, local_action_dim)
            for obs_dim, local_action_dim in zip(self.obs_dims, self.action_dims)
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

    def _joint_actions(
        self, states: torch.Tensor, actor: StructuredCentralizedActor,
    ) -> torch.Tensor:
        """分别调用局部策略，仅为 Critic 和旧环境接口拼接动作；states 为联合观测。"""
        return torch.cat([
            actor(obs, i)
            for i, obs in enumerate(torch.split(states, self.obs_dims, dim=-1))
        ], dim=-1)

    @torch.no_grad()
    def select_local_action(
        self, observation: np.ndarray, agent_index: int,
        explore: bool = True, noise_scale: float | None = None,
    ) -> np.ndarray:
        """仅根据指定园区的观测生成动作；无探索时确定性输出，索引须有效。"""
        obs = torch.as_tensor(observation, dtype=torch.float32, device=self.device)
        action = self.actor(obs, agent_index).cpu().numpy().flatten()
        if explore:
            scale = self.expl_noise_std if noise_scale is None else float(noise_scale)
            action = action + np.random.normal(0, scale, size=action.shape)
        return np.clip(action, -self.max_action, self.max_action).astype(np.float32)

    @torch.no_grad()
    def select_action(
        self,
        state: np.ndarray,
        explore: bool = True,
        noise_scale: float | None = None,
    ) -> np.ndarray:
        """
        兼容旧调用：拆分完整状态，分别生成本地动作后拼接返回。

        ``noise_scale`` 未指定时使用默认探索噪声；为 0 时输出确定性动作。
        """
        state_t = torch.as_tensor(state, dtype=torch.float32, device=self.device).unsqueeze(0)  # (1, S)
        # Actor 的输出层已经包含 Tanh，此处不能再次压缩，否则动作范围会退化到约 [-0.76, 0.76]。
        a = self._joint_actions(state_t, self.actor).cpu().numpy().flatten()
        if explore:
            actual_noise_scale = self.expl_noise_std if noise_scale is None else float(noise_scale)
            a = a + np.random.normal(0, actual_noise_scale, size=a.shape)
        return np.clip(a, -self.max_action, self.max_action).astype(np.float32)

    def train(self) -> None:
        """拟合两类 Critic，再分别更新共享主干和各头部；样本不足时跳过。"""
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
        # 回放按固定园区顺序拆分，园区评价器仅使用自己的 transition 分量。
        local_states = torch.split(states, self.obs_dims, dim=-1)
        local_actions = torch.split(actions, self.action_dims, dim=-1)
        local_next_states = torch.split(next_states, self.obs_dims, dim=-1)

        with torch.no_grad():
            next_actions = self._joint_actions(next_states, self.actor_target)
            local_next_actions = torch.split(next_actions, self.action_dims, dim=-1)
            global_target = rewards.sum(dim=1, keepdim=True) + (1.0 - dones) * self.gamma \
                * self.global_critic_target(next_states, next_actions)
            target_values = [
                rewards[:, i:i + 1] + (1.0 - dones) * self.gamma
                * target(local_next_states[i], local_next_actions[i])
                for i, target in enumerate(self.critic_targets)
            ]

        for critic, optimizer, target_q, critic_states, critic_actions in zip(
            [self.global_critic, *self.critics],
            [self.global_critic_optimizer, *self.critic_optimizers],
            [global_target, *target_values],
            [states, *local_states],
            [actions, *local_actions],
        ):
            critic_loss = nn.MSELoss()(critic(critic_states, critic_actions), target_q)
            optimizer.zero_grad(set_to_none=True)
            critic_loss.backward()
            torch.nn.utils.clip_grad_norm_(critic.parameters(), 1.0)
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)

        self._update_backbone(states)
        self._update_heads(states)

        # 软更新
        with torch.no_grad():
            for tp, p in zip(self.global_critic_target.parameters(), self.global_critic.parameters()):
                tp.lerp_(p, self.tau)
            for tp, p in zip(self.critic_targets.parameters(), self.critics.parameters()):
                tp.data.mul_(1 - self.tau).add_(self.tau * p.data)
            for tp, p in zip(self.actor_target.parameters(), self.actor.parameters()):
                tp.data.mul_(1 - self.tau).add_(self.tau * p.data)

    def _update_backbone(self, states: torch.Tensor) -> None:
        """中心 Q 更新共享主干；states 为联合观测，头部及 Critic 参数保持固定。"""
        self.actor.zero_grad(set_to_none=True)
        self.global_critic.requires_grad_(False)
        self.actor.action_heads.requires_grad_(False)
        try:
            # 冻结头部参数但保留对输入特征的导数，中心信号才能穿过头部回到主干。
            actions = self._joint_actions(states, self.actor)
            loss = -self.global_critic(states, actions).mean()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.actor.backbone.parameters(), 1.0)
            self.backbone_optimizer.step()
        finally:
            self.actor.action_heads.requires_grad_(True)
            self.global_critic.requires_grad_(True)
            self.backbone_optimizer.zero_grad(set_to_none=True)

    def _update_heads(self, states: torch.Tensor) -> None:
        """拆分 states 为本地观测，园区 Q 仅更新对应头部；主干特征停止梯度。"""
        self.actor.zero_grad(set_to_none=True)
        local_obs = torch.split(states, self.obs_dims, dim=-1)
        # 用更新后的主干重新计算特征并停止梯度，避免局部目标更新共享参数。
        with torch.no_grad():
            features = [self.actor.backbone(obs) for obs in local_obs]
        self.critics.requires_grad_(False)
        try:
            for i, (head, critic, optimizer) in enumerate(zip(
                self.actor.action_heads, self.critics, self.head_optimizers,
            )):
                local_action = head(features[i]) * self.max_action
                loss = -critic(local_obs[i], local_action).mean()
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(head.parameters(), 1.0)
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
        finally:
            self.critics.requires_grad_(True)

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
        """保存共享 Actor、中心 Critic 和园区 Critic 权重；不保存优化器状态。"""
        save_dir = Path(directory)
        save_dir.mkdir(parents=True, exist_ok=True)
        torch.save(self.actor.state_dict(), save_dir / "actor.pth")
        torch.save(self.critics.state_dict(), save_dir / "critics.pth")
        torch.save(self.global_critic.state_dict(), save_dir / "global_critic.pth")

    def load(self, directory: str = "./model_pth/ddpg") -> None:
        """加载相同园区配置的新结构权重并同步目标网络；旧架构须重新训练。"""
        save_dir = Path(directory)
        if not (save_dir / "global_critic.pth").is_file():
            raise FileNotFoundError("缺少 global_critic.pth，旧 DDPG 架构权重不兼容，请重新训练")
        self.actor.load_state_dict(
            torch.load(save_dir / "actor.pth", map_location=self.device, weights_only=True)
        )
        self.critics.load_state_dict(
            torch.load(save_dir / "critics.pth", map_location=self.device, weights_only=True)
        )
        self.actor_target.load_state_dict(self.actor.state_dict())
        self.critic_targets.load_state_dict(self.critics.state_dict())
        self.global_critic.load_state_dict(
            torch.load(save_dir / "global_critic.pth", map_location=self.device, weights_only=True)
        )
        self.global_critic_target.load_state_dict(self.global_critic.state_dict())
