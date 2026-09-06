import torch.nn as nn
import torch
from collections.abc import Sequence

# ----------------------------
# Networks
# ----------------------------


class Actor(nn.Module):
    def __init__(self, obs_dim, action_dim, max_action):
        super().__init__()
        self.head = nn.Sequential(
            nn.Linear(obs_dim, 64),
            nn.ReLU(),
        )
        self.base = nn.Sequential(
            nn.Linear(64, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, action_dim),
        )
        self.foot = nn.Sequential(
            nn.Tanh()
        )

    def forward(self, obs):
        out = self.head(obs)
        out = self.base(out)
        out = self.foot(out)
        return out


class StructuredCentralizedActor(nn.Module):
    """
    面向多园区集中控制的结构化 Actor。

    每个园区的观测先由独立编码器提取局部特征，再将全部局部特征融合为
    全局上下文。每个动作头同时使用本园区特征和全局上下文，因此既保留
    园区自身状态，也能利用其他园区的信息协调联合动作。

    Args:
        obs_dims: 按固定园区顺序排列的观测维度。
        action_dims: 与 ``obs_dims`` 顺序一致的动作维度。
        max_action: Actor 输出动作的绝对值上限。
        local_hidden_dim: 每个园区局部编码器的输出维度。
        global_hidden_dim: 全局融合特征维度。
    """

    def __init__(
        self,
        obs_dims: Sequence[int],
        action_dims: Sequence[int],
        max_action: float = 1.0,
        local_hidden_dim: int = 64,
        global_hidden_dim: int = 128,
    ) -> None:
        super().__init__()
        self.obs_dims = tuple(int(dim) for dim in obs_dims)
        self.action_dims = tuple(int(dim) for dim in action_dims)
        if not self.obs_dims or len(self.obs_dims) != len(self.action_dims):
            raise ValueError("obs_dims 和 action_dims 必须非空且长度一致")
        if any(dim <= 0 for dim in self.obs_dims + self.action_dims):
            raise ValueError("观测维度和动作维度必须均大于 0")

        self.max_action = float(max_action)
        self.local_encoders = nn.ModuleList(
            nn.Sequential(
                nn.Linear(obs_dim, local_hidden_dim),
                nn.ReLU(),
                nn.Linear(local_hidden_dim, local_hidden_dim),
                nn.ReLU(),
            )
            for obs_dim in self.obs_dims
        )
        self.global_fusion = nn.Sequential(
            nn.Linear(local_hidden_dim * len(self.obs_dims), global_hidden_dim),
            nn.ReLU(),
            nn.Linear(global_hidden_dim, global_hidden_dim),
            nn.ReLU(),
        )
        self.action_heads = nn.ModuleList(
            nn.Sequential(
                nn.Linear(local_hidden_dim + global_hidden_dim, global_hidden_dim),
                nn.ReLU(),
                nn.Linear(global_hidden_dim, action_dim),
                nn.Tanh(),
            )
            for action_dim in self.action_dims
        )

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        """
        根据拼接后的全部园区观测生成联合动作。

        输入最后一维必须等于 ``sum(obs_dims)``；输出按输入园区顺序拼接，
        最后一维等于 ``sum(action_dims)``。
        """
        if obs.shape[-1] != sum(self.obs_dims):
            raise ValueError(
                f"联合观测维度不匹配: expected={sum(self.obs_dims)}, "
                f"actual={obs.shape[-1]}"
            )

        local_observations = torch.split(obs, self.obs_dims, dim=-1)
        local_features = [
            encoder(local_obs)
            for encoder, local_obs in zip(self.local_encoders, local_observations)
        ]
        global_context = self.global_fusion(torch.cat(local_features, dim=-1))

        local_actions = [
            action_head(torch.cat([local_feature, global_context], dim=-1))
            for action_head, local_feature in zip(self.action_heads, local_features)
        ]
        return torch.cat(local_actions, dim=-1) * self.max_action

class Critic(nn.Module):
    """Centralized critic: input = concat(all_obs, all_actions)"""
    def __init__(self, total_obs_dim, total_action_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(total_obs_dim + total_action_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, 1)
        )

    def forward(self, all_obs, all_actions):
        x = torch.cat([all_obs, all_actions], dim=1)
        return self.net(x)

# split an original model into a base and a head
class BaseHeadSplit(nn.Module):
    def __init__(self, head ,base, foot):
        super(BaseHeadSplit, self).__init__()
        self.head = head
        self.base = base
        self.foot = foot

    def forward(self, x):
        out = self.head(x)
        out = self.base(out)
        out = self.foot(out)

        return out
