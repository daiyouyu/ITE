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
    本地观测经过共享主干和园区专属头部，输出该园区的动作。

    各园区观测必须具有相同维度及字段语义；主干只有一份参数，
    每次调用仅处理指定园区的观测，不融合其他园区的信息。

    Args:
        obs_dims: 按固定园区顺序排列的观测维度。
        action_dims: 与 ``obs_dims`` 顺序一致的动作维度。
        max_action: Actor 输出动作的绝对值上限。
        local_hidden_dim: 主干第一层宽度。
        global_hidden_dim: 主干输出和动作头隐藏层宽度。
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
        if len(set(self.obs_dims)) != 1:
            raise ValueError("共享主干要求各园区观测维度一致，且字段语义对齐")

        self.max_action = float(max_action)
        self.backbone = nn.Sequential(
            nn.Linear(self.obs_dims[0], local_hidden_dim),
            nn.ReLU(),
            nn.Linear(local_hidden_dim, global_hidden_dim),
            nn.ReLU(),
        )
        self.action_heads = nn.ModuleList(
            nn.Sequential(
                nn.Linear(global_hidden_dim, global_hidden_dim),
                nn.ReLU(),
                nn.Linear(global_hidden_dim, action_dim),
                nn.Tanh(),
            )
            for action_dim in self.action_dims
        )

    def forward(self, obs: torch.Tensor, agent_index: int) -> torch.Tensor:
        """
        根据本地观测和园区索引生成该园区动作。

        obs 最后一维必须等于该园区观测维度；返回最后一维为该园区动作维度。
        园区索引越界或观测维度不匹配时抛出 ValueError。
        """
        if not 0 <= agent_index < len(self.obs_dims):
            raise ValueError(f"园区索引越界: agent_index={agent_index}")
        if obs.shape[-1] != self.obs_dims[agent_index]:
            raise ValueError(
                f"本地观测维度不匹配: expected={self.obs_dims[agent_index]}, "
                f"actual={obs.shape[-1]}"
            )

        return self.action_heads[agent_index](self.backbone(obs)) * self.max_action

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
