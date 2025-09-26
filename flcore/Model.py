import torch.nn as nn
import torch

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