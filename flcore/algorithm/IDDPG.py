# -*- coding: utf-8 -*-
import copy
import torch
import os
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
import random
from collections import deque

# 复用你的模型定义（含 head/base/foot，便于抽 proto）
from flcore.Model import Actor, Critic
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"device: {device}")

# ----------------------------
# 重放缓存（与 MADDPG 保持一致：存 joint，方便一次性采样）
# ----------------------------
class JointReplayBuffer:
    def __init__(self, max_size=100000):
        self.max_size = int(max_size)
        self.buffer = deque(maxlen=self.max_size)

    def add(self, joint_obs, joint_actions, joint_rewards, joint_next_obs, dones):
        self.buffer.append((joint_obs, joint_actions, joint_rewards, joint_next_obs, dones))

    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        obs_b, acts_b, rews_b, next_obs_b, dones_b = zip(*batch)
        return (
            np.array(obs_b, dtype=np.float32),
            np.array(acts_b, dtype=np.float32),
            np.array(rews_b, dtype=np.float32),
            np.array(next_obs_b, dtype=np.float32),
            np.array(dones_b, dtype=np.float32),
        )

    def size(self):
        return len(self.buffer)


# ----------------------------
# IDDPG：每个 agent 独立 Actor-Critic（本地观测 + 本地动作的 Critic）
# 保持与 MADDPG.py 相同的外部接口：select_actions/update/Fed_Aggergate/save/load
# ----------------------------
class IDDPG:
    def __init__(self, obs_dims, action_dims, max_actions,
                 lr_actor=1e-3, lr_critic=1e-3, gamma=0.99, tau=0.01,
                 batch_size=256, buffer_size=100000):
        """
        obs_dims: list[int] 每个 agent 的观测维度
        action_dims: list[int] 每个 agent 的动作维度
        max_actions: list[float or array] 每个 agent 的动作界（用于 clip）
        """
        self.n_agents = len(obs_dims)
        self.obs_dims = obs_dims
        self.action_dims = action_dims
        self.max_actions = max_actions

        # 独立 actor / critic
        self.actors, self.actor_targets, self.actor_opts = [], [], []
        self.critics, self.critic_targets, self.critic_opts = [], [], []

        # proto 历史（为联邦聚合准备）
        self.proto_history = [[] for _ in range(self.n_agents)]
        self.Federated_proto = []
        self.template = None  # 用于按权重累加 actor 参数

        # 初始化每个 agent 的网络
        for i in range(self.n_agents):

            actor = Actor(obs_dims[i], action_dims[i], max_actions[i]).to(device)
            actor_t = copy.deepcopy(actor).to(device)
            opt_a = optim.Adam(actor.parameters(), lr=lr_actor)
            self.actors.append(actor)
            self.actor_targets.append(actor_t)
            self.actor_opts.append(opt_a)

            # 注意：IDDPG 的 Critic 是本地 critic，只吃自己 agent 的 obs + act
            critic = Critic(obs_dims[i], action_dims[i]).to(device)
            critic_t = copy.deepcopy(critic).to(device)
            opt_c = optim.Adam(critic.parameters(), lr=lr_critic)
            self.critics.append(critic)
            self.critic_targets.append(critic_t)
            self.critic_opts.append(opt_c)

        self.replay = JointReplayBuffer(max_size=buffer_size)
        self.gamma = float(gamma)
        self.tau = float(tau)
        self.batch_size = int(batch_size)

        # 模板网络用于联邦加权（拷一个 actor 结构当累加容器）
        self.template = copy.deepcopy(self.actors[0]).to(device)

        # 为了切片方便，预先计算 joint 各段位置
        self._obs_slices = []
        self._act_slices = []
        o_l, a_l = 0, 0
        for od, ad in zip(self.obs_dims, self.action_dims):
            self._obs_slices.append(slice(o_l, o_l + od))
            self._act_slices.append(slice(a_l, a_l + ad))
            o_l += od
            a_l += ad

    # 观测 -> 动作（可加噪）
    def select_actions(self, obs_list, noise_scale=0.1):
        actions = []
        for i, obs in enumerate(obs_list):
            s = torch.FloatTensor(np.asarray(obs).reshape(1, -1)).to(device)
            a = self.actors[i](s).detach().cpu().numpy().flatten()
            if noise_scale and noise_scale > 0:
                a = a + np.random.normal(0, noise_scale, size=a.shape)
            # clip 到动作边界（支持标量或逐维）
            max_bound = self.max_actions[i]
            if np.isscalar(max_bound):
                a = np.clip(a, -max_bound, max_bound)
            else:
                a = np.clip(a, -np.asarray(max_bound, dtype=np.float32), np.asarray(max_bound, dtype=np.float32))
            actions.append(a.astype(np.float32))
        return actions

    # 一步更新（与 MADDPG 的外形一致；但每个 agent 的 critic 只看自己的 obs/act）
    def update(self):
        if self.replay.size() < self.batch_size:
            return

        obs_b, acts_b, rews_b, next_obs_b, dones_b = self.replay.sample(self.batch_size)
        # 转 tensor（joint）
        obs_b_t = torch.FloatTensor(obs_b).to(device)               # [B, sum(obs)]
        acts_b_t = torch.FloatTensor(acts_b).to(device)             # [B, sum(act)]
        next_obs_b_t = torch.FloatTensor(next_obs_b).to(device)     # [B, sum(obs)]
        rews_b_t = torch.FloatTensor(rews_b).to(device)             # [B, n_agents]
        dones_b_t = torch.FloatTensor(dones_b).to(device)           # [B, n_agents]

        self.Federated_proto = []

        for i in range(self.n_agents):
            # 切出 agent i 的本地 obs/action
            oi = obs_b_t[:, self._obs_slices[i]]
            ai = acts_b_t[:, self._act_slices[i]]
            noi = next_obs_b_t[:, self._obs_slices[i]]

            # -------- critic 更新（本地）--------
            with torch.no_grad():
                next_ai = self.actor_targets[i](noi)
                q_next = self.critic_targets[i](noi, next_ai)
                td_target = rews_b_t[:, i:i+1] + (1.0 - dones_b_t[:, i:i+1]) * (self.gamma * q_next)

            q_curr = self.critics[i](oi, ai)
            loss_q = nn.MSELoss()(q_curr, td_target)
            self.critic_opts[i].zero_grad()
            loss_q.backward()
            self.critic_opts[i].step()

            # -------- actor 更新（本地 PG）--------
            # 计算 proto 并记录（供联邦）
            out_i = self.actors[i].head(oi)
            proto_i = self.actors[i].base(out_i)        # [B, Feat]
            act_pred = self.actors[i].foot(proto_i)     # [B, A_i]

            # actor loss：最大化本地 critic 的 Q（等价最小化 -Q）
            actor_loss = - self.critics[i](oi, act_pred).mean()
            self.actor_opts[i].zero_grad()
            actor_loss.backward()
            self.actor_opts[i].step()

            # 累计 batch 维度上的 proto 均值，入历史
            with torch.no_grad():
                self.proto_history[i].append(proto_i.mean(dim=0).detach().cpu())
                self.Federated_proto.append(proto_i.detach())

            # -------- 软更新 target --------
            for p, p_t in zip(self.actors[i].parameters(), self.actor_targets[i].parameters()):
                p_t.data.copy_(self.tau * p.data + (1.0 - self.tau) * p_t.data)
            for p, p_t in zip(self.critics[i].parameters(), self.critic_targets[i].parameters()):
                p_t.data.copy_(self.tau * p.data + (1.0 - self.tau) * p_t.data)

    # 与 MADDPG 一致的联邦聚合：按 proto 距离自适应加权聚合 actor 参数
    def Fed_Aggergate(self):
        if self.replay.size() < self.batch_size:
            return
        if any(len(h) == 0 for h in self.proto_history):
            return

        # 1) 历史 proto 平均
        avg_proto = []
        for j in range(self.n_agents):
            hist = self.proto_history[j]  # list of cpu tensors (Feat,)
            avg = torch.stack(hist, dim=0).mean(dim=0).to(device)
            avg_proto.append(avg)

        # 2) 用当前一个 batch 估计“参考 proto”
        obs_b, _, _, _, _ = self.replay.sample(self.batch_size)
        obs_b_t = torch.FloatTensor(obs_b).to(device)
        proto_ref = []
        with torch.no_grad():
            for i in range(self.n_agents):
                oi = obs_b_t[:, self._obs_slices[i]]
                out_i = self.actors[i].head(oi)
                p_i = self.actors[i].base(out_i)
                proto_ref.append(p_i.mean(dim=0))  # (Feat,)

        # 3) 基于 L1 距离的权重矩阵
        eps = 1e-8
        Federated_w = []
        for i in range(self.n_agents):
            row = []
            for j in range(self.n_agents):
                d = F.l1_loss(proto_ref[i], avg_proto[j], reduction='mean').item()
                row.append(1.0 / (d + eps))
            w = np.array(row, dtype=np.float64)
            w = w / (w.sum() + 1e-12)
            Federated_w.append(w)

        # 4) 依据权重做参数聚合，分别得到每个 agent 的新 actor
        with torch.no_grad():
            new_actors = []
            for i in range(self.n_agents):
                for p in self.template.parameters():
                    p.data.zero_()
                for j in range(self.n_agents):
                    w_ij = Federated_w[i][j]
                    for tp, pj in zip(self.template.parameters(), self.actors[j].parameters()):
                        tp.data.add_(pj.data, alpha=w_ij)
                new_actors.append(copy.deepcopy(self.template))

            # 回写
            for i in range(self.n_agents):
                for p, np_ in zip(self.actors[i].parameters(), new_actors[i].parameters()):
                    p.data.copy_(np_.data)

        # 5) 清空历史
        self.proto_history = [[] for _ in range(self.n_agents)]

    # 保存 / 加载（对齐 MADDPG）
    def save(self, prefix="iddpg",Fed=False):
        file_path = f"./model_pth/{prefix}"
        if not os.path.exists(file_path):
            os.makedirs(file_path)
        for i in range(self.n_agents):
            torch.save(self.actors[i].state_dict(), f"{file_path}/{Fed}_actor_{i}.pth")
            torch.save(self.critics[i].state_dict(), f"{file_path}/{Fed}_critic_{i}.pth")

    def load(self, prefix="iddpg",Fed=False):
        for i in range(self.n_agents):
            self.actors[i].load_state_dict(torch.load(f"./model_pth/{prefix}/{Fed}_actor_{i}.pth", map_location=device,weights_only=True))
            self.critics[i].load_state_dict(torch.load(f"./model_pth/{prefix}/{Fed}_critic_{i}.pth", map_location=device,weights_only=True))
