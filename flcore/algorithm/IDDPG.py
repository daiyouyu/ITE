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
    def update(self, online_agents=None):
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
                # 断联园区继续本地更新，但不对外提供动作描述符。
                if online_agents is None or i in online_agents:
                    self.proto_history[i].append(act_pred.mean(dim=0).detach().cpu())
                    self.Federated_proto.append(proto_i.detach())
                else:
                    self.proto_history[i].clear()

            # -------- 软更新 target --------
            for p, p_t in zip(self.actors[i].parameters(), self.actor_targets[i].parameters()):
                p_t.data.copy_(self.tau * p.data + (1.0 - self.tau) * p_t.data)
            for p, p_t in zip(self.critics[i].parameters(), self.critic_targets[i].parameters()):
                p_t.data.copy_(self.tau * p.data + (1.0 - self.tau) * p_t.data)

    def Fed_Aggergate(self, method='DSFA', online_agents=None):
        """按在线园区集合聚合，返回完整权重矩阵（离线行列为零）。

        Args:
            method: DSFA、AllDSFA 或 FedAvg。
            online_agents: 本轮参与通信的园区索引，None 表示全部在线。

        Returns:
            成功时返回 n_agents × n_agents 权重矩阵；历史不足时返回 None。
        """
        online = list(range(self.n_agents)) if online_agents is None else list(online_agents)
        if len(set(online)) != len(online) or any(i < 0 or i >= self.n_agents for i in online):
            raise ValueError("online_agents 包含无效或重复的园区索引")
        if not online:
            return None
        if method not in ('DSFA', 'AllDSFA', 'FedAvg'):
            raise ValueError(f"未知聚合方法: {method}")
        if self.replay.size() < self.batch_size:
            return None
        if method != 'FedAvg' and any(not self.proto_history[i] for i in online):
            return None

        weights = np.zeros((self.n_agents, self.n_agents), dtype=np.float64)
        if method == 'FedAvg':
            weights[np.ix_(online, online)] = 1.0 / len(online)
        else:
            avg_proto = {
                j: torch.stack(self.proto_history[j], dim=0).mean(dim=0).to(device)
                for j in online
            }
            obs_b, _, _, _, _ = self.replay.sample(self.batch_size)
            obs_b_t = torch.as_tensor(obs_b, dtype=torch.float32, device=device)
            with torch.no_grad():
                proto_ref = {
                    i: self.actors[i](obs_b_t[:, self._obs_slices[i]]).mean(dim=0)
                    for i in online
                }
            for i in online:
                similarity = np.array([
                    1.0 / (F.l1_loss(proto_ref[i], avg_proto[j], reduction='mean').item() + 1e-8)
                    for j in online
                ], dtype=np.float64)
                weights[i, online] = similarity / similarity.sum()

        # 全部新参数取自同一聚合前快照，避免园区遍历顺序影响结果。
        with torch.no_grad():
            actor_snapshot = [[p.detach().clone() for p in actor.parameters()] for actor in self.actors]
            actor_updates = {}
            for i in online:
                actor_updates[i] = [
                    sum((weights[i, j] * actor_snapshot[j][k] for j in online))
                    for k in range(len(actor_snapshot[i]))
                ]
            for i, parameters in actor_updates.items():
                for target, value in zip(self.actors[i].parameters(), parameters):
                    target.copy_(value)

            if method == 'AllDSFA':
                critic_snapshot = [[p.detach().clone() for p in critic.parameters()] for critic in self.critics]
                for i in online:
                    for k, target in enumerate(self.critics[i].parameters()):
                        target.copy_(sum((weights[i, j] * critic_snapshot[j][k] for j in online)))

        # 仅清除参与通信的描述符。断联园区由 update 清理，恢复时从近期更新重新积累。
        for i in online:
            self.proto_history[i].clear()
        return weights

    # 保存 / 加载（对齐 MADDPG）
    def save(self, prefix="iddpg",Fed=False):
        file_path = f"./model_pth/{prefix}"
        if not os.path.exists(file_path):
            os.makedirs(file_path)
        for i in range(self.n_agents):
            torch.save(self.actors[i].state_dict(), f"{file_path}/{Fed}_actor_{i}.pth")
            torch.save(self.critics[i].state_dict(), f"{file_path}/{Fed}_critic_{i}.pth")

    def checkpoint_state(self):
        """导出可继续训练的完整状态，包括网络、优化器及经验缓存。"""
        return {
            "actors": [model.state_dict() for model in self.actors],
            "actor_targets": [model.state_dict() for model in self.actor_targets],
            "critics": [model.state_dict() for model in self.critics],
            "critic_targets": [model.state_dict() for model in self.critic_targets],
            "actor_opts": [opt.state_dict() for opt in self.actor_opts],
            "critic_opts": [opt.state_dict() for opt in self.critic_opts],
            "replay": list(self.replay.buffer),
            "proto_history": self.proto_history,
        }

    def load_checkpoint_state(self, state):
        """加载完整训练状态；网络及优化器顺序与园区索引一致。"""
        for name in ("actors", "actor_targets", "critics", "critic_targets"):
            for model, weights in zip(getattr(self, name), state[name]):
                model.load_state_dict(weights)
        for name in ("actor_opts", "critic_opts"):
            for optimizer, weights in zip(getattr(self, name), state[name]):
                optimizer.load_state_dict(weights)
        self.replay.buffer = deque(state["replay"], maxlen=self.replay.max_size)
        self.proto_history = state["proto_history"]

    def save_to_directory(self, directory):
        """保存分支模型参数至独立目录，沿用原模型文件命名规则。"""
        os.makedirs(directory, exist_ok=True)
        for i in range(self.n_agents):
            torch.save(self.actors[i].state_dict(), os.path.join(directory, f"True_actor_{i}.pth"))
            torch.save(self.critics[i].state_dict(), os.path.join(directory, f"True_critic_{i}.pth"))

    def load(self, prefix="iddpg",Fed=False):
        for i in range(self.n_agents):
            self.actors[i].load_state_dict(torch.load(f"./model_pth/{prefix}/{Fed}_actor_{i}.pth", map_location=device,weights_only=True))
            self.critics[i].load_state_dict(torch.load(f"./model_pth/{prefix}/{Fed}_critic_{i}.pth", map_location=device,weights_only=True))
