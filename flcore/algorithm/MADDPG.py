
import copy
import torch
import os
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import random
import numpy as np
from collections import deque
#导入模型
from flcore.Model import Actor,Critic

# ----------------------------
# MADDPG agent wrapper
# ----------------------------

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ----------------------------
# Replay buffer (joint)
# ----------------------------
class JointReplayBuffer:
    def __init__(self, max_size=100000):
        self.max_size = int(max_size)
        self.buffer = deque(maxlen=self.max_size)

    def add(self, joint_obs, joint_actions, joint_rewards, joint_next_obs, dones):
        # joint_* are numpy arrays / lists concatenated in fixed order
        # dones: list/array of per-agent done flags for the step
        self.buffer.append((joint_obs, joint_actions, joint_rewards, joint_next_obs, dones))

    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        obs_b, acts_b, rews_b, next_obs_b, dones_b = zip(*batch)
        # rews_b: list of per-step arrays (len = n_agents) -> convert to (B, n_agents)
        return (
            np.array(obs_b, dtype=np.float32),
            np.array(acts_b, dtype=np.float32),
            np.array(rews_b, dtype=np.float32),
            np.array(next_obs_b, dtype=np.float32),
            np.array(dones_b, dtype=np.float32),
        )

    def size(self):
        return len(self.buffer)

class MADDPG:
    def __init__(self, obs_dims, action_dims, max_actions,
                 lr_actor=1e-3, lr_critic=1e-3, gamma=0.95, tau=0.01,
                 batch_size=256, buffer_size=100000):
        """
        obs_dims: list of obs dim per agent
        action_dims: list of action dim per agent
        max_actions: list of max_action per agent (for scaling tanh outputs)
        """
        self.n_agents = len(obs_dims)
        self.obs_dims = obs_dims
        self.action_dims = action_dims
        self.max_actions = max_actions

        self.actors = []
        self.actor_targets = []
        self.actor_opts = []

        # critics and optimizers (one critic per agent, centralized inputs)
        self.critics = []
        self.critic_targets = []
        self.critic_opts = []

        self.Federated_proto = []
        self.proto_history = [[] for _ in range(self.n_agents)]

        total_obs = sum(obs_dims)
        total_action = sum(action_dims)

        for i in range(self.n_agents):
            actor = Actor(obs_dims[i], action_dims[i], max_actions[i]).to(device)
            # foot = actor.foot,
            # base = actor.base,
            # head = actor.head,
            # actor = BaseHeadSplit(head,base,foot)
            actor_t = copy.deepcopy(actor).to(device)
            opt_a = optim.Adam(actor.parameters(), lr=lr_actor)
            self.actors.append(actor)
            self.actor_targets.append(actor_t)
            self.actor_opts.append(opt_a)

            critic = Critic(total_obs, total_action).to(device)
            critic_t = copy.deepcopy(critic).to(device)
            opt_c = optim.Adam(critic.parameters(), lr=lr_critic)
            self.critics.append(critic)
            self.critic_targets.append(critic_t)
            self.critic_opts.append(opt_c)

        # replay
        self.replay = JointReplayBuffer(max_size=buffer_size)
        self.template = copy.deepcopy(self.actors[0])
        self.gamma = gamma
        self.tau = tau
        self.batch_size = batch_size

    def select_actions(self, obs_list, noise_scale=0.1):
        """obs_list: list of per-agent observations (numpy)"""
        actions = []
        for i, obs in enumerate(obs_list):
            s = torch.FloatTensor(obs.reshape(1, -1)).to(device)
            a = self.actors[i](s).detach().cpu().numpy().flatten()
            if noise_scale > 0:
                a = a + np.random.normal(0, noise_scale, size=a.shape)
            max_bound = self.max_actions[i]
            if np.isscalar(max_bound):
                a = np.clip(a, -max_bound, max_bound)
            else:
                max_arr = np.asarray(max_bound, dtype=np.float32)
                a = np.clip(a, -max_arr, max_arr)
            actions.append(a.astype(np.float32))
        return actions

    def update(self):
        if self.replay.size() < self.batch_size:
            return

        obs_b, acts_b, rews_b, next_obs_b, dones_b = self.replay.sample(self.batch_size)
        # shapes:
        # obs_b: (B, total_obs_dim)
        # acts_b: (B, total_action_dim)
        # rews_b: (B, n_agents)
        # next_obs_b: (B, total_obs_dim)
        # dones_b: (B, n_agents)

        # Convert to tensors
        obs_b_t = torch.FloatTensor(obs_b).to(device)
        acts_b_t = torch.FloatTensor(acts_b).to(device)
        next_obs_b_t = torch.FloatTensor(next_obs_b).to(device)
        rews_b_t = torch.FloatTensor(rews_b).to(device)  # shape (B, n_agents)
        dones_b_t = torch.FloatTensor(dones_b).to(device)  # shape (B, n_agents)

        # 计算所有 agent 的 proto（base 输出）并在 batch 维度取均值，追加到历史列表
        with torch.no_grad():
            obs_splits_all = torch.split(obs_b_t, self.obs_dims, dim=1)
            for j in range(self.n_agents):
                out_j = self.actors[j].head(obs_splits_all[j])
                proto_j = self.actors[j].base(out_j)
                proto_mean_j = proto_j.mean(dim=0).detach().cpu()  # (Feat,)
                self.proto_history[j].append(proto_mean_j)

        # For each agent, compute targets and update critic & actor
        self.Federated_proto = []
        for i in range(self.n_agents):
            # --------------------
            # Critic update
            # --------------------
            with torch.no_grad():
                # build next actions by actor_targets
                next_actions = []
                obs_splits_next = torch.split(next_obs_b_t, self.obs_dims, dim=1)
                for j in range(self.n_agents):
                    a_next = self.actor_targets[j](obs_splits_next[j])
                    next_actions.append(a_next)
                next_actions_cat = torch.cat(next_actions, dim=1)  # (B, total_action)

                # compute target Q using agent i's critic_target
                q_next = self.critic_targets[i](next_obs_b_t, next_actions_cat)
                # reward for agent i: rews_b_t[:, i:i+1]
                td_target = rews_b_t[:, i:i+1] + (1.0 - dones_b_t[:, i:i+1]) * (self.gamma * q_next)

            # current Q
            q_curr = self.critics[i](obs_b_t, acts_b_t)
            loss_q = nn.MSELoss()(q_curr, td_target.detach())
            self.critic_opts[i].zero_grad()
            loss_q.backward()
            self.critic_opts[i].step()

            # --------------------
            # Actor update (policy gradient)
            # --------------------
            obs_splits = torch.split(obs_b_t, self.obs_dims, dim=1)
            protos=[]
            curr_actions = []

            for j in range(self.n_agents):
                if j == i:
                    out = self.actors[j].head(obs_splits[j])
                    proto = self.actors[j].base(out)
                    curr_a = self.actors[j].foot(proto)
                else:
                    # detach other agents' actions (treat as constant)
                    out = self.actors[j].head(obs_splits[j]).detach()
                    proto = self.actors[j].base(out).detach()
                    curr_a = self.actors[j].foot(proto).detach()
                protos.append(proto)
                curr_actions.append(curr_a)
            curr_actions_cat = torch.cat(curr_actions, dim=1)
            proto_stack = torch.stack(protos)
            proto_cat = torch.mean(proto_stack, dim=0)

            # actor loss: maximize critic i's Q -> minimize -Q
            actor_loss = -self.critics[i](obs_b_t, curr_actions_cat).mean()
            self.Federated_proto.append(proto_cat)

            self.actor_opts[i].zero_grad()
            actor_loss.backward()
            self.actor_opts[i].step()


            # --------------------
            # Soft update targets for this agent (actor + critic)
            # --------------------

            for p, p_t in zip(self.actors[i].parameters(), self.actor_targets[i].parameters()):
                p_t.data.copy_(self.tau * p.data + (1.0 - self.tau) * p_t.data)
            for p, p_t in zip(self.critics[i].parameters(), self.critic_targets[i].parameters()):
                p_t.data.copy_(self.tau * p.data + (1.0 - self.tau) * p_t.data)

    def Fed_Aggergate(self):
        Federated_w = [[],[],[],[],[]]

        if self.replay.size() < self.batch_size:
            return

            # 历史为空也跳过（避免首次或长时间未 update 的情况）
        if any(len(h) == 0 for h in self.proto_history):
            return

        # 1) 计算每个 agent 的“历史参考 proto” = 对历史条目求平均
        avg_proto = []  # list of tensors (Feat,) on device
        for j in range(self.n_agents):
            hist = self.proto_history[j]  # list of cpu tensors (Feat,)
            avg = torch.stack(hist, dim=0).mean(dim=0).to(device)  # (Feat,)
            avg_proto.append(avg)

        obs_b,  _, _, _, _ = self.replay.sample(self.batch_size)

        obs_b_t = torch.FloatTensor(obs_b).to(device)
        obs_splits = torch.split(obs_b_t, self.obs_dims, dim=1)

        proto_ref = []  # 当前 batch 的 per-agent proto 均值 (Feat,)
        with torch.no_grad():
            for i in range(self.n_agents):
                out_i = self.actors[i].head(obs_splits[i])
                p_i = self.actors[i].base(out_i)
                proto_ref.append(p_i.mean(dim=0))  # (Feat,)

        # 3) 计算联邦权重矩阵 W[i][j]  ~  1 / ( L1(proto_ref[i], avg_proto[j]) + eps )
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

        # 4) 依据 W 做参数加权聚合（为每个 i 聚合得到一个新 actor_i）
        with torch.no_grad():
            new_actors = []
            for i in range(self.n_agents):
                # 先把模板清零
                for p in self.template.parameters():
                    p.data.zero_()
                # 累加各个 agent 的参数 * 权重
                for j in range(self.n_agents):
                    w_ij = Federated_w[i][j]
                    for tp, pj in zip(self.template.parameters(), self.actors[j].parameters()):
                        tp.data.add_(pj.data, alpha=w_ij)
                # 收集该行（对应 agent i）的聚合结果
                new_actors.append(copy.deepcopy(self.template))

            # 回写
            for i in range(self.n_agents):
                for p, np_ in zip(self.actors[i].parameters(), new_actors[i].parameters()):
                    p.data.copy_(np_.data)

        # 5) 清空历史，开始新一轮累计
        self.proto_history = [[] for _ in range(self.n_agents)]

    # 保存 / 加载（对齐 MADDPG）
    def save(self, prefix="maddpg",Fed=False):
        file_path = f"./model_pth/{prefix}"
        if not os.path.exists(file_path):
            os.makedirs(file_path)
        for i in range(self.n_agents):
            torch.save(self.actors[i].state_dict(), f"{file_path}/{Fed}_actor_{i}.pth")
            torch.save(self.critics[i].state_dict(), f"{file_path}/{Fed}_critic_{i}.pth")


    def load(self, prefix="maddpg",Fed=False):
            for i in range(self.n_agents):
                self.actors[i].load_state_dict(torch.load(f"./model_pth/{prefix}/{Fed}_actor_{i}.pth", map_location=device,weights_only=True))
                self.critics[i].load_state_dict(torch.load(f"./model_pth/{prefix}/{Fed}_critic_{i}.pth", map_location=device,weights_only=True))
