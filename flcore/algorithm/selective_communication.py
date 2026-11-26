"""DSFA 两阶段通信仿真：摘要筛选、按需 Actor 传输及有效载荷统计。"""

import random

import numpy as np
import torch


def select_neighbors(
    distances: np.ndarray, candidates: list[int], strategy: str,
    neighbors: int, rng: np.random.Generator,
) -> list[int]:
    """按距离选邻居；neighbors 不含自身，在线邻居不足时全部选取。

    mixed 保留 K-1 个最近邻，再随机探索一个；K=1 时退化为随机。
    距离相同按园区索引排序，保证结果可复现。
    """
    if strategy not in {'all', 'nearest', 'farthest', 'random', 'mixed'}:
        raise ValueError(f'未知通信选择策略: {strategy}')
    if neighbors < 0:
        raise ValueError('neighbors 不能为负数')
    if not np.isfinite(distances[candidates]).all():
        raise ValueError('候选动作距离包含非有限数值')
    # K=0 保留第一阶段摘要通信，但第二阶段不请求 Actor；自身仍参与聚合。
    if neighbors == 0:
        count = 0
    elif strategy == 'all':
        count = len(candidates)
    else:
        count = min(neighbors, len(candidates))
    ordered = sorted(candidates, key=lambda j: (float(distances[j]), j))
    if strategy in {'all', 'nearest'}:
        return ordered[:count]
    if strategy == 'farthest':
        return sorted(candidates, key=lambda j: (-float(distances[j]), j))[:count]
    if strategy == 'random':
        return sorted(rng.choice(candidates, size=count, replace=False).tolist())
    chosen = ordered[:max(0, count - 1)]
    if count:
        chosen.append(int(rng.choice([j for j in ordered if j not in chosen])))
    return chosen


def aggregate_selected(
    model, online_agents: list[int] | None = None, *, strategy: str = 'nearest',
    neighbors: int = 1, seed: int = 0,
) -> np.ndarray | None:
    """执行一次两阶段 DSFA，返回完整权重矩阵，历史不足则返回 None。

    使用本地历史动作均值到候选当前动作的距离。命名动作只比较共同项，
    以 RMSE 作为距离。自身始终参与，缺少共同动作的邻居不进入候选集。
    离线园区不收发。按有向单播累计 float32 摘要、int64 版本号、
    两个 int64 请求字段及 Actor 参数字节，不包含网络协议头和重传。
    通信记录保存在 model.communication_records；本函数不建立真实连接。
    """
    online = list(range(model.n_agents)) if online_agents is None else list(online_agents)
    if len(set(online)) != len(online) or any(i < 0 or i >= model.n_agents for i in online):
        raise ValueError('online_agents 包含无效或重复索引')
    # 在修改模型或消耗训练随机数之前校验配置。
    select_neighbors(np.zeros(model.n_agents), [], strategy, neighbors, np.random.default_rng(seed))
    if not online or model.replay.size() < model.batch_size:
        return None
    if any(not model.proto_history[i] for i in online):
        return None
    if not hasattr(model, 'communication_records'):
        model.communication_records = []
        model.communication_rng = np.random.default_rng(seed)
        model.communication_replay_rng = random.Random(seed)
    round_id = len(model.communication_records)
    match_indices = model.match_indices
    replay_rng = model.communication_replay_rng if match_indices is not None else None
    obs, _, _, _, _ = model.replay.sample(model.batch_size, rng=replay_rng)
    device = next(model.actors[online[0]].parameters()).device
    obs = torch.as_tensor(obs, dtype=torch.float32, device=device)
    with torch.no_grad():
        # 第一阶段只发布动作摘要；历史行为参考始终留在本地。
        descriptors = {}
        for j in online:
            summary = model.actors[j](obs[:, model._obs_slices[j]]).mean(0).cpu()
            descriptors[j] = summary if match_indices is None else {
                name: summary[index] for name, index in match_indices[j].items()
            }
        references = {i: torch.stack(model.proto_history[i]).mean(0).cpu() for i in online}
        distances = np.full((model.n_agents, model.n_agents), np.nan)
        weights = np.zeros_like(distances)
        selected = np.zeros_like(distances, dtype=bool)
        for i in online:
            candidates = []
            for j in online:
                if match_indices is None:
                    distances[i, j] = torch.linalg.vector_norm(references[i] - descriptors[j]).item()
                else:
                    common = sorted(match_indices[i].keys() & descriptors[j].keys())
                    if not common:
                        continue
                    local = references[i][[match_indices[i][name] for name in common]]
                    peer = torch.stack([descriptors[j][name] for name in common])
                    distances[i, j] = (local - peer).square().mean().sqrt().item()
                if j != i:
                    candidates.append(j)
            # 没有可比邻居时保留自身主干，包括自身没有配置任何匹配动作的园区。
            peers = select_neighbors(distances[i], candidates,
                                     strategy, neighbors, model.communication_rng)
            if not peers:
                weights[i, i] = 1.0
                continue
            sources = [i, *peers]
            selected[i, peers] = True
            inverse = 1.0 / (distances[i, sources] + 1e-8)
            weights[i, sources] = inverse / inverse.sum()

        # 第二阶段仅请求选中模型。先生成所有接收方更新，再统一提交，
        # 保证每次传输的都是本轮聚合前参数，避免原地更新污染后续接收方。
        updates = {}
        actor_bytes = 0
        for i in online:
            sources = [i, *np.flatnonzero(selected[i]).tolist()]
            if len(sources) == 1:
                continue
            parameters = [model.actor_aggregation_parameters(j) for j in sources]
            updates[i] = [sum(weights[i, j] * parameters[pos][k].detach()
                              for pos, j in enumerate(sources))
                          for k in range(len(parameters[0]))]
            for peer_parameters in parameters[1:]:
                actor_bytes += sum(p.numel() * p.element_size() for p in peer_parameters)
        for i, values in updates.items():
            for parameter, value in zip(model.actor_aggregation_parameters(i), values):
                parameter.copy_(value)

    # 异构模式的名称和顺序由实验配置预先共享，每轮只发送值和版本。
    descriptor_bytes = (len(online) - 1) * sum(
        (descriptors[j].numel() * descriptors[j].element_size()
         if match_indices is None else sum(value.element_size() for value in descriptors[j].values()))
        + 8 for j in online
    )
    request_bytes = int(selected.sum()) * 16
    model.communication_records.append({
        'round': round_id, 'strategy': strategy, 'neighbors': neighbors,
        'online': online, 'selected': selected.tolist(), 'weights': weights.tolist(),
        'descriptor_bytes': descriptor_bytes, 'request_bytes': request_bytes,
        'actor_bytes': actor_bytes,
        'total_bytes': descriptor_bytes + request_bytes + actor_bytes,
    })
    for i in online:
        model.proto_history[i].clear()
    return weights
