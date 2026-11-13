"""DSFA 两阶段通信仿真：摘要筛选、按需 Actor 传输及有效载荷统计。"""

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
    count = len(candidates) if strategy == 'all' else min(neighbors, len(candidates))
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

    使用本地历史动作均值到候选当前动作的 L2 距离。自身始终参与，
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
    round_id = len(model.communication_records)
    obs, _, _, _, _ = model.replay.sample(model.batch_size)
    device = next(model.actors[online[0]].parameters()).device
    obs = torch.as_tensor(obs, dtype=torch.float32, device=device)
    with torch.no_grad():
        # 第一阶段只发布动作摘要；历史行为参考始终留在本地。
        descriptors = {
            j: model.actors[j](obs[:, model._obs_slices[j]]).mean(0).cpu()
            for j in online
        }
        references = {i: torch.stack(model.proto_history[i]).mean(0).cpu() for i in online}
        distances = np.full((model.n_agents, model.n_agents), np.nan)
        weights = np.zeros_like(distances)
        selected = np.zeros_like(distances, dtype=bool)
        for i in online:
            for j in online:
                distances[i, j] = torch.linalg.vector_norm(references[i] - descriptors[j]).item()
            peers = select_neighbors(distances[i], [j for j in online if j != i],
                                     strategy, neighbors, model.communication_rng)
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
            parameters = [list(model.actors[j].parameters()) for j in sources]
            updates[i] = [sum(weights[i, j] * parameters[pos][k].detach()
                              for pos, j in enumerate(sources))
                          for k in range(len(parameters[0]))]
            for j in sources[1:]:
                actor_bytes += sum(p.numel() * p.element_size() for p in model.actors[j].parameters())
        for i, values in updates.items():
            for parameter, value in zip(model.actors[i].parameters(), values):
                parameter.copy_(value)

    descriptor_bytes = sum(descriptors[j].numel() * descriptors[j].element_size() + 8
                           for i in online for j in online if i != j)
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
