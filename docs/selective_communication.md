# 默认并行六组实验

```powershell
.\.venv\Scripts\python.exe run_communication.py
```

默认 nearest、farthest、random × neighbors=2、3，共六组；六进程（不超过 CPU 数），每进程一个 PyTorch 线程，训练数据 1 天，训练轮数仍为 1000。可用 --workers 2 限制并发。

快速检查（12 轮，足够积累 batch_size=128 并发生通信）：

```powershell
.\.venv\Scripts\python.exe run_communication.py --epochs 12
```

单组：--strategy nearest --neighbors 2。每批独立保存 config.json、summary.json；各组独立保存 training.log、模型、奖励和通信记录。

一天数据下使用跨 episode 累计步数预热、更新和通信；前 24 步预热，不再每个 episode 重新随机探索。经验不足时跳过聚合，约第 6 个 episode 开始出现通信。

四园区下 K=3 已覆盖全部邻居，三种策略选择集合相同，主要用于一致性对照；K=2 才能比较选择策略的影响。一天数据仅用于快速试验，不能替代全年评估。

nearest 选动作距离最小者，farthest 选最大者；mixed 选 K-1 个最近邻和一个随机邻居，K=1 时为随机。自身不占 K 名额，始终参与聚合。在线邻居不足 K 时全部选取。各策略均保持逆距离聚合，不反转权重。比较混合策略时请用 nearest K=2 对照，并重复多个 seed。

距离统一为本地历史动作摘要均值与候选当前 Actor 在采样本地观测上的批均值之间的 L2 距离，并非两个历史均值，也不是单个实时动作。历史摘要来自本地网络更新时的批次预测。

每次运行在 result/communication 下独立保存模型、奖励及 communication.json。按有向单播统计摘要、版本号、请求和实际参数张量的有效载荷字节，不包含协议头和重传。这是单进程逻辑通信仿真，并非真实网络服务。all 也计入请求，属于相同协议下的全邻居基线。

普通训练入口保持原调用方式；独立通信实验暂不接入双分支断联训练。聚合接口本身支持 online_agents，离线园区不收发且不聚合。原全量 DSFA/AllDSFA 的距离方向和 L2 度量已与论文统一，因此旧实验结果不能视作新实现的结果，需重新训练验证。
