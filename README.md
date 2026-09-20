# [WIP] 重力四子棋

本仓库实现 5×5×5 三维重力四子棋：黑白双方轮流在 5×5 平面的某一列顶部落子，在任意三维方向率先连成四子的一方获胜。

仓库提供两类 AI：

- [Minimax AI](./minimax)：Bitboard、Zobrist Hash、Alpha-Beta/PVS、置换表、历史启发、对称性去重和强制威胁延伸；
- [类 AlphaZero 深度强化学习 AI](./dqn)：通过自对弈训练策略/价值网络，并结合 MCTS 选点。

## Web 人机对战

[Web UI](./web) 提供 Three.js 2.5D 棋盘、人机对战、逐手胜率曲线、全部落点估值、分层观察和逐手复盘。优化后的 C++ Minimax 通过 WebAssembly 运行在独立 Worker 中，不阻塞界面；本地开发与构建方式见 [web/README.md](./web/README.md)。

## Minimax 性能

优化版在 Apple M4 Pro、Clang 21、搜索深度 9 的空棋盘首步测试中：

- 原始版本：41.965 秒；
- 优化版本：0.647 秒；
- 端到端提速：64.9×；
- 两个版本均选择中心 `(3,3)`。

100 局新旧引擎对战结果：

- 同深度 5：优化版 53 胜，旧版 47 胜；
- 同为每步 100 ms：优化版 69 胜，旧版 31 胜。

实现、测试方法和逐步数据见 [Minimax benchmark](./minimax/BENCHMARK.md)。
