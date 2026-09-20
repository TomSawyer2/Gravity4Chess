# 基于 Minimax 的重力四子棋 AI

该目录包含原始 Minimax 引擎、优化后的引擎、正确性测试和可复现的 A/B benchmark。

优化版保持原有 5×5×5 棋规和评估权重，并移除了会影响搜索可靠性的置换表边界问题。主要优化包括：

- 落子/撤销代替每个节点复制棋盘；
- 增量维护 302 条胜线的计数和评估值；
- 仅检查最后落子涉及的胜线；
- 固定容量置换表，key 包含行棋方；
- 正确保存 `alphaOriginal` / `betaOriginal`；
- PVS、aspiration window、killer/history/战术走法排序；
- 根节点棋盘对称性去重；
- 只延伸立即获胜与唯一防守线的威胁搜索，降低 horizon effect；
- 迭代加深、节点数、NPS、TT 命中率及单步耗时统计。

Null Move、LMR 和 futility pruning 没有放入优化版，因为它们可能通过选择性裁剪降低确定性。

## 文件

- `gravity_engine.hpp`：棋盘、原始基线引擎和优化引擎；
- `play_cpp_opti.cpp`：优化版人机对战入口；
- `play_cpp_opti_legacy.cpp`：优化前源码快照；
- `engine_tests.cpp`：胜线、增量估值、make/unmake 和精确搜索回归；
- `benchmark.cpp`：固定深度性能测试和新旧引擎对战；
- `BENCHMARK.md`：完整测试方法与结果；
- `benchmark_*.csv`：逐步原始数据。

## 构建和测试

```bash
cd minimax
make
make test
```

也可以使用 CMake：

```bash
cmake -S minimax -B minimax/build -DCMAKE_BUILD_TYPE=Release
cmake --build minimax/build -j
ctest --test-dir minimax/build --output-on-failure
```

## 运行

默认搜索 9 层：

```bash
./minimax/play_cpp_opti
```

限制每步时间或修改 TT 大小：

```bash
./minimax/play_cpp_opti --depth 32 --time-ms 1000 --tt-mb 64
```

## Benchmark

固定深度性能：

```bash
./minimax/benchmark \
  --bench-depth 7 --depth-games 0 --time-games 0 \
  --output minimax/benchmark_performance.csv
```

100 局同深度对战：

```bash
./minimax/benchmark \
  --bench-depth 0 --depth-games 100 --time-games 0 \
  --match-depth 5 --opening-ply 4 \
  --output minimax/benchmark_equal_depth_100.csv
```

100 局同时间对战：

```bash
./minimax/benchmark \
  --bench-depth 0 --depth-games 0 --time-games 100 \
  --move-ms 100 --opening-ply 4 \
  --output minimax/benchmark_equal_time_100.csv
```

详见 [BENCHMARK.md](./BENCHMARK.md)。
