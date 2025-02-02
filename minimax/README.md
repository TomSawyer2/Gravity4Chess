# 基于 Minimax 算法的重力四子棋 AI

## 简介

在本文件夹中，使用 Minimax 搜索 + Alpha-Beta 剪枝 + 置换表实现了重力四子棋 AI。此后，使用多线程实现了更快的游戏 AI。进一步，使用 Bitboard 存储棋盘状态，结合 Zobrist Hash、历史启发等优化手段，成功在 5×5×5 的三维棋盘上完成深度搜索，构建了一个高效的人工智能对弈系统。对于具体技术细节，见[基于 Minimax 算法的重力四子棋 AI 技术报告](https://github.com/TomSawyer2/Gravity4Chess/tree/main/minimax/Report.md)

## 文件结构

- `main.py`：使用 Python 实现的游戏代码
- `play_cpp.py`：使用 C++ 实现的游戏代码（更快）
- `play_cpp_multi.cpp`：使用 C++ 实现的最底层多线程搜索游戏代码（更快）
- `play_cpp_opti.cpp`：使用 Bitboard、Zobrist Hash、历史启发优化后的 C++ 游戏代码（更快）
- `data_analysis.py`：分析对局记录耗时数据

- `play_cpp.exe`：编译后的游戏可执行文件
- `play_cpp_multi.exe`：编译后的多线程搜索游戏可执行文件
- `play_opti_7.exe`：编译后的优化游戏可执行文件（搜索深度为 7）
- `play_opti_8.exe`：编译后的优化游戏可执行文件（搜索深度为 8）
- `play_opti_9.exe`：编译后的优化游戏可执行文件（搜索深度为 9）

配置详解：

```cpp
MAX_DEPTH = 7; // 最大搜索深度
```

## 实战效果

与计客智能四子棋 AI 难度三进行对战，结果如下：

`draw.log`：搜索深度为 8 时的对战记录，当棋子耗完时仍未决出胜负，最终可以由 Minimax 算法取得胜利，搜索时间大致相当

对战数据：

共计算了 38 步

- 总用时：80.28800009s
- 平均用时：2.112842107631579s
- 最大用时：8.907s
- 最小用时：0s
- 用时标准差：2.9496884075970855
- 用时方差：8.70066170191263
- 用时中位数：0.0

`win.log`：搜索深度为 9 时的对战记录，Minimax 算法在搜索时间上较慢，但能战胜计客智能四子棋 AI 难度三

对战数据：

- 共计算了 22 步
- 总用时：879.08299999s
- 平均用时：39.95831818136364s
- 最大用时：117.191s
- 最小用时：0s
- 用时标准差：34.07500764994248
- 用时方差：1161.1061463436386
- 用时中位数：37.278499999999994

## 编译

```bash
g++ -std=c++17 -O3 -march=native -flto -funroll-loops -DNDEBUG -o ./minimax/game_opti_8.exe ./minimax/play_cpp_opti.cpp
```

## 运行

```bash
./minimax/game_opti_8.exe
```
