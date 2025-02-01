# 基于 Minimax 算法的重力四子棋 AI

## 简介

在本文件夹中，使用 Bitboard 存储棋盘状态，结合 Zobrist Hash、Alpha-Beta + Minimax 搜索、置换表、历史启发等优化手段，成功在 5×5×5 的三维棋盘上完成深度搜索，构建了一个高效的人工智能对弈系统。

## 文件结构

- `main.py`：使用 Python 实现的游戏代码
- `play_cpp.py`：使用 C++ 实现的游戏代码（更快）
- `play_cpp_multi.cpp`：使用 C++ 实现的最底层多线程搜索游戏代码（更快）
- `play_cpp_opti.cpp`：使用 Bitboard、Zobrist Hash、历史启发优化后的 C++ 游戏代码（更快）
- `data_analysis.py`：分析对局记录耗时数据

- `game_multi.exe`：编译后的多线程搜索游戏可执行文件
- `game_opti_7.exe`：编译后的优化游戏可执行文件（搜索深度为7）
- `game_opti_8.exe`：编译后的优化游戏可执行文件（搜索深度为8）
- `game_opti_9.exe`：编译后的优化游戏可执行文件（搜索深度为9）

C++ 文件为了保持与终端输出字符编码一致性，使用GBK编码。

配置详解：

```cpp
MAX_DEPTH = 7; // 最大搜索深度
```

## 实战效果

与计客智能四子棋AI难度三进行对战，结果如下：

- `draw.log`：搜索深度为8时的对战记录，当棋子耗完时仍未决出胜负，最终可以由Minimax算法取得胜利，搜索时间大致相当
- `win.log`：搜索深度为9时的对战记录，Minimax算法在搜索时间上较慢，但能战胜计客智能四子棋AI难度三

## 编译

```bash
g++ -std=c++17 -O3 -march=native -flto -funroll-loops -DNDEBUG -o ./minimax/game_opti.exe ./minimax/play_cpp_opti.cpp
```

## 运行

```bash
./minimax/game_opti_8.exe
```
