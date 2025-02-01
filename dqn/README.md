# 基于类 AlphaZero 深度强化学习的重力四子棋 AI

## 简介

在本文件夹中，使用类似 AlphaZero 的深度强化学习（Deep Q-Learning）进行自对弈训练神经网络，实现了一个简单的重力四子棋 AI。由于训练时间较短且优化很差，只训练了 1000EP（每个 EP 自对弈 20 轮），仅能战胜搜索深度为 2 的 Minimax AI。

## 文件结构

### 环境部分

- `gpu_test.py`：测试 Pytorch 环境下 GPU 是否可用
- `setup.py`：将 C++文件编译为 Python 模块
- `plot.py`：读取`train.log`文件并绘制训练曲线，输出为`merged_plot.png`

### 模型部分

- `train.py`：模型训练代码
- `mcts.cpp`：蒙特卡洛树搜索代码

### 游戏部分

- `play.py`：在纯 Python 环境下运行的游戏代码
- `play_cpp.py`：蒙特卡洛树搜索运行 C++环境下运行的游戏代码（更快）
- `pure_mcts.py`：纯 Python 实现的蒙特卡洛树搜索对战（仅用于测试）

## 训练

环境：

- Python 3.9.21
- Pytorch 2.3.1
- CUDA 11.8

运行以下命令进行训练：

```bash
pip install numpy pytorch
pip install . # 编译 C++ 代码
python train.py
```

配置详解：

```python
CONFIG = {
    "BOARD_ROWS": 5,               # 棋盘行数
    "BOARD_COLS": 5,               # 棋盘列数
    "MAX_STACK": 5,                # 棋盘最大堆叠数
    "RESIDUAL_BLOCKS": 10,         # 残差块数量
    "RES_FILTERS": 320,            # 残差块卷积核数量
    "POLICY_SIZE": 25,             # 动作空间大小(BOARD_ROWS*BOARD_COLS)
    "NUM_WORKERS": 2,              # 训练时使用的进程数
    "MCTS_SIMS": 6000,             # MCTS 模拟次数
    "GAMES_PER_WORKER": 10,        # 每个进程自对弈的游戏数
    "EPISODES": 2500,              # 训练 EP 数
    "UPDATE_FREQ": 20,             # 更新频率
    "BATCH_SIZE": 256,             # 批大小
    "REPLAY_BUFFER_SIZE": 50000,   # 经验回放缓冲区大小
    "LEARNING_RATE": 3e-5,         # 学习率
    "TEMP_INIT": 1.0,              # 初始温度
    "TEMP_FINAL": 0.01,            # 最终温度
    "TEMP_DECAY_EPISODES": 2000,   # 温度衰减 EP 数
    "SAVE_INTERVAL": 50,           # 保存间隔
    "CHECKPOINT_DIR": "./ckpt/",   # 存放权重/日志等
    "REPLAY_FILE": "replay_buffer.pkl", # 经验回放文件
    "TRAIN_STATE_FILE": "train_state.json", # 训练状态文件
    "EVAL_GAMES": 1,               # 评估时对局数
    "MINIMAX_DEPTH": 5,            # 评估时 Minimax 搜索深度
    "LOG_FILE": "train.log",       # 日志文件
}
```

每隔 `SAVE_INTERVAL` 个 EP 保存一次模型和回放文件，支持断点续训。

## 对战

选择 `play.py` 或 `play_cpp.py`中任意一个文件，将模型路径修改为训练好的模型路径，运行以下命令进行对战：

```bash
python play.py
```

## 训练结果（参考）

交替使用RTX 4090、RTX 3080训练约一周，训练曲线如下：

![训练结果](./merged_plot.png)

在训练过程中多次出现平台期，通过调整学习率、增加MCTS搜索次数进行突破。
