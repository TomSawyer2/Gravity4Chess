import os
import sys
import time
import threading

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

import mcts_cpp


# -------------------- 全局配置 --------------------
CONFIG = {
    "BOARD_ROWS": 5,
    "BOARD_COLS": 5,
    "MAX_STACK": 5,
    "POLICY_SIZE": 25,
    "MCTS_SIMS": 6000,
    "C_PUCT": 1.0,
    "AI_PLAYER": "W",
    "HUMAN_PLAYER": "B",
    "MODEL_CHECKPOINT": "./ckpt/ckpt_ep975.pth",
}


# -------------------- 1. 环境定义 --------------------
class StackedConnect4Env:
    """
    简化的 5x5 堆叠连四环境 (Python仅用于人机对战的显示与输入)。
    """
    def __init__(self, config=CONFIG):
        self.rows = config["BOARD_ROWS"]
        self.cols = config["BOARD_COLS"]
        self.max_stack = config["MAX_STACK"]
        self.reset()

    def reset(self):
        self.board = [[[] for _ in range(self.cols)] for _ in range(self.rows)]
        self.current_player = "W"  # 先手
        self.done = False
        self.winner = None
        return self._get_obs()

    def step(self, action):
        if self.done:
            return self._get_obs(), 0, True
        r, c = divmod(action, self.cols)
        # 堆叠超限 => 对手胜
        if len(self.board[r][c]) >= self.max_stack:
            self.done = True
            self.winner = "B" if self.current_player == "W" else "W"
            return self._get_obs(), -1, True

        self.board[r][c].append(self.current_player)

        # 检查连4
        w = self._check_winner_full()
        if w is not None:
            self.done = True
            self.winner = w
            if w == self.current_player:
                return self._get_obs(), 1, True
            else:
                return self._get_obs(), -1, True

        # 检查平局
        if self._board_full():
            self.done = True
            self.winner = None
            return self._get_obs(), 0, True

        # 切换玩家
        self.current_player = "B" if self.current_player == "W" else "W"
        return self._get_obs(), 0, False

    def _board_full(self):
        for r in range(self.rows):
            for c in range(self.cols):
                if len(self.board[r][c]) < self.max_stack:
                    return False
        return True

    def _check_winner_full(self):
        directions_3d = [
            (0, 1, 0), (1, 0, 0), (1, 1, 0), (1, -1, 0),
            (0, 0, 1), (1, 0, 1), (-1, 0, 1), (0, 1, 1),
            (0, -1, 1), (1, 1, 1), (1, -1, 1), (-1, 1, 1), (-1, -1, 1),
        ]
        for r in range(self.rows):
            for c in range(self.cols):
                stack = self.board[r][c]
                for l, piece in enumerate(stack):
                    for dr, dc, dl in directions_3d:
                        rr, cc, ll = r, c, l
                        chain_count = 1
                        for _ in range(3):
                            rr += dr
                            cc += dc
                            ll += dl
                            if (
                                0 <= rr < self.rows and
                                0 <= cc < self.cols and
                                0 <= ll < len(self.board[rr][cc])
                            ):
                                if self.board[rr][cc][ll] == piece:
                                    chain_count += 1
                                else:
                                    break
                            else:
                                break
                        if chain_count >= 4:
                            return piece
        return None

    def _get_obs(self):
        obs = np.zeros((self.rows, self.cols, 2), dtype=np.float32)
        for r in range(self.rows):
            for c in range(self.cols):
                w_count = sum(1 for x in self.board[r][c] if x == "W")
                b_count = sum(1 for x in self.board[r][c] if x == "B")
                obs[r, c, 0] = w_count
                obs[r, c, 1] = b_count
        return obs

    def legal_actions(self):
        acts = []
        for i in range(self.rows * self.cols):
            r, c = divmod(i, self.cols)
            if len(self.board[r][c]) < self.max_stack:
                acts.append(i)
        return acts

    def clone(self):
        newE = StackedConnect4Env()
        newE.rows = self.rows
        newE.cols = self.cols
        newE.max_stack = self.max_stack
        newE.board = [[col_stack[:] for col_stack in row] for row in self.board]
        newE.current_player = self.current_player
        newE.done = self.done
        newE.winner = self.winner
        return newE


# -------------------- 2. 训练时的网络结构 (示例) --------------------
class ResidualBlock(nn.Module):
    def __init__(self, filters=128):
        super().__init__()
        self.conv1 = nn.Conv2d(filters, filters, kernel_size=3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(filters)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(filters, filters, kernel_size=3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(filters)
        self.drop = nn.Dropout(0.1)

    def forward(self, x):
        shortcut = x
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.drop(x)
        x += shortcut
        x = self.relu(x)
        return x


class AlphaZeroResNet(nn.Module):
    def __init__(self, board_size=(5, 5), num_res_blocks=10, filters=320, policy_size=25):
        super().__init__()
        self.conv_init = nn.Conv2d(2, filters, 3, padding=1, bias=False)
        self.bn_init = nn.BatchNorm2d(filters)
        self.relu_init = nn.ReLU(inplace=True)

        blocks = []
        for _ in range(num_res_blocks):
            blocks.append(ResidualBlock(filters))
        self.res_blocks = nn.Sequential(*blocks)

        # policy head
        self.policy_conv = nn.Conv2d(filters, 2, kernel_size=1)
        self.policy_bn = nn.BatchNorm2d(2)
        self.policy_relu = nn.ReLU(inplace=True)
        self.policy_fc = nn.Linear(2 * board_size[0] * board_size[1], policy_size)

        # value head
        self.value_conv = nn.Conv2d(filters, 1, kernel_size=1)
        self.value_bn = nn.BatchNorm2d(1)
        self.value_relu = nn.ReLU(inplace=True)
        self.value_fc1 = nn.Linear(board_size[0] * board_size[1], 128)
        self.value_drop = nn.Dropout(0.1)
        self.value_fc2 = nn.Linear(128, 1)

    def forward(self, x):
        x = self.conv_init(x)
        x = self.bn_init(x)
        x = self.relu_init(x)
        x = self.res_blocks(x)

        # policy
        p = self.policy_conv(x)
        p = self.policy_bn(p)
        p = self.policy_relu(p)
        p = p.reshape(p.size(0), -1)
        logits = self.policy_fc(p)

        # value
        v = self.value_conv(x)
        v = self.value_bn(v)
        v = self.value_relu(v)
        v = v.reshape(v.size(0), -1)
        v = self.value_fc1(v)
        v = F.relu(v)
        v = self.value_drop(v)
        v = self.value_fc2(v)
        v = torch.tanh(v)
        return logits, v


def load_trained_model(checkpoint_path, device):
    model = AlphaZeroResNet(
        board_size=(CONFIG["BOARD_ROWS"], CONFIG["BOARD_COLS"]),
        num_res_blocks=10,
        filters=320,
        policy_size=CONFIG["POLICY_SIZE"],
    )
    model.to(device)
    if os.path.exists(checkpoint_path):
        model.load_state_dict(torch.load(checkpoint_path, map_location=device))
        print(f"模型已加载自: {checkpoint_path}")
    else:
        print(f"模型检查点文件 {checkpoint_path} 不存在！")
        sys.exit(1)
    model.eval()
    return model


# -------------------- 3. 显示用函数 --------------------
def print_board(board):
    print("当前棋盘（底→顶）:")
    for r in range(CONFIG["BOARD_ROWS"]):
        row_str = []
        for c in range(CONFIG["BOARD_COLS"]):
            stack = board[r][c]
            if len(stack) == 0:
                row_str.append("[ ]")
            else:
                row_str.append("[" + "".join(stack) + "]")
        print(" ".join(row_str))
    print()


# -------------------- 4. AI回合: 多线程 + 旋转符号 --------------------
spinner_chars = ['|', '/', '-', '\\']

def ai_move_with_spinner(board_cpp, nn_infer_py, sims, c_puct, temp):
    """
    通过多线程调用 C++ MCTS，在主线程打印旋转符号 + 耗时。
    返回最终action。
    """
    result_action = None
    done_flag = False

    def run_mcts():
        nonlocal result_action, done_flag
        # 真正进行 C++ MCTS 计算
        result_action = mcts_cpp.mcts_get_action_once(
            board_cpp,
            nn_infer_py,  # Python回调: obs -> (logits, value)
            sims,
            c_puct,
            temp
        )
        done_flag = True

    # 启动子线程跑 MCTS
    t = threading.Thread(target=run_mcts)
    t.start()

    # 主线程做旋转动画
    start_time = time.time()
    spin_idx = 0
    while not done_flag:
        elapsed = time.time() - start_time
        spin_char = spinner_chars[spin_idx % len(spinner_chars)]
        spin_idx += 1
        # 同行覆盖打印
        sys.stdout.write(
            f"\r\033[90m[AI计算] {spin_char} 耗时={elapsed:.2f}s\033[0m"
        )
        sys.stdout.flush()
        time.sleep(0.2)

    # 等子线程结束，拿到action
    t.join()
    print()  # 换行
    return result_action


# -------------------- 5. 人机对战 --------------------
def human_vs_ai(model, device):
    env = StackedConnect4Env(CONFIG)
    print("欢迎来到 5x5 堆叠连四游戏！")
    print(f"您是 '{CONFIG['HUMAN_PLAYER']}'，AI 是 '{CONFIG['AI_PLAYER']}'.")
    print("棋盘坐标从1开始（行 列），如输入 '1 1' 表示第一行第一列。")
    print_board(env.board)

    # Python => C++ 的网络推理回调
    def nn_infer_py(obs_list):
        # obs_list 是一个 5x5x2 的嵌套列表, 需转 np.ndarray
        obs_np = np.array(obs_list, dtype=np.float32)
        obs_tensor = torch.from_numpy(obs_np).unsqueeze(0).to(device)  # [1,5,5,2]
        obs_tensor = obs_tensor.permute(0, 3, 1, 2).float()            # [1,2,5,5]
        with torch.no_grad():
            logits, v = model(obs_tensor)
        logits = logits.cpu().numpy()[0]  # shape=[25]
        v = v.cpu().numpy()[0, 0]         # 标量
        return (logits.tolist(), float(v))

    # 创建 C++ 侧的 Board
    board_cpp = mcts_cpp.Board()

    # 一个小函数：把Python env状态同步到C++ board
    def sync_env_to_cpp(env_py, board_c):
        board_c.reset()
        # 这里可用"重放落子"或者在C++端加set_state(...)，略
        # (示例简化，假设一次性没有冲突即可)
        for r in range(env_py.rows):
            for c in range(env_py.cols):
                for piece in env_py.board[r][c]:
                    # 如果 c++ board current_player != piece => 做一次空step切换
                    # 或者写个 set_current_player(piece)
                    # demo省略...
                    board_c.step(r * env_py.cols + c)

        # 最后看看 env_py.current_player 和 C++ current_player 是否匹配
        # 也要处理 env_py.done / winner 等，demo省略

    while True:
        if env.done:
            if env.winner == CONFIG["HUMAN_PLAYER"]:
                print("\033[94m恭喜，您赢了！\033[0m")
            elif env.winner == CONFIG["AI_PLAYER"]:
                print("\033[91mAI赢了！\033[0m")
            else:
                print("\033[93m平局！\033[0m")
            break

        if env.current_player == CONFIG["HUMAN_PLAYER"]:
            # 人类回合
            while True:
                try:
                    move_str = input("请输入您的落子（行 列）：").strip()
                    rs, cs = move_str.split()
                    r = int(rs) - 1
                    c = int(cs) - 1
                    action = r * CONFIG["BOARD_COLS"] + c
                    if action not in env.legal_actions():
                        print("非法落子，请重新输入。")
                        continue
                    env.step(action)
                    print_board(env.board)
                    break
                except ValueError:
                    print("输入格式有误，请重新输入。")
                except Exception as e:
                    print(f"发生错误: {e}")
        else:
            # AI回合
            print("AI 正在思考...")
            # 同步Python env -> c++ board (这里仅示例)
            sync_env_to_cpp(env, board_cpp)

            action = ai_move_with_spinner(
                board_cpp,
                nn_infer_py,
                sims=CONFIG["MCTS_SIMS"],
                c_puct=CONFIG["C_PUCT"],
                temp=0
            )
            r, c = divmod(action, CONFIG["BOARD_COLS"])
            env.step(action)
            print(f"\033[92mAI 落子: 行 {r+1}, 列 {c+1}\033[0m")
            print_board(env.board)


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"使用设备: {device}")

    model = load_trained_model(CONFIG["MODEL_CHECKPOINT"], device)
    human_vs_ai(model, device)


if __name__ == "__main__":
    main()
