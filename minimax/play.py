import os
import math
import sys

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

# ---------------------- 1. CONFIG ----------------------
CONFIG = {
    "BOARD_ROWS": 5,
    "BOARD_COLS": 5,
    "MAX_STACK": 5,
    "RESIDUAL_BLOCKS": 10,  # 最新网络的残差块数
    "RES_FILTERS": 320,  # 最新网络的通道数
    "POLICY_SIZE": 25,  # 5x5动作
    "MCTS_SIMS": 3000,  # 最新MCTS模拟次数
    "C_PUCT": 1.0,
    "AI_PLAYER": "B",
    "HUMAN_PLAYER": "W",
    "MODEL_CHECKPOINT": "./ckpt/ckpt_ep500.pth",  # 请根据实际路径修改
}


# ---------------------- 2. ENVIRONMENT ----------------------
class StackedConnect4Env:
    """
    5x5 堆叠连四环境:
    - 每格可堆叠 MAX_STACK 层.
    - current_player: 'W' 或 'B'
    - winner: 'W'/'B' 表示谁赢, None 表示未结束
    - done: 是否结束
    """

    def __init__(self, config=CONFIG):
        self.rows = config["BOARD_ROWS"]
        self.cols = config["BOARD_COLS"]
        self.max_stack = config["MAX_STACK"]
        self.reset()

    def reset(self):
        self.board = [[[] for _ in range(self.cols)] for _ in range(self.rows)]
        self.current_player = "W"  # 玩家1（白棋）先手
        self.done = False
        self.winner = None
        return self._get_obs()

    def step(self, action):
        if self.done:
            return self._get_obs(), 0, True
        r, c = divmod(action, self.cols)
        if len(self.board[r][c]) >= self.max_stack:
            # 非法动作 => 对手直接胜
            self.done = True
            self.winner = "B" if self.current_player == "W" else "W"
            return self._get_obs(), -1, True

        self.board[r][c].append(self.current_player)

        # 改为“全盘扫描”
        w = self._check_winner_full()
        if w is not None:
            self.done = True
            self.winner = w  # 'W' 或 'B'
            # 如果 w == self.current_player，可以返回奖励=1，否则=?
            if w == self.current_player:
                return self._get_obs(), 1, True
            else:
                return self._get_obs(), -1, True

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
        """
        全盘扫描：若有任意 4 连成线返回 'W' or 'B'，否则 None
        """
        directions_3d = [
            (0, 1, 0),
            (1, 0, 0),
            (1, 1, 0),
            (1, -1, 0),
            (0, 0, 1),
            (1, 0, 1),
            (-1, 0, 1),
            (0, 1, 1),
            (0, -1, 1),
            (1, 1, 1),
            (1, -1, 1),
            (-1, 1, 1),
            (-1, -1, 1),
        ]
        for r in range(self.rows):
            for c in range(self.cols):
                stack = self.board[r][c]
                for l, piece in enumerate(stack):
                    # piece 要么是 'W' 要么是 'B'
                    for dr, dc, dl in directions_3d:
                        rr, cc, ll = r, c, l
                        chain_count = 1
                        for _ in range(3):
                            rr += dr
                            cc += dc
                            ll += dl
                            if (
                                0 <= rr < self.rows
                                and 0 <= cc < self.cols
                                and 0 <= ll < len(self.board[rr][cc])
                            ):
                                if self.board[rr][cc][ll] == piece:
                                    chain_count += 1
                                else:
                                    break
                            else:
                                break
                        if chain_count >= 4:
                            return piece  # 'W' 或 'B'
        return None

    def _get_obs(self):
        """
        返回 shape=(5,5,2) 的numpy数组:
        - channel0: 'W' 棋子个数
        - channel1: 'B' 棋子个数
        """
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


# ---------------------- 3. RESNET MODEL ----------------------
class ResidualBlock(nn.Module):
    def __init__(self, filters=128):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Conv2d(filters, filters, kernel_size=3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(filters)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(filters, filters, kernel_size=3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(filters)
        self.dropout = nn.Dropout(0.1)

    def forward(self, x):
        shortcut = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.dropout(out)

        out += shortcut
        out = self.relu(out)
        return out


class AlphaZeroResNet(nn.Module):
    def __init__(
        self, board_size=(5, 5), num_res_blocks=6, filters=128, policy_size=25
    ):
        super(AlphaZeroResNet, self).__init__()
        self.conv_init = nn.Conv2d(2, filters, kernel_size=3, padding=1, bias=False)
        self.bn_init = nn.BatchNorm2d(filters)
        self.relu_init = nn.ReLU(inplace=True)

        self.res_blocks = nn.Sequential(
            *[ResidualBlock(filters) for _ in range(num_res_blocks)]
        )

        # Policy head
        self.policy_conv = nn.Conv2d(filters, 2, kernel_size=1)
        self.policy_bn = nn.BatchNorm2d(2)
        self.policy_relu = nn.ReLU(inplace=True)
        self.policy_flat = nn.Flatten()
        self.policy_fc = nn.Linear(2 * board_size[0] * board_size[1], policy_size)

        # Value head
        self.value_conv = nn.Conv2d(filters, 1, kernel_size=1)
        self.value_bn = nn.BatchNorm2d(1)
        self.value_relu = nn.ReLU(inplace=True)
        self.value_flat = nn.Flatten()
        self.value_fc1 = nn.Linear(board_size[0] * board_size[1], 128)
        self.value_dropout = nn.Dropout(0.1)
        self.value_fc2 = nn.Linear(128, 1)

    def forward(self, x):
        x = self.conv_init(x)
        x = self.bn_init(x)
        x = self.relu_init(x)

        x = self.res_blocks(x)

        # Policy head
        p = self.policy_conv(x)
        p = self.policy_bn(p)
        p = self.policy_relu(p)
        p = self.policy_flat(p)
        logits = self.policy_fc(p)

        # Value head
        v = self.value_conv(x)
        v = self.value_bn(v)
        v = self.value_relu(v)
        v = self.value_flat(v)
        v = self.value_fc1(v)
        v = self.value_dropout(v)
        v = self.value_fc2(v)
        v = torch.tanh(v)
        return logits, v


# ---------------------- 4. MCTS ----------------------
class Node:
    def __init__(self, env, parent=None):
        self.env = env
        self.parent = parent
        self.children = {}
        self.P = {}
        self.Q = {}
        self.N = {}
        self.value = 0
        self.player = env.current_player


def mcts_search(root, model, device, simulations=100, c_puct=1.0):
    model.eval()
    with torch.no_grad():
        for _ in range(simulations):
            node = root
            # Selection
            while True:
                if node.env.done or len(node.children) == 0:
                    break
                best_a, best_ucb = None, -1e9
                for a in node.env.legal_actions():
                    ucb = _ucb_score(node, a, c_puct)
                    if ucb > best_ucb:
                        best_ucb = ucb
                        best_a = a
                if best_a is None:
                    break
                node = node.children.get(best_a, None)
                if node is None:
                    break

            # Expansion
            if not node.env.done and len(node.children) == 0:
                obs = node.env._get_obs()
                obs_tensor = torch.from_numpy(obs).unsqueeze(0).to(device)
                obs_tensor = obs_tensor.permute(0, 3, 1, 2).float()
                logits, v = model(obs_tensor)
                logits = logits.cpu().numpy()[0]
                v = v.cpu().numpy()[0, 0]

                acts = node.env.legal_actions()
                maxA = np.max(logits[acts])
                exp_logits = np.exp(logits[acts] - maxA)
                sum_exp = np.sum(exp_logits) + 1e-8
                for a in acts:
                    node.P[a] = np.exp(logits[a] - maxA) / sum_exp
                    node.Q[a] = 0
                    node.N[a] = 0
                node.value = v
                for a in acts:
                    new_env = node.env.clone()
                    new_env.step(a)
                    child = Node(new_env, node)
                    node.children[a] = child

            # Backprop
            leaf_value = node.value
            leaf_player = node.player
            temp_node = node
            while temp_node:
                if temp_node.player == leaf_player:
                    val = leaf_value
                else:
                    val = -leaf_value
                if temp_node.parent:
                    pnode = temp_node.parent
                    for aa, ch in pnode.children.items():
                        if ch is temp_node:
                            pnode.N[aa] += 1
                            pnode.Q[aa] += (val - pnode.Q[aa]) / pnode.N[aa]
                            break
                temp_node = temp_node.parent


def _ucb_score(node, a, c_puct):
    sumN = sum(node.N[aa] for aa in node.N)
    Qa = node.Q[a]
    Pa = node.P[a]
    Na = node.N[a]
    U = c_puct * Pa * math.sqrt(sumN + 1e-8) / (1 + Na)
    return Qa + U


def get_mcts_policy(root, temp=1e-3):
    counts = [0] * CONFIG["POLICY_SIZE"]
    for a in root.N:
        counts[a] = root.N[a]
    counts = np.array(counts, dtype=np.float32)

    if np.all(counts == 0):
        return np.ones(CONFIG["POLICY_SIZE"], dtype=np.float32) / CONFIG["POLICY_SIZE"]

    if temp < 1e-8:
        best_a = np.argmax(counts)
        probs = np.zeros_like(counts)
        probs[best_a] = 1.0
        return probs

    counts_clipped = np.clip(counts, 1e-8, None)
    log_counts = np.log(counts_clipped) / temp
    cmax = np.max(log_counts)
    exps = np.exp(log_counts - cmax)
    sum_exps = np.sum(exps)
    if sum_exps < 1e-8:
        return np.ones_like(counts) / len(counts)
    return exps / sum_exps


# ---------------------- 5. LOAD TRAINED MODEL ----------------------
def load_trained_model(checkpoint_path, device):
    model = AlphaZeroResNet(
        board_size=(CONFIG["BOARD_ROWS"], CONFIG["BOARD_COLS"]),
        num_res_blocks=CONFIG["RESIDUAL_BLOCKS"],
        filters=CONFIG["RES_FILTERS"],
        policy_size=CONFIG["POLICY_SIZE"],
    )
    model.to(device)
    model.eval()
    if os.path.exists(checkpoint_path):
        model.load_state_dict(torch.load(checkpoint_path, map_location=device))
        print(f"模型已加载自: {checkpoint_path}")
    else:
        print(f"模型检查点文件 {checkpoint_path} 不存在！请检查路径。")
        sys.exit(1)
    return model


# ---------------------- 6. DISPLAY FUNCTIONS ----------------------
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


# ---------------------- 7. GAME LOOP ----------------------
def human_vs_ai(model, device):
    env = StackedConnect4Env(CONFIG)
    print("欢迎来到 5x5 堆叠连四游戏！")
    print(f"您是 '{CONFIG['HUMAN_PLAYER']}'，AI 是 '{CONFIG['AI_PLAYER']}'.")
    print("棋盘坐标从1开始，例如输入 '1 1' 表示第一行第一列。")
    print_board(env.board)

    while True:
        # 检查游戏是否结束
        if env.done:
            if env.winner == CONFIG["HUMAN_PLAYER"]:
                print("\033[94m恭喜，您赢了！\033[0m")
            elif env.winner == CONFIG["AI_PLAYER"]:
                print("\033[91mAI赢了！\033[0m")
            else:
                print("\033[93m平局！\033[0m")
            break

        if env.current_player == CONFIG["HUMAN_PLAYER"]:
            # 玩家回合
            while True:
                try:
                    move_str = input("请输入您的落子（行 列，例如1 1）：").strip()
                    rs, cs = move_str.split()
                    r = int(rs) - 1
                    c = int(cs) - 1
                    action = r * CONFIG["BOARD_COLS"] + c
                    if action not in env.legal_actions():
                        print("非法落子，请重新输入！")
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
            root = Node(env.clone())
            mcts_search(
                root,
                model,
                device,
                simulations=CONFIG["MCTS_SIMS"],
                c_puct=CONFIG["C_PUCT"],
            )
            pi = get_mcts_policy(root, temp=1e-3)  # 低温度，偏向最优
            action = np.argmax(pi)
            r, c = divmod(action, CONFIG["BOARD_COLS"])
            env.step(action)
            print(f"\033[92mAI 落子: 行 {r+1}, 列 {c+1}\033[0m")
            print_board(env.board)


# ---------------------- 8. MAIN ----------------------
def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"使用设备: {device}")

    model = load_trained_model(CONFIG["MODEL_CHECKPOINT"], device)
    human_vs_ai(model, device)


if __name__ == "__main__":
    main()
