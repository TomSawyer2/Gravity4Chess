import math
import random
import sys

# ---------------------- 1. CONFIG ----------------------
CONFIG = {
    "BOARD_ROWS": 5,
    "BOARD_COLS": 5,
    "MAX_STACK": 5,
    "MCTS_SIMS": 2000,   # MCTS 模拟次数，可根据性能调整
    "C_PUCT": 1.4,       # UCB 常数，可微调
    "AI_PLAYER": "B",    # AI 使用 'B'
    "HUMAN_PLAYER": "W", # 人类玩家使用 'W'
}


# ---------------------- 2. ENVIRONMENT ----------------------
class StackedConnect4Env:
    """
    5x5 堆叠连四环境 (纯逻辑，无神经网络):
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
        self.current_player = CONFIG["HUMAN_PLAYER"]  # 人类先手
        self.done = False
        self.winner = None

    def step(self, action):
        """
        执行动作 action, action=(row*cols+col).
        若该位置超过 max_stack, 则视为非法(对手直接获胜).
        """
        if self.done:
            return

        r, c = divmod(action, self.cols)
        if len(self.board[r][c]) >= self.max_stack:
            # 非法动作 -> 对手胜
            self.done = True
            self.winner = (
                CONFIG["AI_PLAYER"] if self.current_player == CONFIG["HUMAN_PLAYER"]
                else CONFIG["HUMAN_PLAYER"]
            )
            return

        self.board[r][c].append(self.current_player)

        # 检查胜负
        w = self._check_winner_full()
        if w is not None:
            self.done = True
            self.winner = w
            return

        # 检查是否平局
        if self._board_full():
            self.done = True
            self.winner = None
            return

        # 切换玩家
        self.current_player = (
            CONFIG["AI_PLAYER"] if self.current_player == CONFIG["HUMAN_PLAYER"]
            else CONFIG["HUMAN_PLAYER"]
        )

    def clone(self):
        new_env = StackedConnect4Env()
        new_env.rows = self.rows
        new_env.cols = self.cols
        new_env.max_stack = self.max_stack
        new_env.board = [[col_stack[:] for col_stack in row] for row in self.board]
        new_env.current_player = self.current_player
        new_env.done = self.done
        new_env.winner = self.winner
        return new_env

    def legal_actions(self):
        if self.done:
            return []
        acts = []
        for i in range(self.rows * self.cols):
            r, c = divmod(i, self.cols)
            if len(self.board[r][c]) < self.max_stack:
                acts.append(i)
        return acts

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
            (0, 1, 0), (1, 0, 0), (1, 1, 0), (1, -1, 0),
            (0, 0, 1), (1, 0, 1), (-1, 0, 1), (0, 1, 1),
            (0, -1, 1), (1, 1, 1), (1, -1, 1),
            (-1, 1, 1), (-1, -1, 1),
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
                            if (0 <= rr < self.rows and
                                0 <= cc < self.cols and
                                0 <= ll < len(self.board[rr][cc])):
                                if self.board[rr][cc][ll] == piece:
                                    chain_count += 1
                                else:
                                    break
                            else:
                                break
                        if chain_count >= 4:
                            return piece
        return None


# ---------------------- 3. MCTS NODE & FORCED KILL CHECK ----------------------
class Node:
    """MCTS节点，不依赖NN。用随机rollout来估值。"""
    def __init__(self, env, parent=None):
        self.env = env
        self.parent = parent
        self.children = {}
        self.W = 0  # 累计价值
        self.N = 0  # 访问次数


def check_forced_moves(env: StackedConnect4Env):
    """
    检测“单步必杀/必防”：
    1) 若当前玩家能一步制胜，则 forced_actions 包含这些动作。
    2) 否则，检查对手下一步能否制胜 -> forced_actions包含防守动作。
    若没有，则返回空列表。
    """
    forced_actions = []
    current = env.current_player
    opponent = "B" if current == "W" else "W"

    # 1) 当前玩家是否能一步就赢
    for a in env.legal_actions():
        tmp = env.clone()
        tmp.step(a)
        if tmp.done and tmp.winner == current:
            forced_actions.append(a)
    if forced_actions:
        return forced_actions

    # 2) 对手下一步是否能赢 -> 需要防守
    for a in env.legal_actions():
        tmp = env.clone()
        tmp.step(a)  # 当前玩家先走
        if tmp.done:
            continue
        # 检查对手下一步
        opp_moves = tmp.legal_actions()
        for oa in opp_moves:
            tmp2 = tmp.clone()
            tmp2.step(oa)
            if tmp2.done and tmp2.winner == opponent:
                # 对手下这一步能赢 -> 我方必须在本回合下 a 去防守
                forced_actions.append(a)
                break

    return forced_actions


# ---------------------- 4. MCTS 核心函数 ----------------------
def mcts_search(root: Node, sims: int, c_puct: float = 1.4):
    """进行 MCTS 搜索 sims 次模拟。"""
    for _ in range(sims):
        node = root

        # ----------(A) Selection-----------
        while node.children and not node.env.done:
            # 在已扩展子节点里选择 UCB 最大者
            best_a = None
            best_ucb = -1e9
            for a, child in node.children.items():
                ucb = _ucb_score(child, c_puct)
                if ucb > best_ucb:
                    best_ucb = ucb
                    best_a = a
            node = node.children[best_a]

        # ----------(B) Expansion-----------
        if not node.env.done and len(node.children) == 0:
            forced_actions = check_forced_moves(node.env)
            acts = node.env.legal_actions()
            if forced_actions:
                acts = forced_actions

            for a in acts:
                tmp_env = node.env.clone()
                tmp_env.step(a)
                child = Node(tmp_env, parent=node)
                node.children[a] = child

        # ----------(C) Simulation (Rollout)----------
        leaf = node
        value = rollout(leaf.env.clone())  # 随机直到结束

        # ----------(D) Backpropagation----------
        temp = leaf
        while temp is not None:
            temp.N += 1
            # 根据 AI_PLAYER 的视角来记录W
            # 如果 leaf.env.current_player == AI_PLAYER，那么 value=1表示AI赢
            # 但 current_player 在游戏过程中会改变，因此这里需要根据节点是谁在落子做区分
            # 这里为了简化：若 leaf.env.winner == AI_PLAYER 则value=1; ==HUMAN则=-1
            # rollout函数已处理, 我们只简单加 W += value
            temp.W += value
            temp = temp.parent


def _ucb_score(node: Node, c_puct: float):
    """UCB1 = Q + c_puct * sqrt(ln(Np)/(1+Nc))"""
    if node.N == 0:
        return float('inf')
    Q = node.W / node.N
    # 父节点访问次数
    parent = node.parent
    Np = parent.N if parent else 1  # parent 可能为空
    return Q + c_puct * math.sqrt(math.log(Np + 1) / (node.N + 1e-8))


def rollout(env: StackedConnect4Env):
    """
    随机模拟直到终局:
    - 如果AI_PLAYER赢 -> return +1
    - 如果HUMAN_PLAYER赢 -> return -1
    - 平局 -> return 0
    """
    while not env.done:
        acts = env.legal_actions()
        if not acts:
            break
        a = random.choice(acts)
        env.step(a)

    if env.winner == CONFIG["AI_PLAYER"]:
        return +1
    elif env.winner == CONFIG["HUMAN_PLAYER"]:
        return -1
    else:
        return 0


# ---------------------- 5. 人机对战 ----------------------
def print_board(env: StackedConnect4Env):
    """打印当前棋盘 (行列从上到下，堆叠从左到右)."""
    print("当前棋盘:")
    for r in range(env.rows):
        row_str = []
        for c in range(env.cols):
            stack = env.board[r][c]
            if len(stack) == 0:
                row_str.append("[ ]")
            else:
                row_str.append("[" + "".join(stack) + "]")
        print(" ".join(row_str))
    print()

def human_vs_mcts():
    env = StackedConnect4Env()
    env.reset()

    print("欢迎来到 5x5 堆叠连四 (纯MCTS + 强制杀招检测)！")
    print(f"您使用 '{CONFIG['HUMAN_PLAYER']}'，AI 使用 '{CONFIG['AI_PLAYER']}'。")
    print(f"若想阻挡AI或击败AI，请输入 行 列，例如 '1 1' 表示第一行第一列。")
    print(f"MCTS模拟次数 = {CONFIG['MCTS_SIMS']}")
    print_board(env)

    while not env.done:
        if env.current_player == CONFIG["HUMAN_PLAYER"]:
            # 人类玩家回合
            while True:
                move_str = input("请落子 (行 列): ").strip()
                try:
                    rs, cs = move_str.split()
                    r = int(rs) - 1
                    c = int(cs) - 1
                    action = r * env.cols + c
                    if action not in env.legal_actions():
                        print("非法或堆叠上限，请重新输入。")
                        continue
                    env.step(action)
                    break
                except:
                    print("输入格式不正确，请重新输入。")

            print_board(env)
        else:
            # AI回合 (MCTS)
            print("AI 正在思考中...")
            root = Node(env.clone(), parent=None)
            mcts_search(root, sims=CONFIG["MCTS_SIMS"], c_puct=CONFIG["C_PUCT"])
            # 选择访问次数最高的子节点
            best_a, best_child = None, None
            best_n = -1
            for a, ch in root.children.items():
                if ch.N > best_n:
                    best_n = ch.N
                    best_a = a
                    best_child = ch

            if best_a is None:
                # 没找到子节点，说明无合法动作
                break
            env.step(best_a)
            r, c = divmod(best_a, env.cols)
            print(f"AI 落子: 行 {r+1}, 列 {c+1}")
            print_board(env)

    # 游戏结束
    if env.winner == CONFIG["HUMAN_PLAYER"]:
        print("\033[94m恭喜，您赢了！\033[0m")
    elif env.winner == CONFIG["AI_PLAYER"]:
        print("\033[91mAI 胜利了...\033[0m")
    else:
        print("\033[93m平局！\033[0m")

def main():
    human_vs_mcts()

if __name__ == "__main__":
    main()
