import math
import copy
import hashlib
import time
import sys

PLAYER = 'B'  # 玩家(白棋)
AI = 'W'      # AI(黑棋)

ROWS = 5
COLS = 5
MAX_STACK = 5  # 每格最多可堆叠 5 层

# -- 全局变量，用于实时打印搜索进度/转圈 --
nodes_explored = 0        # 本次搜索已访问的节点数
search_start_time = 0     # 本次搜索开始时间
spinner = ["|","/","-","\\"]  # 转圈动画用

transposition_table = {}

def create_board():
    return [[[] for _ in range(COLS)] for _ in range(ROWS)]

def is_valid_move(board, row, col):
    if 0 <= row < ROWS and 0 <= col < COLS:
        return len(board[row][col]) < MAX_STACK
    return False

def make_move(board, row, col, piece):
    new_board = copy.deepcopy(board)
    new_board[row][col].append(piece)
    return new_board

def is_board_full(board):
    for r in range(ROWS):
        for c in range(COLS):
            if len(board[r][c]) < MAX_STACK:
                return False
    return True

def board_to_str(board):
    rows_str = []
    for r in range(ROWS):
        cols_str = []
        for c in range(COLS):
            stack_str = ''.join(board[r][c])
            cols_str.append(stack_str + '|')
        rows_str.append('#'.join(cols_str))
    return '##'.join(rows_str)

def hash_board(board):
    return hashlib.md5(board_to_str(board).encode('utf-8')).hexdigest()

def check_winner(board):
    """
    若有一方 4 连成线返回 'W' or 'B'，否则 None
    """
    directions_3d = [
        (0,1,0), (1,0,0), (1,1,0), (1,-1,0),
        (0,0,1), (1,0,1), (-1,0,1),
        (0,1,1), (0,-1,1),
        (1,1,1), (1,-1,1), (-1,1,1), (-1,-1,1),
    ]
    for r in range(ROWS):
        for c in range(COLS):
            stack = board[r][c]
            for l, p in enumerate(stack):
                for (dr, dc, dl) in directions_3d:
                    rr, cc, ll = r, c, l
                    chain_count = 1
                    for _ in range(3):
                        rr += dr
                        cc += dc
                        ll += dl
                        if (0 <= rr < ROWS and 0 <= cc < COLS and 0 <= ll < len(board[rr][cc])):
                            if board[rr][cc][ll] == p:
                                chain_count += 1
                            else:
                                break
                        else:
                            break
                    if chain_count >= 4:
                        return p
    return None

# ---------- 多类预估分 (模式匹配) ----------
black_pattern_score = {
    "BBBB": 9999999,
    "BBB-": 5000,
    "BB-B": 5000,
    "B-BB": 5000,
    "-BBB": 5000,
    "BB--": 200,
    "B-B-": 150,
    "B--B": 100,
    "-BB-": 200,
    "B---": 10,
    "-B--": 10,
    "--B-": 10,
    "---B": 10,
    "----": 0
}

white_pattern_score = {
    "WWWW": 9999999,
    "WWW-": 5000,
    "WW-W": 5000,
    "W-WW": 5000,
    "-WWW": 5000,
    "WW--": 200,
    "W-W-": 150,
    "W--W": 100,
    "-WW-": 200,
    "W---": 10,
    "-W--": 10,
    "--W-": 10,
    "---W": 10,
    "----": 0
}

def get_all_lines(board):
    directions_3d = [
        (0,1,0), (1,0,0), (1,1,0), (1,-1,0),
        (0,0,1), (1,0,1), (-1,0,1),
        (0,1,1), (0,-1,1),
        (1,1,1), (1,-1,1), (-1,1,1), (-1,-1,1),
    ]
    lines = []
    for r in range(ROWS):
        for c in range(COLS):
            stack_height = len(board[r][c])
            for l in range(stack_height):
                for (dr, dc, dl) in directions_3d:
                    coords = [(r,c,l)]
                    rr, cc, ll = r, c, l
                    for _ in range(3):
                        rr += dr
                        cc += dc
                        ll += dl
                        if 0 <= rr < ROWS and 0 <= cc < COLS and 0 <= ll < MAX_STACK:
                            coords.append((rr, cc, ll))
                        else:
                            break
                    if len(coords) == 4:
                        lines.append(coords)
    return lines

def evaluate(board):
    w = check_winner(board)
    if w == AI:
        return 99999999
    elif w == PLAYER:
        return -99999999

    ai_potential = 0
    player_potential = 0
    lines = get_all_lines(board)

    for coords in lines:
        pieces = []
        for (r, c, l) in coords:
            if len(board[r][c]) > l:
                pieces.append(board[r][c][l])
            else:
                pieces.append(None)

        if 'B' in pieces and 'W' in pieces:
            continue  # 同时有 B、W，不对任何人加分

        pattern = ''.join(
            'B' if p == AI else ('W' if p == PLAYER else '-')
            for p in pieces
        )
        if 'B' in pattern and 'W' not in pattern:
            ai_potential += black_pattern_score.get(pattern, 0)
        elif 'W' in pattern and 'B' not in pattern:
            player_potential += white_pattern_score.get(pattern, 0)
    
    return ai_potential - player_potential

# ---------------- Minimax + Alpha-Beta ----------------
def minimax(board, depth, alpha, beta, maximizing):
    """
    在这里进行节点计数，并用转圈动画做“进度条”效果。
    """
    global nodes_explored, search_start_time, spinner

    # 先计一次节点
    nodes_explored += 1

    # 每当访问节点数到一定阈值，就刷新“进度条”到同一行
    refresh_threshold = 200  # 每访问200个节点刷新一次
    if nodes_explored % refresh_threshold == 0:
        # 计算已用时间
        spent = time.time() - search_start_time
        # 选一个动画字符
        spin_char = spinner[(nodes_explored // refresh_threshold) % len(spinner)]
        # 在同一行打印，用 \r 回到行首覆盖
        sys.stdout.write(
            f"\r\033[90m[AI搜索] {spin_char}  已访问节点={nodes_explored}, 耗时={spent:.2f}s\033[0m"
        )
        sys.stdout.flush()

    # 置换表
    bkey = hash_board(board)
    if bkey in transposition_table:
        stored_depth, stored_alpha, stored_beta, stored_val, stored_move = transposition_table[bkey]
        if stored_depth >= depth and stored_alpha <= alpha and stored_beta >= beta:
            return (stored_val, stored_move[0], stored_move[1])

    # 终止条件
    score = evaluate(board)
    if abs(score) >= 99999999 or depth == 0 or is_board_full(board):
        return (score, None, None)

    if maximizing:
        best_val = -math.inf
        best_move = (None, None)
        moves = []
        for r in range(ROWS):
            for c in range(COLS):
                if is_valid_move(board, r, c):
                    moves.append((r,c))
        center_r, center_c = ROWS//2, COLS//2
        moves.sort(key=lambda m: abs(m[0]-center_r) + abs(m[1]-center_c))

        for (r,c) in moves:
            new_board = make_move(board, r, c, AI)
            val, _, _ = minimax(new_board, depth-1, alpha, beta, False)
            if val > best_val:
                best_val = val
                best_move = (r,c)
            alpha = max(alpha, best_val)
            if alpha >= beta:
                break

        transposition_table[bkey] = (depth, alpha, beta, best_val, best_move)
        return (best_val, best_move[0], best_move[1])
    else:
        best_val = math.inf
        best_move = (None, None)
        moves = []
        for r in range(ROWS):
            for c in range(COLS):
                if is_valid_move(board, r, c):
                    moves.append((r,c))
        center_r, center_c = ROWS//2, COLS//2
        moves.sort(key=lambda m: abs(m[0]-center_r) + abs(m[1]-center_c))

        for (r,c) in moves:
            new_board = make_move(board, r, c, PLAYER)
            val, _, _ = minimax(new_board, depth-1, alpha, beta, True)
            if val < best_val:
                best_val = val
                best_move = (r,c)
            beta = min(beta, best_val)
            if alpha >= beta:
                break

        transposition_table[bkey] = (depth, alpha, beta, best_val, best_move)
        return (best_val, best_move[0], best_move[1])

def search_best_move(board, max_depth, maximizing):
    best_score = None
    best_move = (None, None)
    for d in range(1, max_depth+1):
        val, r, c = minimax(board, d, -math.inf, math.inf, maximizing)
        if r is not None and c is not None:
            best_score = val
            best_move = (r, c)
        if best_score is not None and abs(best_score) >= 99999999:
            break
    return best_score, best_move[0], best_move[1]

# ------------ 单步紧急防守/必杀 ------------
def check_immediate_win_or_defense(board):
    # AI单步必杀
    for r in range(ROWS):
        for c in range(COLS):
            if is_valid_move(board, r, c):
                temp_board = make_move(board, r, c, AI)
                if check_winner(temp_board) == AI:
                    return (r, c)
    # 防守对手
    for r in range(ROWS):
        for c in range(COLS):
            if is_valid_move(board, r, c):
                temp_board = make_move(board, r, c, PLAYER)
                if check_winner(temp_board) == PLAYER:
                    return (r, c)
    return None

# ------------------- 显示 & 主循环 -------------------
def print_board(board):
    print("当前棋盘（底→顶）:")
    for r in range(ROWS):
        row_str = []
        for c in range(COLS):
            stack = board[r][c]
            if len(stack) == 0:
                row_str.append("[ ]")
            else:
                row_str.append("[" + "".join(stack) + "]")
        print(" ".join(row_str))
    print()

def main():
    global transposition_table, nodes_explored, search_start_time
    board = create_board()
    current_player = PLAYER  # 玩家 先手
    MAX_DEPTH = 6

    # 标识AI第一手
    ai_first_move_done = True

    while True:
        print_board(board)

        w = check_winner(board)
        if w is not None:
            print(f"游戏结束，胜者：{w}")
            break
        if is_board_full(board):
            print("棋盘已满，平局！")
            break

        if current_player == PLAYER:
            print("请落子，输入行列(如1 1):")
            move_str = input().strip()
            try:
                rs, cs = move_str.split()
                r = int(rs) - 1
                c = int(cs) - 1
                if not is_valid_move(board, r, c):
                    print("非法落子，请重新输入！")
                    continue
                board = make_move(board, r, c, PLAYER)
                current_player = AI
            except:
                print("输入格式有误，请重新输入。")
        else:
            print("开始计算...")

            transposition_table.clear()
            nodes_explored = 0
            search_start_time = time.time()

            if not ai_first_move_done:
                # 第一手浅层搜索
                shallow_depth = 2
                val, r, c = minimax(board, shallow_depth, -math.inf, math.inf, True)
                # 搜索完成后，打一行换行，避免覆盖后续输出
                print()  # <-- 打印换行，把进度条那行结束
                spent = time.time() - search_start_time

                if r is not None and c is not None:
                    board = make_move(board, r, c, AI)
                    print(f"\033[92m【AI第一手】落子: 行 {r+1}, 列 {c+1}, 耗时 {spent:.2f}s, 节点={nodes_explored}\033[0m")
                ai_first_move_done = True
            else:
                # 查单步必杀/防守
                urgent_move = check_immediate_win_or_defense(board)
                if urgent_move is not None:
                    # 打印换行，把进度条那行结束
                    print()
                    r, c = urgent_move
                    board = make_move(board, r, c, AI)
                    print(f"\033[92m【AI紧急策略】落子: 行 {r+1}, 列 {c+1}\033[0m")
                else:
                    # 正常迭代加深
                    best_score, rr, cc = search_best_move(board, MAX_DEPTH, True)
                    print()  # 搜索完也要换行
                    if rr is not None and cc is not None:
                        spent = time.time() - search_start_time
                        print(f"\033[92mAI 落子: 行 {rr+1}, 列 {cc+1}, 耗时 {spent:.2f}s, 节点={nodes_explored}, score={best_score}\033[0m")
                        board = make_move(board, rr, cc, AI)

            current_player = PLAYER

    print_board(board)
    print("游戏结束！")

if __name__ == "__main__":
    main()
