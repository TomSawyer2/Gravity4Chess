#include <bits/stdc++.h> 
// 如果编译器或环境不支持 <bits/stdc++.h>，
// 请自行包含所需头文件，如:
// #include <iostream>
// #include <vector>
// #include <string>
// #include <unordered_map>
// #include <chrono>
// #include <ctime>
// #include <sstream>
// #include <iomanip>
// #include <algorithm>
// #include <cctype>
// #include <climits>
// #include <cmath>
// #include <limits>
// #include <functional>
// #include <stdlib.h>

using namespace std;

/*
 * 为了与 Python hashlib.md5 一致，这里内置了一个简单的 MD5 实现。
 * 若你的环境可用 OpenSSL 或其他 MD5 库，可以直接调用库函数。
 */

// =============== 开始：MD5 相关 ===============
namespace {
static const uint32_t MD5_INIT_STATE[4] = {
    0x67452301UL, 0xEFCDAB89UL,
    0x98BADCFEUL, 0x10325476UL
};

static const uint32_t MD5_SINE_TABLE[64] = {
    0xd76aa478UL, 0xe8c7b756UL, 0x242070dbUL, 0xc1bdceeeUL,
    0xf57c0fafUL, 0x4787c62aUL, 0xa8304613UL, 0xfd469501UL,
    0x698098d8UL, 0x8b44f7afUL, 0xffff5bb1UL, 0x895cd7beUL,
    0x6b901122UL, 0xfd987193UL, 0xa679438eUL, 0x49b40821UL,
    0xf61e2562UL, 0xc040b340UL, 0x265e5a51UL, 0xe9b6c7aaUL,
    0xd62f105dUL, 0x02441453UL, 0xd8a1e681UL, 0xe7d3fbc8UL,
    0x21e1cde6UL, 0xc33707d6UL, 0xf4d50d87UL, 0x455a14edUL,
    0xa9e3e905UL, 0xfcefa3f8UL, 0x676f02d9UL, 0x8d2a4c8aUL,
    0xfffa3942UL, 0x8771f681UL, 0x6d9d6122UL, 0xfde5380cUL,
    0xa4beea44UL, 0x4bdecfa9UL, 0xf6bb4b60UL, 0xbebfbc70UL,
    0x289b7ec6UL, 0xeaa127faUL, 0xd4ef3085UL, 0x04881d05UL,
    0xd9d4d039UL, 0xe6db99e5UL, 0x1fa27cf8UL, 0xc4ac5665UL,
    0xf4292244UL, 0x432aff97UL, 0xab9423a7UL, 0xfc93a039UL,
    0x655b59c3UL, 0x8f0ccc92UL, 0xffeff47dUL, 0x85845dd1UL,
    0x6fa87e4fUL, 0xfe2ce6e0UL, 0xa3014314UL, 0x4e0811a1UL,
    0xf7537e82UL, 0xbd3af235UL, 0x2ad7d2bbUL, 0xeb86d391UL
};

inline uint32_t F(uint32_t x, uint32_t y, uint32_t z) { return (x & y) | ((~x) & z); }
inline uint32_t G(uint32_t x, uint32_t y, uint32_t z) { return (x & z) | (y & (~z)); }
inline uint32_t H(uint32_t x, uint32_t y, uint32_t z) { return x ^ y ^ z; }
inline uint32_t I(uint32_t x, uint32_t y, uint32_t z) { return y ^ (x | (~z)); }

inline uint32_t rotate_left(uint32_t x, int n) { return (x << n) | (x >> (32 - n)); }

inline void FF(uint32_t &a, uint32_t b, uint32_t c, uint32_t d,
               uint32_t M, uint32_t s, uint32_t t) {
    a = b + rotate_left(a + F(b, c, d) + M + t, s);
}
inline void GG(uint32_t &a, uint32_t b, uint32_t c, uint32_t d,
               uint32_t M, uint32_t s, uint32_t t) {
    a = b + rotate_left(a + G(b, c, d) + M + t, s);
}
inline void HH(uint32_t &a, uint32_t b, uint32_t c, uint32_t d,
               uint32_t M, uint32_t s, uint32_t t) {
    a = b + rotate_left(a + H(b, c, d) + M + t, s);
}
inline void II(uint32_t &a, uint32_t b, uint32_t c, uint32_t d,
               uint32_t M, uint32_t s, uint32_t t) {
    a = b + rotate_left(a + I(b, c, d) + M + t, s);
}

static inline void to_bytes(uint64_t val, uint8_t *bytes) {
    for (int i = 0; i < 8; i++) {
        bytes[i] = (uint8_t) (val >> (8 * i));
    }
}

static inline uint32_t to_uint32(const uint8_t *bytes) {
    return (uint32_t) bytes[0]
         | ((uint32_t) bytes[1] << 8)
         | ((uint32_t) bytes[2] << 16)
         | ((uint32_t) bytes[3] << 24);
}

} // end anonymous namespace

std::string md5(const std::string &input)
{
    // 初始化
    uint32_t h[4];
    for (int i = 0; i < 4; i++) {
        h[i] = MD5_INIT_STATE[i];
    }

    // 预处理
    uint64_t message_len_bits = (uint64_t) input.size() * 8;
    // 复制输入数据
    std::vector<uint8_t> msg(input.begin(), input.end());
    // 添加 0x80
    msg.push_back(0x80);
    // 添加 0x00，直到长度 mod 64 = 56
    while ((msg.size() % 64) != 56) {
        msg.push_back(0x00);
    }
    // 添加原始长度（64bit，低位在前）
    uint8_t length_bytes[8];
    to_bytes(message_len_bits, length_bytes);
    for (int i = 0; i < 8; i++) {
        msg.push_back(length_bytes[i]);
    }

    // 每 64 字节处理
    for (size_t offset = 0; offset < msg.size(); offset += 64) {
        uint32_t a = h[0];
        uint32_t b = h[1];
        uint32_t c = h[2];
        uint32_t d = h[3];

        const uint8_t *chunk = &msg[offset];
        uint32_t M[16];
        for (int i = 0; i < 16; i++) {
            M[i] = to_uint32(chunk + i * 4);
        }

        // 64 轮
        // 第一轮
        FF(a, b, c, d, M[0],  7,  MD5_SINE_TABLE[0]);
        FF(d, a, b, c, M[1],  12, MD5_SINE_TABLE[1]);
        FF(c, d, a, b, M[2],  17, MD5_SINE_TABLE[2]);
        FF(b, c, d, a, M[3],  22, MD5_SINE_TABLE[3]);
        FF(a, b, c, d, M[4],  7,  MD5_SINE_TABLE[4]);
        FF(d, a, b, c, M[5],  12, MD5_SINE_TABLE[5]);
        FF(c, d, a, b, M[6],  17, MD5_SINE_TABLE[6]);
        FF(b, c, d, a, M[7],  22, MD5_SINE_TABLE[7]);
        FF(a, b, c, d, M[8],  7,  MD5_SINE_TABLE[8]);
        FF(d, a, b, c, M[9],  12, MD5_SINE_TABLE[9]);
        FF(c, d, a, b, M[10], 17, MD5_SINE_TABLE[10]);
        FF(b, c, d, a, M[11], 22, MD5_SINE_TABLE[11]);
        FF(a, b, c, d, M[12], 7,  MD5_SINE_TABLE[12]);
        FF(d, a, b, c, M[13], 12, MD5_SINE_TABLE[13]);
        FF(c, d, a, b, M[14], 17, MD5_SINE_TABLE[14]);
        FF(b, c, d, a, M[15], 22, MD5_SINE_TABLE[15]);

        // 第二轮
        GG(a, b, c, d, M[1],  5,  MD5_SINE_TABLE[16]);
        GG(d, a, b, c, M[6],  9,  MD5_SINE_TABLE[17]);
        GG(c, d, a, b, M[11], 14, MD5_SINE_TABLE[18]);
        GG(b, c, d, a, M[0],  20, MD5_SINE_TABLE[19]);
        GG(a, b, c, d, M[5],  5,  MD5_SINE_TABLE[20]);
        GG(d, a, b, c, M[10], 9,  MD5_SINE_TABLE[21]);
        GG(c, d, a, b, M[15], 14, MD5_SINE_TABLE[22]);
        GG(b, c, d, a, M[4],  20, MD5_SINE_TABLE[23]);
        GG(a, b, c, d, M[9],  5,  MD5_SINE_TABLE[24]);
        GG(d, a, b, c, M[14], 9,  MD5_SINE_TABLE[25]);
        GG(c, d, a, b, M[3],  14, MD5_SINE_TABLE[26]);
        GG(b, c, d, a, M[8],  20, MD5_SINE_TABLE[27]);
        GG(a, b, c, d, M[13], 5,  MD5_SINE_TABLE[28]);
        GG(d, a, b, c, M[2],  9,  MD5_SINE_TABLE[29]);
        GG(c, d, a, b, M[7],  14, MD5_SINE_TABLE[30]);
        GG(b, c, d, a, M[12], 20, MD5_SINE_TABLE[31]);

        // 第三轮
        HH(a, b, c, d, M[5],  4,  MD5_SINE_TABLE[32]);
        HH(d, a, b, c, M[8],  11, MD5_SINE_TABLE[33]);
        HH(c, d, a, b, M[11], 16, MD5_SINE_TABLE[34]);
        HH(b, c, d, a, M[14], 23, MD5_SINE_TABLE[35]);
        HH(a, b, c, d, M[1],  4,  MD5_SINE_TABLE[36]);
        HH(d, a, b, c, M[4],  11, MD5_SINE_TABLE[37]);
        HH(c, d, a, b, M[7],  16, MD5_SINE_TABLE[38]);
        HH(b, c, d, a, M[10], 23, MD5_SINE_TABLE[39]);
        HH(a, b, c, d, M[13], 4,  MD5_SINE_TABLE[40]);
        HH(d, a, b, c, M[0],  11, MD5_SINE_TABLE[41]);
        HH(c, d, a, b, M[3],  16, MD5_SINE_TABLE[42]);
        HH(b, c, d, a, M[6],  23, MD5_SINE_TABLE[43]);
        HH(a, b, c, d, M[9],  4,  MD5_SINE_TABLE[44]);
        HH(d, a, b, c, M[12], 11, MD5_SINE_TABLE[45]);
        HH(c, d, a, b, M[15], 16, MD5_SINE_TABLE[46]);
        HH(b, c, d, a, M[2],  23, MD5_SINE_TABLE[47]);

        // 第四轮
        II(a, b, c, d, M[0],  6,  MD5_SINE_TABLE[48]);
        II(d, a, b, c, M[7],  10, MD5_SINE_TABLE[49]);
        II(c, d, a, b, M[14], 15, MD5_SINE_TABLE[50]);
        II(b, c, d, a, M[5],  21, MD5_SINE_TABLE[51]);
        II(a, b, c, d, M[12], 6,  MD5_SINE_TABLE[52]);
        II(d, a, b, c, M[3],  10, MD5_SINE_TABLE[53]);
        II(c, d, a, b, M[10], 15, MD5_SINE_TABLE[54]);
        II(b, c, d, a, M[1],  21, MD5_SINE_TABLE[55]);
        II(a, b, c, d, M[8],  6,  MD5_SINE_TABLE[56]);
        II(d, a, b, c, M[15], 10, MD5_SINE_TABLE[57]);
        II(c, d, a, b, M[6],  15, MD5_SINE_TABLE[58]);
        II(b, c, d, a, M[13], 21, MD5_SINE_TABLE[59]);
        II(a, b, c, d, M[4],  6,  MD5_SINE_TABLE[60]);
        II(d, a, b, c, M[11], 10, MD5_SINE_TABLE[61]);
        II(c, d, a, b, M[2],  15, MD5_SINE_TABLE[62]);
        II(b, c, d, a, M[9],  21, MD5_SINE_TABLE[63]);

        // 更新
        h[0] += a;
        h[1] += b;
        h[2] += c;
        h[3] += d;
    }

    // 输出结果
    std::ostringstream oss;
    for (int i = 0; i < 4; i++) {
        for (int shift = 0; shift < 32; shift += 8) {
            uint8_t byte = (h[i] >> shift) & 0xff;
            oss << std::hex << std::setw(2) << std::setfill('0') << (int)byte;
        }
    }
    return oss.str();
}
// =============== 结束：MD5 相关 ===============

// 全局常量
static const char PLAYER = 'W'; // 玩家(白棋)
static const char AI     = 'B'; // AI(黑棋)

static const int ROWS = 5;
static const int COLS = 5;
static const int MAX_STACK = 5; // 每格最多堆叠 5 层

// 全局变量
static long long nodes_explored = 0;       // 本次搜索已访问的节点数
static double search_start_time = 0.0;     // 本次搜索开始时间 (秒)
static vector<string> spinner = {"|", "/", "-", "\\"}; // 转圈动画符号

// 置换表结构
struct TranspositionEntry {
    int depth;
    double alpha;
    double beta;
    double value;
    pair<int,int> bestMove;
};

// 使用 unordered_map 实现置换表
static unordered_map<string, TranspositionEntry> transposition_table;


//==================== 棋盘相关函数 ====================

// 三维棋盘：board[r][c] 是一个 vector<char>，表示该格堆叠的所有棋子
// 其中最底层为 board[r][c][0], 最顶层为 board[r][c].back()

// 创建空棋盘
vector<vector<vector<char>>> create_board() {
    return vector<vector<vector<char>>>(ROWS, vector<vector<char>>(COLS, vector<char>()));
}

// 检查该位置 (row, col) 是否可放子
bool is_valid_move(const vector<vector<vector<char>>>& board, int row, int col) {
    if (row < 0 || row >= ROWS || col < 0 || col >= COLS) return false;
    return (int)board[row][col].size() < MAX_STACK;
}

// 落子，返回新的棋盘
vector<vector<vector<char>>> make_move(const vector<vector<vector<char>>>& board, int row, int col, char piece) {
    vector<vector<vector<char>>> new_board = board; // 深复制
    new_board[row][col].push_back(piece);
    return new_board;
}

// 检查棋盘是否已经满
bool is_board_full(const vector<vector<vector<char>>>& board) {
    for (int r = 0; r < ROWS; r++) {
        for (int c = 0; c < COLS; c++) {
            if ((int)board[r][c].size() < MAX_STACK) {
                return false;
            }
        }
    }
    return true;
}

// 将棋盘转为可散列字符串
string board_to_str(const vector<vector<vector<char>>>& board) {
    // Python中做法：每一格用  stack_str + '|'  拼接，然后行之间用 '#'，最后整个用 "##" 拼接
    // 这里也尽量保持同样逻辑
    // 形如: (格1内容)|#(格2内容)|#...##(下一行)
    // 为了和 python 完全一致，尽量模仿其格式
    ostringstream oss;
    for (int r = 0; r < ROWS; r++) {
        for (int c = 0; c < COLS; c++) {
            // 拼接该 stack
            for (char ch : board[r][c]) {
                oss << ch;
            }
            oss << "|";
            if (c < COLS - 1) {
                oss << "#";
            }
        }
        if (r < ROWS - 1) {
            oss << "##";
        }
    }
    return oss.str();
}

// 对棋盘做 MD5 并返回十六进制字符串
string hash_board(const vector<vector<vector<char>>>& board) {
    return md5(board_to_str(board));
}

//==================== 获胜检测 ====================
// 如果有任意一方形成 4 连线则返回 'W' or 'B'，否则返回 '\0'
char check_winner(const vector<vector<vector<char>>>& board) {
    // 三维方向集合
    vector<array<int,3>> directions_3d = {
        {0,1,0}, {1,0,0}, {1,1,0}, {1,-1,0},
        {0,0,1}, {1,0,1}, {-1,0,1},
        {0,1,1}, {0,-1,1},
        {1,1,1}, {1,-1,1}, {-1,1,1}, {-1,-1,1},
    };

    for (int r = 0; r < ROWS; r++) {
        for (int c = 0; c < COLS; c++) {
            const auto& stack = board[r][c];
            for (int l = 0; l < (int)stack.size(); l++) {
                char p = stack[l];
                for (auto& dir : directions_3d) {
                    int dr = dir[0];
                    int dc = dir[1];
                    int dl = dir[2];
                    int rr = r, cc = c, ll = l;
                    int chain_count = 1;
                    for (int step = 0; step < 3; step++) {
                        rr += dr;
                        cc += dc;
                        ll += dl;
                        if (rr >= 0 && rr < ROWS && cc >= 0 && cc < COLS) {
                            if (ll >= 0 && ll < (int)board[rr][cc].size()) {
                                if (board[rr][cc][ll] == p) {
                                    chain_count++;
                                } else {
                                    break;
                                }
                            } else {
                                break;
                            }
                        } else {
                            break;
                        }
                    }
                    if (chain_count >= 4) {
                        return p; // 'W' or 'B'
                    }
                }
            }
        }
    }
    return '\0';
}

//==================== 模式匹配打分 ====================

// 对应 Python 中 black_pattern_score
static unordered_map<string, int> black_pattern_score = {
    {"BBBB", 9999999},
    {"BBB-", 5000}, {"BB-B", 5000}, {"B-BB", 5000}, {"-BBB", 5000},
    {"BB--", 200},  {"B-B-", 150},  {"B--B", 100},  {"-BB-", 200},
    {"B---", 10},   {"-B--", 10},   {"--B-", 10},   {"---B", 10},
    {"----", 0}
};

// 对应 Python 中 white_pattern_score
static unordered_map<string, int> white_pattern_score = {
    {"WWWW", 9999999},
    {"WWW-", 5000}, {"WW-W", 5000}, {"W-WW", 5000}, {"-WWW", 5000},
    {"WW--", 200},  {"W-W-", 150},  {"W--W", 100},  {"-WW-", 200},
    {"W---", 10},   {"-W--", 10},   {"--W-", 10},   {"---W", 10},
    {"----", 0}
};

// 获取棋盘上所有 4 连坐标线，每条线包含 4 个 (r,c,l)
vector<vector<array<int,3>>> get_all_lines(const vector<vector<vector<char>>>& board) {
    vector<array<int,3>> directions_3d = {
        {0,1,0}, {1,0,0}, {1,1,0}, {1,-1,0},
        {0,0,1}, {1,0,1}, {-1,0,1},
        {0,1,1}, {0,-1,1},
        {1,1,1}, {1,-1,1}, {-1,1,1}, {-1,-1,1},
    };

    vector<vector<array<int,3>>> lines;
    // 枚举每个可能的起点 + 每个方向
    // 若能取到长度 = 4 的序列，则加入到 lines
    for (int r = 0; r < ROWS; r++) {
        for (int c = 0; c < COLS; c++) {
            int stack_height = (int)board[r][c].size();
            for (int l = 0; l < stack_height; l++) {
                for (auto &dir : directions_3d) {
                    int dr = dir[0];
                    int dc = dir[1];
                    int dl = dir[2];
                    vector<array<int,3>> coords;
                    coords.push_back({r, c, l});
                    int rr = r, cc = c, ll = l;
                    for (int step = 0; step < 3; step++) {
                        rr += dr;
                        cc += dc;
                        ll += dl;
                        if (rr >= 0 && rr < ROWS && cc >= 0 && cc < COLS && ll >= 0 && ll < MAX_STACK) {
                            coords.push_back({rr, cc, ll});
                        } else {
                            break;
                        }
                    }
                    if ((int)coords.size() == 4) {
                        lines.push_back(coords);
                    }
                }
            }
        }
    }
    return lines;
}

// 评估函数
double evaluate(const vector<vector<vector<char>>>& board) {
    // 若出现必胜局面则立即返回极值
    char w = check_winner(board);
    if (w == AI) {
        return 99999999.0;
    } else if (w == PLAYER) {
        return -99999999.0;
    }

    // 否则按照模式匹配打分
    long long ai_potential = 0;
    long long player_potential = 0;

    vector<vector<array<int,3>>> lines = get_all_lines(board);
    for (auto &lineCoords : lines) {
        // 收集这4格的棋子情况
        // 形如: [ 'W', 'W', 'W', (null) ] -> 可能也有空
        vector<char> pieces;
        for (auto &coord : lineCoords) {
            int rr = coord[0];
            int cc = coord[1];
            int ll = coord[2];
            if (ll < (int)board[rr][cc].size()) {
                pieces.push_back(board[rr][cc][ll]);
            } else {
                pieces.push_back('\0');
            }
        }
        // 检查是否含有 'B' 和 'W' 同时出现
        bool hasB = false, hasW = false;
        for (char p : pieces) {
            if (p == AI) hasB = true;
            if (p == PLAYER) hasW = true;
        }
        // 若两者都存在，则对双方都无贡献
        if (hasB && hasW) {
            continue;
        }
        // 构建模式字符串（和 Python 一致：AI->'B', PLAYER->'W', 空->'-'）
        // 注意：Python 代码里 AI='W', PLAYER='B' (示例中)，你可以根据需要调整。
        // 但这里为了维持一致，需要再三确认：上面定义：PLAYER='B', AI='W'？
        // Python 原码写的是 PLAYER='B', AI='W'，那 evaluate 里却写了 pattern 中 'B' if p==AI ...
        // 其实 Python 里是：AI='W' => 'B' if p==AI ... 似乎是一种"反转"。
        // 不过只要与 black_pattern_score, white_pattern_score 对应好即可。
        // 这里严格照着 Python 代码中的写法：
        //   pattern = 'B' if p==AI else ('W' if p==PLAYER else '-')
        
        ostringstream pattern_oss;
        for (char p : pieces) {
            if (p == AI) {
                // AI='W'，但在 pattern 中要写 'B' ?
                // Python 代码确实这么写了： 'B' if p==AI ...
                // 这看上去很拧巴，但为了与原 Python 代码输出一模一样，只能照做
                pattern_oss << 'B';
            }
            else if (p == PLAYER) {
                // PLAYER='B' => 在 pattern 中写 'W'
                pattern_oss << 'W';
            }
            else {
                pattern_oss << '-';
            }
        }
        string pattern = pattern_oss.str();

        // 计算加分
        if (pattern.find('B') != string::npos && pattern.find('W') == string::npos) {
            // 只有B (AI) 
            if (black_pattern_score.count(pattern)) {
                ai_potential += black_pattern_score[pattern];
            }
        } else if (pattern.find('W') != string::npos && pattern.find('B') == string::npos) {
            // 只有W (玩家)
            if (white_pattern_score.count(pattern)) {
                player_potential += white_pattern_score[pattern];
            }
        }
    }

    return (double)ai_potential - (double)player_potential;
}

//==================== Minimax + Alpha-Beta ====================

// 获取当前时间 (秒)
double now_in_seconds() {
    using namespace std::chrono;
    auto tp = high_resolution_clock::now();
    auto dur = tp.time_since_epoch();
    return (double)duration_cast<milliseconds>(dur).count() / 1000.0;
}

pair<double,pair<int,int>> minimax(
    const vector<vector<vector<char>>>& board, 
    int depth, double alpha, double beta, bool maximizing
) {
    // 节点计数
    nodes_explored += 1;

    // 每访问一定阈值节点就刷新“转圈动画”
    static const long long refresh_threshold = 200;
    if (nodes_explored % refresh_threshold == 0) {
        double spent = now_in_seconds() - search_start_time;
        // 计算转圈字符
        // spinner[(nodes_explored // refresh_threshold) % len(spinner)]
        int idx = (int)((nodes_explored / refresh_threshold) % spinner.size());
        string spin_char = spinner[idx];
        // \r回到行首覆盖
        cerr << "\r\033[90m[AI搜索] " << spin_char
             << "  已访问节点=" << nodes_explored
             << ", 耗时=" << fixed << setprecision(2) << spent << "s\033[0m";
        cerr.flush();
    }

    // 置换表查找
    string bkey = hash_board(board);
    if (transposition_table.count(bkey)) {
        auto &entry = transposition_table[bkey];
        // 若满足相同或更深层 && alpha/beta 范围兼容，则直接返回
        if (entry.depth >= depth && entry.alpha <= alpha && entry.beta >= beta) {
            return { entry.value, entry.bestMove };
        }
    }

    // 检查终止条件
    double score = evaluate(board);
    if (fabs(score) >= 99999999.0 || depth == 0 || is_board_full(board)) {
        return { score, { -1, -1 } };
    }

    // alpha-beta 搜索
    if (maximizing) {
        double best_val = -numeric_limits<double>::infinity();
        pair<int,int> best_move = { -1, -1 };

        // 获取所有可行步
        vector<pair<int,int>> moves;
        for (int r = 0; r < ROWS; r++) {
            for (int c = 0; c < COLS; c++) {
                if (is_valid_move(board, r, c)) {
                    moves.push_back({r,c});
                }
            }
        }
        // 简单排序，让靠近中心的步优先
        int center_r = ROWS / 2;
        int center_c = COLS / 2;
        sort(moves.begin(), moves.end(), [=](auto &m1, auto &m2){
            int dist1 = abs(m1.first - center_r) + abs(m1.second - center_c);
            int dist2 = abs(m2.first - center_r) + abs(m2.second - center_c);
            return dist1 < dist2;
        });

        for (auto &mv : moves) {
            int r = mv.first;
            int c = mv.second;
            auto new_board = make_move(board, r, c, AI);
            auto ret = minimax(new_board, depth - 1, alpha, beta, false);
            double val = ret.first;
            if (val > best_val) {
                best_val = val;
                best_move = {r, c};
            }
            alpha = max(alpha, best_val);
            if (alpha >= beta) {
                break; // 剪枝
            }
        }

        // 存入置换表
        TranspositionEntry newEntry = { depth, alpha, beta, best_val, best_move };
        transposition_table[bkey] = newEntry;

        return { best_val, best_move };
    } else {
        double best_val = numeric_limits<double>::infinity();
        pair<int,int> best_move = { -1, -1 };

        // 获取所有可行步
        vector<pair<int,int>> moves;
        for (int r = 0; r < ROWS; r++) {
            for (int c = 0; c < COLS; c++) {
                if (is_valid_move(board, r, c)) {
                    moves.push_back({r,c});
                }
            }
        }
        // 同样排序
        int center_r = ROWS / 2;
        int center_c = COLS / 2;
        sort(moves.begin(), moves.end(), [=](auto &m1, auto &m2){
            int dist1 = abs(m1.first - center_r) + abs(m1.second - center_c);
            int dist2 = abs(m2.first - center_r) + abs(m2.second - center_c);
            return dist1 < dist2;
        });

        for (auto &mv : moves) {
            int r = mv.first;
            int c = mv.second;
            auto new_board = make_move(board, r, c, PLAYER);
            auto ret = minimax(new_board, depth - 1, alpha, beta, true);
            double val = ret.first;
            if (val < best_val) {
                best_val = val;
                best_move = {r, c};
            }
            beta = min(beta, best_val);
            if (alpha >= beta) {
                break; // 剪枝
            }
        }

        // 存入置换表
        TranspositionEntry newEntry = { depth, alpha, beta, best_val, best_move };
        transposition_table[bkey] = newEntry;

        return { best_val, best_move };
    }
}

// 迭代加深搜索
pair<double, pair<int,int>> search_best_move(const vector<vector<vector<char>>>& board, int max_depth, bool maximizing) {
    double best_score = 0.0;
    pair<int,int> best_move = { -1, -1 };
    // 迭代加深
    for (int d = 1; d <= max_depth; d++) {
        auto ret = minimax(board, d, -numeric_limits<double>::infinity(), numeric_limits<double>::infinity(), maximizing);
        double val = ret.first;
        auto mv = ret.second;
        if (mv.first != -1 && mv.second != -1) {
            best_score = val;
            best_move = mv;
        }
        // 若已达胜负极值，停止
        if (fabs(best_score) >= 99999999.0) {
            break;
        }
    }
    return { best_score, best_move };
}

//==================== 单步必杀/防守 ====================
pair<int,int> check_immediate_win_or_defense(const vector<vector<vector<char>>>& board) {
    // AI 单步必杀
    for (int r = 0; r < ROWS; r++) {
        for (int c = 0; c < COLS; c++) {
            if (is_valid_move(board, r, c)) {
                auto temp_board = make_move(board, r, c, AI);
                if (check_winner(temp_board) == AI) {
                    return {r, c};
                }
            }
        }
    }
    // 防守对手
    for (int r = 0; r < ROWS; r++) {
        for (int c = 0; c < COLS; c++) {
            if (is_valid_move(board, r, c)) {
                auto temp_board = make_move(board, r, c, PLAYER);
                if (check_winner(temp_board) == PLAYER) {
                    return {r, c};
                }
            }
        }
    }
    return {-1, -1};
}

//==================== 打印/交互逻辑 ====================
void print_board(const vector<vector<vector<char>>>& board) {
    cout << "当前棋盘（底→顶）:" << endl;
    for (int r = 0; r < ROWS; r++) {
        for (int c = 0; c < COLS; c++) {
            if (board[r][c].empty()) {
                cout << "[ ] ";
            } else {
                cout << "[";
                for (auto &ch : board[r][c]) {
                    cout << ch;
                }
                cout << "] ";
            }
        }
        cout << endl;
    }
    cout << endl;
}

int main() {
    ios::sync_with_stdio(false);
    cin.tie(nullptr);

    // 初始化
    auto board = create_board();
    char current_player = AI;
    int MAX_DEPTH = 7;

    // 用于模拟 Python 里 AI 是否第一手
    bool ai_first_move_done = false; 

    while (true) {
        print_board(board);
        char w = check_winner(board);
        if (w != '\0') {
            cout << "游戏结束，胜者：" << w << endl;
            break;
        }
        if (is_board_full(board)) {
            cout << "棋盘已满，平局！" << endl;
            break;
        }

        if (current_player == PLAYER) {
            cout << "请落子，输入行列(如 1 1):" << endl;
            string move_str;
            getline(cin, move_str);
            if (move_str.empty()) {
                continue;
            }
            try {
                // 解析输入
                int rr, cc;
                {
                    std::stringstream ss(move_str);
                    ss >> rr >> cc;
                }
                rr -= 1; 
                cc -= 1;
                if (!is_valid_move(board, rr, cc)) {
                    cout << "非法落子，请重新输入！" << endl;
                    continue;
                }
                // 玩家落子
                board = make_move(board, rr, cc, PLAYER);
                current_player = AI;
            } catch (...) {
                cout << "输入格式有误，请重新输入。" << endl;
            }
        } else {
            // AI 回合
            cout << "开始计算..." << endl;
            transposition_table.clear();
            nodes_explored = 0;
            search_start_time = now_in_seconds();

            if (!ai_first_move_done) {
                // 第一手浅层搜索
                int shallow_depth = 5;
                auto ret = minimax(board, shallow_depth, -numeric_limits<double>::infinity(), numeric_limits<double>::infinity(), true);
                // 换行
                cerr << endl;
                double spent = now_in_seconds() - search_start_time;
                auto mv = ret.second;
                if (mv.first != -1 && mv.second != -1) {
                    board = make_move(board, mv.first, mv.second, AI);
                    cout << "\033[92m【AI第一手】落子: 行 " << mv.first + 1 
                         << ", 列 " << mv.second + 1 
                         << ", 耗时 " << fixed << setprecision(2) << spent << "s, 节点=" << nodes_explored << "\033[0m" 
                         << endl;
                }
                ai_first_move_done = true;
            } else {
                // 查单步必杀/防守
                auto urgent_move = check_immediate_win_or_defense(board);
                if (urgent_move.first != -1) {
                    // 有紧急策略
                    cerr << endl; // 打印换行
                    board = make_move(board, urgent_move.first, urgent_move.second, AI);
                    cout << "\033[92m【AI紧急策略】落子: 行 " << urgent_move.first + 1
                         << ", 列 " << urgent_move.second + 1 
                         << "\033[0m" << endl;
                } else {
                    // 正常迭代加深
                    auto ret = search_best_move(board, MAX_DEPTH, true);
                    double best_score = ret.first;
                    auto mv = ret.second;
                    cerr << endl; // 结束进度条
                    if (mv.first != -1 && mv.second != -1) {
                        double spent = now_in_seconds() - search_start_time;
                        cout << "\033[92mAI 落子: 行 " << mv.first + 1
                             << ", 列 " << mv.second + 1
                             << ", 耗时 " << fixed << setprecision(2) << spent << "s"
                             << ", 节点=" << nodes_explored
                             << ", score=" << best_score << "\033[0m" << endl;
                        board = make_move(board, mv.first, mv.second, AI);
                    }
                }
            }
            current_player = PLAYER;
        }
    }

    print_board(board);
    cout << "游戏结束！" << endl;
    return 0;
}
