#include <bits/stdc++.h>
#include <future>
#include <mutex>
#include <thread>
using namespace std;

/*
 * ���ü� MD5 ʵ�֡�
 */
namespace {
static const uint32_t MD5_INIT_STATE[4] = {
    0x67452301UL, 0xEFCDAB89UL,
    0x98BADCFEUL, 0x10325476UL};
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
    0xf7537e82UL, 0xbd3af235UL, 0x2ad7d2bbUL, 0xeb86d391UL};

inline uint32_t F(uint32_t x, uint32_t y, uint32_t z) {
  return (x & y) | ((~x) & z);
}
inline uint32_t G(uint32_t x, uint32_t y, uint32_t z) {
  return (x & z) | (y & (~z));
}
inline uint32_t H(uint32_t x, uint32_t y, uint32_t z) {
  return x ^ y ^ z;
}
inline uint32_t I(uint32_t x, uint32_t y, uint32_t z) {
  return y ^ (x | (~z));
}
inline uint32_t rotate_left(uint32_t x, int n) {
  return (x << n) | (x >> (32 - n));
}

inline void FF(uint32_t& a, uint32_t b, uint32_t c, uint32_t d, uint32_t M, uint32_t s, uint32_t t) {
  a = b + rotate_left(a + F(b, c, d) + M + t, s);
}
inline void GG(uint32_t& a, uint32_t b, uint32_t c, uint32_t d, uint32_t M, uint32_t s, uint32_t t) {
  a = b + rotate_left(a + G(b, c, d) + M + t, s);
}
inline void HH(uint32_t& a, uint32_t b, uint32_t c, uint32_t d, uint32_t M, uint32_t s, uint32_t t) {
  a = b + rotate_left(a + H(b, c, d) + M + t, s);
}
inline void II(uint32_t& a, uint32_t b, uint32_t c, uint32_t d, uint32_t M, uint32_t s, uint32_t t) {
  a = b + rotate_left(a + I(b, c, d) + M + t, s);
}

static inline void to_bytes(uint64_t val, uint8_t* bytes) {
  for (int i = 0; i < 8; i++) {
    bytes[i] = (uint8_t)(val >> (8 * i));
  }
}
static inline uint32_t to_uint32(const uint8_t* bytes) {
  return (uint32_t)bytes[0] | ((uint32_t)bytes[1] << 8) | ((uint32_t)bytes[2] << 16) | ((uint32_t)bytes[3] << 24);
}
}  // end anonymous namespace

string md5(const string& input) {
  uint32_t h[4];
  for (int i = 0; i < 4; i++) {
    h[i] = MD5_INIT_STATE[i];
  }
  uint64_t msg_len_bits = (uint64_t)input.size() * 8;
  vector<uint8_t> msg(input.begin(), input.end());
  msg.push_back(0x80);
  while ((msg.size() % 64) != 56) {
    msg.push_back(0x00);
  }
  uint8_t length_bytes[8];
  to_bytes(msg_len_bits, length_bytes);
  for (int i = 0; i < 8; i++) {
    msg.push_back(length_bytes[i]);
  }
  for (size_t offset = 0; offset < msg.size(); offset += 64) {
    uint32_t a = h[0], b = h[1], c = h[2], d = h[3];
    const uint8_t* chunk = &msg[offset];
    uint32_t M[16];
    for (int i = 0; i < 16; i++) {
      M[i] = to_uint32(chunk + i * 4);
    }
    // 64 ��
    FF(a, b, c, d, M[0], 7, MD5_SINE_TABLE[0]);
    FF(d, a, b, c, M[1], 12, MD5_SINE_TABLE[1]);
    FF(c, d, a, b, M[2], 17, MD5_SINE_TABLE[2]);
    FF(b, c, d, a, M[3], 22, MD5_SINE_TABLE[3]);
    FF(a, b, c, d, M[4], 7, MD5_SINE_TABLE[4]);
    FF(d, a, b, c, M[5], 12, MD5_SINE_TABLE[5]);
    FF(c, d, a, b, M[6], 17, MD5_SINE_TABLE[6]);
    FF(b, c, d, a, M[7], 22, MD5_SINE_TABLE[7]);
    FF(a, b, c, d, M[8], 7, MD5_SINE_TABLE[8]);
    FF(d, a, b, c, M[9], 12, MD5_SINE_TABLE[9]);
    FF(c, d, a, b, M[10], 17, MD5_SINE_TABLE[10]);
    FF(b, c, d, a, M[11], 22, MD5_SINE_TABLE[11]);
    FF(a, b, c, d, M[12], 7, MD5_SINE_TABLE[12]);
    FF(d, a, b, c, M[13], 12, MD5_SINE_TABLE[13]);
    FF(c, d, a, b, M[14], 17, MD5_SINE_TABLE[14]);
    FF(b, c, d, a, M[15], 22, MD5_SINE_TABLE[15]);
    GG(a, b, c, d, M[1], 5, MD5_SINE_TABLE[16]);
    GG(d, a, b, c, M[6], 9, MD5_SINE_TABLE[17]);
    GG(c, d, a, b, M[11], 14, MD5_SINE_TABLE[18]);
    GG(b, c, d, a, M[0], 20, MD5_SINE_TABLE[19]);
    GG(a, b, c, d, M[5], 5, MD5_SINE_TABLE[20]);
    GG(d, a, b, c, M[10], 9, MD5_SINE_TABLE[21]);
    GG(c, d, a, b, M[15], 14, MD5_SINE_TABLE[22]);
    GG(b, c, d, a, M[4], 20, MD5_SINE_TABLE[23]);
    GG(a, b, c, d, M[9], 5, MD5_SINE_TABLE[24]);
    GG(d, a, b, c, M[14], 9, MD5_SINE_TABLE[25]);
    GG(c, d, a, b, M[3], 14, MD5_SINE_TABLE[26]);
    GG(b, c, d, a, M[8], 20, MD5_SINE_TABLE[27]);
    GG(a, b, c, d, M[13], 5, MD5_SINE_TABLE[28]);
    GG(d, a, b, c, M[2], 9, MD5_SINE_TABLE[29]);
    GG(c, d, a, b, M[7], 14, MD5_SINE_TABLE[30]);
    GG(b, c, d, a, M[12], 20, MD5_SINE_TABLE[31]);
    HH(a, b, c, d, M[5], 4, MD5_SINE_TABLE[32]);
    HH(d, a, b, c, M[8], 11, MD5_SINE_TABLE[33]);
    HH(c, d, a, b, M[11], 16, MD5_SINE_TABLE[34]);
    HH(b, c, d, a, M[14], 23, MD5_SINE_TABLE[35]);
    HH(a, b, c, d, M[1], 4, MD5_SINE_TABLE[36]);
    HH(d, a, b, c, M[4], 11, MD5_SINE_TABLE[37]);
    HH(c, d, a, b, M[7], 16, MD5_SINE_TABLE[38]);
    HH(b, c, d, a, M[10], 23, MD5_SINE_TABLE[39]);
    HH(a, b, c, d, M[13], 4, MD5_SINE_TABLE[40]);
    HH(d, a, b, c, M[0], 11, MD5_SINE_TABLE[41]);
    HH(c, d, a, b, M[3], 16, MD5_SINE_TABLE[42]);
    HH(b, c, d, a, M[6], 23, MD5_SINE_TABLE[43]);
    HH(a, b, c, d, M[9], 4, MD5_SINE_TABLE[44]);
    HH(d, a, b, c, M[12], 11, MD5_SINE_TABLE[45]);
    HH(c, d, a, b, M[15], 16, MD5_SINE_TABLE[46]);
    HH(b, c, d, a, M[2], 23, MD5_SINE_TABLE[47]);
    II(a, b, c, d, M[0], 6, MD5_SINE_TABLE[48]);
    II(d, a, b, c, M[7], 10, MD5_SINE_TABLE[49]);
    II(c, d, a, b, M[14], 15, MD5_SINE_TABLE[50]);
    II(b, c, d, a, M[5], 21, MD5_SINE_TABLE[51]);
    II(a, b, c, d, M[12], 6, MD5_SINE_TABLE[52]);
    II(d, a, b, c, M[3], 10, MD5_SINE_TABLE[53]);
    II(c, d, a, b, M[10], 15, MD5_SINE_TABLE[54]);
    II(b, c, d, a, M[1], 21, MD5_SINE_TABLE[55]);
    II(a, b, c, d, M[8], 6, MD5_SINE_TABLE[56]);
    II(d, a, b, c, M[15], 10, MD5_SINE_TABLE[57]);
    II(c, d, a, b, M[6], 15, MD5_SINE_TABLE[58]);
    II(b, c, d, a, M[13], 21, MD5_SINE_TABLE[59]);
    II(a, b, c, d, M[4], 6, MD5_SINE_TABLE[60]);
    II(d, a, b, c, M[11], 10, MD5_SINE_TABLE[61]);
    II(c, d, a, b, M[2], 15, MD5_SINE_TABLE[62]);
    II(b, c, d, a, M[9], 21, MD5_SINE_TABLE[63]);
    h[0] += a;
    h[1] += b;
    h[2] += c;
    h[3] += d;
  }
  ostringstream oss;
  for (int i = 0; i < 4; i++) {
    for (int shift = 0; shift < 32; shift += 8) {
      uint8_t byte = (h[i] >> shift) & 0xff;
      oss << hex << setw(2) << setfill('0') << (int)byte;
    }
  }
  return oss.str();
}

//================================================================
static const char PLAYER = 'W';  // ���(����)
static const char AI = 'B';      // AI(����)

static const int ROWS = 5;
static const int COLS = 5;
static const int MAX_STACK = 5;  // ÿ�����ѵ� 5 ��
static const int MAX_DEPTH = 7;

// �û����ṹ
struct TranspositionEntry {
  int depth;
  double alpha;
  double beta;
  double value;
  pair<int, int> bestMove;
};

// ȫ���û��� + ������
static unordered_map<string, TranspositionEntry> transposition_table;
static std::mutex transposition_table_mutex;

// ʱ�丨��
static inline double now_in_seconds() {
  using namespace std::chrono;
  auto tp = high_resolution_clock::now();
  auto dur = tp.time_since_epoch();
  return double(duration_cast<milliseconds>(dur).count()) / 1000.0;
}

//================================================================
// ���̲���
vector<vector<vector<char>>> create_board() {
  return vector<vector<vector<char>>>(ROWS, vector<vector<char>>(COLS));
}
bool is_valid_move(const vector<vector<vector<char>>>& board, int r, int c) {
  if (r < 0 || r >= ROWS || c < 0 || c >= COLS)
    return false;
  return (int)board[r][c].size() < MAX_STACK;
}
vector<vector<vector<char>>> make_move(const vector<vector<vector<char>>>& board, int r, int c, char piece) {
  auto new_board = board;
  new_board[r][c].push_back(piece);
  return new_board;
}
bool is_board_full(const vector<vector<vector<char>>>& board) {
  for (int rr = 0; rr < ROWS; rr++) {
    for (int cc = 0; cc < COLS; cc++) {
      if ((int)board[rr][cc].size() < MAX_STACK)
        return false;
    }
  }
  return true;
}

//================================================================
// ����ת�ַ���
string board_to_str(const vector<vector<vector<char>>>& board) {
  ostringstream oss;
  for (int r = 0; r < ROWS; r++) {
    for (int c = 0; c < COLS; c++) {
      for (char p : board[r][c]) {
        oss << p;
      }
      oss << "|";
      if (c < COLS - 1)
        oss << "#";
    }
    if (r < ROWS - 1)
      oss << "##";
  }
  return oss.str();
}
string hash_board(const vector<vector<vector<char>>>& board) {
  return md5(board_to_str(board));
}

//================================================================
// ʤ�����
char check_winner(const vector<vector<vector<char>>>& board) {
  static vector<array<int, 3>> directions_3d = {
      {0, 1, 0},
      {1, 0, 0},
      {1, 1, 0},
      {1, -1, 0},
      {0, 0, 1},
      {1, 0, 1},
      {-1, 0, 1},
      {0, 1, 1},
      {0, -1, 1},
      {1, 1, 1},
      {1, -1, 1},
      {-1, 1, 1},
      {-1, -1, 1},
  };
  for (int r = 0; r < ROWS; r++) {
    for (int c = 0; c < COLS; c++) {
      const auto& stk = board[r][c];
      for (int l = 0; l < (int)stk.size(); l++) {
        char p = stk[l];
        for (auto& dir : directions_3d) {
          int rr = r, cc = c, ll = l;
          int chain = 1;
          for (int step = 0; step < 3; step++) {
            rr += dir[0];
            cc += dir[1];
            ll += dir[2];
            if (rr >= 0 && rr < ROWS && cc >= 0 && cc < COLS) {
              if (ll >= 0 && ll < (int)board[rr][cc].size()) {
                if (board[rr][cc][ll] == p)
                  chain++;
                else
                  break;
              } else
                break;
            } else
              break;
          }
          if (chain >= 4)
            return p;
        }
      }
    }
  }
  return '\0';
}

//================================================================
// ģʽƥ����
static unordered_map<string, int> black_pattern_score = {
    {"BBBB", 9999999},
    {"BBB-", 5000},
    {"BB-B", 5000},
    {"B-BB", 5000},
    {"-BBB", 5000},
    {"BB--", 200},
    {"B-B-", 150},
    {"B--B", 100},
    {"-BB-", 200},
    {"B---", 10},
    {"-B--", 10},
    {"--B-", 10},
    {"---B", 10},
    {"----", 0}};
static unordered_map<string, int> white_pattern_score = {
    {"WWWW", 9999999},
    {"WWW-", 5000},
    {"WW-W", 5000},
    {"W-WW", 5000},
    {"-WWW", 5000},
    {"WW--", 200},
    {"W-W-", 150},
    {"W--W", 100},
    {"-WW-", 200},
    {"W---", 10},
    {"-W--", 10},
    {"--W-", 10},
    {"---W", 10},
    {"----", 0}};

vector<vector<array<int, 3>>> get_all_lines(const vector<vector<vector<char>>>& board) {
  static vector<array<int, 3>> directions_3d = {
      {0, 1, 0},
      {1, 0, 0},
      {1, 1, 0},
      {1, -1, 0},
      {0, 0, 1},
      {1, 0, 1},
      {-1, 0, 1},
      {0, 1, 1},
      {0, -1, 1},
      {1, 1, 1},
      {1, -1, 1},
      {-1, 1, 1},
      {-1, -1, 1},
  };
  vector<vector<array<int, 3>>> lines;
  for (int r = 0; r < ROWS; r++) {
    for (int c = 0; c < COLS; c++) {
      int stkh = (int)board[r][c].size();
      for (int l = 0; l < stkh; l++) {
        for (auto& dir : directions_3d) {
          vector<array<int, 3>> coords;
          coords.push_back({r, c, l});
          int rr = r, cc = c, ll = l;
          for (int step = 0; step < 3; step++) {
            rr += dir[0];
            cc += dir[1];
            ll += dir[2];
            if (rr >= 0 && rr < ROWS && cc >= 0 && cc < COLS && ll >= 0 && ll < MAX_STACK) {
              coords.push_back({rr, cc, ll});
            } else
              break;
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

double evaluate(const vector<vector<vector<char>>>& board) {
  char w = check_winner(board);
  if (w == AI)
    return 99999999.0;
  if (w == PLAYER)
    return -99999999.0;

  long long ai_potential = 0, player_potential = 0;
  auto lines = get_all_lines(board);
  for (auto& ln : lines) {
    vector<char> pieces;
    for (auto& xyz : ln) {
      int rr = xyz[0], cc = xyz[1], ll = xyz[2];
      if (ll < (int)board[rr][cc].size())
        pieces.push_back(board[rr][cc][ll]);
      else
        pieces.push_back('\0');
    }
    bool hasB = false, hasW = false;
    for (char p : pieces) {
      if (p == AI)
        hasB = true;
      if (p == PLAYER)
        hasW = true;
    }
    // ��ͬʱ�� B / W��������
    if (hasB && hasW)
      continue;

    // ����ģʽ��
    ostringstream oss;
    for (char p : pieces) {
      if (p == AI)
        oss << 'B';
      else if (p == PLAYER)
        oss << 'W';
      else
        oss << '-';
    }
    string pattern = oss.str();
    if (pattern.find('B') != string::npos && pattern.find('W') == string::npos) {
      // ֻ�� B
      if (black_pattern_score.count(pattern)) {
        ai_potential += black_pattern_score[pattern];
      }
    } else if (pattern.find('W') != string::npos && pattern.find('B') == string::npos) {
      // ֻ�� W
      if (white_pattern_score.count(pattern)) {
        player_potential += white_pattern_score[pattern];
      }
    }
  }
  return (double)ai_potential - (double)player_potential;
}

//================================================================
// �û�������
bool tt_lookup(const string& key, int depth, double alpha, double beta, double& val, pair<int, int>& bestMove) {
  lock_guard<mutex> guard(transposition_table_mutex);
  auto it = transposition_table.find(key);
  if (it == transposition_table.end())
    return false;
  auto& entry = it->second;
  if (entry.depth >= depth && entry.alpha <= alpha && entry.beta >= beta) {
    val = entry.value;
    bestMove = entry.bestMove;
    return true;
  }
  return false;
}
void tt_store(const string& key, int depth, double alpha, double beta, double val, pair<int, int> bestMove) {
  lock_guard<mutex> guard(transposition_table_mutex);
  TranspositionEntry e{depth, alpha, beta, val, bestMove};
  transposition_table[key] = e;
}

//================================================================
// ���̰߳汾 minimax
pair<double, pair<int, int>> minimax_single(
    const vector<vector<vector<char>>>& board,
    int depth,
    double alpha,
    double beta,
    bool maximizing) {
  string bkey = hash_board(board);
  double val;
  pair<int, int> mv(-1, -1);
  if (tt_lookup(bkey, depth, alpha, beta, val, mv)) {
    return {val, mv};
  }

  double sc = evaluate(board);
  if (fabs(sc) >= 99999999.0 || depth == 0 || is_board_full(board)) {
    return {sc, {-1, -1}};
  }

  if (maximizing) {
    double best_val = -numeric_limits<double>::infinity();
    pair<int, int> best_move(-1, -1);
    // ö�����п�����
    vector<pair<int, int>> moves;
    for (int r = 0; r < ROWS; r++) {
      for (int c = 0; c < COLS; c++) {
        if (is_valid_move(board, r, c))
          moves.push_back({r, c});
      }
    }
    // ����(��������)
    int cr = ROWS / 2, cc = COLS / 2;
    sort(moves.begin(), moves.end(), [&](auto& m1, auto& m2) {
      int d1 = abs(m1.first - cr) + abs(m1.second - cc);
      int d2 = abs(m2.first - cr) + abs(m2.second - cc);
      return d1 < d2;
    });
    for (auto& m : moves) {
      auto nb = make_move(board, m.first, m.second, AI);
      auto ret = minimax_single(nb, depth - 1, alpha, beta, false);
      if (ret.first > best_val) {
        best_val = ret.first;
        best_move = m;
      }
      alpha = max(alpha, best_val);
      if (alpha >= beta)
        break;
    }
    tt_store(bkey, depth, alpha, beta, best_val, best_move);
    return {best_val, best_move};
  } else {
    double best_val = numeric_limits<double>::infinity();
    pair<int, int> best_move(-1, -1);
    // ö��
    vector<pair<int, int>> moves;
    for (int r = 0; r < ROWS; r++) {
      for (int c = 0; c < COLS; c++) {
        if (is_valid_move(board, r, c))
          moves.push_back({r, c});
      }
    }
    int cr = ROWS / 2, cc = COLS / 2;
    sort(moves.begin(), moves.end(), [&](auto& m1, auto& m2) {
      int d1 = abs(m1.first - cr) + abs(m1.second - cc);
      int d2 = abs(m2.first - cr) + abs(m2.second - cc);
      return d1 < d2;
    });
    for (auto& m : moves) {
      auto nb = make_move(board, m.first, m.second, PLAYER);
      auto ret = minimax_single(nb, depth - 1, alpha, beta, true);
      if (ret.first < best_val) {
        best_val = ret.first;
        best_move = m;
      }
      beta = min(beta, best_val);
      if (alpha >= beta)
        break;
    }
    tt_store(bkey, depth, alpha, beta, best_val, best_move);
    return {best_val, best_move};
  }
}

//================================================================
// �������� (���ڸ��ڵ�)
pair<double, pair<int, int>> minimax_root_parallel(
    const vector<vector<vector<char>>>& board,
    int depth,
    bool maximizing) {
  // �ռ����п����߷�
  vector<pair<int, int>> moves;
  for (int r = 0; r < ROWS; r++) {
    for (int c = 0; c < COLS; c++) {
      if (is_valid_move(board, r, c))
        moves.push_back({r, c});
    }
  }
  if (moves.empty()) {
    // �޿���ʱ��ֱ������
    double sc = evaluate(board);
    return {sc, {-1, -1}};
  }

  int cr = ROWS / 2, cc = COLS / 2;
  sort(moves.begin(), moves.end(), [&](auto& m1, auto& m2) {
    int d1 = abs(m1.first - cr) + abs(m1.second - cc);
    int d2 = abs(m2.first - cr) + abs(m2.second - cc);
    return d1 < d2;
  });

  // ���У���ÿ���߷�����һ���߳̽��е��߳� minimax
  vector<future<pair<double, pair<int, int>>>> futs;
  futs.reserve(moves.size());
  for (auto& mv : moves) {
    futs.push_back(async(std::launch::async, [&, mv]() {
      auto nb = make_move(board, mv.first, mv.second, (maximizing ? AI : PLAYER));
      auto ret = minimax_single(nb, depth - 1,
                                -numeric_limits<double>::infinity(),
                                numeric_limits<double>::infinity(),
                                !maximizing);
      // ���� {value, rootMove}
      return make_pair(ret.first, mv);
    }));
  }

  // �鲢���
  double best_val = maximizing ? -numeric_limits<double>::infinity()
                               : numeric_limits<double>::infinity();
  pair<int, int> best_move(-1, -1);
  for (auto& f : futs) {
    auto res = f.get();
    double val = res.first;
    auto mv = res.second;
    if (maximizing) {
      if (val > best_val) {
        best_val = val;
        best_move = mv;
      }
    } else {
      if (val < best_val) {
        best_val = val;
        best_move = mv;
      }
    }
  }
  return {best_val, best_move};
}

//================================================================
// ���յĵ�������������ֻ�������ȲŲ���
pair<double, pair<int, int>> search_best_move(
    const vector<vector<vector<char>>>& board,
    int max_depth,
    bool maximizing) {
  pair<double, pair<int, int>> best_result = {0.0, {-1, -1}};
  for (int d = 1; d <= max_depth; d++) {
    if (d < max_depth) {
      // ���������ȣ����߳� root ����
      auto ret = minimax_single(board, d,
                                -numeric_limits<double>::infinity(),
                                numeric_limits<double>::infinity(),
                                maximizing);
      if (ret.second.first != -1) {
        best_result = ret;
      }
      if (fabs(best_result.first) >= 99999999.0) {
        break;
      }
    } else {
      // d == max_depth ʱ�����в�������
      auto ret = minimax_root_parallel(board, d, maximizing);
      if (ret.second.first != -1) {
        best_result = ret;
      }
      break;  // �����Ⱥ�Ͳ��ټ���
    }
  }
  return best_result;
}

//================================================================
// ������ɱ/����
pair<int, int> check_immediate_win_or_defense(const vector<vector<vector<char>>>& board) {
  // AI ��ɱ
  for (int r = 0; r < ROWS; r++) {
    for (int c = 0; c < COLS; c++) {
      if (is_valid_move(board, r, c)) {
        auto nb = make_move(board, r, c, AI);
        if (check_winner(nb) == AI) {
          return {r, c};
        }
      }
    }
  }
  // ���ض���
  for (int r = 0; r < ROWS; r++) {
    for (int c = 0; c < COLS; c++) {
      if (is_valid_move(board, r, c)) {
        auto nb = make_move(board, r, c, PLAYER);
        if (check_winner(nb) == PLAYER) {
          return {r, c};
        }
      }
    }
  }
  return {-1, -1};
}

//================================================================
// չʾ����ѭ��
void print_board(const vector<vector<vector<char>>>& board) {
  cout << "��ǰ���̣��ס�����:" << endl;
  for (int r = 0; r < ROWS; r++) {
    for (int c = 0; c < COLS; c++) {
      if (board[r][c].empty()) {
        cout << "[ ] ";
      } else {
        cout << "[";
        for (char ch : board[r][c])
          cout << ch;
        cout << "] ";
      }
    }
    cout << endl;
  }
  cout << endl;
}

int main() {
  // �������м��� C++ I/O������Ҫ�ֶ� flush/endl
  ios::sync_with_stdio(false);
  cin.tie(nullptr);

  auto board = create_board();
  board[1] = {{}, {'B'}, {'W'}, {'W'}, {}};
  board[2] = {{}, {'W', 'W'}, {'B', 'B', 'W'}, {'B', 'B'}, {}};
  board[3] = {{}, {}, {'B'}, {'W'}, {}};

  char current_player = AI;

  bool ai_first_move_done = true;

  while (true) {
    // ��ʾ����
    print_board(board);

    char w = check_winner(board);
    if (w != '\0') {
      cout << "��Ϸ������ʤ�ߣ�" << w << endl;
      break;
    }
    if (is_board_full(board)) {
      cout << "����������ƽ�֣�" << endl;
      break;
    }

    if (current_player == PLAYER) {
      cout << "�����ӣ���������(�� 1 1):" << endl;
      // endl���Զ� flush��ȷ����ʾ�ɼ�
      string line;
      if (!std::getline(cin, line)) {
        // ����ȡʧ��(EOF��)������������
        break;
      }
      if (line.empty()) {
        continue;
      }
      int rr, cc;
      try {
        {
          stringstream ss(line);
          ss >> rr >> cc;
        }
        rr -= 1;
        cc -= 1;
        if (!is_valid_move(board, rr, cc)) {
          cout << "�Ƿ����ӣ����������룡" << endl;
          continue;
        }
        board = make_move(board, rr, cc, PLAYER);
        current_player = AI;
      } catch (...) {
        cout << "�����ʽ�������������롣" << endl;
      }
    } else {
      cout << "��ʼ����..." << endl;
      // ͬ�� endl���������

      // ����û���
      {
        lock_guard<mutex> guard(transposition_table_mutex);
        transposition_table.clear();
      }
      double start_time = now_in_seconds();

      if (!ai_first_move_done) {
        // AI��һ�֣�����ǳ������
        int shallow_depth = 7;
        auto ret = minimax_root_parallel(board, shallow_depth, true);
        double spent = now_in_seconds() - start_time;
        auto mv = ret.second;
        if (mv.first != -1) {
          board = make_move(board, mv.first, mv.second, AI);
          cout << "��AI��һ�֡�����: �� " << (mv.first + 1)
               << ", �� " << (mv.second + 1)
               << ", ��ʱ " << fixed << setprecision(2) << spent << "s" << endl;
        }
        ai_first_move_done = true;
      } else {
        // �鵥����ɱ�����
        auto urgent = check_immediate_win_or_defense(board);
        if (urgent.first != -1) {
          // ��������
          board = make_move(board, urgent.first, urgent.second, AI);
          double spent = now_in_seconds() - start_time;
          cout << "��AI�������ԡ�����: �� " << (urgent.first + 1)
               << ", �� " << (urgent.second + 1)
               << ", ��ʱ " << fixed << setprecision(2) << spent << "s" << endl;
        } else {
          // ������������ (�����Ȳ���)
          auto best_ret = search_best_move(board, MAX_DEPTH, true);
          double spent = now_in_seconds() - start_time;
          double best_score = best_ret.first;
          auto mv = best_ret.second;
          if (mv.first != -1) {
            cout << "AI ����: �� " << (mv.first + 1)
                 << ", �� " << (mv.second + 1)
                 << ", ��ʱ " << fixed << setprecision(2) << spent << "s"
                 << ", score=" << best_score << endl;
            board = make_move(board, mv.first, mv.second, AI);
          }
        }
      }
      current_player = PLAYER;
    }
  }

  print_board(board);
  cout << "��Ϸ������" << endl;
  return 0;
}
