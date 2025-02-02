#include <bits/stdc++.h>

using namespace std;

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

std::string md5(const std::string& input) {
  uint32_t h[4];
  for (int i = 0; i < 4; i++) {
    h[i] = MD5_INIT_STATE[i];
  }

  uint64_t message_len_bits = (uint64_t)input.size() * 8;
  std::vector<uint8_t> msg(input.begin(), input.end());
  msg.push_back(0x80);
  while ((msg.size() % 64) != 56) {
    msg.push_back(0x00);
  }
  uint8_t length_bytes[8];
  to_bytes(message_len_bits, length_bytes);
  for (int i = 0; i < 8; i++) {
    msg.push_back(length_bytes[i]);
  }

  for (size_t offset = 0; offset < msg.size(); offset += 64) {
    uint32_t a = h[0];
    uint32_t b = h[1];
    uint32_t c = h[2];
    uint32_t d = h[3];

    const uint8_t* chunk = &msg[offset];
    uint32_t M[16];
    for (int i = 0; i < 16; i++) {
      M[i] = to_uint32(chunk + i * 4);
    }

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

  std::ostringstream oss;
  for (int i = 0; i < 4; i++) {
    for (int shift = 0; shift < 32; shift += 8) {
      uint8_t byte = (h[i] >> shift) & 0xff;
      oss << std::hex << std::setw(2) << std::setfill('0') << (int)byte;
    }
  }
  return oss.str();
}

static const char PLAYER = 'W';
static const char AI = 'B';

static const int ROWS = 5;
static const int COLS = 5;
static const int MAX_STACK = 5;
static const int MAX_DEPTH = 7;

static long long nodes_explored = 0;
static double search_start_time = 0.0;
static vector<string> spinner = {"|", "/", "-", "\\"};

struct TranspositionEntry {
  int depth;
  double alpha;
  double beta;
  double value;
  pair<int, int> bestMove;
};

static unordered_map<string, TranspositionEntry> transposition_table;

vector<vector<vector<char>>> create_board() {
  return vector<vector<vector<char>>>(ROWS, vector<vector<char>>(COLS, vector<char>()));
}

bool is_valid_move(const vector<vector<vector<char>>>& board, int row, int col) {
  if (row < 0 || row >= ROWS || col < 0 || col >= COLS)
    return false;
  return (int)board[row][col].size() < MAX_STACK;
}

vector<vector<vector<char>>> make_move(const vector<vector<vector<char>>>& board, int row, int col, char piece) {
  vector<vector<vector<char>>> new_board = board;
  new_board[row][col].push_back(piece);
  return new_board;
}

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

string board_to_str(const vector<vector<vector<char>>>& board) {
  ostringstream oss;
  for (int r = 0; r < ROWS; r++) {
    for (int c = 0; c < COLS; c++) {
      // ƴ�Ӹ� stack
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

string hash_board(const vector<vector<vector<char>>>& board) {
  return md5(board_to_str(board));
}

char check_winner(const vector<vector<vector<char>>>& board) {
  vector<array<int, 3>> directions_3d = {
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
            return p;  // 'W' or 'B'
          }
        }
      }
    }
  }
  return '\0';
}


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
  vector<array<int, 3>> directions_3d = {
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
      int stack_height = (int)board[r][c].size();
      for (int l = 0; l < stack_height; l++) {
        for (auto& dir : directions_3d) {
          int dr = dir[0];
          int dc = dir[1];
          int dl = dir[2];
          vector<array<int, 3>> coords;
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

double evaluate(const vector<vector<vector<char>>>& board) {
  char w = check_winner(board);
  if (w == AI) {
    return 99999999.0;
  } else if (w == PLAYER) {
    return -99999999.0;
  }

  long long ai_potential = 0;
  long long player_potential = 0;

  vector<vector<array<int, 3>>> lines = get_all_lines(board);
  for (auto& lineCoords : lines) {
    vector<char> pieces;
    for (auto& coord : lineCoords) {
      int rr = coord[0];
      int cc = coord[1];
      int ll = coord[2];
      if (ll < (int)board[rr][cc].size()) {
        pieces.push_back(board[rr][cc][ll]);
      } else {
        pieces.push_back('\0');
      }
    }
    bool hasB = false, hasW = false;
    for (char p : pieces) {
      if (p == AI)
        hasB = true;
      if (p == PLAYER)
        hasW = true;
    }
    if (hasB && hasW) {
      continue;
    }

    ostringstream pattern_oss;
    for (char p : pieces) {
      if (p == AI) {
        pattern_oss << 'B';
      } else if (p == PLAYER) {
        pattern_oss << 'W';
      } else {
        pattern_oss << '-';
      }
    }
    string pattern = pattern_oss.str();

    if (pattern.find('B') != string::npos && pattern.find('W') == string::npos) {
      if (black_pattern_score.count(pattern)) {
        ai_potential += black_pattern_score[pattern];
      }
    } else if (pattern.find('W') != string::npos && pattern.find('B') == string::npos) {
      if (white_pattern_score.count(pattern)) {
        player_potential += white_pattern_score[pattern];
      }
    }
  }

  return (double)ai_potential - (double)player_potential;
}

//==================== Minimax + Alpha-Beta ====================
double now_in_seconds() {
  using namespace std::chrono;
  auto tp = high_resolution_clock::now();
  auto dur = tp.time_since_epoch();
  return (double)duration_cast<milliseconds>(dur).count() / 1000.0;
}

pair<double, pair<int, int>> minimax(
    const vector<vector<vector<char>>>& board,
    int depth,
    double alpha,
    double beta,
    bool maximizing) {
  nodes_explored += 1;

  static const long long refresh_threshold = 200;
  if (nodes_explored % refresh_threshold == 0) {
    double spent = now_in_seconds() - search_start_time;
    // spinner[(nodes_explored // refresh_threshold) % len(spinner)]
    int idx = (int)((nodes_explored / refresh_threshold) % spinner.size());
    string spin_char = spinner[idx];
    cerr << "\r\033[90m[AI Searching] " << spin_char
         << "  Nodes Explored=" << nodes_explored
         << ", Spending=" << fixed << setprecision(2) << spent << "s\033[0m";
    cerr.flush();
  }

  string bkey = hash_board(board);
  if (transposition_table.count(bkey)) {
    auto& entry = transposition_table[bkey];
    if (entry.depth >= depth && entry.alpha <= alpha && entry.beta >= beta) {
      return {entry.value, entry.bestMove};
    }
  }

  double score = evaluate(board);
  if (fabs(score) >= 99999999.0 || depth == 0 || is_board_full(board)) {
    return {score, {-1, -1}};
  }

  if (maximizing) {
    double best_val = -numeric_limits<double>::infinity();
    pair<int, int> best_move = {-1, -1};

    vector<pair<int, int>> moves;
    for (int r = 0; r < ROWS; r++) {
      for (int c = 0; c < COLS; c++) {
        if (is_valid_move(board, r, c)) {
          moves.push_back({r, c});
        }
      }
    }
    int center_r = ROWS / 2;
    int center_c = COLS / 2;
    sort(moves.begin(), moves.end(), [=](auto& m1, auto& m2) {
      int dist1 = abs(m1.first - center_r) + abs(m1.second - center_c);
      int dist2 = abs(m2.first - center_r) + abs(m2.second - center_c);
      return dist1 < dist2;
    });

    for (auto& mv : moves) {
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
        break;
      }
    }

    TranspositionEntry newEntry = {depth, alpha, beta, best_val, best_move};
    transposition_table[bkey] = newEntry;

    return {best_val, best_move};
  } else {
    double best_val = numeric_limits<double>::infinity();
    pair<int, int> best_move = {-1, -1};

    vector<pair<int, int>> moves;
    for (int r = 0; r < ROWS; r++) {
      for (int c = 0; c < COLS; c++) {
        if (is_valid_move(board, r, c)) {
          moves.push_back({r, c});
        }
      }
    }
    int center_r = ROWS / 2;
    int center_c = COLS / 2;
    sort(moves.begin(), moves.end(), [=](auto& m1, auto& m2) {
      int dist1 = abs(m1.first - center_r) + abs(m1.second - center_c);
      int dist2 = abs(m2.first - center_r) + abs(m2.second - center_c);
      return dist1 < dist2;
    });

    for (auto& mv : moves) {
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
        break;
      }
    }

    TranspositionEntry newEntry = {depth, alpha, beta, best_val, best_move};
    transposition_table[bkey] = newEntry;

    return {best_val, best_move};
  }
}

pair<double, pair<int, int>> search_best_move(const vector<vector<vector<char>>>& board, int max_depth, bool maximizing) {
  double best_score = 0.0;
  pair<int, int> best_move = {-1, -1};
  for (int d = 1; d <= max_depth; d++) {
    auto ret = minimax(board, d, -numeric_limits<double>::infinity(), numeric_limits<double>::infinity(), maximizing);
    double val = ret.first;
    auto mv = ret.second;
    if (mv.first != -1 && mv.second != -1) {
      best_score = val;
      best_move = mv;
    }
    if (fabs(best_score) >= 99999999.0) {
      break;
    }
  }
  return {best_score, best_move};
}

pair<int, int> check_immediate_win_or_defense(const vector<vector<vector<char>>>& board) {
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

void print_board(const vector<vector<vector<char>>>& board) {
  cout << "Current board (bottom to top): " << endl;
  for (int r = 0; r < ROWS; r++) {
    for (int c = 0; c < COLS; c++) {
      if (board[r][c].empty()) {
        cout << "[ ] ";
      } else {
        cout << "[";
        for (auto& ch : board[r][c]) {
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

  auto board = create_board();
  char current_player = AI;

  bool ai_first_move_done = false;

  while (true) {
    print_board(board);
    char w = check_winner(board);
    if (w != '\0') {
      cout << "Game end! Winner is " << w << endl;
      break;
    }
    if (is_board_full(board)) {
      cout << "Board full!" << endl;
      break;
    }

    if (current_player == PLAYER) {
      cout << "Please input position to insert (e.g. 1[ROW] 1[COL]): " << endl;
      string move_str;
      getline(cin, move_str);
      if (move_str.empty()) {
        continue;
      }
      try {
        int rr, cc;
        {
          std::stringstream ss(move_str);
          ss >> rr >> cc;
        }
        rr -= 1;
        cc -= 1;
        if (!is_valid_move(board, rr, cc)) {
          cout << "Invalid input!" << endl;
          continue;
        }
        board = make_move(board, rr, cc, PLAYER);
        current_player = AI;
      } catch (...) {
        cout << "Unexpected error!" << endl;
      }
    } else {
      cout << "AI thinking..." << endl;
      transposition_table.clear();
      nodes_explored = 0;
      search_start_time = now_in_seconds();

      if (!ai_first_move_done) {
        int shallow_depth = 5;
        auto ret = minimax(board, shallow_depth, -numeric_limits<double>::infinity(), numeric_limits<double>::infinity(), true);
        cerr << endl;
        double spent = now_in_seconds() - search_start_time;
        auto mv = ret.second;
        if (mv.first != -1 && mv.second != -1) {
          board = make_move(board, mv.first, mv.second, AI);
          cout << "\033[92mAI first move: Row " << mv.first + 1
               << ", Col " << mv.second + 1
               << ", Spending " << fixed << setprecision(2) << spent << "s, Nodes Explored=" << nodes_explored 
               << "\033[0m" << endl;
        }
        ai_first_move_done = true;
      } else {
        auto urgent_move = check_immediate_win_or_defense(board);
        if (urgent_move.first != -1) {
          cerr << endl;
          board = make_move(board, urgent_move.first, urgent_move.second, AI);
          cout << "\033[92mAI urgent move: Row " << urgent_move.first + 1
               << ", Col " << urgent_move.second + 1
               << "\033[0m" << endl;
        } else {
          auto ret = search_best_move(board, MAX_DEPTH, true);
          double best_score = ret.first;
          auto mv = ret.second;
          cerr << endl;
          if (mv.first != -1 && mv.second != -1) {
            double spent = now_in_seconds() - search_start_time;
            cout << "\033[92mAI deciding: Row " << mv.first + 1
                 << ", Col " << mv.second + 1
                 << ", Spending " << fixed << setprecision(2) << spent << "s"
                 << ", Nodes Explored=" << nodes_explored
                 << ", score=" << best_score << "\033[0m" << endl;
            board = make_move(board, mv.first, mv.second, AI);
          }
        }
      }
      current_player = PLAYER;
    }
  }

  print_board(board);
  cout << "Game end!" << endl;
  return 0;
}
