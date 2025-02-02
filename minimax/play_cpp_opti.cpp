#include <bits/stdc++.h>
using namespace std;

/*
========================================
  1. Board Size & Constants
========================================
*/
static const int ROWS = 5;
static const int COLS = 5;
static const int MAX_STACK = 5;  // max 5 layers per position
static const int MAX_DEPTH = 9;

static const char PLAYER = 'W';  // Player
static const char AI = 'B';      // AI

// Position index: index = r*25 + c*5 + l
inline int indexOf(int r, int c, int l) {
  return r * 25 + c * 5 + l;
}

/*
========================================
  2. BitBoard Representation (2 uint64_t)
========================================
*/
struct BitBoard {
  uint64_t lo;  // lower 64 bits
  uint64_t hi;  // higher 64 bits

  BitBoard()
      : lo(0ULL), hi(0ULL) {}
  BitBoard(uint64_t l, uint64_t h)
      : lo(l), hi(h) {}

  // set pos bit=1
  inline void setBit(int pos) {
    if (pos < 64) {
      lo |= (1ULL << pos);
    } else {
      pos -= 64;
      hi |= (1ULL << pos);
    }
  }
  // reset pos bit=0
  inline void resetBit(int pos) {
    if (pos < 64) {
      lo &= ~(1ULL << pos);
    } else {
      pos -= 64;
      hi &= ~(1ULL << pos);
    }
  }
  // test pos bit
  inline bool testBit(int pos) const {
    if (pos < 64) {
      return ((lo >> pos) & 1ULL) != 0ULL;
    } else {
      pos -= 64;
      return ((hi >> pos) & 1ULL) != 0ULL;
    }
  }
  inline BitBoard operator&(const BitBoard& rhs) const {
    return BitBoard(lo & rhs.lo, hi & rhs.hi);
  }
  inline BitBoard operator|(const BitBoard& rhs) const {
    return BitBoard(lo | rhs.lo, hi | rhs.hi);
  }
  inline BitBoard operator^(const BitBoard& rhs) const {
    return BitBoard(lo ^ rhs.lo, hi ^ rhs.hi);
  }
  inline void operator^=(const BitBoard& rhs) {
    lo ^= rhs.lo;
    hi ^= rhs.hi;
  }
  inline bool empty() const {
    return (lo == 0ULL && hi == 0ULL);
  }
};

/*
========================================
  3. Board Structure
     - blackBB, whiteBB
     - topIndex[r][c]
     - boardHash (Zobrist)
========================================
*/
struct Board {
  BitBoard blackBB;
  BitBoard whiteBB;
  int topIndex[ROWS][COLS];  // stack height
  uint64_t boardHash;        // zobrist hash

  Board() {
    memset(topIndex, 0, sizeof(topIndex));
    boardHash = 0ULL;
  }
};

/*
========================================
  4. Zobrist Hash
     - each index in [0..124], color=0(B), 1(W)
========================================
*/
static uint64_t ZOBRIST[125][2];

static void initZobrist() {
  static mt19937_64 rng(random_device{}());
  for (int i = 0; i < 125; i++) {
    for (int c = 0; c < 2; c++) {
      uint64_t val = ((uint64_t)rng() << 32) ^ rng();
      ZOBRIST[i][c] = val;
    }
  }
}

// calculate Zobrist hash for the Board (using XOR)
uint64_t calcZobrist(const Board& bd) {
  uint64_t h = 0ULL;
  // black
  for (int pos = 0; pos < 125; pos++) {
    if (bd.blackBB.testBit(pos)) {
      h ^= ZOBRIST[pos][0];  // color=0(B)
    }
  }
  // white
  for (int pos = 0; pos < 125; pos++) {
    if (bd.whiteBB.testBit(pos)) {
      h ^= ZOBRIST[pos][1];  // color=1(W)
    }
  }
  return h;
}

/*
========================================
  5. Functions: isValid, makeMove, isFull
========================================
*/
bool isValidMove(const Board& bd, int r, int c) {
  if (r < 0 || r >= ROWS || c < 0 || c >= COLS)
    return false;
  return (bd.topIndex[r][c] < MAX_STACK);
}

Board makeMove(const Board& bd, int r, int c, bool isBlack) {
  Board newbd = bd;
  int layer = newbd.topIndex[r][c];
  newbd.topIndex[r][c]++;
  int pos = indexOf(r, c, layer);

  if (isBlack) {
    newbd.blackBB.setBit(pos);
    // XOR zobrist
    newbd.boardHash ^= ZOBRIST[pos][0];
  } else {
    newbd.whiteBB.setBit(pos);
    newbd.boardHash ^= ZOBRIST[pos][1];
  }
  return newbd;
}

bool isBoardFull(const Board& bd) {
  for (int r = 0; r < ROWS; r++) {
    for (int c = 0; c < COLS; c++) {
      if (bd.topIndex[r][c] < MAX_STACK)
        return false;
    }
  }
  return true;
}

/*
========================================
  6. Winning Conditions
     - check for 4 consecutive lines
     - check blackBB / whiteBB states
========================================
*/
// lines: every 4 index positions
static vector<array<int, 4>> allLines;

static void buildAllLines() {
  allLines.clear();
  static const int dirs[13][3] = {
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
      for (int l = 0; l < MAX_STACK; l++) {
        int base = indexOf(r, c, l);
        for (auto& d : dirs) {
          vector<int> tmp;
          tmp.push_back(base);
          int rr = r, cc = c, ll = l;
          for (int step = 0; step < 3; step++) {
            rr += d[0];
            cc += d[1];
            ll += d[2];
            if (rr < 0 || rr >= ROWS || cc < 0 || cc >= COLS || ll < 0 || ll >= MAX_STACK) {
              break;
            }
            tmp.push_back(indexOf(rr, cc, ll));
          }
          if ((int)tmp.size() == 4) {
            array<int, 4> line;
            for (int i = 0; i < 4; i++) {
              line[i] = tmp[i];
            }
            allLines.push_back(line);
          }
        }
      }
    }
  }
}

bool checkWin(const BitBoard& b) {
  for (auto& ln : allLines) {
    if (b.testBit(ln[0]) && b.testBit(ln[1]) &&
        b.testBit(ln[2]) && b.testBit(ln[3])) {
      return true;
    }
  }
  return false;
}

char checkWinner(const Board& bd) {
  if (checkWin(bd.blackBB))
    return 'B';
  if (checkWin(bd.whiteBB))
    return 'W';
  return '\0';
}

/*
========================================
  7. Heuristics
     - evaluate all lines, count the blocked and open positions
     - (also consider historical move order)
========================================
*/
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

double evaluate(const Board& bd) {
  if (checkWin(bd.blackBB))
    return 99999999.0;
  if (checkWin(bd.whiteBB))
    return -99999999.0;

  long long blackScore = 0, whiteScore = 0;
  for (auto& ln : allLines) {
    bool hasB = false, hasW = false;
    int Bcount = 0, Wcount = 0;
    for (int i = 0; i < 4; i++) {
      int pos = ln[i];
      bool b = bd.blackBB.testBit(pos);
      bool w = bd.whiteBB.testBit(pos);
      if (b) {
        hasB = true;
        Bcount++;
      }
      if (w) {
        hasW = true;
        Wcount++;
      }
      if (hasB && hasW)
        break;
    }
    if (hasB && !hasW) {
      switch (Bcount) {
        case 4:
          blackScore += 9999999;
          break;
        case 3:
          blackScore += 5000;
          break;
        case 2:
          blackScore += 200;
          break;
        case 1:
          blackScore += 10;
          break;
        default:
          break;
      }
    } else if (hasW && !hasB) {
      switch (Wcount) {
        case 4:
          whiteScore += 9999999;
          break;
        case 3:
          whiteScore += 5000;
          break;
        case 2:
          whiteScore += 200;
          break;
        case 1:
          whiteScore += 10;
          break;
        default:
          break;
      }
    }
  }
  return (double)blackScore - (double)whiteScore;
}

/*
========================================
  8. User Interface + Game Loop + Minimax
========================================
*/
enum class BoundType { EXACT,
                       LOWER,
                       UPPER };

struct TTEntry {
  int depth;
  double value;
  BoundType boundType;
  pair<int, int> bestMove;
};

struct U64Hash {
  size_t operator()(uint64_t x) const {
    return std::hash<uint64_t>()(x);
  }
};

static unordered_map<uint64_t, TTEntry, U64Hash> transpositionTable;
static int historyHeuristic[ROWS][COLS];

static inline double now_in_seconds() {
  using namespace std::chrono;
  auto tp = high_resolution_clock::now();
  auto dur = tp.time_since_epoch();
  return double(duration_cast<milliseconds>(dur).count()) / 1000.0;
}

bool ttLookup(uint64_t key, int depth, double alpha, double beta, double& val, pair<int, int>& bestMv) {
  auto it = transpositionTable.find(key);
  if (it == transpositionTable.end())
    return false;
  const TTEntry& e = it->second;
  if (e.depth < depth)
    return false;

  if (e.boundType == BoundType::EXACT) {
    val = e.value;
    bestMv = e.bestMove;
    return true;
  } else if (e.boundType == BoundType::LOWER && e.value > alpha) {
    if (e.value >= beta) {
      val = e.value;
      bestMv = e.bestMove;
      return true;
    }
    alpha = max(alpha, e.value);
  } else if (e.boundType == BoundType::UPPER && e.value < beta) {
    if (e.value <= alpha) {
      val = e.value;
      bestMv = e.bestMove;
      return true;
    }
    beta = min(beta, e.value);
  }
  return alpha >= beta;
}

void ttStore(uint64_t key, int depth, double val, double alpha, double beta, pair<int, int> bestMv) {
  BoundType bt;
  if (val <= alpha)
    bt = BoundType::UPPER;
  else if (val >= beta)
    bt = BoundType::LOWER;
  else
    bt = BoundType::EXACT;
  TTEntry entry{depth, val, bt, bestMv};
  transpositionTable[key] = entry;
}

void orderMoves(vector<pair<int, int>>& moves, const pair<int, int>& ttMove) {
  if (ttMove.first != -1) {
    auto it = find(moves.begin(), moves.end(), ttMove);
    if (it != moves.end()) {
      iter_swap(moves.begin(), it);
    }
  }
  if (moves.size() > 1) {
    sort(moves.begin() + 1, moves.end(), [&](auto& m1, auto& m2) {
      int h1 = historyHeuristic[m1.first][m1.second];
      int h2 = historyHeuristic[m2.first][m2.second];
      return h1 > h2;
    });
  }
}

/*
========================================
  9. Minimax
========================================
*/
pair<double, pair<int, int>> minimaxSingle(
    const Board& bd,
    int depth,
    double alpha,
    double beta,
    bool maximizing);

double doNullMove(const Board& bd, int depth, double alpha, double beta) {
  int R = 1;
  auto ret = minimaxSingle(bd, depth - 1 - R, alpha, beta, false);
  return ret.first;
}

static bool allowNullMove = true;
bool nullMoveAllowed(const Board& bd, int depth) {
  if (!allowNullMove)
    return false;
  if (depth < 2)
    return false;
  return true;
}

pair<double, pair<int, int>> minimaxSingle(
    const Board& bd,
    int depth,
    double alpha,
    double beta,
    bool maximizing) {
  // TT
  uint64_t key = bd.boardHash;
  {
    double val;
    pair<int, int> bm(-1, -1);
    if (ttLookup(key, depth, alpha, beta, val, bm)) {
      return {val, bm};
    }
  }

  double sc = evaluate(bd);
  if (fabs(sc) >= 99999999.0 || depth == 0 || isBoardFull(bd)) {
    return {sc, {-1, -1}};
  }

  // Null Move
  if (maximizing && nullMoveAllowed(bd, depth)) {
    double nm = doNullMove(bd, depth, alpha, beta);
    if (nm >= beta) {
      return {nm, {-1, -1}};
    }
  }

  // Move Generation
  vector<pair<int, int>> moves;
  moves.reserve(25);
  for (int r = 0; r < ROWS; r++) {
    for (int c = 0; c < COLS; c++) {
      if (isValidMove(bd, r, c)) {
        moves.push_back({r, c});
      }
    }
  }
  if (moves.empty()) {
    return {sc, {-1, -1}};
  }

  // TT best
  pair<int, int> ttMove(-1, -1);
  {
    auto it = transpositionTable.find(key);
    if (it != transpositionTable.end()) {
      ttMove = it->second.bestMove;
    }
  }
  orderMoves(moves, ttMove);

  if (maximizing) {
    double bestVal = -1e15;
    pair<int, int> bestMove(-1, -1);

    for (auto& mv : moves) {
      auto nbd = makeMove(bd, mv.first, mv.second, true /* black */);
      auto ret = minimaxSingle(nbd, depth - 1, alpha, beta, false);
      double val = ret.first;
      if (val > bestVal) {
        bestVal = val;
        bestMove = mv;
      }
      alpha = max(alpha, bestVal);
      if (alpha >= beta) {
        historyHeuristic[mv.first][mv.second] += depth * depth;
        break;
      }
    }
    ttStore(key, depth, bestVal, alpha, beta, bestMove);
    return {bestVal, bestMove};
  } else {
    double bestVal = +1e15;
    pair<int, int> bestMove(-1, -1);
    for (auto& mv : moves) {
      auto nbd = makeMove(bd, mv.first, mv.second, false /* white */);
      auto ret = minimaxSingle(nbd, depth - 1, alpha, beta, true);
      double val = ret.first;
      if (val < bestVal) {
        bestVal = val;
        bestMove = mv;
      }
      beta = min(beta, bestVal);
      if (alpha >= beta) {
        historyHeuristic[mv.first][mv.second] += depth * depth;
        break;
      }
    }
    ttStore(key, depth, bestVal, alpha, beta, bestMove);
    return {bestVal, bestMove};
  }
}

/*
========================================
 10. Search Best Move
========================================
*/
pair<double, pair<int, int>> searchBestMove(const Board& bd, int maxDepth, bool maximizing) {
  pair<double, pair<int, int>> bestRes = {0.0, {-1, -1}};
  for (int d = 1; d <= maxDepth; d++) {
    auto ret = minimaxSingle(bd, d, -1e15, +1e15, maximizing);
    if (ret.second.first != -1) {
      bestRes = ret;
    }
    if (fabs(bestRes.first) >= 99999999.0) {
      break;
    }
  }
  return bestRes;
}

/*
========================================
 11. Check Immediate Win or Defense
========================================
*/
pair<int, int> checkImmediateWinOrDefense(const Board& bd) {
  for (int r = 0; r < ROWS; r++) {
    for (int c = 0; c < COLS; c++) {
      if (isValidMove(bd, r, c)) {
        auto nb = makeMove(bd, r, c, true);
        if (checkWin(nb.blackBB)) {
          return {r, c};
        }
      }
    }
  }
  for (int r = 0; r < ROWS; r++) {
    for (int c = 0; c < COLS; c++) {
      if (isValidMove(bd, r, c)) {
        auto nb = makeMove(bd, r, c, false);
        if (checkWin(nb.whiteBB)) {
          return {r, c};
        }
      }
    }
  }
  return {-1, -1};
}

/*
========================================
 12. Print Board
========================================
*/
void printBoard(const Board& bd) {
  cout << "Current board (bottom to top): \n";
  for (int r = 0; r < ROWS; r++) {
    for (int c = 0; c < COLS; c++) {
      int h = bd.topIndex[r][c];
      if (h == 0) {
        cout << "[ ] ";
      } else {
        cout << "[";
        for (int layer = 0; layer < h; layer++) {
          int pos = indexOf(r, c, layer);
          bool b = bd.blackBB.testBit(pos);
          bool w = bd.whiteBB.testBit(pos);
          if (b)
            cout << 'B';
          else if (w)
            cout << 'W';
          else
            cout << '?';
        }
        cout << "] ";
      }
    }
    cout << "\n";
  }
  cout << "\n";
}

int main() {
  ios::sync_with_stdio(false);
  cin.tie(nullptr);

  initZobrist();
  buildAllLines();
  memset(historyHeuristic, 0, sizeof(historyHeuristic));

  Board board;
  // boardHash=0; black/whiteBB=0; topIndex=0

  char currentPlayer = AI;
  bool aiFirstMoveDone = false;

  while (true) {
    printBoard(board);
    char w = checkWinner(board);
    if (w != '\0') {
      cout << "Game end! Winner is: " << w << "\n";
      break;
    }
    if (isBoardFull(board)) {
      cout << "Board full!\n";
      break;
    }

    if (currentPlayer == PLAYER) {
      cout << "Please input position to insert (e.g. 1[ROW] 1[COL]): " << flush;
      string line;
      if (!getline(cin, line))
        break;
      if (line.empty())
        continue;
      int rr, cc;
      {
        stringstream ss(line);
        ss >> rr >> cc;
      }
      rr--;
      cc--;
      if (!isValidMove(board, rr, cc)) {
        cout << "Invalid input!\n";
        continue;
      }
      board = makeMove(board, rr, cc, false /*white*/);
      currentPlayer = AI;
    } else {
      // AI
      cout << "AI thinking...\n";
      double st = now_in_seconds();
      if (!aiFirstMoveDone) {
        auto ret = searchBestMove(board, MAX_DEPTH, true);
        double sp = now_in_seconds() - st;
        auto mv = ret.second;
        if (mv.first != -1) {
          board = makeMove(board, mv.first, mv.second, true);
          cout << "\033[92mAI deciding: Row " << mv.first + 1 << ", Col " << mv.second + 1
               << ", Spending " << sp << "s\n"
               << "\033[0m" << endl;;
        }
        aiFirstMoveDone = true;
      } else {
        auto urgent = checkImmediateWinOrDefense(board);
        if (urgent.first != -1) {
          board = makeMove(board, urgent.first, urgent.second, true);
          double sp = now_in_seconds() - st;
          cout << "\033[92mAI urgent move: Row " << urgent.first + 1
               << ", Col" << urgent.second + 1
               << ", Spending " << sp << "s\n"
               << "\033[0m" << endl;
        } else {
          auto bestRet = searchBestMove(board, MAX_DEPTH, true);
          double sp = now_in_seconds() - st;
          double bestScore = bestRet.first;
          auto mv = bestRet.second;
          if (mv.first != -1) {
            cout << "\033[92mAI deciding: Row " << mv.first + 1 << ", Col" << mv.second + 1
                 << ", Spending " << sp << "s"
                 << ", score=" << bestScore
                 << "\033[0m" << endl;;
            board = makeMove(board, mv.first, mv.second, true);
          }
        }
      }
      currentPlayer = PLAYER;
    }
  }

  printBoard(board);
  cout << "Game end!\n";
  return 0;
}
