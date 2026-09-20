#pragma once

#include <algorithm>
#include <array>
#include <cassert>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <limits>
#include <random>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

namespace gravity4 {

constexpr int kRows = 5;
constexpr int kCols = 5;
constexpr int kLayers = 5;
constexpr int kColumns = kRows * kCols;
constexpr int kCells = kColumns * kLayers;
constexpr int kLineCount = 302;
constexpr int kMateScore = 99'999'999;
constexpr int kInfinity = 1'000'000'000;
constexpr int kNoMove = -1;

enum class Side : uint8_t { Black = 0, White = 1 };

inline Side opposite(Side side) {
  return side == Side::Black ? Side::White : Side::Black;
}

inline int columnOf(int row, int col) {
  return row * kCols + col;
}

inline int positionOf(int column, int layer) {
  return column * kLayers + layer;
}

inline int rowOf(int column) {
  return column / kCols;
}

inline int colOf(int column) {
  return column % kCols;
}

struct BitBoard {
  uint64_t lo = 0;
  uint64_t hi = 0;

  void set(int pos) {
    if (pos < 64) {
      lo |= 1ULL << pos;
    } else {
      hi |= 1ULL << (pos - 64);
    }
  }

  void reset(int pos) {
    if (pos < 64) {
      lo &= ~(1ULL << pos);
    } else {
      hi &= ~(1ULL << (pos - 64));
    }
  }

  bool test(int pos) const {
    if (pos < 64) return ((lo >> pos) & 1ULL) != 0;
    return ((hi >> (pos - 64)) & 1ULL) != 0;
  }

  bool operator==(const BitBoard& rhs) const {
    return lo == rhs.lo && hi == rhs.hi;
  }
};

struct WinningLine {
  std::array<uint8_t, 4> cells{};
  uint64_t loMask = 0;
  uint64_t hiMask = 0;
};

struct Geometry {
  std::array<WinningLine, kLineCount> lines{};
  std::array<std::vector<uint16_t>, kCells> linesByCell{};
  std::array<std::array<uint8_t, kColumns>, 8> transforms{};

  Geometry() {
    buildLines();
    buildTransforms();
  }

 private:
  void buildLines() {
    static constexpr int directions[13][3] = {
        {0, 1, 0},   {1, 0, 0},   {1, 1, 0},   {1, -1, 0},
        {0, 0, 1},   {1, 0, 1},   {-1, 0, 1},  {0, 1, 1},
        {0, -1, 1},  {1, 1, 1},   {1, -1, 1},  {-1, 1, 1},
        {-1, -1, 1},
    };

    int lineCount = 0;
    for (int row = 0; row < kRows; ++row) {
      for (int col = 0; col < kCols; ++col) {
        for (int layer = 0; layer < kLayers; ++layer) {
          for (const auto& direction : directions) {
            int r = row;
            int c = col;
            int l = layer;
            WinningLine line;
            bool valid = true;

            for (int step = 0; step < 4; ++step) {
              if (r < 0 || r >= kRows || c < 0 || c >= kCols ||
                  l < 0 || l >= kLayers) {
                valid = false;
                break;
              }

              const int pos = positionOf(columnOf(r, c), l);
              line.cells[step] = static_cast<uint8_t>(pos);
              if (pos < 64) {
                line.loMask |= 1ULL << pos;
              } else {
                line.hiMask |= 1ULL << (pos - 64);
              }

              r += direction[0];
              c += direction[1];
              l += direction[2];
            }

            if (!valid) continue;
            assert(lineCount < kLineCount);
            lines[lineCount] = line;
            for (int pos : line.cells) {
              linesByCell[pos].push_back(static_cast<uint16_t>(lineCount));
            }
            ++lineCount;
          }
        }
      }
    }
    assert(lineCount == kLineCount);
  }

  void buildTransforms() {
    for (int row = 0; row < kRows; ++row) {
      for (int col = 0; col < kCols; ++col) {
        const int source = columnOf(row, col);
        const std::array<std::pair<int, int>, 8> destinations = {{
            {row, col},
            {col, kRows - 1 - row},
            {kRows - 1 - row, kCols - 1 - col},
            {kCols - 1 - col, row},
            {row, kCols - 1 - col},
            {kRows - 1 - row, col},
            {col, row},
            {kCols - 1 - col, kRows - 1 - row},
        }};

        for (int transform = 0; transform < 8; ++transform) {
          transforms[transform][source] = static_cast<uint8_t>(
              columnOf(destinations[transform].first,
                       destinations[transform].second));
        }
      }
    }
  }
};

inline const Geometry& geometry() {
  static const Geometry value;
  return value;
}

struct ZobristKeys {
  std::array<std::array<uint64_t, 2>, kCells> piece{};
  uint64_t whiteToMove = 0;

  ZobristKeys() {
    uint64_t state = 0x6a09e667f3bcc909ULL;
    for (auto& cell : piece) {
      cell[0] = splitMix64(state);
      cell[1] = splitMix64(state);
    }
    whiteToMove = splitMix64(state);
  }

 private:
  static uint64_t splitMix64(uint64_t& state) {
    uint64_t z = (state += 0x9e3779b97f4a7c15ULL);
    z = (z ^ (z >> 30)) * 0xbf58476d1ce4e5b9ULL;
    z = (z ^ (z >> 27)) * 0x94d049bb133111ebULL;
    return z ^ (z >> 31);
  }
};

inline const ZobristKeys& zobrist() {
  static const ZobristKeys value;
  return value;
}

inline int lineContribution(int blackCount, int whiteCount) {
  static constexpr int score[5] = {0, 10, 200, 5000, 9'999'999};
  if (blackCount != 0 && whiteCount != 0) return 0;
  if (blackCount != 0) return score[blackCount];
  if (whiteCount != 0) return -score[whiteCount];
  return 0;
}

struct MoveUndo {
  uint8_t move = 0;
  uint8_t pos = 0;
};

struct Board {
  BitBoard black;
  BitBoard white;
  std::array<uint8_t, kColumns> heights{};
  std::array<uint8_t, kLineCount> blackLineCount{};
  std::array<uint8_t, kLineCount> whiteLineCount{};
  uint64_t hash = 0;
  uint32_t legalMask = (1U << kColumns) - 1U;
  int evaluation = 0;
  uint8_t ply = 0;

  const BitBoard& pieces(Side side) const {
    return side == Side::Black ? black : white;
  }

  bool isLegal(int move) const {
    return move >= 0 && move < kColumns &&
           ((legalMask >> move) & 1U) != 0;
  }

  int nextPosition(int move) const {
    return positionOf(move, heights[move]);
  }

  bool isFull() const {
    return ply == kCells;
  }

  bool wouldWin(Side side, int move) const {
    if (!isLegal(move)) return false;
    const int pos = nextPosition(move);
    const auto& ownCounts = side == Side::Black ? blackLineCount : whiteLineCount;
    const auto& otherCounts = side == Side::Black ? whiteLineCount : blackLineCount;

    for (int lineId : geometry().linesByCell[pos]) {
      if (ownCounts[lineId] == 3 && otherCounts[lineId] == 0) return true;
    }
    return false;
  }

  MoveUndo makeMove(int move, Side side) {
    assert(isLegal(move));
    const int pos = nextPosition(move);

    for (int lineId : geometry().linesByCell[pos]) {
      evaluation -= lineContribution(blackLineCount[lineId],
                                     whiteLineCount[lineId]);
      if (side == Side::Black) {
        ++blackLineCount[lineId];
      } else {
        ++whiteLineCount[lineId];
      }
      evaluation += lineContribution(blackLineCount[lineId],
                                     whiteLineCount[lineId]);
    }

    if (side == Side::Black) {
      black.set(pos);
    } else {
      white.set(pos);
    }
    hash ^= zobrist().piece[pos][static_cast<int>(side)];

    ++heights[move];
    ++ply;
    if (heights[move] == kLayers) legalMask &= ~(1U << move);

    return MoveUndo{static_cast<uint8_t>(move), static_cast<uint8_t>(pos)};
  }

  void unmakeMove(const MoveUndo& undo, Side side) {
    const int move = undo.move;
    const int pos = undo.pos;

    if (heights[move] == kLayers) legalMask |= 1U << move;
    --heights[move];
    --ply;

    hash ^= zobrist().piece[pos][static_cast<int>(side)];
    if (side == Side::Black) {
      black.reset(pos);
    } else {
      white.reset(pos);
    }

    for (int lineId : geometry().linesByCell[pos]) {
      evaluation -= lineContribution(blackLineCount[lineId],
                                     whiteLineCount[lineId]);
      if (side == Side::Black) {
        --blackLineCount[lineId];
      } else {
        --whiteLineCount[lineId];
      }
      evaluation += lineContribution(blackLineCount[lineId],
                                     whiteLineCount[lineId]);
    }
  }

  bool hasWinAt(Side side, int pos) const {
    const auto& counts = side == Side::Black ? blackLineCount : whiteLineCount;
    for (int lineId : geometry().linesByCell[pos]) {
      if (counts[lineId] == 4) return true;
    }
    return false;
  }

  bool hasWin(Side side) const {
    const BitBoard& board = pieces(side);
    for (const WinningLine& line : geometry().lines) {
      if ((board.lo & line.loMask) == line.loMask &&
          (board.hi & line.hiMask) == line.hiMask) {
        return true;
      }
    }
    return false;
  }

  bool invariantUnder(int transform) const {
    const auto& mapping = geometry().transforms[transform];
    for (int move = 0; move < kColumns; ++move) {
      const int mapped = mapping[move];
      if (heights[move] != heights[mapped]) return false;
      for (int layer = 0; layer < heights[move]; ++layer) {
        const int sourcePos = positionOf(move, layer);
        const int mappedPos = positionOf(mapped, layer);
        if (black.test(sourcePos) != black.test(mappedPos) ||
            white.test(sourcePos) != white.test(mappedPos)) {
          return false;
        }
      }
    }
    return true;
  }

  std::vector<int> rootMovesModuloSymmetry() const {
    std::vector<int> symmetries;
    for (int transform = 0; transform < 8; ++transform) {
      if (invariantUnder(transform)) symmetries.push_back(transform);
    }

    std::vector<int> moves;
    uint32_t mask = legalMask;
    while (mask != 0) {
      const int move = __builtin_ctz(mask);
      mask &= mask - 1;

      int representative = move;
      for (int transform : symmetries) {
        representative = std::min(
            representative,
            static_cast<int>(geometry().transforms[transform][move]));
      }
      if (move == representative) moves.push_back(move);
    }
    return moves;
  }
};

inline int slowEvaluate(const Board& board) {
  if (board.hasWin(Side::Black)) return kMateScore;
  if (board.hasWin(Side::White)) return -kMateScore;

  int score = 0;
  for (const WinningLine& line : geometry().lines) {
    int blackCount = 0;
    int whiteCount = 0;
    for (int pos : line.cells) {
      blackCount += board.black.test(pos);
      whiteCount += board.white.test(pos);
    }
    score += lineContribution(blackCount, whiteCount);
  }
  return score;
}

struct SearchLimits {
  int maxDepth = 9;
  int timeLimitMs = 0;
  uint64_t nodeLimit = 0;
};

struct SearchStats {
  uint64_t nodes = 0;
  uint64_t evaluations = 0;
  uint64_t ttProbes = 0;
  uint64_t ttHits = 0;
  uint64_t ttCutoffs = 0;
  uint64_t betaCutoffs = 0;
  int completedDepth = 0;
  int score = 0;
  int move = kNoMove;
  double elapsedMs = 0;

  double nps() const {
    return elapsedMs > 0 ? nodes * 1000.0 / elapsedMs : 0;
  }
};

struct SearchResult {
  int move = kNoMove;
  int score = 0;
  SearchStats stats;
};

class SearchClock {
 public:
  void start(const SearchLimits& limits) {
    limits_ = limits;
    start_ = std::chrono::steady_clock::now();
    deadline_ = start_ + std::chrono::milliseconds(limits.timeLimitMs);
  }

  bool expired(uint64_t nodes) const {
    if (limits_.nodeLimit != 0 && nodes >= limits_.nodeLimit) return true;
    if (limits_.timeLimitMs == 0) return false;
    return std::chrono::steady_clock::now() >= deadline_;
  }

  double elapsedMs() const {
    const auto elapsed = std::chrono::steady_clock::now() - start_;
    return std::chrono::duration<double, std::milli>(elapsed).count();
  }

 private:
  SearchLimits limits_;
  std::chrono::steady_clock::time_point start_;
  std::chrono::steady_clock::time_point deadline_;
};

struct LegacyBoard {
  BitBoard black;
  BitBoard white;
  int height[kRows][kCols]{};
  uint64_t hash = 0;
};

inline LegacyBoard toLegacyBoard(const Board& board) {
  LegacyBoard legacy;
  legacy.black = board.black;
  legacy.white = board.white;
  legacy.hash = board.hash;
  for (int move = 0; move < kColumns; ++move) {
    legacy.height[rowOf(move)][colOf(move)] = board.heights[move];
  }
  return legacy;
}

class LegacyEngine {
 public:
  SearchResult search(const Board& board, Side side, const SearchLimits& limits) {
    limits_ = limits;
    stats_ = {};
    stopped_ = false;
    clock_.start(limits);

    const LegacyBoard legacy = toLegacyBoard(board);
    const int urgent = immediateMove(legacy, side);
    if (urgent != kNoMove) {
      stats_.elapsedMs = clock_.elapsedMs();
      stats_.move = urgent;
      stats_.score = side == Side::Black ? kMateScore : -kMateScore;
      return SearchResult{urgent, stats_.score, stats_};
    }
    std::pair<int, int> best = {0, kNoMove};

    for (int depth = 1; depth <= limits.maxDepth; ++depth) {
      const auto result = minimax(legacy, depth, -kInfinity, kInfinity,
                                  side == Side::Black);
      if (stopped_) break;
      if (result.second != kNoMove) best = result;
      stats_.completedDepth = depth;
      if (std::abs(best.first) >= kMateScore) break;
    }

    stats_.elapsedMs = clock_.elapsedMs();
    stats_.move = best.second;
    stats_.score = best.first;
    return SearchResult{best.second, best.first, stats_};
  }

 private:
  enum class Bound : uint8_t { Exact, Lower, Upper };

  struct Entry {
    int depth = 0;
    int value = 0;
    Bound bound = Bound::Exact;
    int bestMove = kNoMove;
  };

  SearchLimits limits_;
  SearchStats stats_;
  SearchClock clock_;
  bool stopped_ = false;
  std::unordered_map<uint64_t, Entry> table_;
  int history_[kRows][kCols]{};

  bool shouldStop() {
    if ((stats_.nodes & 2047ULL) != 0) return false;
    if (!clock_.expired(stats_.nodes)) return false;
    stopped_ = true;
    return true;
  }

  static bool valid(const LegacyBoard& board, int move) {
    return board.height[rowOf(move)][colOf(move)] < kLayers;
  }

  static LegacyBoard makeMove(const LegacyBoard& board, int move, Side side) {
    LegacyBoard next = board;
    const int row = rowOf(move);
    const int col = colOf(move);
    const int layer = next.height[row][col]++;
    const int pos = positionOf(move, layer);
    if (side == Side::Black) {
      next.black.set(pos);
    } else {
      next.white.set(pos);
    }
    next.hash ^= zobrist().piece[pos][static_cast<int>(side)];
    return next;
  }

  static bool full(const LegacyBoard& board) {
    for (int move = 0; move < kColumns; ++move) {
      if (valid(board, move)) return false;
    }
    return true;
  }

  static bool win(const BitBoard& pieces) {
    for (const WinningLine& line : geometry().lines) {
      if (pieces.test(line.cells[0]) && pieces.test(line.cells[1]) &&
          pieces.test(line.cells[2]) && pieces.test(line.cells[3])) {
        return true;
      }
    }
    return false;
  }

  int evaluate(const LegacyBoard& board) {
    ++stats_.evaluations;
    if (win(board.black)) return kMateScore;
    if (win(board.white)) return -kMateScore;

    int score = 0;
    for (const WinningLine& line : geometry().lines) {
      int blackCount = 0;
      int whiteCount = 0;
      for (int pos : line.cells) {
        blackCount += board.black.test(pos);
        whiteCount += board.white.test(pos);
        if (blackCount != 0 && whiteCount != 0) break;
      }
      score += lineContribution(blackCount, whiteCount);
    }
    return score;
  }

  int immediateMove(const LegacyBoard& board, Side side) {
    for (int move = 0; move < kColumns; ++move) {
      if (!valid(board, move)) continue;
      const LegacyBoard next = makeMove(board, move, side);
      if (win(side == Side::Black ? next.black : next.white)) return move;
    }
    const Side opponent = opposite(side);
    for (int move = 0; move < kColumns; ++move) {
      if (!valid(board, move)) continue;
      const LegacyBoard next = makeMove(board, move, opponent);
      if (win(opponent == Side::Black ? next.black : next.white)) return move;
    }
    return kNoMove;
  }

  bool lookup(uint64_t key, int depth, int alpha, int beta,
              int& value, int& bestMove) {
    ++stats_.ttProbes;
    const auto it = table_.find(key);
    if (it == table_.end() || it->second.depth < depth) return false;
    ++stats_.ttHits;

    const Entry& entry = it->second;
    if (entry.bound == Bound::Exact) {
      value = entry.value;
      bestMove = entry.bestMove;
      ++stats_.ttCutoffs;
      return true;
    }
    if (entry.bound == Bound::Lower && entry.value > alpha) {
      if (entry.value >= beta) {
        value = entry.value;
        bestMove = entry.bestMove;
        ++stats_.ttCutoffs;
        return true;
      }
      alpha = std::max(alpha, entry.value);
    } else if (entry.bound == Bound::Upper && entry.value < beta) {
      if (entry.value <= alpha) {
        value = entry.value;
        bestMove = entry.bestMove;
        ++stats_.ttCutoffs;
        return true;
      }
      beta = std::min(beta, entry.value);
    }
    return alpha >= beta;
  }

  void store(uint64_t key, int depth, int value, int alpha, int beta,
             int bestMove) {
    Bound bound = Bound::Exact;
    if (value <= alpha) {
      bound = Bound::Upper;
    } else if (value >= beta) {
      bound = Bound::Lower;
    }
    table_[key] = Entry{depth, value, bound, bestMove};
  }

  void orderMoves(std::vector<int>& moves, int ttMove) {
    if (ttMove != kNoMove) {
      const auto it = std::find(moves.begin(), moves.end(), ttMove);
      if (it != moves.end()) std::iter_swap(moves.begin(), it);
    }
    if (moves.size() > 1) {
      std::sort(moves.begin() + 1, moves.end(), [&](int lhs, int rhs) {
        return history_[rowOf(lhs)][colOf(lhs)] >
               history_[rowOf(rhs)][colOf(rhs)];
      });
    }
  }

  std::pair<int, int> minimax(const LegacyBoard& board, int depth,
                              int alpha, int beta, bool maximizing) {
    ++stats_.nodes;
    if (shouldStop()) return {0, kNoMove};

    int cachedValue = 0;
    int cachedMove = kNoMove;
    if (lookup(board.hash, depth, alpha, beta, cachedValue, cachedMove)) {
      return {cachedValue, cachedMove};
    }

    const int staticScore = evaluate(board);
    if (std::abs(staticScore) >= kMateScore || depth == 0 || full(board)) {
      return {staticScore, kNoMove};
    }

    if (maximizing && depth >= 2) {
      const auto nullResult = minimax(board, depth - 2, alpha, beta, false);
      if (stopped_) return {0, kNoMove};
      if (nullResult.first >= beta) return {nullResult.first, kNoMove};
    }

    std::vector<int> moves;
    moves.reserve(kColumns);
    for (int move = 0; move < kColumns; ++move) {
      if (valid(board, move)) moves.push_back(move);
    }
    if (moves.empty()) return {staticScore, kNoMove};

    int ttMove = kNoMove;
    const auto tt = table_.find(board.hash);
    if (tt != table_.end()) ttMove = tt->second.bestMove;
    orderMoves(moves, ttMove);

    int bestMove = kNoMove;
    if (maximizing) {
      int bestValue = -kInfinity;
      for (int move : moves) {
        const LegacyBoard next = makeMove(board, move, Side::Black);
        const int value = minimax(next, depth - 1, alpha, beta, false).first;
        if (stopped_) return {0, kNoMove};
        if (value > bestValue) {
          bestValue = value;
          bestMove = move;
        }
        alpha = std::max(alpha, bestValue);
        if (alpha >= beta) {
          ++stats_.betaCutoffs;
          history_[rowOf(move)][colOf(move)] += depth * depth;
          break;
        }
      }
      store(board.hash, depth, bestValue, alpha, beta, bestMove);
      return {bestValue, bestMove};
    }

    int bestValue = kInfinity;
    for (int move : moves) {
      const LegacyBoard next = makeMove(board, move, Side::White);
      const int value = minimax(next, depth - 1, alpha, beta, true).first;
      if (stopped_) return {0, kNoMove};
      if (value < bestValue) {
        bestValue = value;
        bestMove = move;
      }
      beta = std::min(beta, bestValue);
      if (alpha >= beta) {
        ++stats_.betaCutoffs;
        history_[rowOf(move)][colOf(move)] += depth * depth;
        break;
      }
    }
    store(board.hash, depth, bestValue, alpha, beta, bestMove);
    return {bestValue, bestMove};
  }
};

class OptimizedEngine {
 public:
  explicit OptimizedEngine(size_t tableMegabytes = 32) {
    size_t entries = 1;
    const size_t targetBytes = tableMegabytes * 1024ULL * 1024ULL;
    while ((entries << 1) * sizeof(TTEntry) <= targetBytes) entries <<= 1;
    table_.resize(entries);
    tableMask_ = entries - 1;
    for (auto& plyKillers : killers_) {
      plyKillers[0] = kNoMove;
      plyKillers[1] = kNoMove;
    }
  }

  SearchResult search(Board board, Side side, const SearchLimits& limits) {
    limits_ = limits;
    stats_ = {};
    stopped_ = false;
    rootSide_ = side;
    ++generation_;
    clock_.start(limits);

    int bestMove = kNoMove;
    int bestScore = 0;

    for (int depth = 1; depth <= limits.maxDepth; ++depth) {
      int alpha = -kInfinity;
      int beta = kInfinity;
      int delta = 1000;

      if (depth >= 3 && std::abs(bestScore) < kMateScore / 2) {
        alpha = std::max(-kInfinity, bestScore - delta);
        beta = std::min(kInfinity, bestScore + delta);
      }

      RootResult iteration;
      while (true) {
        iteration = searchRoot(board, side, depth, alpha, beta);
        if (stopped_) break;

        if (iteration.score <= alpha && alpha > -kInfinity) {
          alpha = std::max(-kInfinity, alpha - delta);
          delta *= 4;
          continue;
        }
        if (iteration.score >= beta && beta < kInfinity) {
          beta = std::min(kInfinity, beta + delta);
          delta *= 4;
          continue;
        }
        break;
      }

      if (stopped_) break;
      if (iteration.move != kNoMove) {
        bestMove = iteration.move;
        bestScore = iteration.score;
      }
      stats_.completedDepth = depth;
      if (std::abs(bestScore) >= kMateScore - kCells) break;
    }

    stats_.elapsedMs = clock_.elapsedMs();
    stats_.move = bestMove;
    stats_.score = bestScore;
    return SearchResult{bestMove, bestScore, stats_};
  }

 private:
  enum class Bound : uint8_t { Empty, Exact, Lower, Upper };

  struct TTEntry {
    uint64_t key = 0;
    int32_t score = 0;
    int16_t depth = -1;
    uint16_t generation = 0;
    Bound bound = Bound::Empty;
    uint8_t move = 255;
  };

  struct ScoredMove {
    int move = kNoMove;
    int score = 0;
    bool tactical = false;
  };

  struct RootResult {
    int score = 0;
    int move = kNoMove;
  };

  SearchLimits limits_;
  SearchStats stats_;
  SearchClock clock_;
  bool stopped_ = false;
  Side rootSide_ = Side::Black;
  uint16_t generation_ = 0;
  std::vector<TTEntry> table_;
  size_t tableMask_ = 0;
  int history_[2][kColumns]{};
  int killers_[kCells + 1][2]{};

  bool shouldStop() {
    if ((stats_.nodes & 1023ULL) != 0) return false;
    if (!clock_.expired(stats_.nodes)) return false;
    stopped_ = true;
    return true;
  }

  uint64_t key(const Board& board, Side side) const {
    return board.hash ^
           (side == Side::White ? zobrist().whiteToMove : 0ULL);
  }

  static int perspectiveScore(const Board& board, Side side) {
    return side == Side::Black ? board.evaluation : -board.evaluation;
  }

  static int scoreToTable(int value, int searchPly) {
    if (value > kMateScore - kCells) return value + searchPly;
    if (value < -kMateScore + kCells) return value - searchPly;
    return value;
  }

  static int scoreFromTable(int value, int searchPly) {
    if (value > kMateScore - kCells) return value - searchPly;
    if (value < -kMateScore + kCells) return value + searchPly;
    return value;
  }

  bool probe(uint64_t positionKey, int depth, int searchPly,
             int& alpha, int& beta, int& value, int& ttMove) {
    ++stats_.ttProbes;
    const TTEntry& entry = table_[positionKey & tableMask_];
    if (entry.bound == Bound::Empty || entry.key != positionKey) return false;

    ++stats_.ttHits;
    ttMove = entry.move == 255 ? kNoMove : entry.move;
    if (entry.depth < depth) return false;

    const int decodedScore = scoreFromTable(entry.score, searchPly);
    if (entry.bound == Bound::Exact) {
      value = decodedScore;
      ++stats_.ttCutoffs;
      return true;
    }
    if (entry.bound == Bound::Lower) alpha = std::max(alpha, decodedScore);
    if (entry.bound == Bound::Upper) beta = std::min(beta, decodedScore);
    if (alpha < beta) return false;

    value = decodedScore;
    ++stats_.ttCutoffs;
    return true;
  }

  void store(uint64_t positionKey, int depth, int searchPly, int value,
             int alphaOriginal, int betaOriginal, int bestMove) {
    TTEntry& entry = table_[positionKey & tableMask_];
    const bool samePosition = entry.bound != Bound::Empty && entry.key == positionKey;
    if (!samePosition && entry.generation == generation_ && entry.depth > depth) return;
    if (samePosition && entry.depth > depth && entry.bound == Bound::Exact) return;

    Bound bound = Bound::Exact;
    if (value <= alphaOriginal) {
      bound = Bound::Upper;
    } else if (value >= betaOriginal) {
      bound = Bound::Lower;
    }

    entry.key = positionKey;
    entry.score = scoreToTable(value, searchPly);
    entry.depth = static_cast<int16_t>(depth);
    entry.generation = generation_;
    entry.bound = bound;
    entry.move = bestMove == kNoMove ? 255 : static_cast<uint8_t>(bestMove);
  }

  int moveScore(const Board& board, Side side, int move, int ttMove,
                int searchPly) const {
    if (move == ttMove) return 2'000'000;
    if (board.wouldWin(side, move)) return 1'500'000;
    if (board.wouldWin(opposite(side), move)) return 1'000'000;
    if (killers_[searchPly][0] == move) return 800'000;
    if (killers_[searchPly][1] == move) return 700'000;

    const int centerDistance = std::abs(rowOf(move) - 2) +
                               std::abs(colOf(move) - 2);
    return history_[static_cast<int>(side)][move] + (4 - centerDistance) * 32;
  }

  int generateMoves(const Board& board, Side side, int ttMove, int searchPly,
                    std::array<ScoredMove, kColumns>& moves,
                    const std::vector<int>* rootMoves = nullptr) const {
    int count = 0;
    if (rootMoves != nullptr) {
      for (int move : *rootMoves) {
        moves[count++] = ScoredMove{
            move,
            moveScore(board, side, move, ttMove, searchPly),
            board.wouldWin(side, move) || board.wouldWin(opposite(side), move)};
      }
    } else {
      uint32_t mask = board.legalMask;
      while (mask != 0) {
        const int move = __builtin_ctz(mask);
        mask &= mask - 1;
        moves[count++] = ScoredMove{
            move,
            moveScore(board, side, move, ttMove, searchPly),
            board.wouldWin(side, move) || board.wouldWin(opposite(side), move)};
      }
    }

    for (int i = 1; i < count; ++i) {
      ScoredMove value = moves[i];
      int j = i;
      while (j > 0 && moves[j - 1].score < value.score) {
        moves[j] = moves[j - 1];
        --j;
      }
      moves[j] = value;
    }
    return count;
  }

  int threatSearch(Board& board, Side side, int alpha, int beta,
                   int lastPos, int searchPly, int remaining,
                   bool countNode) {
    if (countNode) {
      ++stats_.nodes;
      if (shouldStop()) return 0;
    }

    if (lastPos >= 0 && board.hasWinAt(opposite(side), lastPos)) {
      return -kMateScore + searchPly;
    }
    if (board.isFull()) {
      ++stats_.evaluations;
      return perspectiveScore(board, side);
    }

    int forcedBlock = kNoMove;
    int opponentThreats = 0;
    uint32_t mask = board.legalMask;
    while (mask != 0) {
      const int move = __builtin_ctz(mask);
      mask &= mask - 1;
      if (board.wouldWin(side, move)) {
        return kMateScore - searchPly - 1;
      }
      if (board.wouldWin(opposite(side), move)) {
        forcedBlock = move;
        ++opponentThreats;
      }
    }

    if (opponentThreats >= 2) return -kMateScore + searchPly + 2;
    if (opponentThreats == 0 || remaining == 0) {
      ++stats_.evaluations;
      return perspectiveScore(board, side);
    }

    const MoveUndo undo = board.makeMove(forcedBlock, side);
    const int value = -threatSearch(board, opposite(side), -beta, -alpha,
                                    undo.pos, searchPly + 1,
                                    remaining - 1, true);
    board.unmakeMove(undo, side);
    return value;
  }

  int negamax(Board& board, Side side, int depth, int alpha, int beta,
              int lastPos, int searchPly) {
    ++stats_.nodes;
    if (shouldStop()) return 0;

    if (lastPos >= 0 && board.hasWinAt(opposite(side), lastPos)) {
      return -kMateScore + searchPly;
    }
    if (depth == 0) {
      return threatSearch(board, side, alpha, beta, lastPos,
                          searchPly, 4, false);
    }
    if (board.isFull()) {
      ++stats_.evaluations;
      return perspectiveScore(board, side);
    }

    const int alphaOriginal = alpha;
    const int betaOriginal = beta;
    const uint64_t positionKey = key(board, side);
    int ttValue = 0;
    int ttMove = kNoMove;
    if (probe(positionKey, depth, searchPly, alpha, beta, ttValue, ttMove)) {
      return ttValue;
    }

    std::array<ScoredMove, kColumns> moves;
    const int moveCount = generateMoves(board, side, ttMove, searchPly, moves);
    if (moveCount == 0) {
      ++stats_.evaluations;
      return perspectiveScore(board, side);
    }

    int bestValue = -kInfinity;
    int bestMove = kNoMove;
    bool firstMove = true;

    for (int i = 0; i < moveCount; ++i) {
      const ScoredMove candidate = moves[i];
      const MoveUndo undo = board.makeMove(candidate.move, side);

      int value;
      if (board.hasWinAt(side, undo.pos)) {
        value = kMateScore - searchPly - 1;
      } else if (firstMove) {
        value = -negamax(board, opposite(side), depth - 1,
                         -beta, -alpha, undo.pos, searchPly + 1);
      } else {
        value = -negamax(board, opposite(side), depth - 1,
                         -alpha - 1, -alpha, undo.pos, searchPly + 1);
        if (!stopped_ && value > alpha && value < beta) {
          value = -negamax(board, opposite(side), depth - 1,
                           -beta, -alpha, undo.pos, searchPly + 1);
        }
      }

      board.unmakeMove(undo, side);
      if (stopped_) return 0;
      firstMove = false;

      if (value > bestValue) {
        bestValue = value;
        bestMove = candidate.move;
      }
      alpha = std::max(alpha, bestValue);
      if (alpha >= beta) {
        ++stats_.betaCutoffs;
        if (!candidate.tactical) {
          if (killers_[searchPly][0] != candidate.move) {
            killers_[searchPly][1] = killers_[searchPly][0];
            killers_[searchPly][0] = candidate.move;
          }
          int& history = history_[static_cast<int>(side)][candidate.move];
          history += depth * depth;
          if (history > 1'000'000) {
            for (auto& sideHistory : history_) {
              for (int& item : sideHistory) item /= 2;
            }
          }
        }
        break;
      }
    }

    store(positionKey, depth, searchPly, bestValue,
          alphaOriginal, betaOriginal, bestMove);
    return bestValue;
  }

  RootResult searchRoot(Board& board, Side side, int depth, int alpha, int beta) {
    const int alphaOriginal = alpha;
    const int betaOriginal = beta;
    const uint64_t positionKey = key(board, side);

    int ignoredValue = 0;
    int ttMove = kNoMove;
    int probeAlpha = alpha;
    int probeBeta = beta;
    if (probe(positionKey, depth, 0, probeAlpha, probeBeta,
              ignoredValue, ttMove)) {
      const TTEntry& entry = table_[positionKey & tableMask_];
      if (entry.bound == Bound::Exact && ttMove != kNoMove) {
        return RootResult{ignoredValue, ttMove};
      }
    }

    const std::vector<int> uniqueRootMoves = board.rootMovesModuloSymmetry();
    std::array<ScoredMove, kColumns> moves;
    const int moveCount = generateMoves(board, side, ttMove, 0, moves,
                                        &uniqueRootMoves);
    if (moveCount == 0) return RootResult{perspectiveScore(board, side), kNoMove};

    int bestValue = -kInfinity;
    int bestMove = kNoMove;
    bool firstMove = true;

    for (int i = 0; i < moveCount; ++i) {
      const int move = moves[i].move;
      const MoveUndo undo = board.makeMove(move, side);

      int value;
      if (board.hasWinAt(side, undo.pos)) {
        value = kMateScore - 1;
      } else if (firstMove) {
        value = -negamax(board, opposite(side), depth - 1,
                         -beta, -alpha, undo.pos, 1);
      } else {
        value = -negamax(board, opposite(side), depth - 1,
                         -alpha - 1, -alpha, undo.pos, 1);
        if (!stopped_ && value > alpha && value < beta) {
          value = -negamax(board, opposite(side), depth - 1,
                           -beta, -alpha, undo.pos, 1);
        }
      }

      board.unmakeMove(undo, side);
      if (stopped_) return RootResult{};
      firstMove = false;

      if (value > bestValue) {
        bestValue = value;
        bestMove = move;
      }
      alpha = std::max(alpha, bestValue);
      if (alpha >= beta) {
        ++stats_.betaCutoffs;
        break;
      }
    }

    store(positionKey, depth, 0, bestValue,
          alphaOriginal, betaOriginal, bestMove);
    return RootResult{bestValue, bestMove};
  }
};

inline char sideChar(Side side) {
  return side == Side::Black ? 'B' : 'W';
}

inline std::string moveName(int move) {
  if (move == kNoMove) return "--";
  return std::to_string(rowOf(move) + 1) + "," +
         std::to_string(colOf(move) + 1);
}

}  // namespace gravity4
