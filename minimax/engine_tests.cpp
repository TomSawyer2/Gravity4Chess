#include "gravity_engine.hpp"

#include <cstdlib>
#include <iostream>
#include <random>

using namespace gravity4;

namespace {

void require(bool condition, const char* message) {
  if (!condition) {
    std::cerr << "FAIL: " << message << '\n';
    std::exit(1);
  }
}

int slowHeuristic(const Board& board) {
  int score = 0;
  for (const WinningLine& line : geometry().lines) {
    int black = 0;
    int white = 0;
    for (int pos : line.cells) {
      black += board.black.test(pos);
      white += board.white.test(pos);
    }
    score += lineContribution(black, white);
  }
  return score;
}

bool sameBoard(const Board& lhs, const Board& rhs) {
  return lhs.black == rhs.black && lhs.white == rhs.white &&
         lhs.heights == rhs.heights &&
         lhs.blackLineCount == rhs.blackLineCount &&
         lhs.whiteLineCount == rhs.whiteLineCount &&
         lhs.hash == rhs.hash && lhs.legalMask == rhs.legalMask &&
         lhs.evaluation == rhs.evaluation && lhs.ply == rhs.ply;
}

int referenceThreat(Board& board, Side side, int alpha, int beta,
                    int lastPos, int searchPly, int remaining) {
  if (lastPos >= 0 && board.hasWinAt(opposite(side), lastPos)) {
    return -kMateScore + searchPly;
  }
  if (board.isFull()) {
    return side == Side::Black ? board.evaluation : -board.evaluation;
  }

  int forcedBlock = kNoMove;
  int opponentThreats = 0;
  uint32_t mask = board.legalMask;
  while (mask != 0) {
    const int move = __builtin_ctz(mask);
    mask &= mask - 1;
    if (board.wouldWin(side, move)) return kMateScore - searchPly - 1;
    if (board.wouldWin(opposite(side), move)) {
      forcedBlock = move;
      ++opponentThreats;
    }
  }

  if (opponentThreats >= 2) return -kMateScore + searchPly + 2;
  if (opponentThreats == 0 || remaining == 0) {
    return side == Side::Black ? board.evaluation : -board.evaluation;
  }

  const MoveUndo undo = board.makeMove(forcedBlock, side);
  const int value = -referenceThreat(board, opposite(side), -beta, -alpha,
                                     undo.pos, searchPly + 1, remaining - 1);
  board.unmakeMove(undo, side);
  return value;
}

int referenceNegamax(Board& board, Side side, int depth, int alpha, int beta,
                     int lastPos, int searchPly) {
  if (lastPos >= 0 && board.hasWinAt(opposite(side), lastPos)) {
    return -kMateScore + searchPly;
  }
  if (depth == 0) {
    return referenceThreat(board, side, alpha, beta, lastPos, searchPly, 4);
  }
  if (board.isFull()) {
    return side == Side::Black ? board.evaluation : -board.evaluation;
  }

  int best = -kInfinity;
  uint32_t mask = board.legalMask;
  while (mask != 0) {
    const int move = __builtin_ctz(mask);
    mask &= mask - 1;
    const MoveUndo undo = board.makeMove(move, side);
    const int value = -referenceNegamax(board, opposite(side), depth - 1,
                                        -beta, -alpha, undo.pos,
                                        searchPly + 1);
    board.unmakeMove(undo, side);
    best = std::max(best, value);
    alpha = std::max(alpha, best);
    if (alpha >= beta) break;
  }
  return best;
}

int referenceRoot(Board board, Side side, int depth) {
  int best = -kInfinity;
  uint32_t mask = board.legalMask;
  while (mask != 0) {
    const int move = __builtin_ctz(mask);
    mask &= mask - 1;
    const MoveUndo undo = board.makeMove(move, side);
    const int value = -referenceNegamax(board, opposite(side), depth - 1,
                                        -kInfinity, kInfinity, undo.pos, 1);
    board.unmakeMove(undo, side);
    best = std::max(best, value);
  }
  return best;
}

void testGeometry() {
  size_t references = 0;
  for (const auto& lines : geometry().linesByCell) references += lines.size();
  require(references == kLineCount * 4, "winning-line reverse index");
  require(geometry().linesByCell[positionOf(12, 2)].size() == 26,
          "center cell should participate in 26 lines");
}

void testIncrementalBoard() {
  std::mt19937 rng(20260920);
  for (int game = 0; game < 100; ++game) {
    Board board;
    Side side = Side::Black;
    for (int turn = 0; turn < 80 && !board.isFull(); ++turn) {
      std::vector<int> legal;
      for (int move = 0; move < kColumns; ++move) {
        if (board.isLegal(move)) legal.push_back(move);
      }
      const int move = legal[rng() % legal.size()];
      const Board before = board;
      const MoveUndo undo = board.makeMove(move, side);

      require(board.evaluation == slowHeuristic(board),
              "incremental evaluation must match slow evaluation");
      require(board.hasWinAt(side, undo.pos) == board.hasWin(side),
              "last-move win check must match full-board win check");

      board.unmakeMove(undo, side);
      require(sameBoard(board, before), "make/unmake must restore the board");
      board.makeMove(move, side);
      if (board.hasWinAt(side, undo.pos)) break;
      side = opposite(side);
    }
  }
}

void testKnownWin() {
  Board board;
  int lastPos = -1;
  for (int move = 0; move < 4; ++move) {
    lastPos = board.makeMove(move, Side::Black).pos;
  }
  require(board.hasWinAt(Side::Black, lastPos), "known horizontal win");
  require(board.hasWin(Side::Black), "full scan known horizontal win");
}

void testSearchAgainstReference() {
  const std::array<std::vector<int>, 5> openings = {{
      {},
      {12, 0},
      {6, 18, 12},
      {2, 22, 7, 17},
      {10, 14, 11, 13, 12},
  }};

  for (const auto& opening : openings) {
    Board board;
    Side side = Side::Black;
    for (int move : opening) {
      const MoveUndo undo = board.makeMove(move, side);
      require(!board.hasWinAt(side, undo.pos), "test opening must be non-terminal");
      side = opposite(side);
    }

    const int expected = referenceRoot(board, side, 3);
    OptimizedEngine engine(4);
    const SearchResult actual = engine.search(board, side, SearchLimits{3, 0, 0});
    require(actual.move != kNoMove && board.isLegal(actual.move),
            "optimized engine must return a legal move");
    require(actual.score == expected,
            "optimized PVS/TT score must match exact reference search");
  }
}

void testLegacySmoke() {
  Board board;
  LegacyEngine engine;
  const SearchResult result = engine.search(board, Side::Black,
                                            SearchLimits{3, 0, 0});
  require(result.move != kNoMove && board.isLegal(result.move),
          "legacy engine must return a legal move");
  require(result.stats.completedDepth == 3, "legacy engine completes depth 3");
}

void testRootMoveAnalysis() {
  Board board;
  OptimizedEngine engine(4);
  const SearchResult result = engine.search(
      board, Side::Black, SearchLimits{3, 0, 0, true});

  require(result.rootMoves.size() == kColumns,
          "root analysis must expand every legal column");
  int bestScore = -kInfinity;
  bool foundBestMove = false;
  std::array<bool, kColumns> seen{};
  for (const auto& candidate : result.rootMoves) {
    require(board.isLegal(candidate.move),
            "root analysis candidates must be legal");
    require(!seen[candidate.move],
            "root analysis candidates must be unique");
    seen[candidate.move] = true;
    bestScore = std::max(bestScore, candidate.score);
    if (candidate.move == result.move) foundBestMove = true;
  }
  require(foundBestMove, "root analysis must include the selected move");
  require(bestScore == result.score,
          "root analysis best score must match the search result");
  require(result.rootMoves[0].score == result.rootMoves[4].score &&
              result.rootMoves[0].score == result.rootMoves[20].score &&
              result.rootMoves[0].score == result.rootMoves[24].score,
          "symmetric root moves must share their expanded score");
}

}  // namespace

int main() {
  testGeometry();
  testIncrementalBoard();
  testKnownWin();
  testSearchAgainstReference();
  testLegacySmoke();
  testRootMoveAnalysis();
  std::cout << "All engine tests passed.\n";
  return 0;
}
