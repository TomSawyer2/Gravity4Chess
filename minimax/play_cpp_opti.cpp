#include "gravity_engine.hpp"

#include <iomanip>
#include <iostream>
#include <sstream>
#include <string>

using namespace gravity4;

namespace {

void printBoard(const Board& board) {
  std::cout << "Current board (bottom to top):\n";
  for (int row = 0; row < kRows; ++row) {
    for (int col = 0; col < kCols; ++col) {
      const int move = columnOf(row, col);
      std::cout << '[';
      for (int layer = 0; layer < board.heights[move]; ++layer) {
        const int pos = positionOf(move, layer);
        std::cout << (board.black.test(pos) ? 'B' : 'W');
      }
      std::cout << "] ";
    }
    std::cout << '\n';
  }
  std::cout << '\n';
}

int parseIntArg(int argc, char** argv, const std::string& name, int fallback) {
  for (int i = 1; i + 1 < argc; ++i) {
    if (argv[i] == name) return std::stoi(argv[i + 1]);
  }
  return fallback;
}

void printSearch(const SearchResult& result) {
  std::cout << "AI deciding: Row " << rowOf(result.move) + 1
            << ", Col " << colOf(result.move) + 1
            << ", time=" << std::fixed << std::setprecision(3)
            << result.stats.elapsedMs / 1000.0 << "s"
            << ", depth=" << result.stats.completedDepth
            << ", nodes=" << result.stats.nodes
            << ", nps=" << static_cast<uint64_t>(result.stats.nps())
            << ", tt-hit="
            << (result.stats.ttProbes == 0
                    ? 0.0
                    : 100.0 * result.stats.ttHits / result.stats.ttProbes)
            << "%"
            << ", score=" << result.score << '\n';
}

}  // namespace

int main(int argc, char** argv) {
  std::ios::sync_with_stdio(false);
  std::cin.tie(nullptr);

  const int maxDepth = parseIntArg(argc, argv, "--depth", 9);
  const int timeMs = parseIntArg(argc, argv, "--time-ms", 0);
  const int ttMegabytes = parseIntArg(argc, argv, "--tt-mb", 32);

  Board board;
  OptimizedEngine engine(ttMegabytes);
  Side current = Side::Black;
  int lastPos = -1;

  while (true) {
    printBoard(board);
    if (lastPos >= 0 && board.hasWinAt(opposite(current), lastPos)) {
      std::cout << "Game end! Winner is " << sideChar(opposite(current)) << ".\n";
      break;
    }
    if (board.isFull()) {
      std::cout << "Board full: draw.\n";
      break;
    }

    if (current == Side::White) {
      std::cout << "Please input position (row col): " << std::flush;
      std::string line;
      if (!std::getline(std::cin, line)) break;

      int row = 0;
      int col = 0;
      std::stringstream parser(line);
      if (!(parser >> row >> col)) {
        std::cout << "Invalid input.\n";
        continue;
      }

      if (row < 1 || row > kRows || col < 1 || col > kCols) {
        std::cout << "Invalid move.\n";
        continue;
      }
      const int move = columnOf(row - 1, col - 1);
      if (!board.isLegal(move)) {
        std::cout << "Invalid move.\n";
        continue;
      }
      lastPos = board.makeMove(move, current).pos;
    } else {
      std::cout << "AI thinking...\n";
      const SearchResult result = engine.search(
          board, current, SearchLimits{maxDepth, timeMs, 0});
      if (result.move == kNoMove) {
        std::cerr << "Search did not complete a legal iteration.\n";
        return 2;
      }
      printSearch(result);
      lastPos = board.makeMove(result.move, current).pos;
    }

    current = opposite(current);
  }
  return 0;
}
