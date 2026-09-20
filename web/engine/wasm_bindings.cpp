#include "../../minimax/gravity_engine.hpp"
#include "wasm_api.hpp"

#include <algorithm>
#include <cmath>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

#ifdef __EMSCRIPTEN__
#include <emscripten/bind.h>
#endif

namespace gravity4::web {
namespace {

std::string escapeJson(const std::string& value) {
  std::ostringstream output;
  for (const char ch : value) {
    switch (ch) {
      case '"': output << "\\\""; break;
      case '\\': output << "\\\\"; break;
      case '\n': output << "\\n"; break;
      case '\r': output << "\\r"; break;
      case '\t': output << "\\t"; break;
      default: output << ch; break;
    }
  }
  return output.str();
}

std::vector<int> parseMoves(const std::string& movesCsv) {
  std::vector<int> moves;
  if (movesCsv.empty()) return moves;

  std::stringstream stream(movesCsv);
  std::string token;
  while (std::getline(stream, token, ',')) {
    if (token.empty()) throw std::invalid_argument("empty move token");
    size_t consumed = 0;
    const int move = std::stoi(token, &consumed);
    if (consumed != token.size() || move < 0 || move >= kColumns) {
      throw std::invalid_argument("move must be an integer from 0 to 24");
    }
    moves.push_back(move);
  }
  return moves;
}

double sideWinRate(int score) {
  if (score >= kMateScore - kCells) return 1.0;
  if (score <= -kMateScore + kCells) return 0.0;
  constexpr double kScoreScale = 6000.0;
  const double exponent = std::clamp(-score / kScoreScale, -30.0, 30.0);
  return 1.0 / (1.0 + std::exp(exponent));
}

double blackWinRate(int score, Side sideToMove) {
  const double currentSideRate = sideWinRate(score);
  return sideToMove == Side::Black ? currentSideRate : 1.0 - currentSideRate;
}

void writeBoard(std::ostringstream& output, const Board& board) {
  output << "\"heights\":[";
  for (int move = 0; move < kColumns; ++move) {
    if (move != 0) output << ',';
    output << static_cast<int>(board.heights[move]);
  }
  output << "],\"stacks\":[";
  for (int move = 0; move < kColumns; ++move) {
    if (move != 0) output << ',';
    output << '[';
    for (int layer = 0; layer < board.heights[move]; ++layer) {
      if (layer != 0) output << ',';
      const int pos = positionOf(move, layer);
      output << '"' << (board.black.test(pos) ? 'B' : 'W') << '"';
    }
    output << ']';
  }
  output << ']';
}

std::string errorJson(const std::string& message) {
  return "{\"ok\":false,\"error\":\"" + escapeJson(message) + "\"}";
}

}  // namespace

std::string analyzePosition(const std::string& movesCsv, int maxDepth,
                            int timeLimitMs, int tableMegabytes) {
  try {
    if (maxDepth < 1 || maxDepth > kCells) {
      throw std::invalid_argument("maxDepth must be between 1 and 125");
    }
    if (timeLimitMs < 0) {
      throw std::invalid_argument("timeLimitMs cannot be negative");
    }
    if (tableMegabytes < 1 || tableMegabytes > 256) {
      throw std::invalid_argument("tableMegabytes must be between 1 and 256");
    }

    const std::vector<int> moves = parseMoves(movesCsv);
    Board board;
    Side side = Side::Black;
    Side winner = Side::Black;
    bool hasWinner = false;
    int lastPos = -1;

    for (size_t ply = 0; ply < moves.size(); ++ply) {
      if (hasWinner) {
        throw std::invalid_argument("moves continue after the game was won");
      }
      const int move = moves[ply];
      if (!board.isLegal(move)) {
        throw std::invalid_argument("move targets a full or invalid column");
      }
      lastPos = board.makeMove(move, side).pos;
      if (board.hasWinAt(side, lastPos)) {
        winner = side;
        hasWinner = true;
      }
      side = opposite(side);
    }

    std::ostringstream output;
    output << std::fixed << std::setprecision(6);
    output << "{\"ok\":true,\"engineVersion\":1,\"ply\":"
           << static_cast<int>(board.ply) << ",\"sideToMove\":\""
           << sideChar(side) << "\",";
    writeBoard(output, board);

    if (hasWinner || board.isFull()) {
      output << ",\"terminal\":true,\"winner\":";
      if (hasWinner) {
        output << '"' << sideChar(winner) << '"';
      } else {
        output << "null";
      }
      output << ",\"bestMove\":-1,\"score\":0,\"blackWinRate\":"
             << (hasWinner ? (winner == Side::Black ? 1.0 : 0.0) : 0.5)
             << ",\"candidates\":[],\"stats\":{\"depth\":0,"
                "\"elapsedMs\":0,\"nodes\":0,\"nps\":0,"
                "\"ttHitRate\":0}}";
      return output.str();
    }

    OptimizedEngine engine(static_cast<size_t>(tableMegabytes));
    const SearchResult result = engine.search(
        board, side, SearchLimits{maxDepth, timeLimitMs, 0, true});
    if (result.move == kNoMove || !board.isLegal(result.move)) {
      throw std::runtime_error("search did not complete a legal iteration");
    }

    output << ",\"terminal\":false,\"winner\":null,\"bestMove\":"
           << result.move << ",\"score\":" << result.score
           << ",\"blackWinRate\":" << blackWinRate(result.score, side)
           << ",\"candidates\":[";

    for (size_t index = 0; index < result.rootMoves.size(); ++index) {
      const auto& candidate = result.rootMoves[index];
      if (index != 0) output << ',';
      const bool wins = board.wouldWin(side, candidate.move);
      const bool blocks = !wins && board.wouldWin(opposite(side), candidate.move);
      output << "{\"move\":" << candidate.move
             << ",\"row\":" << rowOf(candidate.move)
             << ",\"col\":" << colOf(candidate.move)
             << ",\"layer\":" << static_cast<int>(board.heights[candidate.move])
             << ",\"score\":" << candidate.score
             << ",\"sideWinRate\":" << sideWinRate(candidate.score)
             << ",\"blackWinRate\":" << blackWinRate(candidate.score, side)
             << ",\"tag\":";
      if (wins) {
        output << "\"win\"";
      } else if (blocks) {
        output << "\"block\"";
      } else {
        output << "null";
      }
      output << '}';
    }

    const double ttHitRate = result.stats.ttProbes == 0
                                 ? 0.0
                                 : static_cast<double>(result.stats.ttHits) /
                                       result.stats.ttProbes;
    output << "],\"stats\":{\"depth\":" << result.stats.completedDepth
           << ",\"elapsedMs\":" << result.stats.elapsedMs
           << ",\"nodes\":" << result.stats.nodes
           << ",\"nps\":" << result.stats.nps()
           << ",\"ttHitRate\":" << ttHitRate << "}}";
    return output.str();
  } catch (const std::exception& error) {
    return errorJson(error.what());
  }
}

}  // namespace gravity4::web

#ifdef __EMSCRIPTEN__
EMSCRIPTEN_BINDINGS(gravity4_engine) {
  emscripten::function("analyzePosition", &gravity4::web::analyzePosition);
}
#endif

#ifdef GRAVITY4_ANALYZE_CLI
int main(int argc, char** argv) {
  const std::string moves = argc > 1 ? argv[1] : "";
  const int depth = argc > 2 ? std::stoi(argv[2]) : 5;
  const int timeLimitMs = argc > 3 ? std::stoi(argv[3]) : 0;
  const int tableMegabytes = argc > 4 ? std::stoi(argv[4]) : 16;
  std::cout << gravity4::web::analyzePosition(
                   moves, depth, timeLimitMs, tableMegabytes)
            << '\n';
  return 0;
}
#endif
