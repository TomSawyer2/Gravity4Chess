#include "gravity_engine.hpp"

#include <algorithm>
#include <cstdlib>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <numeric>
#include <random>
#include <string>
#include <vector>

using namespace gravity4;

namespace {

struct PositionCase {
  std::string name;
  Board board;
  Side side = Side::Black;
};

struct MoveSample {
  std::string scenario;
  int game = -1;
  int ply = 0;
  std::string engine;
  Side side = Side::Black;
  SearchResult search;
  std::string result;
};

struct SeriesSummary {
  int optimizedWins = 0;
  int legacyWins = 0;
  int draws = 0;
  std::vector<double> optimizedMoveMs;
  std::vector<double> legacyMoveMs;
  std::vector<int> optimizedDepth;
  std::vector<int> legacyDepth;
  uint64_t optimizedNodes = 0;
  uint64_t legacyNodes = 0;
};

int intArg(int argc, char** argv, const std::string& name, int fallback) {
  for (int i = 1; i + 1 < argc; ++i) {
    if (argv[i] == name) return std::stoi(argv[i + 1]);
  }
  return fallback;
}

std::string stringArg(int argc, char** argv, const std::string& name,
                      const std::string& fallback) {
  for (int i = 1; i + 1 < argc; ++i) {
    if (argv[i] == name) return argv[i + 1];
  }
  return fallback;
}

PositionCase generatePosition(const std::string& name, uint32_t seed,
                              int targetPly) {
  PositionCase result;
  result.name = name;
  std::mt19937 rng(seed);
  Side side = Side::Black;

  while (result.board.ply < targetPly && !result.board.isFull()) {
    std::vector<int> quietMoves;
    std::vector<int> legalMoves;
    for (int move = 0; move < kColumns; ++move) {
      if (!result.board.isLegal(move)) continue;
      legalMoves.push_back(move);
      if (!result.board.wouldWin(side, move)) quietMoves.push_back(move);
    }

    const std::vector<int>& candidates = quietMoves.empty() ? legalMoves : quietMoves;
    if (candidates.empty()) break;
    const int move = candidates[rng() % candidates.size()];
    const MoveUndo undo = result.board.makeMove(move, side);
    if (result.board.hasWinAt(side, undo.pos)) break;
    side = opposite(side);
  }

  result.side = side;
  return result;
}

bool hasUrgentMove(const Board& board, Side side) {
  for (int move = 0; move < kColumns; ++move) {
    if (!board.isLegal(move)) continue;
    if (board.wouldWin(side, move) || board.wouldWin(opposite(side), move)) {
      return true;
    }
  }
  return false;
}

PositionCase generateQuietPosition(const std::string& name, uint32_t seed,
                                   int targetPly) {
  for (uint32_t attempt = 0; attempt < 10'000; ++attempt) {
    PositionCase position = generatePosition(name, seed + attempt, targetPly);
    if (position.board.ply == targetPly &&
        !hasUrgentMove(position.board, position.side)) {
      return position;
    }
  }
  std::cerr << "Could not generate quiet position " << name << '\n';
  std::exit(2);
}

std::vector<PositionCase> performancePositions() {
  return {
      generatePosition("empty", 1, 0),
      generateQuietPosition("opening-4", 101, 4),
      generateQuietPosition("opening-8", 202, 8),
      generateQuietPosition("midgame-12", 303, 12),
      generateQuietPosition("midgame-16", 404, 16),
      generateQuietPosition("midgame-20", 505, 20),
      generateQuietPosition("midgame-24", 606, 24),
      generateQuietPosition("midgame-28", 707, 28),
  };
}

double percentile(std::vector<double> values, double p) {
  if (values.empty()) return 0;
  std::sort(values.begin(), values.end());
  const size_t index = static_cast<size_t>(
      std::ceil(p * static_cast<double>(values.size()))) - 1;
  return values[std::min(index, values.size() - 1)];
}

double average(const std::vector<double>& values) {
  if (values.empty()) return 0;
  return std::accumulate(values.begin(), values.end(), 0.0) / values.size();
}

double averageDepth(const std::vector<int>& values) {
  if (values.empty()) return 0;
  return static_cast<double>(
             std::accumulate(values.begin(), values.end(), int64_t{0})) /
         values.size();
}

int firstLegal(const Board& board) {
  return board.legalMask == 0 ? kNoMove : __builtin_ctz(board.legalMask);
}

void appendSample(std::vector<MoveSample>& samples, const std::string& scenario,
                  int game, int ply, const std::string& engine, Side side,
                  const SearchResult& search, const std::string& result = "") {
  samples.push_back(MoveSample{scenario, game, ply, engine, side, search, result});
}

void printSearchRow(const std::string& position, const std::string& engine,
                    const SearchResult& result) {
  std::cout << std::left << std::setw(14) << position
            << std::setw(11) << engine
            << " move=" << std::setw(5) << moveName(result.move)
            << " depth=" << std::setw(2) << result.stats.completedDepth
            << " time=" << std::setw(10) << std::fixed << std::setprecision(3)
            << result.stats.elapsedMs << "ms"
            << " nodes=" << std::setw(12) << result.stats.nodes
            << " nps=" << static_cast<uint64_t>(result.stats.nps()) << '\n';
}

void runPerformanceSuite(int depth, std::vector<MoveSample>& samples) {
  std::cout << "\n=== Fixed-depth performance (depth " << depth << ") ===\n";
  double legacyMs = 0;
  double optimizedMs = 0;
  uint64_t legacyNodes = 0;
  uint64_t optimizedNodes = 0;

  int caseIndex = 0;
  for (const PositionCase& position : performancePositions()) {
    LegacyEngine legacy;
    OptimizedEngine optimized;
    const SearchLimits limits{depth, 0, 0};

    const SearchResult oldResult = legacy.search(position.board, position.side, limits);
    const SearchResult newResult = optimized.search(position.board, position.side, limits);

    printSearchRow(position.name, "legacy", oldResult);
    printSearchRow(position.name, "optimized", newResult);
    appendSample(samples, "fixed-depth", caseIndex, position.board.ply,
                 "legacy", position.side, oldResult);
    appendSample(samples, "fixed-depth", caseIndex, position.board.ply,
                 "optimized", position.side, newResult);

    legacyMs += oldResult.stats.elapsedMs;
    optimizedMs += newResult.stats.elapsedMs;
    legacyNodes += oldResult.stats.nodes;
    optimizedNodes += newResult.stats.nodes;
    ++caseIndex;
  }

  std::cout << "Fixed-depth total: legacy=" << legacyMs
            << "ms optimized=" << optimizedMs
            << "ms speedup=" << legacyMs / std::max(optimizedMs, 0.001)
            << "x nodes=" << legacyNodes << " -> " << optimizedNodes << "\n";
}

std::string playGame(int gameId, const std::string& scenario,
                     const PositionCase& opening, bool optimizedBlack,
                     const SearchLimits& legacyLimits,
                     const SearchLimits& optimizedLimits,
                     std::vector<MoveSample>& samples,
                     SeriesSummary& summary) {
  Board board = opening.board;
  Side side = opening.side;
  LegacyEngine legacy;
  OptimizedEngine optimized;
  int lastPos = -1;

  while (!board.isFull()) {
    const bool useOptimized =
        (side == Side::Black) == optimizedBlack;
    const std::string engineName = useOptimized ? "optimized" : "legacy";
    SearchResult result = useOptimized
                              ? optimized.search(board, side, optimizedLimits)
                              : legacy.search(board, side, legacyLimits);
    if (result.move == kNoMove || !board.isLegal(result.move)) {
      result.move = firstLegal(board);
    }

    if (useOptimized) {
      summary.optimizedMoveMs.push_back(result.stats.elapsedMs);
      summary.optimizedDepth.push_back(result.stats.completedDepth);
      summary.optimizedNodes += result.stats.nodes;
    } else {
      summary.legacyMoveMs.push_back(result.stats.elapsedMs);
      summary.legacyDepth.push_back(result.stats.completedDepth);
      summary.legacyNodes += result.stats.nodes;
    }

    lastPos = board.makeMove(result.move, side).pos;
    appendSample(samples, scenario, gameId, board.ply, engineName, side, result);

    if (board.hasWinAt(side, lastPos)) {
      const bool optimizedWon = useOptimized;
      if (optimizedWon) {
        ++summary.optimizedWins;
      } else {
        ++summary.legacyWins;
      }
      samples.back().result = optimizedWon ? "optimized-win" : "legacy-win";
      return samples.back().result;
    }
    side = opposite(side);
  }

  ++summary.draws;
  if (!samples.empty()) samples.back().result = "draw";
  return "draw";
}

void printSeriesSummary(const std::string& name, const SeriesSummary& summary) {
  const int games = summary.optimizedWins + summary.legacyWins + summary.draws;
  const double scoreRate = games == 0
                               ? 0
                               : (summary.optimizedWins + 0.5 * summary.draws) /
                                     static_cast<double>(games);

  std::cout << "\n=== " << name << " summary ===\n"
            << "optimized wins=" << summary.optimizedWins
            << ", legacy wins=" << summary.legacyWins
            << ", draws=" << summary.draws
            << ", optimized score rate=" << std::fixed << std::setprecision(1)
            << scoreRate * 100.0 << "%\n"
            << "optimized move ms: avg=" << average(summary.optimizedMoveMs)
            << " p50=" << percentile(summary.optimizedMoveMs, 0.50)
            << " p95=" << percentile(summary.optimizedMoveMs, 0.95)
            << " avg-depth=" << averageDepth(summary.optimizedDepth) << '\n'
            << "legacy move ms:    avg=" << average(summary.legacyMoveMs)
            << " p50=" << percentile(summary.legacyMoveMs, 0.50)
            << " p95=" << percentile(summary.legacyMoveMs, 0.95)
            << " avg-depth=" << averageDepth(summary.legacyDepth) << '\n';
}

void runSeries(const std::string& name, int games, int openingPly,
               const SearchLimits& legacyLimits,
               const SearchLimits& optimizedLimits,
               std::vector<MoveSample>& samples) {
  if (games % 2 != 0) ++games;
  SeriesSummary summary;

  std::cout << "\n=== " << name << " head-to-head: " << games << " games ===\n";
  for (int game = 0; game < games; ++game) {
    const int pair = game / 2;
    const PositionCase opening = generatePosition(
        "match-opening-" + std::to_string(pair), 9000 + pair, openingPly);
    const bool optimizedBlack = (game % 2 == 0);
    const std::string result = playGame(game, name, opening, optimizedBlack,
                                        legacyLimits, optimizedLimits,
                                        samples, summary);
    std::cout << "game " << std::setw(2) << game + 1
              << " optimized=" << (optimizedBlack ? "B" : "W")
              << " result=" << result << '\n';
  }
  printSeriesSummary(name, summary);
}

void writeCsv(const std::string& path, const std::vector<MoveSample>& samples) {
  std::ofstream output(path);
  output << "scenario,game,ply,engine,side,move,move_row,move_col,elapsed_ms,depth,nodes,nps,"
            "evaluations,tt_probes,tt_hits,tt_cutoffs,beta_cutoffs,score,result\n";
  output << std::fixed << std::setprecision(6);
  for (const MoveSample& sample : samples) {
    const SearchStats& stats = sample.search.stats;
    output << sample.scenario << ',' << sample.game << ',' << sample.ply << ','
           << sample.engine << ',' << sideChar(sample.side) << ','
           << sample.search.move << ','
           << (sample.search.move == kNoMove ? 0 : rowOf(sample.search.move) + 1) << ','
           << (sample.search.move == kNoMove ? 0 : colOf(sample.search.move) + 1) << ','
           << stats.elapsedMs << ','
           << stats.completedDepth << ',' << stats.nodes << ',' << stats.nps()
           << ',' << stats.evaluations << ',' << stats.ttProbes << ','
           << stats.ttHits << ',' << stats.ttCutoffs << ',' << stats.betaCutoffs
           << ',' << sample.search.score << ',' << sample.result << '\n';
  }
}

}  // namespace

int main(int argc, char** argv) {
  const int benchmarkDepth = intArg(argc, argv, "--bench-depth", 7);
  const int equalDepth = intArg(argc, argv, "--match-depth", 5);
  const int games = intArg(argc, argv, "--games", 12);
  const int depthGames = intArg(argc, argv, "--depth-games", games);
  const int timeGames = intArg(argc, argv, "--time-games", games);
  const int moveTimeMs = intArg(argc, argv, "--move-ms", 100);
  const int openingPly = intArg(argc, argv, "--opening-ply", 4);
  const std::string outputPath = stringArg(
      argc, argv, "--output", "benchmark_results.csv");

  std::vector<MoveSample> samples;
  if (benchmarkDepth > 0) runPerformanceSuite(benchmarkDepth, samples);

  if (depthGames > 0) {
    runSeries("equal-depth", depthGames, openingPly,
              SearchLimits{equalDepth, 0, 0},
              SearchLimits{equalDepth, 0, 0}, samples);
  }

  if (timeGames > 0) {
    runSeries("equal-time", timeGames, openingPly,
              SearchLimits{32, moveTimeMs, 0},
              SearchLimits{32, moveTimeMs, 0}, samples);
  }

  writeCsv(outputPath, samples);
  std::cout << "\nCSV written to " << outputPath << '\n';
  return 0;
}
