#include "wasm_api.hpp"

#include <cstdlib>
#include <iostream>
#include <string>

namespace {

void require(bool condition, const char* message) {
  if (!condition) {
    std::cerr << "FAIL: " << message << '\n';
    std::exit(1);
  }
}

int occurrences(const std::string& value, const std::string& needle) {
  int count = 0;
  size_t offset = 0;
  while ((offset = value.find(needle, offset)) != std::string::npos) {
    ++count;
    offset += needle.size();
  }
  return count;
}

}  // namespace

int main() {
  const std::string opening = gravity4::web::analyzePosition("", 3, 0, 4);
  require(opening.find("\"ok\":true") != std::string::npos,
          "opening analysis succeeds");
  require(opening.find("\"sideToMove\":\"B\"") != std::string::npos,
          "black moves first");
  require(occurrences(opening, "\"move\":") == 25,
          "opening analysis contains all 25 candidates");

  const std::string terminal = gravity4::web::analyzePosition(
      "0,5,1,6,2,7,3", 3, 0, 4);
  require(terminal.find("\"terminal\":true") != std::string::npos,
          "terminal position is recognized");
  require(terminal.find("\"winner\":\"B\"") != std::string::npos,
          "terminal response identifies black as winner");

  const std::string invalid = gravity4::web::analyzePosition("25", 3, 0, 4);
  require(invalid.find("\"ok\":false") != std::string::npos,
          "invalid input is returned as a structured error");

  std::cout << "All web engine contract tests passed.\n";
  return 0;
}
