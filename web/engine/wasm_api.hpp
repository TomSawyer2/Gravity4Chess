#pragma once

#include <string>

namespace gravity4::web {

std::string analyzePosition(const std::string& movesCsv, int maxDepth,
                            int timeLimitMs, int tableMegabytes);

}  // namespace gravity4::web
