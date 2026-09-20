CXX ?= c++
CXXFLAGS ?= -std=c++17 -O3 -march=native -flto -funroll-loops -DNDEBUG \
	-Wall -Wextra -Wpedantic

.PHONY: all test bench clean

all: play_cpp_opti play_cpp_opti_legacy benchmark engine_tests

play_cpp_opti: play_cpp_opti.cpp gravity_engine.hpp
	$(CXX) $(CXXFLAGS) play_cpp_opti.cpp -o $@

play_cpp_opti_legacy: play_cpp_opti_legacy.cpp
	$(CXX) $(CXXFLAGS) play_cpp_opti_legacy.cpp -o $@

benchmark: benchmark.cpp gravity_engine.hpp
	$(CXX) $(CXXFLAGS) benchmark.cpp -o $@

engine_tests: engine_tests.cpp gravity_engine.hpp
	$(CXX) $(CXXFLAGS) engine_tests.cpp -o $@

test: engine_tests
	./engine_tests

bench: benchmark
	./benchmark --bench-depth 7 --match-depth 5 --games 12 --move-ms 100

clean:
	$(RM) play_cpp_opti play_cpp_opti_legacy benchmark engine_tests
