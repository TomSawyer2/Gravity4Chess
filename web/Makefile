CXX ?= c++
CXXFLAGS ?= -std=c++17 -O3 -DNDEBUG -Wall -Wextra -Wpedantic

.PHONY: native-test wasm benchmark-cli clean

native-test: engine/wasm_bindings_test
	./engine/wasm_bindings_test

engine/wasm_bindings_test: engine/wasm_bindings.cpp engine/wasm_bindings_test.cpp \
		engine/wasm_api.hpp ../minimax/gravity_engine.hpp
	$(CXX) $(CXXFLAGS) engine/wasm_bindings.cpp engine/wasm_bindings_test.cpp -o $@

wasm:
	./scripts/build-wasm.sh

benchmark-cli: engine/analyze_cli

engine/analyze_cli: engine/wasm_bindings.cpp engine/wasm_api.hpp \
		../minimax/gravity_engine.hpp
	$(CXX) $(CXXFLAGS) -march=native -flto -DGRAVITY4_ANALYZE_CLI \
		engine/wasm_bindings.cpp -o $@

clean:
	$(RM) engine/wasm_bindings_test engine/analyze_cli
