#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
WEB_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
OUTPUT_DIR="${WEB_DIR}/public/wasm"

if [[ -d /opt/homebrew/opt/emscripten/libexec ]]; then
  export PATH="/opt/homebrew/opt/emscripten/libexec/binaryen/bin:/opt/homebrew/opt/emscripten/libexec/llvm/bin:/opt/homebrew/bin:${PATH}"
  if [[ -x /opt/homebrew/opt/python@3.14/bin/python3.14 ]]; then
    export EMSDK_PYTHON="/opt/homebrew/opt/python@3.14/bin/python3.14"
  fi
fi

if ! command -v em++ >/dev/null 2>&1; then
  echo "error: em++ was not found. Install and activate the Emscripten SDK first." >&2
  exit 1
fi

mkdir -p "${OUTPUT_DIR}"

em++ "${WEB_DIR}/engine/wasm_bindings.cpp" \
  -std=c++17 \
  -O3 \
  -flto \
  -DNDEBUG \
  -msimd128 \
  -lembind \
  --no-entry \
  -sWASM=1 \
  -sMODULARIZE=1 \
  -sEXPORT_ES6=1 \
  -sEXPORT_NAME=createGravity4Engine \
  -sENVIRONMENT=web,worker,node \
  -sINCOMING_MODULE_JS_API=locateFile,wasmBinary \
  -sFILESYSTEM=0 \
  -sALLOW_MEMORY_GROWTH=1 \
  -sINITIAL_MEMORY=67108864 \
  -sMAXIMUM_MEMORY=536870912 \
  -sASSERTIONS=0 \
  -o "${OUTPUT_DIR}/gravity4-engine.js"

echo "WASM engine written to ${OUTPUT_DIR}"
