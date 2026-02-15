#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BENCH_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
if [[ -n "${SENGOO_ROOT:-}" ]]; then
  PROJECT_ROOT="${SENGOO_ROOT}"
else
  PROJECT_ROOT="$(cd "${BENCH_ROOT}/.." && pwd)"
fi
cd "${PROJECT_ROOT}"

SAMPLE="${BENCH_ROOT}/tests/hello_print.sg"
OUT_DIR="${BENCH_ROOT}/tests/build/e2e"
LLVM_OUT="${OUT_DIR}/hello_print.ll"

mkdir -p "${OUT_DIR}"

if [[ ! -f "${SAMPLE}" ]]; then
  echo "[e2e] sample not found: ${SAMPLE}" >&2
  exit 2
fi

echo "[e2e] checking sample program..."
cargo run -q -p sgc -- check "${SAMPLE}"

echo "[e2e] building llvm ir sample..."
cargo run -q -p sgc -- build "${SAMPLE}" --emit-llvm --output "${LLVM_OUT}"

if [[ ! -f "${LLVM_OUT}" ]]; then
  echo "[e2e] expected LLVM IR output not found: ${LLVM_OUT}" >&2
  exit 1
fi

echo "[e2e] smoke test passed"
