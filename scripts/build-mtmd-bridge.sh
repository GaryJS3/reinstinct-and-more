#!/usr/bin/env bash
set -euo pipefail

root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
build_dir="${REINSTINCT_MTMD_BUILD_DIR:-$root/build/mtmd-bridge}"

git -C "$root" submodule update --init --recursive third_party/llama.cpp
cmake -S "$root/mtmd-bridge" -B "$build_dir" \
  -DCMAKE_BUILD_TYPE=Release \
  -DGGML_HIP=ON
cmake --build "$build_dir" --parallel
printf 'Built %s\n' "$build_dir/libreinstinct_mtmd.so"
