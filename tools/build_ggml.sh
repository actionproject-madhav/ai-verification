#!/bin/bash
# Build the reference encoder (libggml-base, from llama.cpp) the four ways lineage_check.derive tries, on x86-64 Linux.
# The builds differ only in whether the compiler fuses a multiply and an add, which decides some roundings.
#   tools/build_ggml.sh <dir>                      prints the four library paths, one per line
#   export LINEAGE_GGML=$(tools/build_ggml.sh <dir> | paste -sd, -)
# Needs git, cmake, gcc and clang. The commit is the one the census in docs/RESULTS.md was measured with.
set -e
DIR=${1:-$HOME/ggml}
COMMIT=dfdfca22bce36f66b4cfa3e6c96f0c4d92dac0fe
mkdir -p "$DIR" && cd "$DIR"
[ -d llama.cpp ] || git clone -q https://github.com/ggml-org/llama.cpp
cd llama.cpp && git fetch -q origin && git checkout -q $COMMIT
build() {
  name=$1 cc=$2 cxx=$3; shift 3
  [ -f build-$name/bin/libggml-base.so ] || {
    cmake -S . -B build-$name -DBUILD_SHARED_LIBS=ON -DLLAMA_CURL=OFF -DCMAKE_BUILD_TYPE=Release -DGGML_NATIVE=OFF \
      -DCMAKE_C_COMPILER=$cc -DCMAKE_CXX_COMPILER=$cxx "-DCMAKE_C_FLAGS=$*" "-DCMAKE_CXX_FLAGS=$*" > build-$name.log 2>&1
    cmake --build build-$name --target ggml-base -j 8 >> build-$name.log 2>&1; }
  echo "$PWD/build-$name/bin/libggml-base.so"
}
build plain gcc g++ -ffp-contract=off          # no fused multiply-add anywhere
build fma-chains gcc g++ -march=haswell         # gcc, fused where the Haswell tuning chooses
build fma gcc g++ -mfma -mavx2 -ffp-contract=fast
build clang-fma clang clang++ -mfma -mavx2      # clang fuses within an expression only
