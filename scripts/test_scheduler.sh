#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$repo_root"

for sanitizer in address,undefined thread; do
    build_dir="build/scheduler-${sanitizer%%,*}"
    cmake -S scheduler -B "$build_dir" -G Ninja -DCMAKE_BUILD_TYPE=Debug -DSANITIZE="$sanitizer" >/dev/null
    cmake --build "$build_dir" >/dev/null
    ctest --test-dir "$build_dir" --output-on-failure
done
