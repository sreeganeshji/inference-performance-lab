#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$repo_root"

if [[ "$(stat -f -c %T "$repo_root")" == nfs* ]]; then
    export UV_LINK_MODE="${UV_LINK_MODE:-copy}"
fi

if ! command -v uv >/dev/null 2>&1; then
    echo "error: uv is required: https://docs.astral.sh/uv/" >&2
    exit 1
fi

uv python install
uv sync --locked

echo "Bootstrap complete."
echo "Run project commands with: uv run --locked <command>"
