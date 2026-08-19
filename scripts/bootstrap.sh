#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$repo_root"

if [[ "$(stat -f -c %T "$repo_root")" == nfs* ]]; then
    export UV_LINK_MODE="${UV_LINK_MODE:-copy}"
fi

if ! command -v uv >/dev/null 2>&1; then
    echo "uv not found; installing it..."
    curl -LsSf https://astral.sh/uv/install.sh | sh
    export PATH="$HOME/.local/bin:$PATH"
fi

uv --version

if [[ -x /usr/bin/python3.12 ]]; then
    python_bin=/usr/bin/python3.12
else
    uv python install 3.12
    python_bin="$(uv python find 3.12)"
fi

if [[ ! -x .venv/bin/python ]]; then
    echo "Creating or replacing unusable virtual environment..."
    uv venv --clear --python "$python_bin" .venv
fi

uv sync --locked --python "$python_bin"

if ! uv run --locked --no-sync python -c "import torch, vllm"; then
    echo "Environment payload is inconsistent; reinstalling locked packages..."
    uv sync --locked --python "$python_bin" --reinstall
fi

echo "Bootstrap complete."
echo "Run project commands with: uv run --locked --no-sync <command>"