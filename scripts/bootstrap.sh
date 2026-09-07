#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$repo_root"

profile=full

if (($# > 1)); then
    echo "Usage: $0 [--kernels-only]" >&2
    exit 2
fi

case "${1:-}" in
    "")
        ;;
    --kernels-only)
        profile=kernels
        ;;
    -h | --help)
        echo "Usage: $0 [--kernels-only]"
        echo
        echo "Without arguments, install the full vLLM serving environment."
        echo "Use --kernels-only for PyTorch, CUDA extension builds, and kernel tests."
        exit 0
        ;;
    *)
        echo "error: unsupported argument: $1" >&2
        echo "Usage: $0 [--kernels-only]" >&2
        exit 2
        ;;
esac

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

sync_args=(sync --locked --python "$python_bin")
if [[ "$profile" == full ]]; then
    sync_args+=(--extra serving)
fi

uv "${sync_args[@]}"

required_imports=(torch)
if [[ "$profile" == full ]]; then
    required_imports+=(vllm)
fi

import_check="$(IFS=,; echo "${required_imports[*]}")"
if ! uv run --locked --no-sync python -c "import ${import_check}"; then
    echo "Environment payload is inconsistent; reinstalling locked packages..."
    uv "${sync_args[@]}" --reinstall
fi

echo "Bootstrap complete ($profile profile)."
echo "Run project commands with: uv run --locked --no-sync <command>"
