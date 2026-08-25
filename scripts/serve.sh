#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
source "$repo_root/scripts/env.sh"
cd "$repo_root"

model="Qwen/Qwen2.5-7B-Instruct"
revision="a09a35458c702b33eeacc393d103063234e8bc28"
fused_rmsnorm_provider="${FUSED_RMSNORM_PROVIDER:-vllm_c}"

case "$fused_rmsnorm_provider" in
    vllm_c)
        # An empty allowlist prevents our general plugin from loading.
        export VLLM_PLUGINS=""
        ;;
    inference_lab)
        export VLLM_PLUGINS="inference_performance_lab"
        ;;
    *)
        echo "error: unsupported FUSED_RMSNORM_PROVIDER=$fused_rmsnorm_provider" >&2
        exit 2
        ;;
esac

ir_op_priority="$(
    printf '{"fused_add_rms_norm":["%s"]}' \
        "$fused_rmsnorm_provider"
)"

echo "Fused RMSNorm provider: $fused_rmsnorm_provider"

exec uv run --locked --no-sync vllm serve "$model" \
    --revision "$revision" \
    --served-model-name qwen2.5-7b-instruct \
    --dtype bfloat16 \
    --max-model-len 4096 \
    --gpu-memory-utilization 0.90 \
    --no-enable-prefix-caching \
    --generation-config vllm \
    --seed 0 \
    --ir-op-priority "$ir_op_priority" \
    --host 127.0.0.1 \
    --port "${VLLM_PORT:-8000}"