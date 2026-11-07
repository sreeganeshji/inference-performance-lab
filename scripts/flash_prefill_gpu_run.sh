#!/usr/bin/env bash
# Prefill-attention measurements (csrc/flash_prefill_attention_kernel.cu); see docs/flash-prefill.md.
#
#   scripts/flash_prefill_gpu_run.sh test       correctness tests on this GPU
#   scripts/flash_prefill_gpu_run.sh peak       cuBLAS BF16 GEMM throughput (the "% of peak" denominator)
#   scripts/flash_prefill_gpu_run.sh steps      every development step at four shapes
#   scripts/flash_prefill_gpu_run.sh grid       Qwen2.5-7B grid, batch 1-8 x 512-16K, against FA2 and SDPA
#   scripts/flash_prefill_gpu_run.sh accuracy   max abs / rel error per shape against the FP32 reference
#   scripts/flash_prefill_gpu_run.sh ncu        Nsight Compute: final kernel, step 3 and FA2
#   scripts/flash_prefill_gpu_run.sh all        every step in order
#
# Uses the venv from scripts/attention_gpu_run.sh setup. Every GPU step holds an flock on
# $IPL_GPU_LOCK (default ~/.cache/gpu-bench.lock) so concurrent jobs on a shared GPU never overlap.
# Results go to results/data/attention/prefill/; Nsight Compute reports to ~/.cache/ipl-flash-ncu.
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$repo_root"

venv="${IPL_VENV:-$HOME/.venvs/ipl-vllm}"
python="$venv/bin/python"
export PATH="$venv/bin:$PATH"
export PYTHONPATH="$repo_root/src:$repo_root/benchmarks"
export PYTHONDONTWRITEBYTECODE=1

lock="${IPL_GPU_LOCK:-$HOME/.cache/gpu-bench.lock}"
mkdir -p "$(dirname "$lock")"
gpu_tag="${GPU_TAG:-$(nvidia-smi --query-gpu=name --format=csv,noheader | head -1 |
    awk '{name=tolower($0); gsub(/nvidia |geforce /, "", name); gsub(/[^a-z0-9]+/, "", name); print name}')}"
out_dir="results/data/attention/prefill"
ncu_dir="${IPL_NCU_DIR:-$HOME/.cache/ipl-flash-ncu}"
peak="$out_dir/$gpu_tag-bf16-gemm-peak.json"
mkdir -p "$out_dir" "$out_dir/ncu" "$ncu_dir"

gpu() { flock "$lock" "$@"; }

machine() {
    {
        echo "# $(date -u +%Y-%m-%dT%H:%M:%SZ)"
        nvidia-smi --query-gpu=name,driver_version,memory.total,clocks.max.sm,clocks.max.mem,power.limit,pcie.link.gen.max \
            --format=csv
        nvcc --version | tail -2
        ncu --version | tail -1
        "$python" -c "import torch, vllm; p = torch.cuda.get_device_properties(0); print('torch', torch.__version__, 'cuda', torch.version.cuda, 'vllm', vllm.__version__); print('sm', f'{p.major}.{p.minor}', 'SMs', p.multi_processor_count, 'L2 bytes', p.L2_cache_size, 'smem/SM', p.shared_memory_per_multiprocessor)"
        uname -sr
    } | tee "$out_dir/$gpu_tag-machine.txt"
}

run_tests() {
    gpu "$python" -m pytest -p no:cacheprovider -q tests/test_flash_prefill_attention.py
}

measure_peak() {
    gpu "$python" benchmarks/measure_bf16_tensor_peak.py --output "$peak" | tee "$out_dir/$gpu_tag-bf16-gemm-peak.md"
}

steps() {
    gpu "$python" benchmarks/benchmark_flash_prefill_attention.py --peak "$peak" \
        --batch-sizes 1 --seq-lens 512 2048 8192 --trials 9 \
        --implementations lab-simt lab-mma lab-swizzle lab-cp_async lab-fast_exp2 lab-hoisted lab-heavy_first \
        lab-skip_masked vllm-fa2 sdpa-flash \
        --output "$out_dir/$gpu_tag-steps.jsonl" | tee "$out_dir/$gpu_tag-steps.md"
    gpu "$python" benchmarks/benchmark_flash_prefill_attention.py --peak "$peak" \
        --batch-sizes 4 --seq-lens 8192 --trials 9 \
        --implementations lab-simt lab-mma lab-swizzle lab-cp_async lab-fast_exp2 lab-hoisted lab-heavy_first \
        lab-skip_masked vllm-fa2 sdpa-flash \
        --output "$out_dir/$gpu_tag-steps.jsonl" | grep '^| 4 ' | tee -a "$out_dir/$gpu_tag-steps.md"
}

grid() {
    gpu "$python" benchmarks/benchmark_flash_prefill_attention.py --peak "$peak" \
        --batch-sizes 1 2 4 8 --seq-lens 512 1024 2048 4096 8192 16384 --trials 9 \
        --implementations lab-heavy_first lab-skip_masked vllm-fa2 sdpa-flash \
        --output "$out_dir/$gpu_tag-qwen2.5-7b-grid.jsonl" | tee "$out_dir/$gpu_tag-qwen2.5-7b-grid.md"
}

accuracy() {
    gpu "$python" benchmarks/flash_prefill_accuracy.py --output "$out_dir/$gpu_tag-accuracy.jsonl" |
        tee "$out_dir/$gpu_tag-accuracy.md"
}

profile() { # profile <report name> <implementation> <batch> <seq>
    gpu ncu --set full --import-source yes -k 'regex:flash_prefill_mma|flash_fwd_kernel' --launch-skip 1 -c 1 -f \
        -o "$ncu_dir/$1" "$python" benchmarks/profile_flash_prefill_attention.py \
        --implementation "$2" --batch-size "$3" --seq-len "$4"
}

nsight() {
    # Calibration: what "tensor pipe active" reads for cuBLAS's own BF16 GEMM on this GPU.
    gpu ncu --set full -k 'regex:gemm|xmma|cutlass' --launch-skip 1 -c 1 -f -o "$ncu_dir/cublas-8192" \
        "$python" benchmarks/measure_bf16_tensor_peak.py --profile-once 8192 8192 8192

    for shape in "1 8192" "1 1024"; do
        read -r batch seq <<<"$shape"
        tag="${batch}x${seq}"
        profile "lab-cp_async-$tag" lab-cp_async "$batch" "$seq"
        profile "lab-skip_masked-$tag" lab-skip_masked "$batch" "$seq"
        profile "vllm-fa2-$tag" vllm-fa2 "$batch" "$seq"
        "$python" benchmarks/summarize_ncu_prefill.py "step-3-cp_async=$ncu_dir/lab-cp_async-$tag.ncu-rep" \
            "final-skip_masked=$ncu_dir/lab-skip_masked-$tag.ncu-rep" "vllm-fa2=$ncu_dir/vllm-fa2-$tag.ncu-rep" |
            tee "$out_dir/ncu/$gpu_tag-$tag-summary.md"

        for name in lab-skip_masked vllm-fa2; do
            ncu --import "$ncu_dir/$name-$tag.ncu-rep" --page details >"$out_dir/ncu/$gpu_tag-$tag-$name-details.txt"
        done
    done

    # Every tensor-core step at 1x8192, for the decision journal.
    for name in lab-mma lab-swizzle lab-hoisted; do
        profile "$name-1x8192" "$name" 1 8192
    done

    "$python" benchmarks/summarize_ncu_prefill.py "1-mma=$ncu_dir/lab-mma-1x8192.ncu-rep" \
        "2-swizzle=$ncu_dir/lab-swizzle-1x8192.ncu-rep" "3-cp_async=$ncu_dir/lab-cp_async-1x8192.ncu-rep" \
        "4b-hoisted=$ncu_dir/lab-hoisted-1x8192.ncu-rep" "6-skip_masked=$ncu_dir/lab-skip_masked-1x8192.ncu-rep" \
        "vllm-fa2=$ncu_dir/vllm-fa2-1x8192.ncu-rep" | tee "$out_dir/ncu/$gpu_tag-1x8192-steps-summary.md"

    "$python" benchmarks/summarize_ncu_prefill.py "cublas-gemm-8192=$ncu_dir/cublas-8192.ncu-rep" \
        "final-skip_masked-1x8192=$ncu_dir/lab-skip_masked-1x8192.ncu-rep" |
        tee "$out_dir/ncu/$gpu_tag-cublas-calibration-summary.md"
}

case "${1:-all}" in
    test) run_tests ;;
    machine) machine ;;
    peak) measure_peak ;;
    steps) steps ;;
    grid) grid ;;
    accuracy) accuracy ;;
    ncu) nsight ;;
    all) machine && run_tests && measure_peak && steps && grid && accuracy && nsight ;;
    *) echo "usage: $0 {test|machine|peak|steps|grid|accuracy|ncu|all}" >&2; exit 2 ;;
esac
