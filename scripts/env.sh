#!/usr/bin/env bash

_inference_lab_configure_env() {
    local repo_root
    local cache_root

    repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
    cache_root="${INFERENCE_LAB_CACHE_ROOT:-$(dirname "$repo_root")/huggingface}"

    export HF_HUB_CACHE="${HF_HUB_CACHE:-$cache_root/hub}"
    export HF_XET_CACHE="${HF_XET_CACHE:-$cache_root/xet}"

    mkdir -p "$HF_HUB_CACHE" "$HF_XET_CACHE"
}

_inference_lab_configure_env
unset -f _inference_lab_configure_env