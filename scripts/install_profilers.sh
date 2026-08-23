#!/usr/bin/env bash
set -euo pipefail

nsys_package="${NSYS_PACKAGE:-nsight-systems-2026.1.3}"
ncu_package="${NCU_PACKAGE:-nsight-compute-2025.4.1}"

if ! command -v apt-get >/dev/null 2>&1; then
    echo "error: this installer requires an apt-based system" >&2
    exit 1
fi

if ((EUID == 0)); then
    as_root=()
elif command -v sudo >/dev/null 2>&1; then
    as_root=(sudo)
else
    echo "error: root privileges or sudo are required" >&2
    exit 1
fi

package_is_installed() {
    dpkg-query -W -f='${Status}' "$1" 2>/dev/null |
        grep -qx 'install ok installed'
}

packages=()

if ! package_is_installed "$nsys_package"; then
    packages+=("$nsys_package")
fi

if ! package_is_installed "$ncu_package"; then
    packages+=("$ncu_package")
fi

if ((${#packages[@]})); then
    "${as_root[@]}" apt-get update
    "${as_root[@]}" apt-get install -y "${packages[@]}"
fi

ensure_on_path() {
    local tool="$1"
    local package="$2"
    local source_path
    local target_path="/usr/local/bin/$tool"

    if command -v "$tool" >/dev/null 2>&1; then
        return
    fi

    source_path="$(
        dpkg -L "$package" |
            awk -v tool="$tool" '
                index($0, "/target/") == 0 &&
                $0 ~ ("/" tool "$") &&
                !found {
                    value = $0
                    found = 1
                }
                END { print value }
            '
    )"

    if [[ -z "$source_path" || ! -x "$source_path" ]]; then
        echo "error: could not locate $tool in $package" >&2
        exit 1
    fi

    if [[ -e "$target_path" && ! -L "$target_path" ]]; then
        echo "error: $target_path exists and is not a symlink" >&2
        exit 1
    fi

    "${as_root[@]}" ln -sfn "$source_path" "$target_path"
}

ensure_on_path nsys "$nsys_package"
ensure_on_path ncu "$ncu_package"

nsys --version
ncu --version

echo
echo "Profiler installation complete."
echo "Note: Nsight Compute hardware counters may require sudo."