#!/usr/bin/env bash
#
# Wraps ./build/profile in `ncu` and writes a .ncu-rep into `out/`
#
# Usage:
#   scripts/ncu.sh [case] [iters]
#   NCU_SET={full|basic} (default full), NCU=<ncu binary>

set -euo pipefail

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
repo_root="$(cd "$script_dir/.." && pwd)"
profile_bin="$repo_root/build/profile"
out_dir="$repo_root/out"

NCU="${NCU:-ncu}"
NCU_SET="${NCU_SET:-full}"

log() { printf '\033[1;34m==>\033[0m %s\n' "$*" >&2; }
die() { printf '\033[1;31merror:\033[0m %s\n' "$*" >&2; exit 1; }

command -v "$NCU" >/dev/null || die "ncu not found (set \$NCU or add it to PATH)."
if [[ ! -x "$profile_bin" ]]; then
  log "profile not built; building target 'profile'..."
  cmake --build "$repo_root/build" -j --target profile \
    || die "build failed; configure first: cmake -S '$repo_root' -B '$repo_root/build'"
fi
mkdir -p "$out_dir"

run_one() {
  local case="$1" iters="${2:-}"
  local report="$out_dir/profile_${case}"
  log "profiling '$case' (iters=${iters:-default}), set=$NCU_SET -> ${report}.ncu-rep"
  # Omit the iters arg entirely when unset, so profile uses the example's depth
  "$NCU" \
    --set "$NCU_SET" \
    --nvtx \
    --target-processes all \
    --kernel-name-base demangled \
    -f -o "$report" \
    "$profile_bin" "$case" ${iters:+"$iters"}
}

if [[ $# -ge 1 ]]; then
  run_one "$1" "${2:-}"
else
  run_one plant 11
  run_one dragon 24
fi

log "done. GUI:  ncu-ui out/profile_<case>.ncu-rep"
log "CLI: ncu --import out/profile_<case>.ncu-rep --page details | less"
