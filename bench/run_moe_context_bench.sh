#!/usr/bin/env bash
set -euo pipefail

ROOT=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)
cd "$ROOT"
source "$ROOT/setup/env.sh"

exec python "$ROOT/bench/moe_context_bench.py" "$@"