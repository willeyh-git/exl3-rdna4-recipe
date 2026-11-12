#!/bin/bash
ROOT=$(cd "$(dirname "$0")" && pwd)
exec "$ROOT/run_tabby.sh" --config "$ROOT/config.gemma4.yml" "$@"