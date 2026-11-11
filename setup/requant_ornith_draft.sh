#!/bin/bash
# Convert the BF16 Ornith DFlash2 draft to EXL3 with the pinned ExLlamaV3 converter. Uses the GPU.
#
#   setup/requant_ornith_draft.sh            3.0 bpw -> models/Ornith-1.5-35B-A3B-DFlash2-EXL3-3.0bpw
#   setup/requant_ornith_draft.sh --dry-run  print the command without converting
#
# Source: jzinno/Ornith-1.5-35B-A3B-DFlash2, revision 9b4852c05fd00b672b7434b1bb105bc03c8682b0.
set -euo pipefail
ROOT=$(cd "$(dirname "$0")/.." && pwd)
EXLLAMAV3_DIR=${EXLLAMAV3_DIR:-$ROOT/exllamav3}
MODELS_DIR=${MODELS_DIR:-models}
[[ "$MODELS_DIR" = /* ]] || MODELS_DIR="$ROOT/$MODELS_DIR"
SOURCE=${SOURCE:-$MODELS_DIR/Ornith-1.5-35B-A3B-DFlash2-bf16}
OUTPUT=${OUTPUT:-$MODELS_DIR/Ornith-1.5-35B-A3B-DFlash2-EXL3-3.0bpw}
WORK=${WORK:-$ROOT/requant_work/ornith-dflash2-3.0bpw}

dry_run=false
[ "$#" -le 1 ] || { echo "usage: $0 [--dry-run]" >&2; exit 2; }
case "${1:-}" in
    "") ;;
    --dry-run) dry_run=true ;;
    *) echo "usage: $0 [--dry-run]" >&2; exit 2 ;;
esac

if $dry_run; then
    printf 'source %q\n' "$ROOT/setup/env.sh"
    printf 'python %q -i %q -o %q -w %q -b 3.0 --out_scales always -cb mul1 -d 0\n' \
        "$EXLLAMAV3_DIR/convert.py" "$SOURCE" "$OUTPUT" "$WORK"
    exit 0
fi

[ -f "$SOURCE/config.json" ] && [ -f "$SOURCE/model.safetensors" ] || {
    echo "missing BF16 draft in $SOURCE; run setup/download_models.sh ornith-dflash2-bf16" >&2; exit 1; }
[ -f "$EXLLAMAV3_DIR/convert.py" ] || { echo "missing $EXLLAMAV3_DIR/convert.py; run setup/build_exllamav3.sh" >&2; exit 1; }
python3 - "$SOURCE/config.json" <<'PY'
import json
import sys

with open(sys.argv[1], encoding="utf-8") as config_file:
    config = json.load(config_file)

if config.get("architectures") != ["DFlash2DraftModel"]:
    raise SystemExit("expected a DFlash2DraftModel checkpoint")
if config.get("hidden_size") != 2048 or config.get("num_target_layers") != 40:
    raise SystemExit("checkpoint metadata does not match the Ornith-1.5 2048-wide, 40-layer target")
if not config.get("dflash_config", {}).get("target_layer_ids"):
    raise SystemExit("checkpoint is missing DFlash2 target-layer metadata")
print("validated Ornith DFlash2 metadata (hidden size 2048, 40 target layers)")
PY

[ ! -e "$OUTPUT" ] && [ ! -L "$OUTPUT" ] && [ ! -e "$WORK" ] && [ ! -L "$WORK" ] || {
    echo "$OUTPUT or $WORK already exists; inspect it before choosing a new output/work path" >&2; exit 1; }

source "$ROOT/setup/env.sh"
mkdir -p "$(dirname "$WORK")"
python "$EXLLAMAV3_DIR/convert.py" -i "$SOURCE" -o "$OUTPUT" -w "$WORK" -b 3.0 \
    --out_scales always -cb mul1 -d 0
echo "draft written to $OUTPUT (you can delete $WORK after verifying the output)"