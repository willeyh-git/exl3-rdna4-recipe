#!/bin/bash
# Download the models into models/ (or MODELS_DIR; the 3.0 bpw draft into DRAFT_DIR, default the same). Uses the `hf`
# CLI from huggingface_hub (in the venv).
#
#   setup/download_models.sh            main model (3.0 bpw self-calibrated, ~13.4 GB) + the 3.0 bpw DFlash2 draft (0.9 GB)
#   setup/download_models.sh main       main model only
#   setup/download_models.sh draft      the 3.0 bpw DFlash2 draft only (our requant of inco.ai's draft, on Hugging Face)
#   setup/download_models.sh 3.0bpw-plain  turboderp's plain 3.0 bpw quant, the default until 2026-10-09 (13.8 GB)
#   setup/download_models.sh draft-bf16 bf16 DFlash2 draft, to make the 3.0 bpw draft yourself (setup/requant_draft.sh)
#   setup/download_models.sh draft-mia  ready-made 5.0 bpw DFlash2 draft by Mia-AiLab (1.5 GB, no requant needed,
#                                       but it leaves less room for context)
#   setup/download_models.sh 2.5bpw     the 2.5 bpw main model, only for the GPQA/perplexity comparison
#   setup/download_models.sh ornith     Ornith-1.5-35B-A3B EXL3 4.00 bpw (24 GB)
# In a terminal, existing model directories can be discovered under a supplied models directory and linked into place.
# Non-interactive runs keep downloading as before.
set -euo pipefail
ROOT=$(cd "$(dirname "$0")/.." && pwd)
MODELS_DIR=${MODELS_DIR:-models}
DRAFT_DIR=${DRAFT_DIR:-$MODELS_DIR}
VENV=${VENV:-$ROOT/.venv}
[ -x "$VENV/bin/hf" ] && export PATH="$VENV/bin:$PATH"
cd "$ROOT"

DISCOVERED_MODELS=()

is_model_dir() {
    local model_path=$1 weight
    [ -f "$model_path/config.json" ] || return 1
    for weight in "$model_path"/*.safetensors; do
        [ -f "$weight" ] && return 0
    done
    return 1
}

discover_model_dirs() {
    local search_dir=$1 candidate found
    DISCOVERED_MODELS=()
    search_dir=$(cd "$search_dir" && pwd -P) || return 1

    if is_model_dir "$search_dir"; then
        DISCOVERED_MODELS+=("$search_dir")
    fi
    for candidate in "$search_dir"/*; do
        [ -d "$candidate" ] || continue
        is_model_dir "$candidate" || continue
        candidate=$(cd "$candidate" && pwd -P) || continue
        if [ "${#DISCOVERED_MODELS[@]}" -gt 0 ]; then
            for found in "${DISCOVERED_MODELS[@]}"; do
                [ "$found" = "$candidate" ] && continue 2
            done
        fi
        DISCOVERED_MODELS+=("$candidate")
    done
}

init_model_discovery() {
    [ -t 0 ] || return 0
    local entered search_dir=$MODELS_DIR index
    read -r -p "Directory to search for existing models [$MODELS_DIR]: " entered || return 0
    [ -z "$entered" ] || search_dir=$entered
    case "$search_dir" in
        "~") search_dir=$HOME ;;
        "~/"*) search_dir="$HOME/${search_dir:2}" ;;
    esac
    [[ "$search_dir" = /* ]] || search_dir="$ROOT/$search_dir"
    if [ ! -d "$search_dir" ]; then
        if [ -n "$entered" ]; then
            echo "Models directory does not exist: $search_dir" >&2
            return 1
        fi
        echo "Models directory does not exist yet; continuing with downloads: $search_dir"
        return 0
    fi
    discover_model_dirs "$search_dir"
    if [ "${#DISCOVERED_MODELS[@]}" -eq 0 ]; then
        echo "No model directories containing config.json and safetensors weights found in $search_dir or its immediate subdirectories."
        return 0
    fi

    printf 'Found existing model directories:\n'
    for index in "${!DISCOVERED_MODELS[@]}"; do
        printf '  %d) %s\n' "$((index + 1))" "${DISCOVERED_MODELS[$index]}"
    done
}

link_existing_model() {
    local model_path=$1 target=$2 target_parent target_path
    is_model_dir "$model_path" || {
        echo "Expected config.json and at least one .safetensors weight in $model_path" >&2
        return 1
    }
    mkdir -p "$(dirname "$target")" || { echo "Could not create the model directory." >&2; return 1; }
    target_parent=$(cd "$(dirname "$target")" && pwd -P) || {
        echo "Could not resolve the model directory: $target" >&2
        return 1
    }
    target_path="$target_parent/$(basename "$target")"
    if [ "$model_path" = "$target_path" ]; then
        echo "Model is already at $target"
        return 0
    fi
    if [ -e "$target" ] || [ -L "$target" ]; then
        echo "Refusing to replace existing model path: $target" >&2
        return 1
    fi
    ln -s "$model_path" "$target" || { echo "Could not link $model_path to $target" >&2; return 1; }
    echo "Using existing model: $target -> $model_path"
}

use_existing_model() {
    local target=$1 selection selected_number
    if [ -L "$target" ]; then
        if is_model_dir "$target"; then
            echo "Using existing model at $target"
            return 0
        fi
        echo "Existing model link is incomplete: $target" >&2
        exit 1
    fi
    [ -t 0 ] && [ "${#DISCOVERED_MODELS[@]}" -gt 0 ] || return 1

    printf 'Choose an existing directory for %s (0 downloads it):\n' "$(basename "$target")"
    for selection in "${!DISCOVERED_MODELS[@]}"; do
        printf '  %d) %s\n' "$((selection + 1))" "${DISCOVERED_MODELS[$selection]}"
    done
    while true; do
        read -r -p "Selection [0]: " selection || return 1
        [ -n "$selection" ] || return 1
        if [[ "$selection" =~ ^[0-9]+$ ]]; then
            selected_number=$((10#$selection))
            [ "$selected_number" -eq 0 ] && return 1
            if [ "$selected_number" -le "${#DISCOVERED_MODELS[@]}" ]; then
                link_existing_model "${DISCOVERED_MODELS[$((selected_number - 1))]}" "$target" || exit 1
                return 0
            fi
        fi
        printf 'Enter 0 to download or a number from 1 to %d.\n' "${#DISCOVERED_MODELS[@]}"
    done
}

download_from_hf() {
    local repo=$1 revision=$2 target=$3
    command -v hf >/dev/null || {
        echo "hf CLI not found; run setup/make_venv.sh first or pip install huggingface_hub" >&2
        exit 1
    }
    if [ -n "$revision" ]; then
        hf download "$repo" --revision "$revision" --local-dir "$target"
    else
        hf download "$repo" --local-dir "$target"
    fi
}

# fetch REPO REVISION PATH: download, then write .download-complete (the revision) only after it finishes.
# Rerunning resumes and only fetches what's missing.
fetch() {
    local repo=$1 revision=$2 target=$3
    if use_existing_model "$target"; then
        return 0
    fi
    download_from_hf "$repo" "$revision" "$target"
    echo "$repo@$revision" > "$target/.download-complete"
}
download_model() {
    local repo=$1 revision=$2 target=$3
    if use_existing_model "$target"; then
        return 0
    fi
    download_from_hf "$repo" "$revision" "$target"
}
main() {
    # Branch SC_3.00bpw_H4 of turboderp's repo: self-calibrated 3.0 bpw with a 4-bit head (pinned to the tested commit)
    fetch turboderp/Qwen3.8-27B-exl3 86b95530f9d365d90dd24d01a997a8de0cdf6c55 "$MODELS_DIR/Qwen3.8-27B-EXL3-SC3.0bpw-H4"
}
main_plain() {
    # Branch 3.00bpw (commit 6fe61ad620abfe97c5b49f9722c2bceeea4ccc28 when tested)
    download_model turboderp/Qwen3.8-27B-exl3 3.00bpw "$MODELS_DIR/Qwen3.8-27B-EXL3-3.0bpw"
}
draft() {
    # setup/requant_draft.sh's output for incoai/Qwen3.8-27B-DFlash2 at 015e795, byte-identical to what we run
    fetch parth-thakre/Qwen3.8-27B-DFlash2-EXL3-3.0bpw 3d915b126ceecdb3ce41587ca19faa782a91fee0 "$DRAFT_DIR/Qwen3.8-27B-DFlash2-EXL3-3.0bpw"
}
draft_bf16() {
    # The DFlash2 draft (inco.ai / z-lab, Apache-2.0; mirrored at z-lab/Qwen3.8-27B-DFlash2), at the revision we converted
    download_model incoai/Qwen3.8-27B-DFlash2 015e795645c74b1a0eeef3b570031fb62e769bc5 \
        "$MODELS_DIR/Qwen3.8-27B-DFlash2-bf16"
}
draft_mia() {
    download_model Mia-AiLab/Qwen3.8-27B-DFlash2-EXL3-5.0bpw "" \
        "$MODELS_DIR/Qwen3.8-27B-DFlash2-EXL3-5.0bpw"
}
main_25() {
    # Branch 2.50bpw (commit 0cd1091272291d008e06976acb81e25946cdc767 when tested)
    download_model turboderp/Qwen3.8-27B-exl3 2.50bpw "$MODELS_DIR/Qwen3.8-27B-EXL3-2.5bpw"
}
ornith() {
    fetch JRamirez-UAB/Ornith-1.5-35B-A3B-EXL3-4.00bpw-24GB ac1d94ac17373eaea6818cd815591d003177916b \
        "$MODELS_DIR/Ornith-1.5-35B-A3B-EXL3-4.00bpw-24GB"
}

action=${1:-default}
case "$action" in
    default|main|draft|3.0bpw-plain|draft-bf16|draft-mia|2.5bpw|ornith) ;;
    *) echo "usage: $0 [main|draft|3.0bpw-plain|draft-bf16|draft-mia|2.5bpw|ornith]" >&2; exit 2 ;;
esac
init_model_discovery
case "$action" in
    default)       main; draft ;;
    main)          main ;;
    draft)         draft ;;
    3.0bpw-plain)  main_plain ;;
    draft-bf16)    draft_bf16 ;;
    draft-mia)     draft_mia ;;
    2.5bpw)        main_25 ;;
    ornith)        ornith ;;
esac
