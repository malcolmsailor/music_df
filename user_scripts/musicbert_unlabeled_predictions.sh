#!/bin/bash

run_name=$1
split=$2
if [[ -z $split ]]; then
    echo 'Usage: bash musicbert_unlabeled_predictions.sh [run_name] [split] <other args to pass through>'
    exit 1
fi

shift
shift

# Get the directory of the current script
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Get the parent directory
PARENT_DIR="$(dirname "$SCRIPT_DIR")"

PREDICTIONS_BASE=/Users/malcolm/output/musicbert_unlabeled/saved_predictions/"${run_name}"
OUTPUT_FOLDER=/Users/malcolm/output/plot_unlabeled_predictions/"${run_name}"
set -x
python "${PARENT_DIR}"/music_df/scripts/plot_predictions.py \
    --config-file "${PARENT_DIR}"/music_df/scripts/configs/plot_unlabeled_predictions1.yaml \
    metadata="$PREDICTIONS_BASE"/metadata_test.txt \
    predictions="$PREDICTIONS_BASE"/"${split}"/predictions \
    output_folder="$OUTPUT_FOLDER" \
    "${@}"
set +x
