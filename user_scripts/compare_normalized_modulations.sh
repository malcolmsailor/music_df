DIR1="/Users/malcolm/datasets/chord_tones/salami_slice_dedoubled_no_suspensions_q16"

cd "$DIR1"
FILENAME=$(fd --extension csv . | fzf)
F1="$DIR1/$FILENAME"
DIR2="/Users/malcolm/output/normalize_modulations"
F2="$DIR2/$FILENAME"

OUTPUT_DIR="/Users/malcolm/output/normalized_modulations_comparisons"
OUTPUT_FILE="$OUTPUT_DIR/${FILENAME%.csv}.pdf"

mkdir -p $(dirname "$OUTPUT_FILE")

python /Users/malcolm/google_drive/python/malmus/music_df/main/music_df/scripts/compare_annotations.py \
    input_file1="$F1" input_file2="$F2" output_file="$OUTPUT_FILE"
