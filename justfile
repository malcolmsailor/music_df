demo-transform-chain *ARGS:
    python tests/demos/demo_transforms.py ~/datasets/ClassicalMusicArchivesClean \
        --max-files 20 \
        ~/tmp/transform_chain_out \
        --transforms 'merge_repeated_notes,split_notes_at_barlines,dedouble_unisons_across_instruments,dedouble_octaves,dedouble_octaves_within_instrument' \
        --params temp_transform_params.yaml \
        {{ARGS}}
#        --transforms 'merge_repeated_notes,split_notes_at_barlines,dedouble_unisons_across_instruments,dedouble_octaves,dedouble_octaves_within' \
