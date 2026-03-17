demo-transform-chain:
    python tests/demos/demo_transforms.py ~/datasets/ClassicalMusicArchivesClean \
        --max-files 20 \
        ~/tmp/transform_chain_out \
        --transforms 'merge_repeated_notes,split_notes_at_barlines,dedouble_instruments' \
        --params temp_transform_params.yaml
#        --transforms 'merge_repeated_notes,split_notes_at_barlines,dedouble_instruments,dedouble_octaves,dedouble_octaves_within' \
