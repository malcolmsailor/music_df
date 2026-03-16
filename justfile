demo-transform-chain:
    python tests/demos/demo_chain.py ~/datasets/ClassicalMusicArchivesClean \
        --max-files 20 \
        --transforms 'merge_repeated_notes,split_notes_at_barlines,dedouble_instruments,dedouble_octaves,dedouble_octaves_within' \
        ~/tmp/transform_chain_out
