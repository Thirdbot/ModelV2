"""Grounded seismic VLM — the main model.

    data/       dataset access + config
    model/      encoder · detector · decoder · narrator · geology adapter
    train/      stage1_geology · stage2_detector · stage3_narrator · train (runner)
    test/       evaluate (held-out copy score + faithfulness swap)
    inference/  infer (narrate from facts)

The model: image -> frozen NCS encoder -> detector facts (class-driven
measurement) -> digit-token bridge -> Qwen decoder (frozen geology adapter +
trainable grounding adapter) -> grounded language that copies the exact numbers,
plus <SEG> -> mask decoder for display masks.
"""
