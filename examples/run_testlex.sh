#!/bin/bash
# ============================================================================
# Example: Run FastLex on the included test lexicon (18 words).
#
# This is a good sanity check to make sure everything is working.
# It should complete in under a second.
# ============================================================================

# Run from the fastlex-github directory.

python fastlex.py \
  --lexicon-path data/testlex.csv \
  --output-path output/testlex_flex.csv \
  --orth-col Item \
  --pron-col Pronunciation \
  --delimiter-phono space \
  --delimiter-ortho none \
  --n-jobs 1 \
  --progress \
  --oldk 5 \
  --pldk 5 \
  --old-count 3 --old-parts \
  --pld-count 3 --pld-parts
