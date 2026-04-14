#!/bin/bash
# ============================================================================
# Example: Run FastLex on the English Lexicon Project (~35,500 forms).
#
# ELP (Balota et al., 2007) is a large-scale database of lexical decision and
# naming data for English. This subset includes ARPABET transcriptions.
#
# NOTE: You must supply your own copy of the ELP lexicon. Place the CSV in
# data/ with columns "Word" (orthography) and "Pron_arpabet"
# (space-delimited ARPABET).
#
# Expected runtime: ~6.5 minutes with full pipeline, ~3 minutes without LED.
# ============================================================================

# Run from the fastlex-github directory.

python fastlex.py \
  --lexicon-path data/ELP_lex_min_missing_added.csv \
  --output-path output/elp_flex.csv \
  --orth-col Word \
  --pron-col Pron_arpabet \
  --delimiter-phono space \
  --delimiter-ortho none \
  --n-jobs 8 \
  --progress \
  --oldk 20 \
  --pldk 20 \
  --old-count 5 --old-parts \
  --pld-count 5 --pld-parts
