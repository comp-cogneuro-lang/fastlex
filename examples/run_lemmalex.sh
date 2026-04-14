#!/bin/bash
# ============================================================================
# Example: Run FastLex on LemmaLex (~17,700 English lemma forms).
#
# LemmaLex is the base lexicon from LexFindR (Li, Crinnion, & Magnuson, 2022).
# It includes orthography and ARPABET phonemic transcriptions.
#
# NOTE: You must supply your own copy of lemmalex. You can export it from the
# LexFindR R package, or contact the authors. Place the CSV in data/ with
# columns "Item" (orthography) and "Pronunciation" (space-delimited ARPABET).
#
# Expected runtime: ~2 minutes with full pipeline, ~1 minute without LED.
# ============================================================================

# Run from the fastlex-github directory.

python fastlex.py \
  --lexicon-path data/lemmalex_min.csv \
  --output-path output/lemmalex_flex.csv \
  --orth-col Item \
  --pron-col Pronunciation \
  --delimiter-phono space \
  --delimiter-ortho none \
  --n-jobs 8 \
  --progress \
  --oldk 20 \
  --pldk 20 \
  --old-count 5 --old-parts \
  --pld-count 5 --pld-parts
