#!/bin/bash
# ============================================================================
# Example: Run FastLex on EsPal (~278,000 Spanish forms).
#
# EsPal (Duchon et al., 2013) is a comprehensive Spanish lexical resource.
# Because Spanish has rich morphology, EsPal is much larger than ELP and
# provides a demanding test of FastLex's scalability.
#
# NOTE: You must supply your own copy of the EsPal lexicon. Place the CSV in
# data/ with columns "word" (orthography) and "es_phon_structure"
# (phonological form, null-delimited).
#
# WARNING: This is a large job. Using 8+ cores will tax your machine. Expect
# ~250 minutes for the full pipeline, ~102 minutes without LED counts.
# ============================================================================

# Run from the fastlex-github directory.

python fastlex.py \
  --lexicon-path data/espal-lex.csv \
  --output-path output/espal_flex.csv \
  --orth-col word \
  --pron-col es_phon_structure \
  --delimiter-phono none \
  --delimiter-ortho none \
  --n-jobs 8 \
  --progress \
  --oldk 20 \
  --pldk 20 \
  --old-count 5 --old-parts \
  --pld-count 5 --pld-parts
