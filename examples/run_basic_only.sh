#!/bin/bash
# ============================================================================
# Example: Run FastLex with ONLY basic neighbor metrics (no OLD/PLD, no LED).
#
# This is much faster than the full pipeline. Useful when you only need
# conventional neighbor counts (DAS neighbors, cohorts, nohorts, homoforms,
# uniqueness points) and do not need edit-distance summaries.
#
# This example uses the ELP lexicon, but the same approach works for any
# lexicon. Just omit the --oldk, --pldk, --old-count, and --pld-count flags.
# ============================================================================

# Run from the fastlex-github directory.

python fastlex.py \
  --lexicon-path data/ELP_lex_min_missing_added.csv \
  --output-path output/elp_basic_only.csv \
  --orth-col Word \
  --pron-col Pron_arpabet \
  --delimiter-phono space \
  --delimiter-ortho none \
  --progress
