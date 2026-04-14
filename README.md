# FastLex

**A fast Python tool for computing lexical neighborhood metrics.**

---

## What is FastLex?

FastLex computes a suite of lexical neighborhood metrics from a tabular lexicon (a CSV file with orthographic forms, pronunciations, or both). It was created as a fast, flexible alternative to the R package [LexFindR](https://github.com/maglab-uconn/LexFindR) (Li, Crinnion, & Magnuson, 2021). LexFindR is well-tested and easy to use, but its R-native implementation becomes slow for large lexica and expensive metrics like OLD-K. FastLex uses hash-based indexing, BK-trees, and multiprocessing to dramatically reduce computation time -- from hours to minutes (or minutes to seconds) depending on lexicon size and which metrics are requested.

FastLex produces the same metrics as LexFindR, so results can be directly compared and validated against LexFindR output. We have done this extensively with the English Lexicon Project (ELP), eSPAL (Spanish), LemmaLex (the base LexFindR lexicon), and a Basque lexicon.

---

## Metrics computed

**Basic neighbor counts** (computed via hash-based indexing; very fast, scales linearly with lexicon size):

* *Cohorts* -- items sharing the same first two tokens (phonemes or characters)
* *Nohorts* -- cohorts that are also structural neighbors (edit distance 1)
* *Deletion neighbors* -- items obtainable by deleting one token
* *Addition neighbors* -- items obtainable by adding one token
* *Substitution neighbors* -- same-length items differing in exactly one token
* *Homoforms* -- other entries with the same token sequence (homophones for phonology, homographs for orthography)
* *DAS neighbors* (`das_nb`) -- the sum of deletion, addition, substitution neighbors and homoforms
* *Uniqueness point (UP)* -- the first position at which the item's prefix becomes unique

All of the above are computed for both orthography and phonology (suffixed `_or` and `_ph` in the output). Cohort, neighbor, and nohort counts are reported independently. To obtain non-overlapping counts, subtract nohorts from cohorts and from DAS neighbors. The resulting three values (cohorts-less-nohorts, neighbors-less-nohorts, and nohorts) are non-overlapping.

**Edit-distance summaries** (computed via BK-tree search; opt-in, slower but parallelizable):

* *OLD-K* -- mean orthographic Levenshtein distance to the *K* nearest neighbors
* *PLD-K* -- mean phonological Levenshtein distance to the *K* nearest neighbors
* *LED counts* -- number of neighbors at each exact Levenshtein distance from 2 to *D*
* *LED parts* -- LED counts split by relative length (same, shorter, longer). When *D*==1, same-length LED neighbors are substitution neighbors (or homoforms), shorter are deletion neighbors, and longer are addition neighbors. At larger values of *D*, these values do not distinguish between items that might be described as mixes of different kinds of edits, etc.

---

## Quick start

The simplest way to run FastLex is with a `.src` file (a shell command template). For example, to compute all metrics for the English Lexicon Project:

```bash
python fastlex.py \
  --lexicon-path ./data/ELP_lex_min_missing_added.csv \
  --output-path ./output/elp_flex.csv \
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
```

Ready-made example scripts are included in `examples/` for each lexicon (see [Lexica](#lexica) below). To run one:

```bash
cd examples
bash run_elp.sh
```

If you only need the basic neighbor metrics (cohorts, dels, adds, subs, etc.) and don't need OLD/PLD or LED counts, simply omit the `--oldk`, `--pldk`, `--old-count`, and `--pld-count` flags. The basic metrics are very fast even for large lexica.

---

## Requirements

* Python 3.8+
* pandas
* tqdm

Optional but recommended:

* [RapidFuzz](https://github.com/maxbachmann/RapidFuzz) -- provides a fast C++ Levenshtein backend. FastLex detects it automatically and falls back to a pure-Python implementation if it is not installed. For large lexica with OLD/PLD computation, RapidFuzz makes a substantial difference.

Install dependencies:

```bash
pip install pandas tqdm rapidfuzz
```

---

## Command-line arguments

| Argument | Required | Description |
|---|---|---|
| `--lexicon-path` | yes | Path to the input CSV |
| `--output-path` | yes | Path for the output CSV (input columns + computed metrics) |
| `--orth-col` | yes | Column name for orthographic forms |
| `--pron-col` | yes | Column name for pronunciations |
| `--delimiter-phono` | no | Tokenization for phonology: `space` (default), `none` (character-level), or a literal delimiter |
| `--delimiter-ortho` | no | Tokenization for orthography: `none` (default, character-level), `space`, or a literal delimiter |
| `--n-jobs` | no | Number of parallel workers for edit-distance computation (default: 1) |
| `--progress` | no | Show progress bars |
| `--test-sample` | no | Subsample N items for quick testing |
| `--oldk K` | no | Compute OLD-K (e.g., `--oldk 20`) |
| `--pldk K` | no | Compute PLD-K (e.g., `--pldk 20`) |
| `--old-count D` | no | Compute orthographic LED counts for distances 2..D |
| `--pld-count D` | no | Compute phonological LED counts for distances 2..D |
| `--old-parts` | no | Split OLD LED counts by relative length (requires `--old-count`) |
| `--pld-parts` | no | Split PLD LED counts by relative length (requires `--pld-count`) |

---

## Lexica

We include `.src` run-scripts for the following lexica:

| Example script | Language | Lexicon | Columns |
|---|---|---|---|
| `examples/run_elp.sh` | English | English Lexicon Project | Word, Pron_arpabet |
| `examples/run_espal.sh` | Spanish | eSPAL | word, es_phon_structure |
| `examples/run_lemmalex.sh` | English | LemmaLex (from LexFindR) | Item, Pronunciation |
| `examples/run_basque.sh` | Basque | Neurospeech Basque dictionary | Word, Pronunciation |
| `examples/run_testlex.sh` | English | Small test lexicon (included) | Item, Pronunciation |

The small `testlex.csv` (18 words) is included in `data/` so you can verify everything works out of the box. Full lexica (ELP, EsPal, LemmaLex, Basque) are not distributed here due to size and licensing; see each example script for details.

---

## How it works (briefly)

The basic neighbor metrics avoid pairwise comparisons entirely. Instead, FastLex tokenizes each item once, hashes the token sequences, and then generates candidate neighbors via dictionary lookups. For example, deletion neighbors are found by generating all possible one-token-shorter sequences and looking them up in a hash table. Addition neighbors fall out for free (if B is a deletion neighbor of A, then A is an addition neighbor of B). Substitution neighbors use a "wildcard index" where each item contributes one key per token position. All of this scales as O(N * L), where N is the number of items and L is the average token length.

For edit-distance metrics (OLD-K, PLD-K, LED counts), FastLex builds a BK-tree -- a data structure that indexes strings under a metric and supports efficient radius queries using the triangle inequality. OLD/PLD-K uses an expanding-radius search to find the K nearest neighbors, while LED counts use a fixed-radius search. For phonological metrics, multi-character phoneme tokens (like ARPABET symbols) are mapped to single Unicode Private Use Area characters so that standard string edit distance operates at the token level.

When `--n-jobs` > 1, the edit-distance stage is parallelized across queries using Python's multiprocessing. Each worker builds its own BK-tree and processes a chunk of queries independently. If [RapidFuzz](https://github.com/maxbachmann/RapidFuzz) is installed, its C++ Levenshtein implementation is used automatically.

---

## Output

FastLex writes two files:

1. **The output CSV** -- the original input columns plus all computed metrics. Orthographic metrics are suffixed `_or`, phonological metrics `_ph`.

2. **A timings CSV** (same name with `_timings` appended) -- one row per computation stage, with wall-clock seconds. Useful for benchmarking and diagnosing performance.

---

## Project structure

```
fastlex.py                 Main computation engine
README.md                  This file
INSTALL.txt                Installation and usage guide
requirements.txt           Python dependencies
data/                      Input lexica (testlex.csv included; see examples for others)
examples/                  Shell scripts showing how to run FastLex on each lexicon
scripts/
  benchmark_scaling.py     Benchmarking script for scaling analysis
output/                    Generated results (not tracked in git)
```

---

## References

* Li, Z., Crinnion, A. M., & Magnuson, J. S. (2021). LexFindR: A fast, simple, and extensible R package for finding similar-sounding words and computing phonological distance measures. *Behavior Research Methods*.
* Yarkoni, T., Balota, D. A., & Yap, M. J. (2008). Moving beyond Coltheart's N: A new measure of orthographic similarity. *Psychonomic Bulletin & Review*, 15(5), 971-979. (Introduces OLD-20.)
* Suárez, L., Tan, S. H., Yap, M. J., & Goh, W. D. (2011). Observing neighborhood effects without neighbors. *Psychonomic Bulletin & Review*, 18(3), 605-611. (Introduces PLD-20.)

---

## Contact

James S. Magnuson  
University of Connecticut & Basque Center on Cognition, Brain, and Language
