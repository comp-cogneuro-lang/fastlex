#!/usr/bin/env python3
"""
fastlex.py

Neighbor metrics computation for a lexicon, including:

PHONO (based on --pron-col tokenization)
  - length (in tokens)
  - cohorts (same first 2 tokens)
  - nohorts (cohorts restricted to "real" neighbors)
  - deletion neighbors (dels)
  - addition neighbors (adds)
  - substitution neighbors (subs)
  - homoforms (exact same token sequence)
  - PLD-K (Phonological Levenshtein Distance to K nearest neighbors), column: PLD{K}_ph
  - LED counts: number of neighbors at exact Levenshtein distance 1..D, columns: LED{d}_ph

ORTHO (based on --orth-col tokenization)
  - same neighbor metrics as above, suffixed with _or
  - OLD-K (Orthographic Levenshtein Distance to K nearest neighbors), column: OLD{K}_or
  - LED counts: number of neighbors at exact Levenshtein distance 1..D, columns: LED{d}_or

-------------------------------------------------------------------------------
Definitions (distance metrics)

OLD-K:
  Mean Levenshtein (edit) distance from a word to its K nearest *other* words
  (orthographically). We compute exact Levenshtein distance.

PLD-K (parallel to OLD-K, without feature weighting):
  Mean Levenshtein (edit) distance from a pronunciation to its K nearest *other*
  pronunciations, where the unit is a phoneme token (as produced by
  --delimiter-phono tokenization). We use uniform costs:
    insertion = deletion = substitution = 1

LED{d} counts:
  For each item, LED{d} is the count of *other* items whose Levenshtein distance
  is exactly d (for d in [led_min, led_max]).
  - We compute exact distances via BK-tree radius search up to led_max.
  - We exclude self matches (distance 0 to itself).
  - We deduplicate by unique normalized form (type-level) in the distance stage,
    because BK-tree is built on unique terms. This mirrors standard OLD/PLD
    practice. (Neighbor counts like 'identicals' still reflect duplicates at the
    row level via the neighbor-count indexing stage.)

Correctness guarantee:
  OLD-K / PLD-K and LED counts are exact (true Levenshtein distance) using a
  BK-tree plus expanding-radius search (for OLD/PLD) and fixed-radius search up
  to led_max (for LED counts). We then take the K smallest exact distances (OLD/PLD).

Speedups in this version:
  1) Optional RapidFuzz backend (exact Levenshtein implemented in fast C++).
     If RapidFuzz is installed, we use it automatically. Otherwise we fall back
     to a pure-Python DP implementation.
  2) Parallelization across queries using multiprocessing (--n-jobs).
     Each worker builds its own BK-tree once (initializer), then processes a
     chunk of queries.

-------------------------------------------------------------------------------
Example usage:

  # Basic neighbor metrics only (fast):
  python fastlex.py \
      --lexicon-path lexicon/ELP_arpabet.csv \
      --output-path lexicon/ELP_lexicon_with_neighbors.csv \
      --orth-col Word \
      --pron-col Pron_arpabet \
      --delimiter-phono space \
      --delimiter-ortho none

  # Add OLD-20 and PLD-20:
  python fastlex.py \
      --lexicon-path lexicon/ELP_arpabet.csv \
      --output-path lexicon/ELP_lexicon_with_neighbors.csv \
      --orth-col Word \
      --pron-col Pron_arpabet \
      --delimiter-phono space \
      --delimiter-ortho none \
      --oldk 20 \
      --pldk 20 \
      --n-jobs 8

  # Add LED2..LED5 counts (+ parts) for both ORTHO and PHONO:
  python fastlex.py \
      --lexicon-path lexicon/ELP_arpabet.csv \
      --output-path lexicon/ELP_lexicon_with_neighbors.csv \
      --orth-col Word \
      --pron-col Pron_arpabet \
      --delimiter-phono space \
      --delimiter-ortho none \
      --old-count 5 --old-parts \
      --pld-count 5 --pld-parts \
      --n-jobs 8

-------------------------------------------------------------------------------

-------------------------------------------------------------------------------
"""

from __future__ import annotations

import argparse
import heapq
import os
from pathlib import Path
import multiprocessing as mp
from collections import defaultdict
from typing import Any, Dict, List, Optional, Set, Tuple

import pandas as pd
from tqdm import tqdm

# Optional fast exact Levenshtein backend
try:
    from rapidfuzz.distance import Levenshtein as RFLev  # type: ignore
    _HAVE_RAPIDFUZZ = True
except Exception:
    _HAVE_RAPIDFUZZ = False

import time
import warnings

def check_requested_cpus(n_jobs: int) -> int:
    """Print CPU availability and warn on oversubscription.

    Returns an effective n_jobs value capped to available logical CPUs when known.
    """
    avail = os.cpu_count() or 0
    if avail > 0:
        print(f"[CPU] Available logical CPUs: {avail}")
        print(f"[CPU] Requested n_jobs: {n_jobs}")
        if n_jobs > avail:
            warnings.warn(
                f"Requested n_jobs={n_jobs}, but only {avail} CPUs are available. "
                "This may degrade performance due to oversubscription.",
                RuntimeWarning,
            )
        elif n_jobs > int(0.8 * avail):
            warnings.warn(
                f"Requested n_jobs={n_jobs}, which is >80% of available CPUs ({avail}). "
                "Consider leaving headroom for the OS.",
                RuntimeWarning,
            )
        return min(int(n_jobs), int(avail)) if n_jobs is not None else int(avail)
    else:
        print("[CPU] Available logical CPUs: (unknown)")
        print(f"[CPU] Requested n_jobs: {n_jobs}")
        return int(n_jobs)


def print_parallel_plan(*, n_jobs_req: int, n_jobs_eff: int,
                        do_ortho_edit: bool, do_phono_edit: bool,
                        do_ortho_dims: bool, do_phono_dims: bool) -> None:
    """Explain which parts will use multiprocessing."""
    # Basic neighbor metrics + UP are single-process.
    if do_phono_dims:
        print("[PLAN] PHONO basic metrics (cohorts/nohorts/dels/adds/subs/homoforms) + UP: single-process")
    if do_ortho_dims:
        print("[PLAN] ORTHO basic metrics (cohorts/nohorts/dels/adds/subs/homoforms) + UP: single-process")

    if n_jobs_eff <= 1:
        if do_ortho_edit or do_phono_edit:
            print("[PLAN] Edit-distance summaries (OLDk/PLDk + LED counts/parts): single-process (n_jobs=1)")
        return

    if do_ortho_edit:
        print(f"[PLAN] ORTHO edit-distance summaries (OLDk/LED counts/parts): multiprocessing with {n_jobs_eff} worker processes")
    if do_phono_edit:
        print(f"[PLAN] PHONO edit-distance summaries (PLDk/LED counts/parts): multiprocessing with {n_jobs_eff} worker processes")

    if not (do_ortho_edit or do_phono_edit):
        print("[PLAN] No edit-distance summaries requested; multiprocessing will not be used.")

# ----------------------------------------------------------------------------
# PIPELINE OVERVIEW
# This script computes a suite of lexical neighbor metrics from a tabular lexicon.
# The core steps are:
#   1) Tokenize orthographic forms (characters or delimiter-split) and pronunciations.
#   2) Compute exact-edit-distance based neighborhoods (substitution / deletion / addition).
#   3) Compute K-nearest mean edit distance (OLD-k / PLD-k) using an incremental top-k heap.
#   4) Optionally compute LED counts (exact distance 1..D), with optional length partitions
#      (same length vs shorter vs longer) using a BK-tree radius search.
#   5) Merge ORTHO + PHONO results back onto the original lexicon and write a CSV.
# Design note: heavy steps are parallelized across query items via multiprocessing; workers
# share a read-only BK-tree and a term->indices map to support duplicates in the input.
# ----------------------------------------------------------------------------

# =============================================================================
# Delimiter parsing and tokenization
# =============================================================================

def parse_delimiter_arg(arg: Optional[str]) -> Optional[str]:
    """
    Interpret a delimiter argument from CLI into a usable delimiter.

    Conventions:
      - None -> None
      - "none" -> None (character-level tokenization)
      - "space" or "whitespace" -> " " (use Python .split() i.e., any whitespace)
      - otherwise: treat as literal delimiter string (passed to .split(delimiter))

    Returns:
      delimiter string or None
    """
    if arg is None:
        return None

    s = str(arg).strip().lower()
    if s in {"none", "null", "nil"}:
        return None
    if s in {"space", "whitespace", "ws"}:
        return " "
    return arg


def tokenize_strings(strings: List[str], delimiter: Optional[str]) -> List[List[str]]:
    """
    Convert each string into a list of tokens.

    If delimiter is None:
      - tokenization is character-by-character.

    If delimiter is " ":
      - uses .split(), which splits on any whitespace and collapses multiple spaces.

    Else:
      - uses .split(delimiter) and drops empty tokens.
    """
    tokens_list: List[List[str]] = []

    for s in strings:
        if s is None:
            s = ""
        s = str(s)

        if delimiter is None:
            tokens = list(s)
        elif delimiter == " ":
            tokens = s.split()
        else:
            tokens = [t for t in s.split(delimiter) if t != ""]

        tokens_list.append(tokens)

    return tokens_list


# =============================================================================
# Neighbor metrics: dels/adds/subs/identicals/cohorts/nohorts
# =============================================================================


def encode_token_lists_to_pua_strings(
    toks_list: List[List[str]],
    base_codepoint: int = 0xE000,
) -> Tuple[List[str], Dict[str, str]]:
    """Encode a list of token lists into single-character strings.

    Motivation
    ----------
    We often need Levenshtein distance over *tokens* (e.g., phoneme symbols) where
    tokens can be multi-character strings. A naive join to a character string
    makes edit distance operate over characters (wrong). A tuple-of-tokens
    representation is correct, but can be slower / incompatible with some
    optimized string-distance backends.

    This helper maps each unique token to a single Unicode codepoint from the
    Private Use Area (PUA), producing a compact string per item. Levenshtein over
    these encoded strings is exactly token-level Levenshtein.

    Returns
    -------
    encoded : List[str]
        Encoded strings aligned with toks_list.
    token_to_char : Dict[str, str]
        The token->character mapping used (useful for debugging/inspection).
    """
    # Collect unique tokens (drop empty strings defensively)
    uniq = sorted({t for toks in toks_list for t in toks if t != ""})

    # Private Use Area range: U+E000 .. U+F8FF
    max_chars = 0xF8FF - base_codepoint + 1
    if len(uniq) > max_chars:
        raise ValueError(
            f"Too many unique tokens to encode ({len(uniq)}). "
            f"PUA capacity from base {hex(base_codepoint)} is {max_chars}."
        )

    token_to_char: Dict[str, str] = {tok: chr(base_codepoint + i) for i, tok in enumerate(uniq)}
    encoded = ["".join(token_to_char[t] for t in toks) for toks in toks_list]
    return encoded, token_to_char

# ----------------------------------------------------------------------------
# NEIGHBOR METRICS FROM TOKEN SEQUENCES
# Given a list of token sequences, compute classic neighborhood sets:
#   - cohorts: items sharing the first two tokens
#   - nohorts: cohorts restricted to true neighbors (edit distance == 1)
#   - dels/adds/subs: deletion/addition/substitution neighbors under edit distance 1
# The key trick is symmetry: if A is a deletion neighbor of B, then B is an addition
# neighbor of A. We exploit that to fill both structures in one pass over deletion variants.
# ----------------------------------------------------------------------------

def _lcp_len(a: str, b: str) -> int:
    """Length of longest common prefix of two strings."""
    n = min(len(a), len(b))
    i = 0
    while i < n and a[i] == b[i]:
        i += 1
    return i


def uniqueness_points(strings: List[str]) -> List[int]:
    """
    Compute uniqueness point (UP) for each string in `strings`.

    UP is the first 1-based position where the prefix becomes unique.
    If not unique by the final position (prefix of a longer word OR duplicates),
    UP = len(string) + 1.

    Notes:
      - Works for orthographic forms (characters) directly.
      - For phonology, pass the single-character encoded token string (phones-as-chars)
        so UP is measured in tokens, not raw characters.
    """
    n = len(strings)
    order = sorted(range(n), key=lambda i: strings[i])
    ups: List[int] = [0] * n

    for pos, idx in enumerate(order):
        w = strings[idx]
        if not w:
            ups[idx] = 1
            continue

        lcp_prev = _lcp_len(w, strings[order[pos - 1]]) if pos > 0 else 0
        lcp_next = _lcp_len(w, strings[order[pos + 1]]) if (pos + 1) < n else 0

        m = lcp_prev if lcp_prev > lcp_next else lcp_next
        L = len(w)

        ups[idx] = (L + 1) if m >= L else (m + 1)

    return ups

def _parse_lexicon_filename(lexicon_path: str):
    """
    Parse lexicon metadata from the input lexicon filename.

    Supported sampled-lexicon pattern:
        <lex>-lex-<NN>k-v<RR>.csv
      e.g. elp-lex-02k-v01.csv  -> lexicon='elp', sample_size=2000, run=1

    Fallback:
      If the filename doesn't match, lexicon is set to the file stem
      and sample_size/run are returned as None.

    Returns
    -------
    lexicon_label : str
    sample_size   : Optional[int]
    run           : Optional[int]
    """
    import re
    from pathlib import Path

    stem = Path(lexicon_path).stem  # no suffix, e.g. "elp-lex-02k-v01" or "ELP_lex_min"

    m = re.match(r"^(?P<lex>.+?)-lex-(?P<size>\d+)k-v(?P<run>\d+)$", stem)
    if not m:
        return stem, None, None

    lex = m.group("lex")
    size_k = int(m.group("size"))   # handles leading zeros like "02"
    run = int(m.group("run"))       # handles leading zeros like "01"

    return lex, size_k * 1000, run


class TimingLog:
    """
    Timing collector that writes a tidy timings CSV.

    Output columns:
      tool, lexicon, items, operation, seconds, sample_size, run
    """
    def __init__(
        self,
        *,
        tool: str,
        lexicon: str,
        items: int,
        sample_size: Optional[int] = None,
        run: Optional[int] = None,
    ):
        self.tool = str(tool)
        self.lexicon = str(lexicon)
        self.items = int(items)
        self.sample_size = None if sample_size is None else int(sample_size)
        self.run = None if run is None else int(run)

        # list of (operation, seconds)
        self._rows = []

    def add(self, key: str, seconds: float):
        if key is None:
            return
        self._rows.append((str(key), float(seconds)))

    def to_csv(self, path: str):
        """
        Write one row per operation.
        """
        import csv

        with open(path, "w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)

            # Always write header
            w.writerow(["tool", "lexicon", "items", "operation", "seconds", "sample_size", "run"])

            for op, sec in self._rows:
                w.writerow([
                    self.tool,
                    self.lexicon,
                    self.items,
                    op,
                    f"{sec:.6f}",
                    "" if self.sample_size is None else str(self.sample_size),
                    "" if self.run is None else str(self.run),
                ])


def _fmt_elapsed(sec: float) -> str:
    return f"{sec:.6f} sec elapsed"


def compute_neighbor_counts(
    strings: List[str],
    delimiter: Optional[str],
    *,
    label: Optional[str] = None,
    show_progress: bool = False,
    timings: Optional["TimingLog"] = None,
    key_suffix: str = "",
) -> pd.DataFrame:
    """
    Compute neighbor metrics on a list of raw strings given a delimiter.

    Returned columns:
      - string
      - length
      - cohorts
      - nohorts
      - deletion_neighbors
      - addition_neighbors
      - substitution_neighbors
      - homoforms

    Definitions:
      - homoforms: exact match in token sequence (excluding self)
      - cohorts: share first 2 tokens (only if len >= 2)
      - dels: sequences obtainable by deleting exactly one token
      - adds: inverse of dels (counts longer items that delete-to this item)
      - subs: same length; differ in exactly one position
      - nohorts: cohorts ∩ (identicals ∪ dels ∪ adds ∪ subs)

    Notes:
      - We compute neighbor sets, then take counts.
      - Self is always excluded.
      - Duplicate forms in the lexicon naturally create identical neighbors.
    """
    # -------------------------------------------------------------------------
    # Why this section is structured this way
    #
    # The goal here is to avoid naive pairwise comparisons across the lexicon.
    # A brute-force approach would compare every item to every other item:
    #
    #   for i in items:
    #       for j in items:
    #           compare(i, j)
    #
    # which scales as O(N^2) and becomes infeasible for large lexica.
    #
    # FastLex instead:
    #   - tokenizes each item once,
    #   - hashes token sequences and partial sequences,
    #   - and generates candidate neighbors using these hash structures
    #     (e.g., identical forms, deletions, substitutions, cohorts),
    #     rather than by direct comparison.
    #
    # As a result, most computations in this section scale as:
    #
    #   O(N * L)
    #
    # where L is the average token length, plus constant-time hash lookups.
    #
    # This is why these "dims" computations scale roughly linearly with lexicon
    # size and are dramatically faster than edit-distance–based methods.
    # -------------------------------------------------------------------------

    t_all0 = time.perf_counter()
    tokens_list = tokenize_strings(strings, delimiter=delimiter)

    """
        GOAL: This block prepares index structures that allow FastLex to:
                - avoid pairwise comparisons (O(N²))
                - compute neighbor counts using hashing and grouping
                - keep everything aligned to row indices of the lexicon
        WHY:  This single structure enables fast computation of:
                - identicals (homoforms)
                - deletion neighbors -- and addition neighbors are the inverse, 
                  so they can be computed in one pass (if word B is a deletion 
                  neighbor of word A, A is an addition neighbor of B)
              All of that becomes O(1) dictionary lookup instead of scanning the 
              lexicon -- a huge time savings.
    """
    # Tokens are the elements of the string, which have been tokenized (search for PUA for details)
    # Map full token sequence -> list of indices having that sequence
    # seq_to_indicies is a hash table (dictionary); we need tuples (immutable and hashable) 
    
    seq_to_indices: Dict[Tuple[str, ...], List[int]] = defaultdict(list)
    for i, toks in enumerate(tokens_list):
        seq_to_indices[tuple(toks)].append(i)

    lengths = [len(toks) for toks in tokens_list]

    identical_sets: List[Set[int]] = [set() for _ in range(len(tokens_list))]
    del_sets: List[Set[int]] = [set() for _ in range(len(tokens_list))]
    add_sets: List[Set[int]] = [set() for _ in range(len(tokens_list))]
    sub_sets: List[Set[int]] = [set() for _ in range(len(tokens_list))]
    cohort_sets: List[Set[int]] = [set() for _ in range(len(tokens_list))]

    # 1) IDENTICALS
    # -------------------------------------------------------------------------
    # Identical neighbors (homoforms)
    #
    # Goal:
    #   Identify items that have exactly the same token sequence (i.e., are
    #   identical in form).
    #
    # Strategy:
    #   Earlier, we built a hash table (seq_to_indices) mapping each full
    #   token sequence to the list of lexicon indices that share it.
    #
    #   For each group with more than one member, all items in the group
    #   are identical neighbors of one another.
    #
    #   Each item i receives all other indices in its group as identical
    #   neighbors (excluding itself).
    #
    # Performance:
    #   This is a simple grouping operation over hashed token sequences.
    #   It runs in linear time with respect to the number of items and
    #   requires no pairwise comparisons.
    # -------------------------------------------------------------------------

    t0 = time.perf_counter()
    for _, idxs in seq_to_indices.items():
        if len(idxs) <= 1:
            continue
        idxset = set(idxs)
        for i in idxs:
            identical_sets[i] = idxset - {i}
    if label:
        dt = time.perf_counter() - t0
        print(f"[{label}] identicals\nidenticals done: {_fmt_elapsed(dt)}", flush=True)
        if timings is not None:
            timings.add(f"identicals{key_suffix}", dt)

    # 2) COHORTS (first 2 tokens)
    # -------------------------------------------------------------------------
    # Cohort neighbors (shared initial overlap)
    #
    # Goal:
    #   Identify cohort neighbors: items that share the same initial token
    #   sequence. Here, cohorts are defined as items that share the first
    #   two tokens.
    #
    # Strategy:
    #   We construct a hash table mapping the first two tokens of each item
    #   to the list of lexicon indices with that prefix.
    #
    #   Items in the same prefix group form a cohort. For each item, all
    #   other items with the same prefix are assigned as cohort neighbors
    #   (excluding itself).
    #
    # Performance:
    #   This is another hash-based grouping operation. Each item contributes
    #   a single prefix key, and cohort relations are recovered via constant-
    #   time lookups. The overall complexity is linear in the number of items.
    #
    # Note:
    #   The cohort definition here uses an overlap of exactly two tokens,
    #   matching common definitions in spoken-word recognition models.
    # -------------------------------------------------------------------------

    t0 = time.perf_counter()
    prefix2_to_indices: Dict[Tuple[str, str], List[int]] = defaultdict(list)
    for i, toks in enumerate(tokens_list):
        if len(toks) >= 2:
            prefix2_to_indices[(toks[0], toks[1])].append(i)

    for _, idxs in prefix2_to_indices.items():
        if len(idxs) <= 1:
            continue
        idxset = set(idxs)
        for i in idxs:
            cohort_sets[i] = idxset - {i}
    if label:
        dt = time.perf_counter() - t0
        print(f"[{label}] cohorts (overlap=2)\ncohorts done: {_fmt_elapsed(dt)}", flush=True)
        if timings is not None:
            timings.add(f"cohorts{key_suffix}", dt)

    # 3) DELS and ADDS
    # -------------------------------------------------------------------------
    # Deletion and addition neighbors (computed together)
    #
    # Goal:
    #   Identify deletion and addition neighbors efficiently. Two items are
    #   neighbors if one can be obtained from the other by deleting or
    #   inserting exactly one token.
    #
    # Key observation:
    #   Deletions and additions are inverse relations:
    #     - If item B can be formed by deleting one token from item A,
    #       then A is a deletion neighbor of B and B is an addition
    #       neighbor of A.
    #
    # Strategy:
    #   For each item, we generate all possible "shorter" token sequences
    #   by deleting one token at each position. Each shortened sequence
    #   is looked up in a hash table (seq_to_indices) that maps token
    #   sequences to lexicon indices.
    #
    #   If the shortened sequence exists in the lexicon:
    #     - the current item has a deletion neighbor
    #     - the shorter item has an addition neighbor
    #
    #   Both relationships are recorded simultaneously in a single pass.
    #
    # Performance:
    #   Each item generates O(L) shortened sequences, where L is the token
    #   length. All neighbor lookups are constant-time hash operations.
    #   This avoids any pairwise comparisons and yields roughly O(N * L)
    #   complexity overall.
    #
    # Notes:
    #   Sets are used to store neighbors so that duplicate neighbors
    #   (e.g., arising from repeated tokens) are handled automatically.
    # -------------------------------------------------------------------------

    t0 = time.perf_counter()
    it = enumerate(tokens_list)
    if show_progress:
        it = tqdm(it, total=len(tokens_list), desc=f"[{label or 'NEIGH'}] dels/adds", leave=False)
    for i, toks in it:
        if not toks:
            continue
        L = len(toks)
        for pos in range(L):
            shorter = tuple(toks[:pos] + toks[pos + 1 :])
            if shorter in seq_to_indices:
                for j in seq_to_indices[shorter]:
                    if j != i:
                        del_sets[i].add(j)
                        add_sets[j].add(i)
    if label:
        # dels and adds are computed symmetrically in one pass
        dt = time.perf_counter() - t0
        print(f"[{label}] deletion neighbors (D)\ndels done: {_fmt_elapsed(dt)}", flush=True)
        if timings is not None:
            timings.add(f"dels{key_suffix}", dt)
        print(f"[{label}] addition neighbors (A)\nadds done: {_fmt_elapsed(dt)}", flush=True)
        if timings is not None:
            timings.add(f"adds{key_suffix}", dt)

    # 4) SUBS via wildcard index
    # -------------------------------------------------------------------------
    # Substitution neighbors via a "wildcard" hash index
    #
    # Goal:
    #   Identify substitution neighbors efficiently, without pairwise
    #   comparisons. Two items are substitution neighbors if:
    #     - they have the same length, and
    #     - they differ in exactly one token position.
    #
    # Strategy:
    #   For each token sequence, we generate L "wildcard" keys (one per
    #   position), where the token at that position is removed. Each key
    #   is defined by:
    #
    #     (sequence_length, position, sequence_without_that_position)
    #
    #   Items that share the same wildcard key are identical everywhere
    #   except possibly at the wildcarded position.
    #
    #   These items are grouped together in a hash table ("wildcard_index").
    #
    # Resolution:
    #   Within each wildcard group, we further split items by the actual
    #   token at the wildcarded position. Items with different tokens at
    #   that position are substitution neighbors of one another.
    #
    # Performance:
    #   This avoids O(N^2) comparisons. Instead, each item contributes O(L)
    #   wildcard keys, and all neighbor relations are recovered via hash
    #   lookups and set operations. Overall complexity is roughly O(N * L),
    #   where L is token length.
    #
    #   This is why substitution neighbors can be computed efficiently
    #   even for large lexica.
    # -------------------------------------------------------------------------

    t0 = time.perf_counter()
    wildcard_index: Dict[Tuple[int, int, Tuple[str, ...]], List[int]] = defaultdict(list)
    it2 = enumerate(tokens_list)
    if show_progress:
        it2 = tqdm(it2, total=len(tokens_list), desc=f"[{label or 'NEIGH'}] subs-index", leave=False)
    for i, toks in it2:
        L = len(toks)
        if L == 0:
            continue
        seq = tuple(toks)
        for pos in range(L):
            key = (L, pos, seq[:pos] + seq[pos + 1 :])
            wildcard_index[key].append(i)

    for (L, pos, _), idxs in wildcard_index.items():
        if len(idxs) <= 1:
            continue
        token_to_idxs: Dict[str, List[int]] = defaultdict(list)
        for i in idxs:
            tok = tokens_list[i][pos]
            token_to_idxs[tok].append(i)

        all_idxs_set = set(idxs)
        for _, group in token_to_idxs.items():
            others = all_idxs_set - set(group)
            for i in group:
                sub_sets[i].update(others)
    if label:
        dt = time.perf_counter() - t0
        print(f"[{label}] substitution neighbors (S)\nsubs done: {_fmt_elapsed(dt)}", flush=True)
        if timings is not None:
            timings.add(f"subs{key_suffix}", dt)

    # 5) NOHORTS
    # -------------------------------------------------------------------------
    # Nohorts (cohort neighbors that are also DAS/LED ≤ 1 neighbors)
    #
    # Goal:
    #   Compute the number of nohorts for each item. A nohort is defined as
    #   an item that is both:
    #     - a cohort neighbor (shares the first 2 tokens), and
    #     - a true structural neighbor (identical, deletion, addition,
    #       or substitution neighbor).
    #
    # Strategy:
    #   For each item, we first form the set of all true neighbors by
    #   taking the union of:
    #       identicals ∪ deletions ∪ additions ∪ substitutions
    #
    #   We then intersect this set with the item's cohort set. The size of
    #   this intersection is the nohort count.
    #
    # Performance:
    #   This uses fast set union and intersection operations. Since all
    #   neighbor sets were already computed in earlier steps, no additional
    #   structural comparisons are needed here. The per-item cost is small,
    #   and the overall complexity is linear in the number of items.
    #
    # -------------------------------------------------------------------------

    t0 = time.perf_counter()
    nohorts_counts: List[int] = []
    
    # Iterate over all items, with optional progress bar
    it3 = range(len(tokens_list))
    if show_progress:
        it3 = tqdm(it3, total=len(tokens_list), desc=f"[{label or 'NEIGH'}] nohorts", leave=False)
    for i in it3:
        # Union of all true neighbor types for item i
        true_neighbors = identical_sets[i] | del_sets[i] | add_sets[i] | sub_sets[i]
        # Nohorts as union of cohorts and 'true' neighbors
        nohorts_counts.append(len(cohort_sets[i] & true_neighbors))
    
    # optional timing / logging
    if label:
        dt = time.perf_counter() - t0
        print(f"[{label}] nohorts\nnohorts done: {_fmt_elapsed(dt)}", flush=True)
        if timings is not None:
            timings.add(f"nohorts{key_suffix}", dt)

    # Assemble output DataFrame. Each row corresponds to one lexical item,
    # with counts of different neighbor types.
    del_counts = [len(s) for s in del_sets]
    add_counts = [len(s) for s in add_sets]
    sub_counts = [len(s) for s in sub_sets]
    hom_counts = [len(s) for s in identical_sets]
    das_counts = [d + a + s + h for d, a, s, h in zip(del_counts, add_counts, sub_counts, hom_counts)]

    df_out = pd.DataFrame({
        "string": strings,
        "length": lengths,
        "cohorts": [len(s) for s in cohort_sets],
        "nohorts": nohorts_counts,
        "das_nb": das_counts,
        "deletion_neighbors": del_counts,
        "addition_neighbors": add_counts,
        "substitution_neighbors": sub_counts,
        "homoforms": hom_counts,
    })

    # Final timing for all dims computations
    if label:
        dt = time.perf_counter() - t_all0
        print(f"Computed {label} dims: {_fmt_elapsed(dt)}", flush=True)
        if timings is not None:
            timings.add(f"dims{key_suffix}", dt)
    return df_out


# =============================================================================
# BK-tree Levenshtein engine (exact distance; optional RapidFuzz backend)
# =============================================================================

#def _levenshtein_cutoff(a: str, b: str, max_dist: Optional[int] = None) -> int:
def _levenshtein_cutoff(a: Any, b: Any, max_dist: Optional[int] = None) -> int:

    """
    Exact Levenshtein distance with optional cutoff.

    If RapidFuzz is available, we use its exact implementation (fast C++).
    Otherwise we use a pure-Python DP implementation.

    Cutoff contract:
      - if max_dist is not None and dist > max_dist, return max_dist + 1
    """
    # RapidFuzz only applies to *strings* (character-level edit distance).
    # For token sequences, fall back to DP below.
    if _HAVE_RAPIDFUZZ and isinstance(a, str) and isinstance(b, str):
        if max_dist is None:
            return int(RFLev.distance(a, b))
        d = int(RFLev.distance(a, b, score_cutoff=max_dist))
        return d if d <= max_dist else (max_dist + 1)

    # if _HAVE_RAPIDFUZZ:
    #     if max_dist is None:
    #         return int(RFLev.distance(a, b))
    #     d = int(RFLev.distance(a, b, score_cutoff=max_dist))
    #     return d if d <= max_dist else (max_dist + 1)

    # ---------------------- pure-python fallback ----------------------
    # Dynamic programming (Wagner–Fischer) edit distance.
    #
    # We compute a DP table where DP[i][j] is the edit distance between:
    #   - the first i symbols of `a`
    #   - the first j symbols of `b`
    #
    # To keep memory O(min(len(a), len(b))), we only keep:
    #   - `prev`: the previous row DP[i-1][0..lb]
    #   - `cur` : the current  row DP[i][0..lb]
    #
    # Variable naming inside the loops:
    #   - i, ca: position and symbol from `a` (1-indexed for DP convenience)
    #   - j, cb: position and symbol from `b`
    #   - ins / dele / sub: candidate costs for insertion, deletion, substitution
    #   - v: the minimum of those three candidates (i.e., DP[i][j])
    #   - row_min: the smallest DP value in the current row (used for cutoff pruning)
    if a == b:
        # Exact match: Levenshtein distance is 0 regardless of cutoff.
        return 0

    la, lb = len(a), len(b)
    if max_dist is not None and abs(la - lb) > max_dist:
        # Length difference is a hard lower bound on edit distance.
        # If it already exceeds the cutoff, we can stop immediately.
        return max_dist + 1

    if lb > la:
        # Iterate over the longer string as `a` and the shorter as `b` so
        # our DP rows have length lb+1 and stay as small as possible.
        a, b = b, a
        la, lb = lb, la

    # Base case for DP row i=0: distance from empty prefix of `a` to b[:j] is j insertions.
    prev = list(range(lb + 1))
    for i, ca in enumerate(a, start=1):
        # DP[i][0] = i deletions (turn a[:i] into empty string).
        cur = [i]
        row_min = i  # Track smallest value in this row for cutoff-based early exit.
        for j, cb in enumerate(b, start=1):
            # Three possible last operations to transform a[:i] -> b[:j]:
            #   insertion    (insert cb):      DP[i][j-1] + 1
            #   deletion     (delete ca):      DP[i-1][j] + 1
            #   substitution (ca -> cb):       DP[i-1][j-1] + (ca != cb)
            ins = cur[j - 1] + 1
            dele = prev[j] + 1
            sub = prev[j - 1] + (ca != cb)

            # Take the cheapest of the three options.
            v = ins if ins < dele else dele
            v = sub if sub < v else v

            cur.append(v)
            if v < row_min:
                row_min = v

        if max_dist is not None and row_min > max_dist:
            # All entries in this row are already > max_dist, and subsequent rows
            # can only stay the same or increase by at most 1 per step; therefore
            # the final distance cannot come back under the cutoff.
            return max_dist + 1
        prev = cur  # Slide the DP window: current row becomes previous row.

    d = prev[-1]  # Final cell DP[la][lb] = distance between full sequences.
    if max_dist is not None and d > max_dist:
        # Enforce the cutoff contract: callers can treat (max_dist+1) as "too far".
        return max_dist + 1
    return d

class _BKNode:
    """BK-tree node for approximate matching under edit distance.

    NOTE: terms can be strings (orthography) or tuples of tokens (phonology).
    They must be hashable and support len() and iteration.
    """
    __slots__ = ("term", "children")

    def __init__(self, term: Any):
        self.term: Any = term
        self.children: Dict[int, "_BKNode"] = {}



# ----------------------------------------------------------------------------
# BK-TREE IMPLEMENTATION (FOR FAST FIXED-RADIUS LED SEARCH)
# A BK-tree indexes strings under a metric (here: exact Levenshtein distance).
# It supports efficient queries of the form: 'return all items within radius r'.
# We use it for LED counts (distance 1..D) because BK search prunes branches using
# the triangle inequality, avoiding a full O(N) scan per query.
# ----------------------------------------------------------------------------

def _bk_insert(root: _BKNode, term: Any) -> None:
    # Insert `term` into an existing BK-tree rooted at `root`.
    #
    # Each edge from a node is labeled by the edit distance between the node's term and the child term.
    # Insertion is deterministic given the distance function:
    #   - compute d = dist(term, node.term)
    #   - follow / create the child edge labeled d
    #   - repeat until an empty slot is found
    node = root
    while True:
        d = _levenshtein_cutoff(term, node.term, None)
        child = node.children.get(d)
        if child is None:
            node.children[d] = _BKNode(term)
            return
        node = child


def _bk_search_within(root: _BKNode, query: Any, radius: int, out: List[Tuple[Any, int]]) -> None:
    """BK-tree radius search (iterative DFS).

    We want all dictionary terms `t` such that dist(query, t) <= radius.

    BK-trees index children by their edit distance from the parent term.
    If the current node term is `x` at distance d = dist(query, x), then for any child
    on edge `e = dist(x, child.term)`, the triangle inequality gives:

        |d - e| <= dist(query, child.term) <= d + e

    Therefore, a child can only possibly fall within `radius` if e is in [d-radius, d+radius].
    We exploit that to prune almost all branches in practice.

    Notes:
      - We call _levenshtein_cutoff(..., radius) so distance computation can also short-circuit.
      - `out` is appended with (term, distance) pairs; caller can post-process duplicates / best-dist.
    """
    stack = [root]
    while stack:
        node = stack.pop()
        d = _levenshtein_cutoff(query, node.term, radius)
        if d <= radius:
            out.append((node.term, d))

        lo = d - radius
        hi = d + radius
        for edge_dist, child in node.children.items():
            if lo <= edge_dist <= hi:
                stack.append(child)


# =============================================================================
# Parallel LD-K and LED counts using BK-tree
# =============================================================================

_G_ROOT: Optional[_BKNode] = None
_G_K_EFF: int = 0
_G_LED_MIN: int = 1
_G_LED_MAX: int = 5
_G_DO_LEDK: bool = False  # Default OFF: enable via CLI flags
_G_DO_LED_PARTS: bool = False  # Default OFF: enable via CLI flags
# Map from term string -> length in *token* units for the metric being computed.
# For ORTHO this is just len(string), but for PHONO we pass token counts.
_G_TERM_LEN: Optional[Dict[Any, int]] = None
_G_TERM_TO_INDICES: Optional[Dict[Any, List[int]]] = None  # term -> original indices (duplicates/homophones)
_G_TERM_MULT: Optional[Dict[Any, int]] = None  # term -> multiplicity (len(term_to_indices[term]))


# 
def _worker_init(
    unique_terms: List[Any],
    term_len: Dict[Any, int],
    term_to_indices: Dict[Any, List[int]],
    term_mult: Dict[Any, int],
    k_eff: int,
    led_min: int,
    led_max: int,
    do_ledk: bool,
    do_led_parts: bool,
) -> None:
    """Worker initializer: build a BK-tree once per process and store shared lookup maps."""
    global _G_ROOT, _G_TERM_LEN, _G_TERM_TO_INDICES, _G_TERM_MULT
    global _G_K_EFF, _G_LED_MIN, _G_LED_MAX, _G_DO_LEDK, _G_DO_LED_PARTS

    _G_K_EFF = int(k_eff)
    _G_LED_MIN = int(led_min)
    _G_LED_MAX = int(led_max)
    _G_DO_LEDK = bool(do_ledk)
    _G_DO_LED_PARTS = bool(do_led_parts) if do_ledk else False
    _G_TERM_LEN = term_len
    _G_TERM_TO_INDICES = term_to_indices
    _G_TERM_MULT = term_mult

    if not unique_terms:
        _G_ROOT = None
        return

    root = _BKNode(unique_terms[0])
    for t in unique_terms[1:]:
        _bk_insert(root, t)
    _G_ROOT = root

# ----------------------------------------------------------------------------
# PER-QUERY COMPUTATION: OLD/PLD K-NEAREST + (OPTIONAL) LED COUNTS
#
# This function is executed *inside a worker process* for one query item q.
# It assumes the worker has already been initialized with a BK-tree over the
# lexicon terms (via _worker_init), stored in the global _G_ROOT.
#
# What we compute for one query q:
#
#   (A) LED totals (optional):
#       For each exact edit distance d in [led_min, led_max], compute
#       how many lexicon terms are exactly distance d away from q.
#
#       Implementation detail:
#         - We use a BK-tree radius search up to led_max:
#               _bk_search_within(root, q, led_max, hits)
#           which returns candidate terms along with their distance d.
#         - BK-trees prune the search space using the triangle inequality:
#             if node.term is at distance dist(q, node.term) = D,
#             then only child edges labeled in [D - r, D + r] can contain
#             terms within radius r of q.
#             - This avoids scanning all N terms, which would be infeasible.
#             - This follows from a basic property of edit distance: the distance
#               between two words cannot change arbitrarily when moving through the 
#               tree. By skipping whole branches at once, the BK-tree avoids 
#               checking most of the lexicon and makes edit-distance search 
#               feasible for large vocabularies.
#
#       Duplicate handling / multiplicity:
#         - The lexicon may contain duplicates (same surface form repeated),
#           e.g., homographs or repeated entries.
#         - The BK-tree is built over *unique terms*, so a match to a term
#           represents potentially multiple lexicon rows.
#         - term_mult[term] stores how many times this term occurs in the
#           original lexicon. We multiply counts by term_mult so LED counts
#           reflect the raw lexicon, not the deduplicated BK-tree.
#
#   (B) LED parts (optional):
#       Further split LED counts by relative length of neighbor vs query:
#         - same length  -> substitution-like changes
#         - shorter      -> requires at least one deletion somewhere
#         - longer       -> requires at least one insertion somewhere
#
#       Implementation detail:
#         - We precompute entries = [(d, mult, tlen), ...] from the LED search,
#           so the "parts" pass can run without repeated dictionary lookups.
#
#   (C) OLD/PLD-K ("LD-K") mean distance:
#       Compute the mean Levenshtein distance to the K nearest *other* terms.
#
#       Implementation detail:
#         - We use the BK-tree as an expanding-radius candidate generator:
#             radius = 0, 1, 2, ...
#             collect all unique terms within that radius
#             stop once we have at least K candidate distances (accounting for multiplicity)
#         - We then take the K smallest distances (heapq.nsmallest) and average them.
#
#       A subtlety:
#         - We must exclude the query term itself (distance 0 to itself).
#           However, if the term appears multiple times in the lexicon,
#           *other* duplicates should still count as valid neighbors.
#           That is why we use term_to_indices + multiplicity and reduce mult by 1
#           only when term == q.
#
# Timing outputs:
#   - t_led_counts: time for the BK-tree radius search + tallying LED totals
#   - t_led_parts: extra time for splitting LED totals into same/short/long
#   - t_ldk:       time for the expanding-radius LD-K computation
# ----------------------------------------------------------------------------
def _compute_ldk_and_led_for_query(
    q_info: Tuple[int, Any, int]
) -> Tuple[float, List[int], List[int], List[int], List[int], float, float, float]:
    """
    Worker-side computation for a single query.

    Returns:
      (ldk_value,
       led_counts, led_counts_same, led_counts_short, led_counts_long,
       t_ldk, t_led_counts, t_led_parts)

    Timing semantics:
      - t_ldk: time spent computing the K-nearest mean distance (OLD/PLD).
      - t_led_counts: time spent computing LED total counts (LED2..D).
      - t_led_parts: incremental time spent computing LED parts (same/short/long).
    """
    
    # Unpack query:
    #   q_idx: row index in the lexicon (used elsewhere; not central here)
    #   q:     the query term (string or encoded token-string)
    #   q_len: token length of query (used for LED parts classification)
    q_idx, q, q_len = q_info
    
    # BK-tree root built in worker init; if missing, we cannot compute distances.
    root = _G_ROOT
    if root is None:
        return (0.0, [], [], [], [], 0.0, 0.0, 0.0)

    # Worker-side cached metadata (built once in _worker_init):
    #   term_len       : {term -> token length}   (for parts split; avoids recomputing)
    #   term_to_indices: {term -> list(indices)}  (for multiplicity in raw lexicon)
    #   term_mult      : {term -> multiplicity}   (fast multiplicity lookup)
    term_len = _G_TERM_LEN or {}
    term_to_indices = _G_TERM_TO_INDICES or {}
    term_mult = _G_TERM_MULT or {}

    # Control parameters (also set globally in _worker_init):
    #   k_eff        : effective K for LD-K (may be reduced if lexicon smaller)
    #   led_min/max  : LED distance range [led_min, led_max]
    #   do_ledk      : whether to compute LED totals at all
    #   do_led_parts : whether to compute LED parts split (same/short/long)
    k_eff = _G_K_EFF
    led_min = _G_LED_MIN
    led_max = _G_LED_MAX
    do_ledk = _G_DO_LEDK
    do_led_parts = _G_DO_LED_PARTS

    # Timers returned for instrumentation / profiling
    t_ldk = 0.0
    t_led_counts = 0.0
    t_led_parts = 0.0

    # ---------------- LED totals (+ optional parts) ----------------
    # These arrays are indexed by (d - led_min), so:
    #   led_counts[0] corresponds to distance led_min
    #   led_counts[-1] corresponds to distance led_max
    led_counts: List[int]
    led_counts_same: List[int]
    led_counts_short: List[int]
    led_counts_long: List[int]

    if do_ledk:
        t0 = time.perf_counter()

        # BK-tree radius search:
        #   hits is a list of (term, d) for terms within distance <= led_max.
        # Note: This returns unique terms as stored in the BK-tree.
        hits: List[Tuple[Any, int]] = []
        _bk_search_within(root, q, led_max, hits)

        # Pre-filter and pre-compute multiplicity + term length once.
        # We store a compact triple (d, mult, tlen) so the optional 'parts'
        # computation can run without repeated dictionary lookups.
        entries: List[Tuple[int, int, int]] = []  # (d, mult, tlen)

        # Initialize LED total counts across the distance range.
        led_counts = [0 for _ in range(led_max - led_min + 1)]

        for term, d in hits:
            # Ignore distances outside our requested window.
            if not (led_min <= d <= led_max):
                continue

            # Multiplicity: how many times this unique term occurs in the original lexicon.
            # This inflates counts to match the raw lexicon rather than deduped BK-tree.
            mult = term_mult.get(term, 1)
            if mult <= 0:
                mult = 1

            # Precomputed token length for this term; fallback to len(term)
            # (len(term) is correct for single-character token encodings).
            tlen = term_len.get(term)
            if tlen is None:
                tlen = len(term)
            entries.append((d, mult, int(tlen)))
            
            # Add multiplicity-weighted count at exact distance d.
            led_counts[d - led_min] += mult

        t_led_counts += (time.perf_counter() - t0)

        if do_led_parts:
            t1 = time.perf_counter()

            # Initialize part-specific counts across the same distance bins.
            led_counts_same = [0 for _ in range(led_max - led_min + 1)]
            led_counts_short = [0 for _ in range(led_max - led_min + 1)]
            led_counts_long = [0 for _ in range(led_max - led_min + 1)]

            # Classify each LED hit by neighbor length relative to query length.
            # This provides a rough structural interpretation:
            #   same length -> substitution-like neighbors
            #   shorter     -> must include at least one deletion
            #   longer      -> must include at least one insertion
            for d, mult, tlen in entries:
                idx = d - led_min
                if tlen == q_len:
                    led_counts_same[idx] += mult
                elif tlen < q_len:
                    led_counts_short[idx] += mult
                else:
                    led_counts_long[idx] += mult

            t_led_parts += (time.perf_counter() - t1)
        else:
            # Explicit empty lists when parts not requested (keeps downstream simple)
            led_counts_same, led_counts_short, led_counts_long = [], [], []
    else:
        # If LED not requested, return empty LED arrays
        led_counts, led_counts_same, led_counts_short, led_counts_long = [], [], [], []

    # ---------------- LD-K (K nearest mean distance) ----------------
    # If K is not requested (or reduced to 0), skip.
    if k_eff <= 0:
        return (0.0, led_counts, led_counts_same, led_counts_short, led_counts_long, t_ldk, t_led_counts, t_led_parts)

    t2 = time.perf_counter()

    # Expanding-radius search state:
    #   radius     : current BK-tree radius being queried
    #   candidates : collected distances (replicated according to multiplicity, up to K)
    #   seen_terms : ensures each unique term is processed at most once across radii
    radius = 0
    candidates: List[int] = []
    seen_terms: Set[Any] = set()

    while True:
        ring_hits: List[Tuple[Any, int]] = []
        _bk_search_within(root, q, radius, ring_hits)

        for term, d in ring_hits:
            # Prevent double-counting the same unique term (BK search returns it again
            # at larger radii).
            if term in seen_terms:
                continue
            seen_terms.add(term)

            # Multiplicity in raw lexicon (number of rows corresponding to this term)
            mult = len(term_to_indices.get(term, []))
            if mult <= 0:
                mult = 1
            if term == q:
                # Exclude the query token itself; allow other duplicates/homophones
                mult = max(0, mult - 1)
            if mult == 0:
                continue

            # Add up to the number of distances we still need to reach K.
            # We replicate distance d by multiplicity, but we only take as many as needed.
            need = k_eff - len(candidates)
            take = min(need, mult)
            candidates.extend([d] * take)

        # Stop once we have >= K candidate distances.
        if len(candidates) >= k_eff:
            break

        # Otherwise widen the search radius and try again.
        radius += 1
        
        # Safety break: prevents infinite loops if something odd happens
        # (e.g., extremely sparse neighbor structure). The bound here is heuristic.
        if radius > max(2, len(q) + 2):
            break

    # Compute the LD-K mean from collected distances.
    # If we have >= K, take the K smallest distances (ties naturally handled).
    if not candidates:
        ldk_val = float("nan")
    elif len(candidates) >= k_eff:
        k_smallest = heapq.nsmallest(k_eff, candidates)
        ldk_val = sum(k_smallest) / k_eff
    else:
        # Fallback: not enough candidates found; average what we have. 
        # Should this generate a warning? Not clear it would happen with 
        # a lexicon with at least k+1 words! 
        ldk_val = sum(candidates) / len(candidates)

    t_ldk += (time.perf_counter() - t2)

    return (ldk_val, led_counts, led_counts_same, led_counts_short, led_counts_long, t_ldk, t_led_counts, t_led_parts)

def _normalize_items(items, normalize_case: bool):
    """
    Normalize items for edit-distance computations.

    - For ORTHO: items are strings, we can casefold if requested.
    - For PHONO: items are sequences (tuples of tokens). We must NOT touch them.
    """
    out = []
    for x in items:
        if isinstance(x, str):
            if normalize_case:
                out.append(x.casefold())
            else:
                out.append(x)
        else:
            # token sequences (tuple/list of tokens) -> leave unchanged
            out.append(x)
    return out

# ----------------------------------------------------------------------------
# BATCH DRIVER FOR OLD/PLD + LED
# This is the main engine used by both ORTHO (OLD) and PHONO (PLD) pipelines.
# It builds the BK-tree once, then maps per-query computations across all unique terms.
# Outputs are returned in 'per original row' order (not per unique term), so downstream
# code can directly attach columns back onto the original DataFrame.
# LED toggles:
#   do_ledk=False      => skip all LED work (no radius search, no LED dicts returned)
#   do_led_parts=False => compute only total LED(d), omit same/shorter/longer splits
# ----------------------------------------------------------------------------
def compute_ldk_and_led(
    items: List[Any],
    k: int,
    led_min: int,
    led_max: int,
    normalize_case: bool,
    progress_label: str,
    n_jobs: int,
    chunksize: int,
    item_lengths: Optional[List[int]] = None,
    do_ledk: bool = False,
    do_led_parts: bool = False,
) -> Tuple[
    List[float],
    Dict[int, List[int]],
    Dict[int, List[int]],
    Dict[int, List[int]],
    Dict[int, List[int]],
    Dict[str, float],
]:
    """
    Compute LD-K (exact) plus optional LED counts (exact) for all items.

    Returns:
      ldk_values,
      led_by_d,
      led_same_by_d,
      led_short_by_d,
      led_long_by_d,
      timing dict with keys: {'ldk','led_counts','led_parts'}

    Timing semantics:
      - 'ldk': time spent computing the K-nearest mean distance (OLD/PLD).
      - 'led_counts': time spent computing LED total counts (LED2..D).
      - 'led_parts': incremental time spent computing LED parts (same/short/long).
    """
    if k <= 0:
        raise ValueError("k must be >= 1 for LD-K.")
    if led_min < 1:
        raise ValueError("led_min must be >= 1.")
    if led_max < led_min:
        raise ValueError("led_max must be >= led_min.")

    norms = _normalize_items(items, normalize_case=normalize_case)

    # Lengths used to split LED neighbors into same-length vs shorter vs longer.
    # For ORTHO metrics, this is character length. For PHONO metrics, callers pass
    # token counts via item_lengths.
    if item_lengths is None:
        lens = [len(x) for x in norms]
    else:
        if len(item_lengths) != len(norms):
            raise ValueError("item_lengths must have same length as items")
        lens = [int(L) for L in item_lengths]

    queries: List[Tuple[int, Any, int]] = [(i, norms[i], lens[i]) for i in range(len(norms))]

    # First occurrence length for each unique term (used for all items).
    term_len: Dict[Any, int] = {}
    for _, t, L in queries:
        if t not in term_len:
            term_len[t] = L

    n = len(norms)
    if n == 0:
        return ([], {}, {}, {}, {}, {"ldk": 0.0, "led_counts": 0.0, "led_parts": 0.0})

    k_eff = min(k, max(0, n - 1))

    do_ledk = bool(do_ledk)
    do_led_parts = bool(do_led_parts) if do_ledk else False

    # Multiplicity map (for duplicates/homophones)
    term_to_indices: Dict[Any, List[int]] = {}
    for i, t in enumerate(norms):
        term_to_indices.setdefault(t, []).append(i)
    unique_terms = list(term_to_indices.keys())
    term_mult = {t: len(idxs) for t, idxs in term_to_indices.items()}

    # ------------------------- timing accumulators -------------------------
    t_ldk = 0.0
    t_led_counts = 0.0
    t_led_parts = 0.0

    # Single-process path
    if n_jobs <= 1:
        # Reuse the same _compute_ldk_and_led_for_query function that workers use,
        # by setting the module-level globals directly instead of spawning processes.
        # This keeps the LD-K + LED logic in one place (DRY).
        global _G_ROOT, _G_K_EFF, _G_LED_MIN, _G_LED_MAX
        global _G_DO_LEDK, _G_DO_LED_PARTS, _G_TERM_LEN, _G_TERM_TO_INDICES, _G_TERM_MULT

        root = _BKNode(unique_terms[0])
        for t in unique_terms[1:]:
            _bk_insert(root, t)

        _G_ROOT = root
        _G_K_EFF = k_eff
        _G_LED_MIN = led_min
        _G_LED_MAX = led_max
        _G_DO_LEDK = do_ledk
        _G_DO_LED_PARTS = do_led_parts
        _G_TERM_LEN = term_len
        _G_TERM_TO_INDICES = term_to_indices
        _G_TERM_MULT = term_mult

        desc = f"[{progress_label}] Computing {progress_label}-{k_eff}"
        if do_ledk:
            desc += f" + LED{led_min}-{led_max}" + (" (parts)" if do_led_parts else "")

        results = [
            _compute_ldk_and_led_for_query(q_info)
            for q_info in tqdm(queries, desc=desc)
        ]

        ldk_vals = [r[0] for r in results]
        t_ldk = sum(r[5] for r in results)
        t_led_counts = sum(r[6] for r in results)
        t_led_parts = sum(r[7] for r in results)

        if not do_ledk:
            return (ldk_vals, {}, {}, {}, {}, {"ldk": t_ldk, "led_counts": 0.0, "led_parts": 0.0})

        led_by_d: Dict[int, List[int]] = {d: [] for d in range(led_min, led_max + 1)}
        led_same_by_d: Dict[int, List[int]] = {d: [] for d in range(led_min, led_max + 1)}
        led_short_by_d: Dict[int, List[int]] = {d: [] for d in range(led_min, led_max + 1)}
        led_long_by_d: Dict[int, List[int]] = {d: [] for d in range(led_min, led_max + 1)}

        if not do_led_parts:
            for _, counts, _, _, _, *_tim in results:
                for d in range(led_min, led_max + 1):
                    led_by_d[d].append(counts[d - led_min])
            return (ldk_vals, led_by_d, {}, {}, {}, {"ldk": t_ldk, "led_counts": t_led_counts, "led_parts": 0.0})

        for _, counts, counts_same, counts_short, counts_long, *_tim in results:
            for d in range(led_min, led_max + 1):
                idx = d - led_min
                led_by_d[d].append(counts[idx])
                led_same_by_d[d].append(counts_same[idx])
                led_short_by_d[d].append(counts_short[idx])
                led_long_by_d[d].append(counts_long[idx])

        return (ldk_vals, led_by_d, led_same_by_d, led_short_by_d, led_long_by_d, {"ldk": t_ldk, "led_counts": t_led_counts, "led_parts": t_led_parts})

    # Multi-process path
    ctx = mp.get_context("spawn")
    n_jobs_eff = min(int(n_jobs), os.cpu_count() or int(n_jobs))

    with ctx.Pool(
        processes=n_jobs_eff,
        initializer=_worker_init,
        initargs=(unique_terms, term_len, term_to_indices, term_mult, k_eff, led_min, led_max, do_ledk, do_led_parts),
    ) as pool:
        it = pool.imap(_compute_ldk_and_led_for_query, queries, chunksize=chunksize)
        results = list(
            tqdm(
                it,
                total=len(norms),
                desc=f"[{progress_label}] Computing {progress_label}-{k_eff}"
                     + (f" + LED{led_min}-{led_max}" if do_ledk else "")
                     + (f" (n_jobs={n_jobs_eff})"),
            )
        )

    ldk_vals = [r[0] for r in results]

    # Sum internal timings from workers
    t_ldk = sum(r[5] for r in results)
    t_led_counts = sum(r[6] for r in results)
    t_led_parts = sum(r[7] for r in results)

    if not do_ledk:
        return (ldk_vals, {}, {}, {}, {}, {"ldk": t_ldk, "led_counts": 0.0, "led_parts": 0.0})

    led_by_d: Dict[int, List[int]] = {d: [] for d in range(led_min, led_max + 1)}
    led_same_by_d: Dict[int, List[int]] = {d: [] for d in range(led_min, led_max + 1)}
    led_short_by_d: Dict[int, List[int]] = {d: [] for d in range(led_min, led_max + 1)}
    led_long_by_d: Dict[int, List[int]] = {d: [] for d in range(led_min, led_max + 1)}

    if not do_led_parts:
        for ldk, counts, _, _, _, *_tim in results:
            for d in range(led_min, led_max + 1):
                led_by_d[d].append(counts[d - led_min])
        return (ldk_vals, led_by_d, {}, {}, {}, {"ldk": t_ldk, "led_counts": t_led_counts, "led_parts": 0.0})

    for _, counts, counts_same, counts_short, counts_long, *_tim in results:
        for d in range(led_min, led_max + 1):
            idx = d - led_min
            led_by_d[d].append(counts[idx])
            led_same_by_d[d].append(counts_same[idx])
            led_short_by_d[d].append(counts_short[idx])
            led_long_by_d[d].append(counts_long[idx])

    return (ldk_vals, led_by_d, led_same_by_d, led_short_by_d, led_long_by_d, {"ldk": t_ldk, "led_counts": t_led_counts, "led_parts": t_led_parts})



def compute_old_and_led_ortho(
    words: List[str],
    k: int,
    led_min: int,
    led_max: int,
    n_jobs: int,
    chunksize: int,
    do_ledk: bool = False,
    do_led_parts: bool = False,
) -> Tuple[
    List[float],
    Dict[int, List[int]],
    Dict[int, List[int]],
    Dict[int, List[int]],
    Dict[int, List[int]],
    Dict[str, float],
]:
    """ORTHO: compute OLD-K and optional LED counts (case-insensitive)."""
    return compute_ldk_and_led(
        items=words,
        k=k,
        led_min=led_min,
        led_max=led_max,
        normalize_case=True,
        progress_label="OLD",
        n_jobs=n_jobs,
        chunksize=chunksize,
        do_ledk=do_ledk,
        do_led_parts=do_led_parts,
    )


def compute_pld_and_led_phono(
    prons: Optional[List[str]] = None,
    delimiter: Optional[str] = None,
    k: int = 20,
    led_min: int = 1,
    led_max: int = 5,
    n_jobs: int = 1,
    chunksize: int = 50,
    do_ledk: bool = False,
    do_led_parts: bool = False,
    # ----------------------------
    # Backward-compatible aliases
    # ----------------------------
    pron_strings: Optional[List[str]] = None,
    delimiter_phono: Optional[str] = None,
) -> Tuple[
    List[float],
    Dict[int, List[int]],
    Dict[int, List[int]],
    Dict[int, List[int]],
    Dict[int, List[int]],
    Dict[str, float],
]:
    """PHONO: compute PLD-K and optional LED counts at the phoneme-token level."""
    if prons is None and pron_strings is not None:
        prons = pron_strings
    if delimiter is None and delimiter_phono is not None:
        delimiter = delimiter_phono
    if prons is None:
        raise TypeError("compute_pld_and_led_phono(): must provide 'prons' (or 'pron_strings').")

    toks_list = tokenize_strings(prons, delimiter=delimiter)
    token_lens = [len(toks) for toks in toks_list]
    encoded, _token_to_char = encode_token_lists_to_pua_strings(toks_list)

    return compute_ldk_and_led(
        items=encoded,
        k=k,
        led_min=led_min,
        led_max=led_max,
        normalize_case=False,
        progress_label="PLD",
        n_jobs=n_jobs,
        chunksize=chunksize,
        item_lengths=token_lens,
        do_ledk=do_ledk,
        do_led_parts=do_led_parts,
    )

# =============================================================================
# End-to-end pipeline
# =============================================================================

# ----------------------------------------------------------------------------
# TOP-LEVEL PIPELINE ORCHESTRATION
# Reads the lexicon, tokenizes orthography and pronunciation, computes ORTHO + PHONO
# neighbor metrics, and merges everything into a single output CSV.
# Practical notes:
#   - Orthography can be tokenized by characters (delimiter 'none') or by a chosen delimiter.
#   - Pronunciation tokenization is typically whitespace-delimited (e.g., ARPABET phones).
#   - LED flags control which LED columns are created and written to disk.
# ----------------------------------------------------------------------------

def run_neighbor_pipeline(
    lexicon_path: str,
    output_path: str,
    orth_col: str,
    pron_col: str,
    delimiter_phono: Optional[str] = " ",
    delimiter_ortho: Optional[str] = None,
    test_sample: Optional[int] = None,
    oldk: Optional[int] = None,
    pldk: Optional[int] = None,
    old_count: Optional[int] = None,
    pld_count: Optional[int] = None,
    old_parts: bool = False,
    pld_parts: bool = False,
    n_jobs: int = 1,
    chunksize: int = 50,
    show_progress: bool = False,
) -> None:
    """Load lexicon CSV, compute neighbor metrics, optional OLD-k/PLD-k, optional LED counts, merge, save."""

    # Validate LED parts dependencies (CLI should enforce now, but we keep a defensive check here). It does 
    # not make sense to specify parts if counts are not already specified, and we don't want to just turn 
    # those on, because the user should specify the degree of LEDistances we will go (from 2..xld_count)
    if old_parts and old_count is None:
        raise ValueError("--old-parts requires --old-count.")
    if pld_parts and pld_count is None:
        raise ValueError("--pld-parts requires --pld-count.")

    df = pd.read_csv(
        lexicon_path,
        dtype=str,
        # do NOT treat strings like 'null', 'NA', 'true' as missing or anything other than strings
        keep_default_na=False,
        na_filter=False,
    )

    # error if user has not  correctly specified form and pron columns. WE SHOULD CHANGE THIS SO THAT 
    # IF ONE OF THESE 2 IS NOT SPECIFIED IT IS SKIPPED RATHER THAN CHECKED. 
    if orth_col not in df.columns:
        raise ValueError(f"orth_col '{orth_col}' not found in columns: {list(df.columns)}")
    if pron_col not in df.columns:
        raise ValueError(f"pron_col '{pron_col}' not found in columns: {list(df.columns)}")

    # When test_sample is None we ignore it
    if test_sample is not None: 
        if test_sample <= 0:
            raise ValueError("--test-sample must be a positive integer.")
        if test_sample < len(df):
            df = df.sample(n=test_sample, random_state=123).reset_index(drop=True)

    df[orth_col] = df[orth_col].fillna("").astype(str)
    df[pron_col] = df[pron_col].fillna("").astype(str)

    # 1) Neighbor metrics (PHONO)
    t_script0 = time.perf_counter()
    # timings = TimingLog()
    # Derive lexicon/sample metadata from the INPUT lexicon filename (not the output filename)
    lex_label, sample_size, run_id = _parse_lexicon_filename(lexicon_path)
    
    timings = TimingLog(
        tool="fastlex",
        lexicon=lex_label,
        items=len(df),
        sample_size=sample_size,
        run=run_id,
    )

    print("Computing PHONO neighbor metrics...")
    pron_strings = df[pron_col].tolist()

    # Save single-character (PUA) encoding of pronunciation tokens (for inspection/reproducibility)
    # We do this even when delimiter is none, implying single character elements. But this does not 
    # hurt anything -- overhead is tiny -- and makes it consistent always. 
    pron_col_enc = f"{pron_col}_enc"
    try:
        toks_list_enc = tokenize_strings(pron_strings, delimiter=delimiter_phono)
        pron_encoded, _token_map = encode_token_lists_to_pua_strings(toks_list_enc)
        df[pron_col_enc] = pron_encoded
    except Exception as e:
        print(f"[warn] Could not create pronunciation encoding column '{pron_col_enc}': {e}", flush=True)

    # call compute_neighbor_counts for phono
    neighbor_df_ph = compute_neighbor_counts(pron_strings, delimiter=delimiter_phono, label="PHONO", show_progress=show_progress, timings=timings, key_suffix="_ph")
    neighbor_df_ph = neighbor_df_ph.rename(columns={
        "length": "length_ph",
        "das_nb": "das_nb_ph",
        "deletion_neighbors": "dels_ph",
        "addition_neighbors": "adds_ph",
        "substitution_neighbors": "subs_ph",
        "homoforms": "homoforms_ph",
        "cohorts": "cohorts_ph",
        "nohorts": "nohorts_ph",
    })

    # 2) Neighbor metrics (ORTHO)
    print("Computing ORTHO neighbor metrics...")
    word_strings = df[orth_col].tolist()
    
    # call compute_neighbor_counts for ortho
    neighbor_df_or = compute_neighbor_counts(word_strings, delimiter=delimiter_ortho, label="ORTHO", show_progress=show_progress, timings=timings, key_suffix="_or")
    neighbor_df_or = neighbor_df_or.rename(columns={
        "length": "length_or",
        "das_nb": "das_nb_or",
        "deletion_neighbors": "dels_or",
        "addition_neighbors": "adds_or",
        "substitution_neighbors": "subs_or",
        "homoforms": "homoforms_or",
        "cohorts": "cohorts_or",
        "nohorts": "nohorts_or",
    })


    # 3) ORTHO: optional OLD-k and/or LED counts (2..old_count)
    do_led_or = old_count is not None
    do_led_or_parts = bool(old_parts) if do_led_or else False
    led_or_max = int(old_count) if do_led_or else 2  # default to 2 (= led_min) when LED not requested

    if oldk is not None or do_led_or:
        k_for_or = int(oldk) if oldk is not None else 1  # if only LED requested, compute dummy OLD-1 and discard

        t0 = time.perf_counter()
        old_vals, led_or, led_or_same, led_or_shorter, led_or_longer, t_or = compute_old_and_led_ortho(
            words=word_strings,
            k=k_for_or,
            led_min=2,
            led_max=led_or_max,
            n_jobs=n_jobs,
            chunksize=chunksize,
            do_ledk=do_led_or,
            do_led_parts=do_led_or_parts,
        )
        dt_wall = time.perf_counter() - t0
    
        # Record separate internal timings (these are the sums of per-query timers).
        if oldk is not None:
            timings.add("oldk_or", float(t_or.get("ldk", 0.0)))
        if do_led_or:
            timings.add("old_count_or", float(t_or.get("led_counts", 0.0)))
        if do_led_or_parts:
            timings.add("old_parts_or", float(t_or.get("led_parts", 0.0)))
    
        # Console summary (keep one combined wall-clock timer for the whole call)
        if oldk is not None and not do_led_or:
            print(f"[ORTHO] OLD{int(oldk)}\nOLD{int(oldk)} done: {_fmt_elapsed(dt_wall)}", flush=True)
        elif oldk is None and do_led_or and not do_led_or_parts:
            print(f"[ORTHO] LED2-LED{led_or_max} counts\nLED counts done: {_fmt_elapsed(dt_wall)}", flush=True)
        elif oldk is None and do_led_or and do_led_or_parts:
            print(f"[ORTHO] LED2-LED{led_or_max} counts (parts)\nLED counts+parts done: {_fmt_elapsed(dt_wall)}", flush=True)
        elif oldk is not None and do_led_or and not do_led_or_parts:
            print(f"[ORTHO] OLD{int(oldk)} + LED2-LED{led_or_max}\nOLD+LED done: {_fmt_elapsed(dt_wall)}", flush=True)
        else:
            print(f"[ORTHO] OLD{int(oldk)} + LED2-LED{led_or_max} (parts)\nOLD+LED+parts done: {_fmt_elapsed(dt_wall)}", flush=True)
    
        if oldk is not None:
            old_col = f"OLD{int(oldk)}_or"
            print(f"Computed {old_col}.")
            neighbor_df_or[old_col] = old_vals
    
        if do_led_or:
            print(f"Computed ORTHO LED2-LED{led_or_max} counts.")
            for d in range(2, led_or_max + 1):
                neighbor_df_or[f"LED{d}_or"] = led_or[d]
                if do_led_or_parts:
                    neighbor_df_or[f"LED{d}_or_same"]    = led_or_same[d]
                    neighbor_df_or[f"LED{d}_or_shorter"] = led_or_shorter[d]
                    neighbor_df_or[f"LED{d}_or_longer"]  = led_or_longer[d]
    
    
    
    # 4) PHONO: optional PLD-k and/or LED counts (2..pld_count)
    do_led_ph = pld_count is not None
    do_led_ph_parts = bool(pld_parts) if do_led_ph else False
    led_ph_max = int(pld_count) if do_led_ph else 2  # default to 2 (= led_min) when LED not requested
    
    if pldk is not None or do_led_ph:
        k_for_ph = int(pldk) if pldk is not None else 1  # if only LED requested, compute dummy PLD-1 and discard
    
        t0 = time.perf_counter()
        pld_vals, led_ph, led_ph_same, led_ph_shorter, led_ph_longer, t_ph = compute_pld_and_led_phono(
            pron_strings=pron_strings,
            delimiter_phono=delimiter_phono,
            k=k_for_ph,
            led_min=2,
            led_max=led_ph_max,
            n_jobs=n_jobs,
            chunksize=chunksize,
            do_ledk=do_led_ph,
            do_led_parts=do_led_ph_parts,
        )
        dt_wall = time.perf_counter() - t0
    
        # Record separate internal timings (sums of per-query timers).
        if pldk is not None:
            timings.add("pldk_ph", float(t_ph.get("ldk", 0.0)))
        if do_led_ph:
            timings.add("pld_count_ph", float(t_ph.get("led_counts", 0.0)))
        if do_led_ph_parts:
            timings.add("pld_parts_ph", float(t_ph.get("led_parts", 0.0)))
    
        if pldk is not None and not do_led_ph:
            print(f"[PHONO] PLD{int(pldk)}\nPLD-k done: {_fmt_elapsed(dt_wall)}", flush=True)
        elif pldk is None and do_led_ph and not do_led_ph_parts:
            print(f"[PHONO] LED2..LED{led_ph_max} counts\npld-count done: {_fmt_elapsed(dt_wall)}", flush=True)
        elif pldk is None and do_led_ph and do_led_ph_parts:
            print(f"[PHONO] LED2..LED{led_ph_max} counts + parts\npld-count+parts done: {_fmt_elapsed(dt_wall)}", flush=True)
        elif pldk is not None and do_led_ph and not do_led_ph_parts:
            print(f"[PHONO] PLD{int(pldk)} + LED2..LED{led_ph_max} counts\npldk+pld-count done: {_fmt_elapsed(dt_wall)}", flush=True)
        else:
            print(f"[PHONO] PLD{int(pldk)} + LED2..LED{led_ph_max} counts + parts\npldk+pld-count+parts done: {_fmt_elapsed(dt_wall)}", flush=True)
    
        if pldk is not None:
            pld_col = f"PLD{int(pldk)}_ph"
            print(f"Computed {pld_col}.")
            neighbor_df_ph[pld_col] = pld_vals
    
        if do_led_ph:
            print(f"Computed PHONO LED2-LED{led_ph_max} counts.")
            for d in range(2, led_ph_max + 1):
                neighbor_df_ph[f"LED{d}_ph"] = led_ph[d]
                if do_led_ph_parts:
                    neighbor_df_ph[f"LED{d}_ph_same"]    = led_ph_same[d]
                    neighbor_df_ph[f"LED{d}_ph_shorter"] = led_ph_shorter[d]
                    neighbor_df_ph[f"LED{d}_ph_longer"]  = led_ph_longer[d]


    # 5) Merge by row alignment and save
    ph_cols = ["length_ph", "cohorts_ph", "nohorts_ph", "das_nb_ph", "dels_ph", "adds_ph", "subs_ph", "homoforms_ph"]
    if pldk is not None:
        ph_cols.append(f"PLD{int(pldk)}_ph")
    if do_led_ph:
        ph_cols += [f"LED{d}_ph" for d in range(2, led_ph_max + 1)]
        if do_led_ph_parts:
            ph_cols += [f"LED{d}_ph_same" for d in range(2, led_ph_max + 1)]
            ph_cols += [f"LED{d}_ph_shorter" for d in range(2, led_ph_max + 1)]
            ph_cols += [f"LED{d}_ph_longer" for d in range(2, led_ph_max + 1)]

    or_cols = ["length_or", "cohorts_or", "nohorts_or", "das_nb_or", "dels_or", "adds_or", "subs_or", "homoforms_or"]
    if oldk is not None:
        or_cols.append(f"OLD{int(oldk)}_or")
    if do_led_or:
        or_cols += [f"LED{d}_or" for d in range(2, led_or_max + 1)]
        if do_led_or_parts:
            or_cols += [f"LED{d}_or_same" for d in range(2, led_or_max + 1)]
            or_cols += [f"LED{d}_or_shorter" for d in range(2, led_or_max + 1)]
            or_cols += [f"LED{d}_or_longer" for d in range(2, led_or_max + 1)]

    df_out = df.copy()

    # Join computed neighbor metrics onto the original dataframe.
    # The input lexicon may already contain columns with the same names
    # (e.g., from a previous run). To avoid pandas' "columns overlap" error,
    # we proactively drop any overlapping columns before joining.
    for _cols, _ndf in ((ph_cols, neighbor_df_ph), (or_cols, neighbor_df_or)):
        overlap = [c for c in _cols if c in df_out.columns]
        if overlap:
            df_out = df_out.drop(columns=overlap)

        # Align by row order/index (neighbor dfs are constructed in the same order as df)
        _ndf_aligned = _ndf.reindex(df_out.index)
        df_out = df_out.join(_ndf_aligned[_cols])


    # 6) Uniqueness points (UP)
    # Only compute if the corresponding columns are present.
    # UP is 1-based; if not unique by the final position, UP = length + 1.
    if orth_col and (orth_col in df_out.columns):
        try:
            t0 = time.perf_counter()
            df_out["UP_or"] = uniqueness_points(df_out[orth_col].astype(str).tolist())
            dt = time.perf_counter() - t0
            print(f"[ORTHO] uniqueness point (UP)\nUPs done: {_fmt_elapsed(dt)}", flush=True)
            timings.add("ups_or", dt)
        except Exception as e:
            print(f"[warn] Could not compute UP_or: {e}", flush=True)

    if pron_col and (pron_col in df_out.columns):
        try:
            t0 = time.perf_counter()
            pron_col_enc = f"{pron_col}_enc"
            if pron_col_enc in df_out.columns:
                ph_for_up = df_out[pron_col_enc].astype(str).tolist()
            else:
                # Fallback: tokenize + PUA-encode on the fly (measures UP in tokens)
                toks_list_enc = tokenize_strings(df_out[pron_col].astype(str).tolist(), delimiter=delimiter_phono)
                ph_for_up, _ = encode_token_lists_to_pua_strings(toks_list_enc)
            df_out["UP_ph"] = uniqueness_points(ph_for_up)
            dt = time.perf_counter() - t0
            print(f"[PHONO] uniqueness point (UP)\nUPs done: {_fmt_elapsed(dt)}", flush=True)
            timings.add("ups_ph", dt)
        except Exception as e:
            print(f"[warn] Could not compute UP_ph: {e}", flush=True)

    df_out.to_csv(output_path, index=False)
    print(f"Saved: {output_path}")

    # Write timings CSV next to the output file
    try:
        out_p = Path(output_path)
        timings_path = str(out_p.with_name(out_p.stem + "_timings" + out_p.suffix))
        timings.add("all", time.perf_counter() - t_script0)
        timings.to_csv(timings_path)
        print(f"Timings saved: {timings_path}")
    except Exception as e:
        print(f"[warn] Could not write timings CSV: {e}", flush=True)
    print(f"Full script done.: {_fmt_elapsed(time.perf_counter() - t_script0)}", flush=True)
    print("Done.")
    

# =============================================================================
# CLI
# =============================================================================

# ----------------------------------------------------------------------------
# COMMAND-LINE INTERFACE
# Expensive edit-distance summaries are OPT-IN:
#   --oldk K      => OLD-K (mean edit distance to K nearest orthographic neighbors)
#   --pldk K      => PLD-K (mean edit distance to K nearest phonological neighbors)
#   --old-count D => orthographic LED2..LED{D} counts (exact distance counts; LED1 is redundant with basic neighbor metrics)
#   --pld-count D => phonological LED2..LED{D} counts (exact distance counts; LED1 is redundant with basic neighbor metrics)
#   --old-parts / --pld-parts add same/shorter/longer splits for each LED{d} (requires the corresponding *-count flag).
# ----------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compute neighbor metrics (phono + ortho), with optional OLD-k / PLD-k and optional LED count columns."
    )

    parser.add_argument("--lexicon-path", required=True, help="Path to input lexicon CSV.")
    parser.add_argument("--output-path", required=True, help="Path to output CSV with metrics appended.")

    parser.add_argument("--orth-col", required=True, help="Column name for orthographic form.")
    parser.add_argument("--pron-col", required=True, help="Column name for pronunciation / phonology.")

    parser.add_argument(
        "--delimiter-phono",
        default="space",
        help="Delimiter for phonology tokenization: none | space | whitespace | literal string. Default: space."
    )
    parser.add_argument(
        "--delimiter-ortho",
        default="none",
        help="Delimiter for orthography tokenization: none | space | whitespace | literal string. Default: none."
    )

    parser.add_argument(
        "--test-sample",
        type=int,
        default=None,
        help="Optional subsample size for quick testing (random_state fixed)."
    )

    # --- OPT-IN distance summaries (mean edit distance to k nearest neighbors) ---
    parser.add_argument(
        "--oldk",
        type=int,
        default=None,
        help="Compute OLD-k (mean Levenshtein distance to k closest orthographic neighbors). Example: --oldk 20"
    )
    parser.add_argument(
        "--pldk",
        type=int,
        default=None,
        help="Compute PLD-k (mean Levenshtein distance to k closest phonological neighbors). Example: --pldk 20"
    )

    # --- OPT-IN LED count summaries (counts at exact distances 1..d) ---
    parser.add_argument(
        "--old-count",
        type=int,
        default=None,
        help="Compute orthographic LED2..LEDd neighbor counts (LED1 is redundant with basic neighbor metrics). Example: --old-count 5"
    )
    parser.add_argument(
        "--pld-count",
        type=int,
        default=None,
        help="Compute phonological LED2..LEDd neighbor counts (LED1 is redundant with basic neighbor metrics). Example: --pld-count 5"
    )

    # --- Optional LED 'parts' (requires corresponding *-count flag) ---
    parser.add_argument(
        "--old-parts",
        action="store_true",
        help="Also compute orthographic LED parts (same/shorter/longer) for each d. Requires --old-count."
    )
    parser.add_argument(
        "--pld-parts",
        action="store_true",
        help="Also compute phonological LED parts (same/shorter/longer) for each d. Requires --pld-count."
    )

    # --- Parallelism controls (used for OLD/PLD and LED) ---
    parser.add_argument(
        "--n-jobs",
        type=int,
        default=1,
        help="Number of worker processes for distance computations (OLD/PLD/LED). Use >1 to parallelize."
    )
    parser.add_argument(
        "--chunksize",
        type=int,
        default=50,
        help="Multiprocessing imap chunksize for distance computations (default 50)."
    )

    parser.add_argument(
        "--progress",
        action="store_true",
        help="Show per-part progress bars for basic neighbor metrics (may add slight overhead)."
    )

    t0 = time.perf_counter() # for overall time report at end
    args = parser.parse_args()
    n_jobs_eff = check_requested_cpus(args.n_jobs) # aims for 80% of available cpus, warns if you ask for too many, etc.

    # Validate dependencies and value ranges
    if args.old_parts and args.old_count is None:
        parser.error("--old-parts requires --old-count.")
    if args.pld_parts and args.pld_count is None:
        parser.error("--pld-parts requires --pld-count.")

    if args.oldk is not None and args.oldk <= 0:
        parser.error("--oldk must be a positive integer.")
    if args.pldk is not None and args.pldk <= 0:
        parser.error("--pldk must be a positive integer.")

    if args.old_count is not None and args.old_count < 2:
        parser.error("--old-count must be >= 2 (d=1 is redundant with the basic neighbor metrics).")
    if args.pld_count is not None and args.pld_count < 2:
        parser.error("--pld-count must be >= 2 (d=1 is redundant with the basic neighbor metrics).")

    delimiter_ph = parse_delimiter_arg(args.delimiter_phono)
    delimiter_or = parse_delimiter_arg(args.delimiter_ortho)

    # we could just insist people install it because it speeds things up so much, 
    # but maybe it will break in a future version, so this allows a fallback.
    if _HAVE_RAPIDFUZZ:
        print("RapidFuzz detected: using fast exact Levenshtein backend.")
    else:
        print("RapidFuzz not detected: using pure-Python Levenshtein (slower).")

    # Report/plan parallelism
    #    - Print messages listing which things will be done via parallelism and which not... 
    do_ortho_dims = bool(args.orth_col)
    do_phono_dims = bool(args.pron_col)
    do_ortho_edit = do_ortho_dims and (args.oldk is not None or args.old_count is not None)
    do_phono_edit = do_phono_dims and (args.pldk is not None or args.pld_count is not None)
    print_parallel_plan(
        n_jobs_req=int(args.n_jobs),
        n_jobs_eff=int(n_jobs_eff),
        do_ortho_edit=do_ortho_edit,
        do_phono_edit=do_phono_edit,
        do_ortho_dims=do_ortho_dims,
        do_phono_dims=do_phono_dims,
    )

    # Here's the main stuff
    run_neighbor_pipeline(
        lexicon_path=args.lexicon_path,
        output_path=args.output_path,
        orth_col=args.orth_col,
        pron_col=args.pron_col,
        delimiter_phono=delimiter_ph,
        delimiter_ortho=delimiter_or,
        test_sample=args.test_sample,
        oldk=args.oldk,
        pldk=args.pldk,
        old_count=args.old_count,
        pld_count=args.pld_count,
        old_parts=args.old_parts,
        pld_parts=args.pld_parts,
        n_jobs=n_jobs_eff,
        chunksize=args.chunksize,
        show_progress=args.progress,
    )

    elapsed = time.perf_counter() - t0
    h, rem = divmod(elapsed, 3600)
    m, s = divmod(rem, 60)
    print(f"Finished in {int(h)}:{int(m):02d}:{s:05.2f}")


if __name__ == "__main__":
    main()
