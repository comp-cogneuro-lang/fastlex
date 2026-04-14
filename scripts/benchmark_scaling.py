#!/usr/bin/env python3
"""
Benchmark FastLex processing time across lexicon sizes.

Runs fastlex.py on sampled lexica at increasing sizes (2k, 4k, 8k, 16k, 32k)
and on the full lexica, with two configurations:
  (1) Full pipeline: basic metrics + OLD-k + PLD-k + LED counts + LED parts
  (2) No-LED pipeline: basic metrics + OLD-k + PLD-k only

For each run, wall-clock time is recorded. Results are written to a tidy CSV
suitable for the scaling analysis reported in the paper (fitting T ~ k * N^p).

The script also collects per-operation timings from FastLex's own _timings.csv
output files, which break down time by operation (dims, oldk, pldk, led, etc.).

Usage:
  python benchmark_scaling.py --lexicon elp
  python benchmark_scaling.py --lexicon espal
  python benchmark_scaling.py --lexicon all

  # Override defaults:
  python benchmark_scaling.py --lexicon elp --n-jobs 8 --reps 3

Requirements:
  - fastlex.py must be in the parent directory (../fastlex.py)
  - Sampled lexica must be in ../sampled_lexica/<lexicon>/
  - Full lexica must be in ../data/
"""

from __future__ import annotations

import argparse
import csv
import os
import subprocess
import sys
import time
from pathlib import Path


# ---------------------------------------------------------------------------
# Lexicon configurations
# ---------------------------------------------------------------------------

LEXICON_CONFIGS = {
    "elp": {
        "full_path": "data/ELP_lex_min_missing_added.csv",
        "orth_col": "Word",
        "pron_col": "Pron_arpabet",
        "delimiter_phono": "space",
        "delimiter_ortho": "none",
        "sample_dir": "sampled_lexica/elp",
        # Sampled file pattern: elp-lex-{size}k-v{rep}.csv
        "sample_sizes_k": [2, 4, 8, 16, 32],
    },
    "espal": {
        "full_path": "data/espal-lex.csv",
        "orth_col": "word",
        "pron_col": "es_phon_structure",
        "delimiter_phono": "none",
        "delimiter_ortho": "none",
        "sample_dir": "sampled_lexica/espal",
        # Sampled file pattern: espal-lex-{size}k-v{rep}.csv
        "sample_sizes_k": [2, 4, 8, 16, 32, 64, 96],
    },
}


# ---------------------------------------------------------------------------
# Build fastlex command
# ---------------------------------------------------------------------------

def build_cmd(
    python: str,
    fastlex_path: str,
    lexicon_path: str,
    output_path: str,
    cfg: dict,
    n_jobs: int,
    include_led: bool,
) -> list[str]:
    """Build the fastlex.py command list."""
    cmd = [
        python, fastlex_path,
        "--lexicon-path", lexicon_path,
        "--output-path", output_path,
        "--orth-col", cfg["orth_col"],
        "--pron-col", cfg["pron_col"],
        "--delimiter-phono", cfg["delimiter_phono"],
        "--delimiter-ortho", cfg["delimiter_ortho"],
        "--n-jobs", str(n_jobs),
        "--progress",
        "--oldk", "20",
        "--pldk", "20",
    ]
    if include_led:
        cmd += ["--old-count", "5", "--old-parts"]
        cmd += ["--pld-count", "5", "--pld-parts"]
    return cmd


# ---------------------------------------------------------------------------
# Run one benchmark
# ---------------------------------------------------------------------------

def run_one(
    cmd: list[str],
    log_dir: Path,
    label: str,
) -> float:
    """Run a single FastLex invocation and return wall-clock seconds."""
    log_out = log_dir / f"{label}.stdout.txt"
    log_err = log_dir / f"{label}.stderr.txt"

    print(f"  Running: {label} ...", end="", flush=True)
    t0 = time.perf_counter()

    with open(log_out, "w", encoding="utf-8") as fo, \
         open(log_err, "w", encoding="utf-8") as fe:
        p = subprocess.run(cmd, stdout=fo, stderr=fe)

    elapsed = time.perf_counter() - t0

    if p.returncode != 0:
        print(f" FAILED (exit {p.returncode}) [{elapsed:.1f}s]")
        print(f"    See: {log_err}")
    else:
        print(f" OK [{elapsed:.1f}s]")

    return elapsed


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser(
        description="Benchmark FastLex scaling across lexicon sizes."
    )
    ap.add_argument(
        "--lexicon", required=True,
        choices=list(LEXICON_CONFIGS.keys()) + ["all"],
        help="Which lexicon to benchmark (or 'all')."
    )
    ap.add_argument("--n-jobs", type=int, default=8, help="Number of parallel workers (default: 8).")
    ap.add_argument("--reps", type=int, default=3, help="Number of sample replicates per size (default: 3).")
    ap.add_argument("--python", default=sys.executable, help="Python interpreter.")
    ap.add_argument("--skip-full", action="store_true", help="Skip the full (unsampled) lexicon runs.")
    ap.add_argument("--out-csv", default="benchmark_results.csv", help="Output CSV path.")
    ap.add_argument("--dry-run", action="store_true", help="Print commands without executing.")
    args = ap.parse_args()

    # Resolve paths relative to the dist/ directory (parent of scripts/)
    dist_dir = Path(__file__).resolve().parent.parent
    fastlex_path = str(dist_dir / "fastlex.py")

    if not Path(fastlex_path).exists():
        raise FileNotFoundError(f"fastlex.py not found at {fastlex_path}")

    # Output and log directories
    out_dir = dist_dir / "output" / "benchmark"
    log_dir = out_dir / "logs"
    out_dir.mkdir(parents=True, exist_ok=True)
    log_dir.mkdir(parents=True, exist_ok=True)

    # Which lexicons to benchmark
    if args.lexicon == "all":
        lexicons = list(LEXICON_CONFIGS.keys())
    else:
        lexicons = [args.lexicon]

    # Collect results
    results: list[dict] = []

    for lex_name in lexicons:
        cfg = LEXICON_CONFIGS[lex_name]
        print(f"\n{'='*60}")
        print(f"Benchmarking: {lex_name}")
        print(f"{'='*60}")

        # --- Sampled lexica at each size ---
        for size_k in cfg["sample_sizes_k"]:
            for rep in range(1, args.reps + 1):
                sample_file = f"{lex_name}-lex-{size_k:02d}k-v{rep:02d}.csv"
                sample_path = str(dist_dir / cfg["sample_dir"] / sample_file)

                if not Path(sample_path).exists():
                    print(f"  [skip] {sample_file} not found")
                    continue

                # Count actual rows (subtract 1 for header)
                with open(sample_path, "r") as f:
                    n_items = sum(1 for _ in f) - 1

                for include_led, pipeline_label in [(True, "full"), (False, "no_led")]:
                    label = f"{lex_name}_{size_k:02d}k_v{rep:02d}_{pipeline_label}"
                    out_csv = str(out_dir / f"{label}.csv")

                    cmd = build_cmd(
                        python=args.python,
                        fastlex_path=fastlex_path,
                        lexicon_path=sample_path,
                        output_path=out_csv,
                        cfg=cfg,
                        n_jobs=args.n_jobs,
                        include_led=include_led,
                    )

                    if args.dry_run:
                        print(f"  [dry-run] {label}: {' '.join(cmd)}")
                        continue

                    elapsed = run_one(cmd, log_dir, label)

                    results.append({
                        "lexicon": lex_name,
                        "source": sample_file,
                        "n_items": n_items,
                        "sample_size_k": size_k,
                        "replicate": rep,
                        "pipeline": pipeline_label,
                        "n_jobs": args.n_jobs,
                        "wall_seconds": f"{elapsed:.2f}",
                    })

        # --- Full lexicon ---
        if not args.skip_full:
            full_path = str(dist_dir / cfg["full_path"])
            if not Path(full_path).exists():
                print(f"  [skip] Full lexicon not found: {full_path}")
            else:
                with open(full_path, "r") as f:
                    n_items = sum(1 for _ in f) - 1

                for include_led, pipeline_label in [(True, "full"), (False, "no_led")]:
                    label = f"{lex_name}_full_{pipeline_label}"
                    out_csv = str(out_dir / f"{label}.csv")

                    cmd = build_cmd(
                        python=args.python,
                        fastlex_path=fastlex_path,
                        lexicon_path=full_path,
                        output_path=out_csv,
                        cfg=cfg,
                        n_jobs=args.n_jobs,
                        include_led=include_led,
                    )

                    if args.dry_run:
                        print(f"  [dry-run] {label}: {' '.join(cmd)}")
                        continue

                    elapsed = run_one(cmd, log_dir, label)

                    results.append({
                        "lexicon": lex_name,
                        "source": Path(full_path).name,
                        "n_items": n_items,
                        "sample_size_k": None,
                        "replicate": None,
                        "pipeline": pipeline_label,
                        "n_jobs": args.n_jobs,
                        "wall_seconds": f"{elapsed:.2f}",
                    })

    # --- Write results CSV ---
    if args.dry_run:
        print("\n[dry-run] No results to write.")
        return

    out_csv_path = dist_dir / "output" / args.out_csv
    fieldnames = [
        "lexicon", "source", "n_items", "sample_size_k", "replicate",
        "pipeline", "n_jobs", "wall_seconds",
    ]
    with open(out_csv_path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(results)

    print(f"\nResults written to: {out_csv_path}")
    print(f"Total runs: {len(results)}")


if __name__ == "__main__":
    main()
