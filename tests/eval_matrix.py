#!/usr/bin/env python3
"""eval_matrix.py — analyse the channel×load detection matrix.

For each /tmp/matrix-<channel>-<load>.csv + .json cell that exists, this
script slices the raw latency samples into idle and load windows using the
contention marks (c_idle1→c_load, c_load→c_load_end, c_idle2→c_end), then
computes three detection metrics:

  median_ratio   — median(load) / median(idle)   — how much load shifts the
                   central tendency.
  throughput_pct — (count_load/dur_load - count_idle/dur_idle) /
                   count_idle/dur_idle * 100  — rate change under load.
  AUC            — P(load sample > idle sample), the threshold-free effect
                   size equivalent to the Mann-Whitney U / Cliff's delta.
                   AUC ≈ 0.5 → no separation; →1 load is higher; →0 load
                   is lower.  Implemented via rank-based formula; no scipy.

The AUC is the headline number: it tells a reader at a glance which
(channel, load) pairs show meaningful contention sensing.

Usage:
    python3 tests/eval_matrix.py [--csv-dir /tmp]

Output:
    Per-cell detail lines, then a tidy channel×load matrix of AUC and
    median-ratio values.
"""

import argparse
import glob
import json
import os
import sys
from pathlib import Path

import numpy as np


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

# The three canonical cells.  The "expected" load for each channel is what
# the channel was designed to sense; the cross-cell is the control.
CHANNELS = ["read", "cache", "flush"]
LOADS    = ["dd", "burn"]
CELL_DIR = "/tmp"


# ---------------------------------------------------------------------------
# AUC (Mann-Whitney)
# ---------------------------------------------------------------------------

def auc_mw(a: np.ndarray, b: np.ndarray) -> float:
    """Return AUC = P(b > a) via the rank-based formula.

    Uses the identity:  U = sum of ranks of b in merged array  minus
    n_b*(n_b+1)/2,  then AUC = U / (n_a * n_b).

    Ties are broken by averaging ranks (midrank), which is the standard
    Mann-Whitney convention.  No scipy required.

    Parameters
    ----------
    a : idle samples
    b : load samples

    Returns
    -------
    float in [0, 1] — P(load > idle)
    """
    if len(a) == 0 or len(b) == 0:
        return float("nan")

    n_a, n_b = len(a), len(b)
    # Merge and rank
    combined = np.concatenate([a, b])
    labels   = np.concatenate([np.zeros(n_a), np.ones(n_b)])  # 0=idle, 1=load
    order    = np.argsort(combined, kind="stable")
    sorted_labels = labels[order]
    sorted_vals   = combined[order]

    # Assign midranks (1-based)
    ranks = np.empty(len(combined), dtype=float)
    i = 0
    while i < len(combined):
        j = i
        while j < len(combined) and sorted_vals[j] == sorted_vals[i]:
            j += 1
        midrank = (i + j + 1) / 2.0  # average of 1-based positions i+1..j
        ranks[i:j] = midrank
        i = j

    # U statistic for group b (load)
    u_b = ranks[sorted_labels == 1].sum() - n_b * (n_b + 1) / 2.0
    return float(u_b / (n_a * n_b))


# ---------------------------------------------------------------------------
# Data loading and slicing
# ---------------------------------------------------------------------------

def load_cell(csv_path: str, json_path: str):
    """Load a cell's CSV and marks JSON.

    Returns (timestamps_ms, latencies_us, marks_dict).
    marks_dict maps name→t_ms.
    """
    data = np.genfromtxt(csv_path, delimiter=",", skip_header=1)
    if data.ndim == 1:
        data = data.reshape(1, -1)
    timestamps = data[:, 0]
    latencies  = data[:, 1]

    with open(json_path) as fh:
        marks_list = json.load(fh)
    marks = {m["name"]: m["t_ms"] for m in marks_list}

    return timestamps, latencies, marks


def slice_contention(timestamps, latencies, marks):
    """Slice samples into idle and load windows from contention marks.

    idle = [c_idle1, c_load) ∪ [c_idle2, c_end)
    load = [c_load, c_load_end)

    Returns (idle_lat, load_lat, idle_dur_ms, load_dur_ms).
    """
    required = {"c_idle1", "c_load", "c_load_end", "c_idle2", "c_end"}
    missing  = required - set(marks)
    if missing:
        raise ValueError(f"Missing marks: {missing}")

    t_idle1    = marks["c_idle1"]
    t_load     = marks["c_load"]
    t_load_end = marks["c_load_end"]
    t_idle2    = marks["c_idle2"]
    t_end      = marks["c_end"]

    mask_idle = (
        ((timestamps >= t_idle1) & (timestamps < t_load)) |
        ((timestamps >= t_idle2) & (timestamps < t_end))
    )
    mask_load = (timestamps >= t_load) & (timestamps < t_load_end)

    idle_dur = (t_load - t_idle1) + (t_end - t_idle2)   # ms
    load_dur = t_load_end - t_load                        # ms

    return latencies[mask_idle], latencies[mask_load], idle_dur, load_dur


# ---------------------------------------------------------------------------
# Per-cell analysis
# ---------------------------------------------------------------------------

def analyse_cell(channel: str, load: str, csv_dir: str):
    """Analyse one (channel, load) cell.

    Returns a dict with keys: auc, median_ratio, throughput_pct,
    n_idle, n_load, or None if data files are missing.
    """
    csv_path  = os.path.join(csv_dir, f"matrix-{channel}-{load}.csv")
    json_path = os.path.join(csv_dir, f"matrix-{channel}-{load}.json")

    if not (os.path.exists(csv_path) and os.path.exists(json_path)):
        return None

    timestamps, latencies, marks = load_cell(csv_path, json_path)
    idle_lat, load_lat, idle_dur, load_dur = slice_contention(
        timestamps, latencies, marks
    )

    if len(idle_lat) == 0 or len(load_lat) == 0:
        return {
            "auc": float("nan"), "median_ratio": float("nan"),
            "throughput_pct": float("nan"),
            "n_idle": len(idle_lat), "n_load": len(load_lat),
        }

    med_idle = float(np.median(idle_lat))
    med_load = float(np.median(load_lat))
    median_ratio = med_load / med_idle if med_idle != 0 else float("nan")

    rate_idle = len(idle_lat) / idle_dur if idle_dur > 0 else float("nan")
    rate_load = len(load_lat) / load_dur if load_dur > 0 else float("nan")
    throughput_pct = (
        (rate_load - rate_idle) / rate_idle * 100.0
        if rate_idle and rate_idle != 0 else float("nan")
    )

    auc = auc_mw(idle_lat, load_lat)

    return {
        "auc":            auc,
        "median_ratio":   median_ratio,
        "throughput_pct": throughput_pct,
        "n_idle":         len(idle_lat),
        "n_load":         len(load_lat),
        "med_idle_us":    med_idle,
        "med_load_us":    med_load,
    }


# ---------------------------------------------------------------------------
# Pretty printing
# ---------------------------------------------------------------------------

def fmt_auc(v):
    if v is None or (isinstance(v, float) and np.isnan(v)):
        return "  n/a  "
    return f" {v:.3f} "


def fmt_ratio(v):
    if v is None or (isinstance(v, float) and np.isnan(v)):
        return "  n/a  "
    return f" {v:5.2f}x"


def fmt_pct(v):
    if v is None or (isinstance(v, float) and np.isnan(v)):
        return "  n/a  "
    return f"{v:+7.1f}%"


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--csv-dir", default=CELL_DIR,
                        help="Directory containing matrix-*.csv / .json files "
                             f"(default: {CELL_DIR})")
    args = parser.parse_args()

    csv_dir = args.csv_dir

    # Discover which cells actually exist
    available = set()
    for ch in CHANNELS:
        for ld in LOADS:
            csv  = os.path.join(csv_dir, f"matrix-{ch}-{ld}.csv")
            json_ = os.path.join(csv_dir, f"matrix-{ch}-{ld}.json")
            if os.path.exists(csv) and os.path.exists(json_):
                available.add((ch, ld))

    if not available:
        print(f"No matrix-*.csv / .json files found in {csv_dir}")
        print("Run tests/matrix.sh first (or copy /tmp/frost-*.csv).")
        sys.exit(1)

    print("=" * 70)
    print("FROST channel×load detection matrix")
    print("=" * 70)
    print(f"Data directory : {csv_dir}")
    print(f"Cells found    : {len(available)}")
    print()

    # Collect results
    results = {}
    for ch in CHANNELS:
        for ld in LOADS:
            key = (ch, ld)
            if key not in available:
                results[key] = None
                continue
            print(f"--- {ch} × {ld} ---")
            r = analyse_cell(ch, ld, csv_dir)
            results[key] = r
            if r:
                print(f"  n_idle={r['n_idle']:6d}  med_idle={r['med_idle_us']:8.1f} µs")
                print(f"  n_load={r['n_load']:6d}  med_load={r['med_load_us']:8.1f} µs")
                print(f"  median_ratio   = {r['median_ratio']:.4f}x  "
                      f"({'↑ load higher' if r['median_ratio'] > 1 else '↓ load lower'})")
                print(f"  throughput_pct = {r['throughput_pct']:+.1f}%")
                print(f"  AUC            = {r['auc']:.4f}  "
                      f"(0.5=random, →1 load higher latency, →0 load lower latency)")
            print()

    # ---- AUC matrix ----
    print("=" * 70)
    print("AUC matrix  (P(load_latency > idle_latency); 0.5 = no effect)")
    print("=" * 70)
    col_w = 12
    header = f"{'channel':<10}" + "".join(f"{ld:^{col_w}}" for ld in LOADS)
    print(header)
    print("-" * len(header))
    for ch in CHANNELS:
        row = f"{ch:<10}"
        for ld in LOADS:
            r = results.get((ch, ld))
            if r is None:
                cell = "   -    "
            else:
                auc = r["auc"]
                cell = fmt_auc(auc)
            row += f"{cell:^{col_w}}"
        print(row)
    print()

    # ---- Median-ratio matrix ----
    print("=" * 70)
    print("Median-ratio matrix  (median_load / median_idle; 1.0 = no effect)")
    print("=" * 70)
    print(header)
    print("-" * len(header))
    for ch in CHANNELS:
        row = f"{ch:<10}"
        for ld in LOADS:
            r = results.get((ch, ld))
            if r is None:
                cell = "   -    "
            else:
                cell = fmt_ratio(r["median_ratio"])
            row += f"{cell:^{col_w}}"
        print(row)
    print()

    # ---- Throughput-%change matrix ----
    print("=" * 70)
    print("Throughput %change matrix  (sample-rate under load vs idle)")
    print("=" * 70)
    print(header)
    print("-" * len(header))
    for ch in CHANNELS:
        row = f"{ch:<10}"
        for ld in LOADS:
            r = results.get((ch, ld))
            if r is None:
                cell = "   -    "
            else:
                cell = fmt_pct(r["throughput_pct"])
            row += f"{cell:^{col_w}}"
        print(row)
    print()

    # ---- Interpretation guide ----
    print("=" * 70)
    print("Interpretation")
    print("=" * 70)
    print("  AUC > 0.70 or < 0.30  — strong detection")
    print("  AUC > 0.60 or < 0.40  — moderate detection")
    print("  AUC ≈ 0.50             — no detection (channel blind to this load)")
    print()
    print("Expected pattern:")
    print("  read   × dd   — strong (SSD read timing perturbed by disk writes)")
    print("  read   × burn — weak   (CPU load does not hit SSD read path)")
    print("  cache  × burn — strong (LLC thrash raises cache-probe latency)")
    print("  cache  × dd   — weak   (disk I/O alone does not evict LLC)")
    print("  flush  × dd   — strong (Firefox write-flush timing perturbed by disk)")
    print("  flush  × burn — weak   (CPU burn alone does not slow flush)")


if __name__ == "__main__":
    main()
