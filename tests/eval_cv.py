"""Leakage-robust cross-validation evaluator for the FROST timing-channel fingerprinter.

Three evaluation schemes are run in order to expose whether reported accuracies
are real or inflated by temporal/session drift:

  (a) Random RepeatedStratifiedKFold -- the optimistic baseline used in analyze.py.
      Windows are randomly assigned to folds so a window collected at t=10s and one
      collected at t=170s may both appear in the *same training fold*.  When
      hardware features drift monotonically over a session (e.g. SSD caches
      warming, LLC settling), the classifier silently learns "early vs late" and
      conflates that temporal gradient with the idle/nunl class signal.  This
      produces inflated accuracy.

  (b) Leakage-robust GroupKFold with contiguous time blocks -- windows are sorted
      by collection time and divided into N_BLOCKS contiguous groups.  GroupKFold
      guarantees no group appears in both train and test, so the classifier must
      generalise *across* time epochs.  Two block counts are reported: 4 blocks
      (captures local drift) and 2 blocks / half-split (captures the full
      session-length drift; this is the most conservative estimate).

  (c) Idle-only NULL control -- all idle windows are relabelled with two fake
      binary classes (first-half vs second-half, and alternating even/odd) and
      then classified using the *same* random StratifiedKFold as scheme (a).
      A ground-truth null MUST score ~50%.  If it scores well above 50% it proves
      that the random KFold is inflated by session drift; the model is not learning
      the true class distinction at all, it is learning "early-session vs
      late-session".  The alternating null (neighbouring windows get opposite
      labels) acts as a sanity check: it should score at or below chance because
      consecutive windows look alike (positive autocorrelation in latency features).

Why this matters:
  The 16 idle and 16 nunl windows in frost-read.csv were collected sequentially
  over ~165 seconds.  Summary statistics (p95, mean latency, throughput) show a
  clear monotonic trend across the session: early windows have higher latency and
  lower throughput than late windows.  Random KFold mixes early and late windows
  in training, so the model learns this temporal gradient for free.  A 93% random
  CV accuracy accompanied by a 93% null-half score means the classifier is
  essentially fitting a time-index, not a fingerprint.

Usage:
    python3 tests/eval_cv.py              # defaults to 'read' channel
    python3 tests/eval_cv.py cache
    python3 tests/eval_cv.py flush
"""

import os
import sys
import json
import argparse
import warnings

import numpy as np

# Allow running from repo root or from tests/
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "analysis"))

from dataset import Recording          # noqa: E402
from features import build_matrix      # noqa: E402

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import (
    RepeatedStratifiedKFold,
    GroupKFold,
    cross_val_score,
)

# ── Configuration ─────────────────────────────────────────────────────────────

RF_PARAMS = dict(n_estimators=300, random_state=0, n_jobs=-1)

# Repeated random CV parameters
RANDOM_CV_SPLITS = 5
RANDOM_CV_REPEATS = 40

# Number of contiguous time blocks for the grouped split
N_BLOCKS_COARSE = 4   # moderate granularity
N_BLOCKS_STRICT = 2   # hardest: first-half train, second-half test (and vice versa)

# Leakage thresholds
NULL_HALF_LEAKAGE_THRESHOLD = 0.60    # null_half > this -> LEAKAGE SUSPECTED
GAP_LEAKAGE_THRESHOLD = 0.15         # |random - grouped_strict| > this -> LEAKAGE SUSPECTED

# ── Data loading ──────────────────────────────────────────────────────────────

def load_channel(channel: str):
    """Load CSV + marks for one channel; return (all_recs, groups_array).

    all_recs is sorted by collection order (pairs: idle_i, nunl_i for i=1..N).
    groups_array is a parallel integer array where each pair shares a group id,
    so GroupKFold can honour the paired structure.
    """
    csv_path   = f"/tmp/frost-{channel}.csv"
    marks_path = f"/tmp/frost-{channel}-marks.json"

    if not os.path.exists(csv_path):
        raise FileNotFoundError(
            f"{csv_path} not found.  Run channel_probe.js first."
        )

    # Read raw trace
    t_all, l_all = [], []
    with open(csv_path) as fh:
        fh.readline()  # skip header
        for line in fh:
            parts = line.split(",")
            if len(parts) < 2:
                continue
            try:
                t_all.append(float(parts[0]))
                l_all.append(float(parts[1]))
            except ValueError:
                pass
    t_all = np.asarray(t_all, dtype=np.float64)
    l_all = np.asarray(l_all, dtype=np.float64)

    with open(marks_path) as fh:
        marks = {m["name"]: m["t_ms"] for m in json.load(fh)}

    recs, groups = [], []
    pair_idx = 1
    while f"idle_{pair_idx}_s" in marks:
        for label in ("idle", "nunl"):
            key_s = f"{label}_{pair_idx}_s"
            key_e = f"{label}_{pair_idx}_e"
            if key_e not in marks:
                continue
            a, b = marks[key_s], marks[key_e]
            mask = (t_all >= a) & (t_all < b)
            wt, wl = t_all[mask], l_all[mask]
            if wl.size > 5:
                recs.append(Recording(
                    label=label,
                    t_ms=wt,
                    lat_us=wl,
                    window_s=(b - a) / 1000.0,
                ))
                groups.append(pair_idx)
        pair_idx += 1

    return recs, np.asarray(groups, dtype=int)


# ── Feature matrix ────────────────────────────────────────────────────────────

def prepare_matrix(recs):
    """Return (X, y) with NaN replaced by 0."""
    X, y, _ = build_matrix(recs)
    X = np.nan_to_num(X)
    return X, y


# ── Scheme (a): random repeated stratified KFold ──────────────────────────────

def eval_random_cv(X, y, n_splits=None, n_repeats=RANDOM_CV_REPEATS):
    """RepeatedStratifiedKFold with as many splits as the minority class allows."""
    n_min = min((y == c).sum() for c in np.unique(y))
    folds = min(n_splits or RANDOM_CV_SPLITS, int(n_min))
    if folds < 2:
        return None, None, folds

    clf = RandomForestClassifier(**RF_PARAMS)
    rcv = RepeatedStratifiedKFold(
        n_splits=folds, n_repeats=n_repeats, random_state=0
    )
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        scores = cross_val_score(clf, X, y, cv=rcv)
    return scores.mean(), scores.std(), folds


# ── Scheme (b): leakage-robust GroupKFold with contiguous time blocks ─────────

def assign_contiguous_blocks(n_samples: int, n_blocks: int) -> np.ndarray:
    """Assign windows to contiguous time-ordered blocks.

    Windows are assumed to already be sorted in collection order (they come
    out of load_channel in that order).  Block 0 = earliest samples, block
    n_blocks-1 = latest samples.  Each block is as balanced as possible and
    always contains samples from both classes because idle/nunl windows are
    interleaved in time (idle_i immediately precedes nunl_i).
    """
    return (np.arange(n_samples) * n_blocks) // n_samples


def eval_grouped_cv(X, y, n_blocks: int):
    """GroupKFold with contiguous time blocks."""
    block_ids = assign_contiguous_blocks(len(y), n_blocks)

    # Verify each block has at least both classes; degenerate if not
    for b in range(n_blocks):
        block_labels = y[block_ids == b]
        if len(np.unique(block_labels)) < 2:
            return None, None, n_blocks  # not enough diversity to score

    clf = RandomForestClassifier(**RF_PARAMS)
    gkf = GroupKFold(n_splits=n_blocks)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        scores = cross_val_score(clf, X, y, cv=gkf, groups=block_ids)
    return scores.mean(), scores.std(), n_blocks


# ── Scheme (c): idle-only NULL control ────────────────────────────────────────

def eval_null_control(recs, n_repeats=RANDOM_CV_REPEATS):
    """Classify idle-only windows with fake temporal labels.

    Two fake labelling schemes:
      half  -- first-half idle windows get 'A', second-half get 'B'.
               Detects whether the random KFold is exploiting session-level
               feature drift (a monotonic trend over the full session).
      alt   -- even-indexed idle windows get 'A', odd-indexed get 'B'.
               Neighbouring windows share a label so a classifier that learns
               local autocorrelations should score *below* chance (they look
               alike, not different).  A value well above 50% here would be
               unusual and would suggest a non-monotonic artefact.

    Both are evaluated with the *same* random StratifiedKFold used in scheme
    (a) so that any inflation is directly comparable.
    """
    idle_recs = [r for r in recs if r.label == "idle"]
    if len(idle_recs) < 4:
        return None, None, None, None

    X_idle, _ = prepare_matrix(idle_recs)
    n = len(idle_recs)

    y_half = np.array(["A" if j < n // 2 else "B" for j in range(n)])
    y_alt  = np.array(["A" if j % 2 == 0 else "B" for j in range(n)])

    folds = min(RANDOM_CV_SPLITS, n // 2)
    if folds < 2:
        return None, None, None, None

    clf = RandomForestClassifier(**RF_PARAMS)
    rcv = RepeatedStratifiedKFold(
        n_splits=folds, n_repeats=n_repeats, random_state=0
    )
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        sc_half = cross_val_score(clf, X_idle, y_half, cv=rcv)
        sc_alt  = cross_val_score(clf, X_idle, y_alt,  cv=rcv)

    return sc_half.mean(), sc_half.std(), sc_alt.mean(), sc_alt.std()


# ── Verdict logic ─────────────────────────────────────────────────────────────

def compute_verdict(
    random_mean,
    grouped_strict_mean,
    null_half_mean,
):
    """Return (is_leakage_suspected, reason_lines)."""
    reasons = []
    leakage = False

    if null_half_mean is not None and null_half_mean > NULL_HALF_LEAKAGE_THRESHOLD:
        leakage = True
        reasons.append(
            f"null-half accuracy {null_half_mean*100:.1f}% > {NULL_HALF_LEAKAGE_THRESHOLD*100:.0f}% "
            f"-- random KFold is learning a session-time gradient, NOT the fingerprint signal."
        )

    if (
        random_mean is not None
        and grouped_strict_mean is not None
        and (random_mean - grouped_strict_mean) > GAP_LEAKAGE_THRESHOLD
    ):
        leakage = True
        reasons.append(
            f"random CV ({random_mean*100:.1f}%) exceeds strict grouped CV "
            f"({grouped_strict_mean*100:.1f}%) by "
            f"{(random_mean - grouped_strict_mean)*100:.1f}pp > "
            f"{GAP_LEAKAGE_THRESHOLD*100:.0f}pp -- temporal proximity between "
            f"train and test windows is inflating the random-CV estimate."
        )

    if not reasons:
        reasons.append(
            "null-half is near chance and random/grouped gap is small -- "
            "no strong evidence of temporal leakage."
        )

    return leakage, reasons


# ── Printing helpers ──────────────────────────────────────────────────────────

_RED    = "\033[31m"
_GREEN  = "\033[32m"
_YELLOW = "\033[33m"
_BOLD   = "\033[1m"
_RESET  = "\033[0m"


def _fmt(mean, std, label=""):
    if mean is None:
        return f"{label}: N/A (insufficient data)"
    return f"{label}: {mean*100:.1f}% +- {std*100:.1f}%"


def _section(title):
    print(f"\n{_BOLD}{'─'*60}{_RESET}")
    print(f"{_BOLD}{title}{_RESET}")
    print(f"{_BOLD}{'─'*60}{_RESET}")


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "channel",
        nargs="?",
        default="read",
        help="Channel name (default: read).  Loads /tmp/frost-<channel>.csv",
    )
    args = parser.parse_args()

    channel = args.channel

    # ── Load data ────────────────────────────────────────────────────────────
    print(f"\nLoading channel: {_BOLD}{channel}{_RESET}")
    try:
        recs, groups = load_channel(channel)
    except FileNotFoundError as exc:
        print(f"ERROR: {exc}")
        sys.exit(1)
    X, y = prepare_matrix(recs)

    counts = {c: int((y == c).sum()) for c in np.unique(y)}
    print(f"Windows loaded: {len(recs)}  ({counts})")
    print(f"Feature matrix: {X.shape[0]} samples x {X.shape[1]} features")

    if len(recs) < 4:
        print("ERROR: not enough windows to evaluate (need >= 4).")
        sys.exit(1)

    # ── Scheme (a) ───────────────────────────────────────────────────────────
    _section("(a) Random RepeatedStratifiedKFold  [OPTIMISTIC BASELINE]")
    print(
        "  Windows are randomly assigned to folds.  Train and test may include\n"
        "  windows from the same time epoch, leaking temporal feature drift."
    )
    m_rand, s_rand, folds_a = eval_random_cv(X, y)
    print(f"  {_fmt(m_rand, s_rand, f'{folds_a}-fold x{RANDOM_CV_REPEATS} reps')}")

    # ── Scheme (b) ───────────────────────────────────────────────────────────
    _section(
        "(b) Leakage-robust GroupKFold  [CONTIGUOUS TIME BLOCKS]"
    )
    print(
        "  Windows sorted by collection time are assigned to contiguous blocks.\n"
        "  GroupKFold guarantees no time-block appears in both train and test.\n"
        "  The 2-block (half-split) is the strictest: maximum temporal gap."
    )

    m_g4, s_g4, _ = eval_grouped_cv(X, y, n_blocks=N_BLOCKS_COARSE)
    m_g2, s_g2, _ = eval_grouped_cv(X, y, n_blocks=N_BLOCKS_STRICT)
    print(f"  {_fmt(m_g4, s_g4, f'{N_BLOCKS_COARSE}-block GroupKFold')}")
    print(f"  {_fmt(m_g2, s_g2, f'{N_BLOCKS_STRICT}-block GroupKFold (half-split, most conservative)')}")

    # ── Scheme (c) ───────────────────────────────────────────────────────────
    _section("(c) Idle-only NULL control  [MUST BE ~CHANCE]")
    print(
        "  Only idle windows are used; labels are fabricated (no real class signal).\n"
        "\n"
        "  null_half: label = 'A' if window is in first half of session, else 'B'.\n"
        "    Evaluated with the SAME random KFold as scheme (a).\n"
        "    If accuracy >> 50%: session-level feature drift is leaking into\n"
        "    the random KFold -- the model is learning 'time', not 'class'.\n"
        "\n"
        "  null_alt:  label = 'A' for even-indexed windows, 'B' for odd-indexed.\n"
        "    Adjacent idle windows share a label; a model that merely exploits\n"
        "    local autocorrelation will score BELOW chance (they look alike).\n"
        "    A value well above 50% here is unusual and warrants investigation."
    )
    m_nh, s_nh, m_na, s_na = eval_null_control(recs)
    print(f"  {_fmt(m_nh, s_nh, 'null_half (random KFold)')}  [target: ~50%]")
    print(f"  {_fmt(m_na, s_na, 'null_alt  (random KFold)')}  [target: ~50%]")

    # ── Verdict ──────────────────────────────────────────────────────────────
    _section("VERDICT")
    leakage, reasons = compute_verdict(m_rand, m_g2, m_nh)

    if leakage:
        tag = f"{_RED}{_BOLD}LEAKAGE SUSPECTED{_RESET}"
    else:
        tag = f"{_GREEN}{_BOLD}NO LEAKAGE DETECTED{_RESET}"

    print(f"  {tag}")
    for reason in reasons:
        print(f"  * {reason}")

    # Summary table
    print(f"\n  {'Scheme':<45} {'Accuracy':>10}")
    print(f"  {'─'*57}")
    print(f"  {'(a) random StratKFold (optimistic)':<45} {_fmt(m_rand, s_rand):>10}")
    print(f"  {'(b) grouped 4-block (coarse temporal gap)':<45} {_fmt(m_g4, s_g4):>10}")
    print(f"  {'(b) grouped 2-block / half-split (max gap)':<45} {_fmt(m_g2, s_g2):>10}")
    print(f"  {'(c) null_half (idle-only, random KFold)':<45} {_fmt(m_nh, s_nh):>10}")
    print(f"  {'(c) null_alt  (idle-only, random KFold)':<45} {_fmt(m_na, s_na):>10}")
    print()


if __name__ == "__main__":
    main()
