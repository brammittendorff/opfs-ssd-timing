"""Statistical rigor layer for FROST channel fingerprint evaluation.

WHY EACH METRIC MATTERS
========================

Single-run CV accuracy (what analyze.py reports)
  Guards against: nothing — it is one point estimate on one dataset with high
  variance when N is small (here: 16 windows/class). Two back-to-back runs on
  the same machine can differ by ±10–15 pp purely from split randomness.

(a) Bootstrap confidence interval on CV accuracy
  Guards against: over-interpreting a lucky split. By resampling windows with
  replacement thousands of times and recomputing CV accuracy each time, we get
  the *sampling distribution* of the estimator. The 95% CI tells you the range
  of accuracies you'd plausibly observe across repeated data collection sessions
  from the same distribution. A wide CI (e.g. [70%, 100%]) means the 93% point
  estimate is not reliable; a narrow CI means the data density is sufficient.

(b) Permutation p-value
  Guards against: spurious accuracy from overfitting, dataset artifacts, or
  structural bias (e.g. time ordering leaking into splits). Shuffling labels
  destroys the class-signal while preserving all feature correlations and
  CV-split structure. If the real accuracy is not significantly above the null
  distribution of permuted accuracies, the "fingerprint" is a mirage. Standard
  threshold: p < 0.05 (i.e., fewer than 5% of permutations match or beat the
  real score).

(c) Between-run CI (multi-run aggregator)
  Guards against: session-specific confounds — a particular OS state, thermal
  state, or background process profile that happens to separate idle from nu.nl
  in one session but not in others. Within-dataset CV reuses the same OS state
  for every window, so its variance estimate is too optimistic. A true between-
  run CI requires collecting the CSV K independent times (different OS/browser
  states), computing accuracy on each, and taking the CI of those K numbers.
  When only one run is available, bootstrap pseudo-runs bound the within-session
  variance but CANNOT bound the between-session variance — they will be too
  narrow. This function documents that limitation explicitly.

(d) Sample-size / power note
  Guards against: planning future experiments with insufficient data. Given the
  observed accuracy and the bootstrap spread, estimates how many windows/class
  are needed to shrink the 95% CI half-width below a target (default ±5 pp).

Usage
-----
  cd tests/
  python3 stats.py              # reads channel: read (default)
  python3 stats.py cache        # reads channel: cache
  python3 stats.py flush        # reads channel: flush

  # For a real between-run CI, re-run the probe K times:
  #   node tests/channel_probe.js read /tmp/frost-read-run1.csv
  #   node tests/channel_probe.js read /tmp/frost-read-run2.csv
  #   ...
  # then call aggregate_runs([run1_paths, run2_paths, ...])
"""

import json
import os
import sys
import time
from collections import Counter

import numpy as np
from joblib import Parallel, delayed
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import (
    StratifiedKFold,
    cross_val_score,
    permutation_test_score,
)

# Make analysis/ importable from tests/
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "analysis"))
from dataset import Recording   # noqa: E402
from features import build_matrix  # noqa: E402

# ---------------------------------------------------------------------------
# Classifier configuration
# ---------------------------------------------------------------------------
# Use 300 trees for the single real-accuracy estimate (matches analyze.py),
# but drop to 50 for bootstrap/permutation iterations: accuracy is nearly
# identical on this dataset while being ~6x faster per call.
N_TREES_MAIN = 300
N_TREES_FAST = 50
RANDOM_STATE = 0


# ---------------------------------------------------------------------------
# Data loading helpers
# ---------------------------------------------------------------------------

def _load_csv(path):
    """Load timestamp_ms,latency_us CSV into parallel numpy arrays."""
    ts, lats = [], []
    with open(path) as fh:
        fh.readline()  # skip header
        for line in fh:
            parts = line.split(",")
            if len(parts) < 2:
                continue
            try:
                ts.append(float(parts[0]))
                lats.append(float(parts[1]))
            except ValueError:
                continue
    return np.asarray(ts, dtype=np.float64), np.asarray(lats, dtype=np.float64)


def _load_marks(path):
    """Load marks JSON into {name: t_ms} dict."""
    return {m["name"]: m["t_ms"] for m in json.load(open(path))}


def load_windows(csv_path, marks_path):
    """Slice idle_i and nunl_i windows from a raw CSV using the marks file.

    Returns a list of Recording objects and a Counter of windows per class.
    Skips windows with <= 5 samples (too sparse for meaningful features).
    """
    t, l = _load_csv(csv_path)
    marks = _load_marks(marks_path)

    recs = []
    i = 1
    while f"idle_{i}_s" in marks:
        for label in ("idle", "nunl"):
            t_start = marks[f"{label}_{i}_s"]
            t_end = marks[f"{label}_{i}_e"]
            mask = (t >= t_start) & (t < t_end)
            wt, wl = t[mask], l[mask]
            if wl.size > 5:
                recs.append(Recording(
                    label=label,
                    t_ms=wt,
                    lat_us=wl,
                    window_s=(t_end - t_start) / 1000.0,
                ))
        i += 1

    counts = Counter(r.label for r in recs)
    return recs, counts


def build_Xy(recs):
    """Build feature matrix X and label vector y from a list of Recordings."""
    X, y, _ = build_matrix(recs)
    X = np.nan_to_num(X)
    return X, y


# ---------------------------------------------------------------------------
# (a) Bootstrap confidence interval
# ---------------------------------------------------------------------------

def _bootstrap_iteration(seed, X, y, label_indices, n_folds):
    """Single stratified bootstrap resample + CV accuracy (runs in worker)."""
    rng = np.random.default_rng(seed)
    boot_idx = np.concatenate([
        rng.choice(idxs, size=len(idxs), replace=True)
        for idxs in label_indices.values()
    ])
    Xb, yb = X[boot_idx], y[boot_idx]
    clf = RandomForestClassifier(
        n_estimators=N_TREES_FAST, random_state=seed % (2**31), n_jobs=1
    )
    cv = StratifiedKFold(n_folds, shuffle=True, random_state=seed % (2**31))
    return cross_val_score(clf, Xb, yb, cv=cv).mean()


def bootstrap_ci(X, y, n_boot=500, ci=(2.5, 97.5)):
    """Compute bootstrap confidence interval on stratified CV accuracy.

    Parameters
    ----------
    X : ndarray [N, F]
    y : ndarray [N]
    n_boot : int
        Number of bootstrap resamples (>=500 recommended for stable 95% CI).
    ci : tuple
        Lower and upper percentiles for the confidence interval.

    Returns
    -------
    dict with keys: mean, ci_lo, ci_hi, std, scores
    """
    labels = sorted(set(y))
    label_indices = {lab: np.where(y == lab)[0] for lab in labels}
    n_folds = min(5, min(len(v) for v in label_indices.values()))

    scores = np.array(Parallel(n_jobs=-1)(
        delayed(_bootstrap_iteration)(s, X, y, label_indices, n_folds)
        for s in range(n_boot)
    ))

    lo, hi = np.percentile(scores, list(ci))
    return {
        "mean": float(scores.mean()),
        "ci_lo": float(lo),
        "ci_hi": float(hi),
        "std": float(scores.std()),
        "scores": scores,
    }


# ---------------------------------------------------------------------------
# (b) Permutation test
# ---------------------------------------------------------------------------

def permutation_test(X, y, n_permutations=500):
    """Permutation test against null distribution of shuffled-label accuracy.

    Uses sklearn's permutation_test_score with parallel execution.

    Returns
    -------
    dict with keys: real_acc, null_mean, null_std, null_scores, pvalue, significant
    """
    labels = sorted(set(y))
    label_indices = {lab: np.where(y == lab)[0] for lab in labels}
    n_folds = min(5, min(len(v) for v in label_indices.values()))

    # Main classifier (300 trees) for the real score to match analyze.py.
    clf_main = RandomForestClassifier(
        n_estimators=N_TREES_MAIN, random_state=RANDOM_STATE, n_jobs=-1
    )
    # Fast classifier for null distribution (results are statistically identical
    # on this dataset; 50 trees are sufficient to rank permuted accuracy).
    clf_fast = RandomForestClassifier(
        n_estimators=N_TREES_FAST, random_state=RANDOM_STATE, n_jobs=1
    )
    cv = StratifiedKFold(n_folds, shuffle=True, random_state=RANDOM_STATE)

    real_acc = cross_val_score(clf_main, X, y, cv=cv).mean()

    _, null_scores, pvalue = permutation_test_score(
        clf_fast, X, y,
        cv=StratifiedKFold(n_folds, shuffle=True, random_state=RANDOM_STATE),
        n_permutations=n_permutations,
        n_jobs=-1,
        random_state=RANDOM_STATE,
    )

    return {
        "real_acc": float(real_acc),
        "null_mean": float(null_scores.mean()),
        "null_std": float(null_scores.std()),
        "null_scores": null_scores,
        "pvalue": float(pvalue),
        "significant": bool(pvalue < 0.05),
    }


# ---------------------------------------------------------------------------
# (c) Multi-run aggregator
# ---------------------------------------------------------------------------

def aggregate_runs(run_path_pairs):
    """Aggregate fingerprint accuracy across multiple independent collection runs.

    DOCSTRING — VALIDITY NOTE
    -------------------------
    A *real* between-run CI requires running channel_probe.js K independent
    times (different browser sessions, OS states, thermal states) and saving
    distinct CSVs.  Each run yields one accuracy estimate; the CI is then taken
    over those K estimates and captures between-session variance that within-
    dataset CV cannot see.

    Command hint to generate K real runs:
        node tests/channel_probe.js read /tmp/frost-read-run1.csv
        node tests/channel_probe.js read /tmp/frost-read-run2.csv
        ... (repeat K >= 8 times for a meaningful CI)
    Then call:
        aggregate_runs([
            ("/tmp/frost-read-run1.csv", "/tmp/frost-read-run1-marks.json"),
            ("/tmp/frost-read-run2.csv", "/tmp/frost-read-run2-marks.json"),
            ...
        ])

    When only one run exists, this function falls back to bootstrap pseudo-runs
    (stratified resamples) and clearly marks the output as a PSEUDO-RUN CI.
    The pseudo-run CI is a lower bound: it captures within-session window
    variance but will be systematically narrower than a true between-run CI
    because all pseudo-runs share the same OS/browser state.

    Parameters
    ----------
    run_path_pairs : list of (csv_path, marks_path) tuples
        Each tuple is one independent collection run.

    Returns
    -------
    dict with keys: mode, run_accuracies, mean, ci_lo, ci_hi, std, n_runs
    """
    if len(run_path_pairs) < 2:
        # Single run: fall back to bootstrap pseudo-runs
        csv_path, marks_path = run_path_pairs[0]
        recs, counts = load_windows(csv_path, marks_path)
        X, y = build_Xy(recs)
        boot = bootstrap_ci(X, y, n_boot=500)
        return {
            "mode": "PSEUDO-RUN (bootstrap resamples, NOT independent sessions)",
            "run_accuracies": boot["scores"].tolist(),
            "mean": boot["mean"],
            "ci_lo": boot["ci_lo"],
            "ci_hi": boot["ci_hi"],
            "std": boot["std"],
            "n_runs": 500,
        }

    # True multi-run mode: one accuracy per CSV.
    run_accuracies = []
    for csv_path, marks_path in run_path_pairs:
        recs, _ = load_windows(csv_path, marks_path)
        X, y = build_Xy(recs)
        labels_u = sorted(set(y))
        label_indices = {lab: np.where(y == lab)[0] for lab in labels_u}
        n_folds = min(5, min(len(v) for v in label_indices.values()))
        clf = RandomForestClassifier(
            n_estimators=N_TREES_MAIN, random_state=RANDOM_STATE, n_jobs=-1
        )
        cv = StratifiedKFold(n_folds, shuffle=True, random_state=RANDOM_STATE)
        acc = cross_val_score(clf, X, y, cv=cv).mean()
        run_accuracies.append(float(acc))

    arr = np.asarray(run_accuracies)
    n = len(arr)
    # Wilson-style: use t-distribution for small K, normal for K>=30.
    from scipy import stats as sp_stats
    t_crit = sp_stats.t.ppf(0.975, df=n - 1)
    half_width = t_crit * arr.std(ddof=1) / np.sqrt(n)
    mean = float(arr.mean())

    return {
        "mode": "REAL MULTI-RUN",
        "run_accuracies": run_accuracies,
        "mean": mean,
        "ci_lo": float(mean - half_width),
        "ci_hi": float(mean + half_width),
        "std": float(arr.std(ddof=1)),
        "n_runs": n,
    }


# ---------------------------------------------------------------------------
# (d) Sample-size / power note
# ---------------------------------------------------------------------------

def sample_size_estimate(real_acc, boot_std, current_n_per_class,
                         target_half_width=0.05):
    """Estimate windows/class needed to achieve a target CI half-width.

    Two complementary methods:
    1. Bernoulli/binomial: treats each window prediction as a Bernoulli trial.
       n_binom = (1.96 / target_hw)^2 * p * (1-p)
       This is a lower bound — it assumes perfect cross-validation and ignores
       estimator variance from the finite forest.
    2. Bootstrap-scaled: uses the empirical bootstrap std to project how n must
       grow for the CI to narrow. Assumes std ~ 1/sqrt(n) scaling (CLT).
       n_scaled = current_n * (boot_std * 1.96 / target_hw)^2
       This is more realistic for this specific feature pipeline.

    The recommended sample size is the larger of the two (conservative).
    """
    n_binom = (1.96 / target_half_width) ** 2 * real_acc * (1.0 - real_acc)
    n_scaled = current_n_per_class * (boot_std * 1.96 / target_half_width) ** 2
    recommended = max(n_binom, n_scaled)
    return {
        "target_half_width": target_half_width,
        "current_n_per_class": current_n_per_class,
        "n_binom": float(n_binom),
        "n_scaled": float(n_scaled),
        "recommended": float(recommended),
        "already_sufficient": bool(current_n_per_class >= recommended),
    }


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _hr(char="─", width=64):
    print(char * width)


def run_full_report(channel="read"):
    """Load data for `channel` and print the full statistical report."""
    csv_path = f"/tmp/frost-{channel}.csv"
    marks_path = f"/tmp/frost-{channel}-marks.json"

    print()
    _hr("═")
    print(f"  FROST Statistical Report — channel: {channel}")
    _hr("═")

    # Load windows
    t0 = time.time()
    recs, counts = load_windows(csv_path, marks_path)
    print(f"\nData: {csv_path}")
    print(f"Windows: {dict(counts)}")
    if not recs or min(counts.values()) < 2:
        print("ERROR: not enough windows for CV. Aborting.")
        return

    X, y = build_Xy(recs)
    labels_u = sorted(set(y))
    label_indices = {lab: np.where(y == lab)[0] for lab in labels_u}
    n_per_class = min(len(v) for v in label_indices.values())
    n_folds = min(5, n_per_class)

    # ---- (a) Bootstrap CI ----
    _hr()
    print("(a) BOOTSTRAP 95% CI on CV accuracy")
    _hr()
    print("    Resampling windows with replacement (stratified), 500 resamples...")
    boot = bootstrap_ci(X, y, n_boot=500)
    print(f"    Bootstrap mean accuracy : {boot['mean']*100:.1f}%")
    print(f"    95% CI (2.5–97.5 pct)  : [{boot['ci_lo']*100:.1f}%, {boot['ci_hi']*100:.1f}%]")
    print(f"    Bootstrap std           : {boot['std']*100:.1f} pp")
    print()
    print("    Interpretation: you'd expect accuracy between "
          f"{boot['ci_lo']*100:.0f}% and {boot['ci_hi']*100:.0f}% "
          "across repeated collections")
    print("    from the same distribution. A CI touching 50% (chance) would")
    print("    indicate the signal is unreliable.")

    # ---- (b) Permutation test ----
    _hr()
    print("(b) PERMUTATION TEST (500 label-shuffles)")
    _hr()
    print("    Computing null distribution...")
    perm = permutation_test(X, y, n_permutations=500)
    sig_str = "YES (p<0.05)" if perm["significant"] else "NO (p>=0.05)"
    print(f"    Real accuracy           : {perm['real_acc']*100:.1f}%")
    print(f"    Null (shuffled) mean    : {perm['null_mean']*100:.1f}% "
          f"± {perm['null_std']*100:.1f}% std")
    print(f"    p-value                 : {perm['pvalue']:.4f}")
    print(f"    Significant (p<0.05)?   : {sig_str}")
    print()
    if perm["significant"]:
        print("    The classifier performs significantly above chance. The")
        print("    fingerprint signal is real, not an artifact of the CV procedure.")
    else:
        print("    WARNING: result is NOT significant. The fingerprint may be")
        print("    a CV artifact or dataset-specific fluke.")

    # ---- (c) Multi-run aggregator (pseudo-run mode) ----
    _hr()
    print("(c) BETWEEN-RUN CI (pseudo-run mode — one CSV available)")
    _hr()
    agg = aggregate_runs([(csv_path, marks_path)])
    print(f"    Mode: {agg['mode']}")
    print(f"    Bootstrap pseudo-runs   : {agg['n_runs']}")
    print(f"    Pseudo-run mean         : {agg['mean']*100:.1f}%")
    print(f"    Pseudo-run 95% CI       : [{agg['ci_lo']*100:.1f}%, {agg['ci_hi']*100:.1f}%]")
    print()
    print("    IMPORTANT: This CI reflects within-session window variance only.")
    print("    It is systematically NARROWER than a true between-run CI because")
    print("    all pseudo-runs share the same OS/browser session state.")
    print("    To get a real between-run CI, re-run the probe K>=8 times:")
    print("      node tests/channel_probe.js read /tmp/frost-read-run1.csv")
    print("      node tests/channel_probe.js read /tmp/frost-read-run2.csv")
    print("      ... (repeat for each run, saving distinct CSVs)")
    print("    Then call aggregate_runs([...pairs...]) with all run paths.")

    # ---- (d) Sample-size estimate ----
    _hr()
    print("(d) SAMPLE-SIZE / POWER NOTE")
    _hr()
    ss = sample_size_estimate(
        real_acc=perm["real_acc"],
        boot_std=boot["std"],
        current_n_per_class=n_per_class,
        target_half_width=0.05,
    )
    print(f"    Current windows/class   : {ss['current_n_per_class']}")
    print(f"    Target CI half-width    : ±{ss['target_half_width']*100:.0f} pp")
    print(f"    Binomial estimate       : {ss['n_binom']:.0f} windows/class")
    print(f"    Bootstrap-scaled est.   : {ss['n_scaled']:.0f} windows/class")
    print(f"    Recommended (max)       : {ss['recommended']:.0f} windows/class")
    if ss["already_sufficient"]:
        print(f"    Status: current n={ss['current_n_per_class']} is already sufficient "
              f"for ±{ss['target_half_width']*100:.0f}% CI.")
    else:
        deficit = int(np.ceil(ss["recommended"])) - ss["current_n_per_class"]
        print(f"    Status: need ~{deficit} more windows/class "
              f"to achieve ±{ss['target_half_width']*100:.0f}% CI.")

    # ---- Summary ----
    _hr("═")
    print("SUMMARY")
    _hr("═")
    elapsed = time.time() - t0
    print(f"  Channel           : {channel}")
    print(f"  Windows/class     : {n_per_class}")
    print(f"  Real CV accuracy  : {perm['real_acc']*100:.1f}%")
    print(f"  Bootstrap 95% CI  : [{boot['ci_lo']*100:.1f}%, {boot['ci_hi']*100:.1f}%]")
    print(f"  Permutation p     : {perm['pvalue']:.4f}  ({'significant' if perm['significant'] else 'NOT significant'})")
    print(f"  Windows needed    : ~{int(np.ceil(ss['recommended']))} / class for ±5% CI")
    print(f"  Analysis time     : {elapsed:.0f}s")
    _hr("═")
    print()

    return {
        "channel": channel,
        "real_acc": perm["real_acc"],
        "boot_ci_lo": boot["ci_lo"],
        "boot_ci_hi": boot["ci_hi"],
        "boot_mean": boot["mean"],
        "boot_std": boot["std"],
        "pvalue": perm["pvalue"],
        "significant": perm["significant"],
        "sample_size_recommended": ss["recommended"],
    }


if __name__ == "__main__":
    channel = sys.argv[1] if len(sys.argv) > 1 else "read"
    run_full_report(channel)
