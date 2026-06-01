"""Multi-class FROST fingerprinting evaluator.

Loads windows produced by collect_sites.js, builds feature vectors, then:

  CLOSED-WORLD: RandomForest with stratified K-fold CV.
    Reports TOP-1 accuracy, macro-F1, and the confusion matrix.

  OPEN-WORLD: Treat the first ceil(K/2) sites as 'monitored', the rest as
    'unmonitored' (collapsed into one class).  Reports precision, recall,
    and F1 for the monitored class.

  CONTROL: Re-runs closed-world CV with shuffled labels to calibrate chance.

Usage:
    python3 tests/eval_multiclass.py [--channel cache] [--folds 5]
                                     [--monitored 3] [--bin-ms 100]

Reads:  /tmp/frost-sites-<channel>.csv
        /tmp/frost-sites-<channel>-marks.json
"""

import argparse
import json
import math
import os
import sys

import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_recall_fscore_support,
)
from sklearn.model_selection import StratifiedKFold, cross_val_predict

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "analysis"))

from dataset import Recording  # noqa: E402
from features import build_matrix  # noqa: E402

PASS_COLOR = "\033[32mPASS\033[0m"
FAIL_COLOR = "\033[31mFAIL\033[0m"


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_trace(channel: str):
    """Return (t_ms, lat_us) numpy arrays from the raw CSV."""
    csv_path = f"/tmp/frost-sites-{channel}.csv"
    t, lat = [], []
    with open(csv_path) as fh:
        fh.readline()  # skip header
        for line in fh:
            parts = line.split(",")
            if len(parts) < 2:
                continue
            try:
                t.append(float(parts[0]))
                lat.append(float(parts[1]))
            except ValueError:
                continue
    return np.asarray(t, dtype=np.float64), np.asarray(lat, dtype=np.float64)


def load_marks(channel: str):
    """Return (events list, labels dict {cls_idx_str -> url})."""
    marks_path = f"/tmp/frost-sites-{channel}-marks.json"
    raw = json.load(open(marks_path))
    labels_meta = {}
    events = []
    for entry in raw:
        if entry["name"] == "_labels":
            labels_meta = entry.get("labels", {})
        else:
            events.append(entry)
    mark_map = {e["name"]: e["t_ms"] for e in events}
    return mark_map, labels_meta


def slice_window(t, lat, t_start, t_end):
    """Return (t_slice, lat_slice) for samples within [t_start, t_end)."""
    mask = (t >= t_start) & (t < t_end)
    return t[mask], lat[mask]


def build_recordings(channel: str):
    """Parse marks + CSV into a list of Recording objects."""
    t, lat = load_trace(channel)
    mark_map, labels_meta = load_marks(channel)

    # Discover how many classes and windows were collected.
    # Marks are named cls<k>_<i>_s and cls<k>_<i>_e.
    recs = []
    class_indices = sorted(
        {int(k) for k in labels_meta.keys()},
        key=lambda x: x,
    )
    for cls_idx in class_indices:
        label_name = labels_meta[str(cls_idx)]
        # Count windows: find all cls<k>_<i>_s marks.
        i = 1
        while True:
            start_key = f"cls{cls_idx}_{i}_s"
            end_key   = f"cls{cls_idx}_{i}_e"
            if start_key not in mark_map or end_key not in mark_map:
                break
            t_s = mark_map[start_key]
            t_e = mark_map[end_key]
            wt, wl = slice_window(t, lat, t_s, t_e)
            if wl.size > 0:
                recs.append(Recording(
                    label=label_name,
                    t_ms=wt,
                    lat_us=wl,
                    window_s=(t_e - t_s) / 1000.0,
                ))
            i += 1

    return recs, labels_meta


# ---------------------------------------------------------------------------
# Evaluation helpers
# ---------------------------------------------------------------------------

def make_clf():
    return RandomForestClassifier(n_estimators=400, random_state=0, n_jobs=-1)


def closed_world_eval(X, y, class_names, folds: int):
    """Stratified CV -> accuracy, macro-F1, confusion matrix."""
    min_count = min((y == c).sum() for c in class_names)
    actual_folds = max(2, min(folds, min_count))

    cv = StratifiedKFold(n_splits=actual_folds, shuffle=True, random_state=0)
    y_pred = cross_val_predict(make_clf(), X, y, cv=cv)

    acc     = accuracy_score(y, y_pred)
    macro_f1 = f1_score(y, y_pred, labels=class_names, average="macro", zero_division=0)
    cm      = confusion_matrix(y, y_pred, labels=class_names)
    report  = classification_report(y, y_pred, labels=class_names, zero_division=0)

    return {
        "folds":     actual_folds,
        "accuracy":  acc,
        "macro_f1":  macro_f1,
        "cm":        cm,
        "report":    report,
        "y_pred":    y_pred,
    }


def open_world_eval(X, y, class_names, n_monitored: int, folds: int):
    """Collapse unmonitored classes into one, report monitored detection."""
    monitored   = class_names[:n_monitored]
    unmonitored = class_names[n_monitored:]

    def collapse(label):
        return label if label in monitored else "_unmonitored_"

    y_ow = np.array([collapse(lbl) for lbl in y])
    ow_classes = list(monitored) + (["_unmonitored_"] if unmonitored else [])

    min_count = min((y_ow == c).sum() for c in ow_classes)
    actual_folds = max(2, min(folds, min_count))
    if actual_folds < 2:
        return None

    cv = StratifiedKFold(n_splits=actual_folds, shuffle=True, random_state=0)
    y_pred = cross_val_predict(make_clf(), X, y_ow, cv=cv)

    # Treat each monitored site as "positive", unmonitored as "negative".
    # Macro-average precision/recall over monitored labels.
    prec, rec, f1, _ = precision_recall_fscore_support(
        y_ow, y_pred, labels=list(monitored), average="macro", zero_division=0
    )
    return {
        "folds":       actual_folds,
        "monitored":   list(monitored),
        "unmonitored": list(unmonitored),
        "precision":   prec,
        "recall":      rec,
        "f1":          f1,
        "y_ow":        y_ow,
        "y_pred":      y_pred,
        "ow_classes":  ow_classes,
    }


def shuffled_control(X, y, class_names, folds: int, n_repeats: int = 20):
    """Return mean accuracy under permuted labels (calibrates chance level)."""
    min_count = min((y == c).sum() for c in class_names)
    actual_folds = max(2, min(folds, min_count))
    rng = np.random.default_rng(42)
    scores = []
    for seed in range(n_repeats):
        y_perm = rng.permutation(y)
        cv = StratifiedKFold(n_splits=actual_folds, shuffle=True, random_state=seed)
        y_pred = cross_val_predict(make_clf(), X, y_perm, cv=cv)
        scores.append(accuracy_score(y_perm, y_pred))
    return float(np.mean(scores)), float(np.std(scores))


# ---------------------------------------------------------------------------
# Pretty-printing
# ---------------------------------------------------------------------------

def print_confusion_matrix(cm, class_names):
    col_w = max(8, max(len(c) for c in class_names) + 2)
    header = " " * (col_w + 2) + "  ".join(f"{c[:col_w]:>{col_w}}" for c in class_names)
    print(header)
    print(" " * (col_w + 2) + "-" * (len(header) - col_w - 2))
    for name, row in zip(class_names, cm):
        print(f"{name[:col_w]:>{col_w}} |" + "  ".join(f"{v:{col_w}d}" for v in row))


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser(
        description="Multi-class FROST website fingerprinting evaluation"
    )
    ap.add_argument("--channel",   default="cache",
                    help="Channel name (matches /tmp/frost-sites-<channel>.csv)")
    ap.add_argument("--folds",     type=int, default=5,
                    help="Number of CV folds (capped by min class count)")
    ap.add_argument("--monitored", type=int, default=0,
                    help="Number of 'monitored' sites for open-world test "
                         "(0 = ceil(K/2) where K = number of site classes)")
    ap.add_argument("--bin-ms",    type=float, default=100.0,
                    help="Bin width in milliseconds for feature extraction")
    args = ap.parse_args()

    # ---- load data ----
    print(f"\nLoading channel={args.channel} ...")
    recs, labels_meta = build_recordings(args.channel)
    if not recs:
        print("ERROR: no recordings found. Check that collect_sites.js ran successfully.")
        sys.exit(1)

    class_names = []   # preserve discovery order (idle first, then sites in order)
    seen = set()
    for r in recs:
        if r.label not in seen:
            class_names.append(r.label)
            seen.add(r.label)
    counts = {c: sum(r.label == c for r in recs) for c in class_names}

    print(f"{len(recs)} recordings, {len(class_names)} classes")
    for c in class_names:
        print(f"  {counts[c]:3d}  {c}")

    X, y, feat_names = build_matrix(recs, bin_ms=args.bin_ms)
    X = np.nan_to_num(X)
    print(f"Feature matrix: {X.shape[0]} x {X.shape[1]}")

    # ---- closed-world ----
    print("\n" + "=" * 60)
    print("CLOSED-WORLD EVALUATION")
    print("=" * 60)

    min_count = min(counts.values())
    if min_count < 2:
        print("Need >= 2 recordings per class for CV. Collect more windows (increase N_PER).")
        sys.exit(1)

    cw = closed_world_eval(X, y, class_names, folds=args.folds)
    chance = 1.0 / len(class_names)

    print(f"\n{cw['folds']}-fold stratified CV")
    print(f"  TOP-1 accuracy : {cw['accuracy']*100:.1f}%  (chance = {chance*100:.1f}%)")
    print(f"  Macro-F1       : {cw['macro_f1']:.3f}")
    verdict_cw = cw['accuracy'] >= 0.70 or cw['accuracy'] > chance * 1.5
    print(f"  Verdict        : {PASS_COLOR if verdict_cw else FAIL_COLOR}  "
          f"({'clearly above chance' if verdict_cw else 'near chance — collect more windows or use a stronger channel'})")

    print("\nConfusion matrix (rows=true, cols=pred):")
    print_confusion_matrix(cw["cm"], class_names)

    print("\nPer-class report:")
    print(cw["report"])

    # ---- open-world ----
    site_classes = [c for c in class_names if c != "idle"]
    n_monitored = args.monitored if args.monitored > 0 else math.ceil(len(site_classes) / 2)
    n_monitored = min(n_monitored, len(site_classes))

    print("=" * 60)
    print(f"OPEN-WORLD EVALUATION  (monitored={n_monitored}, unmonitored={len(site_classes)-n_monitored})")
    print("=" * 60)

    if len(site_classes) < 2:
        print("Need >= 2 site classes for open-world split. Add more SITES.")
    else:
        # Use site_classes only (drop idle from open-world, as in FROST paper).
        site_recs = [r for r in recs if r.label != "idle"]
        X_ow, y_ow_raw, _ = build_matrix(site_recs, bin_ms=args.bin_ms)
        X_ow = np.nan_to_num(X_ow)

        ow = open_world_eval(X_ow, y_ow_raw, site_classes, n_monitored, folds=args.folds)
        if ow is None:
            print("Not enough samples for open-world CV. Collect more windows.")
        else:
            print(f"\nMonitored sites  : {ow['monitored']}")
            print(f"Unmonitored sites: {ow['unmonitored']}")
            print(f"\n{ow['folds']}-fold stratified CV (unmonitored collapsed to one class)")
            print(f"  Monitored precision (macro): {ow['precision']*100:.1f}%")
            print(f"  Monitored recall    (macro): {ow['recall']*100:.1f}%")
            print(f"  Monitored F1        (macro): {ow['f1']:.3f}")
            verdict_ow = ow["precision"] >= 0.60 and ow["recall"] >= 0.50
            print(f"  Verdict: {PASS_COLOR if verdict_ow else FAIL_COLOR}")

    # ---- shuffled-label control ----
    print("\n" + "=" * 60)
    print("SHUFFLED-LABEL CONTROL (closed-world)")
    print("=" * 60)

    ctrl_mean, ctrl_std = shuffled_control(X, y, class_names, folds=args.folds)
    print(f"\n  Shuffled accuracy: {ctrl_mean*100:.1f}% +/- {ctrl_std*100:.1f}%")
    print(f"  Real accuracy    : {cw['accuracy']*100:.1f}%")
    gap = cw['accuracy'] - ctrl_mean
    sig = gap > 2 * ctrl_std
    print(f"  Gap              : {gap*100:+.1f}pp  {'(significant, > 2 sigma)' if sig else '(not clearly significant)'}")
    print(f"  Verdict          : {PASS_COLOR if sig else FAIL_COLOR}")

    print()


if __name__ == "__main__":
    main()
