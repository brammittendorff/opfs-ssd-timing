"""Classical baseline: RandomForest on summary features.

Fast separability proof - answers "are these activities distinguishable over the
channel at all?" before investing in data collection or the CNN. Prints
cross-validated accuracy, a confusion matrix, and the top feature importances
(expected to be dominated by throughput / p95 / zero_frac).

Usage:
    python3 train_baseline.py dataset.jsonl [--bin-ms 100]
"""

import argparse
import numpy as np

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

from dataset import load_jsonl, labels
from features import build_matrix


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("dataset")
    ap.add_argument("--bin-ms", type=float, default=100.0)
    ap.add_argument("--folds", type=int, default=5)
    args = ap.parse_args()

    recs = load_jsonl(args.dataset)
    classes = labels(recs)
    print(f"{len(recs)} recordings, {len(classes)} classes: {classes}")
    counts = {c: sum(r.label == c for r in recs) for c in classes}
    print("per-class counts:", counts)

    X, y, names = build_matrix(recs, bin_ms=args.bin_ms)
    X = np.nan_to_num(X)

    min_count = min(counts.values())
    folds = max(2, min(args.folds, min_count))
    if min_count < 2:
        print("\nNeed >=2 recordings per class for cross-validation. Capture more.")
        return

    clf = RandomForestClassifier(n_estimators=400, random_state=0, n_jobs=-1)
    cv = StratifiedKFold(n_splits=folds, shuffle=True, random_state=0)
    y_pred = cross_val_predict(clf, X, y, cv=cv)

    acc = accuracy_score(y, y_pred)
    chance = max(counts.values()) / len(recs)
    print(f"\n{folds}-fold CV accuracy: {acc:.3f}  (chance ~ {chance:.3f})")
    print("\nconfusion matrix (rows=true, cols=pred):")
    cm = confusion_matrix(y, y_pred, labels=classes)
    print("        " + "  ".join(f"{c[:7]:>7}" for c in classes))
    for c, row in zip(classes, cm):
        print(f"{c[:7]:>7} " + "  ".join(f"{v:7d}" for v in row))
    print("\n" + classification_report(y, y_pred, labels=classes, zero_division=0))

    # Feature importances from a fit on all data.
    clf.fit(X, y)
    imp = clf.feature_importances_
    order = np.argsort(imp)[::-1][:12]
    print("top features:")
    for i in order:
        print(f"  {names[i]:>18}  {imp[i]:.3f}")


if __name__ == "__main__":
    main()
