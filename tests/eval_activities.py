"""Multi-class, leakage-robust evaluation of the real-activity capture (activities.js).

Classifies idle / web / cp / compute from one channel's trace and reports:
  - random vs GROUPED CV accuracy (grouped = train early windows, test late -> honest),
  - the grouped confusion matrix + macro-F1,
  - per-class one-vs-rest grouped accuracy (which activities the channel can actually pick out),
  - a shuffled-label null control (must be ~chance) + permutation p-value.

Usage:  python3 tests/eval_activities.py <channel>     # cache (default) | read | flush
"""
import os, sys, json
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "analysis")); sys.path.insert(0, HERE)
from dataset import Recording
from features import build_matrix
from eval_cv import assign_contiguous_blocks
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import (RepeatedStratifiedKFold, GroupKFold,
                                      cross_val_score, cross_val_predict)
from sklearn.metrics import confusion_matrix, f1_score, accuracy_score

RF = dict(n_estimators=300, random_state=0, n_jobs=-1)

def load(ch):
    t, l = [], []
    for line in open(f"/tmp/frost-act-{ch}.csv"):
        p = line.split(",")
        try: t.append(float(p[0])); l.append(float(p[1]))
        except ValueError: pass
    t = np.array(t); l = np.array(l)
    M = {m["name"]: m["t_ms"] for m in json.load(open(f"/tmp/frost-act-{ch}-marks.json"))}
    recs = []
    # marks are <cls>_<i>_s/_e, emitted in collection order
    keys = [k for k in M if k.endswith("_s")]
    keys.sort(key=lambda k: M[k])
    for ks in keys:
        cls, i, _ = ks.rsplit("_", 2)
        ke = f"{cls}_{i}_e"
        if ke not in M: continue
        a, b = M[ks], M[ke]; mask = (t >= a) & (t < b)
        wt, wl = t[mask], l[mask]
        if wl.size > 20:
            recs.append(Recording(label=cls, t_ms=wt, lat_us=wl, window_s=(b - a) / 1000.0))
    return recs

def main():
    ch = sys.argv[1] if len(sys.argv) > 1 else "cache"
    recs = load(ch)
    classes = sorted(set(r.label for r in recs))
    counts = {c: sum(r.label == c for r in recs) for c in classes}
    print(f"\n===== channel: {ch}  |  {len(recs)} windows  {counts} =====")
    X, y, _ = build_matrix(recs); X = np.nan_to_num(X)
    nmin = min(counts.values())
    if nmin < 3:
        print("need >=3 windows/class"); return

    folds = min(5, nmin)
    clf = lambda: RandomForestClassifier(**RF)
    chance = max(counts.values()) / len(recs)

    rand = cross_val_score(clf(), X, y, cv=RepeatedStratifiedKFold(n_splits=folds, n_repeats=20, random_state=0)).mean()

    nblk = min(folds, nmin)
    blocks = assign_contiguous_blocks(len(y), nblk)
    gkf = GroupKFold(n_splits=nblk)
    grp = cross_val_score(clf(), X, y, cv=gkf, groups=blocks).mean()
    pred = cross_val_predict(clf(), X, y, cv=gkf, groups=blocks)
    f1 = f1_score(y, pred, average="macro")

    print(f"random k-fold (optimistic, leaks): {rand*100:.0f}%")
    print(f"GROUPED CV (honest, early->late):  {grp*100:.0f}%   (chance {chance*100:.0f}%, {len(classes)} classes)")
    print(f"macro-F1 (grouped):                {f1*100:.0f}%")

    cm = confusion_matrix(y, pred, labels=classes)
    print("\nconfusion (rows=true, cols=pred):")
    print("        " + "".join(f"{c[:7]:>8}" for c in classes))
    for c, row in zip(classes, cm):
        print(f"{c[:7]:>7} " + "".join(f"{v:>8}" for v in row))

    print("\nper-activity detectability (one-vs-rest, grouped CV):")
    for c in classes:
        yb = np.array([1 if v == c else 0 for v in y])
        acc = cross_val_score(clf(), X, yb, cv=GroupKFold(n_splits=nblk), groups=blocks).mean()
        base = max(yb.mean(), 1 - yb.mean())
        print(f"  {c:<8} {acc*100:>3.0f}%  (vs {base*100:.0f}% always-majority)  {'<- detectable' if acc > base + 0.08 else '~ not separable'}")

    # null control + permutation p-value (grouped)
    rng = np.random.default_rng(0)
    null = np.array([cross_val_score(clf(), X, rng.permutation(y), cv=gkf, groups=blocks).mean() for _ in range(30)])
    p = (np.sum(null >= grp) + 1) / (len(null) + 1)
    print(f"\nshuffled-label null (grouped): {null.mean()*100:.0f}% +-{null.std()*100:.0f}%   permutation p={p:.3f}  "
          f"{'(real signal)' if p < 0.05 else '(NOT significant)'}")

if __name__ == "__main__":
    main()
