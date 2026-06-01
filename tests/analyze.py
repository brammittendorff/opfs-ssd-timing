"""Analyze a channel_probe run: contention-detection verdict + nu.nl classifier.

Usage:  python3 analyze.py <channel>      # reads /tmp/frost-<channel>.csv + -marks.json

Prints, per channel:
  - CONTENTION: idle vs load (dd/burn) latency+throughput, PASS if clearly separated.
  - FINGERPRINT: RandomForest CV accuracy on idle-vs-nu.nl windows, PASS if >= 70%.
"""
import os, sys, json
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "analysis"))
from dataset import Recording                      # noqa: E402
from features import build_matrix                  # noqa: E402
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold, cross_val_predict, RepeatedStratifiedKFold, cross_val_score
from sklearn.metrics import accuracy_score, confusion_matrix

PASS = "\033[32mPASS\033[0m"; FAIL = "\033[31mFAIL\033[0m"

def load_run(channel):
    t, l = [], []
    with open(f"/tmp/frost-{channel}.csv") as fh:
        fh.readline()
        for line in fh:
            p = line.split(",")
            if len(p) < 2: continue
            try: t.append(float(p[0])); l.append(float(p[1]))
            except ValueError: pass
    marks = {m["name"]: m["t_ms"] for m in json.load(open(f"/tmp/frost-{channel}-marks.json"))}
    return np.array(t), np.array(l), marks

def win(t, l, a, b):
    m = (t >= a) & (t < b); return t[m], l[m]

def stats(lat, dur_s):
    if not lat.size: return dict(n=0, med=0, p95=0, thr=0)
    return dict(n=lat.size, med=np.median(lat), p95=np.percentile(lat, 95), thr=lat.size/dur_s)

def main():
    ch = sys.argv[1] if len(sys.argv) > 1 else "cache"
    t, l, M = load_run(ch)
    print(f"\n===== channel: {ch}  ({len(t):,} samples, {(t[-1]-t[0])/1000:.0f}s) =====")

    # ---- CONTENTION ----
    it, il = np.concatenate([win(t,l,M['c_idle1'],M['c_load'])[0], win(t,l,M['c_idle2'],M['c_end'])[0]]), \
             np.concatenate([win(t,l,M['c_idle1'],M['c_load'])[1], win(t,l,M['c_idle2'],M['c_end'])[1]])
    idle = stats(il, (M['c_load']-M['c_idle1'] + M['c_end']-M['c_idle2'])/1000)
    lt, ll = win(t, l, M['c_load'], M['c_load_end'])
    load = stats(ll, (M['c_load_end']-M['c_load'])/1000)
    thr_drop = 1 - load['thr']/max(idle['thr'],1e-9)
    p95_rise = load['p95']/max(idle['p95'],1e-9)
    med_rise = load['med']/max(idle['med'],1e-9)
    detected = (thr_drop > 0.20) or (p95_rise > 1.5) or (med_rise > 1.3)
    print(f"CONTENTION  idle: med {idle['med']:.0f}us p95 {idle['p95']:.0f}us thr {idle['thr']:.0f}/s")
    print(f"            load: med {load['med']:.0f}us p95 {load['p95']:.0f}us thr {load['thr']:.0f}/s")
    print(f"            -> throughput -{thr_drop*100:.0f}%, p95 x{p95_rise:.2f}, median x{med_rise:.2f}  [{PASS if detected else FAIL}]")

    # ---- FINGERPRINT ----
    recs = []
    i = 1
    while f"idle_{i}_s" in M:
        for lab in ("idle", "nunl"):
            a, b = M[f"{lab}_{i}_s"], M[f"{lab}_{i}_e"]
            wt, wl = win(t, l, a, b)
            if wl.size > 5:
                recs.append(Recording(label=lab, t_ms=wt, lat_us=wl, window_s=(b-a)/1000))
        i += 1
    counts = {c: sum(r.label==c for r in recs) for c in {r.label for r in recs}}
    X, y, _ = build_matrix(recs); X = np.nan_to_num(X)
    folds = min(5, min(counts.values()))
    if folds >= 2:
        clf = RandomForestClassifier(n_estimators=300, random_state=0, n_jobs=-1)
        # Repeated stratified CV (many seeds) so the accuracy is a distribution, not a
        # single high-variance number - small datasets make one split meaningless.
        rcv = RepeatedStratifiedKFold(n_splits=folds, n_repeats=40, random_state=0)
        scores = cross_val_score(clf, X, y, cv=rcv, n_jobs=-1)
        mean, std = scores.mean(), scores.std()
        lo, hi = np.percentile(scores, 5), np.percentile(scores, 95)
        # Permutation control: shuffle labels -> should sit at chance, calibrates "real".
        rng = np.random.default_rng(0)
        perm = np.array([cross_val_score(clf, X, rng.permutation(y), cv=StratifiedKFold(folds, shuffle=True, random_state=s)).mean()
                         for s in range(20)])
        ok = mean >= 0.70 and (mean - 2*std) > perm.mean()
        print(f"FINGERPRINT idle vs nu.nl: {counts}")
        print(f"            {folds}-fold CV x40 reps: accuracy = {mean*100:.0f}% +-{std*100:.0f}%  (5-95%: {lo*100:.0f}-{hi*100:.0f}%)")
        print(f"            shuffled-label control: {perm.mean()*100:.0f}% +-{perm.std()*100:.0f}%  (this is 'chance' here)")
        print(f"            verdict: {PASS if ok else FAIL}  (need >=70% and clearly above the shuffled control)")
        return detected, ok, mean
    else:
        print("FINGERPRINT: not enough windows"); return detected, False, 0.0

if __name__ == "__main__":
    main()
