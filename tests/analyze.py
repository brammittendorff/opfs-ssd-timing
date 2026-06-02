"""Analyze a channel_probe run: contention-detection verdict + website classifier.

Usage:  python3 analyze.py <channel>      # reads /tmp/frost-<channel>.csv + -marks.json

Prints, per channel:
  - CONTENTION: idle vs load (dd/burn) latency+throughput, PASS if clearly separated.
  - FINGERPRINT: website classifier accuracy reported THREE ways so the number is honest:
      * random KFold      - optimistic; LEAKS session/time drift (train & test windows can
                            be temporally adjacent), so it over-states accuracy.
      * grouped CV        - HONEST headline: train on early windows, test on late ones, so
                            the model must generalise across the session, not memorise drift.
      * idle-vs-idle null - control that MUST sit at ~chance; if it doesn't, the pipeline is
                            classifying time, not the website. (See tests/eval_cv.py.)
    PASS requires the honest grouped accuracy >= 70% AND the null near chance.
"""
import os, sys, json
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "analysis"))
sys.path.insert(0, HERE)                           # for eval_cv (same dir)
from dataset import Recording                      # noqa: E402
from features import build_matrix                  # noqa: E402
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold, cross_val_predict, RepeatedStratifiedKFold, cross_val_score
from sklearn.metrics import accuracy_score, confusion_matrix
# leakage-robust evaluators (grouped CV + idle-vs-idle null) live in eval_cv.py
from eval_cv import eval_random_cv, eval_grouped_cv, eval_null_control, compute_verdict

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

    # ---- FINGERPRINT (leakage-robust) ----
    recs = []
    i = 1
    while f"idle_{i}_s" in M:
        for lab in ("idle", "nunl"):
            a, b = M[f"{lab}_{i}_s"], M[f"{lab}_{i}_e"]
            wt, wl = win(t, l, a, b)
            if wl.size > 5:
                recs.append(Recording(label=lab, t_ms=wt, lat_us=wl, window_s=(b-a)/1000))
        i += 1
    counts = {c: sum(r.label == c for r in recs) for c in {r.label for r in recs}}
    if not recs or min(counts.values()) < 2:
        print("FINGERPRINT: not enough windows (need >=2 idle and >=2 site)")
        return detected, False, 0.0

    X, y, _ = build_matrix(recs); X = np.nan_to_num(X)
    m_rand, s_rand, _ = eval_random_cv(X, y)             # optimistic, leaks drift
    m_g2,  s_g2,  _ = eval_grouped_cv(X, y, 2)           # honest: early->late split
    m_nh,  s_nh, m_na, s_na = eval_null_control(recs)    # idle-vs-idle null control
    leak, _ = compute_verdict(m_rand, m_g2, m_nh)
    headline = m_g2 if m_g2 is not None else m_rand      # the honest, drift-robust number
    ok = headline is not None and headline >= 0.70       # PASS on the grouped number

    def pct(m, s):
        return "N/A" if m is None else f"{m*100:.0f}% +-{(s or 0)*100:.0f}%"
    print(f"FINGERPRINT idle vs site: {counts}")
    print(f"            random KFold  (optimistic, leaks drift): {pct(m_rand, s_rand)}")
    print(f"            grouped CV    (HONEST, early->late split): {pct(m_g2, s_g2)}  <- headline")
    nh = "N/A" if m_nh is None else f"{m_nh*100:.0f}%"
    print(f"            idle-vs-idle null (must be ~50%): {nh}  [{'LEAK' if (m_nh or 0) > 0.6 else 'ok'}]")
    if leak:
        print(f"            WARNING: random CV is inflated by session/time drift - trust the grouped number, not the random one (see eval_cv.py)")
    print(f"            verdict: {PASS if ok else FAIL}  (honest grouped accuracy >= 70%)")
    return detected, ok, headline

if __name__ == "__main__":
    main()
