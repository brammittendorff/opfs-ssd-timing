"""Exploratory analysis of a single FROST trace (CSV or one JSONL recording).

Reproduces the findings that drive the feature design:
  - throughput (reads/sec) collapses under contention (the dominant channel),
  - p95 latency rises ~2x while the median barely moves,
  - periodic background activity shows up as an autocorrelation peak.

Usage:
    python3 analyze_trace.py "~/Downloads/frost-trace (1).csv"
    python3 analyze_trace.py dataset.jsonl          # analyzes the first recording

Writes trace-analysis.png next to a printed summary.
"""

import os
import sys
import numpy as np

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from dataset import load_csv, load_jsonl


def per_bin(rec, bin_ms):
    t0 = rec.t_ms[0]
    T = max(1, int(np.ceil((rec.t_ms[-1] - t0) / bin_ms)) + 1)
    idx = np.clip(np.floor((rec.t_ms - t0) / bin_ms).astype(int), 0, T - 1)
    bin_secs = bin_ms / 1000.0
    thr = np.zeros(T); med = np.zeros(T); p95 = np.zeros(T); zf = np.zeros(T)
    for b in range(T):
        sel = rec.lat_us[idx == b]
        if sel.size == 0:
            continue
        thr[b] = sel.size / bin_secs
        med[b] = np.median(sel)
        p95[b] = np.percentile(sel, 95)
        zf[b] = np.mean(sel == 0.0)
    centers = (np.arange(T) + 0.5) * bin_secs
    return centers, thr, med, p95, zf


def autocorr_period(x, bin_secs, min_lag_s=1.0, max_lag_s=30.0):
    """Dominant period of a 1D series via its autocorrelation.

    The autocorrelation decays monotonically away from lag 0, so a plain argmax
    just returns the search-window boundary. A genuine period shows up as a
    *local* maximum - a lag where the AC rises again - so we pick the strongest
    local peak in the window instead.
    """
    x = x - x.mean()
    if not np.any(x):
        return None, None
    ac = np.correlate(x, x, mode="full")[len(x) - 1:]
    ac /= ac[0]
    lo = max(1, int(min_lag_s / bin_secs))
    hi = min(len(ac) - 2, int(max_lag_s / bin_secs))
    if hi <= lo:
        return None, ac
    # local maxima only (rises then falls)
    peaks = [i for i in range(lo, hi) if ac[i] > ac[i - 1] and ac[i] >= ac[i + 1]]
    if not peaks:
        return None, ac
    peak = max(peaks, key=lambda i: ac[i])
    return peak * bin_secs, ac


def main():
    if len(sys.argv) < 2:
        print(__doc__)
        sys.exit(1)
    path = os.path.expanduser(sys.argv[1])
    if path.endswith(".jsonl"):
        rec = load_jsonl(path)[0]
    else:
        rec = load_csv(path)

    n = rec.lat_us.size
    span = (rec.t_ms[-1] - rec.t_ms[0]) / 1000.0
    sl = np.sort(rec.lat_us)
    def pc(p):
        return sl[min(n - 1, int(p * n))]
    print(f"file: {path}")
    print(f"samples={n:,}  span={span:.1f}s  avg_rate={n/span:,.0f}/s")
    print(f"latency us: min={sl[0]:.0f} med={pc(.5):.0f} p90={pc(.9):.0f} "
          f"p95={pc(.95):.0f} p99={pc(.99):.0f} max={sl[-1]:.0f}")
    print(f"cache-hit (0us) fraction: {np.mean(rec.lat_us == 0.0)*100:.1f}%")

    # 1s bins for the summary print, 250ms bins for plots/autocorr.
    c1, thr1, med1, p951, _ = per_bin(rec, 1000.0)
    idle_thr = np.percentile(thr1, 75)   # idle ~ high-throughput bins
    busy_thr = np.percentile(thr1, 10)   # contention ~ low-throughput bins
    print(f"throughput reads/s: idle(p75)={idle_thr:,.0f}  "
          f"busy(p10)={busy_thr:,.0f}  collapse={idle_thr/max(busy_thr,1):.1f}x")
    print(f"p95 latency us: idle(p25 of bins)={np.percentile(p951,25):.0f}  "
          f"busy(p90 of bins)={np.percentile(p951,90):.0f}")

    cb, thr, med, p95, zf = per_bin(rec, 250.0)
    period, _ = autocorr_period(thr, 0.25)
    if period:
        print(f"dominant activity period (throughput autocorr): ~{period:.1f}s")

    fig, ax = plt.subplots(3, 1, figsize=(11, 8), sharex=True)
    ax[0].plot(cb, thr, lw=0.8, color="#58a6ff"); ax[0].set_ylabel("reads/s")
    ax[0].set_title("throughput (primary contention channel)")
    ax[1].plot(cb, p95, lw=0.8, color="#f0883e", label="p95")
    ax[1].plot(cb, med, lw=0.8, color="#3fb950", label="median")
    ax[1].set_ylabel("latency us"); ax[1].legend(loc="upper right")
    ax[1].set_title("latency: p95 swings, median flat")
    ax[2].plot(cb, zf * 100, lw=0.8, color="#d2a8ff")
    ax[2].set_ylabel("cache-hit %"); ax[2].set_xlabel("time (s)")
    ax[2].set_title("cache-hit fraction (0us reads)")
    for a in ax:
        a.grid(alpha=0.2)
    fig.tight_layout()
    out = os.path.join(os.path.dirname(os.path.abspath(__file__)), "trace-analysis.png")
    fig.savefig(out, dpi=110)
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
