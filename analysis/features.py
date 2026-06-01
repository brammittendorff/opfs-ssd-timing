"""Feature extraction for FROST recordings.

Encodes the key empirical finding from trace analysis: the contention signal is
strongest in *throughput* (reads/sec) and the *cache-hit fraction* (zero-latency
reads), with *p95/p99* latency second; the median is nearly flat. So every
representation here exposes those channels rather than leaning on the median.

Two representations per recording:
  - timeseries [C, T]: per-bin channels over time, for the 1D-CNN.
  - summary vector: distribution stats of each channel, for the sklearn baseline.
"""

import numpy as np

# Per-bin channels (order matters: it defines the CNN input channels and the
# summary-feature layout).
CHANNELS = ["throughput", "median", "p95", "p99", "p999",
            "mean", "std", "zero_frac", "spike_rate"]
# Channels fed to the CNN - the ones real captures showed carry the signal.
# spike_rate (reads >1ms) + the tail percentiles hold the activity signature
# (e.g. a tab-open burst); the median is nearly flat and intentionally omitted.
CNN_CHANNELS = ["throughput", "spike_rate", "p99", "p95"]

SPIKE_US = 1000.0  # a read >1ms is a contention "spike" (paper's threshold)


def bin_recording(rec, bin_ms=100.0):
    """Bin one Recording into fixed-width time bins.

    Returns dict channel -> np.array of length T, where
    T = round(window_s * 1000 / bin_ms). Empty bins get 0 throughput / NaN-free
    latency stats carried from a neutral fill (0), which the CNN normalizer and
    the summary stats both tolerate.
    """
    T = max(1, int(round(rec.window_s * 1000.0 / bin_ms)))
    out = {c: np.zeros(T, dtype=np.float64) for c in CHANNELS}
    if rec.t_ms.size == 0:
        return out, T

    t0 = rec.t_ms[0]
    idx = np.floor((rec.t_ms - t0) / bin_ms).astype(int)
    idx = np.clip(idx, 0, T - 1)
    bin_secs = bin_ms / 1000.0

    for b in range(T):
        sel = rec.lat_us[idx == b]
        if sel.size == 0:
            continue
        out["throughput"][b] = sel.size / bin_secs
        out["median"][b] = np.median(sel)
        out["p95"][b] = np.percentile(sel, 95)
        out["p99"][b] = np.percentile(sel, 99)
        out["p999"][b] = np.percentile(sel, 99.9)
        out["mean"][b] = sel.mean()
        out["std"][b] = sel.std()
        out["zero_frac"][b] = np.mean(sel == 0.0)
        out["spike_rate"][b] = np.mean(sel > SPIKE_US)  # fraction of reads >1ms
    return out, T


def recording_to_timeseries(rec, bin_ms=100.0, T_fixed=None, channels=CNN_CHANNELS):
    """[C, T] float array for the CNN. Pads/truncates to T_fixed if given."""
    binned, T = bin_recording(rec, bin_ms)
    mat = np.stack([binned[c] for c in channels], axis=0)  # [C, T]
    if T_fixed is not None and T != T_fixed:
        if T < T_fixed:
            mat = np.pad(mat, ((0, 0), (0, T_fixed - T)))
        else:
            mat = mat[:, :T_fixed]
    return mat


def recording_to_summary(rec, bin_ms=100.0):
    """1D summary feature vector + parallel feature-name list."""
    binned, _ = bin_recording(rec, bin_ms)
    feats, names = [], []
    pcts = [10, 25, 50, 75, 90]
    for c in CHANNELS:
        v = binned[c]
        # nonzero view for throughput-aware stats (empty bins shouldn't drag stats)
        feats += [v.mean(), v.std(), v.min(), v.max()]
        names += [f"{c}_mean", f"{c}_std", f"{c}_min", f"{c}_max"]
        feats += [np.percentile(v, p) for p in pcts]
        names += [f"{c}_p{p}" for p in pcts]
    return np.asarray(feats, dtype=np.float64), names


def build_matrix(recs, bin_ms=100.0):
    """Summary-feature matrix X [N, F], label vector y, feature names."""
    rows, names = [], None
    for r in recs:
        f, names = recording_to_summary(r, bin_ms)
        rows.append(f)
    X = np.vstack(rows) if rows else np.empty((0, 0))
    y = np.array([r.label for r in recs])
    return X, y, names


def build_tensor(recs, bin_ms=100.0, channels=CNN_CHANNELS):
    """CNN tensor X [N, C, T] (T = max over recordings), label vector y."""
    T_fixed = max((int(round(r.window_s * 1000.0 / bin_ms)) for r in recs), default=1)
    T_fixed = max(1, T_fixed)
    X = np.stack([recording_to_timeseries(r, bin_ms, T_fixed, channels) for r in recs])
    y = np.array([r.label for r in recs])
    return X, y, T_fixed
