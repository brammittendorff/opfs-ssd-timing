"""Load FROST capture data.

Two sources:
  - JSONL datasets exported by the PoC's "Export dataset" button: one labeled
    recording per line, schema:
        {"label", "window_s", "read_size", "spike_filter",
         "t_ms":[...], "lat_us":[...], "batch_rate":[...]}
  - A single raw CSV exported by "Export CSV" (timestamp_ms,latency_us) - used by
    analyze_trace.py for exploratory analysis of one unlabeled trace.
"""

import json
import numpy as np


class Recording:
    """One labeled capture window."""

    def __init__(self, label, t_ms, lat_us, batch_rate=None, window_s=None,
                 read_size=None, spike_filter=None):
        self.label = label
        self.t_ms = np.asarray(t_ms, dtype=np.float64)
        self.lat_us = np.asarray(lat_us, dtype=np.float64)
        self.batch_rate = np.asarray(batch_rate or [], dtype=np.float64)
        # Window length: trust the field, else infer from the timestamps.
        if window_s:
            self.window_s = float(window_s)
        elif self.t_ms.size:
            self.window_s = (self.t_ms[-1] - self.t_ms[0]) / 1000.0
        else:
            self.window_s = 0.0
        self.read_size = read_size
        self.spike_filter = spike_filter

    def __repr__(self):
        return (f"Recording(label={self.label!r}, n={self.lat_us.size}, "
                f"window_s={self.window_s:.1f})")


def load_jsonl(path):
    """Load a JSONL dataset into a list of Recording."""
    recs = []
    with open(path) as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            d = json.loads(line)
            recs.append(Recording(
                label=d["label"], t_ms=d["t_ms"], lat_us=d["lat_us"],
                batch_rate=d.get("batch_rate"), window_s=d.get("window_s"),
                read_size=d.get("read_size"), spike_filter=d.get("spike_filter"),
            ))
    return recs


def load_csv(path, label="trace"):
    """Load a single raw CSV trace (timestamp_ms,latency_us) as one Recording."""
    t, l = [], []
    with open(path) as fh:
        header = fh.readline()  # skip 'timestamp_ms,latency_us'
        for line in fh:
            parts = line.split(",")
            if len(parts) < 2:
                continue
            try:
                t.append(float(parts[0])); l.append(float(parts[1]))
            except ValueError:
                continue
    return Recording(label=label, t_ms=t, lat_us=l)


def labels(recs):
    """Sorted unique label list."""
    return sorted({r.label for r in recs})
