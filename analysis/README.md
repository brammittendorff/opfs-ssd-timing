# FROST fingerprinting - offline pipeline

Takes labeled contention recordings captured by the PoC and trains a classifier to
tell activities apart over the OPFS SSD-timing channel. This is the
classification stage the base PoC deliberately omits.

## Key signal (from trace analysis)

The contention channel is strongest in **throughput (reads/sec)** - it collapses
~5x when another process hits the SSD - and in the **cache-hit fraction** (0 us
reads) and **p95/p99 latency** (~2x swing). The **median latency is nearly flat**.

Short labeled captures (e.g. opening a tab) showed the *aggregate* stats barely
move, but the **tail and its timing** do: the **spike rate** (reads > 1 ms) and
*where* spikes fall in the window carry the activity signature. So the CNN channels
are `throughput, spike_rate, p99, p95` (`features.CNN_CHANNELS`) - median omitted -
and the per-bin `spike_rate` / `p999` channels are first-class in `features.py`.

## Install

```sh
pip install -r requirements.txt
pip install torch        # only for train_cnn.py
```

## Capture data (in the PoC)

1. `python3 ../serve.py`, open in Chrome, **Build & calibrate**, wait for monitoring.
2. Type an **Activity label**, set **Window (s)**, keep **Raw** checked.
3. Click **Record** - during the 3-2-1 pre-roll, trigger the activity in another
   window (open a site, launch an app, `cp` a big file, or do nothing for `idle`).
4. Repeat across activities; aim for >= ~20 trials per class. **Export dataset
   (JSONL)** when done.

Start with a two-class **idle vs active** set to validate the pipeline end-to-end,
then expand to websites / apps / file-ops.

## Analyze / train

```sh
# EDA on any single trace (raw CSV or one JSONL recording) -> trace-analysis.png
python3 analyze_trace.py "~/Downloads/frost-trace (1).csv"

# Fast separability baseline (RandomForest + feature importances)
python3 train_baseline.py frost-dataset-40.jsonl

# Paper-style 1D-CNN on the binned multi-channel trace
python3 train_cnn.py frost-dataset-40.jsonl --epochs 80
```

## Files

| File | Role |
|------|------|
| `dataset.py` | load JSONL datasets / single CSV traces into `Recording` objects |
| `features.py` | bin into time windows; per-bin channels; summary vector + CNN tensor |
| `analyze_trace.py` | EDA: throughput/latency/cache-hit over time, period via autocorrelation |
| `train_baseline.py` | RandomForest CV accuracy, confusion matrix, feature importances |
| `train_cnn.py` | 1D-CNN classifier (PyTorch) |
