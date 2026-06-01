# FROST PoC - OPFS-based SSD timing side channel

A local, self-contained proof-of-concept of the core mechanism from
**FROST: Fingerprinting Remotely using OPFS-based SSD Timing**
(Weissteiner, Weiser, Czerny, Neela, Rauscher, Juffinger, Gruss - TU Graz).

[FROST](https://hannesweissteiner.com/pdfs/frost.pdf)

Pure JavaScript in a browser tab measures **SSD contention** by timing random reads on
a large [OPFS](https://developer.mozilla.org/en-US/docs/Web/API/File_System_API/Origin_private_file_system)
file. No native code, no exploit, no permission prompt. When anything else on the
machine touches the same SSD (you open a site, launch an app, copy a file), read latency
spikes - and you watch it live.

This PoC implements the **live contention monitor**, plus a **labeled-capture harness**
and an **offline fingerprinting pipeline** (see [Fingerprinting](#fingerprinting-capture--offline-training))
that takes a first step toward the paper's CNN classification stage. It does **not**
include the covert channel.

> For testing on your own hardware, for research/education. Don't deploy against others.

## Demo

Live SSD-contention monitor reacting to disk activity:

https://github.com/brammittendorff/opfs-ssd-timing/raw/main/videos/opfs-ssd-timing.mp4

## How does this work? (plain English)

Your computer has one main storage drive (an SSD). Everything that reads or writes
files shares it, like one road that every car in town has to drive on. When the road is
busy, everyone slows down a little.

A normal website is not supposed to be able to see anything happening outside its own
tab. But a browser feature called OPFS lets a website store a big file on your real disk.
This PoC makes such a file, then keeps reading small pieces of it over and over and times
how long each read takes - down to millionths of a second.

Here is the trick: while nothing else is using the disk, those reads are fast and steady.
The moment something else touches the disk - you open another website, launch an app, your
computer saves a file in the background - the shared drive gets busy, and the website's own
reads briefly slow down. By watching for those slowdowns, a plain web page can tell *when*
your computer is doing disk activity, and even start to guess *what* (opening a browser tab
looks different from sitting idle). It never sees your files - only the timing of the
shared road.

That is the whole point of this demo: to show that this leak is real, measure how strong
it is on your machine, and (further down) try to fingerprint activities from the timing
pattern. The page below shows the slowdowns live as a chart so you can see it happen.

One catch: the trick only works if the website's file is bigger than your computer's free
memory. Otherwise the computer keeps the whole file in fast memory and never has to touch
the slow disk, so there is nothing to time. That is why the demo writes a very large file,
and why it does not work on machines with a lot of spare memory unless the file can grow
big enough.

## Run

```sh
cd poc-opfs
python3 serve.py            # serves http://localhost:8000 with COOP/COEP headers
```

Open **http://localhost:8000** in Google Chrome (recommended) or Firefox.

- **Chrome** gives the most headroom: OPFS can grow tens of GB, so the read-latency knee
  is easy to reach.
- **Firefox** works too, but caps OPFS at ~10 GB/origin. If that is smaller than your free
  RAM the read-latency knee is unreachable - use the **write-flush** channel instead (see
  [Channels](#channels)), which needs no large file. The Max-size field is auto-clamped to
  the detected quota, and a mid-fill quota error is handled gracefully.
- The COOP/COEP headers (set by `serve.py`) make the page *cross-origin isolated*, which
  unlocks the high-resolution `performance.now()` timer the attack needs. Opening the
  HTML file directly (`file://`) will **not** work - the timer stays coarsened.

## Use

1. The banner should read `crossOriginIsolated: true`, show a sub-us timer resolution, and
   (once probed) the OPFS quota. If it's red, you opened it wrong (not via `serve.py`).
2. Pick a **Channel** (leave it on `read` for Chrome). Click **Build & calibrate**. In
   `read` mode the worker grows an OPFS file 2 GB at a time and, after
   each step, probes read latency:
   - While the file fits in the OS page cache, the median stays at DRAM speed (a few us).
   - Once the file exceeds available cache, the median **jumps** to SSD speed (tens-
     hundreds of us). That knee is detected and **filling stops automatically** - you
     don't pay for a fixed 40 GB.
3. Monitoring starts. With the chart flat-ish, go to another window and **cause disk
   activity**: launch a heavy app, open a fresh browser profile, or `cp` a big file.
   A cluster of spikes appears in the chart. Idle -> it settles. That contrast is the leak.
4. **Export CSV** dumps the raw trace (`timestamp_ms,latency_us`) for offline analysis.
5. **Reset** deletes the OPFS file to reclaim disk.

### Tuning (UI fields)

| Field | Meaning |
|-------|---------|
| Grow step (GB) | random data written per calibration step |
| Knee threshold (us) | median latency that counts as "now hitting the SSD" |
| Max size (GB) | safety cap; stops growing here even without a clear knee (auto-clamped to the OPFS quota) |
| Read size (kB) | per-read size (4 kB matches the paper) |
| Channel | `read` (large-file knee) or `write-flush` (RAM-independent); see below |

**Negative control:** set Max size below your free RAM (e.g. 8 GB on a 30 GB machine).
Calibration won't find a knee, reads stay cache-served, and disk activity barely shows -
confirming the large-file requirement is what makes the read-latency attack work.

### Channels

| Channel | Probe | Needs file > RAM? | Use when |
|---------|-------|-------------------|----------|
| **read** (default) | time random 4 kB reads | **yes** | Chrome, or any box where the file can exceed free RAM. The paper's method. |
| **write-flush** | write 4 kB, time `handle.flush()` (forced SSD writeback) | **no** | Firefox's 10 GB cap, or high-RAM machines where the file stays cached. Costs extra SSD writes. |

The read channel's signal is read latency (and its throughput / spike-rate / tail). The
write-flush channel measures writeback latency instead, so contention shows even on a
small, fully-cached file - at the cost of continuous small writes (more SSD wear). Note
that the read channel's *throughput* still dips under another process's writes even when
the file is cache-served, so on Firefox a write-heavy target may show up on either channel.
Captured recordings record which `channel` produced them.

## Fingerprinting (capture + offline training)

The live monitor shows *that* the SSD is contended; fingerprinting asks *what caused it*.
The **Capture** panel records fixed-length, labeled slices of the contention signal, and
the `analysis/` Python pipeline trains a classifier on them.

Trace analysis (`analysis/analyze_trace.py` on an exported CSV) shows the channel is
strongest in **throughput (reads/sec)** - it collapses ~5x under sustained contention -
and, for short bursty activities, in the **tail**: the **spike rate** (reads > 1 ms),
**p99.9** and **max** latency. The **median is nearly flat**, so the feature extraction
leans on throughput / spike_rate / tail percentiles, not the median. Capture defaults to
**raw** mode (spike filter off) so the offline pipeline sees the full signal.

**Capture:** after monitoring starts, type an **Activity label**, set a **Window (s)**,
and click **Record**. A 3-2-1 pre-roll lets you trigger the activity (open a site, launch
an app, copy a file, or sit idle) in sync with the window. Repeat across activities, then
**Export dataset (JSONL)**.

**Train (offline):**

```sh
cd analysis
pip install -r requirements.txt          # + `pip install torch` for the CNN
python3 analyze_trace.py "~/Downloads/frost-trace (1).csv"   # EDA -> trace-analysis.png
python3 train_baseline.py frost-dataset-40.jsonl             # RandomForest + importances
python3 train_cnn.py frost-dataset-40.jsonl                  # 1D-CNN (PyTorch)
```

Start with a two-class **idle vs active** set to validate the loop, then expand to
websites / apps / file operations. See `analysis/README.md` for details.

### Worked example: idle vs opening a browser tab

A real capture on a 30 GB / NVMe laptop (file calibrated to the ~42 GB knee), two 8 s
windows per class - `idle` (do nothing) and `open tab` (open a fresh tab during the
window):

| feature | idle (x2) | open tab (x2) |
|---|---|---|
| median / p95 / throughput | 65 us / 125 us / 14k/s | 70 us / 125 us / 14k/s |
| **spike rate** (reads > 1 ms) | **0.6, 0.9 /s** | **4.9, 4.4 /s** |
| p99.9 latency | ~228 us | ~338 us |
| max latency | 1.5-3.8 ms | 9.9-14.4 ms |

The headline stats (median, p95, throughput) are **identical** - opening a tab is too
brief to move them on a fast SSD. The signal is entirely in the **tail**: a ~6x higher
**spike rate** as the page's resources are read/written, plus higher p99.9 and max. This
is why `features.py` exposes a per-bin `spike_rate` channel and the CNN uses
`throughput, spike_rate, p99, p95` (median dropped).

A single-feature leave-one-out check on `spike_rate` separates the two classes perfectly:

```
$ python3 - <<'PY'
import numpy as np; from dataset import load_jsonl
recs = load_jsonl("~/Downloads/frost-dataset-4.jsonl")
feat = np.array([np.sum(r.lat_us>1000)/r.window_s for r in recs])   # spikes/sec
lab  = np.array([r.label for r in recs])
ok = 0
for i in range(len(recs)):                                           # leave-one-out, nearest-centroid
    tr=[j for j in range(len(recs)) if j!=i]
    cents={c: feat[[j for j in tr if lab[j]==c]].mean() for c in set(lab[tr])}
    ok += min(cents, key=lambda c: abs(feat[i]-cents[c]))==lab[i]
print(f"leave-one-out on spike_rate: {ok}/{len(recs)}")
PY
leave-one-out on spike_rate: 4/4
```

> **Caveat on `train_baseline.py` / `train_cnn.py` with tiny datasets.** With only 2
> recordings per class the RandomForest reports ~chance accuracy - not because the classes
> don't separate (they clearly do above) but because 4 samples can't constrain 81 features.
> The classifiers only become trustworthy at ~**15-20 windows per class**; collect that
> many before reading their accuracy. The per-feature check above is the right tool while
> the dataset is small.

## How it works

- `serve.py` - stdlib HTTP server adding `Cross-Origin-Opener-Policy: same-origin` and
  `Cross-Origin-Embedder-Policy: require-corp` (+ no-cache).
- `worker.js` - runs in a Web Worker (required for the synchronous
  `FileSystemSyncAccessHandle`). In `read` mode it grows the OPFS file with real random
  bytes (zero/sparse pages would be cache-served and deduped, flushing each chunk to bound
  dirty memory), detects the page-cache knee, then loops timing random 4 kB reads. In
  `write-flush` mode it skips the big file and instead times `handle.flush()` after a small
  write. Applies the paper's spike filter (samples > 1 ms replaced by the local mean) -
  **toggleable** via `setFilter` so capture can record the raw signal. Re-opening an
  already-calibrated file skips the fill; quota errors are caught.
- `index.html` + `main.js` - environment check, controls, a dependency-free `<canvas>`
  rolling latency chart, live stats, CSV export, and the labeled-capture harness (records
  windows into an in-memory dataset, exports JSONL).
- `analysis/` - offline Python pipeline (feature extraction, RandomForest baseline,
  1D-CNN) that fingerprints activities from captured datasets.

## Caveats / notes

- **Single SSD assumed.** If the browser's OPFS storage and the activity you're testing
  live on *different* physical disks, you won't see cross-disk contention (paper's stated
  limitation). Most laptops/consumer machines have one internal SSD.
- **Real disk I/O.** Calibration writes tens of GB of random data on first build - minutes
  of writes and some SSD wear. Adaptive growth keeps this to the minimum that works.
  The file lives in Chrome's per-origin OPFS storage; **Reset** removes it (or clear site
  data for `localhost:8000` in Chrome).
- `profile-sync-daemon` / tmpfs-backed browser profiles would defeat the zero-interaction
  variant because OPFS would live in RAM, not on the SSD. Unlikely on a default Debian box.

## Mitigation (per the paper)

Cap OPFS to ~1 GB without explicit persistent-storage permission: the file then fits in
RAM, reads are served from the page cache, and the SSD-contention channel disappears.
You can demonstrate this here via the negative control above.
