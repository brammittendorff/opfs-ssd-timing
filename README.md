# FROST PoC - OPFS-based SSD timing side channel

A local, self-contained proof-of-concept of the core mechanism from
**FROST: Fingerprinting Remotely using OPFS-based SSD Timing**
(Weissteiner, Weiser, Czerny, Neela, Rauscher, Juffinger, Gruss - TU Graz).

[FROST](https://hannesweissteiner.com/pdfs/frost.pdf)

Pure JavaScript in a browser tab measures **hardware contention** by timing a shared
resource. No native code, no exploit, no permission prompt. When anything else on the
machine touches that resource (you open a site, launch an app, copy a file), the timing
spikes - and you watch it live.

It offers **three channels** - three different shared resources you can time. They sense
different things, and some are browser-dependent (see [Channels](#channels)):

1. **read** - senses **disk** activity. Time random 4 kB reads on a large
   [OPFS](https://developer.mozilla.org/en-US/docs/Web/API/File_System_API/Origin_private_file_system)
   file once it exceeds free RAM and reads hit the SSD. The paper's method; strongest
   channel. Works in **Chrome and Firefox** (needs OPFS quota > free RAM).
2. **cache-occupancy** - senses **CPU / memory / website** activity. No file at all: time
   a sweep of an LLC-sized buffer (the Shusterman et al. 2019 channel). Works in **Chrome
   and Firefox**, zero setup - the portable default.
3. **write-flush** - senses **disk** writeback, but only where `flush()` truly fsyncs:
   **works on Firefox**, is a **no-op on Chrome**. Needs no big file.

This PoC implements the **live contention monitor**, plus a **labeled-capture harness**
and an **offline fingerprinting pipeline** (see [Fingerprinting](#fingerprinting-capture--offline-training))
that takes a first step toward the paper's CNN classification stage, and a
[**test suite**](tests/) that drives each channel end-to-end and checks it detects
contention and fingerprints a website. It does **not** include the covert channel.

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
python3 serve.py            # from the repo root; serves http://localhost:8000 with COOP/COEP headers
```

Open **http://localhost:8000** in Google Chrome (recommended) or Firefox.

- **Chrome** gives the most headroom: OPFS can grow tens of GB, so the read-latency knee
  is easy to reach.
- **Firefox** works too, but caps OPFS at ~10 GB/origin. If that is smaller than your free
  RAM the read-latency knee is unreachable - use the **write-flush** channel instead (see
  [Channels](#channels)), which needs no large file. The Max-size field is **not** clamped to
  the (conservative) quota estimate - it grows toward your Max and a mid-fill quota error is
  handled gracefully, so you can use all the space the browser actually grants.
- The COOP/COEP headers (set by `serve.py`) make the page *cross-origin isolated*, which
  unlocks the high-resolution `performance.now()` timer the attack needs. Opening the
  HTML file directly (`file://`) will **not** work - the timer stays coarsened.

## Use

1. The banner should read `crossOriginIsolated: true`, show a sub-us timer resolution, and
   (once probed) the OPFS quota. If it's red, you opened it wrong (not via `serve.py`).
2. Pick a **Channel** (see [Channels](#channels) - `cache-occupancy` is the zero-setup
   option that works on any machine; `read` is the paper's method but needs OPFS quota >
   free RAM). Click **Build & calibrate**. In `read` mode the worker grows an OPFS file
   1 GB at a time (the **Grow step** field) and, after each step, probes **what fraction of random reads actually
   hit the SSD** - measured against this machine's own cached baseline, so it's robust to
   coarse/per-device timers (not a fixed-µs median rule):
   - While the file fits in free RAM, ~0% of reads hit the SSD (all cache-served) - keep
     growing. If it reaches the cap still at ~0%, your quota is smaller than RAM and the
     read channel can't work here (use `cache-occupancy`/`write-flush`).
   - Once enough reads miss cache and hit the SSD (the **knee**), filling **stops
     automatically** - and a later **Build** on the same file detects it's already past
     the knee and does **not** re-grow it. (If even a tiny working set is SSD-slow, OPFS
     isn't page-cached on your box and the channel works at any size.)
3. Monitoring starts. With the chart flat-ish, go to another window and **cause disk
   activity**: launch a heavy app, open a fresh browser profile, or `cp` a big file.
   A cluster of spikes appears in the chart. Idle -> it settles. That contrast is the leak.
4. **Export CSV** dumps the raw trace (`timestamp_ms,latency_us`) for offline analysis.
5. **Reset** deletes the OPFS file to reclaim disk.

### Tuning (UI fields)

| Field | Meaning |
|-------|---------|
| Grow step (GB) | random data written per calibration step |
| Knee threshold (us) | fallback only: a read slower than this also counts as an SSD hit. The primary detector is the **fraction of reads that hit the SSD** vs the cached baseline (device-independent), so this rarely matters |
| Max size (GB) | safety cap; stops growing here even without a clear knee. **Not** clamped to the quota estimate (that estimate is conservative) - it tries your full value and stops gracefully if the browser refuses |
| Read size (kB) | per-read size (4 kB matches the paper) |
| Buffer (MB) | cache-occupancy channel only: working-set size to sweep, ~your CPU's LLC. Ignored when **auto-size** is checked (the default) - the worker probes for your LLC and fills this in |
| auto-size | cache-occupancy only: probe geometric buffer sizes for the LLC and pick the most sensitive one automatically (on by default, so you don't hand-tune per machine) |
| Channel | `read` (large-file knee), `write-flush` (RAM-independent), or `cache-occupancy` (LLC, no file); see below |

**Negative control:** set Max size below your free RAM (e.g. 8 GB on a 30 GB machine).
Calibration won't find a knee, reads stay cache-served, and disk activity barely shows -
confirming the large-file requirement is what makes the read-latency attack work.

### Channels

There are **three channels** - three different shared resources you can time. They sense
different things, and write-flush is **browser-dependent**. Pick one in the Channel dropdown:

| Channel | Times | Senses | Works on | Needs |
|---------|-------|--------|----------|-------|
| **read** (default) | random 4 kB reads on a >RAM OPFS file | **disk** I/O | **Chrome + Firefox** | OPFS quota > free RAM (to reach the SSD); Firefox caps quota ~10 GB |
| **cache-occupancy** | a sweep of an LLC-sized buffer (pointer-chase) | **CPU / memory / website** | **Chrome + Firefox** | nothing - no file; **auto-size** finds your LLC |
| **write-flush** | `handle.flush()` after a 4 kB write | **disk** writeback | **Firefox only** | a browser whose `flush()` truly fsyncs |

**Which channel for which activity (measured by [`tests/`](tests/), not assumed):**

- **read** detects sustained disk I/O - its read throughput collapses (~−44% under a `dd`
  write) - and fingerprints a heavy site (wired.com) at **~81%** (honest grouped CV; random
  CV reports 91% but that's inflated by session drift - see the trust note under
  [Tests](#tests)). It needs the OPFS quota to exceed free RAM, or reads stay cache-served
  and there's no knee.
- **cache-occupancy** lights up on CPU/memory contention (sweep median ~×2.9 under
  CPU/LLC-thrash) and fingerprints wired.com at **~97%** grouped CV - and with low drift
  (idle-null 62%), so that's a genuinely strong, leakage-robust result. Zero setup, any
  browser - the portable default for sensing apps/sites, and the best fingerprinter here.
- **write-flush is browser-dependent.** On **Firefox** its `flush()` forces a real
  ~hundreds-of-µs disk writeback, so it detects disk writes (median ~×1.34 under `dd`, no
  big file needed) - this is its intended niche when the quota is too small for `read`. On
  **Chrome** `flush()` is a **no-op** (~5 µs, no fsync per
  [MDN](https://developer.mozilla.org/en-US/docs/Web/API/FileSystemSyncAccessHandle/flush) /
  the [Chromium source](https://chromium.googlesource.com/chromium/src/+/220d5e676a4a9e8d501d293c22256bde6320e50a%5E!/)),
  so it senses no disk there and is best avoided.

**In one line:** for *disk* events use **read** (any browser) or **write-flush** (Firefox);
for *CPU/memory/websites* use **cache-occupancy** (any browser).

The **cache-occupancy** channel (Shusterman et al., USENIX Security 2019) is the same
trace -> classifier shape as FROST on a different resource. It allocates a buffer ~the
size of the CPU's last-level cache and walks every cache line as a randomized
**pointer-chase** - each load's address depends on the previous one, which defeats the
hardware prefetcher and serializes the loads so the whole buffer must stay resident.
When another process touches the cache and evicts the buffer's lines, the sweep slows.
Unlike the SSD channels (where the median is flat and the signal lives in throughput /
the tail), here the **median sweep time itself is the primary signal** - the offline
RandomForest already exposes `median`/`mean` features, so cache captures separate with
no pipeline changes. The sensitive regime is when idle sweeps are *cache-resident* (fast);
if the buffer **exceeds** the LLC every access misses to DRAM, the channel saturates, and
contention barely moves it. The right size is therefore per-machine, so **auto-size** (on
by default) probes geometric buffer sizes, watches per-line sweep latency jump as it spills
out of cache, and picks the largest size still in the fast regime - no hand-tuning. Uncheck
it to set **Buffer (MB)** yourself. (Measured: on a small-LLC box an 8 MB buffer sat at
~17 ms/sweep all-miss and showed no contrast; auto-size dropped to ~1-2 MB and gave a clean
median rise + throughput collapse when competing tabs opened.)

The read channel's signal is read latency (and its throughput / spike-rate / tail). The
write-flush channel times `flush()` instead - **but only on a browser where `flush()` is a
real fsync does that reflect the SSD.** On Firefox it is (idle flush ~480 µs, rising under a
`dd` write), so write-flush surfaces disk contention with no big file - useful when the
quota is below RAM. On Chrome `flush()` returns in ~5 µs without touching the disk, so the
channel has no disk signal there (only a faint CPU-scheduling effect on its write-loop
throughput). Captured recordings record which `channel` produced them.

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

## Tests

[`tests/`](tests/) drives each channel end-to-end in a headless browser (Playwright) and
checks two things per channel: (1) it **detects contention** - idle vs a heavy load (`dd`
disk writes for the SSD channels, CPU/LLC-thrash tabs for cache-occupancy); and (2) it can
**fingerprint a website** - collect labeled idle-vs-`nu.nl` windows, train the RandomForest,
and report accuracy with a shuffled-label control so the number isn't a fluke.

```sh
python3 serve.py 8011 &                                   # serve for the tests
cd <playwright-skill> && CHANNEL=cache N_WIN=18 node run.js .../tests/channel_probe.js
python3 tests/analyze.py cache                            # contention + classifier verdict
```

![three channels: contention + website fingerprint](tests/results.png)

Measured findings (one Linux VM; numbers are environment-specific) - see
[`tests/README.md`](tests/README.md):

| channel | detects contention | fingerprints wired.com (honest **grouped** CV) |
|---|---|---|
| **read** | ✅ `dd` write: throughput −44% | **81%** grouped (random 91% leaks; idle-null 92%) |
| **cache-occupancy** | ✅ CPU/LLC-thrash: median ×2.85 | ✅ **97%** grouped (idle-null only 62% → genuinely real, not drift) |
| **write-flush (Firefox)** | ⚠️ `dd` write: median ×1.27 (borderline) | 59% grouped (not a fingerprint channel) |
| **write-flush (Chrome)** | ❌ `flush()` is a no-op | n/a |

> **Trust note - temporal leakage.** We report the website classifier under **grouped CV**
> (train on early windows, test on late) - not random k-fold, which **leaks session/time
> drift**: an *idle-vs-idle* null control (label idle windows by time alone, no website) can
> score as high as the real task, proving random CV partly learns *when* a window was captured.
> Per channel the leakage differs: **read** has heavy drift (random 91% vs grouped 81%, null
> 92%), so trust the 81%; **cache** has little drift (random 96% vs grouped **97%**, null 62%),
> so its fingerprint is genuinely strong. `analyze.py` prints all three (random / grouped /
> null) every run; `tests/eval_cv.py` is the dedicated diagnostic.

> **Leak / overload check.** Recording is heavy (busy probe loops, large traces, many tabs), so
> `channel_probe.js` samples CPU load, browser+node RSS and live tab-count at 1 Hz and prints a
> verdict each run. Across all three channels: **no CPU overload** (peak ≤43% of a 16-core box),
> **no memory leak** (RSS rises with the OPFS read working-set then falls - not monotonic), and
> **no tab/`dd` leaks** (the `dd` load runs in its own process group with a `timeout` backstop
> and is killed cleanly - an earlier bug orphaned a 20 GB `dd` that contaminated the baseline).

The story is per-resource and per-browser: `read` = disk (any browser), `cache-occupancy`
= CPU/memory/website (any browser), `write-flush` = disk **only on Firefox** (where
`flush()` fsyncs). A light site (nu.nl) sits near the noise floor for the disk channels;
a heavy site (wired.com) is detectable - but at the honest ~72% (read), not 93%.

## How it works

- `serve.py` - stdlib HTTP server adding `Cross-Origin-Opener-Policy: same-origin` and
  `Cross-Origin-Embedder-Policy: require-corp` (+ no-cache).
- `worker.js` - runs in a Web Worker (required for the synchronous
  `FileSystemSyncAccessHandle`). In `read` mode it grows the OPFS file with real random
  bytes (zero/sparse pages would be cache-served and deduped, flushing each chunk to bound
  dirty memory), detects the page-cache knee, then loops timing random 4 kB reads. In
  `write-flush` mode it skips the big file and instead times `handle.flush()` after a small
  write - which only reflects the SSD on a browser where `flush()` is a real fsync (Firefox;
  a no-op on Chrome). In `cache-occupancy` mode it uses no file at all: it **auto-sizes** a buffer to
  the LLC (probing for the spill point), links its cache lines into one random pointer-chase
  cycle, and times a full sweep of it.
  Applies the paper's spike filter (samples > 1 ms replaced by the local mean) -
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
