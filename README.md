# FROST PoC — browser SSD & cache timing side channel

A local, self-contained proof-of-concept of **[FROST: Fingerprinting Remotely using
OPFS-based SSD Timing](https://hannesweissteiner.com/pdfs/frost.pdf)** (Weissteiner et al.,
TU Graz). Pure JavaScript in a browser tab measures **hardware contention** by timing a
shared resource — no native code, no exploit, no permission prompt. When anything else on
the machine touches that resource, the timing spikes, and you watch it live.

> Research/education on your own hardware. Don't deploy against others. (No covert channel included.)

![three channels: contention response + website fingerprint](tests/results.png)

## The three channels

| Channel | Times | Senses | Works on |
|---|---|---|---|
| **read** *(paper's method)* | random 4 kB reads on a `>RAM` OPFS file | **disk** I/O | Chrome + Firefox (needs OPFS quota > free RAM) |
| **cache-occupancy** | a sweep of an LLC-sized buffer (pointer-chase) | **CPU / memory / websites** | Chrome + Firefox, no file — **start here** |
| **write-flush** | `handle.flush()` after a 4 kB write | **disk** writeback | **Firefox only** (Chrome's `flush()` is a no-op) |

For *disk* activity use **read** (any browser) or **write-flush** (Firefox); for
*CPU / memory / websites* use **cache-occupancy** (any browser).

## How it works (plain English)

Your computer's shared parts are like **roads** — when one program uses a road heavily,
everyone else on it slows down. A website can't see other programs, but it *can* time **its
own** trips on a shared road and notice the slowdowns. This PoC watches two roads: the
**SSD** (timing reads of a big OPFS file, or timing a write's flush-to-disk) and the **CPU
cache** (timing a sweep of a cache-sized buffer). It never sees your files — only the timing.
From the slowdowns it tells *when* the machine is busy, and the [offline pipeline](analysis/)
tries to fingerprint *what*.

One catch, **`read` only**: the OPFS file must be **bigger than free RAM**, or it stays fully
cached and never touches the disk. The other two channels have no size requirement.

## Run

```sh
python3 serve.py        # serves http://localhost:8000 with COOP/COEP headers
```

Open **http://localhost:8000** in Chrome or Firefox — the COOP/COEP headers make the page
cross-origin-isolated, unlocking the high-res timer (`file://` won't work). Pick a
**Channel**, click **Build & calibrate**, then cause activity in another window and watch the
chart settle/spike. `cache-occupancy` is the zero-setup default; `read` needs a `>RAM` OPFS
file (Chrome has the most headroom, Firefox caps OPFS at ~10 GB); `write-flush` is the disk
channel for Firefox. [Demo video](https://github.com/brammittendorff/opfs-ssd-timing/raw/main/videos/opfs-ssd-timing.mp4).

## Fingerprinting

The live monitor shows *that* the machine is busy; fingerprinting asks *what*. The
**Capture** panel records labeled windows → JSONL, and the [`analysis/`](analysis/) pipeline
(RandomForest + 1D-CNN) classifies them. See [`analysis/README.md`](analysis/README.md).

## Tests & results

[`tests/`](tests/) drives each channel end-to-end and checks it (a) detects contention and
(b) fingerprints a website — using leakage-robust **grouped CV**, an idle-vs-idle null
control, and a per-run CPU/memory/leak check. Measured on one Linux VM (environment-specific):
cache-occupancy fingerprints wired.com at **~88–97% grouped CV**; `read` detects `dd` writes
(throughput **−80…−95%**); `write-flush` works as a disk channel **only on Firefox**. Full
numbers, methodology and the figure above: [`tests/README.md`](tests/README.md).

## UI fields

- **Grow step / Max size / Read size** — `read` channel: how fast/large the file grows, and
  per-read size. Max is **not** clamped to the quota estimate (that estimate is conservative).
- **Buffer (MB) / auto-size** — `cache` channel: working-set size; **auto-size** (default)
  finds your LLC, so you don't tune by hand.
- **Knee threshold** — vestigial fallback; the real `read` detector is "% of reads that hit
  the SSD", measured against the device's own cached baseline (timer-resolution-independent).

## Caveats

- **Single SSD assumed** — activity on a *different* physical disk than the browser's OPFS
  won't register (paper's stated limitation).
- **Real disk I/O** — `read` writes many GB on first build (some SSD wear); **Reset** removes
  the file.
- **Mitigation (paper):** cap OPFS to ~1 GB → the file fits in RAM → the `read` channel
  disappears.

## How it's built

- `serve.py` — static server adding COOP/COEP (cross-origin isolation).
- `worker.js` — the three channels (OPFS reads / `flush()` / LLC pointer-chase sweep) in a
  Web Worker, with adaptive calibration.
- `index.html` + `main.js` — controls, live `<canvas>` chart, CSV export, labeled-capture harness.
- `analysis/` — offline feature extraction + classifiers. `tests/` — end-to-end test suite.
