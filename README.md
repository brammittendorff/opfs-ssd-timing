# FROST PoC - browser hardware-timing side channels

> A web page that works out what *else* your computer is doing - by timing nothing but its
> own disk and cache access. Pure JavaScript, no permissions, no exploit.

When a program hammers a shared resource - your SSD, your CPU's cache - everything else using
it slows down a touch. A sandboxed browser tab can't *see* other programs, but it **can** time
its own access to those shared resources and watch for the slowdowns. From that timing alone it
can tell *when* you open an app, load a site, or copy a file - and a classifier can start to
guess *what*.

This is a local, self-contained PoC of that idea - three such channels plus an offline
fingerprinting pipeline. Built on **[FROST](https://hannesweissteiner.com/pdfs/frost.pdf)**
(Weissteiner et al., TU Graz - the OPFS/SSD method) and the **cache-occupancy** channel of
**[Shusterman et al.](https://www.usenix.org/conference/usenixsecurity19/presentation/shusterman)**
(USENIX Security 2019).

> Research/education on your own hardware - not for deployment against others. No covert channel included.

## The three channels

| Channel | What it times | What it senses | Browser |
|---|---|---|---|
| **read** *(FROST)* | random 4 kB reads of a `>RAM` OPFS file | **disk** I/O | Chrome / Firefox - needs quota > free RAM |
| **cache-occupancy** | a sweep of an LLC-sized buffer (pointer-chase) | **CPU / memory / websites** | Chrome / Firefox - no file - **start here** |
| **write-flush** | `flush()` after a 4 kB write | **disk** writeback | **Firefox only** (Chrome's `flush()` is a no-op) |

Rule of thumb: **disk** events -> `read` or `write-flush`; **apps & websites** -> `cache-occupancy`.

## Quick start

```sh
python3 serve.py        # http://localhost:8000  (sets the COOP/COEP headers)
```

1. Open **http://localhost:8000** in Chrome or Firefox. The headers make the page
   cross-origin-isolated, which unlocks the high-res timer - opening the HTML directly
   (`file://`) won't work.
2. Pick a **Channel** (`cache-occupancy` is the zero-setup default) and click **Build & calibrate**.
3. Cause activity in another window - open a site, launch an app, `cp` a big file. Watch the
   chart spike; go idle and watch it settle. That contrast is the leak.

[Demo video](https://github.com/brammittendorff/opfs-ssd-timing/raw/main/videos/opfs-ssd-timing.mp4)

## How it works (plain English)

Shared parts of your computer are like **roads**: when one program drives heavily on a road,
everyone else slows down. The page times **its own** trips and notices the slowdowns. It
watches two roads - the **SSD** (timing reads of a big OPFS file, or how long a write takes to
flush to disk) and the **CPU cache** (timing a sweep of a cache-sized buffer). It never sees
your files; it only feels the traffic.

> **One catch - `read` only:** the OPFS file must be **bigger than free RAM**, or it stays
> fully cached and never touches the disk. `cache-occupancy` and `write-flush` have no size
> requirement.

## What's in the box

- **Live monitor** - `index.html` + `main.js` + `worker.js`: the three channels, adaptive
  calibration, a dependency-free rolling chart, CSV export.
- **Capture harness** - record labeled windows of the signal -> JSONL.
- **Fingerprinting** - [`analysis/`](analysis/): feature extraction + RandomForest + 1D-CNN
  that classify *what* caused the contention.
- **Tests** - [`tests/`](tests/): an end-to-end test harness for each channel (see
  [`tests/README.md`](tests/README.md)).

## Caveats

- **One disk assumed** - activity on a *different* physical disk than the browser's OPFS won't register.
- **Real I/O** - `read` writes many GB on first build (some SSD wear); **Reset** removes the file.
- **Mitigation (paper):** cap OPFS to ~1 GB so the file fits in RAM -> the `read` channel disappears.
