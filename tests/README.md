# Channel tests

End-to-end tests that drive each FROST channel in a headless browser. They reuse the offline
pipeline in [`../analysis`](../analysis) (`dataset.py` + `features.py`) for feature extraction
and the RandomForest, so a green test here also exercises that code path.

## Advanced: real-activity fingerprinting

`activities.js` captures labeled windows of **real activities** - `idle`, a real **website**
load, a real 5 GB file **`cp`**, and a CPU/memory **`compute`** process - and `eval_activities.py`
runs a **multi-class, leakage-robust** evaluation (grouped CV = train early windows, test late;
confusion matrix; per-activity one-vs-rest; shuffled-label null + permutation p). This replaces
the earlier synthetic `dd` hammer with representative loads.

```sh
python3 serve.py 8000 &
SKILL=<playwright-skill>
CHANNEL=cache N_PER=8 node "$SKILL/run.js" tests/activities.js   # then CHANNEL=read
python3 tests/eval_activities.py cache   # and read
```

**Measured (one Linux VM, 8 windows/class - environment-specific, small N):**

| one-vs-rest (grouped CV) | **cache** | **read** |
|---|---|---|
| idle | 100% | 81% (~chance) |
| web (real site) | 91% | 78% (~chance) |
| cp (real 5 GB copy) | 97% | 97% |
| compute (CPU/mem) | 97% | 94% |
| **4-class grouped CV** | **97%** | **78%** |
| null / permutation p | 18% / p~0.03 | 22% / p~0.03 |

Honest reading: both channels carry **real signal** (null ~ chance). The **cache** channel
separates all four activities cleanly; the **read** (disk) channel nails the disk-heavy `cp` and
the heavy `compute` (the latter by starving the read loop's CPU), but **can't tell idle from a
light website** - a quick page load barely touches the disk. The `p` floor is 1/31 (30
permutations); N is only 8/class, so treat the exact percentages as indicative, not precise.

## What the basic tests check

For one channel (`read`, `flush`, or `cache`) a single monitoring session runs two phases:

1. **Contention** - idle -> heavy load -> idle. Load is `dd ... oflag=direct` (sustained
   real SSD writes) for the SSD channels and 3 CPU/LLC-thrash tabs for cache-occupancy.
   PASS if the busy window separates from idle (throughput drop, or p95/median rise).
2. **Fingerprint** - `N_WIN` alternating idle / site windows, labeled. Accuracy is reported
   **three ways** so it's honest (see `eval_cv.py`):
   - *random KFold* - optimistic; **leaks** session/time drift (train & test windows can be
     temporally adjacent), so it over-states accuracy.
   - *grouped CV* - the **honest headline**: train on early windows, test on late ones, so the
     model must generalise across the session rather than memorise drift.
   - *idle-vs-idle null* - a control that **must** sit at ~chance; if it doesn't, the pipeline
     is classifying *time*, not the site.
   PASS = honest **grouped** accuracy >= 70%. (We found random CV inflated read 93% -> grouped
   72%, with the null at ~94% - i.e. the 93% was mostly drift.)

## Run

```sh
python3 serve.py 8011 &                  # from the repo root; tests target :8011

# one channel: collect data, then analyze
SKILL=<path-to-playwright-skill>
CHANNEL=cache N_WIN=18 node "$SKILL/run.js" tests/channel_probe.js
python3 tests/analyze.py cache
```

`channel_probe.js` writes `/tmp/frost-<channel>.csv` + `-marks.json`; `analyze.py` reads
them and prints the two verdicts. Env:

- `CHANNEL` - `read` | `flush` | `cache`
- `BROWSER` - `chromium` (default) | `firefox`. **Run write-flush with `BROWSER=firefox`** -
  its `flush()` only fsyncs there (`npx playwright install firefox` first).
- `N_WIN` - windows per class (default 5; use ~18 for a trustworthy classifier number)
- `LOAD` - override the contention load: `dd` (disk) | `burn` (CPU/memory)
- `SITE`, `URL` - target site (default wired.com via the heavy path) and the PoC URL

```sh
# write-flush as a real disk channel (Firefox) vs a dd write:
BROWSER=firefox CHANNEL=flush LOAD=dd N_WIN=1 node "$SKILL/run.js" tests/channel_probe.js
python3 tests/analyze.py flush
```

## Measured results (one Linux VM, small LLC, NVMe; results are environment-specific)

![three channels: contention response + website fingerprint](results.png)

Website fingerprint = idle vs **wired.com** (heavy site), 16 windows/class, RF 5-fold x40
reps; a light site (nu.nl) sits near the noise floor for the disk channels.

| channel | browser | contention | website fingerprint (**grouped** CV = honest) |
|---------|---------|-----------|---------------------|
| **read** | Chrome | `dd` write: throughput -44% | **81%** grouped (random 91% leaks; idle-null 92%) |
| **cache-occupancy** | Chrome | CPU/LLC-thrash: median x2.85 | **97%** grouped (idle-null 62% -> genuinely real) |
| **write-flush** | **Firefox** | `dd` write: median x1.27 (borderline) | 59% grouped (not a fingerprint channel) |
| **write-flush** | Chrome | `flush()` ~5 us **no-op** (no fsync) | n/a |

**Temporal-leakage correction:** website numbers use **grouped CV** (early windows -> late),
not random k-fold, which **leaks session drift** (an idle-vs-idle null control can score as
high as the real task). The leakage differs per channel: **read** has heavy drift (random 91%
vs grouped **81%**, null 92%); **cache** has little (random 96% vs grouped **97%**, null 62%) so
its fingerprint is genuinely strong. `analyze.py` prints random / grouped / null every run;
`eval_cv.py` is the dedicated diagnostic.

**Leak / overload check.** Recording is heavy, so `channel_probe.js` samples CPU load,
browser+node RSS and tab-count at 1 Hz (written to `/tmp/frost-<channel>-resources.json`) and
prints a verdict. All three channels came back clean: peak CPU <=43% of a 16-core box (no
overload), RSS rises with the OPFS read working-set then falls (no monotonic leak), no tab leak,
`dd` killed cleanly. The harness runs `dd` in its own process group with a `timeout` backstop
and bounded size - an earlier bug orphaned a 20 GB `dd` that ran ~20 s into later windows and
inflated results (it's why flush's disk effect was overstated at x1.34 vs the clean x1.27).

**The two takeaways:**

1. **Per resource:** `read` = disk, `cache-occupancy` = CPU/memory/website. They are not
   interchangeable - a heavy site is detected by both, a `dd` write only by `read`, a pure
   CPU/LLC thrash only by `cache`.
2. **Per browser:** `write-flush` only works where `FileSystemSyncAccessHandle.flush()` is
   a real fsync. On **Firefox** it is (idle flush ~480 us vs Chrome's ~5 us), so it becomes
   a genuine disk channel. On **Chrome** `flush()` is a no-op, so it senses no disk - run it
   with `BROWSER=firefox`.

Numbers vary by machine (LLC size, free RAM, SSD, OPFS quota) and browser, which is why the
tests measure rather than assume - and why cache-occupancy auto-sizes its buffer per machine.

## Files

| File | Role |
|------|------|
| `activities.js` / `eval_activities.py` | **advanced real-activity test** - capture idle/web/cp/compute, multi-class grouped-CV + confusion + per-activity detectability + null |
| `channel_probe.js` | Playwright driver: build a channel, run contention + fingerprint phases, export CSV + time-marks; also samples CPU/RSS/tabs at 1 Hz and prints a leak/overload verdict |
| `analyze.py` | slice the trace by marks; print contention verdict + classifier accuracy as random / **grouped** / idle-null |
| `eval_cv.py` | dedicated temporal-leakage diagnostic (random vs grouped CV + idle-vs-idle null) |
| `stats.py` | bootstrap CI + permutation p-value + sample-size estimate for the classifier accuracy |
| `eval_matrix.py` / `matrix.sh` | channel x load detection matrix (AUC / median ratio per cell) |
| `collect_sites.js` / `eval_multiclass.py` | multi-site closed-world + open-world fingerprinting |
