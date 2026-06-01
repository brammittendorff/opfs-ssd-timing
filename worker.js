// FROST PoC worker: owns the OPFS file, grows it until it defeats the page
// cache, then continuously times random reads to surface SSD contention.
//
// Runs in a Web Worker because FileSystemSyncAccessHandle (the fast, synchronous
// OPFS read/write path) is only available off the main thread.

const FILE_NAME = "frost-contention.bin";
const PROBE_READS = 2000;        // reads per calibration probe
const DEBOUNCE = 2;              // consecutive over-threshold probes to confirm the knee
const REPORT_EVERY = 256;        // read/flush: batch this many fast samples before posting
const REPORT_EVERY_CACHE = 4;    // cache: one LLC sweep is already a ~ms-scale sample, batch few
let reportEvery = REPORT_EVERY;  // active batch size (set per channel in startMonitor)
const SPIKE_LIMIT_US = 1000;     // paper's filter: samples > 1 ms are smoothed away
const SPIKE_WINDOW = 100;        // local-average window for the spike filter

let handle = null;               // FileSystemSyncAccessHandle
let fileSize = 0;                // bytes actually written (== logical size)
let readSize = 4096;
let readBuf = null;              // DataView/ArrayBuffer reused for every read
let running = false;             // continuous monitor active
let abortBuild = false;
let spikeFilter = true;          // smooth >1ms samples for the live chart; off for ML capture
let channel = "read";            // "read" (large-file knee), "flush" (RAM-independent), or "cache" (LLC occupancy)
let probeBuf = null;             // small write buffer for the flush probe
let flushOffset = 0;             // rotating write offset for the flush probe

// Cache-occupancy channel: an LLC-sized buffer walked as a randomized pointer-chase.
let chase = null;                // Int32Array over the buffer; each line stores its next-line index
let nLines = 0;                  // number of cache lines in the buffer

const FLUSH_FILE_BYTES = 64 * 1024 * 1024; // flush mode only needs a small file to write into
const LINE_BYTES = 64;           // cache-line size assumed for the sweep stride
let sweepSink = 0;               // dead-code-elimination sink for the sweep accumulator

function log(msg)  { postMessage({ type: "log", msg }); }
function err(msg)  { postMessage({ type: "error", msg }); }

function median(arr) {
  const a = arr.slice().sort((x, y) => x - y);
  const m = a.length >> 1;
  return a.length % 2 ? a[m] : (a[m - 1] + a[m]) / 2;
}

// crypto.getRandomValues throws on any view > 65536 bytes. Fill an arbitrary-size
// write payload by seeding one 64 kB block of real randomness and tiling it across
// the rest. Fine for write buffers (we only need non-zero, non-dedupable bytes, not
// per-byte entropy) - NOT for the cache shuffle, which needs fresh entropy throughout.
function randomFill(view) {
  const SEED = Math.min(view.length, 65536);
  crypto.getRandomValues(view.subarray(0, SEED));
  for (let o = SEED; o < view.length; o += SEED) {
    view.copyWithin(o, 0, Math.min(SEED, view.length - o));
  }
}

async function getDir() {
  return await navigator.storage.getDirectory();
}

// One timed random 4 kB-aligned read. Returns latency in microseconds.
function timedRead() {
  const slots = Math.max(1, Math.floor(fileSize / readSize));
  // Math.random is fine here: we only need offsets spread across the file to
  // defeat prefetch/caching, not cryptographic randomness.
  const offset = Math.floor(Math.random() * slots) * readSize;
  const t0 = performance.now();
  handle.read(readBuf, { at: offset });
  const t1 = performance.now();
  return (t1 - t0) * 1000; // ms -> us
}

// One timed write + flush. Returns the flush latency in microseconds.
// IMPORTANT: contrary to a common assumption, FileSystemSyncAccessHandle.flush() is
// NOT an fsync/fdatasync. Per MDN it commits to the OS ("leave the OS to handle
// [physical writes] when it sees fit") and Chromium implements it asynchronously / as a
// no-op for in-memory backends - there is no fsync primitive in the browser. So on
// Chromium this does NOT reliably force an SSD write: flush() returns in ~1 timer tick
// and the kernel writes back lazily. The channel's contention signal therefore comes
// from write-loop *throughput* / scheduling, not flush latency - it senses CPU/write
// pressure (and websites) more than raw disk writes. RAM-independent variant; the read
// channel is the paper's actual disk method.
function timedFlush() {
  const slots = Math.max(1, Math.floor(fileSize / readSize));
  const offset = (flushOffset++ % slots) * readSize;
  handle.write(probeBuf, { at: offset });
  const t0 = performance.now();
  handle.flush();
  const t1 = performance.now();
  return (t1 - t0) * 1000; // ms -> us
}

// ---- cache-occupancy channel ------------------------------------------------
// Allocate a buffer ~LLC-sized and link its cache lines into one random cycle, with
// each line storing the index of the next line *inside the buffer itself*. Walking
// the chain (`p = buf32[p*STRIDE32]`) is a chain of dependent loads that touches the
// whole buffer in an order the hardware prefetcher can't predict, so a full traversal
// must keep all those lines resident. When another tenant evicts them from the LLC,
// the resulting misses make the sweep visibly slower. No OPFS file, no SSD writes, no
// page-cache knee - just LLC contention.
const STRIDE32 = LINE_BYTES / 4;       // Int32 slots per cache line (16)

function setupCacheBuffer(bytes) {
  nLines = Math.max(2, Math.floor(bytes / LINE_BYTES));
  const ab = new ArrayBuffer(nLines * LINE_BYTES);
  chase = new Int32Array(ab);           // view used to store + follow the next-line links

  // Single random cycle over [0, nLines) via Sattolo's algorithm on an identity
  // order, then write order[k] -> order[k+1] as a link at the head of each line.
  // Randomness is drawn in bulk (per-call crypto overhead would dominate otherwise).
  const order = new Int32Array(nLines);
  for (let i = 0; i < nLines; i++) order[i] = i;
  const rnd = new Uint32Array(Math.min(nLines, 16384)); // 16384*4 = 65536 B, the getRandomValues cap
  let r = rnd.length;
  for (let i = nLines - 1; i > 0; i--) {
    if (r >= rnd.length) { crypto.getRandomValues(rnd); r = 0; }
    const j = rnd[r++] % i;             // Sattolo: 0 <= j < i (one full cycle)
    const t = order[i]; order[i] = order[j]; order[j] = t;
  }
  for (let k = 0; k < nLines; k++) {
    chase[order[k] * STRIDE32] = order[(k + 1) % nLines];
  }
}

// One timed full pointer-chase sweep of the buffer. Each step reads the next line
// index from the current line, so the loads are serialized and span the whole buffer.
// Returns latency in microseconds.
function timedSweep() {
  let p = 0, sum = 0;
  const buf = chase, n = nLines;
  const t0 = performance.now();
  for (let i = 0; i < n; i++) {
    p = buf[p * STRIDE32];              // dependent load: address comes from prev step
    sum += p;
  }
  const t1 = performance.now();
  sweepSink ^= sum;                     // keep `sum` live so the loop can't be elided
  return (t1 - t0) * 1000;              // ms -> us
}

// Auto-size the sweep buffer to ~the LLC so the user doesn't have to guess. Probe
// geometric buffer sizes, timing per-line sweep latency: while the buffer fits in
// cache it stays low, then jumps once the buffer spills to DRAM (exceeds the LLC).
// Pick the largest size still in the fast regime - buffer ~= LLC, where idle sweeps
// are cache-resident and a co-tenant evicting the cache shows up sharply. The LLC
// differs per machine, so this runs per environment instead of hardcoding a size.
const AUTO_MIN_BYTES = 256 * 1024;
const AUTO_MAX_BYTES = 64 * 1024 * 1024;
const KNEE_MULT = 2.5;                  // ns/line this far over baseline => past the LLC

async function autoTuneCacheBuffer() {
  log("auto-sizing cache buffer (probing for the LLC)...");
  let baseline = Infinity, chosen = AUTO_MIN_BYTES, over = 0;
  for (let bytes = AUTO_MIN_BYTES; bytes <= AUTO_MAX_BYTES; bytes *= 2) {
    if (abortBuild) break;
    setupCacheBuffer(bytes);
    timedSweep();                       // warm
    const probes = new Array(7);
    for (let i = 0; i < probes.length; i++) probes[i] = timedSweep();
    const med = median(probes);
    const perLine = (med * 1000) / nLines;          // ns per line
    if (perLine < baseline) baseline = perLine;
    log(`  ${(bytes / 1e6).toFixed(2).padStart(6)} MB -> ${med.toFixed(0).padStart(6)} us/sweep ` +
        `(${perLine.toFixed(1)} ns/line)${perLine > KNEE_MULT * baseline ? " >" : ""}`);
    if (perLine <= KNEE_MULT * baseline) {
      chosen = bytes; over = 0;         // still resident: take the largest such size
    } else if (++over >= 2) {
      break;                            // two sizes past the knee: LLC found, stop probing
    }
    await new Promise((r) => setTimeout(r, 0));
  }
  log(`auto-sized to ${(chosen / 1e6).toFixed(2)} MB (~LLC; idle baseline ${baseline.toFixed(1)} ns/line)`);
  return chosen;
}

// ---- calibration: grow + probe until reads stop being cache-served ----------
async function build(opts) {
  abortBuild = false;
  readSize = opts.readSize;
  readBuf = new Uint8Array(readSize);

  const stepBytes = opts.stepBytes;
  const maxBytes = opts.maxBytes;
  const threshUs = opts.threshUs;
  channel = opts.channel === "flush" ? "flush" : opts.channel === "cache" ? "cache" : "read";

  // Cache-occupancy channel: no OPFS file at all. Allocate the LLC-sized buffer,
  // wire up the pointer-chase, and go straight to monitoring - no fill, no knee.
  if (channel === "cache") {
    if (handle) { try { handle.flush(); handle.close(); } catch (e) {} handle = null; }
    let bytes;
    if (opts.autoTune) {
      bytes = await autoTuneCacheBuffer();
      if (abortBuild) { log("auto-tune aborted"); return; }
      postMessage({ type: "bufsize", bytes });    // reflect the chosen size in the UI
    } else {
      bytes = Math.max(LINE_BYTES * 2, opts.bufferBytes | 0);
    }
    setupCacheBuffer(bytes);
    const actual = nLines * LINE_BYTES;
    postMessage({ type: "size", bytes: actual });
    log(`cache-occupancy channel ready: ${(actual / 1e6).toFixed(1)} MB buffer, ` +
        `${nLines} lines (pointer-chase) - timing LLC sweeps (no file)`);
    postMessage({ type: "built", bytes: actual });
    startMonitor();
    return;
  }

  // Release any handle left open by a previous build/monitor session, otherwise
  // createSyncAccessHandle throws ("another open Access Handle ... same file").
  if (handle) { try { handle.flush(); handle.close(); } catch (e) {} handle = null; }

  const dir = await getDir();
  const fh = await dir.getFileHandle(FILE_NAME, { create: true });
  handle = await fh.createSyncAccessHandle();
  fileSize = handle.getSize();
  log(`opened OPFS file, current size ${(fileSize / 1e9).toFixed(2)} GB`);

  // Write-flush channel: time handle.flush() instead of reads. RAM-independent,
  // so there is no page-cache knee to find and only a small file is needed.
  if (channel === "flush") {
    probeBuf = new Uint8Array(readSize);
    randomFill(probeBuf);            // handles readSize > 64 kB (getRandomValues' cap)
    if (fileSize < FLUSH_FILE_BYTES) {
      const CH = 16 * 1024 * 1024;
      const wb = new Uint8Array(Math.min(CH, FLUSH_FILE_BYTES));
      randomFill(wb);
      let pos = fileSize;
      while (pos < FLUSH_FILE_BYTES) {
        const n = Math.min(wb.length, FLUSH_FILE_BYTES - pos);
        handle.write(wb.subarray(0, n), { at: pos });
        handle.flush();
        pos += n;
      }
      fileSize = handle.getSize();
    }
    postMessage({ type: "size", bytes: fileSize });
    log(`write-flush channel ready at ${(fileSize / 1e9).toFixed(2)} GB - timing flush() (RAM-independent)`);
    postMessage({ type: "built", bytes: fileSize });
    startMonitor();
    return;
  }

  // If the existing file already reads at SSD latency, it was calibrated in a
  // previous session - skip the (slow, SSD-wearing) fill and just continue.
  if (fileSize >= readSize) {
    const probe = new Array(PROBE_READS);
    for (let i = 0; i < PROBE_READS; i++) probe[i] = timedRead();
    const med = median(probe);
    log(`existing file ${(fileSize / 1e9).toFixed(1)} GB, probe median ${med.toFixed(1)} us`);
    if (med >= threshUs) {
      log(`already past the knee - continuing without growing the file`);
      postMessage({ type: "built", bytes: fileSize });
      startMonitor();
      return;
    }
  }

  // A reusable buffer of random bytes to write (avoids zero-page dedup/sparseness).
  const CHUNK = 16 * 1024 * 1024;        // 16 MB write chunks
  const YIELD_BYTES = 512 * 1024 * 1024; // yield to the event loop every 512 MB
  const writeBuf = new Uint8Array(CHUNK);
  randomFill(writeBuf);                  // seed 64 kB of entropy, tile across the chunk

  let over = 0;
  let quotaHit = false;
  while (true) {
    if (abortBuild) { log("build aborted"); break; }
    if (fileSize >= maxBytes) {
      log(`reached the ${(maxBytes / 1e9).toFixed(1)} GB cap with reads still cache-served - ` +
          `your free RAM exceeds the OPFS quota, so the read channel can't reach the SSD on ` +
          `this machine. Switch Channel to "cache-occupancy" (no file, auto-sized) or ` +
          `"write-flush" - neither needs a file bigger than RAM.`);
      break;
    }

    // Grow by one step of real random data.
    const target = Math.min(fileSize + stepBytes, maxBytes);
    let pos = fileSize;
    let sinceYield = 0;
    while (pos < target) {
      if (abortBuild) break;
      const n = Math.min(CHUNK, target - pos);
      // vary a few bytes per chunk so the controller can't trivially dedup
      writeBuf[0] = pos & 0xff; writeBuf[1] = (pos >> 8) & 0xff;
      try {
        handle.write(writeBuf.subarray(0, n), { at: pos });
        // Flush every chunk: keep dirty pages ~CHUNK-bounded instead of letting a
        // whole GB-sized step pile up unwritten - that backlog (with no swap) is
        // what OOM-kills the browser on a from-scratch fill.
        handle.flush();
      } catch (ex) {
        if (ex && (ex.name === "QuotaExceededError" || /quota/i.test(ex.message || ""))) {
          log(`hit the OPFS quota near ${(pos / 1e9).toFixed(1)} GB ` +
              `(browser cap, e.g. Firefox ~10 GB) - stopping fill`);
          quotaHit = true;
          break;
        }
        throw ex;
      }
      pos += n;
      sinceYield += n;
      if (sinceYield >= YIELD_BYTES) {
        // Let the kernel write back / reclaim, and stay responsive to "stop".
        sinceYield = 0;
        fileSize = pos;
        postMessage({ type: "size", bytes: fileSize });
        await new Promise((r) => setTimeout(r, 0));
      }
    }
    fileSize = handle.getSize();
    postMessage({ type: "size", bytes: fileSize });
    if (quotaHit) {
      log(`monitoring at ${(fileSize / 1e9).toFixed(1)} GB - quota-limited; the read ` +
          `channel only leaks if this exceeds free RAM (else try the write-flush channel)`);
      break;
    }

    // Probe: random reads across the whole file, take the median.
    const samples = new Array(PROBE_READS);
    for (let i = 0; i < PROBE_READS; i++) samples[i] = timedRead();
    const med = median(samples);
    log(`size ${(fileSize / 1e9).toFixed(1).padStart(5)} GB -> probe median ` +
        `${med.toFixed(1)} us ${med >= threshUs ? ">" : ""}`);

    if (med >= threshUs) {
      if (++over >= DEBOUNCE) {
        log(`KNEE reached: reads now hit the SSD (median ${med.toFixed(1)} us >= ` +
            `${threshUs} us) at ${(fileSize / 1e9).toFixed(1)} GB. Stopping fill.`);
        break;
      }
    } else {
      over = 0;
    }
  }

  postMessage({ type: "built", bytes: fileSize });
  startMonitor();
}

// ---- continuous monitoring loop ---------------------------------------------
function startMonitor() {
  running = true;
  // A single LLC sweep already costs ~1-2 ms, so batching 256 of them would block
  // the worker (and stall the UI) for hundreds of ms per post. Report far fewer.
  reportEvery = channel === "cache" ? REPORT_EVERY_CACHE : REPORT_EVERY;
  postMessage({ type: "monitoring" });
  loop();
}

function loop() {
  if (!running) return;
  const batch = new Float64Array(reportEvery);
  const probe = channel === "flush" ? timedFlush : channel === "cache" ? timedSweep : timedRead;
  const t0 = performance.now();
  for (let i = 0; i < reportEvery; i++) batch[i] = probe();

  // Spike filter (paper sec4): replace any sample > 1 ms with the local mean.
  // Skipped during ML capture so the offline pipeline sees the raw signal.
  if (spikeFilter) {
    for (let i = 0; i < batch.length; i++) {
      if (batch[i] > SPIKE_LIMIT_US) {
        let sum = 0, cnt = 0;
        const lo = Math.max(0, i - SPIKE_WINDOW / 2);
        const hi = Math.min(batch.length, i + SPIKE_WINDOW / 2);
        for (let j = lo; j < hi; j++) if (j !== i && batch[j] <= SPIKE_LIMIT_US) { sum += batch[j]; cnt++; }
        if (cnt) batch[i] = sum / cnt;
      }
    }
  }
  const elapsedMs = performance.now() - t0;
  postMessage({ type: "samples", batch, rate: (reportEvery / elapsedMs) * 1000 },
              [batch.buffer]);

  // Yield so the worker stays responsive to stop/reset messages.
  setTimeout(loop, 0);
}

async function reset() {
  running = false;
  chase = null; nLines = 0;          // drop the cache-occupancy buffer
  if (handle) { try { handle.close(); } catch (e) {} handle = null; }
  try {
    const dir = await getDir();
    await dir.removeEntry(FILE_NAME);
    log("deleted OPFS file");
  } catch (e) {
    log("nothing to delete (" + e.name + ")");
  }
  fileSize = 0;
  postMessage({ type: "reset" });
}

onmessage = async (e) => {
  const m = e.data;
  try {
    if (m.cmd === "build") await build(m);
    else if (m.cmd === "stop") {
      running = false; abortBuild = true;
      // Close the handle so OPFS durably commits the file's size - otherwise the
      // calibrated file reverts to its last cleanly-closed size on reload.
      if (handle) { try { handle.flush(); handle.close(); } catch (e) {} handle = null; }
      postMessage({ type: "stopped" });
    }
    else if (m.cmd === "reset") await reset();
    else if (m.cmd === "setFilter") { spikeFilter = !!m.on; log(`spike filter ${spikeFilter ? "on" : "off"}`); }
  } catch (ex) {
    err((ex && ex.message) || String(ex));
  }
};
