// Main-thread glue: environment check, worker control, live chart, CSV export.

const $ = (id) => document.getElementById(id);
const banner = $("banner");
const logEl = $("log");

// ---- environment / feature detection ----------------------------------------
const isFirefox = /firefox/i.test(navigator.userAgent);
let opfsQuotaBytes = null;       // OPFS quota (Firefox caps ~10 GB); null until probed

// Storage quota is async; probe it once and refresh the banner when it lands.
async function probeQuota() {
  try {
    if (navigator.storage && navigator.storage.estimate) {
      const est = await navigator.storage.estimate();
      opfsQuotaBytes = est && est.quota ? est.quota : null;
    }
  } catch (e) { /* ignore */ }
  checkEnv();
  return opfsQuotaBytes;
}

function checkEnv() {
  const hasOPFS = !!(navigator.storage && navigator.storage.getDirectory);
  const iso = self.crossOriginIsolated === true;
  // Probe timer granularity: smallest non-zero delta we can observe.
  let res = Infinity;
  for (let i = 0; i < 50000; i++) {
    const a = performance.now(), b = performance.now();
    const d = b - a;
    if (d > 0 && d < res) res = d;
  }
  const resStr = res === Infinity ? "?" : (res * 1000).toFixed(3) + " us";

  const lines = [
    `${isFirefox ? "Firefox" : "browser"} | OPFS: ${hasOPFS ? "yes" : "NO - needs a modern browser"}`,
    `crossOriginIsolated: ${iso ? "true (high-res timer unlocked)" : "FALSE - timers coarsened, run via serve.py"}`,
    `timer resolution: ~${resStr}`,
  ];
  if (opfsQuotaBytes) lines.push(`OPFS quota: ~${(opfsQuotaBytes / 1e9).toFixed(0)} GB`);
  if (isFirefox) lines.push(`read channel needs file > free RAM; if quota is too small, use the write-flush channel`);
  const good = hasOPFS && iso;
  banner.className = "banner " + (good ? "ok" : "bad");
  banner.textContent = lines.join("  |  ");
  return good;
}

// ---- chart ------------------------------------------------------------------
const canvas = $("chart");
const ctx = canvas.getContext("2d");
const W = canvas.width, H = canvas.height;
const PAD = 40;
const CAP = 4000;               // points shown on screen
const view = new Float64Array(CAP);
let viewLen = 0, viewHead = 0;
let yMax = 200;                 // us, auto-scaled

function pushView(v) {
  view[viewHead] = v;
  viewHead = (viewHead + 1) % CAP;
  if (viewLen < CAP) viewLen++;
}

function draw() {
  ctx.fillStyle = "#161b22";
  ctx.fillRect(0, 0, W, H);

  // auto-scale y to recent max (with headroom), clamped to a sane floor
  let mx = 50;
  for (let i = 0; i < viewLen; i++) {
    const idx = (viewHead - viewLen + i + CAP) % CAP;
    if (view[idx] > mx) mx = view[idx];
  }
  yMax = yMax * 0.9 + (mx * 1.2) * 0.1; // smooth
  const yScale = (H - 2 * PAD) / yMax;

  // gridlines + labels
  ctx.strokeStyle = "#21262d"; ctx.fillStyle = "#8b949e";
  ctx.font = "11px monospace"; ctx.lineWidth = 1;
  for (let g = 0; g <= 4; g++) {
    const y = PAD + (H - 2 * PAD) * g / 4;
    ctx.beginPath(); ctx.moveTo(PAD, y); ctx.lineTo(W - PAD, y); ctx.stroke();
    const val = yMax * (1 - g / 4);
    ctx.fillText(val.toFixed(0) + " us", 4, y + 4);
  }

  // trace
  ctx.strokeStyle = "#58a6ff"; ctx.lineWidth = 1; ctx.beginPath();
  const xScale = (W - 2 * PAD) / CAP;
  for (let i = 0; i < viewLen; i++) {
    const idx = (viewHead - viewLen + i + CAP) % CAP;
    const x = PAD + i * xScale;
    const y = H - PAD - Math.min(view[idx], yMax) * yScale;
    if (i === 0) ctx.moveTo(x, y); else ctx.lineTo(x, y);
  }
  ctx.stroke();
  requestAnimationFrame(draw);
}
requestAnimationFrame(draw);

// ---- stats + full trace store -----------------------------------------------
// Ring buffer for CSV export, capped so monitoring can run indefinitely without
// leaking memory (this was what crashed the tab before). ~3M samples ~ 3-4 min
// at 14k reads/s and ~ 36 MB of typed-array storage.
const TRACE_CAP = 3_000_000;
const trT = new Float64Array(TRACE_CAP);   // timestamp ms
const trL = new Float32Array(TRACE_CAP);   // latency us
let trLen = 0, trHead = 0;
function pushTrace(t, us) {
  trT[trHead] = t; trL[trHead] = us;
  trHead = (trHead + 1) % TRACE_CAP;
  if (trLen < TRACE_CAP) trLen++;
}
function clearTrace() { trLen = 0; trHead = 0; }

let recent = [];                // sliding window for median/p95
let total = 0, t0 = performance.now();
let monitoring = false;         // worker is in its read loop

// ---- labeled capture for offline fingerprinting -----------------------------
// Each recording is a fixed-length slice of the live sample stream, tagged with
// an activity label. Raw (unfiltered) samples are kept so the offline pipeline
// picks its own features. Exported as JSONL (one recording per line).
const dataset = [];
let capture = null;             // { label, readSize, spikeFilter, endAt, t_ms, lat_us, batch_rate }

function capCounts() {
  const counts = {};
  for (const r of dataset) counts[r.label] = (counts[r.label] || 0) + 1;
  const parts = Object.keys(counts).sort().map((k) => `${k} <b>${counts[k]}</b>`);
  $("capCounts").innerHTML = dataset.length
    ? `dataset: ${dataset.length} recording(s) - ${parts.join(", ")}`
    : "";
}

function setCapStatus(text, cls = "") {
  const el = $("capStatus");
  el.textContent = text;
  el.className = "capstatus" + (cls ? " " + cls : "");
}

function startRecording() {
  if (!monitoring) { appendLog("build & calibrate first - capture needs the monitor running"); return; }
  if (capture) return;
  const label = ($("capLabel").value || "unlabeled").trim();
  const windowS = Math.max(1, parseFloat($("capWindow").value) || 8);
  const raw = $("capRaw").checked;
  worker.postMessage({ cmd: "setFilter", on: !raw });

  // 3-2-1 pre-roll so the researcher can trigger the external activity in sync.
  let n = 3;
  $("record").disabled = true;
  setCapStatus(`recording "${label}" in ${n}...`, "rec");
  const tick = setInterval(() => {
    n--;
    if (n > 0) { setCapStatus(`recording "${label}" in ${n}...`, "rec"); return; }
    clearInterval(tick);
    capture = {
      label, windowS, readSize: Math.round(parseFloat($("readKB").value) * 1024),
      spikeFilter: !raw, channel: $("channel").value, t_ms: [], lat_us: [], batch_rate: [],
      endAt: performance.now() + windowS * 1000,
    };
    setCapStatus(`* recording "${label}" (${windowS}s)...`, "rec");
  }, 1000);
}

function finishRecording() {
  const rec = capture;
  capture = null;
  dataset.push({
    label: rec.label, window_s: rec.windowS, read_size: rec.readSize,
    spike_filter: rec.spikeFilter, channel: rec.channel, t_ms: rec.t_ms,
    lat_us: rec.lat_us, batch_rate: rec.batch_rate,
  });
  appendLog(`captured "${rec.label}": ${rec.lat_us.length} samples over ${rec.windowS}s`);
  capCounts();
  setCapStatus("ready - record another", "ready");
  $("record").disabled = false;
  $("exportDS").disabled = false;
}

function pctl(arr, p) {
  if (!arr.length) return NaN;
  const a = arr.slice().sort((x, y) => x - y);
  return a[Math.min(a.length - 1, Math.floor(p * a.length))];
}

// ---- worker -----------------------------------------------------------------
const worker = new Worker("worker.js", { type: "module" });

worker.onmessage = (e) => {
  const m = e.data;
  switch (m.type) {
    case "log":   appendLog(m.msg); break;
    case "error": appendLog("ERROR: " + m.msg); break;
    case "size":  $("sSize").textContent = (m.bytes / 1e9).toFixed(1) + " GB"; break;
    case "bufsize": $("bufMB").value = (m.bytes / 1e6).toFixed(2); break;
    case "built": appendLog("calibration done @ " + (m.bytes / 1e9).toFixed(1) + " GB"); break;
    case "monitoring":
      monitoring = true;
      $("build").disabled = true; $("stop").disabled = false; $("csv").disabled = false;
      $("record").disabled = false; setCapStatus("ready - type a label and Record", "ready");
      appendLog("monitoring - go cause some disk activity in another window");
      break;
    case "stopped":
      monitoring = false; capture = null;
      $("build").disabled = false; $("stop").disabled = true;
      $("record").disabled = true; setCapStatus("stopped - build & calibrate to capture");
      appendLog("stopped");
      break;
    case "reset":
      clearTrace(); recent = []; total = 0; viewLen = 0; viewHead = 0;
      monitoring = false; capture = null;
      $("sSize").textContent = "-"; $("sTotal").textContent = "-";
      $("build").disabled = false; $("stop").disabled = true; $("csv").disabled = true;
      $("record").disabled = true; setCapStatus("build & calibrate first");
      break;
    case "samples": ingest(m.batch, m.rate); break;
  }
};

function ingest(batch, rate) {
  const now = performance.now();
  for (let i = 0; i < batch.length; i++) {
    const us = batch[i];
    pushView(us);
    recent.push(us);
    pushTrace(now - t0, us);
    if (capture) { capture.t_ms.push(now - t0); capture.lat_us.push(us); }
  }
  if (capture) {
    capture.batch_rate.push(rate);
    if (now >= capture.endAt) finishRecording();
  }
  total += batch.length;
  if (recent.length > 4000) recent = recent.slice(-4000);
  $("sMedian").textContent = pctl(recent, 0.5).toFixed(1);
  $("sP95").textContent = pctl(recent, 0.95).toFixed(1);
  $("sRate").textContent = Math.round(rate);
  $("sTotal").textContent = total.toLocaleString();
}

function appendLog(msg) {
  logEl.textContent += msg + "\n";
  logEl.scrollTop = logEl.scrollHeight;
}

// ---- controls ---------------------------------------------------------------
$("build").onclick = () => {
  if (!checkEnv()) { appendLog("environment not ready - see banner"); return; }
  clearTrace(); recent = []; total = 0; viewLen = 0; viewHead = 0; t0 = performance.now();
  const ch = $("channel").value;
  let maxBytes = Math.round(parseFloat($("maxGB").value) * 1e9);
  // Don't ask for more than the browser will grant (Firefox ~10 GB) - the write
  // would throw QuotaExceededError mid-fill. Leave 5% headroom.
  if (ch === "read" && opfsQuotaBytes && maxBytes > opfsQuotaBytes * 0.95) {
    maxBytes = Math.floor(opfsQuotaBytes * 0.95);
    appendLog(`max size clamped to ${(maxBytes / 1e9).toFixed(1)} GB (OPFS quota ~${(opfsQuotaBytes / 1e9).toFixed(0)} GB)`);
  }
  worker.postMessage({
    cmd: "build",
    channel: ch,
    stepBytes: Math.round(parseFloat($("stepGB").value) * 1e9),
    maxBytes: maxBytes,
    threshUs: parseFloat($("threshUs").value),
    readSize: Math.round(parseFloat($("readKB").value) * 1024),
    bufferBytes: Math.round(parseFloat($("bufMB").value) * 1e6),
    autoTune: ch === "cache" && $("bufAuto").checked,
  });
  appendLog(ch === "flush" ? "building (write-flush channel)..."
          : ch === "cache" ? "building (cache-occupancy channel)..."
          : "building...");
};
$("stop").onclick = () => worker.postMessage({ cmd: "stop" });
$("reset").onclick = () => worker.postMessage({ cmd: "reset" });
$("csv").onclick = () => {
  const rows = ["timestamp_ms,latency_us"];
  for (let i = 0; i < trLen; i++) {
    const idx = (trHead - trLen + i + TRACE_CAP) % TRACE_CAP;
    rows.push(trT[idx].toFixed(1) + "," + trL[idx].toFixed(2));
  }
  const blob = new Blob([rows.join("\n")], { type: "text/csv" });
  const a = document.createElement("a");
  a.href = URL.createObjectURL(blob);
  a.download = "frost-trace.csv";
  a.click();
  URL.revokeObjectURL(a.href);
};

$("record").onclick = startRecording;
$("exportDS").onclick = () => {
  if (!dataset.length) { appendLog("dataset is empty - record some windows first"); return; }
  const lines = dataset.map((r) => JSON.stringify(r));
  const blob = new Blob([lines.join("\n") + "\n"], { type: "application/x-ndjson" });
  const a = document.createElement("a");
  a.href = URL.createObjectURL(blob);
  a.download = `frost-dataset-${dataset.length}.jsonl`;
  a.click();
  URL.revokeObjectURL(a.href);
  appendLog(`exported ${dataset.length} recording(s)`);
};

checkEnv();
probeQuota();   // async: refreshes the banner with the OPFS quota when it resolves
