// Parametric test driver for the three FROST channels.
//
// For one channel (env CHANNEL = read | flush | cache) it runs one monitoring
// session with two phases and exports a raw CSV + a JSON of labeled time-marks:
//   1. CONTENTION: idle -> heavy load -> idle. Load is `dd` (disk) for the read /
//      write-flush channels and CPU/LLC-thrash tabs for cache-occupancy.
//   2. FINGERPRINT: alternating idle / nu.nl windows, labeled, for the classifier.
//
// Usage (via the playwright-skill run.js):
//   CHANNEL=cache node run.js tests/channel_probe.js
//
// Outputs: /tmp/frost-<CHANNEL>.csv and /tmp/frost-<CHANNEL>-marks.json
const playwright = require('playwright');
const { execSync, spawn } = require('child_process');
const fs = require('fs');
const os = require('os');
const BROWSER = process.env.BROWSER || 'chromium';
const engine = playwright[BROWSER];

const CHANNEL = process.env.CHANNEL || 'cache';
const URL = process.env.URL || 'http://localhost:8011';
const SITE = process.env.SITE || 'https://www.nu.nl/';
const N_WIN = parseInt(process.env.N_WIN || '5', 10);     // idle/nunl windows per class
const CSV = `/tmp/frost-${CHANNEL}.csv`;
const MARKS = `/tmp/frost-${CHANNEL}-marks.json`;
const RES = `/tmp/frost-${CHANNEL}-resources.json`;
const DDFILE = '/tmp/frost_ddload';
const NCPU = os.cpus().length;
const sleep = (ms) => new Promise(r => setTimeout(r, ms));

// --- resource monitor: detect CPU overload / memory leaks in OUR recorder, which
// would both contaminate the side channel and crash long runs. Samples 1 Hz:
//   load1/NCPU (CPU saturation), total RSS of browser+node procs (memory growth),
//   and live browser-tab count (tab leak).
function browserRssMB() {
  try {
    const out = execSync(
      "ps -o rss= -C chrome -C chrome_crashpad_handler -C headless_shell -C firefox -C plugin-container -C node 2>/dev/null | awk '{s+=$1} END{print s}'",
      { encoding: 'utf8' });
    return Math.round((parseInt(out.trim() || '0', 10)) / 1024);
  } catch (e) { return 0; }
}
function startResMon(ctxRef) {
  const samples = [];
  const id = setInterval(() => {
    samples.push({ t: Date.now(), load: +os.loadavg()[0].toFixed(2),
                   rssMB: browserRssMB(), pages: ctxRef.val ? ctxRef.val.pages().length : 0 });
  }, 1000);
  return { samples, stop: () => clearInterval(id) };
}

const BURN_HTML = `<!doctype html><meta charset=utf-8><script>
const N=6*1024*1024,a=new Float64Array(N);for(let i=0;i<N;i++)a[i]=i*2.5;
let s=0,idx=12345;function burn(){for(let k=0;k<3000000;k++){idx=(idx*1103515245+12345)>>>0;s+=a[idx%N];}if(s===-1)document.title=s;setTimeout(burn,0);}burn();
</script>`;

(async () => {
  const browser = await engine.launch({ headless: true });
  const ctx = await browser.newContext({ acceptDownloads: true });
  const ctxRef = { val: ctx };
  const res = startResMon(ctxRef);           // 1 Hz CPU/RSS/tab sampling for the whole run
  const page = await ctx.newPage();
  await page.goto(URL, { waitUntil: 'load' });
  await sleep(400);
  console.log(`channel=${CHANNEL}  banner=`, (await page.locator('#banner').textContent()).trim().split('|')[3]?.trim());

  const log = () => page.locator('#log').textContent();
  const now = () => page.evaluate(() => performance.now());

  // ---- build the chosen channel ----
  await page.click('#reset'); await sleep(400);
  await page.selectOption('#channel', CHANNEL);
  if (CHANNEL === 'read') { await page.fill('#stepGB', '1'); await page.fill('#maxGB', '2'); await page.fill('#threshUs', '25'); }
  await page.click('#build');
  await page.waitForFunction(() => document.getElementById('log').textContent.includes('monitoring'), { timeout: 180000 });
  const tune = (await log()).split('\n').find(l => l.includes('auto-sized to') || l.includes('ready') || l.includes('KNEE') || l.includes('cap with reads'));
  console.log('build:', (tune || '').trim().slice(0, 120));
  const t0 = await now();

  const ev = [];
  const mark = async (name) => { const t = (await now()) - t0; ev.push({ name, t_ms: t }); return t; };

  // ===== PHASE 1: contention =====
  const loadKind = process.env.LOAD || ((CHANNEL === 'cache') ? 'burn' : 'dd');
  await mark('c_idle1'); console.log('[contention] idle 5s'); await sleep(5000);
  await mark('c_load'); console.log(`[contention] LOAD via ${loadKind} 7s`);
  let burners = [], dd = null;
  if (loadKind === 'dd') {
    // oflag=direct bypasses the page cache so every block hits the SSD immediately.
    // detached:true => dd is its own process-group leader so we can kill the WHOLE group
    // (killing the shell child alone would orphan dd, which would keep writing 20s into the
    // recover-idle/fingerprint windows and pollute the baseline). `timeout 9` is a backstop,
    // and count is bounded so a missed kill can't run away.
    dd = spawn('bash', ['-c', `exec timeout 9 dd if=/dev/zero of=${DDFILE} bs=1M count=9000 oflag=direct 2>/dev/null`],
               { detached: true, stdio: 'ignore' });
  } else {
    for (let i = 0; i < 3; i++) { const b = await ctx.newPage(); await b.setContent(BURN_HTML, { waitUntil: 'domcontentloaded' }); burners.push(b); await sleep(200); }
  }
  await sleep(7000);
  await mark('c_load_end');
  if (loadKind === 'dd') {
    try { process.kill(-dd.pid, 'SIGKILL'); } catch (e) {}   // kill the whole process group
    try { execSync(`pkill -9 -f 'dd if=/dev/zero of=${DDFILE}' 2>/dev/null`); } catch (e) {}  // belt-and-suspenders
    try { execSync(`rm -f ${DDFILE}`); } catch (e) {}
  } else {
    for (const b of burners) { try { await b.close(); } catch (e) {} }
  }
  await mark('c_idle2'); console.log('[contention] idle recover 5s'); await sleep(5000);
  await mark('c_end');

  // ===== PHASE 2: fingerprint (alternating idle / nu.nl windows) =====
  console.log(`[fingerprint] collecting ${N_WIN} idle + ${N_WIN} nu.nl windows`);
  for (let i = 1; i <= N_WIN; i++) {
    await mark(`idle_${i}_s`); await sleep(4000); await mark(`idle_${i}_e`);
    await sleep(800);
    await mark(`nunl_${i}_s`);
    const tab = await ctx.newPage();
    if (process.env.HEAVY) {
      // Heavier, more realistic load: wait for full load (subresources/ads) and scroll to
      // force rendering + lazy content + HTTP-cache disk writes - more CPU/disk than DCL.
      try { await tab.goto(SITE, { waitUntil: 'load', timeout: 30000 }); } catch (e) {}
      for (let s = 0; s < 6; s++) { try { await tab.mouse.wheel(0, 2000); } catch (e) {} await sleep(500); }
    } else {
      try { await tab.goto(SITE, { waitUntil: 'domcontentloaded', timeout: 25000 }); } catch (e) {}
      await sleep(3500);
    }
    try { await tab.close(); } catch (e) {}
    await mark(`nunl_${i}_e`);
    await sleep(800);
    console.log(`  window ${i}/${N_WIN} done`);
  }
  await mark('end');

  // --- leak / overload checks before teardown ---
  const openPages = ctx.pages().length;        // should be 1 (the FROST page); >1 = tab leak
  let strayDd = '';
  // -x dd matches only processes literally named "dd" (not the shell/pgrep running this check).
  try { strayDd = execSync(`pgrep -x dd 2>/dev/null || true`, { encoding: 'utf8' }).trim(); } catch (e) {}

  const [dl] = await Promise.all([page.waitForEvent('download', { timeout: 10000 }), page.click('#csv')]);
  await dl.saveAs(CSV);
  fs.writeFileSync(MARKS, JSON.stringify(ev, null, 2));
  console.log('saved', CSV, 'and', MARKS, `(${ev.length} marks)`);

  // --- resource report ---
  res.stop();
  const S = res.samples;
  fs.writeFileSync(RES, JSON.stringify(S, null, 2));
  if (S.length >= 3) {
    const loads = S.map(s => s.load), rss = S.map(s => s.rssMB), pgs = S.map(s => s.pages);
    const peakLoad = Math.max(...loads), peakUtil = peakLoad / NCPU;
    const rss0 = Math.round(rss.slice(0, 3).reduce((a, b) => a + b, 0) / 3);
    const rssEnd = Math.round(rss.slice(-3).reduce((a, b) => a + b, 0) / 3);
    const rssPeak = Math.max(...rss), maxPages = Math.max(...pgs);
    const grew = rssEnd > rss0 * 1.3 && rssEnd > rss0 + 200;   // sustained >30% & >200MB growth
    const overloaded = peakUtil > 0.92;                        // a core pegged ~the whole machine
    const tabLeak = openPages > 1 || maxPages > 4;             // >1 left open, or >4 ever (idle+3 burn = 4 ok)
    console.log(`RESOURCES (${NCPU} cores): peak load ${peakLoad} (util ${(peakUtil*100).toFixed(0)}%/all-cores)`);
    console.log(`          browser+node RSS: ${rss0} -> ${rssEnd} MB (peak ${rssPeak})  ${grew ? '** GROWTH (possible leak)' : 'stable'}`);
    console.log(`          tabs: max ${maxPages} open during run, ${openPages} left open at end  ${tabLeak ? '** TAB LEAK' : 'ok'}`);
    console.log(`          stray dd after load: ${strayDd ? '** ' + strayDd : 'none (clean)'}`);
    console.log(`          verdict: ${(!grew && !overloaded && !tabLeak && !strayDd) ? 'OK - no leaks/overload' : 'CHECK FLAGS ABOVE'}`);
  }
  await browser.close();
})().catch(e => { console.log('SCRIPT ERROR:', e.message); process.exit(1); });
