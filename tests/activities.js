// Advanced, REAL-activity fingerprinting harness. Instead of a synthetic dd hammer, it
// captures labeled windows of representative activities and lets eval_activities.py do a
// proper multi-class, leakage-robust classification.
//
// Activities (real processes / real browser tabs):
//   idle     - nothing
//   web      - load a real website in a tab (CPU + memory + some disk)
//   cp       - copy a 5 GB file on the same disk (buffered disk writeback)
//   compute  - a CPU+memory-bandwidth process (random walk over 64 MB), an "app crunching"
//
// Usage (via the playwright-skill run.js):
//   CHANNEL=cache N_PER=8 node run.js tests/activities.js
// Outputs: /tmp/frost-act-<CHANNEL>.csv + /tmp/frost-act-<CHANNEL>-marks.json
const playwright = require('playwright');
const { execSync, spawn } = require('child_process');
const fs = require('fs');
const os = require('os');

const CHANNEL = process.env.CHANNEL || 'cache';
const BROWSER = process.env.BROWSER || 'chromium';
const N_PER   = parseInt(process.env.N_PER || '8', 10);
const WIN     = parseInt(process.env.WIN || '8', 10) * 1000;
const SITE    = process.env.SITE || 'https://www.wired.com/';
const URL     = process.env.URL || 'http://localhost:8000';
const CSV     = `/tmp/frost-act-${CHANNEL}.csv`;
const MARKS   = `/tmp/frost-act-${CHANNEL}-marks.json`;
const CPSRC = '/tmp/frost_act_src.bin', CPDST = '/tmp/frost_act_dst.bin';
const sleep = (ms) => new Promise(r => setTimeout(r, ms));

// a CPU + memory-bandwidth load that runs for ~`secs` seconds (random walk over 64 MB)
const computeCmd = (secs) =>
  `python3 -c "import time;n=64*1024*1024;a=bytearray(n);e=time.time()+${secs};i=0;s=0
while time.time()<e:
 for _ in range(2000000):
  i=(i*1103515245+12345)%n;s+=a[i]"`;

(async () => {
  // one-time 5 GB source for the copy activity
  execSync(`dd if=/dev/zero of=${CPSRC} bs=1M count=5000 status=none`);
  const browser = await playwright[BROWSER].launch({ headless: true });
  const ctx = await browser.newContext();
  const page = await ctx.newPage();
  await page.goto(URL, { waitUntil: 'load' }); await sleep(400);
  const now = () => page.evaluate(() => performance.now());

  await page.selectOption('#channel', CHANNEL);
  if (CHANNEL === 'read') { await page.fill('#stepGB', '1'); await page.fill('#maxGB', '8'); }
  await page.click('#reset'); await sleep(600);
  await page.click('#build');
  const t0 = await now();
  await page.waitForFunction(() => document.getElementById('log').textContent.includes('monitoring'), { timeout: 180000 });
  console.log(`channel=${CHANNEL} built`);
  const ev = []; const mark = async (n) => ev.push({ name: n, t_ms: (await now()) - t0 });

  async function windowFor(cls, i) {
    await mark(`${cls}_${i}_s`);
    let proc = null, tab = null;
    if (cls === 'web') {
      tab = await ctx.newPage();
      try { await tab.goto(SITE, { waitUntil: 'load', timeout: 20000 }); } catch (e) {}
      for (let s = 0; s < WIN / 1000; s++) { try { await tab.mouse.wheel(0, 1500); } catch (e) {} await sleep(1000); }
    } else if (cls === 'cp') {
      proc = spawn('bash', ['-c', `cp ${CPSRC} ${CPDST} && sync`], { detached: true, stdio: 'ignore' });
      await sleep(WIN);
    } else if (cls === 'compute') {
      proc = spawn('bash', ['-c', computeCmd(WIN / 1000)], { detached: true, stdio: 'ignore' });
      await sleep(WIN);
    } else { // idle
      await sleep(WIN);
    }
    await mark(`${cls}_${i}_e`);
    if (tab) { try { await tab.close(); } catch (e) {} }
    if (proc) { try { process.kill(-proc.pid, 'SIGKILL'); } catch (e) {} }
    try { execSync(`rm -f ${CPDST}`); } catch (e) {}
  }

  const CLASSES = ['idle', 'web', 'cp', 'compute'];
  console.log(`collecting ${N_PER} x {${CLASSES.join(', ')}} (${WIN/1000}s windows)`);
  for (let i = 1; i <= N_PER; i++) {
    for (const cls of CLASSES) { await windowFor(cls, i); await sleep(700); }
    console.log(`  round ${i}/${N_PER} done`);
  }
  await mark('end');

  const [dl] = await Promise.all([page.waitForEvent('download', { timeout: 10000 }), page.click('#csv')]);
  await dl.saveAs(CSV);
  fs.writeFileSync(MARKS, JSON.stringify(ev));
  console.log('saved', CSV, `(${ev.length} marks)`);
  await browser.close();
  try { execSync(`pkill -9 -f frost_act 2>/dev/null; rm -f ${CPSRC} ${CPDST}`); } catch (e) {}
})().catch(e => { console.log('SCRIPT ERROR:', e.message); try { execSync(`rm -f ${CPSRC} ${CPDST}`); } catch (_) {} process.exit(1); });
