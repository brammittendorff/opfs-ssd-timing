// Multi-class FROST website fingerprinting collector.
//
// Builds one timing channel then collects N_PER windows for an 'idle' class
// and N_PER windows for each site in SITES, with windows interleaved
// round-robin across all classes so session drift doesn't align with one label.
//
// Env:
//   CHANNEL   read | cache | flush   (default: cache)
//   BROWSER   chromium | firefox     (default: chromium)
//   N_PER     windows per class      (default: 5)
//   SITES     comma-separated URLs   (default: 5 preset sites)
//
// Outputs:
//   /tmp/frost-sites-<channel>.csv        (raw trace, timestamp_ms,latency_us)
//   /tmp/frost-sites-<channel>-marks.json (marks array + "_labels" metadata)

const playwright = require('playwright');
const fs = require('fs');

const BROWSER  = process.env.BROWSER  || 'chromium';
const CHANNEL  = process.env.CHANNEL  || 'cache';
const N_PER    = parseInt(process.env.N_PER  || '5', 10);
const URL_BASE = process.env.URL      || 'http://localhost:8011';

const DEFAULT_SITES = [
  'https://en.wikipedia.org/wiki/Cat',
  'https://github.com/',
  'https://www.bbc.com/',
  'https://www.wired.com/',
  'https://www.nytimes.com/',
];
const SITES = process.env.SITES
  ? process.env.SITES.split(',').map(s => s.trim()).filter(Boolean)
  : DEFAULT_SITES;

const CSV   = `/tmp/frost-sites-${CHANNEL}.csv`;
const MARKS = `/tmp/frost-sites-${CHANNEL}-marks.json`;

const sleep = (ms) => new Promise(r => setTimeout(r, ms));

// Sanitize a URL into a safe mark-name prefix: keep alphanum, replace rest with '_'.
function urlToKey(url, idx) {
  return `cls${idx}`;  // use numeric index; URL stored in _labels metadata
}

(async () => {
  const engine = playwright[BROWSER];
  if (!engine) { console.error(`Unknown browser: ${BROWSER}`); process.exit(1); }

  // --- classes: index 0 = idle, indices 1..K = sites ---
  const classes = ['idle', ...SITES];  // length K+1
  const K = classes.length;
  console.log(`channel=${CHANNEL}  browser=${BROWSER}  N_PER=${N_PER}`);
  console.log(`classes (${K}): idle + ${SITES.length} sites`);
  SITES.forEach((s, i) => console.log(`  cls${i+1} = ${s}`));

  const browser = await engine.launch({ headless: true });
  const ctx     = await browser.newContext({ acceptDownloads: true });
  const page    = await ctx.newPage();

  await page.goto(URL_BASE, { waitUntil: 'load' });
  await sleep(400);
  console.log('banner:', (await page.locator('#banner').textContent()).trim().split('|')[3]?.trim());

  const logText = () => page.locator('#log').textContent();
  const now     = () => page.evaluate(() => performance.now());

  // ---- build the chosen channel ----
  await page.click('#reset');
  await sleep(400);
  await page.selectOption('#channel', CHANNEL);
  if (CHANNEL === 'read') {
    await page.fill('#stepGB', '1');
    await page.fill('#maxGB', '2');
    await page.fill('#threshUs', '25');
  }
  await page.click('#build');
  await page.waitForFunction(
    () => document.getElementById('log').textContent.includes('monitoring'),
    { timeout: 180000 }
  );
  const tune = (await logText()).split('\n').find(
    l => l.includes('auto-sized') || l.includes('ready') || l.includes('KNEE') || l.includes('cap with reads')
  );
  console.log('build:', (tune || '').trim().slice(0, 120));

  const t0 = await now();
  const ev = [];
  const mark = async (name) => {
    const t = (await now()) - t0;
    ev.push({ name, t_ms: t });
    return t;
  };

  // ---- collect N_PER rounds, round-robin across all classes ----
  // Round r visits class (r % K): class 0 = idle, class k = SITES[k-1].
  // With N_PER rounds per class we need N_PER * K total rounds,
  // but we interleave so we collect all classes evenly in each "lap" of K.
  // Schedule: lap 0 -> [cls0,cls1,...,clsK-1], lap 1 -> [cls0,...], ...
  // We shuffle the order within each lap slightly to break any remaining regularity,
  // but keep class 0 (idle) deterministic here for simplicity and reproducibility.

  const perClass = Array.from({ length: K }, () => 0); // collected count per class
  const total = N_PER * K;

  console.log(`\n[collect] ${total} windows total (${N_PER} per class, ${K} classes), round-robin`);

  for (let lap = 0; lap < N_PER; lap++) {
    // Within each lap visit each class in order 0..K-1.
    for (let cls = 0; cls < K; cls++) {
      const i = perClass[cls] + 1;  // 1-based window index within this class
      const markBase = `cls${cls}_${i}`;

      if (cls === 0) {
        // --- idle window: just wait ---
        await mark(`${markBase}_s`);
        await sleep(4000);
        await mark(`${markBase}_e`);
      } else {
        // --- site window ---
        const site = SITES[cls - 1];
        await mark(`${markBase}_s`);
        const tab = await ctx.newPage();
        try {
          await tab.goto(site, { waitUntil: 'load', timeout: 30000 });
        } catch (e) {
          console.warn(`  [warn] goto ${site} failed: ${e.message.slice(0, 80)}`);
        }
        // Scroll to trigger lazy content / disk writes (FROST-style "heavy" load).
        for (let s = 0; s < 4; s++) {
          try { await tab.mouse.wheel(0, 2000); } catch (_) {}
          await sleep(400);
        }
        try { await tab.close(); } catch (_) {}
        await mark(`${markBase}_e`);
      }

      perClass[cls]++;
      console.log(`  lap ${lap+1}/${N_PER}  cls${cls}  window ${i}/${N_PER} done`);

      // Brief inter-window gap so channel can settle.
      await sleep(600);
    }
  }

  await mark('end');

  // ---- export CSV via UI download ----
  const [dl] = await Promise.all([
    page.waitForEvent('download', { timeout: 10000 }),
    page.click('#csv'),
  ]);
  await dl.saveAs(CSV);

  // ---- write marks JSON (includes _labels metadata) ----
  const out = [
    ...ev,
    {
      name: '_labels',
      // map: cls index (as string) -> URL string, cls0 = 'idle'
      labels: Object.fromEntries(
        classes.map((c, i) => [String(i), c])
      ),
    },
  ];
  fs.writeFileSync(MARKS, JSON.stringify(out, null, 2));

  console.log(`\nsaved ${CSV}  (${ev.length} marks + _labels)`);
  console.log(`saved ${MARKS}`);

  await browser.close();
})().catch(e => {
  console.error('SCRIPT ERROR:', e.message);
  process.exit(1);
});
