// Unit test for the read-channel knee-detection DECISION (mirrors worker.js kneeProbe:
// constants MISS_FRAC=0.30, MIN_SLOW_US=20, and the same formula). It drives the logic
// through every device class with synthetic latency distributions - including the
// cached-file-grows-past-RAM path that a non-page-cached test VM cannot reproduce in a
// real browser. Keep these constants in sync with worker.js.
//   run: node tests/knee_logic_test.js
const MISS_FRAC = 0.30, MIN_SLOW_US = 20;
const pctl = (a, p) => { const s = [...a].sort((x, y) => x - y); return s[Math.min(s.length - 1, Math.floor(p * s.length))]; };

function decideKnee(baseLat, fullLat, threshUs = 25) {
  const baseMed = pctl(baseLat, 0.5);
  const baseP95 = pctl(baseLat, 0.95);
  const slowUs = Math.max(baseP95 * 3, MIN_SLOW_US);     // a read this slow = an SSD miss
  const miss = fullLat.filter(v => v > slowUs).length / fullLat.length;
  const fullMed = pctl(fullLat, 0.5);
  const uncached = baseMed >= MIN_SLOW_US;                // typical tiny-set read is SSD-slow (median, not p95)
  return { past: uncached || miss >= MISS_FRAC || fullMed >= threshUs, miss, uncached, fullMed };
}

// deterministic-ish synthetic latencies (us)
const rep = (n, f) => Array.from({ length: n }, (_, i) => f(i));
const cachedLat = () => rep(3000, i => (i % 2 ? 0 : 5));                 // cache hits: 0-5us
const mixedLat  = (missRate) => rep(3000, i => (i / 3000) < missRate ? 70 + (i % 60) : 5); // SSD misses 70-130us
const uncached  = () => rep(3000, i => 60 + (i % 40));                  // always ~60-100us
// cached baseline with a few transient slow reads (noise) - p95 would falsely read "uncached"
const noisyCachedLat = () => rep(3000, i => (i < 150 ? 90 : (i % 2 ? 0 : 5)));   // 5% slow, median still 5us

const cases = [
  { name: "cached, file < RAM (fits in cache)",        base: cachedLat(), full: cachedLat(),    expect: false },
  { name: "cached, file just under the knee (10% miss)", base: cachedLat(), full: mixedLat(0.10), expect: false },
  { name: "cached, file PAST the knee (>RAM, 50% miss)", base: cachedLat(), full: mixedLat(0.50), expect: true },
  { name: "cached, WARM-RESTART past knee (40% miss)*",  base: cachedLat(), full: mixedLat(0.40), expect: true },
  { name: "uncached OPFS (reads always SSD-slow)",       base: uncached(),  full: uncached(),     expect: true },
  { name: "noisy cached baseline, file < RAM (no false uncached)", base: noisyCachedLat(), full: cachedLat(), expect: false },
];

let ok = 0;
console.log("read-channel knee decision (MISS_FRAC=30%, MIN_SLOW_US=20):\n");
for (const c of cases) {
  const d = decideKnee(c.base, c.full);
  const pass = d.past === c.expect;
  ok += pass ? 1 : 0;
  console.log(`  ${pass ? "PASS" : "FAIL"}  ${c.name}`);
  console.log(`        -> past=${d.past}  (miss ${(d.miss * 100).toFixed(0)}%, median ${d.fullMed}us, uncached=${d.uncached})  expected past=${c.expect}`);
}
console.log(`\n  * the exact bug case: a warm cache makes the median look fast (5us), but the`);
console.log(`    miss-fraction over the whole >RAM file stays high -> correctly 'past knee',`);
console.log(`    so a rebuild does NOT re-grow the file (the 12->21->24 GB bug).`);
console.log(`\n${ok}/${cases.length} cases passed`);
process.exit(ok === cases.length ? 0 : 1);
