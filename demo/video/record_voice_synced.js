// Re-records the 2-minute walkthrough with every action timed to the narration (times are seconds into the voice note).
// Usage: node demo/video/record_voice_synced.js <out_dir> <repo_dir>   -> <out_dir>/a/*.webm (dashboard), <out_dir>/b/*.webm (website)
const { chromium } = require('playwright');
(async () => {
  const [out, repo] = [process.argv[2], process.argv[3]];
  const SPLIT = 41.0;                      // voice note: dashboard part ends here, website part starts
  const b = await chromium.launch();
  const VP = { width: 1280, height: 720 };
  let t0 = 0, base = 0, pg;
  const until = async T => { const ms = (T - base) * 1000 - (Date.now() - t0); if (ms > 0) await pg.waitForTimeout(ms); };
  const caption = t => pg.evaluate(t => {
    let c = document.getElementById('__cap');
    if (!c) { c = document.createElement('div'); c.id = '__cap';
      c.style.cssText = 'position:fixed;left:50%;bottom:18px;transform:translateX(-50%);z-index:99999;max-width:86%;' +
        'background:rgba(15,23,30,.86);color:#fff;font:600 17px/1.35 system-ui,Segoe UI,Arial,sans-serif;' +
        'padding:9px 16px;border-radius:8px;box-shadow:0 4px 14px rgba(0,0,0,.25);text-align:center';
      document.body.appendChild(c); }
    c.textContent = t; }, t);
  const smoothTo = (sel, off = 60) => pg.evaluate(([s, o]) => { const e = document.querySelector(s);
    window.scrollTo({ top: e.getBoundingClientRect().top + scrollY - o, behavior: 'smooth' }); }, [sel, off]);
  const scrollBy = dy => pg.evaluate(d => window.scrollBy({ top: d, behavior: 'smooth' }), dy);
  const typeIn = async (sel, text) => { await pg.click(sel, { clickCount: 3 }); await pg.keyboard.press('Backspace');
    await pg.keyboard.type(text, { delay: 100 }); };

  // ---- part 1: dashboard (voice 0 - 41 s)
  let ctx = await b.newContext({ viewport: VP, recordVideo: { dir: out + '/a', size: VP } });
  t0 = Date.now(); base = 0; pg = await ctx.newPage();
  await pg.goto('file://' + repo + '/dashboard/index.html');
  await until(0.3);  await caption('Climate Push Explorer dashboard (B-ACPN results, works offline)');
  await until(6.2);  await caption('469 districts · ROC-AUC 0.82 · accuracy 0.738 · F1 0.728');
  await until(17.6); await caption('Search any district: Patna (Bihar)'); await typeIn('#q', 'patna');
  await until(20.0); await caption('Group, vulnerability, baseline risk, climate push, uncertainty, main climate factor');
  await until(27.2); await caption('Attention bars: which climate variable drives the push');
  await until(30.6); await caption('Shimla: negative push, climate protects it'); await typeIn('#q', 'shimla');
  await until(32.7); await caption('Kodagu: developed, but climate adds risk'); await typeIn('#q', 'kodagu');
  await until(36.2); await caption('Most climate-driven districts (largest positive push)');
  await until(38.6); await caption('Average climate push for each group');
  await until(SPLIT); await ctx.close();

  // ---- part 2: website (voice 41 s - end)
  ctx = await b.newContext({ viewport: VP, recordVideo: { dir: out + '/b', size: VP } });
  t0 = Date.now(); base = SPLIT; pg = await ctx.newPage();
  await pg.goto('file://' + repo + '/demo/index.html');
  await until(41.4); await caption('Project website: why do some developed districts stay vulnerable?');
  await until(48.7); await smoothTo('#data'); await caption('Five public datasets; the vulnerability index is only the target');
  await until(53.4); await smoothTo('#pipeline'); await caption('Hadoop pipeline: replay of the real run');
  await until(55.2); await pg.click('#replay');
  await until(56.6); await caption('HDFS stores · MapReduce builds yearly indices · Hive 30-yr baseline · Spark climate regimes');
  await until(64.8); await scrollBy(430); await caption('MapReduce run times and data stored in each HDFS zone');
  await until(69.4); await smoothTo('#prep'); await caption('Preprocessing: 469 complete districts, no imputation');
  await until(73.9); await scrollBy(400); await caption('Area weighting avoids a 287 mm rainfall error');
  await until(77.6); await smoothTo('#model'); await caption('Explorer: Kishanganj, the most climate-driven district');
  await until(81.2); await caption('Search: Chennai'); await pg.click('#q'); await pg.keyboard.type('chennai', { delay: 100 });
  await until(83.3); await scrollBy(340); await caption('Baseline, climate push and all thirteen inputs');
  await until(86.6); await smoothTo('#results'); await caption('Six models on the same 25 folds: B-ACPN best, ROC-AUC 0.82');
  await until(92.6); await scrollBy(400); await caption('Confusion matrices of all six models');
  for (const [T, v] of [[93.4, '1'], [93.9, '2'], [94.4, '3'], [94.9, '4'], [95.3, '5'], [95.7, '0']]) { await until(T); await pg.selectOption('#cmSel', v); }
  await until(95.9); await smoothTo('#map', 60); await caption('Climate push of every district: red adds risk, blue protects');
  await until(96.9); await scrollBy(110);
  await until(97.6);
  const box = await pg.locator('#mapsvg').boundingBox();
  for (const [fx, fy] of [[.30, .22], [.18, .45], [.55, .36]]) { await pg.mouse.move(box.x + box.width * fx, box.y + box.height * fy, { steps: 12 }); await pg.waitForTimeout(700); }
  const pb = await pg.locator('#mapsvg path.d[data-c="211"]').boundingBox();
  await until(100.6); await caption('Click Purnia (Bihar) to open it in the explorer');
  await pg.mouse.move(pb.x + pb.width / 2, pb.y + pb.height / 2, { steps: 10 });
  await until(101.7); await pg.mouse.click(pb.x + pb.width / 2, pb.y + pb.height / 2);
  await until(103.5); await smoothTo('#map', 60); await pg.waitForTimeout(700); await scrollBy(700);
  await caption('Developed-but-vulnerable: positive push · resilient poorer districts: negative · both significant');
  await until(110.4); await pg.evaluate(() => window.scrollTo({ top: 0, behavior: 'smooth' })); await caption('Beyond Development · B-ACPN · SRM IST');
  await until(117.5); await ctx.close(); await b.close();
})();
