// Records the 2-minute walkthrough (dashboard, then the demo website) with on-screen captions, no audio.
// Usage: node demo/video/record_full_demo.js <out_dir> <repo_dir>   then join the two .webm files with ffmpeg.
const { chromium } = require('playwright');
(async () => {
  const [out, repo] = [process.argv[2], process.argv[3]];
  const b = await chromium.launch();
  const VP = { width: 1280, height: 720 };

  const caption = async (pg, text) => pg.evaluate(t => {
    let c = document.getElementById('__cap');
    if (!c) { c = document.createElement('div'); c.id = '__cap';
      c.style.cssText = 'position:fixed;left:50%;bottom:18px;transform:translateX(-50%);z-index:99999;max-width:86%;' +
        'background:rgba(15,23,30,.86);color:#fff;font:600 17px/1.35 system-ui,Segoe UI,Arial,sans-serif;' +
        'padding:9px 16px;border-radius:8px;box-shadow:0 4px 14px rgba(0,0,0,.25);text-align:center;transition:opacity .25s';
      document.body.appendChild(c); }
    c.textContent = t; }, text);
  const smoothTo = async (pg, sel, off = 60, ms = 1200) => {
    await pg.evaluate(([s, o]) => { const e = document.querySelector(s);
      window.scrollTo({ top: e.getBoundingClientRect().top + scrollY - o, behavior: 'smooth' }); }, [sel, off]);
    await pg.waitForTimeout(ms); };
  const scrollBy = async (pg, dy, ms = 1000) => { await pg.evaluate(d => window.scrollBy({ top: d, behavior: 'smooth' }), dy); await pg.waitForTimeout(ms); };
  const typeIn = async (pg, sel, text) => { await pg.click(sel, { clickCount: 3 }); await pg.keyboard.press('Backspace');
    await pg.keyboard.type(text, { delay: 110 }); };

  // ---------------- part 1: dashboard (about 40 s)
  let ctx = await b.newContext({ viewport: VP, recordVideo: { dir: out + '/a', size: VP } });
  let pg = await ctx.newPage(); let w = ms => pg.waitForTimeout(ms);
  await pg.goto('file://' + repo + '/dashboard/index.html'); await w(600);
  await caption(pg, 'Part 1 · Climate Push Explorer dashboard (B-ACPN results, works offline)'); await w(4000);
  await caption(pg, 'Headline numbers: 469 districts · ROC-AUC 0.82 · accuracy 0.738 · F1 0.728 · 1.08 GB in HDFS'); await w(4200);
  await caption(pg, 'Search any district: Patna (Bihar)'); await typeIn(pg, '#q', 'patna'); await w(2600);
  await caption(pg, 'Its group, vulnerability, baseline, climate push, uncertainty and main climate factor'); await w(3600);
  await caption(pg, 'Attention bars: which climate variable drives this district\'s push'); await w(3000);
  await caption(pg, 'Shimla: negative push, so climate protects it'); await typeIn(pg, '#q', 'shimla'); await w(4200);
  await caption(pg, 'Kodagu (Karnataka): developed, yet climate adds risk'); await typeIn(pg, '#q', 'kodagu'); await w(4000);
  await typeIn(pg, '#q', ''); await w(300);
  await caption(pg, 'Most climate-driven districts: largest positive climate push'); await scrollBy(pg, 260, 4200);
  await caption(pg, 'Mean climate push by group: anomalies A and B behave as climate predicts'); await scrollBy(pg, 300, 4200);
  await ctx.close();

  // ---------------- part 2: demo website (about 80 s)
  ctx = await b.newContext({ viewport: VP, recordVideo: { dir: out + '/b', size: VP } });
  pg = await ctx.newPage(); w = ms => pg.waitForTimeout(ms);
  await pg.goto('file://' + repo + '/demo/index.html'); await w(500);
  await caption(pg, 'Part 2 · Project demo website: research question and headline results'); await w(4200);
  await smoothTo(pg, '#data'); await caption(pg, 'Step 1 · The five data sources (the vulnerability index is only the target)'); await w(3800);
  await smoothTo(pg, '#pipeline'); await caption(pg, 'Step 2 · Hadoop pipeline: replay of the recorded Colab run'); await w(1200);
  await pg.click('#replay'); await w(6800);
  await caption(pg, 'MapReduce runtime (1, 10, 30 years) and data stored in each HDFS zone'); await scrollBy(pg, 430, 3600);
  await smoothTo(pg, '#prep'); await caption(pg, 'Step 3 · Preprocessing: 640 → 469 districts, no imputation'); await w(3200);
  await caption(pg, 'Area weighting vs nearest grid cell: RMSE 287 mm, MAPE 11.1%'); await scrollBy(pg, 400, 3200);
  await smoothTo(pg, '#model'); await caption(pg, 'Step 4 · B-ACPN explorer: Kishanganj, the most climate-driven district'); await w(3600);
  await caption(pg, 'Search a district: Chennai'); await pg.click('#q'); await pg.keyboard.type('chennai', { delay: 110 }); await w(2600);
  await caption(pg, 'Baseline + climate push → probability, attention bars and all 13 model inputs'); await scrollBy(pg, 340, 3600);
  await smoothTo(pg, '#results'); await caption(pg, 'Step 5 · Six models on the same 25 folds: ROC-AUC, CI, accuracy, F1, LOSO, p-value'); await w(4000);
  await scrollBy(pg, 400, 1000);
  await caption(pg, 'Confusion matrix for each model (25 test folds summed)');
  for (const v of ['1', '2', '3', '4', '5', '0']) { await pg.selectOption('#cmSel', v); await w(900); }
  await smoothTo(pg, '#map', 60, 1000); await caption(pg, 'Step 6 · Climate push of every district: red adds risk, blue protects'); await scrollBy(pg, 110, 900);
  const box = await pg.locator('#mapsvg').boundingBox();
  for (const [fx, fy] of [[.30, .22], [.18, .45], [.55, .36], [.60, .40]]) {
    await pg.mouse.move(box.x + box.width * fx, box.y + box.height * fy, { steps: 14 }); await w(1100); }
  const pb = await pg.locator('#mapsvg path.d[data-c="211"]').boundingBox();
  await caption(pg, 'Click a district on the map to open it in the explorer: Purnia (Bihar)');
  await pg.mouse.move(pb.x + pb.width / 2, pb.y + pb.height / 2, { steps: 16 }); await w(1300);
  await pg.mouse.click(pb.x + pb.width / 2, pb.y + pb.height / 2); await w(3600);
  await smoothTo(pg, '#map', 60, 900); await scrollBy(pg, 700, 1500);
  await caption(pg, 'Group means: anomaly A +0.09, anomaly B −0.17 (both p < 0.001)'); await w(3000);
  await pg.evaluate(() => window.scrollTo({ top: 0, behavior: 'smooth' })); await caption(pg, 'Beyond Development · B-ACPN · SRM IST'); await w(3200);
  await ctx.close(); await b.close();
})();
