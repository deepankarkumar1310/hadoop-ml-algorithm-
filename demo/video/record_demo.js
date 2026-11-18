// Records the 1-minute demo walkthrough. Usage: node demo/video/record_demo.js <out_dir> file:///<path>/demo/index.html  (needs Playwright), then convert the .webm with ffmpeg to MP4.
const { chromium } = require('playwright');
(async () => {
  const out = process.argv[2], url = process.argv[3];
  const b = await chromium.launch();
  const ctx = await b.newContext({ viewport: {width:1280,height:720}, deviceScaleFactor: 1,
                                   recordVideo: { dir: out, size: {width:1280,height:720} } });
  const pg = await ctx.newPage();
  const w = ms => pg.waitForTimeout(ms);
  const glide = async (sel, ms=1200) => { await pg.evaluate(([s])=>{const e=document.querySelector(s);
      window.scrollTo({top:e.getBoundingClientRect().top+scrollY-60,behavior:'smooth'})},[sel]); await w(ms); };
  const scrollBy = async (dy, ms=900) => { await pg.evaluate(d=>window.scrollBy({top:d,behavior:'smooth'}),dy); await w(ms); };
  await pg.goto(url); await w(4500);                                 // 0-5 s: title, question, headline numbers
  await glide('#data', 3500);                                        // 5-9: data sources
  await glide('#pipeline', 1500);                                    // 9-11
  await pg.click('#replay'); await w(7000);                          // 11-18: replay the Hadoop run
  await scrollBy(420, 2500);                                         // 18-21: runtime + HDFS charts
  await glide('#prep', 3500);                                        // 21-25: preprocessing
  await scrollBy(380, 2000);                                         // 25-27
  await glide('#model', 1200);                                       // 27-28: explorer on Kishanganj
  await w(2500);
  await pg.click('#q'); await pg.keyboard.type('chennai', {delay:120}); await w(2500);   // 28-33: search a district
  await scrollBy(330, 2500);                                         // 33-36: attention + inputs
  await glide('#results', 3000);                                     // 36-39: comparison table
  await scrollBy(380, 1500);                                         // 39-41
  await pg.selectOption('#cmSel','5'); await w(1800);                // 41-43: confusion matrix LR
  await pg.selectOption('#cmSel','0'); await w(1500);                // 43-44: back to B-ACPN
  await glide('#map', 1500);                                         // 44-46
  await scrollBy(120, 800);
  const box = await pg.locator('#mapsvg').boundingBox();
  const pts = [[.30,.22],[.55,.36],[.60,.40]];
  for (const [fx,fy] of pts) { await pg.mouse.move(box.x+box.width*fx, box.y+box.height*fy, {steps:15}); await w(1100); }   // 47-50: hover the map
  const pb = await pg.locator('#mapsvg path.d[data-c="211"]').boundingBox();                  // Purnia, Bihar
  await pg.mouse.move(pb.x+pb.width/2, pb.y+pb.height/2, {steps:20}); await w(1500);
  await pg.mouse.click(pb.x+pb.width/2, pb.y+pb.height/2); await w(3600);                    // 52-56: click opens Purnia
  await glide('header.hero', 3500);                                  // 56-60: back to the top
  await ctx.close(); await b.close();
})();
