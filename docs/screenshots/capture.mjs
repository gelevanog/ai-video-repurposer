// Screenshots of the review UI with headless Chrome (puppeteer-core), from projects the evaluation run processed.
//
//   PUPPETEER_FROM=<dir with node_modules/puppeteer-core> PROJECT=<id> CLIP=<clip id> SEEK=<seconds into the clip> \
//     OUTPUTS_PROJECT=<id> node docs/screenshots/capture.mjs http://localhost:3000 [hero|outputs|transcript|projects|evaluation]
import { createRequire } from "module";

const require = createRequire(process.env.PUPPETEER_FROM || import.meta.url);
const puppeteer = require("puppeteer-core");
const [base, ...shots] = process.argv.slice(2);
const out = process.env.OUT_DIR || new URL(".", import.meta.url).pathname;
const sleep = (ms) => new Promise((r) => setTimeout(r, ms));

const browser = await puppeteer.launch({
  executablePath: process.env.CHROME || "/usr/bin/google-chrome",
  headless: "new",
  args: ["--no-sandbox", "--autoplay-policy=no-user-gesture-required"],
});

async function open(url, { width = 1600, height = 1000 } = {}) {
  const page = await browser.newPage();
  await page.setViewport({ width, height, deviceScaleFactor: 1 });
  page.on("console", (msg) => msg.type() === "error" && console.log("page error:", msg.text()));
  await page.goto(url, { waitUntil: "networkidle2", timeout: 120000 });
  await sleep(1200);
  return page;
}

async function seek(page, seconds) {
  // the live preview follows the <video> element's currentTime
  await page.evaluate(async (s) => {
    const video = [...document.querySelectorAll("video")].find((v) => v.src.includes("/media"));
    if (!video) return;
    await new Promise((resolve) => {
      if (video.readyState >= 2) return resolve();
      video.addEventListener("loadeddata", resolve, { once: true });
    });
    video.currentTime = video.currentTime + s;
    await new Promise((resolve) => video.addEventListener("seeked", resolve, { once: true }));
  }, seconds);
  await sleep(900);
}

for (const shot of shots) {
  if (shot === "hero") {
    const page = await open(`${base}/p/${process.env.PROJECT}?clip=${process.env.CLIP}`, { height: 1000 });
    await seek(page, Number(process.env.SEEK || 6));
    await page.screenshot({ path: `${out}hero.png` });
  } else if (shot === "outputs") {
    const page = await open(`${base}/p/${process.env.OUTPUTS_PROJECT || process.env.PROJECT}?tab=outputs`, { height: 1250 });
    await page.screenshot({ path: `${out}outputs.png` });
  } else if (shot === "transcript") {
    const page = await open(`${base}/p/${process.env.TRANSCRIPT_PROJECT || process.env.PROJECT}?tab=transcript`, { height: 950 });
    await page.screenshot({ path: `${out}transcript.png` });
  } else if (shot === "projects") {
    const page = await open(`${base}/`, { height: 900 });
    await page.screenshot({ path: `${out}projects.png` });
  } else if (shot === "evaluation") {
    const page = await open(`${base}/evaluation`, { height: 900 });
    await page.screenshot({ path: `${out}evaluation.png` });
  }
  console.log(`${shot}.png`);
}
await browser.close();
