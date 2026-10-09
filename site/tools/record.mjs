// Records the launch video: deterministic Talus scenes from /video/ (frame by frame), a real-time screencast
// of the actual generator, and an end card; then encodes H.264 MP4 with ffmpeg.
// usage: node build.mjs --video && node serve.mjs   (in another terminal)
//        node tools/record.mjs --out ../reports/video [--ffmpeg C:/ffmpeg/.../ffmpeg.exe]
import { spawn, spawnSync } from "node:child_process";
import { mkdirSync, rmSync, writeFileSync } from "node:fs";
import { tmpdir } from "node:os";
import { join, resolve } from "node:path";

const arg = (k, d) => { const i = process.argv.indexOf(`--${k}`); return i < 0 ? d : process.argv[i + 1]; };
const site = arg("url", "http://127.0.0.1:4173/"), out = resolve(arg("out", join(tmpdir(), "talus-video")));
const ffmpeg = arg("ffmpeg", "ffmpeg"), FPS = 30, W = 1080, H = 1080;
const browser = arg("browser", "C:/Program Files (x86)/Microsoft/Edge/Application/msedge.exe");
const frames = join(out, "frames"), uiOnly = process.argv.includes("--ui-only");  // --ui-only keeps the scene frames
for (const d of uiOnly ? ["b"] : ["a", "b", "c"]) { rmSync(join(frames, d), { recursive: true, force: true }); mkdirSync(join(frames, d), { recursive: true }); }

const port = 9800 + Math.floor(Math.random() * 100);
const proc = spawn(browser, ["--headless=new", `--remote-debugging-port=${port}`, `--user-data-dir=${join(tmpdir(), `talus-rec-${port}`)}`,
  "--enable-unsafe-webgpu", "--no-first-run", "--disable-extensions", `--window-size=${W},${H}`, "--hide-scrollbars", "about:blank"], { stdio: "ignore" });
const sleep = (ms) => new Promise((r) => setTimeout(r, ms));
let target;
for (let i = 0; i < 50 && !target; i++) { await sleep(200); try { target = (await (await fetch(`http://127.0.0.1:${port}/json/list`)).json()).find((t) => t.type === "page"); } catch {} }
const ws = new WebSocket(target.webSocketDebuggerUrl);
await new Promise((r) => ws.addEventListener("open", r));
let id = 0;
const pending = new Map(), listeners = [];
ws.addEventListener("message", (ev) => {
  const m = JSON.parse(ev.data);
  if (m.id && pending.has(m.id)) { pending.get(m.id)(m); pending.delete(m.id); }
  if (m.method === "Runtime.exceptionThrown") console.log("page exception:", m.params.exceptionDetails.exception?.description);
  for (const l of listeners) l(m);
});
const send = (method, params = {}) => new Promise((r) => { const i = ++id; pending.set(i, r); ws.send(JSON.stringify({ id: i, method, params })); });
const evaluate = async (expression) => {
  const r = await send("Runtime.evaluate", { expression, awaitPromise: true, returnByValue: true });
  if (r.result?.exceptionDetails) throw new Error(r.result.exceptionDetails.exception?.description ?? r.result.exceptionDetails.text);
  return r.result?.result?.value;
};
const waitFor = async (expression, ms = 180000) => { const t0 = Date.now(); while (Date.now() - t0 < ms) { if (await evaluate(expression)) return; await sleep(250); } throw new Error(`timeout: ${expression}`); };
const shot = async (path) => {
  const r = await send("Page.captureScreenshot", { format: "png", clip: { x: 0, y: 0, width: W, height: H, scale: 1 } });
  writeFileSync(path, Buffer.from(r.result.data, "base64"));
};

await send("Runtime.enable"); await send("Page.enable");
await send("Emulation.setDeviceMetricsOverride", { width: W, height: H, deviceScaleFactor: 1, mobile: false });

// 1. deterministic scenes
if (!uiOnly) {
  await send("Page.navigate", { url: new URL("video/", site).href });
  await waitFor("!!window.videoReady");
  console.log("video page ready:", JSON.stringify(await evaluate("window.videoReady")));
  if (process.argv.includes("--preview")) {  // a few stills to check the layout, then stop
    for (const t of [0.2, 2.0, 5.0, 9.5, 14.0, 18.9, 21.5]) { await evaluate(`window.renderFrame(${t})`); await shot(join(out, `preview_${t}.png`)); }
    ws.close(); proc.kill(); process.exit(0);
  }
  const scene = async (dir, t0, seconds) => {
    for (let i = 0; i < seconds * FPS; i++) {
      await evaluate(`window.renderFrame(${t0 + i / FPS})`);
      await shot(join(frames, dir, `${String(i).padStart(5, "0")}.png`));
    }
  };
  await scene("a", 0, 20);
  await scene("c", 20, 4.5);
  console.log("scenes captured");
}

// 2. the real generator, recorded in real time
await send("Page.navigate", { url: new URL("?type=mountains&seed=731&steps=12", site).href });
await waitFor("!!window.talusReady && !!document.querySelector('#go')");
await sleep(800);
await evaluate("document.querySelector('#go').click()");                     // warm-up: loads the model once
await waitFor("/Done/.test(document.querySelector('#status').textContent)");
await evaluate(`(() => {
  document.querySelector('#steps input[value="50"]').click();
  const relief = document.querySelector('#p-relief');
  relief.value = relief.min * 0.15 + relief.max * 0.85;
  relief.dispatchEvent(new Event('input', { bubbles: true }));
  const g = document.querySelector('#generate');
  scrollTo(0, g.getBoundingClientRect().top + scrollY - 8);
})()`);
await sleep(1200);
const cast = [];
listeners.push((m) => {
  if (m.method !== "Page.screencastFrame") return;
  cast.push({ t: m.params.metadata.timestamp, data: m.params.data });
  send("Page.screencastFrameAck", { sessionId: m.params.sessionId });
});
await send("Page.startScreencast", { format: "jpeg", quality: 95, maxWidth: W, maxHeight: H, everyNthFrame: 1 });
await sleep(700);
await evaluate("document.querySelector('#go').click()");
await waitFor("/Step/.test(document.querySelector('#status').textContent)");  // the warm-up left "Done" behind
await waitFor("/Done/.test(document.querySelector('#status').textContent)");
await sleep(1600);
await send("Page.stopScreencast");
console.log("status:", await evaluate("document.querySelector('#status').textContent"), `${cast.length} screencast frames`);
const t0 = cast[0].t, dur = cast[cast.length - 1].t - t0;
for (let i = 0, j = 0; i < Math.floor(dur * FPS); i++) {
  while (j + 1 < cast.length && cast[j + 1].t - t0 <= i / FPS) j++;
  writeFileSync(join(frames, "b", `${String(i).padStart(5, "0")}.jpg`), Buffer.from(cast[j].data, "base64"));
}
ws.close();
proc.kill();

// 3. encode: 1080x1080 H.264, yuv420p, faststart (X and Reddit accept it as is)
const mp4 = join(out, "talus-3.mp4");
const enc = spawnSync(ffmpeg, ["-y", "-loglevel", "error",
  "-framerate", String(FPS), "-i", join(frames, "a", "%05d.png"),
  "-framerate", String(FPS), "-i", join(frames, "b", "%05d.jpg"),
  "-framerate", String(FPS), "-i", join(frames, "c", "%05d.png"),
  "-filter_complex", [0, 1, 2].map((k) => `[${k}:v]scale=${W}:${H}:out_range=tv,setsar=1,format=yuv420p[v${k}]`).join(";") + ";[v0][v1][v2]concat=n=3:v=1:a=0[v]",
  "-map", "[v]", "-c:v", "libx264", "-preset", "slow", "-crf", "17", "-pix_fmt", "yuv420p", "-color_range", "tv",
  "-colorspace", "bt709", "-color_primaries", "bt709", "-color_trc", "bt709", "-movflags", "+faststart", mp4], { stdio: "inherit" });
if (enc.status !== 0) throw new Error("ffmpeg failed");
console.log(`wrote ${mp4}`);
