// Headless browser check over the Chrome DevTools Protocol (no extra packages): loads the page, reports
// console errors, optionally runs one generation, and saves screenshots.
// usage: node tools/smoke.mjs --url http://127.0.0.1:4173/ [--browser path] [--generate] [--width 1440] [--out dir]
import { spawn } from "node:child_process";
import { mkdirSync, writeFileSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";

const arg = (k, d) => { const i = process.argv.indexOf(`--${k}`); return i < 0 ? d : process.argv[i + 1]; };
const has = (k) => process.argv.includes(`--${k}`);
const url = arg("url", "http://127.0.0.1:4173/"), width = +arg("width", 1440), height = +arg("height", 900);
const browser = arg("browser", "C:/Program Files (x86)/Microsoft/Edge/Application/msedge.exe"), out = arg("out", join(tmpdir(), "talus-smoke"));
mkdirSync(out, { recursive: true });
const port = 9300 + Math.floor(Math.random() * 500);
const proc = spawn(browser, ["--headless=new", `--remote-debugging-port=${port}`, `--user-data-dir=${join(tmpdir(), `talus-cdp-${port}`)}`,
  "--enable-unsafe-webgpu", "--no-first-run", "--disable-extensions", `--window-size=${width},${height}`,
  ...(arg("hostmap") ? [`--host-resolver-rules=MAP ${arg("hostmap")}`] : []), "about:blank"], { stdio: "ignore" });
const sleep = (ms) => new Promise((r) => setTimeout(r, ms));
let target;
for (let i = 0; i < 50 && !target; i++) {
  await sleep(200);
  try { target = (await (await fetch(`http://127.0.0.1:${port}/json/list`)).json()).find((t) => t.type === "page"); } catch {}
}
const ws = new WebSocket(target.webSocketDebuggerUrl);
await new Promise((r) => ws.addEventListener("open", r));
let id = 0;
const pending = new Map(), logs = [];
ws.addEventListener("message", (ev) => {
  const m = JSON.parse(ev.data);
  if (m.id && pending.has(m.id)) { pending.get(m.id)(m); pending.delete(m.id); }
  if (m.method === "Runtime.consoleAPICalled") logs.push(`${m.params.type}: ${m.params.args.map((a) => a.value ?? a.description).join(" ")}`);
  if (m.method === "Runtime.exceptionThrown") logs.push(`exception: ${m.params.exceptionDetails.exception?.description ?? m.params.exceptionDetails.text}`);
  if (m.method === "Log.entryAdded") logs.push(`${m.params.entry.level}: ${m.params.entry.text} ${m.params.entry.url ?? ""}`);
});
const send = (method, params = {}) => new Promise((r) => { const i = ++id; pending.set(i, r); ws.send(JSON.stringify({ id: i, method, params })); });
const evaluate = async (expression) => (await send("Runtime.evaluate", { expression, awaitPromise: true, returnByValue: true })).result?.result?.value;
const shot = async (name, full = false) => {
  const metrics = full ? await send("Page.getLayoutMetrics") : null;
  const clip = full ? { x: 0, y: 0, width, height: Math.ceil(metrics.result.cssContentSize.height), scale: 1 } : undefined;
  const r = await send("Page.captureScreenshot", { format: "png", captureBeyondViewport: full, ...(clip ? { clip } : {}) });
  writeFileSync(join(out, name), Buffer.from(r.result.data, "base64"));
};

await send("Runtime.enable"); await send("Log.enable"); await send("Page.enable");
await send("Emulation.setDeviceMetricsOverride", { width, height, deviceScaleFactor: +arg("dpr", 1), mobile: width < 600 });
if (has("reduce")) await send("Emulation.setEmulatedMedia", { features: [{ name: "prefers-reduced-motion", value: "reduce" }] });
if (has("nogpu")) await send("Page.addScriptToEvaluateOnNewDocument", { source: "Object.defineProperty(Navigator.prototype, 'gpu', { get: () => undefined });" });
await send("Page.navigate", { url });
await sleep(+arg("wait", 3000));
const info = await evaluate(`({ webgpu: !!navigator.gpu, status: document.querySelector("#status")?.textContent, overflow: document.documentElement.scrollWidth > innerWidth, title: document.title })`);
console.log("loaded", JSON.stringify(info));
await shot("top.png");
await evaluate(`document.querySelectorAll(".reveal").forEach((e) => e.classList.add("in"))`);
await sleep(700);
await shot("full.png", true);
if (has("generate")) {
  await evaluate(`document.querySelector("#go").click()`);
  const t0 = Date.now();
  let s = "";
  while (Date.now() - t0 < +arg("timeout", 240000)) {
    await sleep(1000);
    s = await evaluate(`document.querySelector("#status").textContent + " | " + document.querySelector("#backend").textContent`);
    if (/Done|wrong|Stopped/.test(s)) break;
  }
  console.log("generate:", s, `${((Date.now() - t0) / 1000).toFixed(0)} s`);
  await evaluate(`document.querySelector("#generate").scrollIntoView()`);
  await sleep(600);
  await shot("generated.png");
}
console.log(logs.length ? logs.join("\n") : "no console output");
ws.close();
proc.kill();
