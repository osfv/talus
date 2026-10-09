import * as X from "./exports.js";
import { TerrainView, drawHeightmap } from "./view.js";

const base = new URL("../", import.meta.url);
const $ = (s) => document.querySelector(s);
const html = document.documentElement;
const REDUCE = matchMedia("(prefers-reduced-motion: reduce)").matches;
window.talusReady = true;  // the inline fallback in index.html shows all content if this module never runs

// Fonts first, then the hero; everything else fades in when it scrolls into view.
const ready = () => html.classList.add("is-ready");
setTimeout(ready, 1400);
(document.fonts?.ready ?? Promise.resolve()).then(ready, ready);
const io = new IntersectionObserver((entries) => entries.forEach((e) => { if (e.isIntersecting) { e.target.classList.add("in"); io.unobserve(e.target); } }), { threshold: 0.12 });
document.querySelectorAll("section:not(.hero) .reveal").forEach((el) => io.observe(el));

const get = async (path, kind = "json") => { const r = await fetch(new URL(path, base)); if (!r.ok) throw new Error(`${path}: HTTP ${r.status}`); return r[kind](); };
const [meta, build, heroInfo, heroBin, noiseBin, framesInfo, framesBin] = await Promise.all([
  get("model/talus-3.json"), get("build.json"), get("data/hero.json"), get("data/hero.bin", "arrayBuffer"),
  get("data/bluenoise.bin", "arrayBuffer"), get("data/frames.json"), get("data/frames.bin", "arrayBuffer"),
]);
const world = meta.world, N = world.resolution, P = N * N, noise = new Uint8Array(noiseBin);
const unpack = (buf, i) => Float32Array.from(new Uint16Array(buf, i * P * 2, P), (v) => v / 65535);
const heroMaps = heroInfo.map((info, i) => ({ info, heights: unpack(heroBin, i) }));
const title = (s) => s[0].toUpperCase() + s.slice(1);
const mb = (b) => (b / 1e6).toFixed(b < 1e7 ? 1 : 0);
const firstRun = Object.values(build.files).reduce((a, f) => a + f.bytes, 0);

// ---------- 1-bit top-down render (frames strip, and the fallback when WebGL2 is missing)

function topDown(canvas, h, scale = 2) {
  const M = N * scale, ctx = canvas.getContext("2d"), img = ctx.createImageData(M, M), sea = world.sea_level;
  canvas.width = canvas.height = M;
  const at = (x, y) => h[Math.min(N - 1, Math.max(0, y)) * N + Math.min(N - 1, Math.max(0, x))];
  const L = [-0.55, 0.62, -0.56], k = (world.max_height_m / (world.extent_m / N)) * 1.6;
  for (let y = 0; y < M; y++) for (let x = 0; x < M; x++) {
    const gx = (x + 0.5) / scale - 0.5, gy = (y + 0.5) / scale - 0.5, ix = Math.floor(gx), iy = Math.floor(gy), fx = gx - ix, fy = gy - iy;
    const v = (a, b) => at(ix + a, iy + b);
    const hv = (v(0, 0) * (1 - fx) + v(1, 0) * fx) * (1 - fy) + (v(0, 1) * (1 - fx) + v(1, 1) * fx) * fy;
    let lum;
    if (hv < sea) lum = 0.3 - Math.min(1, (sea - hv) * 12) * 0.26;
    else if ((hv - sea) * world.max_height_m < 6) lum = 1;
    else {
      const dx = (Math.max(sea, v(1, 0)) - Math.max(sea, v(-1, 0))) * k / 2, dy = (Math.max(sea, v(0, 1)) - Math.max(sea, v(0, -1))) * k / 2;
      const l = Math.hypot(dx, 1, dy), d = Math.max(0, (-dx * L[0] + L[1] - dy * L[2]) / l / Math.hypot(...L));
      lum = 0.05 + 0.95 * Math.pow(d, 1.4);
    }
    const on = lum * 255 > noise[(y & 63) * 64 + (x & 63)], i = (y * M + x) * 4;
    img.data[i] = img.data[i + 1] = img.data[i + 2] = on ? 242 : 4;
    img.data[i + 3] = 255;
  }
  ctx.putImageData(img, 0, 0);
}

function makeView(canvas, opts) {
  try { return new TerrainView(canvas, { noise, world, ...opts }); } catch (err) {
    console.warn("WebGL2 unavailable, showing a flat map", err);
    return { set: (h) => topDown(canvas, h, 4) };
  }
}

// ---------- hero: stored Talus-3 maps, one terrain type after another

const heroView = makeView($("#hero-canvas"), { texels: 420 });
const heroCaption = $("#hero-caption");
let heroIndex = 0;
const showHero = (i, ms) => {
  const m = heroMaps[i];
  heroView.set(m.heights, ms);
  heroCaption.textContent = `Talus-3 output: ${m.info.archetype}, seed ${m.info.seed}`;
};
showHero(0, 0);
if (!REDUCE) setInterval(() => { if (!document.hidden) { heroIndex = (heroIndex + 1) % (heroMaps.length - 1); showHero(heroIndex, 1100); } }, 7000);

// ---------- frames strip

const framesEl = $("#frames");
framesInfo.frames.forEach((step, i) => {
  const li = document.createElement("li"), c = document.createElement("canvas"), s = document.createElement("span");
  c.setAttribute("role", "img");
  c.setAttribute("aria-label", `The island after step ${step} of ${framesInfo.steps}`);
  s.className = "mono";
  s.textContent = `Step ${step}`;
  li.append(c, s);
  framesEl.append(li);
  topDown(c, unpack(framesBin, i));
});

// ---------- generator controls

const TYPES = [["any", "Any"], ...meta.archetypes.map((a) => [a, title(a)])];
const PROPS = [
  { key: "relief", q: "relief", label: "Relief", fmt: (v) => `${Math.round(v * world.max_height_m)} m` },
  { key: "water_fraction", q: "water", label: "Water", fmt: (v) => `${Math.round(v * 100)}%` },
  { key: "mean_slope_deg", q: "slope", label: "Mean slope", fmt: (v) => `${v.toFixed(1)}\u00b0` },
  { key: "mean_elevation", q: "elev", label: "Mean elevation", fmt: (v) => `${Math.round(v * world.max_height_m)} m` },
  { key: "spectral_beta", q: "smooth", label: "Smoothness", fmt: (v) => `\u03b2 ${v.toFixed(2)}` },
];
const STEPS = [[12, "Draft, 12 steps"], [25, "25 steps"], [50, "Best, 50 steps"]];

const segment = (root, name, items, value) => {
  root.innerHTML = "";
  for (const [v, label] of items) {
    const l = document.createElement("label"), input = document.createElement("input"), span = document.createElement("span");
    Object.assign(input, { type: "radio", name, value: String(v), checked: String(v) === String(value) });
    span.textContent = label;
    l.append(input, span);
    root.append(l);
  }
};

const propsEl = $("#props");
for (const p of PROPS) {
  const row = document.createElement("div");
  row.className = "prop auto";
  row.innerHTML = `<label for="p-${p.q}">${p.label}</label><output id="o-${p.q}" for="p-${p.q}"></output>
    <label class="check"><input type="checkbox" id="a-${p.q}" checked aria-label="Auto ${p.label.toLowerCase()}" /> Auto</label>
    <input type="range" id="p-${p.q}" aria-describedby="o-${p.q}" />`;
  propsEl.append(row);
  p.row = row; p.range = row.querySelector('input[type="range"]'); p.auto = row.querySelector('input[type="checkbox"]'); p.out = row.querySelector("output");
  p.range.addEventListener("input", () => { p.auto.checked = false; syncProp(p); });
  p.auto.addEventListener("change", () => { if (p.auto.checked) p.range.value = rangeFor(p)[1]; syncProp(p); });
}

const typeValue = () => $('#types input:checked')?.value ?? "any";
const rangeFor = (p) => {
  const [lo, mid, hi] = meta.ranges[typeValue()][p.key];
  return hi - lo > 1e-4 ? [lo, mid, hi] : [lo, mid, lo + 0.01];
};
function syncProp(p) {
  p.row.classList.toggle("auto", p.auto.checked);
  p.out.textContent = p.auto.checked ? "" : p.fmt(+p.range.value);
}
function setRanges() {
  for (const p of PROPS) {
    const [lo, mid, hi] = rangeFor(p), keep = p.auto.checked ? mid : Math.min(hi, Math.max(lo, +p.range.value));
    Object.assign(p.range, { min: lo, max: hi, step: (hi - lo) / 200 });
    p.range.value = keep;
    syncProp(p);
  }
}

function readForm() {
  const type = typeValue(), props = {};
  for (const p of PROPS) if (!p.auto.checked) props[p.key] = +p.range.value;
  const seed = Math.max(0, Math.min(4294967295, Math.floor(+$("#seed").value || 0)));
  return { type, archetype: type === "any" ? null : type, props, seed, steps: +($('#steps input:checked')?.value ?? 25) };
}

function toQuery(req) {
  const q = new URLSearchParams({ type: req.type, seed: req.seed, steps: req.steps });
  for (const p of PROPS) if (p.key in req.props) q.set(p.q, (+req.props[p.key]).toPrecision(4));
  return q.toString();
}

function applyQuery(q) {
  const type = TYPES.some(([t]) => t === q.get("type")) ? q.get("type") : "islands";
  segment($("#types"), "type", TYPES, type);
  segment($("#steps"), "steps", STEPS, [12, 25, 50].includes(+q.get("steps")) ? +q.get("steps") : 25);
  $("#seed").value = q.has("seed") ? q.get("seed") : 209;
  setRanges();
  const hasAny = PROPS.some((p) => q.has(p.q));
  for (const p of PROPS) {
    const v = q.has(p.q) ? +q.get(p.q) : !hasAny && type === "islands" && p.key === "water_fraction" ? 0.62 : null;
    p.auto.checked = v == null || !Number.isFinite(v);
    if (!p.auto.checked) { p.range.max = Math.max(+p.range.max, v); p.range.min = Math.min(+p.range.min, v); p.range.value = v; }
    syncProp(p);
  }
}

applyQuery(new URLSearchParams(location.search));
$("#types").addEventListener("change", setRanges);
$("#reroll").addEventListener("click", () => { $("#seed").value = crypto.getRandomValues(new Uint32Array(1))[0] % 1000000; });

// ---------- the current map, its stats and exports

const genView = makeView($("#gen-canvas"), { texels: 520 });
let current = null;

function stats(h) {
  const sorted = Float32Array.from(h).sort(), q = (f) => sorted[Math.min(P - 1, Math.round(f * (P - 1)))];
  let water = 0;
  for (const v of h) if (v < world.sea_level) water++;
  return { relief: (q(0.98) - q(0.02)) * world.max_height_m, water: water / P };
}

function show(heights, req, source) {
  current = { heights, req, source };
  genView.set(heights, 0);
  drawHeightmap($("#heightmap"), heights, N);
  const s = stats(heights);
  $("#out-title").textContent = `${req.type === "any" ? "Any terrain" : title(req.type)}, seed ${req.seed}`;
  $("#out-meta").textContent = `${N} x ${N} cells, ${Math.round(s.relief)} m relief, ${Math.round(s.water * 100)}% water`;
}

const stored = heroMaps[1];
show(stored.heights, { type: stored.info.archetype, props: stored.info.properties, seed: stored.info.seed, steps: stored.info.steps },
  { ...stored.info, stored: true });

const status = (text) => { $("#status").textContent = text; };
const progress = (f) => { $("#progress span").style.width = `${Math.round(f * 1000) / 10}%`; };
status(`Stored sample. The first run downloads about ${Math.round(firstRun / 1e6)} MB.`);

const stem = () => `talus-3_${current.req.type}_${current.req.seed}`;
document.querySelectorAll("[data-export]").forEach((btn) => btn.addEventListener("click", async () => {
  if (!current) return;
  const h = current.heights, kind = btn.dataset.export;
  if (kind === "png") X.download(await X.png16(h, N), `${stem()}.png`);
  else if (kind === "r16") X.download(X.r16(h, N).blob, `${stem()}_${X.unitySize(N)}.r16`);
  else if (kind === "obj") X.download(X.obj(h, N, world), `${stem()}.obj`);
  else {
    const files = { png16: { name: `${stem()}.png`, resolution: N }, r16: { name: `${stem()}_${X.unitySize(N)}.r16`, resolution: X.unitySize(N) },
                    obj: { name: `${stem()}.obj`, resolution: N } };
    const source = { model: `${meta.name} ${meta.version}`, generator: "talus.tersa.tech", terrain: current.req.type,
                     seed: current.req.seed, steps: current.req.steps, guidance: meta.sampling.guidance, requested: current.req.props, ...current.source };
    X.download(X.sidecar(N, world, files, source), `${stem()}.json`);
  }
}));

$("#share").addEventListener("click", async () => {
  if (!current) return;
  const url = `${location.origin}${location.pathname}?${toQuery(current.req)}`;
  try { await navigator.clipboard.writeText(url); $("#share").textContent = "Copied"; } catch { prompt("Copy this link", url); }
  setTimeout(() => { $("#share").textContent = "Copy link"; }, 1600);
});

// ---------- generation

let talus = null, loading = null, busy = null;
const go = $("#go");

async function ensureModel() {
  if (talus) return talus;
  const { loadTalus } = await import("./runtime.js");
  loading ??= loadTalus(base, meta, (n, total, label) => {
    progress(n / total);
    status(label === "Downloading" ? `Downloading ${mb(n)} of ${mb(total)} MB` : label);
  }).then((r) => (talus = r), (err) => { loading = null; throw err; });
  return loading;
}

async function generate() {
  if (busy) { busy.abort(); return; }
  const req = readForm();
  busy = new AbortController();
  go.textContent = "Stop";
  const signal = busy.signal;
  try {
    const { sampler, backend } = await ensureModel();
    $("#backend").textContent = backend === "WebGPU" ? "WebGPU" : "CPU";
    if (backend !== "WebGPU") status("No WebGPU in this browser, so Talus runs on the CPU. Draft quality is quickest.");
    const t0 = performance.now();
    const result = await sampler.sample({
      archetype: req.archetype, properties: req.props, seed: req.seed, steps: req.steps, signal,
      onStep: async (i, n, preview) => {
        genView.set(preview, 0);
        progress(i / n);
        status(`Step ${i} of ${n}`);
        await new Promise((r) => setTimeout(r, 0));
      },
    });
    show(result.heights, req, { mean_elevation: result.conditions.mean, relief: result.conditions.relief });
    status(`Done in ${((performance.now() - t0) / 1000).toFixed(1)} s on ${backend === "WebGPU" ? "WebGPU" : "the CPU"}`);
    history.replaceState(null, "", `?${toQuery(req)}`);
  } catch (err) {
    if (err.name === "AbortError") { status("Stopped"); if (current) genView.set(current.heights, 0); }
    else { console.error(err); status(`Something went wrong: ${err.message}`); }
  } finally {
    busy = null;
    go.textContent = "Generate terrain";
    progress(0);
  }
}

$("#controls").addEventListener("submit", (e) => { e.preventDefault(); generate(); });
document.querySelector("[data-generate]").addEventListener("click", () => { if (!busy) setTimeout(generate, REDUCE ? 0 : 400); });
