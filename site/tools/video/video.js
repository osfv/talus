// Deterministic frames for the launch video. Every terrain frame is a real Talus-3 sampler state (the noisy
// sample after each DDIM step, decoded to heights), blended between steps for smooth motion.
// tools/record.mjs calls window.renderFrame(t) for each frame time and screenshots the page.
import { loadTalus } from "../js/runtime.js";
import { normals } from "../js/sampler.js";
import { TerrainView } from "../js/view.js";

const base = new URL("../", import.meta.url), $ = (s) => document.querySelector(s);
const meta = await (await fetch(new URL("model/talus-3.json", base))).json();
const noise = new Uint8Array(await (await fetch(new URL("data/bluenoise.bin", base))).arrayBuffer());
const view = new TerrainView($("#c"), { noise, world: meta.world, autoRotate: false, texels: 360, target: [0, 0.16, 0], dist: 2.55 });
const { sampler, backend } = await loadTalus(base, meta);
const P = meta.world.resolution ** 2, STEPS = 25;

const MAPS = [
  { archetype: "ridges", seed: 106, label: "ridges" },
  { archetype: "islands", seed: 209, properties: { water_fraction: 0.62 }, label: "islands, 62% water" },
  { archetype: "mesas", seed: 407, label: "mesas" },
  { archetype: "mountains", seed: 313, label: "mountains" },
];
for (const m of MAPS) {
  const c = sampler.conditions({ archetype: m.archetype, properties: m.properties || {}, seed: m.seed });
  m.states = [sampler.decode(normals(m.seed, P), c.mean, c.relief, new Float32Array(P))];
  await sampler.sample({ conditions: c, seed: m.seed, steps: STEPS, onStep: (i, n, preview, noisy) => { m.states.push(Float32Array.from(noisy)); } });
}

// scene: [start, end) seconds, the map, and when it forms
const SCENES = [
  { from: 0, to: 8, map: 0, form: [0.5, 3.6], titleAt: 0.6, title: "Terrain from noise.", sub: [4.2, "Talus is a small diffusion model that draws game heightmaps."] },
  { from: 8, to: 12, map: 1, form: [8.3, 10.6], titleAt: 8, title: "Pick a terrain type.", sub: [8.3, "Every map starts as random noise and takes shape in 25 steps."] },
  { from: 12, to: 16, map: 2, form: [12.3, 14.6], titleAt: 8, title: "Pick a terrain type.", sub: [8.3, "Every map starts as random noise and takes shape in 25 steps."] },
  { from: 16, to: 20, map: 3, form: [16.3, 18.6], titleAt: 8, title: "Pick a terrain type.", sub: [8.3, "Every map starts as random noise and takes shape in 25 steps."] },
  { from: 20, to: 1e9, map: 0, form: [0, 0], end: true, titleAt: 20, title: "talus.tersa.tech", sub: [20.3, "Runs in your browser. Exports to Unity, Unreal, Godot and Blender."],
    sub2: [20.7, "Free and open source. Made by Tersa."] },
];
// the end card is rendered at t >= 20; tools/record.mjs places it after the recorded UI clip

const h = new Float32Array(P);
const stepFade = (t, t0, d = 0.32) => Math.min(1, Math.max(0, Math.floor(((t - t0) / d) * 4) / 4));
function heights(m, f) {
  const k = Math.min(STEPS - 1, Math.floor(f * STEPS)), a = m.states[k], b = m.states[k + 1], w = f * STEPS - k;
  for (let i = 0; i < P; i++) h[i] = a[i] + (b[i] - a[i]) * w;
  return { h, step: f >= 1 ? STEPS : k };
}

window.renderFrame = (t) => {
  const s = SCENES.find((x) => t >= x.from && t < x.to), m = MAPS[s.map];
  const f = s.end ? 1 : Math.min(1, Math.max(0, (t - s.form[0]) / (s.form[1] - s.form[0])));
  const { h: hh, step } = heights(m, f);
  view.upload(view.texB, hh);
  view.mix = 1;
  view.yaw = -0.75 + 0.16 * t;
  view.target = s.end ? [0, 0.26, 0] : [0, 0.16, 0];
  view.draw();
  $("#title").textContent = s.title;
  $("#title").style.opacity = stepFade(t, s.titleAt, 0.24);
  $("#sub").textContent = s.sub[1];
  $("#sub").style.opacity = stepFade(t, s.sub[0]);
  $("#sub2").textContent = s.sub2 ? s.sub2[1] : "";
  $("#sub2").style.opacity = s.sub2 ? stepFade(t, s.sub2[0]) : 0;
  $("#label").textContent = s.end ? "" : m.label;
  $("#step").textContent = s.end ? "" : `step ${String(step).padStart(2, "0")} / ${STEPS}`;
  $("#url").style.opacity = s.end ? 0 : 1;
  return { step, backend };
};
await document.fonts.ready;
window.videoReady = { backend, maps: MAPS.length };
