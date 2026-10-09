// Pre-generates the maps the page shows before the model loads, with the same JavaScript sampler and seeds as
// the in-browser generator (so "seed 104" on the page regenerates the same map).
// usage: node tools/make-hero.mjs [--try 3]   then pick seeds in HERO below and rerun without --try
import { readFileSync, writeFileSync } from "node:fs";
import { fileURLToPath } from "node:url";
import * as ort from "onnxruntime-node";

import { TalusSampler, parsePrior } from "../src/js/sampler.js";

const HERO = [
  { archetype: "ridges", seed: 106 }, { archetype: "islands", seed: 209, properties: { water_fraction: 0.62 }, frames: true },
  { archetype: "mountains", seed: 313 }, { archetype: "hills", seed: 504 }, { archetype: "mesas", seed: 407 },
  { archetype: "plains", seed: 603 },
];
const STEPS = 25, FRAMES = [1, 3, 6, 10, 16, 25];

const model = new URL("../public/model/", import.meta.url), data = new URL("../src/data/", import.meta.url);
const meta = JSON.parse(readFileSync(new URL("talus-3.json", model)));
const prior = parsePrior(readFileSync(new URL(meta.prior.file, model)).buffer.slice(0), meta);
const session = await ort.InferenceSession.create(fileURLToPath(new URL(meta.models.fp32.file, model)));
const sampler = new TalusSampler(ort, session, meta, prior);
const u16 = (h) => Uint16Array.from(h, (v) => Math.round(Math.min(1, Math.max(0, v)) * 65535));
const tries = process.argv.includes("--try") ? +process.argv[process.argv.indexOf("--try") + 1] : 0;

const maps = [], info = [];
for (const [k, entry] of HERO.entries()) {
  const seeds = tries ? Array.from({ length: tries }, (_, i) => entry.seed + i) : [entry.seed];
  for (const seed of seeds) {
    const frames = [];
    const t0 = performance.now();
    const { heights, conditions } = await sampler.sample({
      archetype: entry.archetype, properties: entry.properties || {}, seed, steps: STEPS, guidance: 2,
      onStep: entry.frames && !tries ? (i, n, preview) => { if (FRAMES.includes(i)) frames.push(u16(preview)); } : null,
    });
    maps.push(u16(heights));
    info.push({ archetype: entry.archetype, seed, properties: entry.properties || {}, steps: STEPS, guidance: 2,
                mean_elevation: conditions.mean, relief: conditions.relief });
    console.log(`${entry.archetype} seed ${seed}: relief ${(conditions.relief * 1200).toFixed(0)} m, ${((performance.now() - t0) / 1000).toFixed(1)} s`);
    if (frames.length) {
      writeFileSync(new URL("frames.bin", data), Buffer.concat(frames.map((f) => Buffer.from(f.buffer))));
      writeFileSync(new URL("frames.json", data), JSON.stringify({ archetype: entry.archetype, seed, steps: STEPS, frames: FRAMES }));
    }
  }
}
const out = tries ? "hero-try" : "hero";
writeFileSync(new URL(`${out}.bin`, data), Buffer.concat(maps.map((m) => Buffer.from(m.buffer))));
writeFileSync(new URL(`${out}.json`, data), JSON.stringify(info, null, 1));
console.log(`wrote src/data/${out}.bin (${maps.length} maps)`);
