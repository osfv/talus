// Parity of the JavaScript sampler with PyTorch: same denoiser (ONNX), same injected noise and conditioning,
// compared against reference samples written by scripts/export_web.py. Run `npm test` after the export.
import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import test from "node:test";
import { fileURLToPath } from "node:url";
import * as ort from "onnxruntime-node";

import { TalusSampler, normals, parsePrior, rng, schedule } from "../src/js/sampler.js";

const dir = new URL("../public/model/", import.meta.url);
const meta = JSON.parse(readFileSync(new URL("talus-3.json", dir)));
const prior = parsePrior(readFileSync(new URL(meta.prior.file, dir)).buffer.slice(0), meta);
const fixtures = JSON.parse(readFileSync(new URL("fixtures/fixtures.json", dir)));
const floats = (name) => { const b = readFileSync(new URL(`fixtures/${name}`, dir)); return new Float32Array(b.buffer, b.byteOffset, b.length / 4); };

async function sampler(kind) {
  const session = await ort.InferenceSession.create(fileURLToPath(new URL(meta.models[kind].file, dir)));
  return new TalusSampler(ort, session, meta, prior);
}

for (const kind of ["fp32", "fp16"]) {
  test(`${kind} model reproduces the PyTorch reference samples`, async () => {
    const s = await sampler(kind);
    for (const f of fixtures) {
      const conditions = { z: Float32Array.from(f.z), known: Uint8Array.from(f.known), label: f.label, mean: f.mean, relief: f.relief };
      const { heights } = await s.sample({ conditions, noise: floats(`${f.stem}.noise.bin`), steps: f.steps, guidance: f.guidance });
      const ref = floats(`${f.stem}.heights.bin`);
      let max = 0;
      for (let i = 0; i < ref.length; i++) max = Math.max(max, Math.abs(heights[i] - ref[i]));
      // heights are normalized (1.0 = 1,200 m): fp32 within 0.6 m, fp16-stored weights within 6 m
      assert.ok(max < (kind === "fp32" ? 5e-4 : 5e-3), `${f.stem}: max |JS - PyTorch| = ${max}`);
    }
  });
}

test("schedules match the exported Python schedules", () => {
  for (const steps of [12, 25, 50]) assert.deepEqual(schedule({ ...meta, schedules: null }, steps), meta.schedules[steps]);
});

test("placement fills open elevation and relief from maps of the requested type", async () => {
  const s = new TalusSampler(ort, null, meta, prior);
  const jm = meta.condition_keys.indexOf("mean_elevation"), jr = meta.condition_keys.indexOf("relief");
  const islands = meta.archetypes.indexOf("islands");
  for (let seed = 0; seed < 20; seed++) {
    const c = s.conditions({ archetype: "islands", properties: { water_fraction: 0.6 }, seed });
    assert.equal(c.label, islands);
    assert.ok(c.known[jm] && c.known[jr]);
    const matches = [...prior.labels.keys()].some((i) => prior.labels[i] === islands
      && prior.conditions[i * meta.prior.keys + jm] === Math.fround(c.mean) && prior.conditions[i * meta.prior.keys + jr] === Math.fround(c.relief));
    assert.ok(matches, "placement must come from one island map in the bank");
  }
  const fixed = s.conditions({ properties: { mean_elevation: 0.3, relief: 0.1 } });
  assert.equal(fixed.mean, 0.3);
  assert.equal(fixed.label, meta.archetypes.length);
});

test("seeded noise is deterministic and standard normal", () => {
  assert.deepEqual(normals(42, 64), normals(42, 64));
  assert.notDeepEqual(normals(42, 64), normals(43, 64));
  const n = normals(1, 40000), mean = n.reduce((a, b) => a + b) / n.length;
  const sd = Math.sqrt(n.reduce((a, b) => a + (b - mean) ** 2, 0) / n.length);
  assert.ok(Math.abs(mean) < 0.03 && Math.abs(sd - 1) < 0.03, `mean ${mean}, sd ${sd}`);
  const u = rng(5);
  assert.ok(Array.from({ length: 1000 }, u).every((v) => v >= 0 && v < 1));
});
