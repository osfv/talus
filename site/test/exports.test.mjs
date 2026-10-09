// Engine exports: 16-bit PNG decodes back to the same samples, RAW is Unity-sized and corner-aligned,
// the OBJ has one vertex and normal per cell, and the sidecar carries the physical scale.
import assert from "node:assert/strict";
import test from "node:test";
import { inflateSync } from "node:zlib";

import { obj, png16, r16, resample, sidecar, toUint16, unitySize } from "../src/js/exports.js";

const N = 64, world = { resolution: N, extent_m: 4096, max_height_m: 1200, sea_level: 0.2 };
const h = Float32Array.from({ length: N * N }, (_, i) => 0.5 + 0.4 * Math.sin((i % N) / 7) * Math.cos(Math.floor(i / N) / 9));

test("16-bit PNG round-trips every sample", async () => {
  const bytes = new Uint8Array(await (await png16(h, N)).arrayBuffer());
  assert.deepEqual([...bytes.slice(0, 8)], [137, 80, 78, 71, 13, 10, 26, 10]);
  const dv = new DataView(bytes.buffer), chunks = {};
  for (let o = 8; o < bytes.length;) {
    const len = dv.getUint32(o), type = String.fromCharCode(...bytes.slice(o + 4, o + 8));
    chunks[type] = bytes.slice(o + 8, o + 8 + len);
    o += 12 + len;
  }
  const ihdr = new DataView(chunks.IHDR.buffer);
  assert.deepEqual([ihdr.getUint32(0), ihdr.getUint32(4), chunks.IHDR[8], chunks.IHDR[9]], [N, N, 16, 0]);
  const raw = inflateSync(chunks.IDAT), want = toUint16(h);
  for (let y = 0; y < N; y++) {
    assert.equal(raw[y * (1 + 2 * N)], 0);
    for (let x = 0; x < N; x++) {
      const o = y * (1 + 2 * N) + 1 + 2 * x;
      assert.equal((raw[o] << 8) | raw[o + 1], want[y * N + x]);
    }
  }
});

test("Unity RAW is 65 x 65 little-endian and keeps the corners", async () => {
  assert.equal(unitySize(64), 65);
  assert.equal(unitySize(65), 65);
  assert.equal(unitySize(128), 129);
  const { blob, size } = r16(h, N), dv = new DataView(await blob.arrayBuffer());
  assert.equal(size, 65);
  assert.equal(dv.byteLength, 65 * 65 * 2);
  const corner = (x, y) => dv.getUint16(2 * (y * 65 + x), true), want = toUint16(h);
  assert.equal(corner(0, 0), want[0]);
  assert.equal(corner(64, 64), want[N * N - 1]);
  const flat = resample(new Float32Array(N * N).fill(0.3), N, 65);
  assert.ok(flat.every((v) => Math.abs(v - 0.3) < 1e-6), "a flat map stays flat");
});

test("OBJ and sidecar describe the map in meters", async () => {
  const text = await obj(h, N, world).text(), lines = text.trim().split("\n");
  assert.equal(lines.filter((l) => l.startsWith("v ")).length, N * N);
  assert.equal(lines.filter((l) => l.startsWith("vn ")).length, N * N);
  assert.equal(lines.filter((l) => l.startsWith("f ")).length, 2 * (N - 1) * (N - 1));
  const [, x, y, z] = lines.find((l) => l.startsWith("v ")).split(" ").map(Number);
  assert.deepEqual([x, z], [32, 32]);
  assert.ok(Math.abs(y - h[0] * 1200) < 1e-3);
  const meta = JSON.parse(await sidecar(N, world, { png16: { name: "a.png", resolution: 64 }, r16: { name: "a.r16", resolution: 65 } }, {}).text());
  assert.equal(meta.footprint_m, 4032);
  assert.equal(meta.files.png16.cell_size_m, 64);
  assert.equal(meta.files.r16.cell_size_m, 63);
});
