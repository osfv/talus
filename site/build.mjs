// Builds dist/: the static site, the browser model files, and the ONNX Runtime files it loads on demand.
// The model comes from public/model (scripts/export_web.py, or the v3.0.0 release assets in CI).
import { cpSync, existsSync, mkdirSync, readFileSync, rmSync, statSync, writeFileSync } from "node:fs";
import { gzipSync } from "node:zlib";

const root = new URL("./", import.meta.url), dist = new URL("dist/", root);
const ORT = ["ort.webgpu.min.mjs", "ort-wasm-simd-threaded.asyncify.mjs"], WASM = "ort-wasm-simd-threaded.asyncify.wasm";
const MODEL = ["talus-3.fp16.onnx", "talus-3.json", "talus-3.prior.bin"];

rmSync(dist, { recursive: true, force: true });
cpSync(new URL("src/", root), dist, { recursive: true });
mkdirSync(new URL("model/", dist), { recursive: true });
for (const f of MODEL) {
  const src = new URL(`public/model/${f}`, root);
  if (!existsSync(src)) throw new Error(`missing ${src.pathname}: run scripts/export_web.py or download the release assets`);
  cpSync(src, new URL(`model/${f}`, dist));
}
const vendor = new URL("vendor/ort/", dist), ortDist = new URL("node_modules/onnxruntime-web/dist/", root);
mkdirSync(vendor, { recursive: true });
for (const f of ORT) cpSync(new URL(f, ortDist), new URL(f, vendor));
// The runtime is 27 MB of WebAssembly but compresses 4x; the page fetches the .gz and inflates it itself,
// because GitHub Pages may not compress .wasm.
const wasm = readFileSync(new URL(WASM, ortDist)), wasmGz = gzipSync(wasm, { level: 9 });
writeFileSync(new URL(`${WASM}.gz`, vendor), wasmGz);
const meta = JSON.parse(readFileSync(new URL("model/talus-3.json", dist)));
const files = {
  model: { url: "model/talus-3.fp16.onnx", bytes: statSync(new URL("model/talus-3.fp16.onnx", dist)).size },
  prior: { url: "model/talus-3.prior.bin", bytes: statSync(new URL("model/talus-3.prior.bin", dist)).size },
  wasm: { url: `vendor/ort/${WASM}.gz`, bytes: wasmGz.length, raw: wasm.length, gzip: true },
};
writeFileSync(new URL("build.json", dist), JSON.stringify({ model: meta.name, version: meta.version, files }, null, 1));
writeFileSync(new URL("CNAME", dist), "talus.tersa.tech\n");
const total = Object.values(files).reduce((a, f) => a + f.bytes, 0);
console.log(`dist/ built; first-run download ${(total / 1e6).toFixed(1)} MB`);
