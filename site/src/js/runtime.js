// Loads the model, the placement bank and ONNX Runtime, with byte-level progress, and keeps them in the
// browser's Cache Storage so the second visit starts at once. Picks WebGPU when the browser offers it.
import { TalusSampler, parsePrior } from "./sampler.js";

const CACHE = "talus-3.0.0";

async function fetchBytes(url, { gzip = false, onBytes = () => {} } = {}) {
  const cache = "caches" in self ? await caches.open(CACHE).catch(() => null) : null;
  let res = cache ? await cache.match(url) : null;
  const fresh = !res;
  if (!res) {
    res = await fetch(url);
    if (!res.ok) throw new Error(`Could not download ${url.split("/").pop()} (HTTP ${res.status})`);
  }
  if (fresh && cache) cache.put(url, res.clone()).catch(() => {});
  const reader = res.body.getReader(), parts = [];
  let loaded = 0;
  for (;;) {
    const { done, value } = await reader.read();
    if (done) break;
    parts.push(value);
    loaded += value.length;
    onBytes(loaded);
  }
  let blob = new Blob(parts);
  if (gzip) blob = await new Response(blob.stream().pipeThrough(new DecompressionStream("gzip"))).blob();
  return blob.arrayBuffer();
}

async function webgpuAvailable() {
  if (!navigator.gpu) return false;
  try { return !!(await navigator.gpu.requestAdapter()); } catch { return false; }
}

// onProgress(loadedBytes, totalBytes, label)
export async function loadTalus(base, meta, onProgress = () => {}) {
  const build = await (await fetch(new URL("build.json", base))).json();
  const files = build.files, total = Object.values(files).reduce((a, f) => a + f.bytes, 0), seen = {};
  const track = (key) => (n) => { seen[key] = n; onProgress(Object.values(seen).reduce((a, b) => a + b, 0), total, "Downloading"); };
  const ortUrl = new URL("vendor/ort/ort.webgpu.min.mjs", base).href;
  const [ort, wasm, model, prior] = await Promise.all([
    import(/* webpackIgnore: true */ ortUrl),
    fetchBytes(new URL(files.wasm.url, base).href, { gzip: files.wasm.gzip, onBytes: track("wasm") }),
    fetchBytes(new URL(files.model.url, base).href, { onBytes: track("model") }),
    fetchBytes(new URL(files.prior.url, base).href, { onBytes: track("prior") }),
  ]);
  ort.env.logLevel = "error";
  ort.env.wasm.wasmPaths = new URL("vendor/ort/", base).href;
  ort.env.wasm.wasmBinary = wasm;
  ort.env.wasm.numThreads = self.crossOriginIsolated ? Math.min(4, navigator.hardwareConcurrency || 1) : 1;
  onProgress(total, total, "Starting the model");
  let session = null, backend = "CPU";
  if (await webgpuAvailable()) {
    try {
      session = await ort.InferenceSession.create(model, { executionProviders: ["webgpu"], graphOptimizationLevel: "all", logSeverityLevel: 3 });
      backend = "WebGPU";
    } catch (err) {
      console.warn("WebGPU session failed; using the CPU", err);
    }
  }
  session ??= await ort.InferenceSession.create(model, { executionProviders: ["wasm"], graphOptimizationLevel: "all", logSeverityLevel: 3 });
  return { sampler: new TalusSampler(ort, session, meta, parsePrior(prior, meta)), backend };
}
