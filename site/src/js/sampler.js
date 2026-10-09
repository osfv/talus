// Talus in JavaScript: DDIM sampling (eta 0) with classifier-free guidance, v-prediction and relative heights.
// Mirrors nullscape.inference.sampler.TerrainSampler.sample for one map at a time. The denoiser is the ONNX
// export from scripts/export_web.py; `ort` is any ONNX Runtime JavaScript build (web or node).
// Seeds drive a JavaScript PRNG, so a seed here gives a different map than the same seed in the Python CLI.

export function rng(seed) {
  // splitmix32 seeding into xoshiro128**, uniform floats in [0, 1)
  let s = seed >>> 0;
  const next32 = () => {
    s = (s + 0x9e3779b9) >>> 0;
    let z = s;
    z = Math.imul(z ^ (z >>> 16), 0x85ebca6b);
    z = Math.imul(z ^ (z >>> 13), 0xc2b2ae35);
    return (z ^ (z >>> 16)) >>> 0;
  };
  let a = next32(), b = next32(), c = next32(), d = next32();
  return () => {
    const r = Math.imul(((Math.imul(b, 5) << 7) | (Math.imul(b, 5) >>> 25)), 9) >>> 0;
    const t = b << 9;
    c ^= a; d ^= b; b ^= c; a ^= d; c ^= t;
    d = (d << 11) | (d >>> 21);
    return r / 4294967296;
  };
}

export function normals(seed, n) {
  const u = rng(seed), out = new Float32Array(n);
  for (let i = 0; i < n; i += 2) {
    const r = Math.sqrt(-2 * Math.log(1 - u())), a = 2 * Math.PI * u();
    out[i] = r * Math.cos(a);
    if (i + 1 < n) out[i + 1] = r * Math.sin(a);
  }
  return out;
}

const roundHalfEven = (v) => {
  const f = Math.floor(v), d = v - f;
  return d > 0.5 || (d === 0.5 && f % 2 !== 0) ? f + 1 : f;
};

export function schedule(meta, steps) {
  if (meta.schedules && meta.schedules[steps]) return meta.schedules[steps];
  const T = meta.alphas_cumprod.length, top = Math.sqrt(T - 1), out = [];
  for (let i = 0; i < steps; i++) {
    const t = roundHalfEven((top * (1 - i / (steps - 1))) ** 2);
    if (out[out.length - 1] !== t) out.push(t);
  }
  return out;
}

export function parsePrior(buffer, meta) {
  const { count, keys } = meta.prior;
  return { conditions: new Float32Array(buffer, 0, count * keys), labels: new Uint8Array(buffer, count * keys * 4, count) };
}

export class TalusSampler {
  constructor(ort, session, meta, prior) {
    this.ort = ort;
    this.session = session;
    this.meta = meta;
    this.prior = prior;
    this.keys = meta.condition_keys;
    this.K = this.keys.length;
    this.R = meta.world.resolution;
    this.unknownLabel = meta.archetypes.length;
  }

  // Raw request -> model conditioning, including the relative-height placement that Python's
  // TerrainSampler._fill_placement does: unspecified mean elevation / relief come from one of the
  // 16 nearest training maps (ties included), same terrain type if one was requested.
  conditions({ archetype = null, properties = {}, seed = 0 } = {}) {
    const { K, keys, meta } = this, mean = meta.cond_mean, std = meta.cond_std;
    const label = archetype == null ? this.unknownLabel : meta.archetypes.indexOf(archetype);
    if (label < 0) throw new Error(`unknown terrain type ${archetype}`);
    const raw = Float64Array.from(mean), known = new Uint8Array(K);
    for (const [key, value] of Object.entries(properties)) {
      const j = keys.indexOf(key);
      if (j < 0) throw new Error(`unknown property ${key}`);
      if (value == null || !Number.isFinite(value)) continue;
      raw[j] = value;
      known[j] = 1;
    }
    const jm = keys.indexOf(meta.height_param.mean_key), jr = keys.indexOf(meta.height_param.relief_key);
    if (!(known[jm] && known[jr])) {
      const { conditions: bc, labels: bl } = this.prior, n = bl.length;
      let cand = [];
      for (let i = 0; i < n; i++) if (label === this.unknownLabel || bl[i] === label) cand.push(i);
      if (!cand.length) throw new Error("the placement bank has no maps of this terrain type");
      if (known.some(Boolean)) {
        const d = cand.map((i) => {
          let s = 0;
          for (let j = 0; j < K; j++) if (known[j]) {
            const diff = (bc[i * K + j] - mean[j]) / std[j] - (raw[j] - mean[j]) / std[j];
            s += diff * diff;
          }
          return s;
        });
        const cutoff = [...d].sort((a, b) => a - b)[Math.min(15, d.length - 1)];
        cand = cand.filter((_, k) => d[k] <= cutoff);
      }
      const pick = cand[Math.floor(rng(seed ^ 0x51ed270b)() * cand.length)];
      for (const j of [jm, jr]) if (!known[j]) { raw[j] = bc[pick * K + j]; known[j] = 1; }
    }
    const z = new Float32Array(K);
    for (let j = 0; j < K; j++) z[j] = known[j] ? (raw[j] - mean[j]) / std[j] : 0;
    return { z, known, label, mean: raw[jm], relief: raw[jr], raw: Array.from(raw) };
  }

  decode(x, mean, relief, out) {
    const hp = this.meta.height_param, r = Math.max(relief, hp.relief_floor) / hp.scale;
    for (let i = 0; i < x.length; i++) out[i] = Math.min(1, Math.max(0, mean + x[i] * r));
    return out;
  }

  // Returns heights [R*R] in [0, 1]. onStep(i, n, previewHeights) sees the predicted clean map after every
  // step; return false from it (or abort `signal`) to stop early.
  async sample({ archetype = null, properties = {}, seed = 0, steps = null, guidance = null, noise = null,
                 conditions = null, onStep = null, signal = null } = {}) {
    const { ort, meta, K, R } = this, P = R * R;
    steps = steps ?? meta.sampling.steps;
    guidance = guidance ?? meta.sampling.guidance;
    const c = conditions ?? this.conditions({ archetype, properties, seed });
    const ts = schedule(meta, steps), ab = meta.alphas_cumprod, clip = meta.x0_clip;
    const guided = guidance !== 1, B = guided ? 2 : 1;
    const x = noise ? Float32Array.from(noise) : normals(seed, P);
    const xIn = new Float32Array(B * P), tIn = new Float32Array(B), cond = new Float32Array(B * K);
    const knownIn = new Float32Array(B * K), label = new Int32Array(B);
    cond.set(c.z);
    for (let j = 0; j < K; j++) knownIn[j] = c.known[j];
    label[0] = c.label;
    if (guided) { cond.set(c.z, K); label[1] = this.unknownLabel; }
    const x0 = new Float32Array(P), preview = new Float32Array(P);
    for (let i = 0; i < ts.length; i++) {
      if (signal?.aborted) throw new DOMException("Generation stopped", "AbortError");
      const a = ab[ts[i]], aPrev = i + 1 < ts.length ? ab[ts[i + 1]] : 1;
      xIn.set(x);
      if (guided) xIn.set(x, P);
      tIn.fill(ts[i]);
      const out = await this.session.run({
        x: new ort.Tensor("float32", xIn, [B, 1, R, R]), t: new ort.Tensor("float32", tIn, [B]),
        cond: new ort.Tensor("float32", cond, [B, K]), known: new ort.Tensor("float32", knownIn, [B, K]),
        label: new ort.Tensor("int32", label, [B]),
      });
      const v = out.v.data, sa = Math.sqrt(a), sb = Math.sqrt(1 - a), sap = Math.sqrt(aPrev), sbp = Math.sqrt(1 - aPrev);
      for (let p = 0; p < P; p++) {
        const vp = guided ? v[P + p] + guidance * (v[p] - v[P + p]) : v[p];
        const x0p = Math.min(clip, Math.max(-clip, sa * x[p] - sb * vp));
        const eps = (x[p] - sa * x0p) / sb;
        x0[p] = x0p;
        x[p] = sap * x0p + sbp * eps;
      }
      out.v.dispose?.();
      if (onStep && (await onStep(i + 1, ts.length, this.decode(x0, c.mean, c.relief, preview))) === false) break;
    }
    return { heights: this.decode(x, c.mean, c.relief, new Float32Array(P)), conditions: c, steps: ts.length };
  }
}
