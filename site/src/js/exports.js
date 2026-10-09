// Engine exports, matching nullscape.export.engine: 16-bit PNG, Unity RAW (.r16, little-endian, resampled to
// 2^n+1), OBJ in meters, and a JSON sidecar with the physical scale an engine needs.

export const toUint16 = (h) => Uint16Array.from(h, (v) => Math.round(Math.min(1, Math.max(0, v)) * 65535));

export function unitySize(resolution) {
  let n = 32;
  while (n + 1 < resolution) n *= 2;
  return n + 1;
}

// Bicubic (a = -0.75), corner-aligned, clamped at the borders: PyTorch's interpolate(mode="bicubic",
// align_corners=True), which the Python exporter uses.
export function resample(h, n, size) {
  const cubic = (x) => {
    const a = -0.75, t = Math.abs(x);
    return t <= 1 ? ((a + 2) * t - (a + 3)) * t * t + 1 : t < 2 ? ((a * t - 5 * a) * t + 8 * a) * t - 4 * a : 0;
  };
  const out = new Float32Array(size * size), s = (n - 1) / (size - 1), at = (x, y) => h[Math.min(n - 1, Math.max(0, y)) * n + Math.min(n - 1, Math.max(0, x))];
  for (let y = 0; y < size; y++) for (let x = 0; x < size; x++) {
    const fx = x * s, fy = y * s, ix = Math.floor(fx), iy = Math.floor(fy);
    let v = 0;
    for (let j = -1; j <= 2; j++) {
      const wy = cubic(fy - (iy + j));
      for (let i = -1; i <= 2; i++) v += wy * cubic(fx - (ix + i)) * at(ix + i, iy + j);
    }
    out[y * size + x] = Math.min(1, Math.max(0, v));
  }
  return out;
}

const CRC = (() => {
  const t = new Uint32Array(256);
  for (let n = 0; n < 256; n++) { let c = n; for (let k = 0; k < 8; k++) c = c & 1 ? 0xedb88320 ^ (c >>> 1) : c >>> 1; t[n] = c >>> 0; }
  return t;
})();
const crc32 = (bytes) => { let c = 0xffffffff; for (const b of bytes) c = CRC[(c ^ b) & 0xff] ^ (c >>> 8); return (c ^ 0xffffffff) >>> 0; };

function chunk(type, data) {
  const out = new Uint8Array(12 + data.length), dv = new DataView(out.buffer);
  dv.setUint32(0, data.length);
  out.set([...type].map((c) => c.charCodeAt(0)), 4);
  out.set(data, 8);
  dv.setUint32(8 + data.length, crc32(out.subarray(4, 8 + data.length)));
  return out;
}

export async function png16(h, n) {
  const raw = new Uint8Array(n * (1 + 2 * n)), v = toUint16(h);
  for (let y = 0; y < n; y++) {
    const row = y * (1 + 2 * n);
    raw[row] = 0;
    for (let x = 0; x < n; x++) { raw[row + 1 + 2 * x] = v[y * n + x] >> 8; raw[row + 2 + 2 * x] = v[y * n + x] & 255; }
  }
  const zipped = new Uint8Array(await new Response(new Blob([raw]).stream().pipeThrough(new CompressionStream("deflate"))).arrayBuffer());
  const ihdr = new Uint8Array(13), dv = new DataView(ihdr.buffer);
  dv.setUint32(0, n); dv.setUint32(4, n);
  ihdr.set([16, 0, 0, 0, 0], 8);  // 16-bit, grayscale, deflate, no filter, no interlace
  return new Blob([new Uint8Array([137, 80, 78, 71, 13, 10, 26, 10]), chunk("IHDR", ihdr), chunk("IDAT", zipped), chunk("IEND", new Uint8Array())],
    { type: "image/png" });
}

export function r16(h, n) {
  const size = unitySize(n), v = toUint16(size === n ? h : resample(h, n, size));
  const out = new DataView(new ArrayBuffer(v.length * 2));
  v.forEach((x, i) => out.setUint16(2 * i, x, true));
  return { blob: new Blob([out.buffer], { type: "application/octet-stream" }), size };
}

export function obj(h, n, world) {
  const cell = world.extent_m / n, z = (x, y) => h[y * n + x] * world.max_height_m;
  const lines = [`# Talus heightmap ${n}x${n}, cell ${cell.toFixed(3)} m`];
  for (let y = 0; y < n; y++) for (let x = 0; x < n; x++) lines.push(`v ${((x + 0.5) * cell).toFixed(3)} ${z(x, y).toFixed(3)} ${((y + 0.5) * cell).toFixed(3)}`);
  for (let y = 0; y < n; y++) for (let x = 0; x < n; x++) {
    // central differences like numpy.gradient (one-sided at the edges)
    const gx = (z(Math.min(n - 1, x + 1), y) - z(Math.max(0, x - 1), y)) / (cell * (x === 0 || x === n - 1 ? 1 : 2));
    const gy = (z(x, Math.min(n - 1, y + 1)) - z(x, Math.max(0, y - 1))) / (cell * (y === 0 || y === n - 1 ? 1 : 2));
    const l = Math.hypot(gx, 1, gy);
    lines.push(`vn ${(-gx / l).toFixed(5)} ${(1 / l).toFixed(5)} ${(-gy / l).toFixed(5)}`);
  }
  for (let y = 0; y < n - 1; y++) for (let x = 0; x < n - 1; x++) {
    const a = y * n + x + 1, b = a + 1, c = a + n, d = c + 1;
    lines.push(`f ${a}//${a} ${c}//${c} ${b}//${b}`, `f ${b}//${b} ${c}//${c} ${d}//${d}`);
  }
  return new Blob([lines.join("\n") + "\n"], { type: "model/obj" });
}

// files: { png16: { name, resolution }, r16: {...}, obj: {...} }. Samples sit at cell centers, so the span
// from the first to the last sample (footprint) is one cell less than the extent; resampling keeps it.
export function sidecar(n, world, files, source) {
  const footprint = (world.extent_m / n) * (n - 1);
  const perFile = Object.fromEntries(Object.entries(files).map(([k, f]) => [k, { ...f, cell_size_m: footprint / (f.resolution - 1) }]));
  return new Blob([JSON.stringify({
    extent_m: world.extent_m, footprint_m: footprint, max_height_m: world.max_height_m, min_height_m: 0,
    sea_level: world.sea_level, sea_level_m: world.sea_level * world.max_height_m,
    encoding: "uint16 = round(h * 65535); height_m = h * max_height_m", byte_order_r16: "little-endian",
    files: perFile, source,
  }, null, 2)], { type: "application/json" });
}

export function download(blob, name) {
  const a = document.createElement("a");
  a.href = URL.createObjectURL(blob);
  a.download = name;
  document.body.append(a);
  a.click();
  a.remove();
  setTimeout(() => URL.revokeObjectURL(a.href), 2000);
}
