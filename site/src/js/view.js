// 1-bit terrain viewer: a WebGL2 block diagram of one heightmap, lit, contoured every 50 m and dithered against
// blue noise into two colors at a low internal resolution. Heights live in a texture, so live denoising only
// uploads 64x64 floats per step. Morphs between maps by blending two height textures.

const VS = `#version 300 es
precision highp float;
precision highp int;
uniform sampler2D uA, uB;
uniform float uMix, uK, uSea, uBase;
uniform mat4 uMVP;
uniform int uN;
out vec3 vNormal;
out float vH, vWater, vSide;
float H(ivec2 p) {
  p = clamp(p, ivec2(0), ivec2(uN - 1));
  return mix(texelFetch(uA, p, 0).r, texelFetch(uB, p, 0).r, uMix);
}
vec3 pos(ivec2 p, float h) { return vec3(vec2(p) / float(uN - 1) - 0.5, max(h, uSea) * uK).xzy; }
void main() {
  int n = uN * uN, id = gl_VertexID;
  if (id < n) {
    ivec2 p = ivec2(id % uN, id / uN);
    float h = H(p), step = 2.0 / float(uN - 1);
    float dx = (max(H(p + ivec2(1, 0)), uSea) - max(H(p - ivec2(1, 0)), uSea)) * uK / step;
    float dz = (max(H(p + ivec2(0, 1)), uSea) - max(H(p - ivec2(0, 1)), uSea)) * uK / step;
    vNormal = normalize(vec3(-dx, 1.0, -dz));
    vH = h; vWater = h < uSea ? 1.0 : 0.0; vSide = 0.0;
    gl_Position = uMVP * vec4(pos(p, h), 1.0);
    return;
  }
  // skirt: the four sides of the block, top edge on the terrain, bottom edge at uBase
  int s = id - n, k = s / 2, side = k / uN, i = k % uN;
  ivec2 p = side == 0 ? ivec2(i, 0) : side == 1 ? ivec2(uN - 1, i) : side == 2 ? ivec2(uN - 1 - i, uN - 1) : ivec2(0, uN - 1 - i);
  vec3 q = pos(p, H(p));
  if (s % 2 == 1) q.y = uBase;
  vNormal = side == 0 ? vec3(0, 0, -1) : side == 1 ? vec3(1, 0, 0) : side == 2 ? vec3(0, 0, 1) : vec3(-1, 0, 0);
  vH = H(p); vWater = 0.0; vSide = 1.0;
  gl_Position = uMVP * vec4(q, 1.0);
}`;

const FS = `#version 300 es
precision highp float;
uniform sampler2D uNoise;
uniform vec3 uInk, uBg, uLight;
uniform float uMaxH, uSea;
in vec3 vNormal;
in float vH, vWater, vSide;
out vec4 color;
void main() {
  float t = texelFetch(uNoise, ivec2(gl_FragCoord.xy) & 63, 0).r;
  vec3 n = normalize(vNormal);
  float lum;
  if (vSide > 0.5) {
    lum = 0.06 + 0.32 * max(dot(n, uLight), 0.0);
  } else if (vWater > 0.5) {
    float depth = clamp((uSea - vH) * 12.0, 0.0, 1.0);
    lum = mix(0.30, 0.04, depth);
  } else {
    lum = 0.05 + 0.95 * pow(max(dot(n, uLight), 0.0), 1.4);
    float m = vH * uMaxH / 50.0, w = fwidth(m);
    float line = abs(fract(m + 0.5) - 0.5) / max(w, 1e-4);
    if (line < 0.6 && w < 0.5) lum = mix(lum, lum > 0.5 ? 0.0 : 1.0, 0.85);
    float shore = (vH - uSea) * uMaxH;
    if (shore < 6.0) lum = 1.0;
  }
  color = vec4(lum > t ? uInk : uBg, 1.0);
}`;

function compile(gl, type, src) {
  const s = gl.createShader(type);
  gl.shaderSource(s, src);
  gl.compileShader(s);
  if (!gl.getShaderParameter(s, gl.COMPILE_STATUS)) throw new Error(gl.getShaderInfoLog(s));
  return s;
}

function perspective(fov, aspect, near, far) {
  const f = 1 / Math.tan(fov / 2), nf = 1 / (near - far);
  return [f / aspect, 0, 0, 0, 0, f, 0, 0, 0, 0, (far + near) * nf, -1, 0, 0, 2 * far * near * nf, 0];
}
function lookAt(eye, at) {
  const sub = (a, b) => a.map((v, i) => v - b[i]), norm = (a) => { const l = Math.hypot(...a); return a.map((v) => v / l); };
  const cross = (a, b) => [a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2], a[0] * b[1] - a[1] * b[0]];
  const dot = (a, b) => a[0] * b[0] + a[1] * b[1] + a[2] * b[2];
  const z = norm(sub(eye, at)), x = norm(cross([0, 1, 0], z)), y = cross(z, x);
  return [x[0], y[0], z[0], 0, x[1], y[1], z[1], 0, x[2], y[2], z[2], 0, -dot(x, eye), -dot(y, eye), -dot(z, eye), 1];
}
function mul(a, b) {
  const o = new Array(16).fill(0);
  for (let c = 0; c < 4; c++) for (let r = 0; r < 4; r++) for (let k = 0; k < 4; k++) o[c * 4 + r] += a[k * 4 + r] * b[c * 4 + k];
  return o;
}

export class TerrainView {
  constructor(canvas, { noise, world, exaggeration = 2.2, autoRotate = true, texels = 480 }) {
    this.canvas = canvas;
    this.world = world;
    this.N = world.resolution;
    this.exaggeration = exaggeration;
    this.texels = texels;
    this.reduce = matchMedia("(prefers-reduced-motion: reduce)").matches;
    this.autoRotate = autoRotate && !this.reduce;
    this.yaw = -0.75; this.pitch = 0.62; this.dist = 1.85;
    this.mix = 1; this.morph = null; this.visible = true; this.dirty = true;
    const gl = canvas.getContext("webgl2", { antialias: false, alpha: false, preserveDrawingBuffer: true });
    if (!gl) throw new Error("WebGL2 is not available");
    this.gl = gl;
    const prog = gl.createProgram();
    gl.attachShader(prog, compile(gl, gl.VERTEX_SHADER, VS));
    gl.attachShader(prog, compile(gl, gl.FRAGMENT_SHADER, FS));
    gl.linkProgram(prog);
    if (!gl.getProgramParameter(prog, gl.LINK_STATUS)) throw new Error(gl.getProgramInfoLog(prog));
    this.prog = prog;
    this.u = Object.fromEntries(["uA", "uB", "uMix", "uK", "uSea", "uBase", "uMVP", "uN", "uNoise", "uInk", "uBg", "uLight", "uMaxH"]
      .map((n) => [n, gl.getUniformLocation(prog, n)]));
    const N = this.N, idx = [];
    for (let y = 0; y < N - 1; y++) for (let x = 0; x < N - 1; x++) {
      const a = y * N + x, b = a + 1, c = a + N, d = c + 1;
      idx.push(a, c, b, b, c, d);
    }
    for (let k = 0; k < 4 * N - 1; k++) {
      const t0 = N * N + 2 * k, t1 = t0 + 2;
      idx.push(t0, t0 + 1, t1, t1, t0 + 1, t1 + 1);
    }
    this.count = idx.length;
    this.vao = gl.createVertexArray();
    gl.bindVertexArray(this.vao);
    const ib = gl.createBuffer();
    gl.bindBuffer(gl.ELEMENT_ARRAY_BUFFER, ib);
    gl.bufferData(gl.ELEMENT_ARRAY_BUFFER, new Uint32Array(idx), gl.STATIC_DRAW);
    const tex = (format, type, w, data, internal) => {
      const t = gl.createTexture();
      gl.bindTexture(gl.TEXTURE_2D, t);
      gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MIN_FILTER, gl.NEAREST);
      gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MAG_FILTER, gl.NEAREST);
      gl.texImage2D(gl.TEXTURE_2D, 0, internal, w, w, 0, format, type, data);
      return t;
    };
    const flat = new Float32Array(N * N).fill(world.sea_level);
    this.texA = tex(gl.RED, gl.FLOAT, N, flat, gl.R32F);
    this.texB = tex(gl.RED, gl.FLOAT, N, flat, gl.R32F);
    this.texNoise = tex(gl.RED, gl.UNSIGNED_BYTE, 64, noise, gl.R8);
    this.current = flat;
    this.setColors([242, 242, 242], [4, 4, 4]);
    this.bindPointer();
    this.observer = new ResizeObserver(() => this.resize());
    this.observer.observe(canvas);
    new IntersectionObserver(([e]) => { this.visible = e.isIntersecting; if (this.visible) this.kick(); }).observe(canvas);
    this.resize();
  }

  setColors(ink, bg) { this.ink = ink.map((v) => v / 255); this.bg = bg.map((v) => v / 255); this.dirty = true; this.kick(); }

  resize() {
    // internal resolution of about `texels` across, in whole device pixels, for the chunky 1-bit look
    const r = this.canvas.getBoundingClientRect(), dpr = Math.min(2, devicePixelRatio || 1);
    const scale = Math.max(2, Math.round((r.width * dpr) / this.texels));
    this.canvas.width = Math.max(1, Math.round((r.width * dpr) / scale));
    this.canvas.height = Math.max(1, Math.round((r.height * dpr) / scale));
    this.dirty = true;
    this.kick();
  }

  upload(tex, h) {
    const gl = this.gl;
    gl.bindTexture(gl.TEXTURE_2D, tex);
    gl.texSubImage2D(gl.TEXTURE_2D, 0, 0, 0, this.N, this.N, gl.RED, gl.FLOAT, h);
  }

  // Show heights immediately, or morph to them over `ms` milliseconds.
  set(h, ms = 0) {
    const next = Float32Array.from(h);
    if (!ms || this.reduce) {
      this.upload(this.texB, next);
      this.mix = 1; this.morph = null;
    } else {
      this.upload(this.texA, this.current);
      this.upload(this.texB, next);
      this.mix = 0; this.morph = { t0: performance.now(), ms };
    }
    this.current = next;
    this.dirty = true;
    this.kick();
  }

  bindPointer() {
    const c = this.canvas;
    let drag = null;
    c.addEventListener("pointerdown", (e) => { drag = { x: e.clientX, y: e.clientY, yaw: this.yaw, pitch: this.pitch }; c.setPointerCapture(e.pointerId); this.autoRotate = false; });
    c.addEventListener("pointermove", (e) => {
      if (!drag) return;
      this.yaw = drag.yaw - (e.clientX - drag.x) * 0.008;
      this.pitch = Math.min(1.35, Math.max(0.18, drag.pitch + (e.clientY - drag.y) * 0.006));
      this.dirty = true; this.kick();
    });
    const end = () => { drag = null; };
    c.addEventListener("pointerup", end);
    c.addEventListener("pointercancel", end);
    c.addEventListener("keydown", (e) => {
      const d = { ArrowLeft: [0.12, 0], ArrowRight: [-0.12, 0], ArrowUp: [0, 0.08], ArrowDown: [0, -0.08] }[e.key];
      if (!d) return;
      e.preventDefault();
      this.autoRotate = false;
      this.yaw += d[0]; this.pitch = Math.min(1.35, Math.max(0.18, this.pitch + d[1]));
      this.dirty = true; this.kick();
    });
  }

  kick() {
    if (!this.visible) {  // offscreen: keep the last frame current (one draw), skip the animation loop
      if (this.dirty) { if (this.morph) { this.mix = 1; this.morph = null; } this.draw(); this.dirty = false; }
      return;
    }
    if (!this.raf) this.raf = requestAnimationFrame((t) => this.frame(t));
  }

  frame(time) {
    this.raf = 0;
    const dt = this.last ? Math.min(0.1, (time - this.last) / 1000) : 0;
    this.last = time;
    if (this.morph) {
      const p = Math.min(1, (time - this.morph.t0) / this.morph.ms);
      this.mix = p < 1 ? Math.floor(p * 12) / 12 : 1;  // stepped, like the rest of the Tersa motion
      if (p >= 1) this.morph = null;
      this.dirty = true;
    }
    if (this.autoRotate) { this.yaw += dt * 0.12; this.dirty = true; }
    if (this.dirty) this.draw();
    this.dirty = false;
    if (this.morph || this.autoRotate) this.kick(); else this.last = 0;
  }

  draw() {
    const gl = this.gl, u = this.u, w = this.canvas.width, h = this.canvas.height, world = this.world;
    gl.viewport(0, 0, w, h);
    gl.clearColor(...this.bg, 1);
    gl.clear(gl.COLOR_BUFFER_BIT | gl.DEPTH_BUFFER_BIT);
    gl.enable(gl.DEPTH_TEST);
    gl.useProgram(this.prog);
    gl.bindVertexArray(this.vao);
    const k = (world.max_height_m / world.extent_m) * this.exaggeration;
    const eye = [Math.cos(this.yaw) * Math.cos(this.pitch) * this.dist, Math.sin(this.pitch) * this.dist, Math.sin(this.yaw) * Math.cos(this.pitch) * this.dist];
    const aspect = w / h, fov = aspect < 1 ? 0.62 / aspect ** 0.5 : 0.62;
    gl.uniformMatrix4fv(u.uMVP, false, mul(perspective(fov, aspect, 0.1, 10), lookAt(eye, [0, 0.05, 0])));
    gl.uniform1f(u.uMix, this.mix);
    gl.uniform1f(u.uK, k);
    gl.uniform1f(u.uSea, world.sea_level);
    gl.uniform1f(u.uBase, -0.06);
    gl.uniform1f(u.uMaxH, world.max_height_m);
    gl.uniform1i(u.uN, this.N);
    gl.uniform3fv(u.uInk, this.ink);
    gl.uniform3fv(u.uBg, this.bg);
    const L = [-0.5, 0.75, -0.42], l = Math.hypot(...L);
    gl.uniform3fv(u.uLight, L.map((v) => v / l));
    [this.texA, this.texB, this.texNoise].forEach((t, i) => { gl.activeTexture(gl.TEXTURE0 + i); gl.bindTexture(gl.TEXTURE_2D, t); });
    gl.uniform1i(u.uA, 0); gl.uniform1i(u.uB, 1); gl.uniform1i(u.uNoise, 2);
    gl.drawElements(gl.TRIANGLES, this.count, gl.UNSIGNED_INT, 0);
  }
}

// Grayscale top-down heightmap, the same pixels the PNG export holds.
export function drawHeightmap(canvas, h, N) {
  canvas.width = N; canvas.height = N;
  const ctx = canvas.getContext("2d"), img = ctx.createImageData(N, N);
  for (let i = 0; i < N * N; i++) {
    const v = Math.round(Math.min(1, Math.max(0, h[i])) * 255);
    img.data[i * 4] = img.data[i * 4 + 1] = img.data[i * 4 + 2] = v;
    img.data[i * 4 + 3] = 255;
  }
  ctx.putImageData(img, 0, 0);
}
