// Local static server for dist/ (no special headers, like GitHub Pages). usage: node serve.mjs [port]
import { createReadStream, existsSync, statSync } from "node:fs";
import { createServer } from "node:http";
import { extname, join, normalize } from "node:path";
import { fileURLToPath } from "node:url";

const root = fileURLToPath(new URL("dist/", import.meta.url)), port = +(process.argv[2] || 4173);
const TYPES = { ".html": "text/html; charset=utf-8", ".css": "text/css", ".js": "text/javascript", ".mjs": "text/javascript",
  ".json": "application/json", ".png": "image/png", ".woff2": "font/woff2", ".wasm": "application/wasm", ".svg": "image/svg+xml",
  ".txt": "text/plain; charset=utf-8", ".xml": "application/xml" };

createServer((req, res) => {
  const path = decodeURIComponent(new URL(req.url, "http://x").pathname);
  let file = normalize(join(root, path));
  if (!file.startsWith(root)) { res.writeHead(403).end(); return; }
  if (existsSync(file) && statSync(file).isDirectory()) file = join(file, "index.html");
  if (!existsSync(file)) { res.writeHead(404, { "content-type": "text/plain" }).end("not found"); return; }
  res.writeHead(200, { "content-type": TYPES[extname(file)] || "application/octet-stream", "content-length": statSync(file).size });
  createReadStream(file).pipe(res);
}).listen(port, "127.0.0.1", () => console.log(`http://127.0.0.1:${port}/`));
