// Minimal static server that sets the cross-origin-isolation headers CrispASR's multithreaded WASM
// needs (SharedArrayBuffer). COEP `credentialless` lets the cross-origin HuggingFace model fetch work
// without the CDN having to send CORP. Run:  node server.mjs   →  http://localhost:8791
import http from 'node:http';
import { readFile } from 'node:fs/promises';
import path from 'node:path';
import { fileURLToPath } from 'node:url';

const ROOT = path.dirname(fileURLToPath(import.meta.url));
const TYPES = {'.html': 'text/html', '.js': 'text/javascript', '.wasm': 'application/wasm', '.json': 'application/json'};

const ts = () => new Date().toISOString().slice(11, 23);
http.createServer(async (req, res) => {
    const t0 = Date.now();
    let rel = decodeURIComponent(req.url.split('?')[0]);
    if (rel === '/') rel = '/index.html';
    const file = path.join(ROOT, path.normalize(rel));
    if (!file.startsWith(ROOT)) { console.log(`${ts()} 403 ${req.url}`); res.writeHead(403); res.end(); return; }
    try {
        const data = await readFile(file);
        // no-store so the browser always re-fetches index.html / the worker / the loader while iterating
        const noCache = /\.(html|js|mjs)$/.test(file);
        res.writeHead(200, {
            'Content-Type': TYPES[path.extname(file)] || 'application/octet-stream',
            ...(process.env.NO_COI ? {} : {'Cross-Origin-Opener-Policy': 'same-origin'}),
            // require-corp works in every browser (incl. Safari); all resources here are same-origin.
            ...(process.env.NO_COI ? {} : {'Cross-Origin-Embedder-Policy': 'require-corp'}),
            'Cross-Origin-Resource-Policy': 'same-origin',
            ...(noCache ? {'Cache-Control': 'no-store'} : {})
        });
        res.end(data);
        console.log(`${ts()} 200 ${req.url}  (${(data.length / 1024 | 0)} KB, ${Date.now() - t0}ms)`);
    } catch {
        console.log(`${ts()} 404 ${req.url}`);
        res.writeHead(404); res.end('not found');
    }
}).listen(8791, () => console.log(`${ts()} Brickwright TTS demo → http://localhost:8791  (logging every request)`));
