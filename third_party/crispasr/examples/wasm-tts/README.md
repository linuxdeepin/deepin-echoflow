# CrispASR WASM TTS demo (on-device, multithreaded)

Fully client-side neural TTS (kokoro) running in the browser via CrispASR's WebAssembly build —
no cloud, no API key. Synthesis runs **multithreaded** on a proxied pthread (`PROXY_TO_PTHREAD`),
driven from a Web Worker so the main thread never blocks.

This directory holds only the small source files. The WASM build and the models are produced/fetched
locally (git-ignored).

## 1. Build the WASM

From the repo root, build the proxy-to-pthread variant (multithreaded, needs COOP/COEP):

```sh
source /path/to/emsdk/emsdk_env.sh
./build-wasm.sh --clean --proxy-to-pthread
cp build-wasm/bin/libwhisper.js build-wasm/bin/libwhisper.wasm examples/wasm-tts/
```

(A `--single-thread` build also works — slower, but needs no COOP/COEP. Then drop the headers in
`server.mjs`.)

## 2. Fetch a model + voice + G2P dict

Into this directory:

- `kokoro.gguf` — the kokoro TTS model
- `voice.gguf` — an English voice (e.g. af_heart); `voice_de.gguf` for German (df_eva)
- `cmudict.dict` — English G2P dictionary; **must** be loaded so pronunciation is correct
- German also needs `olaph_de.txt` + `espeak_de.tsv`

The worker writes `cmudict.dict` to `/home/web_user/.cache/crispasr/` (kokoro's G2P cache path) — do
not change that path or pronunciation garbles.

## 3. Run

```sh
node server.mjs           # serves with COOP/COEP → http://localhost:8791
# or, headless self-test (puppeteer-core + local Chrome):
node test-browser.mjs
```

Open the page, pick a voice, press **Speak**. First synth includes a one-time cmudict load
(~a few seconds); subsequent synths are faster.

## How it works

- `tts-worker.js` — the driver Web Worker. `if (self.name === 'em-pthread') importScripts(loader)`
  guard + instantiate the module at top level (both avoid the Emscripten-in-a-worker bootstrap
  deadlock). Calls `ttsOpenExplicit(model, 'kokoro', N)` then `ttsSynthesizeAsync(text, cb)` — the
  async entry proxies the blocking compute onto the runtime pthread, so the servicer worker stays
  free and ggml's compute threads run without the `pthread_join` deadlock.
- `index.html` — plays the returned Float32Array (24 kHz mono) via Web Audio.
- `server.mjs` — static server sending `Cross-Origin-Opener-Policy: same-origin` +
  `Cross-Origin-Embedder-Policy: require-corp` (SharedArrayBuffer needs cross-origin isolation).
