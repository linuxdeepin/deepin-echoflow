// Generates the c2pa-audio browser demo (self-contained HTML) by inlining the
// exact library source. Each lib file is wrapped in a block scope (its top-level
// `const crypto`/helpers don't collide) and its exports are hoisted onto `LIB`.
import fs from 'fs';

const R = new URL('../js', import.meta.url).pathname;
const OUT = new URL('index.html', import.meta.url).pathname;

const strip = (s) => s.replace(/^export\s+/gm, '');
const cert = strip(fs.readFileSync(`${R}/default-cert.mjs`, 'utf8'));
const signer = strip(fs.readFileSync(`${R}/c2pa.mjs`, 'utf8'));
const verifier = strip(fs.readFileSync(`${R}/c2pa-verify.mjs`, 'utf8'));

const lib = `
const LIB = {};
{ ${cert}
  LIB.CERT = DEFAULT_CERT_PEM; LIB.KEY = DEFAULT_KEY_PEM; }
{ ${signer}
  LIB.signWav = c2paSignWav; LIB.signMp3 = c2paSignMp3; LIB.signM4a = c2paSignM4a; LIB.signFlac = c2paSignFlac; }
{ ${verifier}
  LIB.verify = c2paVerifyWav; }
`;

const html = String.raw`<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>c2pa-audio — sign &amp; verify Content Credentials in the browser</title>
<style>
  :root{
    --bg:#f3f5f7; --panel:#ffffff; --panel-2:#eef1f4; --line:#dde3e9;
    --ink:#131a22; --muted:#5c6875; --faint:#8a97a4;
    --accent:#0d9488; --accent-ink:#0a7d72;
    --good:#16a34a; --bad:#e11d48; --warn:#b45309;
    --mono:ui-monospace,"SF Mono","JetBrains Mono",Menlo,Consolas,monospace;
    --sans:system-ui,-apple-system,"Segoe UI",Roboto,sans-serif;
    --r:12px;
  }
  @media (prefers-color-scheme:dark){
    :root{
      --bg:#0b0f15; --panel:#121a23; --panel-2:#0e151d; --line:#202c38;
      --ink:#e7edf3; --muted:#93a2b1; --faint:#5f6f7e;
      --accent:#2dd4bf; --accent-ink:#5eead4;
      --good:#4ade80; --bad:#fb7185; --warn:#fbbf24;
    }
  }
  :root[data-theme="light"]{--bg:#f3f5f7;--panel:#fff;--panel-2:#eef1f4;--line:#dde3e9;--ink:#131a22;--muted:#5c6875;--faint:#8a97a4;--accent:#0d9488;--accent-ink:#0a7d72;--good:#16a34a;--bad:#e11d48;--warn:#b45309;}
  :root[data-theme="dark"]{--bg:#0b0f15;--panel:#121a23;--panel-2:#0e151d;--line:#202c38;--ink:#e7edf3;--muted:#93a2b1;--faint:#5f6f7e;--accent:#2dd4bf;--accent-ink:#5eead4;--good:#4ade80;--bad:#fb7185;--warn:#fbbf24;}
  *{box-sizing:border-box}
  body{margin:0;background:var(--bg);color:var(--ink);font-family:var(--sans);line-height:1.55;
    -webkit-font-smoothing:antialiased;font-size:15px}
  .wrap{max-width:860px;margin:0 auto;padding:44px 20px 80px}
  header.hero{margin-bottom:30px}
  .kicker{font-family:var(--mono);font-size:11px;letter-spacing:.16em;text-transform:uppercase;
    color:var(--accent-ink);display:flex;align-items:center;gap:8px}
  .kicker::before{content:"";width:26px;height:1px;background:var(--accent)}
  h1{font-size:34px;line-height:1.1;margin:.35em 0 .2em;letter-spacing:-.02em;text-wrap:balance;font-weight:650}
  h1 .lib{font-family:var(--mono);font-weight:600;color:var(--accent-ink)}
  .lede{color:var(--muted);max-width:60ch;margin:0}
  .card{background:var(--panel);border:1px solid var(--line);border-radius:var(--r);
    padding:22px 22px;margin-top:18px}
  .step-h{display:flex;align-items:baseline;gap:10px;margin:0 0 14px}
  .step-n{font-family:var(--mono);font-size:12px;color:var(--faint)}
  .step-h h2{font-size:15px;margin:0;font-weight:600;letter-spacing:-.01em}
  .row{display:flex;flex-wrap:wrap;gap:10px;align-items:center}
  button{font-family:var(--sans);font-size:14px;font-weight:550;cursor:pointer;
    border-radius:9px;border:1px solid var(--line);background:var(--panel-2);color:var(--ink);
    padding:9px 15px;transition:transform .08s ease,border-color .15s,background .15s}
  button:hover{border-color:var(--accent)}
  button:active{transform:translateY(1px)}
  button:disabled{opacity:.4;cursor:not-allowed}
  button.primary{background:var(--accent);border-color:var(--accent);color:#04231f}
  :root[data-theme="dark"] button.primary,@media (prefers-color-scheme:dark){button.primary{color:#04231f}}
  button:focus-visible{outline:2px solid var(--accent);outline-offset:2px}
  .hint{color:var(--faint);font-size:13px}
  .file-in{position:absolute;width:1px;height:1px;opacity:0;pointer-events:none}
  label.upload{display:inline-flex;align-items:center;gap:7px;padding:9px 15px;border:1px dashed var(--line);
    border-radius:9px;cursor:pointer;font-size:14px;font-weight:550}
  label.upload:hover{border-color:var(--accent)}
  canvas.wave{width:100%;height:56px;display:block;margin-top:14px;border-radius:8px;background:var(--panel-2)}
  .meta{font-family:var(--mono);font-size:12.5px;color:var(--muted);margin-top:12px;display:flex;flex-wrap:wrap;gap:6px 18px}
  .meta b{color:var(--ink);font-weight:600}
  .kv{display:grid;grid-template-columns:auto 1fr;gap:6px 16px;font-family:var(--mono);font-size:12.5px;margin-top:4px}
  .kv dt{color:var(--faint)}
  .kv dd{margin:0;color:var(--ink);word-break:break-all}
  .verdict{border-radius:10px;padding:16px 18px;margin-top:4px;border:1px solid var(--line);
    display:flex;gap:14px;align-items:flex-start;background:var(--panel-2)}
  .verdict.ok{border-color:color-mix(in oklab,var(--good) 45%,var(--line));background:color-mix(in oklab,var(--good) 9%,var(--panel))}
  .verdict.bad{border-color:color-mix(in oklab,var(--bad) 45%,var(--line));background:color-mix(in oklab,var(--bad) 9%,var(--panel))}
  .verdict .glyph{font-family:var(--mono);font-weight:700;font-size:20px;line-height:1.2;flex:none}
  .verdict.ok .glyph{color:var(--good)} .verdict.bad .glyph{color:var(--bad)}
  .verdict .title{font-weight:650;font-size:15px}
  .verdict .sub{color:var(--muted);font-size:13px;margin-top:2px}
  .checks{list-style:none;padding:0;margin:12px 0 0;display:grid;gap:7px}
  .checks li{display:flex;align-items:center;gap:9px;font-family:var(--mono);font-size:12.5px;color:var(--muted)}
  .checks .dot{width:8px;height:8px;border-radius:50%;flex:none;background:var(--faint)}
  .checks li.pass .dot{background:var(--good)} .checks li.fail .dot{background:var(--bad)}
  .checks li.pass{color:var(--ink)}
  .muted-note{color:var(--faint);font-size:12.5px;margin-top:12px;font-family:var(--mono)}
  .fade{animation:fade .35s ease both}
  @keyframes fade{from{opacity:0;transform:translateY(4px)}to{opacity:1;transform:none}}
  @media (prefers-reduced-motion:reduce){.fade{animation:none}}
  .grid2{display:grid;gap:18px;grid-template-columns:1fr 1fr}
  @media (max-width:640px){.grid2{grid-template-columns:1fr}}
  footer{margin-top:34px;color:var(--faint);font-size:13px;display:flex;flex-wrap:wrap;gap:6px 18px;font-family:var(--mono)}
  footer a{color:var(--accent-ink);text-decoration:none;border-bottom:1px solid transparent}
  footer a:hover{border-color:var(--accent)}
  .theme-btn{position:fixed;top:16px;right:16px;padding:7px 11px;font-family:var(--mono);font-size:12px}
  .disabled-note{opacity:.5}
  code{font-family:var(--mono);font-size:.92em;background:var(--panel-2);padding:1px 5px;border-radius:5px}
</style>

<button class="theme-btn" id="theme" title="Toggle theme">◐ theme</button>
<div class="wrap">
  <header class="hero">
    <div class="kicker">Content Credentials · pure WebCrypto</div>
    <h1><span class="lib">c2pa-audio</span> — sign &amp; verify audio provenance, right here in your browser</h1>
    <p class="lede">A live demo of the open-source library that embeds and checks C2PA (Content Credentials)
      manifests in audio — ES256 / CBOR / JUMBF / COSE, no server, no Rust, no <code>c2pa-rs</code>.
      Everything below runs on this page via the Web Crypto API. Output is interoperable with the
      c2pa-rs reference implementation.</p>
  </header>

  <section class="card">
    <div class="step-h"><span class="step-n">01</span><h2>Pick some audio</h2></div>
    <div class="row">
      <button class="primary" id="gen">Generate a 1&nbsp;s tone (WAV)</button>
      <label class="upload">Upload audio<input class="file-in" id="file" type="file"
        accept=".wav,.mp3,.m4a,.mp4,.flac,audio/*"></label>
      <span class="hint">WAV · MP3 · M4A/MP4 · FLAC</span>
    </div>
    <canvas class="wave" id="wave" width="820" height="56" aria-hidden="true"></canvas>
    <div class="meta" id="src-meta" hidden></div>
  </section>

  <section class="card disabled-note" id="sign-card">
    <div class="step-h"><span class="step-n">02</span><h2>Sign it with Content Credentials</h2></div>
    <div class="row">
      <button class="primary" id="sign" disabled>Sign</button>
      <button id="download" disabled>Download signed file</button>
      <span class="hint">Uses the library’s bundled self-signed cert.</span>
    </div>
    <dl class="kv" id="manifest" hidden></dl>
  </section>

  <section class="card disabled-note" id="verify-card">
    <div class="step-h"><span class="step-n">03</span><h2>Verify — and try to break it</h2></div>
    <div id="verdict-slot"></div>
    <div class="row" style="margin-top:14px">
      <button id="verify" disabled>Re-verify</button>
      <button id="tamper" disabled>Flip one audio byte ↯</button>
      <span class="hint">Tampering breaks the hard-binding hash — that’s the point.</span>
    </div>
  </section>

  <footer>
    <span>MIT · not affiliated with C2PA / CAI</span>
    <a href="https://github.com/CrispStrobe/c2pa-audio" target="_blank" rel="noopener">GitHub</a>
    <a href="https://pub.dev/packages/c2pa_audio" target="_blank" rel="noopener">pub.dev</a>
    <span>WAV/MP3/M4A/FLAC · C · JS · Dart · Python · Go · C#</span>
  </footer>
</div>

<script type="module">
/*__LIB__*/

// ---------------------------------------------------------------- demo state
const $ = (id) => document.getElementById(id);
const enc = new TextEncoder();
let current = null;   // { bytes: Uint8Array, name, mime, signed?:Uint8Array }
const MIME = { wav:'audio/wav', mp3:'audio/mpeg', m4a:'audio/mp4', mp4:'audio/mp4', flac:'audio/flac' };
const signerFor = { 'audio/wav':LIB.signWav, 'audio/mpeg':LIB.signMp3, 'audio/mp4':LIB.signM4a, 'audio/flac':LIB.signFlac };
const hex = (b, n) => [...b.subarray(0, n)].map((x) => x.toString(16).padStart(2, '0')).join(' ');

function toneWav(seconds = 1, sr = 24000, freq = 220) {
  const n = seconds * sr, buf = new ArrayBuffer(44 + n * 2), dv = new DataView(buf);
  const put = (o, s) => { for (let i = 0; i < s.length; i++) dv.setUint8(o + i, s.charCodeAt(i)); };
  put(0, 'RIFF'); dv.setUint32(4, 36 + n * 2, true); put(8, 'WAVE'); put(12, 'fmt ');
  dv.setUint32(16, 16, true); dv.setUint16(20, 1, true); dv.setUint16(22, 1, true);
  dv.setUint32(24, sr, true); dv.setUint32(28, sr * 2, true); dv.setUint16(32, 2, true);
  dv.setUint16(34, 16, true); put(36, 'data'); dv.setUint32(40, n * 2, true);
  for (let i = 0; i < n; i++) {
    const env = Math.min(1, i / 800, (n - i) / 800);
    dv.setInt16(44 + i * 2, Math.round(9000 * env * Math.sin((2 * Math.PI * freq * i) / sr) * (1 + 0.3 * Math.sin(2 * Math.PI * 3 * i / sr))), true);
  }
  return new Uint8Array(buf);
}

function drawWave(bytes, mark = -1) {
  const c = $('wave'), ctx = c.getContext('2d'), W = c.width, H = c.height;
  ctx.clearRect(0, 0, W, H);
  const css = getComputedStyle(document.documentElement);
  ctx.strokeStyle = css.getPropertyValue('--accent').trim() || '#0d9488';
  ctx.globalAlpha = 0.9; ctx.lineWidth = 1;
  ctx.beginPath();
  const step = Math.max(1, Math.floor(bytes.length / W));
  for (let x = 0; x < W; x++) {
    let lo = 255, hi = 0;
    for (let k = 0; k < step; k++) { const v = bytes[x * step + k] || 128; if (v < lo) lo = v; if (v > hi) hi = v; }
    const y0 = H / 2 - ((hi - 128) / 128) * (H / 2 - 3);
    const y1 = H / 2 - ((lo - 128) / 128) * (H / 2 - 3);
    ctx.moveTo(x + 0.5, y0); ctx.lineTo(x + 0.5, y1);
  }
  ctx.stroke();
  if (mark >= 0) {
    const x = Math.floor((mark / bytes.length) * W);
    ctx.globalAlpha = 1; ctx.fillStyle = css.getPropertyValue('--bad').trim() || '#e11d48';
    ctx.fillRect(x - 1, 0, 3, H);
  }
}

function loadAudio(bytes, name, mime) {
  current = { bytes, name, mime };
  drawWave(bytes);
  const m = $('src-meta'); m.hidden = false;
  m.innerHTML = '<span><b>' + name + '</b></span><span>' + mime + '</span><span>' + bytes.length.toLocaleString() + ' bytes</span><span>first bytes: <b>' + hex(bytes, 8) + '</b></span>';
  $('sign').disabled = false; $('sign-card').classList.remove('disabled-note');
  $('manifest').hidden = true; $('verdict-slot').innerHTML = '';
  $('verify').disabled = $('tamper').disabled = $('download').disabled = true;
  $('verify-card').classList.add('disabled-note');
}

function detectMime(name, bytes) {
  const ext = (name.split('.').pop() || '').toLowerCase();
  if (MIME[ext]) return MIME[ext];
  const s = String.fromCharCode(...bytes.subarray(0, 4));
  if (s === 'RIFF') return 'audio/wav';
  if (s === 'fLaC') return 'audio/flac';
  if (s.startsWith('ID3') || (bytes[0] === 0xff && (bytes[1] & 0xe0) === 0xe0)) return 'audio/mpeg';
  if (String.fromCharCode(...bytes.subarray(4, 8)) === 'ftyp') return 'audio/mp4';
  return 'audio/wav';
}

async function sign() {
  if (!current) return;
  const fn = signerFor[current.mime];
  if (!fn) { alert('Unsupported container for signing.'); return; }
  $('sign').disabled = true; $('sign').textContent = 'Signing…';
  try {
    current.signed = await fn(current.bytes, LIB.CERT, LIB.KEY);
    showManifest(current.signed);
    $('download').disabled = false;
    await verify();
    $('verify').disabled = $('tamper').disabled = false;
    $('verify-card').classList.remove('disabled-note');
  } catch (e) { alert('Sign failed: ' + e.message); }
  $('sign').disabled = false; $('sign').textContent = 'Sign';
}

async function showManifest(signed) {
  const r = await LIB.verify(signed);
  const man = r.manifest || {};
  const act = (man.actions || [])[0] || {};
  const dl = $('manifest'); dl.hidden = false;
  const grew = signed.length - current.bytes.length;
  dl.innerHTML = [
    ['manifest', 'C2PA v2 · JUMBF store (+' + grew.toLocaleString() + ' bytes)'],
    ['generator', man.generatorName || '—'],
    ['action', (act.action || '—') + '  ·  ' + ((act.digitalSourceType || '').split('/').pop() || '—')],
    ['signature', 'COSE_Sign1 · ES256 (ECDSA P-256)'],
    ['signer', 'self-signed (untrusted anchor — marks provenance, not identity)'],
  ].map(([k, v]) => '<dt>' + k + '</dt><dd>' + v + '</dd>').join('');
}

async function verify() {
  const r = await LIB.verify(current.signed);
  const ok = r.valid;
  const slot = $('verdict-slot');
  const checks = [
    ['COSE ES256 signature', r.signatureValid],
    ['hard binding (audio hash)', r.dataHashValid],
    ['assertion hashes', r.assertionsValid],
  ];
  slot.innerHTML =
    '<div class="verdict ' + (ok ? 'ok' : 'bad') + ' fade">' +
      '<div class="glyph">' + (ok ? '✓' : '✗') + '</div>' +
      '<div><div class="title">' + (ok ? 'Valid Content Credentials' : 'Tampered — verification failed') + '</div>' +
        '<div class="sub">' + (ok
          ? 'Verified in-browser against the embedded certificate. (Self-signed ⇒ “untrusted signer”, as expected.)'
          : 'The audio no longer matches the signed hash — exactly what a manifest is supposed to catch.') + '</div>' +
        '<ul class="checks">' + checks.map(([label, pass]) =>
          '<li class="' + (pass ? 'pass' : 'fail') + '"><span class="dot"></span>' + label + '  ' + (pass ? 'ok' : 'FAIL') + '</li>').join('') +
        '</ul>' +
      '</div>' +
    '</div>';
}

function tamper() {
  if (!current || !current.signed) return;
  // flip a byte deep in the audio payload (past the prepended/embedded manifest)
  const i = Math.floor(current.signed.length * 0.82);
  current.signed[i] ^= 0xff;
  drawWave(current.signed, i);
  verify();
  $('tamper').disabled = true; $('tamper').textContent = 'Byte flipped ↯';
}

$('gen').onclick = () => loadAudio(toneWav(), 'tone.wav', 'audio/wav');
$('file').onchange = async (e) => {
  const f = e.target.files[0]; if (!f) return;
  const bytes = new Uint8Array(await f.arrayBuffer());
  loadAudio(bytes, f.name, detectMime(f.name, bytes));
};
$('sign').onclick = sign;
$('verify').onclick = verify;
$('tamper').onclick = tamper;
$('download').onclick = () => {
  const blob = new Blob([current.signed], { type: current.mime });
  const a = document.createElement('a'); a.href = URL.createObjectURL(blob);
  a.download = (current.name.replace(/\.[^.]+$/, '')) + '.signed' + (current.name.match(/\.[^.]+$/) || ['.wav'])[0];
  a.click(); URL.revokeObjectURL(a.href);
};
$('theme').onclick = () => {
  const cur = document.documentElement.getAttribute('data-theme')
    || (matchMedia('(prefers-color-scheme:dark)').matches ? 'dark' : 'light');
  document.documentElement.setAttribute('data-theme', cur === 'dark' ? 'light' : 'dark');
  if (current) drawWave(current.signed || current.bytes);
};

// start with something on screen
loadAudio(toneWav(), 'tone.wav', 'audio/wav');
</script>`;

fs.writeFileSync(OUT, html.replace('/*__LIB__*/', lib));
console.log('wrote', OUT, fs.statSync(OUT).size, 'bytes');
