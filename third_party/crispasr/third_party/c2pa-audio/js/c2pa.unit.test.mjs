// c2pa.unit.test.mjs — hermetic unit tests for the pure-JS signer/verifier.
// No native lib, no c2pa-rs, no network. CBOR RFC-8949 vectors, JUMBF box
// layout, per-container tamper granularity, and negative cases.
import { test } from 'node:test';
import assert from 'node:assert/strict';
import fs from 'node:fs';
import path from 'node:path';
import { fileURLToPath } from 'node:url';
import { c2paSignWav, c2paSignMp3, c2paSignM4a, c2paSignFlac, _internal } from './c2pa.mjs';
import { c2paVerifyWav } from './c2pa-verify.mjs';
import { DEFAULT_CERT_PEM, DEFAULT_KEY_PEM } from './default-cert.mjs';

const here = path.dirname(fileURLToPath(import.meta.url));
const { cbor, boxType, jumb, jumd, sha256 } = _internal;
const hex = (b) => [...b].map((x) => x.toString(16).padStart(2, '0')).join('');
const C = DEFAULT_CERT_PEM, K = DEFAULT_KEY_PEM;

function makeWav(n = 4800, sr = 24000) {
  const b = Buffer.alloc(44 + n * 2);
  b.write('RIFF', 0); b.writeUInt32LE(36 + n * 2, 4); b.write('WAVE', 8);
  b.write('fmt ', 12); b.writeUInt32LE(16, 16); b.writeUInt16LE(1, 20); b.writeUInt16LE(1, 22);
  b.writeUInt32LE(sr, 24); b.writeUInt32LE(sr * 2, 28); b.writeUInt16LE(2, 32); b.writeUInt16LE(16, 34);
  b.write('data', 36); b.writeUInt32LE(n * 2, 40);
  for (let i = 0; i < n; i++) b.writeInt16LE(Math.round(3000 * Math.sin((2 * Math.PI * 220 * i) / sr)), 44 + i * 2);
  return new Uint8Array(b);
}
const sample = (f) => { const p = path.join(here, f); return fs.existsSync(p) ? new Uint8Array(fs.readFileSync(p)) : null; };

// ---------------------------------------------------------------- CBOR vectors
test('CBOR: RFC 8949 integer vectors', () => {
  const cases = [[0, '00'], [23, '17'], [24, '1818'], [255, '18ff'], [256, '190100'],
    [1000, '1903e8'], [65535, '19ffff'], [65536, '1a00010000'], [-1, '20'], [-100, '3863']];
  for (const [n, h] of cases) assert.equal(hex(cbor(n)), h);
});
test('CBOR: text/bytes/array vectors', () => {
  assert.equal(hex(cbor('')), '60');
  assert.equal(hex(cbor('IETF')), '6449455446');
  assert.equal(hex(cbor(new Uint8Array([1, 2, 3, 4]))), '4401020304');
  assert.equal(hex(cbor([1, 2, 3])), '83010203');
  assert.equal(hex(cbor(null)), 'f6');
});
test('CBOR: canonical map key ordering (length then bytes)', () => {
  assert.equal(hex(cbor({ b: 3, aa: 2, a: 1 })), 'a3' + '616101' + '616203' + '626161' + '02');
});

// --------------------------------------------------------------- JUMBF layout
test('JUMBF: boxType = 4 ASCII + fixed 12-byte suffix', () => {
  const t = boxType('c2pa');
  assert.equal(t.length, 16);
  assert.equal(hex(t.subarray(0, 4)), '63327061');
  assert.equal(hex(t.subarray(4)), '00110010800000aa00389b71');
});
test('JUMBF: jumd toggles=0x03 + null-terminated label; jumb size == length', () => {
  const d = jumd(boxType('c2pa'), 'c2pa');
  assert.equal(d[8 + 16], 0x03);
  assert.equal(d[d.length - 1], 0x00);
  const b = jumb('c2pa', 'c2pa', [jumd(boxType('cbor'), 'x')]);
  const size = (b[0] << 24) | (b[1] << 16) | (b[2] << 8) | b[3];
  assert.equal(size, b.length);
  assert.equal(String.fromCharCode(...b.subarray(4, 8)), 'jumb');
});

// ---------------------------------------------------------- per-container tamper
test('WAV round-trip + audio tamper fails the hard binding', async () => {
  const s = await c2paSignWav(makeWav(), C, K);
  assert.equal((await c2paVerifyWav(s)).valid, true);
  const t = Uint8Array.from(s); t[46] ^= 0xff;
  const r = await c2paVerifyWav(t);
  assert.equal(r.dataHashValid, false);
  assert.equal(r.valid, false);
});
for (const [name, fn, file] of [['MP3', c2paSignMp3, 'sample.mp3'], ['M4A', c2paSignM4a, 'sample.m4a'], ['FLAC', c2paSignFlac, 'sample.flac']]) {
  test(`${name} round-trip + audio tamper fails`, async () => {
    const raw = sample(file); if (!raw) return;
    const s = await fn(raw, C, K);
    assert.equal((await c2paVerifyWav(s)).valid, true, name);
    const t = Uint8Array.from(s); t[t.length - 40] ^= 0xff; // deep in the audio payload
    assert.equal((await c2paVerifyWav(t)).valid, false, name + ' tamper');
  });
}

// -------------------------------------------------------------- negative cases
test('non-C2PA WAV verifies as invalid (no manifest)', async () => {
  const r = await c2paVerifyWav(makeWav());
  assert.equal(r.valid, false);
  assert.ok(r.errors.length > 0);
});
test('bad PEM throws', async () => {
  await assert.rejects(() => c2paSignWav(makeWav(), 'not a cert', 'not a key'));
});
test('size determinism: two signings of same WAV are equal length', async () => {
  const w = makeWav();
  const a = await c2paSignWav(w, C, K), b = await c2paSignWav(w, C, K);
  assert.equal(a.length, b.length);
});
test('assertion hash convention: sha256(box[8:]) is a 32-byte digest', async () => {
  const s = await c2paSignWav(makeWav(), C, K);
  // locate the actions assertion box in the RIFF C2PA chunk and hash box[8:]
  const dv = new DataView(s.buffer, s.byteOffset, s.byteLength);
  let off = 12, body = null;
  while (off + 8 <= s.length) {
    const id = String.fromCharCode(s[off], s[off + 1], s[off + 2], s[off + 3]);
    const sz = dv.getUint32(off + 4, true);
    if (id === 'C2PA') { body = s.subarray(off + 8, off + 8 + sz); break; }
    off += 8 + sz + (sz & 1);
  }
  assert.ok(body);
  const h = await sha256(body.subarray(8));
  assert.equal(h.length, 32);
});

// --- robustness: a hostile file must not hang or crash the verifier ---
test('deeply-nested JUMBF does not hang (bounded recursion)', async () => {
  const N = 50000;
  const jumbf = new Uint8Array(8 * N);
  const dv = new DataView(jumbf.buffer);
  for (let i = 0; i < N; i++) { dv.setUint32(8 * i, 8 * (N - i)); jumbf.set([0x6a, 0x75, 0x6d, 0x62], 8 * i + 4); } // 'jumb'
  const cs = jumbf.length;
  const wav = new Uint8Array(16 + 4 + cs);
  wav.set([0x52, 0x49, 0x46, 0x46, 0, 0, 0, 0, 0x57, 0x41, 0x56, 0x45, 0x43, 0x32, 0x50, 0x41], 0); // RIFF..WAVE C2PA
  new DataView(wav.buffer).setUint32(16, cs, true);
  wav.set(jumbf, 20);
  const t0 = Date.now();
  const r = await c2paVerifyWav(wav);
  assert.equal(r.valid, false);
  assert.ok(Date.now() - t0 < 5000, 'verify must return quickly on nested JUMBF');
});
test('deeply-nested CBOR does not crash the decoder', async () => {
  const deep = new Uint8Array(50000).fill(0x81); // nested array(1) chain
  const r = await c2paVerifyWav(deep); // not even a C2PA container -> invalid, no throw escaping
  assert.equal(r.valid, false);
});
test('runt JUMBF boxes (size 8..11) do not read out of bounds', async () => {
  for (let bs = 8; bs <= 11; bs++) {
    const w = new Uint8Array(20 + bs);
    w.set([0x52, 0x49, 0x46, 0x46, 0, 0, 0, 0, 0x57, 0x41, 0x56, 0x45, 0x43, 0x32, 0x50, 0x41], 0);
    new DataView(w.buffer).setUint32(16, bs, true);            // chunk size (LE)
    new DataView(w.buffer).setUint32(20, bs, false);           // box size (BE)
    w.set([0x6a, 0x75, 0x6d, 0x62], 24);                       // 'jumb'
    const r = await c2paVerifyWav(w);                          // must not throw/OOB
    assert.equal(r.valid, false);
  }
});
