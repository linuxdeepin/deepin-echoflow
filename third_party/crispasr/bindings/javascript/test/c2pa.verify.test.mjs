// c2pa.verify.test.mjs — tests for the native JS C2PA verifier (c2pa-verify.mjs).
// Hermetic: signs with the bundled cert then verifies; also verifies a committed
// c2pa-rs reference vector (proving "their signer -> our verifier"); and checks
// tamper rejection. No c2pa-rs / no python needed.
import { test } from 'node:test';
import assert from 'node:assert/strict';
import fs from 'node:fs';
import path from 'node:path';
import { fileURLToPath } from 'node:url';
import { c2paSignWav } from '../c2pa.mjs';
import { c2paVerifyWav } from '../c2pa-verify.mjs';

const here = path.dirname(fileURLToPath(import.meta.url));
const asset = (p) => path.join(here, '..', '..', '..', 'assets', 'c2pa', p);
const fixture = (p) => path.join(here, '..', '..', '..', 'tests', 'assets', 'c2pa', p);
const haveCert = fs.existsSync(asset('crispasr-default-c2pa.crt'));
const cert = haveCert ? fs.readFileSync(asset('crispasr-default-c2pa.crt'), 'utf8') : null;
const key = haveCert ? fs.readFileSync(asset('crispasr-default-c2pa.key'), 'utf8') : null;

function makeWav(n = 4800, sr = 24000) {
  const buf = Buffer.alloc(44 + n * 2);
  buf.write('RIFF', 0); buf.writeUInt32LE(36 + n * 2, 4); buf.write('WAVE', 8);
  buf.write('fmt ', 12); buf.writeUInt32LE(16, 16); buf.writeUInt16LE(1, 20); buf.writeUInt16LE(1, 22);
  buf.writeUInt32LE(sr, 24); buf.writeUInt32LE(sr * 2, 28); buf.writeUInt16LE(2, 32); buf.writeUInt16LE(16, 34);
  buf.write('data', 36); buf.writeUInt32LE(n * 2, 40);
  for (let i = 0; i < n; i++) buf.writeInt16LE(Math.round(3000 * Math.sin((2 * Math.PI * 220 * i) / sr)), 44 + i * 2);
  return new Uint8Array(buf);
}

test('verify: our JS-signed WAV validates (round-trip)', { skip: !haveCert }, async () => {
  const signed = await c2paSignWav(makeWav(), cert, key);
  const r = await c2paVerifyWav(signed);
  assert.equal(r.signatureValid, true, r.errors.join('; '));
  assert.equal(r.dataHashValid, true, r.errors.join('; '));
  assert.equal(r.assertionsValid, true, r.errors.join('; '));
  assert.equal(r.valid, true);
  assert.equal(r.manifest.generatorName, 'CrispASR');
  assert.equal(r.manifest.actions[0].action, 'c2pa.created');
});

test('verify: tampering the audio fails the hard binding', { skip: !haveCert }, async () => {
  const signed = await c2paSignWav(makeWav(), cert, key);
  const tampered = Uint8Array.from(signed);
  tampered[46] ^= 0xff; // flip a byte in the audio payload
  const r = await c2paVerifyWav(tampered);
  assert.equal(r.dataHashValid, false);
  assert.equal(r.valid, false);
  assert.ok(r.errors.some((e) => e.includes('data hash')));
});

test('verify: tampering an assertion fails the assertion binding', { skip: !haveCert }, async () => {
  const signed = await c2paSignWav(makeWav(), cert, key);
  // flip a byte deep in the manifest (inside the JUMBF, past the audio) — most
  // likely lands in an assertion/claim, breaking either the assertion hash, the
  // claim signature, or (if in the exclusion) nothing; require overall invalid.
  const tampered = Uint8Array.from(signed);
  tampered[tampered.length - 40] ^= 0xff;
  const r = await c2paVerifyWav(tampered);
  assert.equal(r.valid, false);
});

test('verify: c2pa-rs reference vector validates (their signer -> our verifier)', async () => {
  const p = fixture('reference-c2pa-rs.wav');
  if (!fs.existsSync(p)) return; // fixture optional
  const r = await c2paVerifyWav(new Uint8Array(fs.readFileSync(p)));
  assert.equal(r.signatureValid, true, r.errors.join('; '));
  assert.equal(r.dataHashValid, true, r.errors.join('; '));
  assert.equal(r.valid, true, r.errors.join('; '));
});

test('verify: non-C2PA WAV reports no manifest', async () => {
  const r = await c2paVerifyWav(makeWav());
  assert.equal(r.valid, false);
  assert.ok(r.errors.some((e) => e.includes('no C2PA chunk')));
});
