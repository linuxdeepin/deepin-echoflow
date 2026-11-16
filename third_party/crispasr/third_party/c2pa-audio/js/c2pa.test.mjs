// Self-contained JS test: sign (bundled cert) -> verify -> tamper -> reference.
import { test } from 'node:test';
import assert from 'node:assert/strict';
import fs from 'node:fs';
import path from 'node:path';
import { fileURLToPath } from 'node:url';
import { c2paSignWav } from './c2pa.mjs';
import { c2paVerifyWav } from './c2pa-verify.mjs';
import { DEFAULT_CERT_PEM, DEFAULT_KEY_PEM } from './default-cert.mjs';

const here = path.dirname(fileURLToPath(import.meta.url));
function makeWav(n = 4800, sr = 24000) {
  const buf = Buffer.alloc(44 + n * 2);
  buf.write('RIFF', 0); buf.writeUInt32LE(36 + n * 2, 4); buf.write('WAVE', 8);
  buf.write('fmt ', 12); buf.writeUInt32LE(16, 16); buf.writeUInt16LE(1, 20); buf.writeUInt16LE(1, 22);
  buf.writeUInt32LE(sr, 24); buf.writeUInt32LE(sr * 2, 28); buf.writeUInt16LE(2, 32); buf.writeUInt16LE(16, 34);
  buf.write('data', 36); buf.writeUInt32LE(n * 2, 40);
  for (let i = 0; i < n; i++) buf.writeInt16LE(Math.round(3000 * Math.sin((2 * Math.PI * 220 * i) / sr)), 44 + i * 2);
  return new Uint8Array(buf);
}

test('sign -> verify round-trip', async () => {
  const signed = await c2paSignWav(makeWav(), DEFAULT_CERT_PEM, DEFAULT_KEY_PEM);
  const r = await c2paVerifyWav(signed);
  assert.equal(r.valid, true, r.errors.join('; '));
});
test('audio tamper fails', async () => {
  const signed = await c2paSignWav(makeWav(), DEFAULT_CERT_PEM, DEFAULT_KEY_PEM);
  signed[46] ^= 0xff;
  assert.equal((await c2paVerifyWav(signed)).valid, false);
});
test('c2pa-rs reference vector validates', async () => {
  const p = path.join(here, 'reference-c2pa-rs.wav');
  if (!fs.existsSync(p)) return;
  assert.equal((await c2paVerifyWav(new Uint8Array(fs.readFileSync(p)))).valid, true);
});

import { c2paSignMp3 } from './c2pa.mjs';
test('MP3 sign -> verify round-trip', async () => {
  const p = path.join(here, 'sample.mp3');
  if (!fs.existsSync(p)) return;
  const signed = await c2paSignMp3(new Uint8Array(fs.readFileSync(p)), DEFAULT_CERT_PEM, DEFAULT_KEY_PEM);
  assert.equal((await c2paVerifyWav(signed)).valid, true);
});
test('c2pa-rs MP3 reference validates (their signer -> our verifier)', async () => {
  const p = path.join(here, 'reference-c2pa-rs.mp3');
  if (!fs.existsSync(p)) return;
  assert.equal((await c2paVerifyWav(new Uint8Array(fs.readFileSync(p)))).valid, true);
});

import { c2paSignM4a } from './c2pa.mjs';
test('M4A sign -> verify round-trip', async () => {
  const p = path.join(here, 'sample.m4a');
  if (!fs.existsSync(p)) return;
  const signed = await c2paSignM4a(new Uint8Array(fs.readFileSync(p)), DEFAULT_CERT_PEM, DEFAULT_KEY_PEM);
  assert.equal((await c2paVerifyWav(signed)).valid, true);
});
test('c2pa-rs M4A reference validates (their signer -> our verifier)', async () => {
  const p = path.join(here, 'reference-c2pa-rs.m4a');
  if (!fs.existsSync(p)) return;
  assert.equal((await c2paVerifyWav(new Uint8Array(fs.readFileSync(p)))).valid, true);
});

import { c2paSignFlac } from './c2pa.mjs';
test('FLAC sign -> verify round-trip', async () => {
  const p = path.join(here, 'sample.flac');
  if (!fs.existsSync(p)) return;
  const signed = await c2paSignFlac(new Uint8Array(fs.readFileSync(p)), DEFAULT_CERT_PEM, DEFAULT_KEY_PEM);
  assert.equal((await c2paVerifyWav(signed)).valid, true);
});
test('c2pa-rs FLAC reference validates (their signer -> our verifier)', async () => {
  const p = path.join(here, 'reference-c2pa-rs.flac');
  if (!fs.existsSync(p)) return;
  assert.equal((await c2paVerifyWav(new Uint8Array(fs.readFileSync(p)))).valid, true);
});
