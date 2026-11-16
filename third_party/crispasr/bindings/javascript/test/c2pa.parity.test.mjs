// c2pa.parity.test.mjs — LIVE parity test: sign a WAV with the native JS
// signer, then validate it with the c2pa-rs reference reader (via the
// c2pa-python CLI). This proves interoperability with the canonical
// implementation — the whole point of re-inventing C2PA in JS.
//
// Skips gracefully when the reference reader is unavailable so CI without the
// python package still passes the unit suite. To run the parity check set
// C2PA_PY to a python that has `c2pa` importable (e.g. a venv), e.g.:
//   C2PA_PY=/path/to/venv/bin/python node --test test/c2pa.parity.test.mjs
import { test } from 'node:test';
import assert from 'node:assert/strict';
import { spawnSync } from 'node:child_process';
import fs from 'node:fs';
import os from 'node:os';
import path from 'node:path';
import { fileURLToPath } from 'node:url';
import { c2paSignWav } from '../c2pa.mjs';

const here = path.dirname(fileURLToPath(import.meta.url));
const asset = (p) => path.join(here, '..', '..', '..', 'assets', 'c2pa', p);
const haveCert = fs.existsSync(asset('crispasr-default-c2pa.crt')) && fs.existsSync(asset('crispasr-default-c2pa.key'));

// Locate a python with the c2pa package importable.
function findC2paPython() {
  const cands = [process.env.C2PA_PY, 'python3', 'python'].filter(Boolean);
  for (const py of cands) {
    const r = spawnSync(py, ['-c', 'import c2pa'], { encoding: 'utf8' });
    if (r.status === 0) return py;
  }
  return null;
}
const py = findC2paPython();

function makeWav(n = 4800, sr = 24000) {
  const buf = Buffer.alloc(44 + n * 2);
  buf.write('RIFF', 0); buf.writeUInt32LE(36 + n * 2, 4); buf.write('WAVE', 8);
  buf.write('fmt ', 12); buf.writeUInt32LE(16, 16); buf.writeUInt16LE(1, 20); buf.writeUInt16LE(1, 22);
  buf.writeUInt32LE(sr, 24); buf.writeUInt32LE(sr * 2, 28); buf.writeUInt16LE(2, 32); buf.writeUInt16LE(16, 34);
  buf.write('data', 36); buf.writeUInt32LE(n * 2, 40);
  for (let i = 0; i < n; i++) buf.writeInt16LE(Math.round(3000 * Math.sin((2 * Math.PI * 220 * i) / sr)), 44 + i * 2);
  return new Uint8Array(buf);
}

// Read a signed WAV with the reference reader; return parsed manifest json.
function readWithReference(pyBin, wavPath) {
  const script = [
    'import sys, json',
    'from c2pa import Reader',
    'with open(sys.argv[1], "rb") as f:',
    '    print(Reader("audio/wav", f).json())',
  ].join('\n');
  const r = spawnSync(pyBin, ['-c', script, wavPath], { encoding: 'utf8' });
  if (r.status !== 0) throw new Error('reference reader failed: ' + (r.stderr || r.stdout));
  return JSON.parse(r.stdout);
}

const skip = !haveCert ? 'bundled cert missing' : !py ? 'c2pa python reader unavailable (set C2PA_PY)' : false;

test('parity: JS-signed WAV validates in the c2pa-rs reference reader', { skip }, async () => {
  const cert = fs.readFileSync(asset('crispasr-default-c2pa.crt'), 'utf8');
  const key = fs.readFileSync(asset('crispasr-default-c2pa.key'), 'utf8');
  const signed = await c2paSignWav(makeWav(), cert, key);

  const tmp = path.join(os.tmpdir(), `crispasr-c2pa-parity-${process.pid}.wav`);
  fs.writeFileSync(tmp, Buffer.from(signed));
  try {
    const m = readWithReference(py, tmp);
    // Validation: the ONLY acceptable status is the self-signed trust warning.
    const statuses = (m.validation_status || []).map((v) => v.code);
    const bad = statuses.filter((c) => c !== 'signingCredential.untrusted');
    assert.deepEqual(bad, [], `unexpected validation failures: ${bad.join(', ')}`);

    // Content parity: generator, assertion, action, digital source type.
    const am = m.manifests[m.active_manifest];
    assert.equal(am.claim_generator_info[0].name, 'CrispASR');
    const actions = am.assertions.find((a) => a.label === 'c2pa.actions.v2');
    assert.ok(actions, 'c2pa.actions.v2 assertion present');
    assert.equal(actions.data.actions[0].action, 'c2pa.created');
    assert.ok(actions.data.actions[0].digitalSourceType.endsWith('trainedAlgorithmicMedia'));
    assert.match(String(am.signature_info?.alg || ''), /Es256/i);
  } finally {
    fs.rmSync(tmp, { force: true });
  }
});

test('parity: tampering the audio makes the reference reader report dataHash.mismatch', { skip }, async () => {
  const cert = fs.readFileSync(asset('crispasr-default-c2pa.crt'), 'utf8');
  const key = fs.readFileSync(asset('crispasr-default-c2pa.key'), 'utf8');
  const signed = await c2paSignWav(makeWav(), cert, key);
  const tampered = Uint8Array.from(signed);
  tampered[60] ^= 0xff; // flip a byte in the audio payload

  const tmp = path.join(os.tmpdir(), `crispasr-c2pa-tamper-${process.pid}.wav`);
  fs.writeFileSync(tmp, Buffer.from(tampered));
  try {
    const m = readWithReference(py, tmp);
    const statuses = (m.validation_status || []).map((v) => v.code);
    assert.ok(
      statuses.some((c) => c.includes('dataHash.mismatch') || c.includes('hashedURI.mismatch')),
      `expected a hash-mismatch status after tamper, got: ${statuses.join(', ')}`,
    );
  } finally {
    fs.rmSync(tmp, { force: true });
  }
});
