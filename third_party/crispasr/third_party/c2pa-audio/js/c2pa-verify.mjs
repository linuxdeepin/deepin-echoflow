// c2pa-verify.mjs — native JS C2PA verifier for WAV (no c2pa-rs). The twin of
// c2pa.mjs: parses the RIFF C2PA chunk + JUMBF tree, decodes the CBOR claim /
// assertions / COSE_Sign1, verifies the ES256 signature against the embedded
// certificate's public key (WebCrypto), and recomputes the hard-binding data
// hash + assertion hashedURIs. Pure WebCrypto — runs in browsers, Node, Deno,
// Workers.
//
// c2paVerifyWav(wavBytes) -> {
//   valid,            // signature + all hash bindings check out
//   signatureValid,   // COSE ES256 verified against the embedded cert
//   dataHashValid,    // hard binding matches the audio
//   assertionsValid,  // every claimed assertion hash matches
//   trusted,          // false for self-signed (no trust-anchor checking here)
//   errors: [string], // human-readable failures
//   manifest: { generatorName, instanceID, actions:[...], alg },
// }
const crypto = globalThis.crypto;
const td = new TextDecoder();

// ------------------------------------------------------------- minimal CBOR decode
function cborDecode(bytes) {
  let pos = 0;
  function u(n) { let v = 0; for (let i = 0; i < n; i++) v = v * 256 + bytes[pos++]; return v; }
  function head() {
    const ib = bytes[pos++];
    const major = ib >> 5, ai = ib & 0x1f;
    let len;
    if (ai < 24) len = ai;
    else if (ai === 24) len = bytes[pos++];
    else if (ai === 25) len = u(2);
    else if (ai === 26) len = u(4);
    else if (ai === 27) len = u(8);
    else len = ai; // 31 = indefinite (unused here)
    return { major, ai, len };
  }
  function value(depth = 0) {
    if (depth > 64 || pos >= bytes.length) throw new Error('cbor: nesting/truncation');
    const h = head();
    switch (h.major) {
      case 0: return h.len;                 // uint
      case 1: return -1 - h.len;            // negative
      case 2: { const b = bytes.slice(pos, pos + h.len); pos += h.len; return b; } // bstr
      case 3: { const b = bytes.slice(pos, pos + h.len); pos += h.len; return td.decode(b); } // tstr
      // array/map: each element is >=1 byte, so a length exceeding what remains
      // is invalid — stop early rather than loop on a bogus huge length.
      case 4: { const a = []; for (let i = 0; i < h.len && pos < bytes.length; i++) a.push(value(depth + 1)); return a; }
      case 5: { const m = new Map(); for (let i = 0; i < h.len && pos < bytes.length; i++) { const k = value(depth + 1); m.set(typeof k === 'string' || typeof k === 'number' ? k : JSON.stringify(k), value(depth + 1)); } return m; }
      case 6: return { __tag: h.len, value: value(depth + 1) };  // tag
      case 7:
        if (h.ai === 20) return false;
        if (h.ai === 21) return true;
        if (h.ai === 22) return null;
        return undefined;
      default: throw new Error('cbor: bad major ' + h.major);
    }
  }
  const v = value();
  return v;
}

// ------------------------------------------------- container store extraction
// Locate the C2PA manifest store — RIFF 'C2PA' chunk (WAV) or ID3v2 GEOB frame
// (MP3). Returns { body } holding the store bytes, or null.
function findC2paChunk(file) {
  // WAV RIFF
  if (file.length >= 12 && file[0] === 0x52 && file[1] === 0x49 && file[2] === 0x46 && file[3] === 0x46) {
    const dv = new DataView(file.buffer, file.byteOffset, file.byteLength);
    let off = 12;
    while (off + 8 <= file.length) {
      const id = String.fromCharCode(file[off], file[off + 1], file[off + 2], file[off + 3]);
      const sz = dv.getUint32(off + 4, true);
      if (id === 'C2PA') return { start: off, size: sz, body: file.subarray(off + 8, off + 8 + sz) };
      off += 8 + sz + (sz & 1);
    }
    return null;
  }
  // MP3 ID3v2 GEOB frame with mime "application/c2pa"
  if (file.length >= 10 && file[0] === 0x49 && file[1] === 0x44 && file[2] === 0x33) {
    const ss = (o) => (file[o] << 21) | (file[o + 1] << 14) | (file[o + 2] << 7) | file[o + 3];
    const end = Math.min(file.length, 10 + ss(6));
    for (let o = 10; o + 10 <= end;) {
      if (file[o] === 0) break; // padding
      const fsz = ss(o + 4);
      if (o + 10 + fsz > end) break;
      const id = String.fromCharCode(file[o], file[o + 1], file[o + 2], file[o + 3]);
      if (id === 'GEOB') {
        const body = file.subarray(o + 10, o + 10 + fsz);
        let p = 1; // skip encoding byte
        const nul = () => { while (p < body.length && body[p] !== 0) p++; const s = p; p++; return s; };
        const mimeEnd = nul(); const mime = String.fromCharCode(...body.subarray(1, mimeEnd));
        nul(); nul(); // filename, description
        if (mime === 'application/c2pa') return { start: o, size: fsz, body: body.subarray(p) };
      }
      o += 10 + fsz;
    }
    return null;
  }
  // ISO BMFF (M4A/MP4): store in a top-level 'uuid' box with the C2PA uuid
  if (file.length >= 8 && file[4] === 0x66 && file[5] === 0x74 && file[6] === 0x79 && file[7] === 0x70) { // 'ftyp'
    const UUID = [0xd8, 0xfe, 0xc3, 0xd6, 0x1b, 0x0e, 0x48, 0x3c, 0x92, 0x97, 0x58, 0x28, 0x87, 0x7e, 0xc4, 0x81];
    let o = 0;
    while (o + 8 <= file.length) {
      let size = ((file[o] << 24) | (file[o + 1] << 16) | (file[o + 2] << 8) | file[o + 3]) >>> 0, hdr = 8;
      if (size === 1) { size = Number(new DataView(file.buffer, file.byteOffset + o + 8, 8).getBigUint64(0)); hdr = 16; }
      else if (size === 0) size = file.length - o;
      if (size < hdr || o + size > file.length) break;
      if (file[o + 4] === 0x75 && file[o + 5] === 0x75 && file[o + 6] === 0x69 && file[o + 7] === 0x64) { // 'uuid'
        let match = o + hdr + 16 <= file.length;
        for (let i = 0; match && i < 16; i++) if (file[o + hdr + i] !== UUID[i]) match = false;
        if (match) {
          const sstart = o + hdr + 16 + 4 + 9 + 8; // reserved + "manifest\0" + merkle offset
          if (sstart <= o + size) return { start: o, size, body: file.subarray(sstart, o + size) };
        }
      }
      o += size;
    }
    return null;
  }
  return null;
}
function rd32be(b, o) { return ((b[o] << 24) | (b[o + 1] << 16) | (b[o + 2] << 8) | b[o + 3]) >>> 0; }
// Walk the JUMBF tree; collect { label -> {box, contentCbor} }.
function walkJumbf(body) {
  const boxes = {};
  let count = 0;
  // Bound recursion depth + box count: real manifests nest ~5 deep with a
  // handful of boxes, but a hostile file must not blow the stack or wedge us.
  function walk(b, depth) {
    if (depth > 32 || count > 4096) return;
    let o = 0;
    while (o + 8 <= b.length) {
      const sz = rd32be(b, o);
      const typ = String.fromCharCode(b[o + 4], b[o + 5], b[o + 6], b[o + 7]);
      if (sz < 8 || o + sz > b.length) break;
      if (typ === 'jumb') {
        const payload = b.subarray(o + 8, o + sz);
        const jsz = rd32be(payload, 0);
        const lblStart = 8 + 16 + 1; // jumd hdr + type uuid + toggles
        const lblEnd = Math.min(jsz, payload.length); // bound by BOTH
        let e = lblStart; while (e < lblEnd && payload[e] !== 0) e++;
        const label = e > lblStart ? String.fromCharCode(...payload.subarray(lblStart, e)) : '';
        // find a cbor content box (first level under the jumd)
        let content = null, po = jsz;
        while (po + 8 <= payload.length) {
          const psz = rd32be(payload, po);
          const pt = String.fromCharCode(payload[po + 4], payload[po + 5], payload[po + 6], payload[po + 7]);
          if (psz < 8 || po + psz > payload.length) break;
          if (pt === 'cbor') content = payload.subarray(po + 8, po + psz);
          po += psz;
        }
        boxes[label] = { box: b.subarray(o, o + sz), content };
        if (++count > 4096) return;
        walk(payload, depth + 1);
      }
      o += sz;
    }
  }
  walk(body, 0);
  return boxes;
}

// ------------------------------------------------------------------- crypto helpers
async function sha256(bytes) { return new Uint8Array(await crypto.subtle.digest('SHA-256', bytes)); }
function eq(a, b) { if (a.length !== b.length) return false; for (let i = 0; i < a.length; i++) if (a[i] !== b[i]) return false; return true; }
function concat(a, b) { const o = new Uint8Array(a.length + b.length); o.set(a, 0); o.set(b, a.length); return o; }
// Extract the 65-byte uncompressed EC point (04||x||y) from an X.509 P-256 cert
// DER: locate the SubjectPublicKeyInfo BIT STRING `03 42 00 04 <64>`.
function extractEcPoint(certDer) {
  for (let i = 0; i + 4 + 65 <= certDer.length; i++) {
    if (certDer[i] === 0x03 && certDer[i + 1] === 0x42 && certDer[i + 2] === 0x00 && certDer[i + 3] === 0x04) {
      return certDer.slice(i + 3, i + 3 + 65); // 04 || x || y
    }
  }
  return null;
}
// Re-encode canonical CBOR bstr/tstr/array head + payload for the Sig_structure.
function cborHead(major, n) {
  if (n < 24) return Uint8Array.of((major << 5) | n);
  if (n < 0x100) return Uint8Array.of((major << 5) | 24, n);
  if (n < 0x10000) return Uint8Array.of((major << 5) | 25, n >> 8, n & 255);
  return Uint8Array.of((major << 5) | 26, (n >>> 24) & 255, (n >> 16) & 255, (n >> 8) & 255, n & 255);
}
function bstr(b) { return concat(cborHead(2, b.length), b); }
function tstr(s) { const b = new TextEncoder().encode(s); return concat(cborHead(3, b.length), b); }

// ------------------------------------------------------------------------ verify
export async function c2paVerifyWav(wavBytes) {
  const errors = [];
  const res = { valid: false, signatureValid: false, dataHashValid: false, assertionsValid: false, trusted: false, errors, manifest: null };
  try {
    const chunk = findC2paChunk(wavBytes);
    if (!chunk) { errors.push('no C2PA chunk in RIFF'); return res; }
    const boxes = walkJumbf(chunk.body);
    for (const need of ['c2pa.claim.v2', 'c2pa.signature', 'c2pa.actions.v2']) {
      if (!boxes[need]) { errors.push('missing JUMBF box: ' + need); return res; }
    }
    if (!boxes['c2pa.hash.data'] && !boxes['c2pa.hash.bmff.v3']) { errors.push('missing hard-binding assertion'); return res; }

    // --- decode claim + signature ---
    const claimBytes = boxes['c2pa.claim.v2'].content;
    const claim = cborDecode(claimBytes);
    const cget = (k) => (claim instanceof Map ? claim.get(k) : claim[k]);
    const cose = cborDecode(boxes['c2pa.signature'].content); // tag(18,[protected,unprotected,payload,sig])
    const coseArr = cose.__tag !== undefined ? cose.value : cose;
    const protectedBstr = coseArr[0];       // serialized protected header (bytes)
    const signature = coseArr[3];           // raw r||s (64)
    const protMap = cborDecode(protectedBstr);
    const certChain = protMap.get(33) || protMap.get('33');
    const certDer = Array.isArray(certChain) ? certChain[0] : certChain;
    if (!certDer) { errors.push('no certificate in COSE protected header'); return res; }

    // --- verify COSE_Sign1 ES256 ---
    const point = extractEcPoint(certDer);
    if (!point) { errors.push('could not extract EC public key from certificate'); return res; }
    const key = await crypto.subtle.importKey('raw', point, { name: 'ECDSA', namedCurve: 'P-256' }, false, ['verify']);
    // Sig_structure = ["Signature1", protected, external_aad(empty), payload(claim)]
    const sigStructure = concat(cborHead(4, 4), concat(concat(tstr('Signature1'), bstr(protectedBstr)), concat(bstr(new Uint8Array(0)), bstr(claimBytes))));
    res.signatureValid = await crypto.subtle.verify({ name: 'ECDSA', hash: 'SHA-256' }, key, signature, sigStructure);
    if (!res.signatureValid) errors.push('COSE ES256 signature does not verify');

    // --- verify assertion hashedURIs (hash = sha256(box without 8-byte header)) ---
    const created = cget('created_assertions') || [];
    const gathered = cget('gathered_assertions') || [];
    let assertionsOk = true;
    for (const a of [...created, ...gathered]) {
      const url = a instanceof Map ? a.get('url') : a.url;
      const storedHash = a instanceof Map ? a.get('hash') : a.hash;
      const label = String(url).split('/').pop();
      const bx = boxes[label];
      if (!bx) { errors.push('assertion box not found: ' + label); assertionsOk = false; continue; }
      const h = await sha256(bx.box.subarray(8));
      if (!eq(h, storedHash)) { errors.push('assertion hash mismatch: ' + label); assertionsOk = false; }
    }
    res.assertionsValid = assertionsOk;

    // --- verify hard binding ---
    if (boxes['c2pa.hash.data']) {
      // byte-range data hash (WAV/MP3)
      const hd = cborDecode(boxes['c2pa.hash.data'].content);
      const hget = (k) => (hd instanceof Map ? hd.get(k) : hd[k]);
      const excl = (hget('exclusions') || [])[0];
      const exStart = excl ? (excl instanceof Map ? excl.get('start') : excl.start) : chunk.start;
      const exLen = excl ? (excl instanceof Map ? excl.get('length') : excl.length) : (8 + chunk.size + (chunk.size & 1));
      const storedFileHash = hget('hash');
      const fileHash = await sha256(concat(wavBytes.subarray(0, exStart), wavBytes.subarray(exStart + exLen)));
      res.dataHashValid = eq(fileHash, storedFileHash);
    } else {
      // BMFF v3 (M4A/MP4): SHA-256 over BE64(offset)++box for each non-excluded top box
      const bh = cborDecode(boxes['c2pa.hash.bmff.v3'].content);
      const bget = (k) => (bh instanceof Map ? bh.get(k) : bh[k]);
      const storedFileHash = bget('hash');
      const excls = bget('exclusions') || [];
      const parts = [];
      let o = 0;
      while (o + 8 <= wavBytes.length) {
        let size = ((wavBytes[o] << 24) | (wavBytes[o + 1] << 16) | (wavBytes[o + 2] << 8) | wavBytes[o + 3]) >>> 0, hdr = 8;
        if (size === 1) { size = Number(new DataView(wavBytes.buffer, wavBytes.byteOffset + o + 8, 8).getBigUint64(0)); hdr = 16; }
        else if (size === 0) size = wavBytes.length - o;
        if (size < hdr || o + size > wavBytes.length) break;
        const btype = '/' + String.fromCharCode(wavBytes[o + 4], wavBytes[o + 5], wavBytes[o + 6], wavBytes[o + 7]);
        let excluded = false;
        for (const ex of excls) {
          const xp = ex instanceof Map ? ex.get('xpath') : ex.xpath;
          if (xp !== btype) continue;
          const dm = ex instanceof Map ? ex.get('data') : ex.data;
          let matches = true;
          if (dm) for (const d of dm) {
            const off = d instanceof Map ? d.get('offset') : d.offset;
            const val = d instanceof Map ? d.get('value') : d.value;
            if (o + off + val.length > o + size) { matches = false; break; }
            for (let i = 0; i < val.length; i++) if (wavBytes[o + off + i] !== val[i]) { matches = false; break; }
            if (!matches) break;
          }
          if (matches) { excluded = true; break; }
        }
        if (!excluded) {
          const be = new Uint8Array(8);
          for (let i = 0; i < 8; i++) be[i] = Number((BigInt(o) >> BigInt(56 - i * 8)) & 0xffn);
          parts.push(be); parts.push(wavBytes.subarray(o, o + size));
        }
        o += size;
      }
      let all = new Uint8Array(0);
      for (const p of parts) all = concat(all, p);
      res.dataHashValid = eq(await sha256(all), storedFileHash);
    }
    if (!res.dataHashValid) errors.push('data hash (hard binding) mismatch');

    // --- surface manifest content ---
    const actionsCbor = cborDecode(boxes['c2pa.actions.v2'].content);
    const actions = (actionsCbor instanceof Map ? actionsCbor.get('actions') : actionsCbor.actions) || [];
    const gen = cget('claim_generator_info');
    res.manifest = {
      generatorName: gen ? (gen instanceof Map ? gen.get('name') : gen.name) : undefined,
      instanceID: cget('instanceID'),
      alg: cget('alg'),
      actions: actions.map((ac) => ({
        action: ac instanceof Map ? ac.get('action') : ac.action,
        digitalSourceType: ac instanceof Map ? ac.get('digitalSourceType') : ac.digitalSourceType,
        softwareAgent: ac instanceof Map ? ac.get('softwareAgent') : ac.softwareAgent,
      })),
    };

    res.valid = res.signatureValid && res.dataHashValid && res.assertionsValid;
    res.trusted = false; // self-signed: no trust-anchor evaluation here
    return res;
  } catch (e) {
    errors.push('verify exception: ' + (e && e.message ? e.message : String(e)));
    return res;
  }
}
