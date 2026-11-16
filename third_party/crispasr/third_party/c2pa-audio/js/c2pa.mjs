// c2pa.mjs — native JS C2PA signer for WAV (no c2pa-rs / no ~9 MB Rust wasm).
// Pure WebCrypto: ECDSA P-256 (ES256) + SHA-256; hand-built canonical CBOR,
// JUMBF box tree, COSE_Sign1, and RIFF chunk embedding. The output is a
// standard C2PA v2 manifest with a c2pa.actions.v2 (c2pa.created,
// trainedAlgorithmicMedia) assertion and a c2pa.hash.data hard binding — it
// validates against the c2pa-rs reference reader (only status:
// signingCredential.untrusted, expected for a self-signed cert).
//
// Runs anywhere WebCrypto exists: browsers, Node ≥16, Deno, Cloudflare
// Workers, and (crucially) the emscripten wasm build — replacing the c2pa-rs
// dependency there entirely.
const crypto = globalThis.crypto;
const te = new TextEncoder();

// ---- minimal canonical CBOR encoder (RFC 8949 deterministic) ----
function cborHead(major, n) {
  if (n < 24) return Uint8Array.of((major << 5) | n);
  if (n < 0x100) return Uint8Array.of((major << 5) | 24, n);
  if (n < 0x10000) return Uint8Array.of((major << 5) | 25, n >> 8, n & 255);
  // 32-bit
  return Uint8Array.of((major << 5) | 26, (n >>> 24) & 255, (n >> 16) & 255, (n >> 8) & 255, n & 255);
}
function cat(arrs) {
  let len = 0; for (const a of arrs) len += a.length;
  const out = new Uint8Array(len); let o = 0; for (const a of arrs) { out.set(a, o); o += a.length; }
  return out;
}
function cbor(v) {
  if (v === null) return Uint8Array.of(0xf6);
  if (typeof v === 'boolean') return Uint8Array.of(v ? 0xf5 : 0xf4);
  if (typeof v === 'number' && Number.isInteger(v)) {
    if (v >= 0) return cborHead(0, v);
    return cborHead(1, -v - 1);
  }
  if (typeof v === 'string') { const b = te.encode(v); return cat([cborHead(3, b.length), b]); }
  if (v instanceof Uint8Array) return cat([cborHead(2, v.length), v]);
  if (Array.isArray(v)) return cat([cborHead(4, v.length), ...v.map(cbor)]);
  if (v && v.__tag !== undefined) return cat([cborHead(6, v.__tag), cbor(v.value)]);
  if (v && typeof v === 'object') {
    // canonical: sort keys by encoded-key bytes (length then lexicographic)
    const entries = Object.entries(v).map(([k, val]) => [cbor(isNaN(+k) || k === '' ? k : (v.__intKeys ? +k : k)), cbor(val), k]);
    // (keys here are always strings for our maps except COSE header which uses intKey())
    entries.sort((a, b) => a[0].length - b[0].length || cmp(a[0], b[0]));
    return cat([cborHead(5, entries.length), ...entries.flatMap(e => [e[0], e[1]])]);
  }
  throw new Error('cbor: unsupported ' + typeof v);
}
function cmp(a, b) { const n = Math.min(a.length, b.length); for (let i = 0; i < n; i++) if (a[i] !== b[i]) return a[i] - b[i]; return a.length - b.length; }
// COSE protected header uses integer keys — encode directly
function coseProtected(certDer) {
  // { 1: -7 (ES256), 33: [certDer] }  (canonical: key 1 before key 33)
  return cat([cborHead(5, 2),
    cbor(1), cbor(-7),
    cbor(33), cat([cborHead(4, 1), cat([cborHead(2, certDer.length), certDer])])]);
}
function tag(t, value) { return { __tag: t, value }; }

// ---- JUMBF boxes ----
const UUID_SUFFIX = hexToBytes('00110010800000aa00389b71');
function boxType(ascii) { return cat([te.encode(ascii), UUID_SUFFIX]); } // 16-byte JUMBF type UUID
function u32be(n) { return Uint8Array.of((n >>> 24) & 255, (n >> 16) & 255, (n >> 8) & 255, n & 255); }
function box(type4, payload) { const sz = 8 + payload.length; return cat([u32be(sz), te.encode(type4), payload]); }
// jumd description box: [16-byte type-uuid][toggles=0x03][label\0]
function jumd(typeUuid, label) {
  return box('jumd', cat([typeUuid, Uint8Array.of(0x03), te.encode(label), Uint8Array.of(0)]));
}
// a jumb superbox: jumd(desc) + content boxes
function jumb(typeAscii, label, contentBoxes) {
  return box('jumb', cat([jumd(boxType(typeAscii), label), ...contentBoxes]));
}
function cborBox(cborBytes) { return box('cbor', cborBytes); }

function hexToBytes(h) { const a = new Uint8Array(h.length / 2); for (let i = 0; i < a.length; i++) a[i] = parseInt(h.substr(i * 2, 2), 16); return a; }
function bytesToHex(b) { return [...b].map(x => x.toString(16).padStart(2, '0')).join(''); }
async function sha256(bytes) { return new Uint8Array(await crypto.subtle.digest('SHA-256', bytes)); }

// Parse PEM cert -> DER bytes; PEM EC private key -> CryptoKey (PKCS8).
function pemToDer(pem, kind) {
  const re = new RegExp(`-----BEGIN ${kind}-----([\\s\\S]*?)-----END ${kind}-----`);
  const m = pem.match(re); if (!m) throw new Error('no ' + kind + ' in PEM');
  return Uint8Array.from(atob(m[1].replace(/\s+/g, '')), c => c.charCodeAt(0));
}

// Sign PCM-derived WAV bytes (already a WAV container) with a C2PA manifest.
// certPem/keyPem: PEM strings. Returns signed WAV Uint8Array.
// Container-agnostic manifest-store builder (JUMBF/CBOR/COSE/ES256). Returns
// { assemble(fileHash, exclStart, exclLen, hashDataAssnHash) -> {store,hashDataBox},
//   assertionHash }. WAV and MP3 reuse this; only the embedding differs.
async function makeManifestBuilder(certPem, keyPem, opts) {
  const gen = opts.generator || { name: 'CrispASR', version: '0.6' };
  const manifestUrn = 'urn:c2pa:' + crypto.randomUUID();
  const instanceID = 'xmp:iid:' + crypto.randomUUID();
  const actionsCbor = cbor({ actions: [{ action: 'c2pa.created', softwareAgent: opts.softwareAgent || 'CrispASR TTS', digitalSourceType: 'http://cv.iptc.org/newscodes/digitalsourcetype/trainedAlgorithmicMedia' }] });
  const actionsBox = jumb('cbor', 'c2pa.actions.v2', [cborBox(actionsCbor)]);
  // c2pa hashed-URI: hash of the assertion box WITHOUT its 8-byte outer header.
  const assertionHash = (box) => sha256(box.subarray(8));
  const actionsHash = await assertionHash(actionsBox);
  function hashDataCbor(fileHash, exclStart, exclLen) {
    return cbor({ exclusions: [{ start: exclStart, length: exclLen }], name: 'jumbf manifest', alg: 'sha256', hash: fileHash, pad: new Uint8Array(8) });
  }
  function buildClaim(hardLabel, hardAssnHash) {
    return cbor({
      instanceID,
      claim_generator_info: gen,
      signature: `self#jumbf=/c2pa/${manifestUrn}/c2pa.signature`,
      created_assertions: [{ url: 'self#jumbf=c2pa.assertions/' + hardLabel, hash: hardAssnHash }],
      gathered_assertions: [{ url: 'self#jumbf=c2pa.assertions/c2pa.actions.v2', hash: actionsHash }],
      alg: 'sha256',
    });
  }
  const certDer = pemToDer(certPem, 'CERTIFICATE');
  const key = await crypto.subtle.importKey('pkcs8', pemToDer(keyPem, 'PRIVATE KEY'), { name: 'ECDSA', namedCurve: 'P-256' }, false, ['sign']);
  const protectedHdr = coseProtected(certDer);
  async function buildCoseBox(claimBytes) {
    const sigStructure = cbor(['Signature1', protectedHdr, new Uint8Array(0), claimBytes]);
    const sig = new Uint8Array(await crypto.subtle.sign({ name: 'ECDSA', hash: 'SHA-256' }, key, sigStructure));
    const cose = tag(18, [protectedHdr, {}, null, sig]); // COSE_Sign1
    return jumb('c2cs', 'c2pa.signature', [cborBox(cbor(cose))]);
  }
  // Generic: assemble the store around a caller-built hard-binding box.
  async function assembleWith(hardBox, hardLabel, hardAssnHash) {
    const assertionsBox = jumb('c2as', 'c2pa.assertions', [actionsBox, hardBox]);
    const claimBytes = buildClaim(hardLabel, hardAssnHash);
    const claimBox = jumb('c2cl', 'c2pa.claim.v2', [cborBox(claimBytes)]);
    const sigBox = await buildCoseBox(claimBytes);
    const manifestBox = jumb('c2ma', manifestUrn, [assertionsBox, claimBox, sigBox]);
    return jumb('c2pa', 'c2pa', [manifestBox]); // top c2pa superbox
  }
  async function assemble(fileHash, exclStart, exclLen, hashDataAssnHash) {
    const hashDataBox = jumb('cbor', 'c2pa.hash.data', [cborBox(hashDataCbor(fileHash, exclStart, exclLen))]);
    const store = await assembleWith(hashDataBox, 'c2pa.hash.data', hashDataAssnHash);
    return { store, hashDataBox };
  }
  return { assemble, assembleWith, assertionHash };
}

const C2PA_UUID = Uint8Array.of(0xd8, 0xfe, 0xc3, 0xd6, 0x1b, 0x0e, 0x48, 0x3c, 0x92, 0x97, 0x58, 0x28, 0x87, 0x7e, 0xc4, 0x81);
// Build the c2pa.hash.bmff.v3 assertion CBOR (box-path exclusions + BMFF hash).
function bmffV3Cbor(bmffHash) {
  return cbor({
    exclusions: [
      { xpath: '/uuid', data: [{ offset: 8, value: C2PA_UUID }] },
      { xpath: '/ftyp' }, { xpath: '/mfra' }, { xpath: '/free' }, { xpath: '/skip' },
    ],
    alg: 'sha256', hash: bmffHash, name: 'jumbf manifest',
  });
}

// Run the 4-pass fixed-point protocol given container callbacks.
//   exclStartFor(store) -> byte offset of the excluded region in the assembled file
//   exclLenFor(store)   -> length of the excluded region (== store-derived)
//   assembleFile(store) -> the full container bytes
async function signWithLayout(builder, exclStartFor, exclLenFor, assembleFile) {
  const { assemble, assertionHash } = builder;
  const ZERO = new Uint8Array(32);
  // fixed point on the exclusion (store size feeds back through its CBOR width)
  let exclStart = exclStartFor(new Uint8Array(0)), exclLen = 0;
  for (let i = 0; i < 6; i++) {
    const st = (await assemble(ZERO, exclStart, exclLen, ZERO)).store;
    const es = exclStartFor(st), el = exclLenFor(st);
    if (es === exclStart && el === exclLen) break;
    exclStart = es; exclLen = el;
  }
  const fileHashExcluding = async (full) => sha256(cat([full.subarray(0, exclStart), full.subarray(exclStart + exclLen)]));
  const tmp = assembleFile((await assemble(ZERO, exclStart, exclLen, ZERO)).store);
  const fileHash = await fileHashExcluding(tmp);
  let a = await assemble(fileHash, exclStart, exclLen, ZERO);
  const hashDataAssnHash = await assertionHash(a.hashDataBox);
  a = await assemble(fileHash, exclStart, exclLen, hashDataAssnHash);
  return assembleFile(a.store);
}

// Sign a WAV (RIFF 'C2PA' chunk appended at the end).
export async function c2paSignWav(wavBytes, certPem, keyPem, opts = {}) {
  const builder = await makeManifestBuilder(certPem, keyPem, opts);
  const chunkStart = wavBytes.length;
  const withChunk = (store) => {
    const pad = (store.length & 1) ? Uint8Array.of(0) : new Uint8Array(0);
    const chunk = cat([te.encode('C2PA'), u32beLE(store.length), store, pad]);
    const out = cat([wavBytes.subarray(0, chunkStart), chunk]);
    new DataView(out.buffer).setUint32(4, out.length - 8, true); // RIFF size = total - 8
    return out;
  };
  return signWithLayout(builder, () => chunkStart, (store) => 8 + store.length + (store.length & 1), withChunk);
}

// Sign an MP3 (manifest store in an ID3v2.4 GEOB frame). Preserves existing
// ID3v2.4 frames. The data-hash exclusion covers exactly the GEOB object.
// Sign a FLAC — same ID3v2 GEOB container mechanism as MP3 (c2pa-rs does this too).
export async function c2paSignFlac(flacBytes, certPem, keyPem, opts = {}) {
  if (flacBytes.length < 4 || flacBytes[0] !== 0x66 || flacBytes[1] !== 0x4c || flacBytes[2] !== 0x61 || flacBytes[3] !== 0x43) throw new Error("not a FLAC file");
  return c2paSignMp3(flacBytes, certPem, keyPem, opts);
}
export async function c2paSignMp3(mp3Bytes, certPem, keyPem, opts = {}) {
  const builder = await makeManifestBuilder(certPem, keyPem, opts);
  const synchsafe = (b, o) => (b[o] << 21) | (b[o + 1] << 14) | (b[o + 2] << 7) | b[o + 3];
  const ssBytes = (n) => Uint8Array.of((n >> 21) & 0x7f, (n >> 14) & 0x7f, (n >> 7) & 0x7f, n & 0x7f);

  let existingFrames = new Uint8Array(0), audioStart = 0;
  if (mp3Bytes.length >= 10 && mp3Bytes[0] === 0x49 && mp3Bytes[1] === 0x44 && mp3Bytes[2] === 0x33 && mp3Bytes[3] === 4) {
    const tagEnd = 10 + synchsafe(mp3Bytes, 6);
    if (tagEnd <= mp3Bytes.length) {
      audioStart = tagEnd;
      const frames = [];
      let o = 10;
      while (o + 10 <= tagEnd) {
        if (mp3Bytes[o] === 0) break; // padding
        const fsz = synchsafe(mp3Bytes, o + 4);
        if (o + 10 + fsz > tagEnd) break;
        frames.push(mp3Bytes.subarray(o, o + 10 + fsz));
        o += 10 + fsz;
      }
      existingFrames = cat(frames);
    }
  }
  const audio = mp3Bytes.subarray(audioStart);
  const geobPrefix = cat([Uint8Array.of(0), te.encode('application/c2pa'), Uint8Array.of(0), Uint8Array.of(0), te.encode('c2pa manifest store'), Uint8Array.of(0)]);
  const exclStart = 10 + existingFrames.length + 10 + geobPrefix.length;

  const assembleFile = (store) => {
    const geobBody = cat([geobPrefix, store]);
    const geob = cat([te.encode('GEOB'), ssBytes(geobBody.length), Uint8Array.of(0, 0), geobBody]);
    const frames = cat([existingFrames, geob]);
    return cat([te.encode('ID3'), Uint8Array.of(4, 0, 0), ssBytes(frames.length), frames, audio]);
  };
  return signWithLayout(builder, () => exclStart, (store) => store.length, assembleFile);
}

// ---- ISO BMFF (M4A/MP4) helpers ----
function be32(b, o) { return ((b[o] << 24) | (b[o + 1] << 16) | (b[o + 2] << 8) | b[o + 3]) >>> 0; }
function be64bytes(n) { const a = new Uint8Array(8); for (let i = 0; i < 8; i++) a[i] = Number((BigInt(n) >> BigInt(56 - i * 8)) & 0xffn); return a; }
function bmffBox(b, o) { // -> {size, hdr}
  let size = be32(b, o), hdr = 8;
  if (size === 1) { size = Number(new DataView(b.buffer, b.byteOffset + o + 8, 8).getBigUint64(0)); hdr = 16; }
  else if (size === 0) size = b.length - o;
  return { size, hdr };
}
function memeq(b, o, ascii) { for (let i = 0; i < ascii.length; i++) if (b[o + i] !== ascii.charCodeAt(i)) return false; return true; }
function isC2paUuid(b, o) { for (let i = 0; i < 16; i++) if (b[o + i] !== C2PA_UUID[i]) return false; return true; }
// SHA-256 over each non-excluded top-level box: BE64(offset) ++ box bytes.
async function bmffV3Hash(file) {
  const parts = [];
  let o = 0;
  while (o + 8 <= file.length) {
    const { size, hdr } = bmffBox(file, o);
    if (size < hdr || o + size > file.length) break;
    let excl = memeq(file, o + 4, 'ftyp') || memeq(file, o + 4, 'free') || memeq(file, o + 4, 'mfra') || memeq(file, o + 4, 'skip');
    if (memeq(file, o + 4, 'uuid') && o + hdr + 16 <= file.length && isC2paUuid(file, o + hdr)) excl = true;
    if (!excl) { parts.push(be64bytes(o)); parts.push(file.subarray(o, o + size)); }
    o += size;
  }
  return sha256(cat(parts));
}
// Adjust stco/co64 offsets by +delta within [start,end) (recurses moov path).
function adjustChunkOffsets(out, start, end, delta) {
  let o = start;
  while (o + 8 <= end) {
    const { size, hdr } = bmffBox(out, o);
    if (size < hdr || o + size > end) break;
    if (memeq(out, o + 4, 'stco') || memeq(out, o + 4, 'co64')) {
      const is64 = memeq(out, o + 4, 'co64');
      let p = o + hdr + 4;
      const count = be32(out, p); p += 4;
      const dv = new DataView(out.buffer, out.byteOffset);
      for (let i = 0; i < count; i++) {
        if (is64) { dv.setBigUint64(p, dv.getBigUint64(p) + BigInt(delta)); p += 8; }
        else { dv.setUint32(p, (dv.getUint32(p) + delta) >>> 0); p += 4; }
      }
    } else if (memeq(out, o + 4, 'moov') || memeq(out, o + 4, 'trak') || memeq(out, o + 4, 'mdia') || memeq(out, o + 4, 'minf') || memeq(out, o + 4, 'stbl')) {
      adjustChunkOffsets(out, o + hdr, o + size, delta);
    }
    o += size;
  }
}

// Sign an M4A/MP4 (ISO BMFF): insert a C2PA 'uuid' box after 'ftyp' and bind
// with a c2pa.hash.bmff.v3 assertion (offset-prepend hash + stco/co64 fixups).
export async function c2paSignM4a(m4aBytes, certPem, keyPem, opts = {}) {
  if (m4aBytes.length < 16 || !memeq(m4aBytes, 4, 'ftyp')) throw new Error('not an ISO BMFF (M4A) file');
  const builder = await makeManifestBuilder(certPem, keyPem, opts);
  const { assembleWith, assertionHash } = builder;
  const ZERO = new Uint8Array(32);
  const ftyp = bmffBox(m4aBytes, 0);
  const insertAt = ftyp.size;
  const uuidBox = (store) => {
    const sz = 8 + 16 + 4 + 9 + 8 + store.length;
    return cat([u32be(sz), te.encode('uuid'), C2PA_UUID, new Uint8Array(4), te.encode('manifest'), Uint8Array.of(0), new Uint8Array(8), store]);
  };
  const buildFile = (store) => {
    const ub = uuidBox(store);
    const out = cat([m4aBytes.subarray(0, insertAt), ub, m4aBytes.subarray(insertAt)]);
    adjustChunkOffsets(out, insertAt + ub.length, out.length, ub.length);
    return out;
  };
  const assembleBmff = async (bmffHash, hardAssnHash) => {
    const box = jumb('cbor', 'c2pa.hash.bmff.v3', [cborBox(bmffV3Cbor(bmffHash))]);
    return { store: await assembleWith(box, 'c2pa.hash.bmff.v3', hardAssnHash), hardBox: box };
  };
  const placeholder = (await assembleBmff(ZERO, ZERO)).store;
  const bhash = await bmffV3Hash(buildFile(placeholder));
  const a3 = await assembleBmff(bhash, ZERO);
  const hardAssnHash = await assertionHash(a3.hardBox);
  const a4 = await assembleBmff(bhash, hardAssnHash);
  return buildFile(a4.store);
}
function u32beLE(n) { return Uint8Array.of(n & 255, (n >> 8) & 255, (n >> 16) & 255, (n >>> 24) & 255); } // little-endian for RIFF

// Internals exported for unit tests / advanced callers. Not part of the stable
// public API — use c2paSignWav().
export const _internal = { cbor, cborHead, coseProtected, boxType, jumb, jumd, cborBox, sha256, pemToDer, cat, hexToBytes, bytesToHex };
