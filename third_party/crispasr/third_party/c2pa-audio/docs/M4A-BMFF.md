# M4A / MP4 (ISO BMFF) C2PA support — IMPLEMENTED

M4A/MP4 sign + verify is implemented and fully interoperable with c2pa-rs
(both directions). Unlike WAV/MP3 (byte-range `c2pa.hash.data`), M4A uses
`c2pa.hash.bmff.v3`.

## Container layout (signed)

```
ftyp
uuid   (C2PA, uuid = d8fec3d61b0e483c92975828877ec481)   <- inserted after ftyp
free
mdat
moov
```

The C2PA `uuid` box payload (after the 16-byte uuid):
`00000000` (reserved) + `"manifest\0"` (9) + 8-byte merkle offset (0) + JUMBF store.

## The BmffHash v3 algorithm (the key)

`hash = SHA-256( for each NON-excluded top-level box, in file order:
BE64(box_file_offset) ++ box_bytes )`.

Excluded boxes: the C2PA `/uuid` box (matched by the uuid value at offset 8),
plus `/ftyp`, `/free`, `/mfra`, `/skip`. Derived from the c2pa-rs source
(`bmff_to_jumbf_exclusions` + `hash_stream_by_alg`'s `bmff_offset` handling:
each included top-level box contributes its 8-byte BE offset before its bytes)
and verified byte-exact against a c2pa-rs reference.

## Signing

1. Insert the `uuid` box (manifest prefix + store) right after `ftyp`.
2. Adjust `stco`/`co64` chunk offsets by +uuid_box_size (mdat shifted).
3. Compute the bmff.v3 hash over the final layout; embed a c2pa.hash.bmff.v3
   assertion (exclusions `/uuid` [data-matched] `/ftyp` `/mfra` `/free` `/skip`).
   Store size is fixed, so no fixed-point iteration is needed.

Implemented in `src/c2pa_native.cpp` (`sign_m4a` / verify) and `js/c2pa.mjs`
(`c2paSignM4a`) + `js/c2pa-verify.mjs`. Fragmented MP4 (Merkle) is out of scope.
