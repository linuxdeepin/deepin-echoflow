#!/usr/bin/env python
"""Re-mux a normal WebM/Opus file into a Chrome MediaRecorder-style *live* WebM:
unknown-size Segment and unknown-size Clusters, one cluster per timeslice,
no Duration/Cues/SeekHead. Mirrors what libwebm's mkvmuxer emits to a
non-seekable writer (which is what Chrome's WebmMuxer uses)."""
import sys

def read_id(d, p):
    b = d[p]
    for i in range(4):
        if b & (0x80 >> i):
            n = i + 1
            break
    else:
        raise ValueError("bad id")
    return int.from_bytes(d[p:p + n], 'big'), p + n

def read_vint(d, p):
    b = d[p]
    for i in range(8):
        if b & (0x80 >> i):
            n = i + 1
            break
    else:
        raise ValueError("bad vint")
    raw = int.from_bytes(d[p:p + n], 'big')
    val = raw & ((1 << (7 * n)) - 1)
    return val, p + n, val == (1 << (7 * n)) - 1

def vint(v):
    for n in range(1, 9):
        if v < (1 << (7 * n)) - 1:
            return (v | (1 << (7 * n))).to_bytes(n, 'big')
    raise ValueError

def eid(v):
    return v.to_bytes((v.bit_length() + 7) // 8, 'big')

def el(i, payload):
    return eid(i) + vint(len(payload)) + payload

def uint_el(i, v, width=None):
    b = v.to_bytes(width or max(1, (v.bit_length() + 7) // 8), 'big')
    return el(i, b)

MASTERS = {0x18538067, 0x1654AE6B, 0xAE, 0x1F43B675, 0xA0, 0x1549A966, 0xE1}

def walk(d, p, end, want, out):
    while p < end:
        i, p2 = read_id(d, p)
        sz, p3, unk = read_vint(d, p2)
        e2 = end if unk else min(p3 + sz, end)
        if i in want:
            out.setdefault(i, []).append((p, p3, e2))
        if i in MASTERS:
            walk(d, p3, e2, want, out)
        p = e2

src = open(sys.argv[1], 'rb').read()
dst = sys.argv[2]
slice_ms = int(sys.argv[3]) if len(sys.argv) > 3 else 100

found = {}
walk(src, 0, len(src), {0x18538067, 0x1654AE6B, 0x1F43B675, 0xA3, 0xE7}, found)

tracks_raw = found[0x1654AE6B][0]
tracks = src[tracks_raw[0]:tracks_raw[2]]

# Collect (abs_timecode_ms, simpleblock_payload) — timecode scale assumed 1e6 (1 ms).
blocks = []
for (cs, cb, ce) in found[0x1F43B675]:
    ct = 0
    p = cb
    while p < ce:
        i, p2 = read_id(src, p)
        sz, p3, unk = read_vint(src, p2)
        e2 = ce if unk else min(p3 + sz, ce)
        if i == 0xE7:
            ct = int.from_bytes(src[p3:e2], 'big')
        elif i == 0xA3:
            body = src[p3:e2]
            # track vint + int16 rel timecode + flags
            _, q, _ = read_vint(body, 0)
            rel = int.from_bytes(body[q:q + 2], 'big', signed=True)
            blocks.append((ct + rel, body))
        p = e2

blocks.sort(key=lambda b: b[0])

# EBML header — hand-written, matching what Chrome emits (docType "webm").
ebml = el(0x1A45DFA3,
          uint_el(0x4286, 1) + uint_el(0x42F7, 1) + uint_el(0x42F2, 4) +
          uint_el(0x42F3, 8) + el(0x4282, b'webm') + uint_el(0x4287, 2) +
          uint_el(0x4285, 2))

info = el(0x1549A966,
          uint_el(0x2AD7B1, 1000000) + el(0x4D80, b'Chrome') + el(0x5741, b'Chrome'))

out = bytearray()
out += ebml
out += eid(0x18538067) + b'\x01\xff\xff\xff\xff\xff\xff\xff'   # Segment, unknown size
out += info
out += tracks

cluster_start = None
cluster_blocks = []

def flush(o, start, blks):
    if not blks:
        return
    o += eid(0x1F43B675) + b'\x01\xff\xff\xff\xff\xff\xff\xff'  # Cluster, unknown size
    o += uint_el(0xE7, start)
    for (ts, body) in blks:
        _, q, _ = read_vint(body, 0)
        rel = ts - start
        nb = bytearray(body)
        nb[q:q + 2] = int(rel).to_bytes(2, 'big', signed=True)
        o += el(0xA3, bytes(nb))

for ts, body in blocks:
    if cluster_start is None:
        cluster_start = ts
    if ts - cluster_start >= slice_ms:
        flush(out, cluster_start, cluster_blocks)
        cluster_start, cluster_blocks = ts, []
    cluster_blocks.append((ts, body))
flush(out, cluster_start, cluster_blocks)

open(dst, 'wb').write(bytes(out))
print(f"wrote {dst}: {len(out)} bytes, {len(blocks)} blocks, slice={slice_ms}ms")
