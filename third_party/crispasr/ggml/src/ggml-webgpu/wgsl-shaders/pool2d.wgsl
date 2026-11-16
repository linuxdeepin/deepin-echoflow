// pool_2d: src [IW, IH, C, N] f32 -> dst [OW, OH, C, N] f32 (contiguous).
// Variants: POOL_MAX / POOL_AVG (matches the CPU reference: out-of-bounds
// taps are skipped for max and contribute zero for avg, which divides by
// the full kernel area). One thread per dst element; 2D dispatch.

@group(0) @binding(0)
var<storage, read_write> src: array<f32>;

@group(0) @binding(1)
var<storage, read_write> dst: array<f32>;

struct Params {
    ne: u32,
    nwg_x: u32,
    offset_src: u32,
    offset_dst: u32,

    // src strides (in elements)
    stride_src1: u32,
    stride_src2: u32,
    stride_src3: u32,

    IW: u32,
    IH: u32,
    OW: u32,
    OH: u32,

    C: u32,            // src ne2 (channels)
    k0: u32,
    k1: u32,
    s0: u32,
    s1: u32,
    p0: u32,
    p1: u32,
};

@group(0) @binding(2)
var<uniform> params: Params;

@compute @workgroup_size(WG_SIZE)
fn main(@builtin(workgroup_id) wid: vec3<u32>,
        @builtin(local_invocation_id) lid: vec3<u32>) {
    let idx = (wid.y * params.nwg_x + wid.x) * WG_SIZE + lid.x;
    if (idx >= params.ne) {
        return;
    }

    var i = idx;
    let plane = params.OH * params.OW;
    let i23 = i / plane;      // fused channel/batch index (contiguous dst)
    i = i % plane;
    let oy = i / params.OW;
    let ox = i % params.OW;

    // src channel/batch offset: dst is contiguous, so i23 splits into
    // (i2, i3) against the src strides.
    let i3 = i23 / params.C;
    let i2 = i23 % params.C;
    let src_base = params.offset_src + i3 * params.stride_src3 + i2 * params.stride_src2;

#ifdef POOL_MAX
    // -FLT_MAX (the decimal literal rounds outside f32 range and WGSL
    // rejects it, so bitcast the exact pattern; matches the CPU init).
    var acc: f32 = bitcast<f32>(0xff7fffffu);
#else
    var acc: f32 = 0.0;
#endif

    for (var ky: u32 = 0; ky < params.k1; ky++) {
        let iy = i32(oy * params.s1 + ky) - i32(params.p1);
        if (iy < 0 || u32(iy) >= params.IH) {
            continue;
        }
        for (var kx: u32 = 0; kx < params.k0; kx++) {
            let ix = i32(ox * params.s0 + kx) - i32(params.p0);
            if (ix < 0 || u32(ix) >= params.IW) {
                continue;
            }
            let v = src[src_base + u32(iy) * params.stride_src1 + u32(ix)];
#ifdef POOL_MAX
            acc = max(acc, v);
#else
            acc += v;
#endif
        }
    }

#ifdef POOL_MAX
    dst[params.offset_dst + idx] = acc;
#else
    dst[params.offset_dst + idx] = acc / f32(params.k0 * params.k1);
#endif
}
