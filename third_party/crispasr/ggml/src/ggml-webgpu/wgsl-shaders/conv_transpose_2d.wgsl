enable f16;

// conv_transpose_2d (p0): kernel src0 [KW, KH, OC, IC] (f16), input src1
// [IW, IH, IC, N] (f32) -> dst [OW, OH, OC, N] (f32, contiguous), where
// OW = (IW-1)*stride + KW, OH = (IH-1)*stride + KH. Gather formulation:
// one thread per dst element, accumulating the taps (ix, iy) with
// ox = ix*stride + kx, oy = iy*stride + ky. 2D dispatch.

@group(0) @binding(0)
var<storage, read_write> kern: array<f16>;

@group(0) @binding(1)
var<storage, read_write> src: array<f32>;

@group(0) @binding(2)
var<storage, read_write> dst: array<f32>;

struct Params {
    ne: u32,
    nwg_x: u32,
    offset_kern: u32,  // in elements
    offset_src: u32,   // in elements
    offset_dst: u32,   // in elements

    // kernel strides (in elements): [KW, KH, OC, IC]
    stride_k1: u32,
    stride_k2: u32,
    stride_k3: u32,

    // input strides (in elements): [IW, IH, IC, N]
    stride_src1: u32,
    stride_src2: u32,
    stride_src3: u32,

    IW: u32,
    IH: u32,
    IC: u32,
    OW: u32,
    OH: u32,
    OC: u32,
    KW: u32,
    KH: u32,
    stride: u32,
};

@group(0) @binding(3)
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
    let i23 = i / plane;
    i = i % plane;
    let oy = i / params.OW;
    let ox = i % params.OW;
    let oc = i23 % params.OC;
    let n = i23 / params.OC;

    var acc: f32 = 0.0;
    for (var ky: u32 = 0; ky < params.KH; ky++) {
        if (oy + 1 < ky + 1) {  // oy < ky, unsigned-safe
            continue;
        }
        let dy = oy - ky;
        if (dy % params.stride != 0) {
            continue;
        }
        let iy = dy / params.stride;
        if (iy >= params.IH) {
            continue;
        }
        for (var kx: u32 = 0; kx < params.KW; kx++) {
            if (ox + 1 < kx + 1) {
                continue;
            }
            let dx = ox - kx;
            if (dx % params.stride != 0) {
                continue;
            }
            let ix = dx / params.stride;
            if (ix >= params.IW) {
                continue;
            }
            let src_row = params.offset_src + n * params.stride_src3 + iy * params.stride_src1 + ix;
            let k_row = params.offset_kern + oc * params.stride_k2 + ky * params.stride_k1 + kx;
            for (var ic: u32 = 0; ic < params.IC; ic++) {
                let v = src[src_row + ic * params.stride_src2];
                let w = f32(kern[k_row + ic * params.stride_k3]);
                acc += v * w;
            }
        }
    }

    dst[params.offset_dst + idx] = acc;
}
