// arange: dst[i] = start + i*step (f32, contiguous)

@group(0) @binding(0)
var<storage, read_write> dst: array<f32>;

struct Params {
    ne: u32,
    offset_dst: u32,   // in elements
    start: f32,
    step: f32,
};

@group(0) @binding(1)
var<uniform> params: Params;

@compute @workgroup_size(WG_SIZE)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    if (gid.x >= params.ne) {
        return;
    }
    dst[params.offset_dst + gid.x] = params.start + f32(gid.x) * params.step;
}
