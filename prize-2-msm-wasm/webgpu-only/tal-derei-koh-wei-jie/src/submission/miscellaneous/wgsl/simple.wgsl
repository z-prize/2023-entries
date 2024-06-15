@group(0) @binding(0)
var<storage, read_write> data: array<u32>;

@compute
@workgroup_size(1)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    data[global_id.x] = data[global_id.x];
}
