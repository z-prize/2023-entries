@group(0) @binding(0)
var<storage, read_write> data: array<u32>;

@compute
@workgroup_size(1)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    var a: array<array<u32, 1u>, 1u>;
    data[global_id.x] = a[0][0] + data[global_id.x];
}
