// Input buffers
@group(0) @binding(0)
var<storage, read> scalar_chunks: array<u32>;
@group(0) @binding(1)
var<storage, read> new_point_indices: array<u32>;
@group(0) @binding(2)
var<storage, read> cluster_start_indices: array<u32>;

// Output buffers
@group(0) @binding(3)
var<storage, read_write> new_scalar_chunks: array<u32>;

@compute
@workgroup_size({{ workgroup_size }})
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let gidx = global_id.x; 
    let gidy = global_id.y; 
    let id = gidx * {{ num_y_workgroups }} + gidy;

    let start_idx = cluster_start_indices[id];

    new_scalar_chunks[id] = scalar_chunks[new_point_indices[start_idx]];

    // When cluster_start_indices terminates
    if (id > 0u && start_idx == 0u) {
        new_scalar_chunks[id] = 0u;
    }
}
