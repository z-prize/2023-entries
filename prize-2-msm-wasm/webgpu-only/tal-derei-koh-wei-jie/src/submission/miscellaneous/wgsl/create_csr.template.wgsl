{{> structs }}
{{> bigint_funcs }}
{{> field_funcs }}
{{> ec_funcs }}
{{> montgomery_product_funcs }}

@group(0) @binding(0)
var<storage, read> points: array<Point>;
@group(0) @binding(1)
var<storage, read> scalar_chunks: array<u32>;
@group(0) @binding(2)
var<storage, read> new_point_indices: array<u32>;
@group(0) @binding(3)
var<storage, read> cluster_start_indices: array<u32>;
@group(0) @binding(4)
var<storage, read> cluster_end_indices: array<u32>;
@group(0) @binding(5)
var<storage, read_write> new_points: array<Point>;
@group(0) @binding(6)
var<storage, read_write> new_scalar_chunks: array<u32>;

@compute
@workgroup_size({{ workgroup_size }})
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    /*new_points[global_id.x] = points[global_id.x];*/

    let gidx = global_id.x; 
    let gidy = global_id.y; 
    let id = gidx * {{ num_y_workgroups }} + gidy;
    var start_idx = cluster_start_indices[id];
    var end_idx = cluster_end_indices[id];

    var acc = points[new_point_indices[start_idx]];
    for (var i = start_idx + 1u; i < end_idx; i ++) {
          acc = add_points(acc, points[new_point_indices[i]]);
    }

    new_points[id] = acc;
    new_scalar_chunks[id] = scalar_chunks[new_point_indices[start_idx]];
}
