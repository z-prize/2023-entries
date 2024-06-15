{{> structs }}
{{> bigint_funcs }}
{{> field_funcs }}
{{> ec_funcs }}
{{> montgomery_product_funcs }}

// Input buffers
@group(0) @binding(0)
var<storage, read> point_x_y: array<BigInt>;
@group(0) @binding(1)
var<storage, read> point_t_z: array<BigInt>;
@group(0) @binding(2)
var<storage, read> new_point_indices: array<u32>;
@group(0) @binding(3)
var<storage, read> cluster_start_indices: array<u32>;
@group(0) @binding(4)
var<storage, read> cluster_end_indices: array<u32>;

// Output buffers
@group(0) @binding(5)
var<storage, read_write> new_point_x_y: array<BigInt>;
@group(0) @binding(6)
var<storage, read_write> new_point_t_z: array<BigInt>;

@compute
@workgroup_size({{ workgroup_size }})
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let gidx = global_id.x; 
    let gidy = global_id.y; 
    let id = gidx * {{ num_y_workgroups }} + gidy;

    let start_idx = cluster_start_indices[id];
    let end_idx = cluster_end_indices[id];
    let acc_point_idx = new_point_indices[start_idx];

    let acc_x = point_x_y[acc_point_idx * 2];
    let acc_y = point_x_y[acc_point_idx * 2 + 1];
    let acc_t = point_t_z[acc_point_idx * 2];
    let acc_z = point_t_z[acc_point_idx * 2 + 1];

    var acc = Point(acc_x, acc_y, acc_t, acc_z);

    for (var i = start_idx + 1u; i < end_idx; i ++) {
        let point_idx = new_point_indices[i];
        let pt_x = point_x_y[point_idx * 2];
        let pt_y = point_x_y[point_idx * 2 + 1];
        let pt_t = point_t_z[point_idx * 2];
        let pt_z = point_t_z[point_idx * 2 + 1];

        let pt = Point(pt_x, pt_y, pt_t, pt_z);

        acc = add_points(acc, pt);
    }

    new_point_x_y[id * 2] = acc.x;
    new_point_x_y[id * 2 + 1] = acc.y;
    new_point_t_z[id * 2] = acc.t;
    new_point_t_z[id * 2 + 1] = acc.z;
}
