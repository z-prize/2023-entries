{{> structs }}
{{> montgomery_product_funcs }}
{{> field_functions }}
{{> bigint_functions }}
{{> curve_functions }}

@group(0) @binding(0)
var<storage, read_write> output: array<Point>;

@group(0) @binding(1)
var<storage, read> col_idx: array<u32>;

@group(0) @binding(2)
var<storage, read> row_ptr: array<u32>;

@group(0) @binding(3)
var<storage, read> points: array<Point>;

@group(0) @binding(4)
var<storage, read_write> loop_index: u32;

fn get_r() -> BigInt {
    var r: BigInt;
{{{ r_limbs }}}
    return r;
}

@compute
@workgroup_size(1)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    // During the first pass, store the point at infinity in the output buffer.
    if (loop_index == 0u) {
        for (var i = 0u; i < arrayLength(&output); i ++) {
            var y: BigInt = get_r();
            var z: BigInt = get_r();

            var inf: Point;
            inf.y = y;
            inf.z = z;
            output[global_id.x + i] = inf;
        }
    }

    // Perform SMTVP on an array of 1s
    let i = loop_index;
    let row_start = row_ptr[global_id.x + i];
    let row_end = row_ptr[global_id.x + i + 1];
    for (var j = row_start; j < row_end; j ++) {
        // Note that points[global_id.x + j] is not multiplied by anything
        // since the vector is an array of 1
        var temp = points[global_id.x + j];
        var col = col_idx[global_id.x + j];
        var ycol = output[global_id.x + col];
        var res = add_points(ycol, temp);

        // Store the result
        output[global_id.x + col] = res;
    }
}
