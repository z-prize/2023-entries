{{> structs }}
{{> montgomery_product_funcs }}
{{> field_functions }}
{{> bigint_functions }}
{{> curve_functions }}

@group(0) @binding(0)
var<storage, read_write> output: array<Point>;

@group(0) @binding(1)
var<storage, read> row_ptr: array<u32>;

@group(0) @binding(2)
var<storage, read> points: array<Point>;

@group(0) @binding(3)
var<storage, read_write> loop_index: u32;

const N = {{ NUM_ROWS_GPU }}u;

fn get_r() -> BigInt {
    var r: BigInt;
{{{ r_limbs }}}
    return r;
}

@compute
@workgroup_size(N)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {    
    var r: BigInt = get_r();
    var inf: Point;
    inf.y = r;
    inf.z = r;

    if (global_id.x < arrayLength(&output)) {
        let row_begin = row_ptr[global_id.x];
        let row_end = row_ptr[global_id.x + 1u];
        var sum = inf;
        for (var j = row_begin; j < row_end; j++) {
            sum = add_points(sum, points[j]);
        }
        output[global_id.x] = sum;
    }
}
