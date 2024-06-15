{{> structs }}
{{> bigint_funcs }}
{{> field_funcs }}
{{> ec_funcs }}
{{> montgomery_product_funcs }}

@group(0) @binding(0)
var<storage, read_write> point_x_y: array<BigInt>;
@group(0) @binding(1)
var<storage, read_write> point_t_z: array<BigInt>;

fn get_r() -> BigInt {
    var r: BigInt;
{{{ r_limbs }}}
    return r;
}

// TODO: refactor this and scalar_mul.template.wgsl as the code is repeated
fn get_paf() -> Point {
    var result: Point;
    let r = get_r();
    result.y = r;
    result.z = r;
    return result;
}

// Double-and-add algo adapted from the ZPrize test harness
fn double_and_add(point: Point, scalar: u32) -> Point {
    // Set result to the point at infinity
    var result: Point = get_paf();

    var s = scalar;
    var temp = point;

    while (s != 0u) {
        if ((s & 1u) == 1u) {
            result = add_points(result, temp);
        }
        temp = double_point(temp);
        s = s >> 1u;
    }
    return result;
}

@compute
@workgroup_size(1)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    // Assume that there are 256 points or fewer
    var id = global_id.x;

    var chunk_size = {{ chunk_size }}u;

    let a_x = point_x_y[0];
    let a_y = point_x_y[1];
    let a_t = point_t_z[0];
    let a_z = point_t_z[1];

    var result = Point(a_x, a_y, a_t, a_z);
    let m = u32(pow(f32(2), f32(chunk_size)));
    for (var i = 1u; i < arrayLength(&point_x_y) / 2u; i ++) {
        let idx = id * 2u;
        let idx1 = idx + 1u;
        let x = point_x_y[idx];
        let y = point_x_y[idx1];
        let t = point_t_z[idx];
        let z = point_t_z[idx1];

        let pt = Point(x, y, t, z);

        result = double_and_add(result, m);
        result = add_points(result, pt);
    }

    point_x_y[0] = result.x;
    point_x_y[1] = result.y;
    point_t_z[0] = result.t;
    point_t_z[1] = result.z;
}
