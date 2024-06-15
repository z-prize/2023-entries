{{> structs }}
{{> bigint_funcs }}
{{> field_funcs }}
{{> barrett_funcs }}
{{> montgomery_product_funcs }}

@group(0) @binding(0)
var<storage, read_write> points: array<Point>;

fn get_r() -> BigInt {
    var r: BigInt;
{{{ r_limbs }}}
    return r;
}

@compute
@workgroup_size({{ workgroup_size }})
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let gidx = global_id.x; 
    let gidy = global_id.y; 

    let id = gidx * {{ num_y_workgroups }} + gidy;
    let pt = points[id];

    var r = get_r();

    var ptx = pt.x;
    var pty = pt.y;
    var xr = field_mul(&ptx, &r);
    var yr = field_mul(&pty, &r);
    let tr = montgomery_product(&xr, &yr);
    let z = r;
    
    let new_pt = Point(xr, yr, tr, z);
    points[id] = new_pt;
}
