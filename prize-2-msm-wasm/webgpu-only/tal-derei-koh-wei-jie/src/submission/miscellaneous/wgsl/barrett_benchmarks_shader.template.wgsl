{{> structs }}
{{> montgomery_product_funcs }}
{{> field_functions }}
{{> bigint_functions }}
{{> barrett_functions }}

@group(0)
@binding(0)
var<storage, read_write> buf: array<BigInt>;

const COST = {{ cost }}u;

@compute
@workgroup_size(64)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let gidx = global_id.x; 
    var a: BigInt = buf[gidx * 2u];
    var b: BigInt = buf[(gidx * 2u) + 1u];

    var c = field_mul(&a, &a);
    for (var i = 1u; i < COST - 1u; i ++) {
        c = field_mul(&c, &a);
    }
    var result = field_mul(&c, &b);

    buf[gidx] = result;
}
