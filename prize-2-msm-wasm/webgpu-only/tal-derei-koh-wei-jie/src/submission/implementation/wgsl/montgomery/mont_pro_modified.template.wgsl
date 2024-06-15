{{> structs }}
{{> montgomery_product_funcs }}
{{> field_functions }}
{{> bigint_functions }}

@group(0)
@binding(0)
var<storage, read_write> buf: array<BigInt>;

const NUM_WORDS = {{ num_words }}u;
const WORD_SIZE = {{ word_size }}u;
const MASK = {{ mask }}u;
const N0 = {{ n0 }}u;
const NSAFE = {{ nsafe }}u;
const TWO_POW_WORD_SIZE = {{ two_pow_word_size }}u;
const COST = {{ cost }}u;

fn get_p() -> BigInt {
    var p: BigInt;
{{{ p_limbs }}}
    return p;
}

/// The Montgomery product algorithm from
/// https://github.com/mitschabaude/montgomery#13-x-30-bit-multiplication
/// It is suitable for limb sizes 14 and 15.
fn montgomery_product(x: ptr<function, BigInt>, y: ptr<function, BigInt>) -> BigInt {
    var s: BigInt;
    var p = get_p();

    for (var i = 0u; i < NUM_WORDS; i ++) {
        var t = s.limbs[0] + (*x).limbs[i] * (*y).limbs[0];
        var tprime = t & MASK;
        var qi = (N0 * tprime) & MASK;
        var c = (t + qi * p.limbs[0]) >> WORD_SIZE;

        for (var j = 1u; j < NUM_WORDS - 1u; j ++) {
            var t = s.limbs[j] + (*x).limbs[i] * (*y).limbs[j] + qi * p.limbs[j];
            if ((j - 1u) % NSAFE == 0u) {
                t += c;
            }

            c = t >> WORD_SIZE;
            if (j % NSAFE == 0u) {
                c = t >> WORD_SIZE;
                s.limbs[j - 1u] = t & MASK;
            } else {
                s.limbs[j - 1u] = t;
            }
        }

        s.limbs[NUM_WORDS - 2u] = (*x).limbs[i] * (*y).limbs[NUM_WORDS - 1u] + qi * p.limbs[NUM_WORDS - 1u];
    }

    var c = 0u;
    for (var i = 0u; i < NUM_WORDS; i ++) {
        var v = s.limbs[i] + c;
        c = v >> WORD_SIZE;
        s.limbs[i] = v & MASK;
    }

    return conditional_reduce(&s, &p);
}

fn conditional_reduce(x: ptr<function, BigInt>, y: ptr<function, BigInt>) -> BigInt {
    /// Determine if x > y.
    var x_gt_y = bigint_gt(x, y);

    if (x_gt_y == 1u) {
        var res: BigInt;
        bigint_sub(x, y, &res);
        return res;
    }

    return *x;
}

@compute
@workgroup_size(1)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let gidx = global_id.x; 
    var a: BigInt = buf[gidx * 2u];
    var b: BigInt = buf[(gidx * 2u) + 1u];

    var c = montgomery_product(&a, &a);
    for (var i = 1u; i < COST - 1u; i ++) {
        c = montgomery_product(&c, &a);
    }
    var result = montgomery_product(&c, &b);

    buf[gidx] = result;
}
