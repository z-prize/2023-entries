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
const COST = {{ cost }}u;

fn get_p() -> BigInt {
    var p: BigInt;
{{{ p_limbs }}}
    return p;
}

fn gen_p_medium_wide() -> BigIntMediumWide {
    var p = get_p();
    var m: BigIntMediumWide;
    for (var i = 0u; i < NUM_WORDS; i ++) {
        m.limbs[i] = p.limbs[i];
    }
    return m;
}

/// The CIOS method for Montgomery multiplication from Tolga Acar's thesis:
/// High-Speed Algorithms & Architectures For Number-Theoretic Cryptosystems
/// https://www.proquest.com/openview/1018972f191afe55443658b28041c118/1
fn montgomery_product(a: ptr<function, BigInt>, b: ptr<function, BigInt>) -> BigInt {
    var n = gen_p_medium_wide();
    var n0 = 65535u;

    var t: array<u32, {{ num_words_plus_two }}u>;
    var x: BigInt;

    for (var i = 0u; i < NUM_WORDS; i ++) {
        var c = 0u;
        for (var j = 0u; j < NUM_WORDS; j ++) {
            var r = t[j] + (*a).limbs[j] * (*b).limbs[i] + c;
            c = r >> WORD_SIZE;
            t[j] = r & MASK;
        }
        var r = t[NUM_WORDS] + c;
        t[NUM_WORDS + 1u] = r >> WORD_SIZE;
        t[NUM_WORDS] = r & MASK;

        var m = (t[0] * n0) % 65536u;
        r = t[0] + m * n.limbs[0];
        c = r >> WORD_SIZE;

        for (var j = 1u; j < NUM_WORDS; j ++) {
            r = t[j] + m * n.limbs[j] + c;
            c = r >> WORD_SIZE;
            t[j - 1u] = r & MASK;
        }

        r = t[NUM_WORDS] + c;
        c = r >> WORD_SIZE;
        t[NUM_WORDS - 1u] = r & MASK;
        t[NUM_WORDS] = t[NUM_WORDS + 1u] + c;
    }

    /// Check if t > n. If so, return n - t. Else, return t.
    var t_lt_n = false;
    for (var idx = 0u; idx < NUM_WORDS + 1u; idx ++) {
        var i = NUM_WORDS - 1u - idx;
        if (t[i] < n.limbs[i]) {
            t_lt_n = true;
            break;
        } else if (t[i] > n.limbs[i]) {
            break;
        }
    }

    var r: BigInt;
    if (t_lt_n) {
        for (var i = 0u; i < NUM_WORDS; i ++) {
            r.limbs[i] = t[i];
        }
        return r;
    } else {
        var borrow = 0u;
        var t_minus_n: BigIntMediumWide;
        for (var i = 0u; i < NUM_WORDS; i ++) {
            t_minus_n.limbs[i] = t[i] - n.limbs[i] - borrow;
            if (t[i] < (n.limbs[i] + borrow)) {
                t_minus_n.limbs[i] = t_minus_n.limbs[i] + 65536u;
                borrow = 1u;
            } else {
                borrow = 0u;
            }
            x.limbs[i] = t_minus_n.limbs[i];
        }
        return x;
    }

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
