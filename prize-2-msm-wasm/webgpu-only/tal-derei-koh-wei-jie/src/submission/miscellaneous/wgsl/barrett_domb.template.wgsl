/*
 * Adapted from https://github.com/ingonyama-zk/modular_multiplication
 */

struct BigIntMediumWide {
    limbs: array<u32, {{ num_words_plus_one }}>
}

struct BigIntWide {
    limbs: array<u32, {{ num_words_mul_two }}>
}

const W_MASK = {{ w_mask }}u;
const SLACK = {{ slack }}u;

fn get_m() -> BigInt {
    var m: BigInt;
{{{ m_limbs }}}
    return m;
}

fn get_p_wide() -> BigIntWide {
    var p: BigIntWide;
{{{ p_limbs }}}
    return p;
}

fn machine_multiply(a: u32, b: u32) -> vec2<u32> {
    let ab = a * b;
    let hi = ab >> WORD_SIZE;
    let lo = ab & MASK;
    return vec2(lo, hi);
}

fn machine_two_digit_add(a: vec2<u32>, b: vec2<u32>) -> vec3<u32> {
    var carry = 0u;
    var res = vec3(0u, 0u, 0u);
    for (var i = 0u; i < 2u; i ++) {
        let sum = a[i] + b[i] + carry;
        res[i] = sum & MASK;
        carry = sum >> WORD_SIZE;
    }
    res[2] = carry;
    return res;
}

/*
 * Bitshift to the left. The shift value must be greater than the word size.
 */
fn mp_shifter_left(a: ptr<function, BigIntWide>, shift: u32) -> BigIntWide {
    var res: BigIntWide;
    var carry = 0u;
    let x = shift - WORD_SIZE;
    for (var i = 1u; i < NUM_WORDS * 2u; i ++) {
        res.limbs[i] = (((*a).limbs[i - 1] << x) & W_MASK) + carry;
        carry = (*a).limbs[i - 1] >> (WORD_SIZE - x);
    }
    return res;
}

fn mp_shifter_right(a: ptr<function, BigIntMediumWide>, shift: u32) -> BigInt {
    var res: BigInt;
    var borrow = 0u;
    let borrow_shift = WORD_SIZE - shift;
    let two_w = 1u << WORD_SIZE;
    for (var idx = 0u; idx < NUM_WORDS; idx ++) {
        let i = NUM_WORDS - idx - 1u;
        let new_borrow = (*a).limbs[i] << borrow_shift;
        res.limbs[i] = (((*a).limbs[i] >> shift) | borrow) % two_w;
        borrow = new_borrow;
    }
    return res;
}

fn mp_msb_multiply(a: ptr<function, BigIntWide>, b: ptr<function, BigInt>) -> BigIntMediumWide {
    var c: BigIntMediumWide;

    for (var l = NUM_WORDS - 1u; l < NUM_WORDS * 2u - 1u; l ++) {
        let i_min = l - (NUM_WORDS - 1u);
        for (var i = i_min; i < NUM_WORDS; i ++) {
            let v = l + 1u - NUM_WORDS;
            let mult_res = machine_multiply((*a).limbs[NUM_WORDS + i], (*b).limbs[l-i]);
            let add_res = machine_two_digit_add(mult_res, vec2(c.limbs[v], c.limbs[v+1]));

            c.limbs[v] = add_res[0];
            c.limbs[v + 1] = add_res[1];
            c.limbs[v + 2] = c.limbs[v + 2] + add_res[2];
        }
    }

    return c;
}

fn mp_lsb_multiply(a: ptr<function, BigInt>, b: ptr<function, BigInt>) -> BigIntMediumWide {
    var c: BigIntMediumWide;
    for (var l = 0u; l < NUM_WORDS; l ++) {
        let i_min = max(0i, i32(l) - (i32(NUM_WORDS) - 1i));
        let i_max = min(i32(l), (i32(NUM_WORDS) - 1i)) + 1i;  // + 1 for inclusive
        for (var i = i_min; i < i_max; i ++) {
            let mult_res = machine_multiply((*a).limbs[i], (*b).limbs[l - u32(i)]);
            let add_res = machine_two_digit_add(mult_res, vec2(c.limbs[l], c.limbs[l + 1]));
            c.limbs[l] = add_res[0];
            c.limbs[l + 1] = add_res[1];
            c.limbs[l + 2] = c.limbs[l + 2] + add_res[2];
        }
    }
    return c;
}

fn mp_adder(a: ptr<function, BigIntMediumWide>, b: ptr<function, BigIntWide>) -> BigIntMediumWide {
    var c: BigIntMediumWide;
    var carry = 0u;
    for (var i = 0u; i < NUM_WORDS; i ++) {
        let x = (*a).limbs[i + 1u] + (*b).limbs[NUM_WORDS + i] + carry;
        c.limbs[i] = x & MASK;
        carry = x >> WORD_SIZE;
    }
    return c;
}

fn mp_subtracter(a: ptr<function, BigIntWide>, b: ptr<function, BigIntMediumWide>) -> BigInt {
    var res: BigInt;
    var borrow = 0u;
    for (var i = 0u; i < NUM_WORDS; i ++) {
        res.limbs[i] = (*a).limbs[i] - (*b).limbs[i] - borrow;
        if ((*a).limbs[i] < ((*b).limbs[i] + borrow)) {
            res.limbs[i] += TWO_POW_WORD_SIZE;
            borrow = 1u;
        } else {
            borrow = 0u;
        }
    }
    return res;
}

// n^2 + (2n - 1)
fn mp_full_multiply(a: ptr<function, BigInt>, b: ptr<function, BigInt>) -> BigIntWide {
    var res: BigIntWide;
    for (var i = 0u; i < NUM_WORDS; i = i + 1u) {
        for (var j = 0u; j < NUM_WORDS; j = j + 1u) {
            let c = (*a).limbs[i] * (*b).limbs[j];
            res.limbs[i+j] += c & W_MASK;
            res.limbs[i+j+1] += c >> WORD_SIZE;
        }   
    }

    // start from 0 and carry the extra over to the next index
    for (var i = 0u; i < 2 * NUM_WORDS - 1; i = i + 1u) {
        res.limbs[i+1] += res.limbs[i] >> WORD_SIZE;
        res.limbs[i] = res.limbs[i] & W_MASK;
    }
    return res;
}

fn mp_subtract_red(a: ptr<function, BigInt>, b: ptr<function, BigInt>) -> BigInt {
    var res = *a;
    var r: BigInt;
    while (bigint_gt(&res, b) == 1u) {
        bigint_sub(&res, b, &r);
        res = r;
    }
    return res;
}

fn field_mul(a: ptr<function, BigInt>, b: ptr<function, BigInt>) -> BigInt {
    var ab = mp_full_multiply(a, b);
    let z = {{ z }}u;

    // AB msb extraction (+ shift)
    var ab_shift: BigIntWide = mp_shifter_left(&ab, z * 2u);

    // L estimation
    var m = get_m();

    var l: BigIntMediumWide = mp_msb_multiply(&ab_shift, &m); // calculate l estimator (MSB multiply)

    var l_add_ab_msb: BigIntMediumWide = mp_adder(&l, &ab_shift);

    var l2: BigInt = mp_shifter_right(&l_add_ab_msb, z);

    var p: BigInt = get_p();

    // LS calculation
    var ls_mw: BigIntMediumWide = mp_lsb_multiply(&l2, &p);

    var result: BigInt = mp_subtracter(&ab, &ls_mw);
    return mp_subtract_red(&result, &p);
}
