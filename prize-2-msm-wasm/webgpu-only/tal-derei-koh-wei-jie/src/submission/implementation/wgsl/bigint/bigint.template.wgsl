fn bigint_double(a: ptr<function, BigInt>, res: ptr<function, BigInt>) -> u32 {
    var carry: u32 = 0u;
    for (var j: u32 = 0u; j < NUM_WORDS; j ++) {
        let c: u32 = ((*a).limbs[j] * 2u) + carry;
        (*res).limbs[j] = c & MASK;
        carry = c >> WORD_SIZE;
    }
    return carry;
}

fn bigint_add(a: ptr<function, BigInt>, b: ptr<function, BigInt>, res: ptr<function, BigInt>) -> u32 {
    var carry: u32 = 0u;
    for (var j: u32 = 0u; j < NUM_WORDS; j ++) {
        let c: u32 = (*a).limbs[j] + (*b).limbs[j] + carry;
        (*res).limbs[j] = c & MASK;
        carry = c >> WORD_SIZE;
    }
    return carry;
}

fn bigint_sub(a: ptr<function, BigInt>, b: ptr<function, BigInt>, res: ptr<function, BigInt>) -> u32 {
    var borrow: u32 = 0u;
    for (var i: u32 = 0u; i < NUM_WORDS; i = i + 1u) {
        (*res).limbs[i] = (*a).limbs[i] - (*b).limbs[i] - borrow;
        if ((*a).limbs[i] < ((*b).limbs[i] + borrow)) {
            (*res).limbs[i] += TWO_POW_WORD_SIZE;
            borrow = 1u;
        } else {
            borrow = 0u;
        }
    }
    return borrow;
}

fn bigint_gt(x: ptr<function, BigInt>, y: ptr<function, BigInt>) -> u32 {
    for (var idx = 0u; idx < NUM_WORDS; idx ++) {
        var i = NUM_WORDS - 1u - idx;
        if ((*x).limbs[i] < (*y).limbs[i]) {
            return 0u;
        } else if ((*x).limbs[i] > (*y).limbs[i]) {
            return 1u;
        }
    }
    return 0u;
}