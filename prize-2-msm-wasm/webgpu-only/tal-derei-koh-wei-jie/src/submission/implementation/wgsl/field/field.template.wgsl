fn fr_add(a: ptr<function, BigInt>, b: ptr<function, BigInt>) -> BigInt { 
    var res: BigInt;
    bigint_add(a, b, &res);
    return fr_reduce(&res);
}

fn fr_reduce(a: ptr<function, BigInt>) -> BigInt {
    var res: BigInt;
    var p: BigInt = get_p();
    var underflow = bigint_sub(a, &p, &res);
    if (underflow == 1u) {
        return *a;
    }

    return res;
}

fn fr_sub(a: ptr<function, BigInt>, b: ptr<function, BigInt>) -> BigInt { 
    var res: BigInt;
    
    var c = bigint_gt(a, b);
    /// a <= b.
    if (c == 0u) { 
        var r: BigInt;
        bigint_sub(b, a, &r);
        var p = get_p();
        bigint_sub(&p, &r, &res);
        return res;
    } 
    /// a > b.
    else { 
        bigint_sub(a, b, &res);
        return res;
    }
}