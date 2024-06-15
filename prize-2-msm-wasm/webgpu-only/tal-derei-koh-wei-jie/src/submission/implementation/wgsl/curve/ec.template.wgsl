fn get_edwards_d() -> BigInt {
    var d: BigInt;
{{{ d_limbs }}}
    return d;
}

fn double_point(p1: Point) -> Point {
    var p1x = p1.x;
    var p1y = p1.y;
    var p1z = p1.z;

    var a = montgomery_product(&p1x, &p1x);
    var b = montgomery_product(&p1y, &p1y);
    var a_p_b = fr_add(&a, &b);
    var z1_m_z1 = montgomery_product(&p1z, &p1z);
    var c = fr_add(&z1_m_z1, &z1_m_z1);
    var p = get_p();
    var d = fr_sub(&p, &a);
    var x1_m_y1 = fr_add(&p1x, &p1y);
    var x1y1_m_x1y1 = montgomery_product(&x1_m_y1, &x1_m_y1);
    var e = fr_sub(&x1y1_m_x1y1, &a_p_b);
    var g = fr_add(&d, &b);
    var f = fr_sub(&g, &c);
    var h = fr_sub(&d, &b);
    var x3 = montgomery_product(&e, &f);
    var y3 = montgomery_product(&g, &h);
    var t3 = montgomery_product(&e, &h);
    var z3 = montgomery_product(&f, &g);
    return Point(x3, y3, t3, z3);
}

/// add-2008-hwcd https://eprint.iacr.org/2008/522.pdf section 3.1, p5 (9M + 2D)
/// https://hyperelliptic.org/EFD/g1p/auto-twisted-extended.html#addition-add-2008-hwcd.
fn add_points(p1: Point, p2: Point) -> Point {
    var p1x = p1.x;
    var p2x = p2.x;
    var p1y = p1.y;
    var p2y = p2.y;
    var p1t = p1.t;
    var p2t = p2.t;
    var p1z = p1.z;
    var p2z = p2.z;

    var a = montgomery_product(&p1x, &p2x);
    var b = montgomery_product(&p1y, &p2y);
    var a_p_b = fr_add(&a, &b);
    var t2 = montgomery_product(&p1t, &p2t);
    var EDWARDS_D = get_edwards_d();
    var c = montgomery_product(&EDWARDS_D, &t2);
    var d = montgomery_product(&p1z, &p2z);
    var xpy = fr_add(&p1x, &p1y);
    var xpy2 = fr_add(&p2x, &p2y);
    var e = montgomery_product(&xpy, &xpy2);
    e = fr_sub(&e, &a_p_b);
    var f = fr_sub(&d, &c);
    var g = fr_add(&d, &c);
    var p = get_p();
    var a_neg = fr_sub(&p, &a);
    var h = fr_sub(&b, &a_neg);
    var added_x = montgomery_product(&e, &f);
    var added_y = montgomery_product(&g, &h);
    var added_t = montgomery_product(&e, &h);
    var added_z = montgomery_product(&f, &g);

    return Point(added_x, added_y, added_t, added_z);
}