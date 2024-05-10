export const CurveWGSL = `
struct AffinePoint {
  x: Field,
  y: Field
}

struct Point {
  x: Field,
  y: Field,
  t: Field,
  z: Field
}

const INF_POINT = Point (U256_ZERO, U256_ONE, U256_ZERO, U256_ONE);

fn is_inf(p: Point) -> bool {
  var a = p.t[0];
  for (var i = 1u; i < 8u; i++) {
    a = a | p.t[i];
  }
  return a == 0u;
}

fn mul_by_a(f: Field) -> Field {
  // mul by a is just negation of f
  return u256_sub(FIELD_ORDER, f);
}

fn add_points(p1: Point, p2: Point) -> Point {
  if (is_inf(p1)) {
    return p2;
  } 
  if (is_inf(p2)) {
    return p1;
  }
  var pt = Point(
    Field(),
    Field(),
    field_mul(p1.z, p2.t),
    field_mul(p1.t, p2.z),
  );
  if (equal(pt.t, pt.z)) {
    return double_point(p1);
  }
  var a = field_double(pt.t);
  var b = field_double(pt.z);
  pt.x = field_add(p2.y, p2.x);
  pt.y = field_add(p1.y, p1.x);
  pt.t = field_add(b, a);
  pt.z = field_sub(b, a);
  a = field_sub(p1.y, p1.x);
  b = field_sub(p2.y, p2.x);  
  pt.x = field_mul(pt.x, a);
  pt.y = field_mul(pt.y, b);
  a = field_sub(pt.y, pt.x);
  b = field_add(pt.y, pt.x);
  pt.x = field_mul(pt.t, a);
  pt.y = field_mul(pt.z, b);
  pt.t = field_mul(pt.t, pt.z);
  pt.z = field_mul(a, b);
  return pt;
}

fn double_point(p: Point) -> Point {
  if (is_inf(p)) {
    return p;
  } 
  var pt = Point(
    field_mul(p.x, p.x),
    field_mul(p.y, p.y),
    field_add(p.x, p.y),
    field_mul(p.z, p.z),
  );
  pt.t = field_mul(pt.t, pt.t);
  pt.t = field_sub(pt.t, pt.x);
  pt.t = field_sub(pt.t, pt.y);
  pt.x = mul_by_a(pt.x);
  var g = field_add(pt.x, pt.y);
  var h = field_sub(pt.x, pt.y);
  pt.z = field_double(pt.z);
  pt.z = field_sub(g, pt.z);
  pt.x = field_mul(pt.t, pt.z);
  pt.y = field_mul(g, h);
  pt.t = field_mul(pt.t, h);
  pt.z = field_mul(pt.z, g);
  return pt;
}
`;