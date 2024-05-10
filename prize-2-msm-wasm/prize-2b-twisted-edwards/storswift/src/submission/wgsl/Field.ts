export const FieldWGSL = 
`
alias Field = U256;
alias FieldWide = array<u32, 16>;
// 8444461749428370424248824938781546531375899335154063827935233455917409239041
const FIELD_ORDER = Field(313222494, 2586617174, 1622428958, 1547153409, 1504343806, 3489660929, 168919040, 1);

// 508595941311779472113692600146818027278633330499214071737745792929336755579
const FIELD_ORDER_RR = Field(18864871, 4025600313, 2815164559, 3856434579, 3425445813, 2288015647, 634746810, 3093398907);
const FIELD_ORDER_M0 = 65535u;
const WIDE_MASK = 65535u;
const WIDE_LIMBS = 16u;

fn field_reduce(a: Field) -> Field {
  if (gte(a, FIELD_ORDER)) {
    return u256_sub(a, FIELD_ORDER);
  }
  return a;
}

fn field_add(a: Field, b: Field) -> Field {
  return field_reduce(u256_add(a, b));
}

fn field_sub(a: Field, b: Field) -> Field {
  var sub = u256_sub(a, b);
  if (gt(b, a)) {
    sub = u256_add(sub, FIELD_ORDER);
  }
  return sub;
}

fn field_double(a: Field) -> Field {
  return field_reduce(u256_double(a));
}

const FIELD_ORDER_WIDE = FieldWide(
  FIELD_ORDER[7] & WIDE_MASK, FIELD_ORDER[7] >> 16u,
  FIELD_ORDER[6] & WIDE_MASK, FIELD_ORDER[6] >> 16u,
  FIELD_ORDER[5] & WIDE_MASK, FIELD_ORDER[5] >> 16u,
  FIELD_ORDER[4] & WIDE_MASK, FIELD_ORDER[4] >> 16u,
  FIELD_ORDER[3] & WIDE_MASK, FIELD_ORDER[3] >> 16u,
  FIELD_ORDER[2] & WIDE_MASK, FIELD_ORDER[2] >> 16u,
  FIELD_ORDER[1] & WIDE_MASK, FIELD_ORDER[1] >> 16u,
  FIELD_ORDER[0] & WIDE_MASK, FIELD_ORDER[0] >> 16u
  );

// Montgomery multiplication
// https://eprint.iacr.org/2022/1400.pdf
fn field_mul(a32: Field, b32: Field) -> Field  {
  var t = FieldWide();
  var a = FieldWide(
    a32[7] & WIDE_MASK, a32[7] >> 16u,
    a32[6] & WIDE_MASK, a32[6] >> 16u,
    a32[5] & WIDE_MASK, a32[5] >> 16u,
    a32[4] & WIDE_MASK, a32[4] >> 16u,
    a32[3] & WIDE_MASK, a32[3] >> 16u,
    a32[2] & WIDE_MASK, a32[2] >> 16u,
    a32[1] & WIDE_MASK, a32[1] >> 16u,
    a32[0] & WIDE_MASK, a32[0] >> 16u,
  );
  var b = FieldWide(
    b32[7] & WIDE_MASK, b32[7] >> 16u,
    b32[6] & WIDE_MASK, b32[6] >> 16u,
    b32[5] & WIDE_MASK, b32[5] >> 16u,
    b32[4] & WIDE_MASK, b32[4] >> 16u,
    b32[3] & WIDE_MASK, b32[3] >> 16u,
    b32[2] & WIDE_MASK, b32[2] >> 16u,
    b32[1] & WIDE_MASK, b32[1] >> 16u,
    b32[0] & WIDE_MASK, b32[0] >> 16u,
  );
  var c: u32 = 0u;
  for(var i = 0u; i < WIDE_LIMBS; i++) {
    var carry = 0u;
    for(var j = 0u; j < WIDE_LIMBS; j++) {
      carry = a[j] * b[i] + t[j] + carry;  
      t[j] = carry & WIDE_MASK;
      carry = carry >> 16u;
    }
    c = carry;

    var m = (0u - t[0]) & WIDE_MASK;
    carry = m * FIELD_ORDER_WIDE[0] + t[0];
    carry = carry >> 16u;
    for(var j = 1u; j < WIDE_LIMBS; j++) {
      carry = m * FIELD_ORDER_WIDE[j] + t[j] + carry;
      t[j - 1] = carry & WIDE_MASK;
      carry = carry >> 16u;
    }
    t[WIDE_LIMBS - 1u] = c + carry;
  }

  var result = Field(
    t[14] | (t[15]<< 16u),
    t[12] | (t[13]<< 16u),
    t[10] | (t[11]<< 16u),
    t[ 8] | (t[ 9]<< 16u),
    t[ 6] | (t[ 7]<< 16u),
    t[ 4] | (t[ 5]<< 16u),
    t[ 2] | (t[ 3]<< 16u),
    t[ 0] | (t[ 1]<< 16u)
  );

  if(gte(result, FIELD_ORDER)) {
    result = u256_sub(result, FIELD_ORDER);
  }
  return result;
}
// Montgomery squaring
// https://eprint.iacr.org/2022/1400.pdf
fn field_sqr(a32: Field) -> Field {
  var t = FieldWide();
  var a = FieldWide(
    a32[7] & WIDE_MASK, a32[7] >> 16u,
    a32[6] & WIDE_MASK, a32[6] >> 16u,
    a32[5] & WIDE_MASK, a32[5] >> 16u,    
    a32[4] & WIDE_MASK, a32[4] >> 16u,
    a32[3] & WIDE_MASK, a32[3] >> 16u,
    a32[2] & WIDE_MASK, a32[2] >> 16u,
    a32[1] & WIDE_MASK, a32[1] >> 16u,
    a32[0] & WIDE_MASK, a32[0] >> 16u,
  );
  var c: u32 = 0u;
  for(var i = 0u; i < WIDE_LIMBS; i++) {
    var carry = a[i] * a[i] + t[i];
    t[i] = carry & WIDE_MASK;
    carry = carry >> 16u;

    var h = 0u;
    for(var j = i + 1u; j < WIDE_LIMBS; j++) {
      var d = a[i] * a[j];
      var d0 = d & WIDE_MASK;
      var d1 = d >> 16u;
      var e = d + t[j] + carry;
      var e0 = e & WIDE_MASK;
      var e1 = e >> 16u;
      carry = d0 + e0;
      t[j] = carry & WIDE_MASK;
      carry = carry >> 16u;
      h = e1 + d1 + carry + h;
      carry = h & WIDE_MASK;
      h = h >> 16u;
    }
    c = carry;

    var m = (0u - t[0]) & WIDE_MASK;
    carry = m * FIELD_ORDER_WIDE[0] + t[0];
    carry = carry >> 16u;
    for(var j = 1u; j < WIDE_LIMBS; j++) {
      carry = m * FIELD_ORDER_WIDE[j] + t[j] + carry;
      t[j - 1] = carry & WIDE_MASK;
      carry = carry >> 16u;
    }
    t[WIDE_LIMBS - 1u] = c + carry;
  }

  var result = Field(
    t[14] | (t[15]<< 16u),
    t[12] | (t[13]<< 16u),
    t[10] | (t[11]<< 16u),
    t[ 8] | (t[ 9]<< 16u),
    t[ 6] | (t[ 7]<< 16u),
    t[ 4] | (t[ 5]<< 16u),
    t[ 2] | (t[ 3]<< 16u),
    t[ 0] | (t[ 1]<< 16u)
  );

  if(gte(result, FIELD_ORDER)) {
    result = u256_sub(result, FIELD_ORDER);
  }
  return result;
}

// assume that the input is NOT 0, as there's no inverse for 0
// this function implements the Guajardo Kumar Paar Pelzl (GKPP) algorithm,
// Algorithm 16 (BEA for inversion in Fp)
fn field_inverse(num: Field) -> Field {
  var u: Field = num;
  var v: Field = FIELD_ORDER;
  var b: Field = U256_ONE;
  var c: Field = U256_ZERO;

  while (!equal(u, U256_ONE) && !equal(v, U256_ONE)) {
    while (is_even(u)) {
      // divide by 2
      u = u256_rs1(u);

      if (is_even(b)) {
        // divide by 2
        b = u256_rs1(b);
      } else {
        b = u256_add(b, FIELD_ORDER);
        b = u256_rs1(b);
      }
    }

    while (is_even(v)) {
      // divide by 2
      v = u256_rs1(v);
      if (is_even(c)) {
        c = u256_rs1(c);
      } else {
        c = u256_add(c, FIELD_ORDER);
        c = u256_rs1(c);
      }
    }

    if (gte(u, v)) {
      u = u256_sub(u, v);
      b = field_sub(b, c);
    } else {
      v = u256_sub(v, u);
      c = field_sub(c, b);
    }
  }

  if (equal(u, U256_ONE)) {
    return field_reduce(b);
  } else {
    return field_reduce(c);
  }
}
`