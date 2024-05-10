export const U256WGSL =
`
alias U256 = array<u32, 8>;
const U256_ONE = U256(0, 0, 0, 0, 0, 0, 0, 1);
const U256_ZERO = U256(0, 0, 0, 0, 0, 0, 0, 0);

// no overflow checking for U256
fn u256_add(a: U256, b: U256) -> U256 {
  var sum = U256(0, 0, 0, 0, 0, 0, 0, 0);
  var carry: u32 = 0u;

  for (var i = 7i; i >= 0i; i--) {
    let c = a[i] + carry; 
    let d = b[i] + c;
    carry = select(0u, 1u, ((c < carry) || (d < c)));
    sum[i]= d;
  }

  return sum;
}

fn u256_rs1(a: U256) -> U256 {
  var right_shifted = U256(0, 0, 0, 0, 0, 0, 0, 0);
  var carry: u32 = 0u;
  for (var i = 0u; i < 8u; i++) {
    var componentResult = a[i] >> 1u;
    componentResult = componentResult | carry;
    right_shifted[i] = componentResult;
    carry = a[i] << 31u;
  }

  return right_shifted;
}

fn is_even(a: U256) -> bool {
  return (a[7u] & 1u) == 0u;
}

// no underflow checking for U256
fn u256_sub(a: U256, b: U256) -> U256 {
  var sub = U256(0, 0, 0, 0, 0, 0, 0, 0);
  var carry: u32 = 0u;
  for (var i = 7i; i >= 0i; i--) {
    let c = a[i] - b[i];
    let d = c - carry;
    carry = select(0u, 1u, ((c > a[i]) || (d > c)));
    sub[i]= d;
  }

  return sub;
}

fn equal(a: U256, b: U256) -> bool {
  var result = a[0] ^ b[0];
  for (var i = 1u; i < 8u; i++) {
    result = result | (a[i] ^ b[i]); 
  }
  return result == 0u;
}

// returns whether a > b
fn gt(a: U256, b: U256) -> bool {
  for (var i = 0u; i < 8u; i++) {
    if (a[i] < b[i]) {
      return false;
    }
    if (a[i] > b[i]) {
      return true;
    }
  }
  return false;
}

// returns whether a >= b
fn gte(a: U256, b: U256) -> bool {
  for (var i = 0u; i < 8u; i++) {
    if (a[i] < b[i]) {
      return false;
    }
    if (a[i] > b[i]) {
      return true;
    }
  }
  return true;
}

fn u256_double(a: U256) -> U256 {
  var double = U256(0, 0, 0, 0, 0, 0, 0, 0);
  var carry: u32 = 0u;

  for (var i = 7i; i >= 0i; i--) {
    double[i] = (a[i] << 1u) | carry;
    carry = a[i] >> 31u;
  }
  return double;
}
`;