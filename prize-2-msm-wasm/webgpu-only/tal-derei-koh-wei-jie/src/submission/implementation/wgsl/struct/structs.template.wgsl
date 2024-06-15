struct BigInt {
    limbs: array<u32, {{ num_words }}>
}

struct BigIntWide {
    limbs: array<u32, {{ num_words_mul_two }}>
}

struct BigIntMediumWide {
    limbs: array<u32, {{ num_words_plus_one }}>
}

struct Point {
  x: BigInt,
  y: BigInt,
  t: BigInt,
  z: BigInt
}