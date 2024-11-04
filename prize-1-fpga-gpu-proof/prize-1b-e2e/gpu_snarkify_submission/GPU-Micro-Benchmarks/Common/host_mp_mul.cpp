/***

Copyright (c) 2024, Snarkify, Inc. All rights reserved.
Dual licensed under the MIT License or the Apache License, Version 2.0.
See LICENSE for details.

Author(s):  Niall Emmart

***/

namespace host {

template<class Field, uint32_t limbs>
class mp_mul {
};

template<class Field>
class mp_mul<Field, 5> {
  public:
  typedef unsigned __int128 uint128_t;

  static uint128_t s(const uint64_t a, const uint32_t shift) {
    uint128_t la=a;

    return la<<shift;
  }

  static uint128_t m(const uint64_t a, const uint64_t b) {
    uint128_t la=a, lb=b;

    return la*lb;
  }

  static void mont_normalize(uint64_t* r, const uint64_t* a) {
    const uint64_t mask52=0x000FFFFFFFFFFFFFull;

    // hard coded for 5 limbs
    const uint64_t* n=Field::const_n.l;
    uint64_t        q[5];
    uint128_t       acc;

    q[0]=Field::qTerm(a[0]);
    acc=a[0] + m(q[0], n[0]);

    acc=(acc>>52) + a[1] + m(q[0], n[1]);
    q[1]=Field::qTerm(acc);
    acc+=m(q[1], n[0]);

    acc=(acc>>52) + a[2] + m(q[0], n[2]) + m(q[1], n[1]);
    q[2]=Field::qTerm(acc);
    acc+=m(q[2], n[0]);

    acc=(acc>>52) + a[3] + m(q[0], n[3]) + m(q[1], n[2]) + m(q[2], n[1]);
    q[3]=Field::qTerm(acc);
    acc+=m(q[3], n[0]);

    acc=(acc>>52) + a[4] + m(q[0], n[4]) + m(q[1], n[3]) + m(q[2], n[2]) + m(q[3], n[1]);
    q[4]=Field::qTerm(acc);
    acc+=m(q[4], n[0]);

    acc=(acc>>52) + m(q[1], n[4]) + m(q[2], n[3]) + m(q[3], n[2]) + m(q[4], n[1]);
    r[0]=acc & mask52;

    acc=(acc>>52) + m(q[2], n[4]) + m(q[3], n[3]) + m(q[4], n[2]);
    r[1]=acc & mask52;

    acc=(acc>>52) + m(q[3], n[4]) + m(q[4], n[3]);
    r[2]=acc & mask52;

    acc=(acc>>52) + m(q[4], n[4]);
    r[3]=acc & mask52;
    r[4]=acc>>52;
  }

  static void mont_sqr(uint64_t* r, const uint64_t* f) {
    const uint64_t mask52=0x000FFFFFFFFFFFFFull;

    const uint64_t* n=Field::const_n.l;
    uint64_t        q[5];
    uint128_t       acc, dbl;

    // hard coded for 5 limbs
    acc=m(f[0], f[0]);
    q[0]=Field::qTerm(acc);
    acc+=m(q[0], n[0]);

    dbl=m(f[0], f[1]);
    acc=(acc>>52) + dbl + dbl + m(q[0], n[1]);
    q[1]=Field::qTerm(acc);
    acc+=m(q[1], n[0]);

    dbl=m(f[0], f[2]);
    acc=(acc>>52) + dbl + dbl + m(f[1], f[1]) + m(q[0], n[2]) + m(q[1], n[1]);
    q[2]=Field::qTerm(acc);
    acc+=m(q[2], n[0]);

    dbl=m(f[0], f[3]) + m(f[1], f[2]);
    acc=(acc>>52) + dbl + dbl + m(q[0], n[3]) + m(q[1], n[2]) + m(q[2], n[1]);
    q[3]=Field::qTerm(acc);
    acc+=m(q[3], n[0]);

    dbl=m(f[0], f[4]) + m(f[1], f[3]);
    acc=(acc>>52) + dbl + dbl + m(f[2], f[2]) + m(q[0], n[4]) + m(q[1], n[3]) + m(q[2], n[2]) + m(q[3], n[1]);
    q[4]=Field::qTerm(acc);
    acc+=m(q[4], n[0]);

    dbl=m(f[1], f[4]) + m(f[2], f[3]);
    acc=(acc>>52) + dbl + dbl + m(q[1], n[4]) + m(q[2], n[3]) + m(q[3], n[2]) + m(q[4], n[1]);
    r[0]=acc & mask52;

    dbl=m(f[2], f[4]);
    acc=(acc>>52) + dbl + dbl + m(f[3], f[3]) + m(q[2], n[4]) + m(q[3], n[3]) + m(q[4], n[2]);
    r[1]=acc & mask52;

    dbl=m(f[3], f[4]);
    acc=(acc>>52) + dbl + dbl + m(q[3], n[4]) + m(q[4], n[3]);
    r[2]=acc & mask52;

    acc=(acc>>52) + m(f[4], f[4]) + m(q[4], n[4]);
    r[3]=acc & mask52;
    r[4]=acc>>52;
  }

  static void mont_mul(uint64_t* r, const uint64_t* a, const uint64_t* b) {
    const uint64_t mask52=0x000FFFFFFFFFFFFFull;

    const uint64_t* n=Field::const_n.l;
    uint64_t        q[5];
    uint128_t       acc;

    // hard coded for 5 limbs
    acc=m(a[0], b[0]);
    q[0]=Field::qTerm(acc);
    acc+=m(q[0], n[0]);

    acc=(acc>>52) + m(a[0], b[1]) + m(a[1], b[0])
                  + m(q[0], n[1]);
    q[1]=Field::qTerm(acc);
    acc+=m(q[1], n[0]);

    acc=(acc>>52) + m(a[0], b[2]) + m(a[1], b[1]) + m(a[2], b[0])
                  + m(q[0], n[2]) + m(q[1], n[1]);
    q[2]=Field::qTerm(acc);
    acc+=m(q[2], n[0]);

    acc=(acc>>52) + m(a[0], b[3]) + m(a[1], b[2]) + m(a[2], b[1]) + m(a[3], b[0])
                  + m(q[0], n[3]) + m(q[1], n[2]) + m(q[2], n[1]);
    q[3]=Field::qTerm(acc);
    acc+=m(q[3], n[0]);

    acc=(acc>>52) + m(a[0], b[4]) + m(a[1], b[3]) + m(a[2], b[2]) + m(a[3], b[1]) + m(a[4], b[0])
                  + m(q[0], n[4]) + m(q[1], n[3]) + m(q[2], n[2]) + m(q[3], n[1]);
    q[4]=Field::qTerm(acc);
    acc+=m(q[4], n[0]);

    acc=(acc>>52) + m(a[1], b[4]) + m(a[2], b[3]) + m(a[3], b[2]) + m(a[4], b[1])
                  + m(q[1], n[4]) + m(q[2], n[3]) + m(q[3], n[2]) + m(q[4], n[1]);
    r[0]=acc & mask52;

    acc=(acc>>52) + m(a[2], b[4]) + m(a[3], b[3]) + m(a[4], b[2])
                  + m(q[2], n[4]) + m(q[3], n[3]) + m(q[4], n[2]);
    r[1]=acc & mask52;

    acc=(acc>>52) + m(a[3], b[4]) + m(a[4], b[3])
                  + m(q[3], n[4]) + m(q[4], n[3]);
    r[2]=acc & mask52;

    acc=(acc>>52) + m(a[4], b[4])
                  + m(q[4], n[4]);
    r[3]=acc & mask52;
    r[4]=acc>>52;
  }

//  static void mont_sqr(uint64_t* r, const uint64_t* f) {
//    mont_mul(r, f, f);
//  }

  static void barrett_mul(uint64_t* r, const uint64_t* a, const uint64_t* b) {
  }
};

template<class Field>
class mp_mul<Field, 7> {
  public:
  typedef unsigned __int128 uint128_t;

  static uint128_t s(const uint64_t a, const uint32_t shift) {
    uint128_t la=a;

    return la<<shift;
  }

  static uint128_t m(const uint64_t a, const uint64_t b) {
    uint128_t la=a, lb=b;

    return la*lb;
  }

  static void mont_shift_right(uint64_t* r, const uint64_t* a, const uint32_t bits) {
    const uint64_t mask55=0x007FFFFFFFFFFFFFull;

    const uint64_t* n=Field::const_n.l;
    uint32_t        shift=55-bits;
    uint64_t        q;
    uint128_t       acc;

    q=Field::qTerm(a[0]<<shift);
    acc=s(a[0], shift) + m(q, n[0]);

    acc=(acc>>55) + s(a[1], shift) + m(q, n[1]);
    r[0]=acc & mask55;

    acc=(acc>>55) + s(a[2], shift) + m(q, n[2]);
    r[1]=acc & mask55;

    acc=(acc>>55) + s(a[3], shift) + m(q, n[3]);
    r[2]=acc & mask55;

    acc=(acc>>55) + s(a[4], shift) + m(q, n[4]);
    r[3]=acc & mask55;

    acc=(acc>>55) + s(a[5], shift) + m(q, n[5]);
    r[4]=acc & mask55;

    acc=(acc>>55) + s(a[6], shift) + m(q, n[6]);
    r[5]=acc & mask55;
    r[6]=acc>>55;
  }

  static void mont_normalize(uint64_t* r, const uint64_t* a) {
    const uint64_t mask55=0x007FFFFFFFFFFFFFull;

    const uint64_t* n=Field::const_n.l;
    uint64_t        q[7];
    uint128_t       acc;

    q[0]=Field::qTerm(a[0]);
    acc=a[0] + m(q[0], n[0]);

    acc=(acc>>55) + a[1] + m(q[0], n[1]);
    q[1]=Field::qTerm(acc);
    acc+=m(q[1], n[0]);

    acc=(acc>>55) + a[2] + m(q[0], n[2]) + m(q[1], n[1]);
    q[2]=Field::qTerm(acc);
    acc+=m(q[2], n[0]);

    acc=(acc>>55) + a[3] + m(q[0], n[3]) + m(q[1], n[2]) + m(q[2], n[1]);
    q[3]=Field::qTerm(acc);
    acc+=m(q[3], n[0]);

    acc=(acc>>55) + a[4] + m(q[0], n[4]) + m(q[1], n[3]) + m(q[2], n[2]) + m(q[3], n[1]);
    q[4]=Field::qTerm(acc);
    acc+=m(q[4], n[0]);

    acc=(acc>>55) + a[5] + m(q[0], n[5]) + m(q[1], n[4]) + m(q[2], n[3]) + m(q[3], n[2]) + m(q[4], n[1]);
    q[5]=Field::qTerm(acc);
    acc+=m(q[5], n[0]);

    acc=(acc>>55) + a[6] + m(q[0], n[6]) + m(q[1], n[5]) + m(q[2], n[4]) + m(q[3], n[3]) + m(q[4], n[2]) + m(q[5], n[1]);
    q[6]=Field::qTerm(acc);
    acc+=m(q[6], n[0]);

    acc=(acc>>55) + m(q[1], n[6]) + m(q[2], n[5]) + m(q[3], n[4]) + m(q[4], n[3]) + m(q[5], n[2]) + m(q[6], n[1]);
    r[0]=acc & mask55;

    acc=(acc>>55) + m(q[2], n[6]) + m(q[3], n[5]) + m(q[4], n[4]) + m(q[5], n[3]) + m(q[6], n[2]);
    r[1]=acc & mask55;

    acc=(acc>>55) + m(q[3], n[6]) + m(q[4], n[5]) + m(q[5], n[4]) + m(q[6], n[3]);
    r[2]=acc & mask55;

    acc=(acc>>55) + m(q[4], n[6]) + m(q[5], n[5]) + m(q[6], n[4]);
    r[3]=acc & mask55;

    acc=(acc>>55) + m(q[5], n[6]) + m(q[6], n[5]);
    r[4]=acc & mask55;

    acc=(acc>>55) + m(q[6], n[6]);
    r[5]=acc & mask55;
    r[6]=acc>>55;
  }

  static void mont_sqr(uint64_t* r, const uint64_t* f) {
    const uint64_t mask55=0x007FFFFFFFFFFFFFull;

    const uint64_t* n=Field::const_n.l;
    uint64_t        q[7];
    uint128_t       acc, dbl;

    // hard coded for 7 limbs
    acc=m(f[0], f[0]);
    q[0]=Field::qTerm(acc);
    acc+=m(q[0], n[0]);

    dbl=m(f[0], f[1]);
    acc=(acc>>55) + dbl + dbl + m(q[0], n[1]);
    q[1]=Field::qTerm(acc);
    acc+=m(q[1], n[0]);

    dbl=m(f[0], f[2]);
    acc=(acc>>55) + dbl + dbl + m(f[1], f[1]) + m(q[0], n[2]) + m(q[1], n[1]);
    q[2]=Field::qTerm(acc);
    acc+=m(q[2], n[0]);

    dbl=m(f[0], f[3]) + m(f[1], f[2]);
    acc=(acc>>55) + dbl + dbl + m(q[0], n[3]) + m(q[1], n[2]) + m(q[2], n[1]);
    q[3]=Field::qTerm(acc);
    acc+=m(q[3], n[0]);

    dbl=m(f[0], f[4]) + m(f[1], f[3]);
    acc=(acc>>55) + dbl + dbl + m(f[2], f[2]) + m(q[0], n[4]) + m(q[1], n[3]) + m(q[2], n[2]) + m(q[3], n[1]);
    q[4]=Field::qTerm(acc);
    acc+=m(q[4], n[0]);

    dbl=m(f[0], f[5]) + m(f[1], f[4]) + m(f[2], f[3]);
    acc=(acc>>55) + dbl + dbl + m(q[0], n[5]) + m(q[1], n[4]) + m(q[2], n[3]) + m(q[3], n[2]) + m(q[4], n[1]);
    q[5]=Field::qTerm(acc);
    acc+=m(q[5], n[0]);

    dbl=m(f[0], f[6]) + m(f[1], f[5]) + m(f[2], f[4]);
    acc=(acc>>55) + dbl + dbl + m(f[3], f[3]) + m(q[0], n[6]) + m(q[1], n[5]) + m(q[2], n[4]) + m(q[3], n[3]) + m(q[4], n[2]) + m(q[5], n[1]);
    q[6]=Field::qTerm(acc);
    acc+=m(q[6], n[0]);

    dbl=m(f[1], f[6]) + m(f[2], f[5]) + m(f[3], f[4]);
    acc=(acc>>55) + dbl + dbl + m(q[1], n[6]) + m(q[2], n[5]) + m(q[3], n[4]) + m(q[4], n[3]) + m(q[5], n[2]) + m(q[6], n[1]);
    r[0]=acc & mask55;

    dbl=m(f[2], f[6]) + m(f[3], f[5]);
    acc=(acc>>55) + dbl + dbl + m(f[4], f[4]) + m(q[2], n[6]) + m(q[3], n[5]) + m(q[4], n[4]) + m(q[5], n[3]) + m(q[6], n[2]);
    r[1]=acc & mask55;

    dbl=m(f[3], f[6]) + m(f[4], f[5]);
    acc=(acc>>55) + dbl + dbl + m(q[3], n[6]) + m(q[4], n[5]) + m(q[5], n[4]) + m(q[6], n[3]);
    r[2]=acc & mask55;

    dbl=m(f[4], f[6]);
    acc=(acc>>55) + dbl + dbl + m(f[5], f[5]) + m(q[4], n[6]) + m(q[5], n[5]) + m(q[6], n[4]);
    r[3]=acc & mask55;

    dbl=m(f[5], f[6]);
    acc=(acc>>55) + dbl + dbl + m(q[5], n[6]) + m(q[6], n[5]);
    r[4]=acc & mask55;

    acc=(acc>>55) + m(f[6], f[6]) + m(q[6], n[6]);
    r[5]=acc & mask55;
    r[6]=acc>>55;
  }

  static void mont_mul(uint64_t* r, const uint64_t* a, const uint64_t* b) {
    const uint64_t mask55=0x007FFFFFFFFFFFFFull;

    const uint64_t* n=Field::const_n.l;
    uint64_t        q[7];
    uint128_t       acc;

    // hard coded for 7 limbs
    acc=m(a[0], b[0]);
    q[0]=Field::qTerm(acc);
    acc+=m(q[0], n[0]);

    acc=(acc>>55) + m(a[0], b[1]) + m(a[1], b[0])
                  + m(q[0], n[1]);
    q[1]=Field::qTerm(acc);
    acc+=m(q[1], n[0]);

    acc=(acc>>55) + m(a[0], b[2]) + m(a[1], b[1]) + m(a[2], b[0])
                  + m(q[0], n[2]) + m(q[1], n[1]);
    q[2]=Field::qTerm(acc);
    acc+=m(q[2], n[0]);

    acc=(acc>>55) + m(a[0], b[3]) + m(a[1], b[2]) + m(a[2], b[1]) + m(a[3], b[0])
                  + m(q[0], n[3]) + m(q[1], n[2]) + m(q[2], n[1]);
    q[3]=Field::qTerm(acc);
    acc+=m(q[3], n[0]);

    acc=(acc>>55) + m(a[0], b[4]) + m(a[1], b[3]) + m(a[2], b[2]) + m(a[3], b[1]) + m(a[4], b[0])
                  + m(q[0], n[4]) + m(q[1], n[3]) + m(q[2], n[2]) + m(q[3], n[1]);
    q[4]=Field::qTerm(acc);
    acc+=m(q[4], n[0]);

    acc=(acc>>55) + m(a[0], b[5]) + m(a[1], b[4]) + m(a[2], b[3]) + m(a[3], b[2]) + m(a[4], b[1]) + m(a[5], b[0])
                  + m(q[0], n[5]) + m(q[1], n[4]) + m(q[2], n[3]) + m(q[3], n[2]) + m(q[4], n[1]);
    q[5]=Field::qTerm(acc);
    acc+=m(q[5], n[0]);

    acc=(acc>>55) + m(a[0], b[6]) + m(a[1], b[5]) + m(a[2], b[4]) + m(a[3], b[3]) + m(a[4], b[2]) + m(a[5], b[1]) + m(a[6], b[0])
                  + m(q[0], n[6]) + m(q[1], n[5]) + m(q[2], n[4]) + m(q[3], n[3]) + m(q[4], n[2]) + m(q[5], n[1]);
    q[6]=Field::qTerm(acc);
    acc+=m(q[6], n[0]);

    acc=(acc>>55) + m(a[1], b[6]) + m(a[2], b[5]) + m(a[3], b[4]) + m(a[4], b[3]) + m(a[5], b[2]) + m(a[6], b[1])
                  + m(q[1], n[6]) + m(q[2], n[5]) + m(q[3], n[4]) + m(q[4], n[3]) + m(q[5], n[2]) + m(q[6], n[1]);
    r[0]=acc & mask55;

    acc=(acc>>55) + m(a[2], b[6]) + m(a[3], b[5]) + m(a[4], b[4]) + m(a[5], b[3]) + m(a[6], b[2])
                  + m(q[2], n[6]) + m(q[3], n[5]) + m(q[4], n[4]) + m(q[5], n[3]) + m(q[6], n[2]);
    r[1]=acc & mask55;

    acc=(acc>>55) + m(a[3], b[6]) + m(a[4], b[5]) + m(a[5], b[4]) + m(a[6], b[3])
                  + m(q[3], n[6]) + m(q[4], n[5]) + m(q[5], n[4]) + m(q[6], n[3]);
    r[2]=acc & mask55;

    acc=(acc>>55) + m(a[4], b[6]) + m(a[5], b[5]) + m(a[6], b[4])
                  + m(q[4], n[6]) + m(q[5], n[5]) + m(q[6], n[4]);
    r[3]=acc & mask55;

    acc=(acc>>55) + m(a[5], b[6]) + m(a[6], b[5])
                  + m(q[5], n[6]) + m(q[6], n[5]);
    r[4]=acc & mask55;

    acc=(acc>>55) + m(a[6], b[6])
                  + m(q[6], n[6]);
    r[5]=acc & mask55;
    r[6]=acc>>55;
  }

  static void barrett_mul(uint64_t* r, const uint64_t* a, const uint64_t* b) {
  }
};

} // namespace host

