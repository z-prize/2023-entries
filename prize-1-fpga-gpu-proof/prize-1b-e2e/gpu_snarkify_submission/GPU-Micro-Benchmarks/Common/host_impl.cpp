/***

Copyright (c) 2024, Snarkify, Inc. All rights reserved.
Dual licensed under the MIT License or the Apache License, Version 2.0.
See LICENSE for details.

Author(s):  Niall Emmart

***/

namespace host {

namespace ff {

template<class Field, bool montgomery>
class impl {
  public:
  typedef Field FIELD;
  uint64_t l[Field::limbs];

  static bool mod_is_zero(const impl& f) {
    return host::mp<Field>::mod_is_zero(f.l);
  }

  static impl zero() {
    impl res;

    for(uint32_t i=0;i<Field::limbs;i++)
      res.l[i]=0;
    return res;
  }

  static impl one() {
    impl res;

    if constexpr(montgomery) {
      for(uint32_t i=0;i<Field::limbs;i++)
        res.l[i]=Field::const_r.l[i];
    }
    else {
      res.l[0]=1;
      for(uint32_t i=1;i<Field::limbs;i++)
        res.l[i]=0;
    }
    return res;
  }

  static impl infinity() {
    const uint32_t last=Field::limbs-1;

    impl res;

    for(int i=0;i<last;i++)
      res.l[i]=0;
    res.l[last]=0x8000000000000000ull;
    return res;
  }

  static bool mod_is_equal(const impl& a, const impl& b) {
    impl check;

    host::mp<Field>::mod_sub(check.l, a.l, b.l);
    return mod_is_zero(check);
  }

  static impl mod_neg(const impl& f) {
    impl res;

    host::mp<Field>::mod_neg(res.l, f.l);
    return res;
  }

  static impl mod_dbl(const impl& f) {
    impl res;

    host::mp<Field>::mod_add(res.l, f.l, f.l);
    return res;
  }

  static impl mod_add(const impl& a, const impl& b) {
    impl res;

    host::mp<Field>::mod_add(res.l, a.l, b.l);
    return res;
  }

  static impl mod_sub(const impl& a, const impl& b) {
    impl res;

    host::mp<Field>::mod_sub(res.l, a.l, b.l);
    return res;
  }

  static impl mod_mul(const impl& a, const impl& b) {
    impl res;

    if constexpr (montgomery)
      host::mp<Field>::mont_mul(res.l, a.l, b.l);
    else
      host::mp<Field>::barrett_mul(res.l, a.l, b.l);

    return res;
  }

  static impl mod_sqr(const impl& f) {
    impl res;

    if constexpr (montgomery)
      host::mp<Field>::mont_sqr(res.l, f.l);
    else
      host::mp<Field>::barrett_sqr(res.l, f.l);
    return res;
  }

  static impl mod_inv(const impl& f) {
    impl res;

    if constexpr (montgomery)
      host::mp<Field>::mont_mul(res.l, f.l, Field::const_r_inv.l);
    else
      res=f;

    host::mp<Field>::invert(res.l, res.l);

    return res;
  }

  static impl mod_pow(const impl& f, const uint64_t k) {
    impl     res=one();
    impl     sqr=f;
    uint64_t local_k=k;

    while(local_k>0) {
      if((local_k & 0x01)!=0)
        res=res * sqr;
      sqr=sqr * sqr;
      local_k=local_k>>1;
    }
    return res;
  }

  static impl mod_pow(const impl& f, const storage<Field, false>& k) {
    impl     res;
    int32_t  bit=Field::bits-1;

    while(bit>=0 && k.l[(bit+1)>>6]==0)
      bit=bit-64;
    while(bit>=0 && !k.test_bit(bit))
      bit--;
    if(bit<0)
      return one();
    res=f;
    bit--;
    while(bit>=0) {
      res=res.sqr();
      if(k.test_bit(bit))
        res=res * f;
      bit--;
    }
    return res;
  }

  static impl generator() {
    impl res;

    if constexpr(montgomery) {
      for(uint32_t i=0;i<Field::limbs;i++)
        res.l[i]=Field::const_generator_montgomery.l[i];
    }
    else {
      for(int i=0;i<Field::limbs;i++)
        res.l[i]=Field::const_generator.l[i];
    }
    return res;
  }

  static impl generator(uint32_t k) {
    storage<Field,false> generator_k;
    uint64_t             mask=(1ull<<k)-1ull;

    // root of unity generator, where res^(2^k)==1.
    generator_k.l[0]=Field::const_storage_n.l[0]-1ull;
    for(uint32_t i=1;i<Field::packed_limbs;i++)
      generator_k.l[i]=Field::const_storage_n.l[i];

    if((generator_k.l[0] & mask)!=0)
      return one();

    host::mp_packed<Field>::shift_right_bits(generator_k.l, generator_k.l, k);
    return generator().pow(generator_k);
  }

  bool is_zero() const {
    return mod_is_zero(*this);
  }

  bool is_infinity() const {
    const uint32_t last=Field::limbs-1;

    return (l[last]>>63)!=0;
  }

  impl dbl() const {
    return mod_dbl(*this);
  }

  impl inv() const {
    return mod_inv(*this);
  }

  impl sqr() const {
    return mod_sqr(*this);
  }

  impl pow(const uint64_t k) const {
    return mod_pow(*this, k);
  }

  impl pow(const storage<Field, false> k) const {
    return mod_pow(*this, k);
  }

  // convenience methods
  bool operator==(const impl& f) const {
    return mod_is_equal(*this, f);
  }

  bool operator!=(const impl& f) const {
    return !mod_is_equal(*this, f);
  }

  impl operator-() const {
    return mod_neg(*this);
  }

  impl operator+(const impl& f) const {
    return mod_add(*this, f);
  }

  impl operator-(const impl& f) const {
    return mod_sub(*this, f);
  }

  impl operator*(const impl& f) const {
    return mod_mul(*this, f);
  }

  impl operator/(const impl& f) const {
    return mod_mul(*this, f.inv());
  }

  operator host::ff::storage<Field,false>() const {
    host::ff::storage<Field,false> res;

    host::ff::convert<Field>::storage_from_impl(res, *this);
    return res;
  }

  operator host::ff::storage<Field,true>() const {
    host::ff::storage<Field,true> res;

    host::ff::convert<Field>::storage_from_impl(res, *this);
    return res;
  }

  operator host::ff::impl<Field,!montgomery>() const {
    host::ff::impl<Field,!montgomery> res;

    host::ff::convert<Field>::impl_from_impl(res, *this);
    return res;
  }
};

} // namespace ff

} // namespace host

