/***

Copyright (c) 2024, Snarkify, Inc. All rights reserved.
Dual licensed under the MIT License or the Apache License, Version 2.0.
See LICENSE for details.

Author(s):  Niall Emmart

***/

namespace host {

namespace ff {

template<class Field, bool montgomery>
class storage {
  public:
  typedef Field FIELD;

  uint64_t l[Field::packed_limbs];

  static storage zero() {
    storage res;

    for(uint32_t i=0;i<Field::packed_limbs;i++)
      res.l[i]=0ull;
    return res;
  }

  bool is_zero() const {
    uint64_t limb_or=l[0];

    for(uint32_t i=1;i<Field::packed_limbs;i++)
      limb_or=limb_or | l[0];
    return limb_or==0;
  }

  bool test_bit(const uint32_t bit) const {
    uint64_t value;

    if(bit>Field::packed_limbs*64)
      return false;
    value=l[bit>>6]>>(bit & 0x3F);
    return (value & 0x01)!=0;
  }

  storage mod_neg() {
    storage res;

    if(is_zero())
      return zero();
    host::mp_packed<Field>::sub(res.l, Field::const_storage_n.l, l);
    return res;
  }

  storage mod_add(const storage& add) {
    storage res;

    host::mp_packed<Field>::add(res.l, l, add.l);
    if(host::mp_packed<Field>::compare(res.l, Field::const_storage_n.l)>=0)
      host::mp_packed<Field>::sub(res.l, res.l, Field::const_storage_n.l);
    return res;
  }

  storage mod_sub(const storage& sub) {
    storage res;

    if(!host::mp_packed<Field>::sub(res.l, l, sub.l))
      host::mp_packed<Field>::add(res.l, res.l, Field::const_storage_n.l);
    return res;
  }

  storage mod_mul(const impl<Field,true>& coeff) {
    impl<Field,true> impl_res;
    storage          res;

    host::mp<Field>::unpack(impl_res.l, l);
    impl_res=impl_res * coeff;
    if(host::mp<Field>::compare(impl_res.l, Field::const_n.l)>=0)
      host::mp<Field>::sub(impl_res.l, impl_res.l, Field::const_n.l);
    host::mp<Field>::pack(res.l, impl_res.l);
    return res;
  }

  bool operator==(const storage& value) {
    return host::mp_packed<Field>::compare(l, value.l)==0;
  }

  bool operator!=(const storage& value) {
    return host::mp_packed<Field>::compare(l, value.l)!=0;
  }

  operator host::ff::impl<Field,false>() const {
    host::ff::impl<Field,false> res;

    host::ff::convert<Field>::impl_from_storage(res, *this);
    return res;
  }

  operator host::ff::impl<Field,true>() const {
    host::ff::impl<Field,true> res;

    host::ff::convert<Field>::impl_from_storage(res, *this);
    return res;
  }
};

} // namespace ff

} // namespace host