/***

Copyright (c) 2024, Snarkify, Inc.  All rights reserved.
Dual licensed under the MIT License or the Apache License, Version 2.0.
See LICENSE for details.

Author(s):  Niall Emmart

***/

namespace host {

namespace ff {

template<class Field>
class convert {
  public:
  static void impl_from_impl(host::ff::impl<Field,false>& r, const host::ff::impl<Field,true>& i) {
    printf("Implement me! i<F,false> = i<F,true>\n");
  }

  static void impl_from_impl(host::ff::impl<Field,true>& r, const host::ff::impl<Field,false>& i) {
    printf("Implement me!\n");
  }

  static void storage_from_storage(host::ff::storage<Field,false>& r, const host::ff::storage<Field,true>& s) {
    printf("Implement me s<F,false> = s<F,true>!\n");
  }

  static void storage_from_storage(host::ff::storage<Field,true>& r, const host::ff::storage<Field,false>& s) {
    printf("Implement me s<F,true> = s<F,false>!\n");
  }

  static void impl_from_storage(host::ff::impl<Field,false>& r, const host::ff::storage<Field,false>& s) {
    mp<Field>::unpack(r.l, s.l);
  }

  static void impl_from_storage(host::ff::impl<Field,true>& r, const host::ff::storage<Field,true>& s) {
    if constexpr (Field::packed_limbs==4)
      printf("Implement me i<F,true> = s<F,true>! (4 limbs)\n");
    else if constexpr (Field::packed_limbs==6) {
      mp<Field>::unpack(r.l, s.l);
      mp<Field>::add(r.l, r.l, r.l);
    }
  }

  static void impl_from_storage(host::ff::impl<Field,false>& r, const host::ff::storage<Field,true>& s) {
    mp<Field>::unpack(r.l, s.l);
    mp<Field>::add(r.l, r.l, r.l);
    mp_mul<Field,Field::limbs>::mont_normalize(r.l, r.l);
  }

  static void impl_from_storage(host::ff::impl<Field,true>& r, const host::ff::storage<Field,false>& s) {
    mp<Field>::unpack(r.l, s.l);
    mp_mul<Field,Field::limbs>::mont_mul(r.l, r.l, Field::const_r_squared.l);
  }

  static void storage_from_impl(host::ff::storage<Field,false>& r, const host::ff::impl<Field,false>& i) {
    // this assumes Barrett mode is 0 to 2n
    mp<Field>::pack(r.l, i.l);
    if(mp_packed<Field>::compare(r.l, Field::const_storage_n.l)>=0)
      mp_packed<Field>::sub(r.l, r.l, Field::const_storage_n.l);
  }

  static void storage_from_impl(host::ff::storage<Field,true>& r, const host::ff::impl<Field,true>& i) {
    if constexpr (Field::packed_limbs==4)
      printf("Implement me s<F,true> = i<F,true>! (4 limbs)\n");
    else if constexpr (Field::packed_limbs==6) {
      mp<Field>::pack(r.l, i.l);
      if((r.l[0] & 0x01)!=0)
        mp_packed<Field>::add(r.l, r.l, Field::const_storage_n.l);
      mp_packed<Field>::shift_right_bits(r.l, r.l, 1);
      if(mp_packed<Field>::compare(r.l, Field::const_storage_n.l)>=0)
        mp_packed<Field>::sub(r.l, r.l, Field::const_storage_n.l);
    }
  }

  static void storage_from_impl(host::ff::storage<Field,false>& r, const host::ff::impl<Field,true>& i) {
    uint64_t l[Field::limbs];

    mp_mul<Field,Field::limbs>::mont_normalize(l, i.l);
    mp<Field>::pack(r.l, l);
    if(mp_packed<Field>::compare(r.l, Field::const_storage_n.l)>=0)
      mp_packed<Field>::sub(r.l, r.l, Field::const_storage_n.l);
  }

  static void storage_from_impl(host::ff::storage<Field,true>& r, const host::ff::impl<Field,false>& i) {
    uint64_t l[Field::limbs];

    mp_mul<Field,Field::limbs>::mont_mul(l, i.l, Field::const_sr_squared.l);
    mp<Field>::pack(r.l, l);
    if(mp_packed<Field>::compare(r.l, Field::const_storage_n.l)>=0)
      mp_packed<Field>::sub(r.l, r.l, Field::const_storage_n.l);
  }
};

} // namespace ff

} // namespace host
