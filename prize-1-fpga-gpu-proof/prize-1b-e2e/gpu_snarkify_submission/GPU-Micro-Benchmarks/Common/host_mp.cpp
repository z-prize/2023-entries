/***

Copyright (c) 2024, Snarkify, Inc. All rights reserved.
Dual licensed under the MIT License or the Apache License, Version 2.0.
See LICENSE for details.

Author(s):  Niall Emmart

***/

#include <stdio.h>
#include <stdint.h>

namespace host {

template<class Field>
class mp_packed {
  public:
  static const uint32_t limbs=Field::packed_limbs;

  static void zero(uint64_t* r) {
    for(uint32_t i=0;i<limbs;i++)
      r[i]=0ull;
  }

  static void set(uint64_t* r, const uint64_t* f) {
    for(uint32_t i=0;i<limbs;i++)
      r[i]=f[i];
  }

  static int32_t compare(uint64_t* a, const uint64_t* b) {
    for(int32_t i=limbs-1;i>=0;i--) {
      if(a[i]!=b[i])
        return a[i]>b[i] ? 1 : -1;
    }
    return 0;
  }

  static uint32_t count_trailing_zero_words(const uint64_t* f) {
    uint32_t count=0;

    for(uint32_t i=0;i<limbs;i++)
      if(f[i]==0)
        count++;
    return count;
  }

  static bool add(uint64_t* r, const uint64_t* a, const uint64_t* b) {
    uint64_t sum, carry=0;

    for(uint32_t i=0;i<limbs;i++) {
      sum=a[i] + b[i] + carry;
      carry=(sum<a[i] || carry+b[i]<b[i]) ? 1 : 0;
      r[i]=sum;
    }
    return carry!=0ull;
  }

  static bool sub(uint64_t* r, const uint64_t* a, const uint64_t* b) {
    uint64_t sum, carry=1;

    for(uint32_t i=0;i<limbs;i++) {
      sum=a[i] + ~b[i] + carry;
      carry=(sum<a[i] || carry+~b[i]<~b[i]) ? 1 : 0;
      r[i]=sum;
    }
    return carry!=0ull;
  }

  static void shift_right_words(uint64_t* r, const uint64_t* f, uint32_t words) {
    for(int i=0;i<limbs-words;i++)
      r[i]=f[i+words];
    for(int i=limbs-words;i<limbs;i++)
      r[i]=0;
  }

  static void shift_left_words(uint64_t* r, const uint64_t* f, uint32_t words) {
    for(int i=limbs-1;i>=words;i--)
      r[i]=f[i-words];
    for(int i=0;i<words;i++)
      r[i]=0;
  }

  static void shift_right_bits(uint64_t* r, const uint64_t* f, uint32_t right) {
    uint32_t left=64-right;

    for(uint32_t i=0;i<limbs-1;i++)
      r[i]=(f[i]>>right) | (f[i+1]<<left);
    r[limbs-1]=f[limbs-1]>>right;
  }

  static void shift_left_bits(uint64_t* r, const uint64_t* f, uint32_t left) {
    uint32_t right=64-left;

    for(int i=limbs-1;i>=1;i--)
      r[i]=(f[i]<<left) | (f[i-1]>>right);
    r[0]=f[0]<<left;
  }
};

template<class Field>
class mp {
  public:
  static const uint32_t limbs=Field::limbs;
  static const uint32_t bits_per_limb=Field::bits_per_limb;
  static const uint64_t limb_mask=Field::limb_mask;

  typedef unsigned __int128 uint128_t;
  typedef mp_packed<Field> packed;
  typedef mp_mul<Field,limbs> mul;

  static uint128_t m(const uint64_t a, const uint64_t b) {
    uint128_t la=a, lb=b;

    return la*lb;
  }

  static void pack(uint64_t* pf, const uint64_t* f) {
    if constexpr(Field::packed_limbs==4) {
      pf[0]=f[0] | (f[1]<<52);
      pf[1]=(f[1]>>12) | (f[2]<<40);
      pf[2]=(f[2]>>24) | (f[3]<<28);
      pf[3]=(f[3]>>36) | (f[4]<<16);
    }
    else if constexpr(Field::packed_limbs==6) {
      pf[0]=f[0] | (f[1]<<55);
      pf[1]=(f[1]>>9) | (f[2]<<46);
      pf[2]=(f[2]>>18) | (f[3]<<37);
      pf[3]=(f[3]>>27) | (f[4]<<28);
      pf[4]=(f[4]>>36) | (f[5]<<19);
      pf[5]=(f[5]>>45) | (f[6]<<10);
    }
  }

  static void unpack(uint64_t* f, const uint64_t* pf) {
    if constexpr(Field::packed_limbs==4) {
      f[0]=pf[0] & limb_mask;
      f[1]=((pf[0]>>52) | (pf[1]<<12)) & limb_mask;
      f[2]=((pf[1]>>40) | (pf[2]<<24)) & limb_mask;
      f[3]=((pf[2]>>28) | (pf[3]<<36)) & limb_mask;
      f[4]=pf[3]>>16;
    }
    else if constexpr(Field::packed_limbs==6) {
      f[0]=pf[0] & limb_mask;
      f[1]=((pf[0]>>55) | (pf[1]<<9)) & limb_mask;
      f[2]=((pf[1]>>46) | (pf[2]<<18)) & limb_mask;
      f[3]=((pf[2]>>37) | (pf[3]<<27)) & limb_mask;
      f[4]=((pf[3]>>28) | (pf[4]<<36)) & limb_mask;
      f[5]=((pf[4]>>19) | (pf[5]<<45)) & limb_mask;
      f[6]=pf[5]>>10;
    }
  }

  static void zero(uint64_t* r) {
    for(uint32_t i=0;i<limbs;i++)
      r[i]=0ull;
  }

  static void set(uint64_t* r, const uint64_t* f) {
    for(uint32_t i=0;i<limbs;i++)
      r[i]=f[i];
  }

  static bool is_limb_equal(const uint64_t* a, const uint64_t* b) {
    uint64_t limb_xor=a[0] ^ b[0];

    for(uint32_t i=1;i<limbs;i++)
      limb_xor=limb_xor | (a[i] ^ b[i]);
    return limb_xor==0ull;
  }

  static int compare(const uint64_t* a, const uint64_t* b) {
    for(int32_t i=limbs-1;i>=0;i--) {
      if(a[i]!=b[i])
        return a[i]>b[i] ? 1 : -1;
    }
    return 0;
  }

  static int64_t add(uint64_t* r, const uint64_t* a, const uint64_t* b) {
    int64_t acc=0;

    for(uint32_t i=0;i<limbs;i++) {
      acc=acc + a[i] + b[i];
      r[i]=acc & limb_mask;
      acc=acc>>bits_per_limb;
    }
    return acc;
  }

  static int64_t sub(uint64_t* r, const uint64_t* a, const uint64_t* b) {
    int64_t acc=0;

    for(uint32_t i=0;i<limbs;i++) {
      acc=acc + a[i] - b[i];
      r[i]=acc & limb_mask;
      acc=acc>>bits_per_limb;
    }
    return acc;
  }

  static void add(uint64_t* r, const uint64_t* a, const uint64_t* b, const uint64_t* adjust) {
    int64_t acc=0;

    for(uint32_t i=0;i<limbs;i++) {
      acc=acc + a[i] + b[i] - adjust[i];
      r[i]=acc & limb_mask;
      acc=acc >> bits_per_limb;
    }
  }

  static void sub(uint64_t* r, const uint64_t* a, const uint64_t* b, const uint64_t* adjust) {
    int64_t acc=0;

    for(uint32_t i=0;i<limbs;i++) {
      acc=acc + a[i] - b[i] + adjust[i];
      r[i]=acc & limb_mask;
      acc=acc >> bits_per_limb;
    }
  }

  static bool mod_is_zero(const uint64_t* f) {
    const uint64_t* adjust;

    adjust=f[0]==0 ? Field::const_zero.l : Field::const_n.l;
    return is_limb_equal(adjust, f);
  }

  static void mod_neg(uint64_t* r, const uint64_t* f) {
    const uint32_t  last=Field::limbs-1;
    const uint64_t* adjust;

    adjust=f[last]<Field::const_n.l[last] ? Field::const_n.l : Field::const_2n.l;
    sub(r, adjust, f);
  }

  static void mod_add(uint64_t* r, const uint64_t* a, const uint64_t* b) {
    const uint32_t  last=Field::limbs-1;
    const uint64_t* adjust;
    uint64_t        top=a[last] + b[last];

    adjust=Field::const_zero.l;
    adjust=(top>Field::const_n.l[last]+1) ? Field::const_n.l : adjust;
    adjust=(top>Field::const_2n.l[last]+1) ? Field::const_2n.l : adjust;

    add(r, a, b, adjust);
  }

  static void mod_sub(uint64_t* r, const uint64_t* a, const uint64_t* b) {
    const uint32_t  last=Field::limbs-1;
    const uint64_t* adjust;
    uint64_t        top=a[last] - b[last] + Field::const_2n.l[last];

    adjust=Field::const_2n.l;
    adjust=(top>Field::const_n.l[last]+1) ? Field::const_n.l : adjust;
    adjust=(top>Field::const_2n.l[last]+1) ? Field::const_zero.l : adjust;

    sub(r, a, b, adjust);
  }

  static void mont_sqr(uint64_t* r, const uint64_t* a) {
    mul::mont_sqr(r, a);
  }

  static void mont_mul(uint64_t* r, const uint64_t* a, const uint64_t* b) {
    mul::mont_mul(r, a, b);
  }

  static void barrett_mul(uint64_t* r, const uint64_t* a, const uint64_t* b) {
    uint64_t l[limbs];

    mul::mont_mul(l, a, Field::const_r_squared.l);
    mul::mont_mul(r, l, b);

//    mul::barrett_mul(r, a, b);
  }

  static void barrett_sqr(uint64_t* r, const uint64_t* a) {
    uint64_t l[limbs];

    mul::mont_mul(l, a, Field::const_r_squared.l);
    mul::mont_mul(r, l, a);

//    mul::barrett_sqr(r, a);
  }

  static void invert(uint64_t* r, const uint64_t* f) {
    uint64_t A[packed::limbs], B[packed::limbs], U[packed::limbs], V[packed::limbs];
    int32_t  cmp;
    uint32_t shift, shift_count=0;

    pack(A, f);
    packed::set(B, Field::const_storage_n.l);
    packed::set(U, Field::const_storage_one.l);
    packed::set(V, Field::const_storage_zero.l);

    if((A[0] & 0x01)==0) {
      if(A[0]==0) {
        shift=packed::count_trailing_zero_words(A);
        packed::shift_right_words(A, A, shift);
        shift_count+=shift*bits_per_limb;
        if(A[0]==0) {
          packed::zero(r);
          return;
        }
      }
      shift=__builtin_ctzll(A[0]);
      packed::shift_right_bits(A, A, shift);
      shift_count+=shift;
    }

    while(true) {
      cmp=packed::compare(A, B);
      if(cmp>0) {
        packed::sub(A, A, B);
        packed::sub(U, U, V);
        if(A[0]==0) {
          shift=packed::count_trailing_zero_words(A);
          packed::shift_right_words(A, A, shift);
          packed::shift_left_words(V, V, shift);
          shift_count+=shift*bits_per_limb;
        }
        shift=__builtin_ctzll(A[0]);
        packed::shift_right_bits(A, A, shift);
        packed::shift_left_bits(V, V, shift);
        shift_count+=shift;
      }
      else if(cmp<0) {
        packed::sub(B, B, A);
        packed::sub(V, V, U);
        if(B[0]==0) {
          shift=packed::count_trailing_zero_words(B);
          packed::shift_right_words(B, B, shift);
          packed::shift_left_words(U, U, shift);
          shift_count+=shift*bits_per_limb;
        }
        shift=__builtin_ctzll(B[0]);
        packed::shift_right_bits(B, B, shift);
        packed::shift_left_bits(U, U, shift);
        shift_count+=shift;
      }
      else
        break;
    }

    packed::add(V, V, Field::const_storage_n.l);

    unpack(r, V);

    while(shift_count>=limbs*bits_per_limb) {
      mul::mont_normalize(r, r);
      shift_count=shift_count - limbs*bits_per_limb;
    }

    if(shift_count!=0) {
      uint64_t t[limbs];

      for(int i=0;i<limbs;i++)
        t[i]=0;

      shift_count=limbs*bits_per_limb-shift_count;

      t[shift_count/bits_per_limb]=1ull<<shift_count%bits_per_limb;

      mul::mont_mul(r, r, t);
    }

    // pack(V, r);
    // printf("res=%016lX%016lX%016lX%016lX%016lX%016lX\n", V[5], V[4], V[3], V[2], V[1], V[0]);
  }
};

} // namespace host
