/***

Copyright (c) 2024, Snarkify, Inc. All rights reserved.
Dual licensed under the MIT License or the Apache License, Version 2.0.
See LICENSE for details.

Author(s):  Niall Emmart

***/

__constant__ __device__ uint32_t zeroConstant=0;

class bls12381_fr {
  static const uint32_t limbs=8;

  public:
  uint32_t __align__(16) l[limbs];

  __device__ __forceinline__ bls12381_fr() {
  }

  __device__ __forceinline__ bls12381_fr(const uint4& low, const uint4& high) {
    l[0]=low.x; l[1]=low.y; l[2]=low.z; l[3]=low.w;
    l[4]=high.x; l[5]=high.y; l[6]=high.z; l[7]=high.w;
  }

  __device__ __forceinline__ uint4 get_low() const {
    return make_uint4(l[0], l[1], l[2], l[3]);
  }

  __device__ __forceinline__ uint4 get_high() const {
    return make_uint4(l[4], l[5], l[6], l[7]);
  }

  __device__ __forceinline__ void set_low(const uint4& low) {
    l[0]=low.x; l[1]=low.y; l[2]=low.z; l[3]=low.w;
  }

  __device__ __forceinline__ void set_high(const uint4& high) {
    l[4]=high.x; l[5]=high.y; l[6]=high.z; l[7]=high.w;
  }

  __device__ __forceinline__ static uint32_t qTerm(uint32_t x) {
    return zeroConstant - x;
  }

  __device__ __forceinline__ static bls12381_fr load(bls12381_fr* ptr) {
    uint4* ptr_u4=(uint4*)ptr;

    return bls12381_fr(ptr_u4[0], ptr_u4[1]);
  }
  
  __device__ __forceinline__ static void store(bls12381_fr* ptr, const bls12381_fr& value) {
    uint4* ptr_u4=(uint4*)ptr;

    ptr_u4[0]=value.get_low();
    ptr_u4[1]=value.get_high();
  }

  __device__ __forceinline__ static bls12381_fr zero() {
    bls12381_fr res;

    mp_zero<limbs>(res.l);
    return res;
  }

  __device__ __forceinline__ static bls12381_fr one() {
    bls12381_fr res;

    res.l[0]=1; res.l[1]=0; res.l[2]=0; res.l[3]=0;
    res.l[4]=0; res.l[5]=0; res.l[6]=0; res.l[7]=0;
    return res;
  }

  __device__ __forceinline__ static bls12381_fr n() {
    bls12381_fr res;

    res.l[0]=0x00000001; res.l[1]=0xFFFFFFFF; res.l[2]=0xFFFE5BFE; res.l[3]=0x53BDA402; 
    res.l[4]=0x09A1D805; res.l[5]=0x3339D808; res.l[6]=0x299D7D48; res.l[7]=0x73EDA753;
    return res;
  }

  __device__ __forceinline__ static bls12381_fr r() {
    bls12381_fr res;

    res.l[0]=0xFFFFFFFE; res.l[1]=0x00000001; res.l[2]=0x00034802; res.l[3]=0x5884B7FA; 
    res.l[4]=0xECBC4FF5; res.l[5]=0x998C4FEF; res.l[6]=0xACC5056F; res.l[7]=0x1824B159;
    return res;
  }

  __device__ __forceinline__ static bls12381_fr r_squared() {
    bls12381_fr res;

    res.l[0]=0xF3F29C6D; res.l[1]=0xC999E990; res.l[2]=0x87925C23; res.l[3]=0x2B6CEDCB; 
    res.l[4]=0x7254398F; res.l[5]=0x05D31496; res.l[6]=0x9F59FF11; res.l[7]=0x0748D9D9;
    return res;
  }

  __device__ __forceinline__ static bls12381_fr r_inverse() {
    bls12381_fr res;

    res.l[0]=0xFE75C040; res.l[1]=0x13F75B69; res.l[2]=0x09DC705F; res.l[3]=0xAB6FCA8F; 
    res.l[4]=0x4F77266A; res.l[5]=0x7204078A; res.l[6]=0x30009D57; res.l[7]=0x1BBE8693;
    return res;
  }

  __device__ __forceinline__ static bls12381_fr root_1_8() {
    bls12381_fr res;

    res.l[0]=0x2C42EFF8; res.l[1]=0x6ECF4995; res.l[2]=0x4D5DFA57; res.l[3]=0x6EC1FA83; 
    res.l[4]=0x5656B15C; res.l[5]=0x3FF5CF38; res.l[6]=0x229E031B; res.l[7]=0x54B80267;
    return res;
  }

  __device__ __forceinline__ static bls12381_fr root_1_4() {
    bls12381_fr res;

    res.l[0]=0xAA89CFB1; res.l[1]=0xF3B05674; res.l[2]=0x6006B9FE; res.l[3]=0x072F0140; 
    res.l[4]=0x25667A26; res.l[5]=0xCE9A0DBF; res.l[6]=0x2D598374; res.l[7]=0x4D2CE405;
    return res;
  }

  __device__ __forceinline__ static bls12381_fr root_3_8() {
    bls12381_fr res;

    res.l[0]=0x3EB2A8DD; res.l[1]=0x111F9842; res.l[2]=0x6F38B840; res.l[3]=0x1F772784; 
    res.l[4]=0x8E3CA13C; res.l[5]=0xB28758A3; res.l[6]=0xED9A48F0; res.l[7]=0x5974F9F7;
    return res;
  }

  __device__ __forceinline__ static bls12381_fr root_5_8() {
    bls12381_fr res;

    res.l[0]=0xD3BD1009; res.l[1]=0x9130B669; res.l[2]=0xB2A061A7; res.l[3]=0xE4FBA97F; 
    res.l[4]=0xB34B26A8; res.l[5]=0xF34408CF; res.l[6]=0x06FF7A2C; res.l[7]=0x1F35A4EC;
    return res;
  }

  __device__ __forceinline__ static bls12381_fr root_3_4() {
    bls12381_fr res;

    res.l[0]=0x55763050; res.l[1]=0x0C4FA98A; res.l[2]=0x9FF7A200; res.l[3]=0x4C8EA2C2; 
    res.l[4]=0xE43B5DDF; res.l[5]=0x649FCA48; res.l[6]=0xFC43F9D3; res.l[7]=0x26C0C34D;
    return res;
  }

  __device__ __forceinline__ static bls12381_fr root_7_8() {
    bls12381_fr res;

    res.l[0]=0xC14D5724; res.l[1]=0xEEE067BC; res.l[2]=0x90C5A3BE; res.l[3]=0x34467C7E; 
    res.l[4]=0x7B6536C9; res.l[5]=0x80B27F64; res.l[6]=0x3C033457; res.l[7]=0x1A78AD5B;
    return res;
  }

  __device__ __forceinline__ static bls12381_fr mod_neg(const bls12381_fr& x) {
    bls12381_fr res;

    mp_sub<limbs>(res.l, n().l, x.l);
    mp_select<limbs>(res.l, mp_logical_or<limbs>(x.l)==0, zero().l, res.l);
    return res;
  }

  __device__ __forceinline__ static bls12381_fr mod_add(const bls12381_fr& a, const bls12381_fr& b) {
    bls12381_fr res, res_minus_n;
    bool        carry;

    mp_add<limbs>(res.l, a.l, b.l);
    carry=mp_sub_carry<limbs>(res_minus_n.l, res.l, n().l); 
    mp_select<limbs>(res.l, carry, res_minus_n.l, res.l);
    return res;
  }

  __device__ __forceinline__ static bls12381_fr mod_sub(const bls12381_fr& a, const bls12381_fr& b) {
    bls12381_fr res, res_plus_n;
    bool        carry;

    carry=mp_sub_carry<limbs>(res.l, a.l, b.l);
    mp_add<limbs>(res_plus_n.l, res.l, n().l);
    mp_select<limbs>(res.l, carry, res.l, res_plus_n.l);
    return res;
  }

  __device__ __forceinline__ static bls12381_fr mod_mul(const bls12381_fr& a, const bls12381_fr& b) {
    uint64_t    evenOdd[limbs];
    bool        carry;
    bls12381_fr res, res_minus_n;

    carry=mp_mul_red_cl<bls12381_fr, limbs>(evenOdd, a.l, b.l, n().l);
    mp_merge_cl<limbs>(res.l, evenOdd, carry); 
    carry=mp_sub_carry<limbs>(res_minus_n.l, res.l, n().l); 
    mp_select<limbs>(res.l, carry, res_minus_n.l, res.l);
    return res;
  }

  __device__ __forceinline__ static bls12381_fr mod_sqr(const bls12381_fr& a) {
    uint64_t    evenOdd[limbs];
    uint32_t    t[limbs];
    bool        carry;
    bls12381_fr res, res_minus_n;

    carry=mp_sqr_red_cl<bls12381_fr, limbs>(evenOdd, t, a.l, n().l);
    mp_merge_cl<limbs>(res.l, evenOdd, carry);
    carry=mp_sub_carry<limbs>(res_minus_n.l, res.l, n().l); 
    mp_select<limbs>(res.l, carry, res_minus_n.l, res.l);
    return res;
  }

  __device__ __forceinline__ static bls12381_fr mod_inv(const bls12381_fr& a) {
    bls12381_fr temp;
    bls12381_fr res;

    temp=mod_mul(a, r_inverse());
    mp_inverse<limbs>(res.l, temp.l, n().l);
    return res;
  }

  __device__ __forceinline__ static bls12381_fr to_montgomery(const bls12381_fr& x) {
    return mod_mul(x, r_squared());
  }

  __device__ __forceinline__ static bls12381_fr from_montgomery(const bls12381_fr& x) {
    return mod_mul(x, one());
  }

  __device__ __forceinline__ bls12381_fr sqr() const {
    return mod_sqr(*this);
  }

  __device__ __forceinline__ bls12381_fr inv() const {
    return mod_inv(*this);
  }

  __device__ __forceinline__ bls12381_fr operator-() const {
    return mod_neg(*this);
  }

  __device__ __forceinline__ bls12381_fr operator+(const bls12381_fr& x) const {
    return mod_add(*this, x);
  }

  __device__ __forceinline__ bls12381_fr operator-(const bls12381_fr& x) const {
    return mod_sub(*this, x);
  }

  __device__ __forceinline__ bls12381_fr operator*(const bls12381_fr& x) const {
    return mod_mul(*this, x);
  }

  __device__ __forceinline__ void dump() const {
    printf("%08X%08X%08X%08X%08X%08X%08X%08X\n", l[7], l[6], l[5], l[4], l[3], l[2], l[1], l[0]);
  }
};
