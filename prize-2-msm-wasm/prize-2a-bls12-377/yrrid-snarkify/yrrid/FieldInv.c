/***

Copyright (c) 2022-2024, Yrrid Software, Inc. and Snarkify Inc.  All rights reserved.
Dual licensed under the MIT License or the Apache License, Version 2.0.
See LICENSE for details.

Author(s):  Niall Emmart
            Sougata Bhattacharya
            Antony Suresh

***/

static inline void packedFieldFromField(PackedField_t* pf, Field_t* f) {
  int64_t l[8]={i64x2_extract_l(f->v0), i64x2_extract_h(f->v0), i64x2_extract_l(f->v1), i64x2_extract_h(f->v1),
                i64x2_extract_l(f->v2), i64x2_extract_h(f->v2), i64x2_extract_l(f->v3), i64x2_extract_h(f->v3)};
  uint64_t mask16=0xFFFFL, mask32=0xFFFFFFFFL, mask48=0xFFFFFFFFFFFFL;

  l[1]+=l[0]>>48;
  l[2]+=l[1]>>48;
  l[3]+=l[2]>>48;
  l[4]+=l[3]>>48;
  l[5]+=l[4]>>48;
  l[6]+=l[5]>>48;
  l[7]+=l[6]>>48;

  pf->f0=(l[0] & mask48) | (l[1]<<48);
  pf->f1=((l[1]>>16) & mask32) | (l[2]<<32);
  pf->f2=((l[2]>>32) & mask16) | (l[3]<<16);
  pf->f3=(l[4] & mask48) | (l[5]<<48);
  pf->f4=((l[5]>>16) & mask32) | (l[6]<<32);
  pf->f5=((l[6]>>32) & mask16) | (l[7]<<16);
}

static inline void packedFieldToField(Field_t* f, PackedField_t* pf) {
  I128 mask=i64x2_splat(0xFFFFFFFFFFFFL);

  f->v0=i64x2_and(i64x2_make(pf->f0, (pf->f0>>48) | (pf->f1<<16)), mask);
  f->v1=i64x2_and(i64x2_make((pf->f1>>32) | (pf->f2<<32), pf->f2>>16), mask);
  f->v2=i64x2_and(i64x2_make(pf->f3, (pf->f3>>48) | (pf->f4<<16)), mask);
  f->v3=i64x2_and(i64x2_make((pf->f4>>32) | (pf->f5<<32), pf->f5>>16), mask);
}

static inline uint64_t packedFieldLimbOR(PackedField_t* pf) {
  return pf->f0 | pf->f1 | pf->f2 | pf->f3 | pf->f4 | pf->f5;
}

static inline void packedFieldSetN(PackedField_t* pf) {
  pf->f0=0x8508C00000000001L;
  pf->f1=0x170B5D4430000000L;
  pf->f2=0x1EF3622FBA094800L;
  pf->f3=0x1A22D9F300F5138FL;
  pf->f4=0xC63B05C06CA1493BL;
  pf->f5=0x01AE3A4617C510EAL;
}

static inline void packedFieldShiftRightOne(PackedField_t* pf) {
  pf->f0=(pf->f0 >> 1) + (pf->f1 << 63);
  pf->f1=(pf->f1 >> 1) + (pf->f2 << 63);
  pf->f2=(pf->f2 >> 1) + (pf->f3 << 63);
  pf->f3=(pf->f3 >> 1) + (pf->f4 << 63);
  pf->f4=(pf->f4 >> 1) + (pf->f5 << 63);
  pf->f5=pf->f5 >> 1;
}

static inline void packedFieldShiftLeft(PackedField_t* pf, uint32_t count) {
  uint32_t left=count, right=64-left;

  if(left!=0) {
    pf->f5=(pf->f5 << left) + (pf->f4 >> right);
    pf->f4=(pf->f4 << left) + (pf->f3 >> right);
    pf->f3=(pf->f3 << left) + (pf->f2 >> right);
    pf->f2=(pf->f2 << left) + (pf->f1 >> right);
    pf->f1=(pf->f1 << left) + (pf->f0 >> right);
    pf->f0=pf->f0 << left;
  }
}

static inline void packedFieldShiftRight(PackedField_t* pf, uint32_t count) {
  uint32_t right=count, left=64-right;

  if(right!=0) {
    pf->f0=(pf->f0 >> right) + (pf->f1 << left);
    pf->f1=(pf->f1 >> right) + (pf->f2 << left);
    pf->f2=(pf->f2 >> right) + (pf->f3 << left);
    pf->f3=(pf->f3 >> right) + (pf->f4 << left);
    pf->f4=(pf->f4 >> right) + (pf->f5 << left);
    pf->f5=(pf->f5 >> right);
  }
}

static inline void packedFieldAddN(PackedField_t* pf) {
  uint64_t s0, s1, s2, s3, s4, s5;

  s0=pf->f0 + 0x8508C00000000001L;
  s1=pf->f1 + 0x170B5D4430000000L;
  s2=pf->f2 + 0x1EF3622FBA094800L;
  s3=pf->f3 + 0x1A22D9F300F5138FL;
  s4=pf->f4 + 0xC63B05C06CA1493BL;
  s5=pf->f5 + 0x01AE3A4617C510EAL,

  pf->f5=s5 + (s4 < pf->f4 ? 1 : 0);
  pf->f4=s4 + (s3 < pf->f3 ? 1 : 0);
  pf->f3=s3 + (s2 < pf->f2 ? 1 : 0);
  pf->f2=s2 + (s1 < pf->f1 ? 1 : 0);
  pf->f1=s1 + (s0 < pf->f0 ? 1 : 0);
  pf->f0=s0;
}

static inline bool packedFieldSub(PackedField_t* r, PackedField_t* a, PackedField_t* b) {
  uint64_t s0, s1, s2, s3, s4, s5;
  uint64_t c0, c1, c2, c3, c4;
  bool     c5;

  s0=a->f0 + ~b->f0;
  s1=a->f1 + ~b->f1;
  s2=a->f2 + ~b->f2;
  s3=a->f3 + ~b->f3;
  s4=a->f4 + ~b->f4;
  s5=a->f5 + ~b->f5;

  c0=(s0+1 <= a->f0) ? 1 : 0;
  c1=(s1 < a->f1 || s1+c0 < c0) ? 1 : 0;
  c2=(s2 < a->f2 || s2+c1 < c1) ? 1 : 0;
  c3=(s3 < a->f3 || s3+c2 < c2) ? 1 : 0;
  c4=(s4 < a->f4 || s4+c3 < c3) ? 1 : 0;
  c5=s5 < a->f5 || s5+c4 < c4;

  r->f0 = s0 + 1;
  r->f1 = s1 + c0;
  r->f2 = s2 + c1;
  r->f3 = s3 + c2;
  r->f4 = s4 + c3;
  r->f5 = s5 + c4;

  return c5;
}

static inline bool packedFieldGreater(PackedField_t* a, PackedField_t* b) {
  if(a->f5 > b->f5) return true;
  if(a->f5 < b->f5) return false;
  if(a->f4 > b->f4) return true;
  if(a->f4 < b->f4) return false;
  if(a->f3 > b->f3) return true;
  if(a->f3 < b->f3) return false;
  if(a->f2 > b->f2) return true;
  if(a->f2 < b->f2) return false;
  if(a->f1 > b->f1) return true;
  if(a->f1 < b->f1) return false;
  return a->f0 > b->f0;
}

static inline void packedFieldSwap(PackedField_t* a, PackedField_t* b) {
  uint64_t swap;

  swap=a->f0; a->f0=b->f0; b->f0=swap;
  swap=a->f1; a->f1=b->f1; b->f1=swap;
  swap=a->f2; a->f2=b->f2; b->f2=swap;
  swap=a->f3; a->f3=b->f3; b->f3=swap;
  swap=a->f4; a->f4=b->f4; b->f4=swap;
  swap=a->f5; a->f5=b->f5; b->f5=swap;
}

void fieldInv(Field_t* r, Field_t* f) {
  PackedField_t A, B;
  PackedField_t T0={1, 0, 0, 0, 0, 0}, T1={0, 0, 0, 0, 0, 0};
  Field_t       local, zero, rInv;

  fieldSetZero(&zero);
  fieldSetRInverse(&rInv);

  fieldMul(&local, &zero, &rInv, f, &rInv, &zero);
  packedFieldFromField(&A, &local);
  packedFieldSetN(&B);

  // this alg is slow, but short to implement.  ~13K cycles per invert.
  // Approx 21500 inverses for a 2^20 point MSM. ==> 280 Mil Cycles.

  while(packedFieldLimbOR(&A)!=0) {
    if((A.f0 & 0x01)!=0) {
      if(packedFieldGreater(&B, &A)) {
        packedFieldSwap(&A, &B);
        packedFieldSwap(&T0, &T1);
      }
      packedFieldSub(&A, &A, &B);
      if(!packedFieldSub(&T0, &T0, &T1))
        packedFieldAddN(&T0);
    }
    packedFieldShiftRightOne(&A);
    if((T0.f0 & 0x01)!=0)
      packedFieldAddN(&T0);
    packedFieldShiftRightOne(&T0);
  }

  packedFieldToField(r, &T1);
}


/*
  // More complicated algorithm.  If we accelerate the mod reduce at the end,
  // it could be about 10% faster.  Potential savings are less than a millisecond,
  // not worth the effort.

  uint32_t totalShift=0, shift;

  while(true) {
    if(A.f0==0) {
      if((A.f1 | A.f2 | A.f3 | A.f4 | A.f5)==0)
        break;
      while(A.f0==0) {
        A.f0=A.f1; A.f1=A.f2; A.f2=A.f3; A.f3=A.f4; A.f4=A.f5; A.f5=0;
        T0.f5=T0.f4; T0.f4=T0.f3; T0.f3=T0.f2; T0.f2=T0.f1; T0.f1=T0.f0; T0.f0=0;
        totalShift+=64;
      }
    }
    shift=__builtin_ctzll(A.f0);
    packedFieldShiftRight(&A, shift);
    packedFieldShiftLeft(&T1, shift);
    totalShift+=shift;
    if(packedFieldGreater(&B, &A)) {
      packedFieldSwap(&A, &B);
      packedFieldSwap(&T0, &T1);
    }
    packedFieldSub(&A, &A, &B);
    packedFieldSub(&T0, &T0, &T1);
  }

  if(T1.f5>=0x8000000000000000L)
    packedFieldAddN(&T1);

  // mod reduce
  for(int i=0;i<totalShift;i++) {
    if((T1.f0 & 0x01)!=0)
      packedFieldAddN(&T1);
    packedFieldShiftRightOne(&T1);
  }
*/
