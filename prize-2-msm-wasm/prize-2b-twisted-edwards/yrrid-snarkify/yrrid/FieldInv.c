/***

Copyright (c) 2022-2024, Yrrid Software, Inc. and Snarkify Inc.  All rights reserved.
Dual licensed under the MIT License or the Apache License, Version 2.0.
See LICENSE for details.

Author(s):  Niall Emmart
            Sougata Bhattacharya
            Antony Suresh

***/

static inline void field256FromFieldPairL(Field256_t* f, FieldPair_t* fp) {
  uint64_t v0, v1, v2, v3, v4;

  // input assertion: fp has been resolved
  v0=i64x2_extract_l(fp->v[0]);
  v1=i64x2_extract_l(fp->v[1]);
  v2=i64x2_extract_l(fp->v[2]);
  v3=i64x2_extract_l(fp->v[3]);
  v4=i64x2_extract_l(fp->v[4]);

  f->f0=(v1<<51) | v0;
  f->f1=(v2<<38) | (v1>>13);
  f->f2=(v3<<25) | (v2>>26);
  f->f3=(v4<<12) | (v3>>39);
}

static inline void field256ToFieldPair(FieldPair_t* fp, Field256_t* f) {
  I128 mask=i64x2_splat(0x7FFFFFFFFFFFFL);

  fp->v[0]=i64x2_splat(f->f0);
  fp->v[1]=i64x2_splat(f->f1);
  fp->v[2]=i64x2_splat(f->f2);
  fp->v[3]=i64x2_splat(f->f3);

  fp->v[4]=u64x2_shr(fp->v[3], 12);
  fp->v[3]=i64x2_or(u64x2_shr(fp->v[2], 25), i64x2_shl(fp->v[3], 39));
  fp->v[2]=i64x2_or(u64x2_shr(fp->v[1], 38), i64x2_shl(fp->v[2], 26));
  fp->v[1]=i64x2_or(u64x2_shr(fp->v[0], 51), i64x2_shl(fp->v[1], 13));

  fp->v[3]=i64x2_and(fp->v[3], mask);
  fp->v[2]=i64x2_and(fp->v[2], mask);
  fp->v[1]=i64x2_and(fp->v[1], mask);
  fp->v[0]=i64x2_and(fp->v[0], mask);
}

static inline uint64_t field256LimbOR(Field256_t* f) {
  return f->f0 | f->f1 | f->f2 | f->f3;
}

static inline void field256SetN(Field256_t* f) {
  f->f0=0x0a11800000000001L;
  f->f1=0x59aa76fed0000001L;
  f->f2=0x60b44d1e5c37b001L;
  f->f3=0x12ab655e9a2ca556L;
}

static inline void field256ShiftRightOne(Field256_t* f) {
  f->f0=(f->f0 >> 1) + (f->f1 << 63);
  f->f1=(f->f1 >> 1) + (f->f2 << 63);
  f->f2=(f->f2 >> 1) + (f->f3 << 63);
  f->f3=f->f3 >> 1;
}

static inline void field256AddN(Field256_t* f) {
  uint64_t s0, s1, s2, s3;

  s0=f->f0 + 0x0a11800000000001L;
  s1=f->f1 + 0x59aa76fed0000001L;
  s2=f->f2 + 0x60b44d1e5c37b001L;
  s3=f->f3 + 0x12ab655e9a2ca556L;

  f->f3=s3 + (s2 < f->f2 ? 1 : 0);
  f->f2=s2 + (s1 < f->f1 ? 1 : 0);
  f->f1=s1 + (s0 < f->f0 ? 1 : 0);
  f->f0=s0;
}

static inline bool field256Sub(Field256_t* r, Field256_t* a, Field256_t* b) {
  uint64_t s0, s1, s2, s3;
  uint64_t c0, c1, c2;
  bool     c3;

  s0=a->f0 + ~b->f0;
  s1=a->f1 + ~b->f1;
  s2=a->f2 + ~b->f2;
  s3=a->f3 + ~b->f3;

  c0=(s0+1 <= a->f0) ? 1 : 0;
  c1=(s1 < a->f1 || s1+c0 < c0) ? 1 : 0;
  c2=(s2 < a->f2 || s2+c1 < c1) ? 1 : 0;
  c3=s3 < a->f3 || s3+c2 < c2;

  r->f0 = s0 + 1;
  r->f1 = s1 + c0;
  r->f2 = s2 + c1;
  r->f3 = s3 + c2;

  return c3;
}

static inline bool field256Greater(Field256_t* a, Field256_t* b) {
  if(a->f3 > b->f3) return true;
  if(a->f3 < b->f3) return false;
  if(a->f2 > b->f2) return true;
  if(a->f2 < b->f2) return false;
  if(a->f1 > b->f1) return true;
  if(a->f1 < b->f1) return false;
  return a->f0 > b->f0;
}

static inline void field256Swap(Field256_t* a, Field256_t* b) {
  uint64_t swap;

  swap=a->f0; a->f0=b->f0; b->f0=swap;
  swap=a->f1; a->f1=b->f1; b->f1=swap;
  swap=a->f2; a->f2=b->f2; b->f2=swap;
  swap=a->f3; a->f3=b->f3; b->f3=swap;
}

void fieldPairInvL(FieldPair_t* r, FieldPair_t* f) {
  Field256_t  A, B;
  Field256_t  T0={1, 0, 0, 0}, T1={0, 0, 0, 0};
  FieldPair_t local, rInv;

  rInv.v[0]=D2I(f64x2_splat(1612108321904644.0));     // rInv mod P
  rInv.v[1]=D2I(f64x2_splat(1509903170359368.0));
  rInv.v[2]=D2I(f64x2_splat(1570590048187764.0));
  rInv.v[3]=D2I(f64x2_splat(48943389467175.0));
  rInv.v[4]=D2I(f64x2_splat(270893018534424.0));

  fieldPairResolveConvert(&local, f);
  fieldPairMul(&local, &local, &rInv);

  fieldPairResolve(&local, &local);
  field256FromFieldPairL(&A, &local);
  field256SetN(&B);

  while(field256LimbOR(&A)!=0) {
    if((A.f0 & 0x01)!=0) {
      if(field256Greater(&B, &A)) {
        field256Swap(&A, &B);
        field256Swap(&T0, &T1);
      }
      field256Sub(&A, &A, &B);
      if(!field256Sub(&T0, &T0, &T1))
        field256AddN(&T0);
    }
    field256ShiftRightOne(&A);
    if((T0.f0 & 0x01)!=0)
      field256AddN(&T0);
    field256ShiftRightOne(&T0);
  }

  field256ToFieldPair(r, &T1);
}
