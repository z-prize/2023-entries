/***

Copyright (c) 2023-2024, Yrrid Software, Inc. and Snarkify Inc.  All rights reserved.
Dual licensed under the MIT License or the Apache License, Version 2.0.
See LICENSE for details.

Author(s):  Niall Emmart

***/

#if defined(WASM)
  #define I(l, h) (__u64x2){l, h}
#endif

#if defined(X86)
  #define I(l, h) l, h
#endif

FieldPair_t lMultiples[7]={
  {I(0x0000000000000L, 0L), I(0x0000000000000L, 0L), I(0x0000000000000L, 0L), I(0x0000000000000L, 0L), I(0x0000000000000L, 0L)},
  {I(0x1800000000001L, 0L), I(0x7DA0000002142L, 0L), I(0x0DEC00566A9DBL, 0L), I(0x2AB305A268F2EL, 0L), I(0x12AB655E9A2CAL, 0L)},
  {I(0x3000000000002L, 0L), I(0x7B40000004284L, 0L), I(0x1BD800ACD53B7L, 0L), I(0x55660B44D1E5CL, 0L), I(0x2556CABD34594L, 0L)},
  {I(0x4800000000003L, 0L), I(0x78E00000063C6L, 0L), I(0x29C401033FD93L, 0L), I(0x001910E73AD8AL, 0L), I(0x3802301BCE85FL, 0L)},
  {I(0x6000000000004L, 0L), I(0x7680000008508L, 0L), I(0x37B00159AA76FL, 0L), I(0x2ACC1689A3CB8L, 0L), I(0x4AAD957A68B29L, 0L)},
  {I(0x7800000000005L, 0L), I(0x742000000A64AL, 0L), I(0x459C01B01514BL, 0L), I(0x557F1C2C0CBE6L, 0L), I(0x5D58FAD902DF3L, 0L)},
  {I(0x1000000000006L, 0L), I(0x71C000000C78DL, 0L), I(0x538802067FB27L, 0L), I(0x003221CE75B14L, 0L), I(0x700460379D0BEL, 0L)},
};

FieldPair_t hMultiples[7]={
  {I(0L, 0x0000000000000L), I(0L, 0x0000000000000L), I(0L, 0x0000000000000L), I(0L, 0x0000000000000L), I(0L, 0x0000000000000L)},
  {I(0L, 0x1800000000001L), I(0L, 0x7DA0000002142L), I(0L, 0x0DEC00566A9DBL), I(0L, 0x2AB305A268F2EL), I(0L, 0x12AB655E9A2CAL)},
  {I(0L, 0x3000000000002L), I(0L, 0x7B40000004284L), I(0L, 0x1BD800ACD53B7L), I(0L, 0x55660B44D1E5CL), I(0L, 0x2556CABD34594L)},
  {I(0L, 0x4800000000003L), I(0L, 0x78E00000063C6L), I(0L, 0x29C401033FD93L), I(0L, 0x001910E73AD8AL), I(0L, 0x3802301BCE85FL)},
  {I(0L, 0x6000000000004L), I(0L, 0x7680000008508L), I(0L, 0x37B00159AA76FL), I(0L, 0x2ACC1689A3CB8L), I(0L, 0x4AAD957A68B29L)},
  {I(0L, 0x7800000000005L), I(0L, 0x742000000A64AL), I(0L, 0x459C01B01514BL), I(0L, 0x557F1C2C0CBE6L), I(0L, 0x5D58FAD902DF3L)},
  {I(0L, 0x1000000000006L), I(0L, 0x71C000000C78DL), I(0L, 0x538802067FB27L), I(0L, 0x003221CE75B14L), I(0L, 0x700460379D0BEL)},
};

FieldPair_t multiples[7]={
  {I(0x0000000000000L, 0x0000000000000L), I(0x0000000000000L, 0x0000000000000L), I(0x0000000000000L, 0x0000000000000L), I(0x0000000000000L, 0x0000000000000L), I(0x0000000000000L, 0x0000000000000L)},
  {I(0x1800000000001L, 0x1800000000001L), I(0x7DA0000002142L, 0x7DA0000002142L), I(0x0DEC00566A9DBL, 0x0DEC00566A9DBL), I(0x2AB305A268F2EL, 0x2AB305A268F2EL), I(0x12AB655E9A2CAL, 0x12AB655E9A2CAL)},
  {I(0x3000000000002L, 0x3000000000002L), I(0x7B40000004284L, 0x7B40000004284L), I(0x1BD800ACD53B7L, 0x1BD800ACD53B7L), I(0x55660B44D1E5CL, 0x55660B44D1E5CL), I(0x2556CABD34594L, 0x2556CABD34594L)},
  {I(0x4800000000003L, 0x4800000000003L), I(0x78E00000063C6L, 0x78E00000063C6L), I(0x29C401033FD93L, 0x29C401033FD93L), I(0x001910E73AD8AL, 0x001910E73AD8AL), I(0x3802301BCE85FL, 0x3802301BCE85FL)},
  {I(0x6000000000004L, 0x6000000000004L), I(0x7680000008508L, 0x7680000008508L), I(0x37B00159AA76FL, 0x37B00159AA76FL), I(0x2ACC1689A3CB8L, 0x2ACC1689A3CB8L), I(0x4AAD957A68B29L, 0x4AAD957A68B29L)},
  {I(0x7800000000005L, 0x7800000000005L), I(0x742000000A64AL, 0x742000000A64AL), I(0x459C01B01514BL, 0x459C01B01514BL), I(0x557F1C2C0CBE6L, 0x557F1C2C0CBE6L), I(0x5D58FAD902DF3L, 0x5D58FAD902DF3L)},
  {I(0x1000000000006L, 0x1000000000006L), I(0x71C000000C78DL, 0x71C000000C78DL), I(0x538802067FB27L, 0x538802067FB27L), I(0x003221CE75B14L, 0x003221CE75B14L), I(0x700460379D0BEL, 0x700460379D0BEL)},
};

void fieldPairLoad(FieldPair_t* fp, I128* f1, I128* f2) {
  I128 mask=i64x2_splat(0x7FFFFFFFFFFFFL);

  fp->v[0]=i64x2_extract_ll(f1[0], f2[0]);
  fp->v[1]=i64x2_extract_hh(f1[0], f2[0]);
  fp->v[2]=i64x2_extract_ll(f1[1], f2[1]);
  fp->v[3]=i64x2_extract_hh(f1[1], f2[1]);

  fp->v[4]=u64x2_shr(fp->v[3], 12);
  fp->v[3]=i64x2_or(u64x2_shr(fp->v[2], 25), i64x2_shl(fp->v[3], 39));
  fp->v[2]=i64x2_or(u64x2_shr(fp->v[1], 38), i64x2_shl(fp->v[2], 26));
  fp->v[1]=i64x2_or(u64x2_shr(fp->v[0], 51), i64x2_shl(fp->v[1], 13));

  fp->v[3]=i64x2_and(fp->v[3], mask);
  fp->v[2]=i64x2_and(fp->v[2], mask);
  fp->v[1]=i64x2_and(fp->v[1], mask);
  fp->v[0]=i64x2_and(fp->v[0], mask);
}

void fieldPairExtractLL(FieldPair_t* r, FieldPair_t* fp1, FieldPair_t* fp2) {
  r->v[0]=i64x2_extract_ll(fp1->v[0], fp2->v[0]);
  r->v[1]=i64x2_extract_ll(fp1->v[1], fp2->v[1]);
  r->v[2]=i64x2_extract_ll(fp1->v[2], fp2->v[2]);
  r->v[3]=i64x2_extract_ll(fp1->v[3], fp2->v[3]);
  r->v[4]=i64x2_extract_ll(fp1->v[4], fp2->v[4]);
}

void fieldPairExtractHH(FieldPair_t* r, FieldPair_t* fp1, FieldPair_t* fp2) {
  r->v[0]=i64x2_extract_hh(fp1->v[0], fp2->v[0]);
  r->v[1]=i64x2_extract_hh(fp1->v[1], fp2->v[1]);
  r->v[2]=i64x2_extract_hh(fp1->v[2], fp2->v[2]);
  r->v[3]=i64x2_extract_hh(fp1->v[3], fp2->v[3]);
  r->v[4]=i64x2_extract_hh(fp1->v[4], fp2->v[4]);
}

void fieldPairStore(I128* l, I128* h, FieldPair_t* fp) {
  I128        v[4];
  FieldPair_t local;

  fieldPairResolve(&local, fp);
  v[0]=i64x2_or(local.v[0], i64x2_shl(local.v[1], 51));
  v[1]=i64x2_or(u64x2_shr(local.v[1], 13), i64x2_shl(local.v[2], 38));
  v[2]=i64x2_or(u64x2_shr(local.v[2], 26), i64x2_shl(local.v[3], 25));
  v[3]=i64x2_or(u64x2_shr(local.v[3], 39), i64x2_shl(local.v[4], 12));

  l[0]=i64x2_make(i64x2_extract_l(v[0]), i64x2_extract_l(v[1]));
  l[1]=i64x2_make(i64x2_extract_l(v[2]), i64x2_extract_l(v[3]));
  h[0]=i64x2_make(i64x2_extract_h(v[0]), i64x2_extract_h(v[1]));
  h[1]=i64x2_make(i64x2_extract_h(v[2]), i64x2_extract_h(v[3]));
}

void fieldPairConvert(FieldPair_t* r, FieldPair_t* fp) {
  I128 cvtI=i64x2_splat(0x4330000000000000L);     // 2^52 in FP64
  F128 cvtD=I2D(cvtI);

  r->v[0]=D2I(f64x2_sub(I2D(i64x2_add(fp->v[0], cvtI)), cvtD));
  r->v[1]=D2I(f64x2_sub(I2D(i64x2_add(fp->v[1], cvtI)), cvtD));
  r->v[2]=D2I(f64x2_sub(I2D(i64x2_add(fp->v[2], cvtI)), cvtD));
  r->v[3]=D2I(f64x2_sub(I2D(i64x2_add(fp->v[3], cvtI)), cvtD));
  r->v[4]=D2I(f64x2_sub(I2D(i64x2_add(fp->v[4], cvtI)), cvtD));
}

void fieldPairResolve(FieldPair_t* r, FieldPair_t* fp) {
  I128      mask=i64x2_splat(0x7FFFFFFFFFFFFL),  local[3];

  local[0]=i64x2_add(fp->v[1], i64x2_shr(fp->v[0], 51));
  local[1]=i64x2_add(fp->v[2], i64x2_shr(local[0], 51));
  local[2]=i64x2_add(fp->v[3], i64x2_shr(local[1], 51));
  r->v[4]=i64x2_add(fp->v[4], i64x2_shr(local[2], 51));

  r->v[0]=i64x2_and(fp->v[0], mask);
  r->v[1]=i64x2_and(local[0], mask);
  r->v[2]=i64x2_and(local[1], mask);
  r->v[3]=i64x2_and(local[2], mask);
}

void fieldPairResolveConvert(FieldPair_t* r, FieldPair_t* fp) {
  fieldPairResolve(r, fp);
  fieldPairConvert(r, r);
}

void fieldPairReduce(FieldPair_t* r, FieldPair_t* fp) {
  uint32_t l, h;

  l=(uint32_t)(i64x2_extract_l(fp->v[4])*0x1B6CL>>61);
  h=(uint32_t)(i64x2_extract_h(fp->v[4])*0x1B6CL>>61);

  if((l | h)==0) {
    *r=*fp;
    return;
  }

  fieldPairSub(r, fp, &lMultiples[l]);
  fieldPairSub(r, r, &hMultiples[h]);
}

void fieldPairReduceResolve(FieldPair_t* r, FieldPair_t* fp) {
  fieldPairReduce(r, fp);
  fieldPairResolve(r, r);
}

void fieldPairFullReduceResolve(FieldPair_t* r, FieldPair_t* fp) {
  I128        high=i64x2_splat(0x12AB655E9A2C9L), check;
  FieldPair_t sub;
  uint64_t    l, h;

  fieldPairReduce(r, fp);
  fieldPairResolve(r, r);

  check=i64x2_sub(high, r->v[4]);
  if((i64x2_extract_l(check) | i64x2_extract_h(check))<=0x7FFFFFFFFFFFFL)
    return;

  // fairly rare, something like 1 in 100
  fieldPairSub(&sub, r, &multiples[1]);
  fieldPairResolve(&sub, &sub);
  l=i64x2_extract_l(sub.v[4]);
  h=i64x2_extract_h(sub.v[4]);
  if(l<0x8000000000000000L) {
    r->v[0]=i64x2_make(i64x2_extract_l(sub.v[0]), i64x2_extract_h(r->v[0]));
    r->v[1]=i64x2_make(i64x2_extract_l(sub.v[1]), i64x2_extract_h(r->v[1]));
    r->v[2]=i64x2_make(i64x2_extract_l(sub.v[2]), i64x2_extract_h(r->v[2]));
    r->v[3]=i64x2_make(i64x2_extract_l(sub.v[3]), i64x2_extract_h(r->v[3]));
    r->v[4]=i64x2_make(i64x2_extract_l(sub.v[4]), i64x2_extract_h(r->v[4]));
  }
  if(h<0x800000000000000L) {
    r->v[0]=i64x2_make(i64x2_extract_l(r->v[0]), i64x2_extract_h(sub.v[0]));
    r->v[1]=i64x2_make(i64x2_extract_l(r->v[1]), i64x2_extract_h(sub.v[1]));
    r->v[2]=i64x2_make(i64x2_extract_l(r->v[2]), i64x2_extract_h(sub.v[2]));
    r->v[3]=i64x2_make(i64x2_extract_l(r->v[3]), i64x2_extract_h(sub.v[3]));
    r->v[4]=i64x2_make(i64x2_extract_l(r->v[4]), i64x2_extract_h(sub.v[4]));
  }
}

void fieldPairAdd(FieldPair_t* r, FieldPair_t* a, FieldPair_t* b) {
  r->v[0]=i64x2_add(a->v[0], b->v[0]);
  r->v[1]=i64x2_add(a->v[1], b->v[1]);
  r->v[2]=i64x2_add(a->v[2], b->v[2]);
  r->v[3]=i64x2_add(a->v[3], b->v[3]);
  r->v[4]=i64x2_add(a->v[4], b->v[4]);
}

void fieldPairSub(FieldPair_t* r, FieldPair_t* a, FieldPair_t* b) {
  r->v[0]=i64x2_sub(a->v[0], b->v[0]);
  r->v[1]=i64x2_sub(a->v[1], b->v[1]);
  r->v[2]=i64x2_sub(a->v[2], b->v[2]);
  r->v[3]=i64x2_sub(a->v[3], b->v[3]);
  r->v[4]=i64x2_sub(a->v[4], b->v[4]);
}

void fieldPairMul(FieldPair_t* r, FieldPair_t* a, FieldPair_t* b) {
  F128     term, c2, c3, c4, bd[5], p[5], lh[5];
  I128     sum[11], c0, c1;
  uint64_t q0, q1;

  // This algorithm uses SIMD FMA to multiply 51 bits * 51 bits -> 102 bits,
  // and builds a modmul algorithm.
  //
  // For more details about how it works, see SupportingFiles/FP_FMA.java

  for(int i=0;i<5;i++)
    bd[i]=I2D(b->v[i]);

  sum[0]=i64x2_splat(0x7990000000000000L);   // have some magic numbers!  ha!
  sum[1]=i64x2_splat(0x6660000000000000L);
  sum[2]=i64x2_splat(0x5330000000000000L);
  sum[3]=i64x2_splat(0x4000000000000000L);
  sum[4]=i64x2_splat(0x2CD0000000000000L);
  sum[5]=i64x2_splat(0x2680000000000000L);
  sum[6]=i64x2_splat(0x39B0000000000000L);
  sum[7]=i64x2_splat(0x4CE0000000000000L);
  sum[8]=i64x2_splat(0x6010000000000000L);
  sum[9]=i64x2_splat(0x7340000000000000L);
  sum[10]=i64x2_splat(0L);

  c0=i64x2_splat(0x7FFFFFFFFFFFFL);
  c1=i64x2_splat(0x4330000000000000L);
  c2=I2D(c1);
  c3=I2D(i64x2_splat(0x4660000000000000L));
  c4=I2D(i64x2_splat(0x4660000000000003L));

  p[0]=f64x2_splat(422212465065985.0);
  p[1]=f64x2_splat(2210018371838274.0);
  p[2]=f64x2_splat(244916305701339.0);
  p[3]=f64x2_splat(751174112677678.0);
  p[4]=f64x2_splat(328437590500042.0);

  for(int i=0;i<5;i++) {
    term=I2D(a->v[i]);
    for(int j=0;j<5;j++)
      lh[j]=f64x2_fma(term, bd[j], c3);
    for(int j=0;j<5;j++)
      sum[j+1]=i64x2_add(sum[j+1], D2I(lh[j]));
    for(int j=0;j<5;j++)
      lh[j]=f64x2_sub(c4, lh[j]);
    for(int j=0;j<5;j++)
      lh[j]=f64x2_fma(term, bd[j], lh[j]);
    for(int j=0;j<5;j++)
      sum[j]=i64x2_add(sum[j], D2I(lh[j]));

    q0=i64x2_extract_l(sum[0])*0x17FFFFFFFFFFFL;
    q1=i64x2_extract_h(sum[0])*0x17FFFFFFFFFFFL;
    term=f64x2_sub(I2D(i64x2_add(i64x2_and(i64x2_make(q0, q1), c0), c1)), c2);

    for(int j=0;j<5;j++)
      lh[j]=f64x2_fma(term, p[j], c3);
    for(int j=0;j<5;j++)
      sum[j+1]=i64x2_add(sum[j+1], D2I(lh[j]));
    for(int j=0;j<5;j++)
      lh[j]=f64x2_sub(c4, lh[j]);
    for(int j=0;j<5;j++)
      lh[j]=f64x2_fma(term, p[j], lh[j]);

    sum[0]=i64x2_add(sum[0], D2I(lh[0]));
    sum[1]=i64x2_add(sum[1], D2I(lh[1]));
    sum[0]=i64x2_add(sum[1], i64x2_shr(sum[0], 51));
    for(int j=1;j<4;j++)
      sum[j]=i64x2_add(sum[j+1], D2I(lh[j+1]));
    sum[4]=sum[5];
    sum[5]=sum[i+6];
  }

  for(int i=0;i<5;i++)
    r->v[i]=sum[i];
}

void fieldPairConvertMul(FieldPair_t* r, FieldPair_t* a, FieldPair_t* b) {
  F128     term, c2, c3, c4, ad[5], bd[5], p[5], lh[5];
  I128     sum[11], t[5], c0, c1;
  uint64_t q0, q1;

  // This algorithm uses SIMD FMA to multiply 51 bits * 51 bits -> 102 bits,
  // and builds a modmul algorithm.
  //
  // For more details about how it works, see SupportingFiles/FP_FMA.java

  c0=i64x2_splat(0x7FFFFFFFFFFFFL);
  c1=i64x2_splat(0x4330000000000000L);
  c2=I2D(c1);
  c3=I2D(i64x2_splat(0x4660000000000000L));
  c4=I2D(i64x2_splat(0x4660000000000003L));

  ad[0]=f64x2_sub(I2D(i64x2_add(a->v[0], c1)), c2);
  ad[1]=f64x2_sub(I2D(i64x2_add(a->v[1], c1)), c2);
  ad[2]=f64x2_sub(I2D(i64x2_add(a->v[2], c1)), c2);
  ad[3]=f64x2_sub(I2D(i64x2_add(a->v[3], c1)), c2);
  ad[4]=f64x2_sub(I2D(i64x2_add(a->v[4], c1)), c2);

  bd[0]=f64x2_sub(I2D(i64x2_add(b->v[0], c1)), c2);
  bd[1]=f64x2_sub(I2D(i64x2_add(b->v[1], c1)), c2);
  bd[2]=f64x2_sub(I2D(i64x2_add(b->v[2], c1)), c2);
  bd[3]=f64x2_sub(I2D(i64x2_add(b->v[3], c1)), c2);
  bd[4]=f64x2_sub(I2D(i64x2_add(b->v[4], c1)), c2);

  sum[0]=i64x2_splat(0x7990000000000000L);   // have some magic numbers!  ha!
  sum[1]=i64x2_splat(0x6660000000000000L);
  sum[2]=i64x2_splat(0x5330000000000000L);
  sum[3]=i64x2_splat(0x4000000000000000L);
  sum[4]=i64x2_splat(0x2CD0000000000000L);
  sum[5]=i64x2_splat(0x2680000000000000L);
  sum[6]=i64x2_splat(0x39B0000000000000L);
  sum[7]=i64x2_splat(0x4CE0000000000000L);
  sum[8]=i64x2_splat(0x6010000000000000L);
  sum[9]=i64x2_splat(0x7340000000000000L);
  sum[10]=i64x2_splat(0L);

  p[0]=f64x2_splat(422212465065985.0);       // prime as 51 bit doubles
  p[1]=f64x2_splat(2210018371838274.0);
  p[2]=f64x2_splat(244916305701339.0);
  p[3]=f64x2_splat(751174112677678.0);
  p[4]=f64x2_splat(328437590500042.0);

  for(int i=0;i<5;i++) {
    for(int j=0;j<5;j++)
      lh[j]=f64x2_fma(ad[i], bd[j], c3);
    for(int j=0;j<5;j++)
      sum[j+1]=i64x2_add(sum[j+1], D2I(lh[j]));
    for(int j=0;j<5;j++)
      lh[j]=f64x2_sub(c4, lh[j]);
    for(int j=0;j<5;j++)
      lh[j]=f64x2_fma(ad[i], bd[j], lh[j]);
    for(int j=0;j<5;j++)
      sum[j]=i64x2_add(sum[j], D2I(lh[j]));

    q0=i64x2_extract_l(sum[0])*0x17FFFFFFFFFFFL;
    q1=i64x2_extract_h(sum[0])*0x17FFFFFFFFFFFL;
    term=f64x2_sub(I2D(i64x2_add(i64x2_and(i64x2_make(q0, q1), c0), c1)), c2);

    for(int j=0;j<5;j++)
      lh[j]=f64x2_fma(term, p[j], c3);
    for(int j=0;j<5;j++)
      sum[j+1]=i64x2_add(sum[j+1], D2I(lh[j]));
    for(int j=0;j<5;j++)
      lh[j]=f64x2_sub(c4, lh[j]);
    for(int j=0;j<5;j++)
      lh[j]=f64x2_fma(term, p[j], lh[j]);

    sum[0]=i64x2_add(sum[0], D2I(lh[0]));
    sum[1]=i64x2_add(sum[1], D2I(lh[1]));
    sum[0]=i64x2_add(sum[1], i64x2_shr(sum[0], 51));
    for(int j=1;j<4;j++)
      sum[j]=i64x2_add(sum[j+1], D2I(lh[j+1]));
    sum[4]=sum[5];
    sum[5]=sum[i+6];
  }

  for(int i=0;i<5;i++)
    r->v[i]=sum[i];
}

void fieldPairResolveConvertMul(FieldPair_t* r, FieldPair_t* a, FieldPair_t* b) {
  F128     term, c2, c3, c4, ad[5], bd[5], p[5], lh[5];
  I128     sum[11], t[5], c0, c1;
  uint64_t q0, q1;

  // This algorithm uses SIMD FMA to multiply 51 bits * 51 bits -> 102 bits,
  // and builds a modmul algorithm.
  //
  // For more details about how it works, see SupportingFiles/FP_FMA.java

  c0=i64x2_splat(0x7FFFFFFFFFFFFL);
  c1=i64x2_splat(0x4330000000000000L);
  c2=I2D(c1);
  c3=I2D(i64x2_splat(0x4660000000000000L));
  c4=I2D(i64x2_splat(0x4660000000000003L));

  t[1]=i64x2_add(a->v[1], i64x2_shr(a->v[0], 51));
  t[2]=i64x2_add(a->v[2], i64x2_shr(t[1], 51));
  t[3]=i64x2_add(a->v[3], i64x2_shr(t[2], 51));
  t[4]=i64x2_add(a->v[4], i64x2_shr(t[3], 51));

  t[0]=i64x2_and(a->v[0], c0);
  t[1]=i64x2_and(t[1], c0);
  t[2]=i64x2_and(t[2], c0);
  t[3]=i64x2_and(t[3], c0);

  ad[0]=f64x2_sub(I2D(i64x2_add(t[0], c1)), c2);
  ad[1]=f64x2_sub(I2D(i64x2_add(t[1], c1)), c2);
  ad[2]=f64x2_sub(I2D(i64x2_add(t[2], c1)), c2);
  ad[3]=f64x2_sub(I2D(i64x2_add(t[3], c1)), c2);
  ad[4]=f64x2_sub(I2D(i64x2_add(t[4], c1)), c2);

  t[1]=i64x2_add(b->v[1], i64x2_shr(b->v[0], 51));
  t[2]=i64x2_add(b->v[2], i64x2_shr(t[1], 51));
  t[3]=i64x2_add(b->v[3], i64x2_shr(t[2], 51));
  t[4]=i64x2_add(b->v[4], i64x2_shr(t[3], 51));

  t[0]=i64x2_and(b->v[0], c0);
  t[1]=i64x2_and(t[1], c0);
  t[2]=i64x2_and(t[2], c0);
  t[3]=i64x2_and(t[3], c0);

  bd[0]=f64x2_sub(I2D(i64x2_add(t[0], c1)), c2);
  bd[1]=f64x2_sub(I2D(i64x2_add(t[1], c1)), c2);
  bd[2]=f64x2_sub(I2D(i64x2_add(t[2], c1)), c2);
  bd[3]=f64x2_sub(I2D(i64x2_add(t[3], c1)), c2);
  bd[4]=f64x2_sub(I2D(i64x2_add(t[4], c1)), c2);

  sum[0]=i64x2_splat(0x7990000000000000L);   // have some magic numbers!  ha!
  sum[1]=i64x2_splat(0x6660000000000000L);
  sum[2]=i64x2_splat(0x5330000000000000L);
  sum[3]=i64x2_splat(0x4000000000000000L);
  sum[4]=i64x2_splat(0x2CD0000000000000L);
  sum[5]=i64x2_splat(0x2680000000000000L);
  sum[6]=i64x2_splat(0x39B0000000000000L);
  sum[7]=i64x2_splat(0x4CE0000000000000L);
  sum[8]=i64x2_splat(0x6010000000000000L);
  sum[9]=i64x2_splat(0x7340000000000000L);
  sum[10]=i64x2_splat(0L);

  p[0]=f64x2_splat(422212465065985.0);       // prime as 51 bit doubles
  p[1]=f64x2_splat(2210018371838274.0);
  p[2]=f64x2_splat(244916305701339.0);
  p[3]=f64x2_splat(751174112677678.0);
  p[4]=f64x2_splat(328437590500042.0);

  for(int i=0;i<5;i++) {
    for(int j=0;j<5;j++)
      lh[j]=f64x2_fma(ad[i], bd[j], c3);
    for(int j=0;j<5;j++)
      sum[j+1]=i64x2_add(sum[j+1], D2I(lh[j]));
    for(int j=0;j<5;j++)
      lh[j]=f64x2_sub(c4, lh[j]);
    for(int j=0;j<5;j++)
      lh[j]=f64x2_fma(ad[i], bd[j], lh[j]);
    for(int j=0;j<5;j++)
      sum[j]=i64x2_add(sum[j], D2I(lh[j]));

    q0=i64x2_extract_l(sum[0])*0x17FFFFFFFFFFFL;
    q1=i64x2_extract_h(sum[0])*0x17FFFFFFFFFFFL;
    term=f64x2_sub(I2D(i64x2_add(i64x2_and(i64x2_make(q0, q1), c0), c1)), c2);

    for(int j=0;j<5;j++)
      lh[j]=f64x2_fma(term, p[j], c3);
    for(int j=0;j<5;j++)
      sum[j+1]=i64x2_add(sum[j+1], D2I(lh[j]));
    for(int j=0;j<5;j++)
      lh[j]=f64x2_sub(c4, lh[j]);
    for(int j=0;j<5;j++)
      lh[j]=f64x2_fma(term, p[j], lh[j]);

    sum[0]=i64x2_add(sum[0], D2I(lh[0]));
    sum[1]=i64x2_add(sum[1], D2I(lh[1]));
    sum[0]=i64x2_add(sum[1], i64x2_shr(sum[0], 51));
    for(int j=1;j<4;j++)
      sum[j]=i64x2_add(sum[j+1], D2I(lh[j+1]));
    sum[4]=sum[5];
    sum[5]=sum[i+6];
  }

  for(int i=0;i<5;i++)
    r->v[i]=sum[i];
}

#if defined(X86)

#include <stdio.h>

void fieldPairDumpRawL(FieldPair_t* fp) {
  for(int i=4;i>=0;i--)
    printf("%016lX ", i64x2_extract_l(fp->v[i]));
  printf("\n");
}

void fieldPairDumpRawH(FieldPair_t* fp) {
  for(int i=4;i>=0;i--)
    printf("%016lX ", i64x2_extract_h(fp->v[i]));
  printf("\n");
}

void fieldPairDumpL(FieldPair_t* fp) {
  FieldPair_t local, one;
  uint64_t    hex[5];

  one.v[0]=i64x2_splat(0x1L);
  one.v[1]=i64x2_splat(0x0L);
  one.v[2]=i64x2_splat(0x0L);
  one.v[3]=i64x2_splat(0x0L);
  one.v[4]=i64x2_splat(0x0L);

  fieldPairResolveConvertMul(&local, fp, &one);
  fieldPairFullReduceResolve(&local, &local);
  for(int i=0;i<5;i++)
    hex[i]=i64x2_extract_l(local.v[i]);

  hex[0]=hex[0] | (hex[1]<<51);
  hex[1]=(hex[1]>>13) | (hex[2]<<38);
  hex[2]=(hex[2]>>26) | (hex[3]<<25);
  hex[3]=(hex[3]>>39) | (hex[4]<<12);
  printf("%016lX%016lX%016lX%016lX\n", hex[3], hex[2], hex[1], hex[0]);
}

void fieldPairDumpH(FieldPair_t* fp) {
  FieldPair_t local, one;
  uint64_t    hex[5];

  one.v[0]=i64x2_splat(0x1L);
  one.v[1]=i64x2_splat(0x0L);
  one.v[2]=i64x2_splat(0x0L);
  one.v[3]=i64x2_splat(0x0L);
  one.v[4]=i64x2_splat(0x0L);

  fieldPairResolveConvertMul(&local, fp, &one);
  fieldPairFullReduceResolve(&local, &local);
  for(int i=0;i<5;i++)
    hex[i]=i64x2_extract_h(local.v[i]);
  hex[0]=hex[0] | (hex[1]<<51);
  hex[1]=(hex[1]>>13) | (hex[2]<<38);
  hex[2]=(hex[2]>>26) | (hex[3]<<25);
  hex[3]=(hex[3]>>39) | (hex[4]<<12);
  printf("%016lX%016lX%016lX%016lX\n", hex[3], hex[2], hex[1], hex[0]);
}

#endif
