/***

Copyright (c) 2023-2024, Yrrid Software, Inc. and Snarkify Inc.  All rights reserved.
Dual licensed under the MIT License or the Apache License, Version 2.0.
See LICENSE for details.

Author(s):  Niall Emmart

***/

#if defined(X86)

void pointPairXYTZ_dumpL(PointPairXYTZ_t* pt) {
  printf("x="); fieldPairDumpL(&pt->x);
  printf("y="); fieldPairDumpL(&pt->y);
  printf("t="); fieldPairDumpL(&pt->t);
  printf("z="); fieldPairDumpL(&pt->z);
}

void pointPairXYTZ_dumpH(PointPairXYTZ_t* pt) {
  printf("x="); fieldPairDumpH(&pt->x);
  printf("y="); fieldPairDumpH(&pt->y);
  printf("t="); fieldPairDumpH(&pt->t);
  printf("z="); fieldPairDumpH(&pt->z);
}

void pointPairXYTZ_dumpRawL(PointPairXYTZ_t* pt) {
  printf("x="); fieldPairDumpRawL(&pt->x);
  printf("y="); fieldPairDumpRawL(&pt->y);
  printf("t="); fieldPairDumpRawL(&pt->t);
  printf("z="); fieldPairDumpRawL(&pt->z);
}

void pointPairXYTZ_dumpRawH(PointPairXYTZ_t* pt) {
  printf("x="); fieldPairDumpRawH(&pt->x);
  printf("y="); fieldPairDumpRawH(&pt->y);
  printf("t="); fieldPairDumpRawH(&pt->t);
  printf("z="); fieldPairDumpRawH(&pt->z);
}

#endif

void pointXYT_prepare(I128* r, I128* pt, bool negate) {
  FieldPair_t fp;

  if(negate) {
    fp.v[0]=pt[0];
    fp.v[1]=pt[1];
    fp.v[2]=pt[2];
    fp.v[3]=pt[3];
    fp.v[4]=pt[4];
    fieldPairSub(&fp, &multiples[1], &fp);
    fieldPairResolve(&fp, &fp);
  }

  r[0]=negate ? fp.v[0] : pt[0];
  r[1]=negate ? fp.v[1] : pt[1];
  r[2]=negate ? fp.v[2] : pt[2];
  r[3]=negate ? fp.v[3] : pt[3];
  r[4]=negate ? fp.v[4] : pt[4];
  r[5]=pt[5];
  r[6]=pt[6];
}

void pointXYTZ_zero(I128* r) {
  r[0]=i64x2_splat(0L);
  r[1]=i64x2_splat(0L);
  r[2]=i64x2_splat(0L);
  r[3]=i64x2_splat(0L);
  r[4]=i64x2_splat(0L);
  r[5]=i64x2_splat(0x6FFFFFFFFFFFAL);
  r[6]=i64x2_splat(0x0E3FFFFFF3872L);
  r[7]=i64x2_splat(0x2C77FDF9804D8L);
  r[8]=i64x2_splat(0x7FCDDE318A4EBL);
  r[9]=i64x2_splat(0x0FFB9FC862F41L);
}

void pointXYTZ_prepare(I128* r, I128* pt, bool negate) {
  FieldPair_t fp;
  uint64_t    v0, v1, v2, v3, v4, mask=0x7FFFFFFFFFFFFL;

  if(negate) {
    fp.v[0]=pt[0];
    fp.v[1]=pt[1];
    fp.v[2]=pt[2];
    fp.v[3]=pt[3];
    fp.v[4]=pt[4];
    fieldPairSub(&fp, &multiples[1], &fp);
    fieldPairResolve(&fp, &fp);
  }

  r[0]=negate ? fp.v[0] : pt[0];
  r[1]=negate ? fp.v[1] : pt[1];
  r[2]=negate ? fp.v[2] : pt[2];
  r[3]=negate ? fp.v[3] : pt[3];
  r[4]=negate ? fp.v[4] : pt[4];

  v0=i64x2_extract_l(pt[5]);
  v1=i64x2_extract_h(pt[5]);
  v2=i64x2_extract_l(pt[6]);
  v3=i64x2_extract_h(pt[6]);
  v4=v3>>12;
  v3=((v2>>25) | (v3<<39)) & mask;
  v2=((v1>>38) | (v2<<26)) & mask;
  v1=((v0>>51) | (v1<<13)) & mask;
  v0=v0 & mask;

  r[5]=i64x2_make(v0, 0x6FFFFFFFFFFFAL);
  r[6]=i64x2_make(v1, 0x0E3FFFFFF3872L);
  r[7]=i64x2_make(v2, 0x2C77FDF9804D8L);
  r[8]=i64x2_make(v3, 0x7FCDDE318A4EBL);
  r[9]=i64x2_make(v4, 0x0FFB9FC862F41L);
}

void pointPairXYT_load(PointPairXYT_t* pair, I128* p1, I128* p2) {
  I128 mask=i64x2_splat(0x7FFFFFFFFFFFFL);

  // Construct a PointPairXYT with p1 in L and p2 in H
  pair->y.v[0]=i64x2_extract_ll(p1[5], p2[5]);
  pair->y.v[1]=i64x2_extract_hh(p1[5], p2[5]);
  pair->y.v[2]=i64x2_extract_ll(p1[6], p2[6]);
  pair->y.v[3]=i64x2_extract_hh(p1[6], p2[6]);

  pair->y.v[4]=u64x2_shr(pair->y.v[3], 12);
  pair->y.v[3]=i64x2_or(u64x2_shr(pair->y.v[2], 25), i64x2_shl(pair->y.v[3], 39));
  pair->y.v[2]=i64x2_or(u64x2_shr(pair->y.v[1], 38), i64x2_shl(pair->y.v[2], 26));
  pair->y.v[1]=i64x2_or(u64x2_shr(pair->y.v[0], 51), i64x2_shl(pair->y.v[1], 13));

  pair->y.v[3]=i64x2_and(pair->y.v[3], mask);
  pair->y.v[2]=i64x2_and(pair->y.v[2], mask);
  pair->y.v[1]=i64x2_and(pair->y.v[1], mask);
  pair->y.v[0]=i64x2_and(pair->y.v[0], mask);

  pair->x.v[0]=i64x2_extract_ll(p1[0], p2[0]);
  pair->x.v[1]=i64x2_extract_ll(p1[1], p2[1]);
  pair->x.v[2]=i64x2_extract_ll(p1[2], p2[2]);
  pair->x.v[3]=i64x2_extract_ll(p1[3], p2[3]);
  pair->x.v[4]=i64x2_extract_ll(p1[4], p2[4]);

  pair->t.v[0]=i64x2_extract_hh(p1[0], p2[0]);
  pair->t.v[1]=i64x2_extract_hh(p1[1], p2[1]);
  pair->t.v[2]=i64x2_extract_hh(p1[2], p2[2]);
  pair->t.v[3]=i64x2_extract_hh(p1[3], p2[3]);
  pair->t.v[4]=i64x2_extract_hh(p1[4], p2[4]);
}

void pointPairXYT_store(I128* p1, I128* p2, PointPairXYT_t* pair) {
  I128 v[4];

  // Separate a PointPairXYT, where L goes to p1 and H goes to p2
  v[0]=i64x2_or(pair->y.v[0], i64x2_shl(pair->y.v[1], 51));
  v[1]=i64x2_or(u64x2_shr(pair->y.v[1], 13), i64x2_shl(pair->y.v[2], 38));
  v[2]=i64x2_or(u64x2_shr(pair->y.v[2], 26), i64x2_shl(pair->y.v[3], 25));
  v[3]=i64x2_or(u64x2_shr(pair->y.v[3], 39), i64x2_shl(pair->y.v[4], 12));

  p1[0]=i64x2_extract_ll(pair->x.v[0], pair->t.v[0]);
  p1[1]=i64x2_extract_ll(pair->x.v[1], pair->t.v[1]);
  p1[2]=i64x2_extract_ll(pair->x.v[2], pair->t.v[2]);
  p1[3]=i64x2_extract_ll(pair->x.v[3], pair->t.v[3]);
  p1[4]=i64x2_extract_ll(pair->x.v[4], pair->t.v[4]);
  p1[5]=i64x2_extract_ll(v[0], v[1]);
  p1[6]=i64x2_extract_ll(v[2], v[3]);

  p2[0]=i64x2_extract_hh(pair->x.v[0], pair->t.v[0]);
  p2[1]=i64x2_extract_hh(pair->x.v[1], pair->t.v[1]);
  p2[2]=i64x2_extract_hh(pair->x.v[2], pair->t.v[2]);
  p2[3]=i64x2_extract_hh(pair->x.v[3], pair->t.v[3]);
  p2[4]=i64x2_extract_hh(pair->x.v[4], pair->t.v[4]);
  p2[5]=i64x2_extract_hh(v[0], v[1]);
  p2[6]=i64x2_extract_hh(v[2], v[3]);
}

void pointPairXYTZ_loadXYT(PointPairXYTZ_t* pair, I128* p1, I128* p2) {
  I128 mask=i64x2_splat(0x7FFFFFFFFFFFFL);

  // Construct a PointPairXYTZ with p1(XYT) going to L and p1(XYT) going to H.
  // Z is initialized to 1.
  pair->y.v[0]=i64x2_extract_ll(p1[5], p2[5]);
  pair->y.v[1]=i64x2_extract_hh(p1[5], p2[5]);
  pair->y.v[2]=i64x2_extract_ll(p1[6], p2[6]);
  pair->y.v[3]=i64x2_extract_hh(p1[6], p2[6]);

  pair->y.v[4]=u64x2_shr(pair->y.v[3], 12);
  pair->y.v[3]=i64x2_or(u64x2_shr(pair->y.v[2], 25), i64x2_shl(pair->y.v[3], 39));
  pair->y.v[2]=i64x2_or(u64x2_shr(pair->y.v[1], 38), i64x2_shl(pair->y.v[2], 26));
  pair->y.v[1]=i64x2_or(u64x2_shr(pair->y.v[0], 51), i64x2_shl(pair->y.v[1], 13));

  pair->y.v[3]=i64x2_and(pair->y.v[3], mask);
  pair->y.v[2]=i64x2_and(pair->y.v[2], mask);
  pair->y.v[1]=i64x2_and(pair->y.v[1], mask);
  pair->y.v[0]=i64x2_and(pair->y.v[0], mask);

  pair->x.v[0]=i64x2_extract_ll(p1[0], p2[0]);
  pair->x.v[1]=i64x2_extract_ll(p1[1], p2[1]);
  pair->x.v[2]=i64x2_extract_ll(p1[2], p2[2]);
  pair->x.v[3]=i64x2_extract_ll(p1[3], p2[3]);
  pair->x.v[4]=i64x2_extract_ll(p1[4], p2[4]);

  pair->t.v[0]=i64x2_extract_hh(p1[0], p2[0]);
  pair->t.v[1]=i64x2_extract_hh(p1[1], p2[1]);
  pair->t.v[2]=i64x2_extract_hh(p1[2], p2[2]);
  pair->t.v[3]=i64x2_extract_hh(p1[3], p2[3]);
  pair->t.v[4]=i64x2_extract_hh(p1[4], p2[4]);

  pair->z.v[0]=i64x2_splat(0x6FFFFFFFFFFFAL);
  pair->z.v[1]=i64x2_splat(0x0E3FFFFFF3872L);
  pair->z.v[2]=i64x2_splat(0x2C77FDF9804D8L);
  pair->z.v[3]=i64x2_splat(0x7FCDDE318A4EBL);
  pair->z.v[4]=i64x2_splat(0x0FFB9FC862F41L);
}

void pointPairXYTZ_loadXYTZ(PointPairXYTZ_t* pair, I128* p1, I128* p2) {
  // construct a PointPairXYTZ with P1 in L, and P2 in H
  pair->x.v[0]=i64x2_extract_ll(p1[0], p2[0]);
  pair->x.v[1]=i64x2_extract_ll(p1[1], p2[1]);
  pair->x.v[2]=i64x2_extract_ll(p1[2], p2[2]);
  pair->x.v[3]=i64x2_extract_ll(p1[3], p2[3]);
  pair->x.v[4]=i64x2_extract_ll(p1[4], p2[4]);

  pair->t.v[0]=i64x2_extract_hh(p1[0], p2[0]);
  pair->t.v[1]=i64x2_extract_hh(p1[1], p2[1]);
  pair->t.v[2]=i64x2_extract_hh(p1[2], p2[2]);
  pair->t.v[3]=i64x2_extract_hh(p1[3], p2[3]);
  pair->t.v[4]=i64x2_extract_hh(p1[4], p2[4]);

  pair->y.v[0]=i64x2_extract_ll(p1[5], p2[5]);
  pair->y.v[1]=i64x2_extract_ll(p1[6], p2[6]);
  pair->y.v[2]=i64x2_extract_ll(p1[7], p2[7]);
  pair->y.v[3]=i64x2_extract_ll(p1[8], p2[8]);
  pair->y.v[4]=i64x2_extract_ll(p1[9], p2[9]);

  pair->z.v[0]=i64x2_extract_hh(p1[5], p2[5]);
  pair->z.v[1]=i64x2_extract_hh(p1[6], p2[6]);
  pair->z.v[2]=i64x2_extract_hh(p1[7], p2[7]);
  pair->z.v[3]=i64x2_extract_hh(p1[8], p2[8]);
  pair->z.v[4]=i64x2_extract_hh(p1[9], p2[9]);
}

void pointPairXYTZ_loadH_XYTZ(PointPairXYTZ_t* pair, PointPairXYTZ_t* acc, I128* pt) {
  // construct a PointPairXYTZ with acc.H in L, and pt in H
  pair->x.v[0]=i64x2_extract_hl(acc->x.v[0], pt[0]);
  pair->x.v[1]=i64x2_extract_hl(acc->x.v[1], pt[1]);
  pair->x.v[2]=i64x2_extract_hl(acc->x.v[2], pt[2]);
  pair->x.v[3]=i64x2_extract_hl(acc->x.v[3], pt[3]);
  pair->x.v[4]=i64x2_extract_hl(acc->x.v[4], pt[4]);

  pair->t.v[0]=i64x2_extract_hh(acc->t.v[0], pt[0]);
  pair->t.v[1]=i64x2_extract_hh(acc->t.v[1], pt[1]);
  pair->t.v[2]=i64x2_extract_hh(acc->t.v[2], pt[2]);
  pair->t.v[3]=i64x2_extract_hh(acc->t.v[3], pt[3]);
  pair->t.v[4]=i64x2_extract_hh(acc->t.v[4], pt[4]);

  pair->y.v[0]=i64x2_extract_hl(acc->y.v[0], pt[5]);
  pair->y.v[1]=i64x2_extract_hl(acc->y.v[1], pt[6]);
  pair->y.v[2]=i64x2_extract_hl(acc->y.v[2], pt[7]);
  pair->y.v[3]=i64x2_extract_hl(acc->y.v[3], pt[8]);
  pair->y.v[4]=i64x2_extract_hl(acc->y.v[4], pt[9]);

  pair->z.v[0]=i64x2_extract_hh(acc->z.v[0], pt[5]);
  pair->z.v[1]=i64x2_extract_hh(acc->z.v[1], pt[6]);
  pair->z.v[2]=i64x2_extract_hh(acc->z.v[2], pt[7]);
  pair->z.v[3]=i64x2_extract_hh(acc->z.v[3], pt[8]);
  pair->z.v[4]=i64x2_extract_hh(acc->z.v[4], pt[9]);
}

void pointPairXYTZ_store(I128* p1, I128* p2, PointPairXYTZ_t* pair) {
  // separate a PointPairXYTZ, L goes to p1, H goes to p2
  p1[0]=i64x2_extract_ll(pair->x.v[0], pair->t.v[0]);
  p1[1]=i64x2_extract_ll(pair->x.v[1], pair->t.v[1]);
  p1[2]=i64x2_extract_ll(pair->x.v[2], pair->t.v[2]);
  p1[3]=i64x2_extract_ll(pair->x.v[3], pair->t.v[3]);
  p1[4]=i64x2_extract_ll(pair->x.v[4], pair->t.v[4]);
  p1[5]=i64x2_extract_ll(pair->y.v[0], pair->z.v[0]);
  p1[6]=i64x2_extract_ll(pair->y.v[1], pair->z.v[1]);
  p1[7]=i64x2_extract_ll(pair->y.v[2], pair->z.v[2]);
  p1[8]=i64x2_extract_ll(pair->y.v[3], pair->z.v[3]);
  p1[9]=i64x2_extract_ll(pair->y.v[4], pair->z.v[4]);

  p2[0]=i64x2_extract_hh(pair->x.v[0], pair->t.v[0]);
  p2[1]=i64x2_extract_hh(pair->x.v[1], pair->t.v[1]);
  p2[2]=i64x2_extract_hh(pair->x.v[2], pair->t.v[2]);
  p2[3]=i64x2_extract_hh(pair->x.v[3], pair->t.v[3]);
  p2[4]=i64x2_extract_hh(pair->x.v[4], pair->t.v[4]);
  p2[5]=i64x2_extract_hh(pair->y.v[0], pair->z.v[0]);
  p2[6]=i64x2_extract_hh(pair->y.v[1], pair->z.v[1]);
  p2[7]=i64x2_extract_hh(pair->y.v[2], pair->z.v[2]);
  p2[8]=i64x2_extract_hh(pair->y.v[3], pair->z.v[3]);
  p2[9]=i64x2_extract_hh(pair->y.v[4], pair->z.v[4]);
}

void pointPairXYTZ_accumulateXYT_unified(PointPairXYTZ_t* acc, PointPairXYT_t* pt) {
  FieldPair_t T0, T1, A, B, C, D, E, F, G, H, k;

  // For a better understanding of these assertions, please see SupportingFiles/AssertionChecking.java

  // Input assertion:  0 <= acc.x, acc.y, acc.t <= 1.01*N
  //                   0 <= acc.z <= 3.20*N
  //
  //                   0 <= pt.x, pt.y, pt.t <= N

  // https://www.hyperelliptic.org/EFD/g1p/auto-twisted-extended-1.html#addition-madd-2008-hwcd-3
  // Assumes acc.z and pt.t have been resolved.
  // Output: acc.z is resolved, acc.x, acc.y, acc.t are not resolved

  k.v[0]=D2I(f64x2_splat(2251799813643824.0));              // 6042 * R mod P
  k.v[1]=D2I(f64x2_splat(1372190158772744.0));
  k.v[2]=D2I(f64x2_splat(1196913093420096.0));
  k.v[3]=D2I(f64x2_splat(985181756304134.0));
  k.v[4]=D2I(f64x2_splat(175725412514789.0));

  // compute A
  fieldPairSub(&T0, &acc->y, &acc->x);
  fieldPairAdd(&T0, &T0, &multiples[2]);      // twoN
  fieldPairSub(&T1, &pt->y, &pt->x);
  fieldPairAdd(&T1, &T1, &multiples[1]);      // oneN
  fieldPairResolveConvertMul(&A, &T0, &T1);

  // compute B
  fieldPairAdd(&T0, &acc->y, &acc->x);
  fieldPairAdd(&T1, &pt->y, &pt->x);
  fieldPairResolveConvertMul(&B, &T0, &T1);

  // compute C and D
  fieldPairResolveConvertMul(&C, &acc->t, &pt->t);
  fieldPairResolveConvert(&C, &C);
  fieldPairMul(&C, &C, &k);
  fieldPairAdd(&D, &acc->z, &acc->z);
  fieldPairReduce(&D, &D);

  // Compute E, F, G, H
  fieldPairSub(&E, &B, &A);
  fieldPairAdd(&E, &E, &multiples[3]);        // threeN
  fieldPairSub(&F, &D, &C);
  fieldPairAdd(&F, &F, &multiples[2]);        // twoN
  fieldPairAdd(&G, &D, &C);
  fieldPairAdd(&H, &B, &A);

  // Prep E, F, G, H for multiplies
  fieldPairResolveConvert(&E, &E);
  fieldPairResolveConvert(&F, &F);
  fieldPairResolveConvert(&G, &G);
  fieldPairResolveConvert(&H, &H);

  // Compute out x, y, t, z
  fieldPairMul(&acc->x, &E, &F);
  fieldPairMul(&acc->y, &G, &H);
  fieldPairMul(&acc->t, &E, &H);
  fieldPairMul(&acc->z, &F, &G);

  fieldPairReduce(&acc->x, &acc->x);
  fieldPairReduce(&acc->y, &acc->y);
  fieldPairReduce(&acc->t, &acc->t);
  fieldPairResolve(&acc->z, &acc->z);

  // Output assertion:  0 <= acc.x, acc.y, acc.t <= 1.01*N
  //                    0 <= acc.z <= 3.20*N
}

void pointPairXYTZ_accumulateXYTZ_unified(PointPairXYTZ_t* acc, PointPairXYTZ_t* pt) {
  FieldPair_t T0, T1, A, B, C, D, E, F, G, H, k;

  // For a better understanding of these assertions, please see SupportingFiles/AssertionChecking.java

  // Input assertion:  0 <= acc.x, acc.y, acc.t <= 1.01*N
  //                   0 <= acc.z <= 3.20*N
  //
  //                   0 <= pt.x, pt.y, pt.t <= 1.01*N
  //                   0 <= pt.z <= 3.20*N

  // https://www.hyperelliptic.org/EFD/g1p/auto-twisted-extended-1.html#addition-add-2008-hwcd-3
  // Assumes acc.z and pt.t have been resolved.
  // Output: acc.z is resolved, acc.x, acc.y, acc.t are not resolved

  k.v[0]=D2I(f64x2_splat(2251799813643824.0));             // 6042 * R mod P
  k.v[1]=D2I(f64x2_splat(1372190158772744.0));
  k.v[2]=D2I(f64x2_splat(1196913093420096.0));
  k.v[3]=D2I(f64x2_splat(985181756304134.0));
  k.v[4]=D2I(f64x2_splat(175725412514789.0));

  // compute A
  fieldPairSub(&T0, &acc->y, &acc->x);
  fieldPairAdd(&T0, &T0, &multiples[2]);      // twoN
  fieldPairSub(&T1, &pt->y, &pt->x);
  fieldPairAdd(&T1, &T1, &multiples[2]);      // twoN
  fieldPairResolveConvertMul(&A, &T0, &T1);

  // compute B
  fieldPairAdd(&T0, &acc->y, &acc->x);
  fieldPairAdd(&T1, &pt->y, &pt->x);
  fieldPairResolveConvertMul(&B, &T0, &T1);

  // compute C and D
  fieldPairResolveConvertMul(&C, &acc->t, &pt->t);
  fieldPairResolveConvert(&C, &C);
  fieldPairMul(&C, &C, &k);
  fieldPairResolveConvertMul(&D, &acc->z, &pt->z);
  fieldPairAdd(&D, &D, &D);
  fieldPairReduce(&D, &D);

  // Compute E, F, G, H
  fieldPairSub(&E, &B, &A);
  fieldPairAdd(&E, &E, &multiples[3]);        // threeN
  fieldPairSub(&F, &D, &C);
  fieldPairAdd(&F, &F, &multiples[2]);        // twoN
  fieldPairAdd(&G, &D, &C);
  fieldPairAdd(&H, &B, &A);

  // Prep E, F, G, H for multiplies
  fieldPairResolveConvert(&E, &E);
  fieldPairResolveConvert(&F, &F);
  fieldPairResolveConvert(&G, &G);
  fieldPairResolveConvert(&H, &H);

  // Compute out x, y, t, z
  fieldPairMul(&acc->x, &E, &F);
  fieldPairMul(&acc->y, &G, &H);
  fieldPairMul(&acc->t, &E, &H);
  fieldPairMul(&acc->z, &F, &G);

  fieldPairReduce(&acc->x, &acc->x);
  fieldPairReduce(&acc->y, &acc->y);
  fieldPairReduce(&acc->t, &acc->t);
  fieldPairResolve(&acc->z, &acc->z);

  // Output assertion:  0 <= acc.x, acc.y, acc.t <= 1.01*N
  //                    0 <= acc.z <= 3.20*N
}

void pointPairXYTZ_accumulateXYT(PointPairXYTZ_t* acc, PointPairXYT_t* pt) {
  FieldPair_t T0, T1, A, B, C, D, E, F, G, H, Z;

  // For a better understanding of these assertions, please see SupportingFiles/AssertionChecking.java

  // Input assertion:  0 <= acc.x, acc.y, acc.t <= 1.01*N
  //                   0 <= acc.z <= 3.20*N
  //
  //                   0 <= pt.x, pt.y, pt.t <= N

  // https://www.hyperelliptic.org/EFD/g1p/auto-twisted-extended-1.html#addition-madd-2008-hwcd-4

  // Assumes acc.z and pt.t have been resolved.
  // Output: acc.z is resolved, acc.x, acc.y, acc.t are not resolved

  // compute A
  fieldPairSub(&T0, &acc->y, &acc->x);
  fieldPairAdd(&T0, &T0, &multiples[2]);      // twoN
  fieldPairAdd(&T1, &pt->y, &pt->x);
  fieldPairResolveConvertMul(&A, &T0, &T1);

  // compute B
  fieldPairAdd(&T0, &acc->y, &acc->x);
  fieldPairSub(&T1, &pt->y, &pt->x);
  fieldPairAdd(&T1, &T1, &multiples[1]);      // oneN
  fieldPairResolveConvertMul(&B, &T0, &T1);

  // compute C, D
  fieldPairConvertMul(&C, &acc->z, &pt->t);
  fieldPairAdd(&C, &C, &C);
  fieldPairAdd(&D, &acc->t, &acc->t);

  // Compute E, F, G, H
  fieldPairAdd(&E, &D, &C);
  fieldPairSub(&F, &B, &A);
  fieldPairAdd(&F, &F, &multiples[2]);        // twoN
  fieldPairAdd(&G, &B, &A);
  fieldPairSub(&H, &D, &C);
  fieldPairAdd(&H, &H, &multiples[3]);        // threeN

  // Prep E, F, G, H for multiplies
  fieldPairResolveConvert(&E, &E);
  fieldPairResolveConvert(&F, &F);
  fieldPairResolveConvert(&G, &G);
  fieldPairResolveConvert(&H, &H);

  fieldPairMul(&Z, &F, &G);
  if((i64x2_extract_l(Z.v[0]) & 0x7FFFFFFFFFF0L)==0 || (i64x2_extract_h(Z.v[0]) & 0x7FFFFFFFFFF0L)==0) {
    // rather than checking is Z is some multiply of P, we just check some bits in the low word,
    // if they are all zero, we just play it safe and call the unified code.  Unrequired calls should be
    // very rare.

    pointPairXYTZ_accumulateXYT_unified(acc, pt);
    return;
  }

  // Compute x, y, t and z
  fieldPairMul(&acc->x, &E, &F);
  fieldPairMul(&acc->y, &G, &H);
  fieldPairMul(&acc->t, &E, &H);

  fieldPairReduce(&acc->x, &acc->x);
  fieldPairReduce(&acc->y, &acc->y);
  fieldPairReduce(&acc->t, &acc->t);
  fieldPairResolve(&acc->z, &Z);

  // Output assertion:  0 <= acc.x, acc.y, acc.t <= 1.01*N
  //                    0 <= acc.z <= 3.20*N
}

void pointPairXYTZ_accumulateXYTZ(PointPairXYTZ_t* acc, PointPairXYTZ_t* pt) {
  FieldPair_t T0, T1, A, B, C, D, E, F, G, H, Z;

  // For a better understanding of these assertions, please see SupportingFiles/AssertionChecking.java

  // Input assertion:  0 <= acc.x, acc.y, acc.t <= 1.01*N
  //                   0 <= acc.z <= 3.20*N
  //
  //                   0 <= pt.x, pt.y, pt.t <= 1.01*N
  //                   0 <= pt.z <= 3.20*N

  // https://www.hyperelliptic.org/EFD/g1p/auto-twisted-extended-1.html#addition-add-2008-hwcd-4
  // Assumes acc.z and pt.t have been resolved.
  // Output: acc.z is resolved, acc.x, acc.y, acc.t are not resolved

  // compute A
  fieldPairSub(&T0, &acc->y, &acc->x);
  fieldPairAdd(&T0, &T0, &multiples[2]);      // twoN
  fieldPairAdd(&T1, &pt->y, &pt->x);
  fieldPairResolveConvertMul(&A, &T0, &T1);

  // compute B
  fieldPairAdd(&T0, &acc->y, &acc->x);
  fieldPairSub(&T1, &pt->y, &pt->x);
  fieldPairAdd(&T1, &T1, &multiples[2]);      // twoN
  fieldPairResolveConvertMul(&B, &T0, &T1);

  // compute C, D
  fieldPairResolveConvertMul(&C, &acc->z, &pt->t);
  fieldPairAdd(&C, &C, &C);
  fieldPairResolveConvertMul(&D, &acc->t, &pt->z);
  fieldPairAdd(&D, &D, &D);

  // Compute E, F, G, H
  fieldPairAdd(&E, &D, &C);
  fieldPairSub(&F, &B, &A);
  fieldPairAdd(&F, &F, &multiples[2]);        // twoN
  fieldPairAdd(&G, &B, &A);
  fieldPairSub(&H, &D, &C);
  fieldPairAdd(&H, &H, &multiples[3]);        // threeN

  // Prep E, F, G, H for multiplies
  fieldPairResolveConvert(&E, &E);
  fieldPairResolveConvert(&F, &F);
  fieldPairResolveConvert(&G, &G);
  fieldPairResolveConvert(&H, &H);

  fieldPairMul(&Z, &F, &G);
  if((i64x2_extract_l(Z.v[0]) & 0x7FFFFFFFFFF0L)==0 || (i64x2_extract_h(Z.v[0]) & 0x7FFFFFFFFFF0L)==0) {
    // rather than checking is Z is some multiply of P, we just check some bits in the low word,
    // if they are all zero, we just play it safe and call the unified code.  Calls that are not
    // requiredshould be very rare.
    pointPairXYTZ_accumulateXYTZ_unified(acc, pt);
    return;
  }

  // Compute x, y, t and z
  fieldPairMul(&acc->x, &E, &F);
  fieldPairMul(&acc->y, &G, &H);
  fieldPairMul(&acc->t, &E, &H);

  fieldPairReduce(&acc->x, &acc->x);
  fieldPairReduce(&acc->y, &acc->y);
  fieldPairReduce(&acc->t, &acc->t);
  fieldPairResolve(&acc->z, &Z);

  // Output assertion:  0 <= acc.x, acc.y, acc.t <= 1.01*N
  //                    0 <= acc.z <= 3.20*N
}
