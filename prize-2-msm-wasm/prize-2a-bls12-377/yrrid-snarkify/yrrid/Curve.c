/***

Copyright (c) 2023-2024, Yrrid Software, Inc. and Snarkify Inc.  All rights reserved.
Dual licensed under the MIT License or the Apache License, Version 2.0.
See LICENSE for details.

Author(s):  Niall Emmart

***/

void pointXYGenerator(PointXY_t* pt) {
  pt->x.v0=i64x2_make(0x33B9772451F4L, 0x169D5658260FL);
  pt->x.v1=i64x2_make(0x10DDC54DD773L, 0x5C1551C469A5L);
  pt->x.v2=i64x2_make(0x62E4425E1698L, 0x6F0652727616L);
  pt->x.v3=i64x2_make(0xFD4DC97D78CCL, 0x00A41206B361L);

  pt->y.v0=i64x2_make(0x961FB8CB81F3L, 0x5F44ADB88193L);
  pt->y.v1=i64x2_make(0xF54A00638D4CL, 0xFAFAF3DAD4DAL);
  pt->y.v2=i64x2_make(0x49E2D655CD18L, 0x01D52814C278L);
  pt->y.v3=i64x2_make(0x3C712EC3DDB4L, 0x007DA9332630L);
}

void pointXYZZInitialize(PointXYZZ_t* acc) {
  fieldSetZero(&acc->x);
  fieldSetZero(&acc->y);
  fieldSetZero(&acc->zz);
  fieldSetZero(&acc->zzz);
}

void pointXYLoad(PointXY_t* r, I128* bucketPtr) {
  fieldLoad(&r->x, bucketPtr);
  fieldLoad(&r->y, bucketPtr+3);
}

void pointXYStore(I128* r, PointXY_t* pt) {
  fieldStore(r, &pt->x);
  fieldStore(r+3, &pt->y);
}

void pointXYToMontgomery(PointXY_t* r, PointXY_t* pt) {
  Field_t rSquared;

  fieldSetRSquared(&rSquared);
  fieldMul(&r->x, &r->y, &rSquared, &pt->x, &rSquared, &pt->y);
}

void pointXYFromMontgomery(PointXY_t* r, PointXY_t* pt) {
  Field_t one;

  one.v0=i64x2_make(1L, 0L);
  one.v1=i64x2_splat(0L);
  one.v2=i64x2_splat(0L);
  one.v3=i64x2_splat(0L);

  fieldMul(&r->x, &r->y, &one, &pt->x, &one, &pt->y);
}

void pointXYZZFromMontgomery(PointXYZZ_t* r, PointXYZZ_t* pt) {
  Field_t one;

  one.v0=i64x2_make(1L, 0L);
  one.v1=i64x2_splat(0L);
  one.v2=i64x2_splat(0L);
  one.v3=i64x2_splat(0L);

  fieldMul(&r->x, &r->y, &one, &pt->x, &one, &pt->y);
  fieldMul(&r->zz, &r->zzz, &one, &pt->zz, &one, &pt->zzz);
}

void pointXYZZDoubleXY(PointXYZZ_t* res, PointXY_t* pt) {
  Field_t u, v, w, xSquared, s, twoS, m;
  Field_t zero, twoN, threeN, fiveN;

  fieldSetZero(&zero);
  fieldSet2N(&twoN);
  fieldSet3N(&threeN);
  fieldSet5N(&fiveN);

  fieldAdd(&u, &pt->y, &pt->y);
  fieldMul(&res->zz, &xSquared, &u, &u, &pt->x, &pt->x);
  fieldMul(&res->zzz, &s, &res->zz, &u, &res->zz, &pt->x);

  fieldAdd(&m, &xSquared, &xSquared);
  fieldAdd(&m, &m, &xSquared);
  fieldPartialResolve(&m);

  fieldAdd(&twoS, &s, &s);
  fieldMul(&res->x, &res->y, &m, &m, &res->zzz, &pt->y);

  fieldSub(&res->x, &res->x, &twoS);
  fieldAdd(&res->x, &res->x, &threeN);

  fieldSub(&s, &s, &res->x);
  fieldAdd(&s, &s, &fiveN);
  fieldMul(&s, &zero, &m, &s, &zero, &zero);

  fieldSub(&res->y, &s, &res->y);
  fieldAdd(&res->y, &res->y, &twoN);

  fieldPartialResolve(&res->x);
  fieldPartialResolve(&res->y);
}

void pointXYZZDoubleXYZZ(PointXYZZ_t *res, PointXYZZ_t* acc) {
  Field_t u, v, w, xSquared, s, twoS, m;
  Field_t zero, twoN, threeN, fiveN;

  fieldSetZero(&zero);
  fieldSet2N(&twoN);
  fieldSet3N(&threeN);
  fieldSet5N(&fiveN);

  fieldAdd(&u, &acc->y, &acc->y);
  fieldMul(&v, &xSquared, &u, &u, &acc->x, &acc->x);
  fieldMul(&w, &s, &v, &u, &v, &acc->x);

  fieldAdd(&m, &xSquared, &xSquared);
  fieldAdd(&m, &m, &xSquared);
  fieldPartialResolve(&m);

  fieldAdd(&twoS, &s, &s);
  fieldMul(&res->x, &res->y, &m, &m, &w, &acc->y);
  fieldMul(&res->zz, &res->zzz, &acc->zz, &v, &acc->zzz, &w);

  fieldSub(&res->x, &res->x, &twoS);
  fieldAdd(&res->x, &res->x, &threeN);

  fieldSub(&s, &s, &res->x);
  fieldAdd(&s, &s, &fiveN);
  fieldMul(&s, &zero, &m, &s, &zero, &zero);

  fieldSub(&res->y, &s, &res->y);
  fieldAdd(&res->y, &res->y, &twoN);

  fieldPartialResolve(&res->x);
  fieldPartialResolve(&res->y);
}

void pointXYZZAccumulateXY(PointXYZZ_t* acc, PointXY_t* pt) {
  Field_t u2, s2, p, pp, ppp, r, rr, q;
  Field_t twoN, fourN, sixN;

  if(fieldIsZero(&acc->zz)) {
    acc->x=pt->x;
    acc->y=pt->y;
    fieldSetOne(&acc->zz);
    fieldSetOne(&acc->zzz);
    return;
  }

  fieldSet2N(&twoN);
  fieldSet4N(&fourN);
  fieldSet6N(&sixN);

  fieldMul(&u2, &s2, &acc->zz, &pt->x, &acc->zzz, &pt->y);

  fieldSub(&p, &u2, &acc->x);
  fieldAdd(&p, &p, &sixN);

  fieldSub(&r, &s2, &acc->y);
  fieldAdd(&r, &r, &fourN);

  if(fieldIsZero(&p) && fieldIsZero(&r)) {
    pointXYZZDoubleXY(acc, pt);
    return;
  }

  fieldMul(&pp, &rr, &p, &p, &r, &r);
  fieldMul(&ppp, &q, &pp, &p, &pp, &acc->x);

  fieldAdd(&acc->x, &q, &q);
  fieldAdd(&acc->x, &acc->x, &ppp);
  fieldSub(&acc->x, &rr, &acc->x);
  fieldAdd(&acc->x, &acc->x, &fourN);
  fieldPartialResolve(&acc->x);

  fieldSub(&q, &q, &acc->x);
  fieldAdd(&q, &q, &sixN);

  fieldMul(&acc->zz, &acc->zzz, &acc->zz, &pp, &acc->zzz, &ppp);

  fieldMul(&acc->y, &q, &acc->y, &ppp, &q, &r);

  fieldSub(&acc->y, &q, &acc->y);
  fieldAdd(&acc->y, &acc->y, &twoN);
  fieldPartialResolve(&acc->y);
}

void pointXYZZAccumulateXYZZ(PointXYZZ_t* acc, PointXYZZ_t* pt) {
  Field_t u1, s1, u2, s2, p, pp, ppp, r, rr, q;
  Field_t twoN, fourN, sixN;

  fieldSet2N(&twoN);
  fieldSet4N(&fourN);
  fieldSet6N(&sixN);

  if(fieldIsZero(&pt->zz))
    return;

  if(fieldIsZero(&acc->zz)) {
    *acc=*pt;
    return;
  }

  fieldMul(&u1, &s1, &acc->x, &pt->zz, &acc->y, &pt->zzz);
  fieldMul(&u2, &s2, &acc->zz, &pt->x, &acc->zzz, &pt->y);

  fieldSub(&p, &u2, &u1);
  fieldAdd(&p, &p, &twoN);

  fieldSub(&r, &s2, &s1);
  fieldAdd(&r, &r, &twoN);

  if(fieldIsZero(&p) && fieldIsZero(&r)) {
    pointXYZZDoubleXYZZ(acc, pt);
    return;
  }

  fieldMul(&pp, &rr, &p, &p, &r, &r);
  fieldMul(&ppp, &q, &pp, &p, &pp, &u1);

  fieldAdd(&acc->x, &q, &q);
  fieldAdd(&acc->x, &acc->x,  &ppp);
  fieldSub(&acc->x, &rr, &acc->x);
  fieldAdd(&acc->x, &acc->x, &fourN);
  fieldPartialResolve(&acc->x);

  fieldSub(&q, &q, &acc->x);
  fieldAdd(&q, &q, &sixN);

  fieldMul(&acc->zz, &acc->zzz, &acc->zz, &pt->zz, &acc->zzz, &pt->zzz);
  fieldMul(&acc->zz, &acc->zzz, &acc->zz, &pp, &acc->zzz, &ppp);

  fieldMul(&acc->y, &q, &s1, &ppp, &q, &r);

  fieldSub(&acc->y, &q, &acc->y);
  fieldAdd(&acc->y, &acc->y, &twoN);
  fieldPartialResolve(&acc->y);
}

void pointXYZZNormalize(PointXY_t* r, PointXYZZ_t* xyzz) {
  Field_t iii, i, ii, zero;

  fieldSetZero(&zero);
  fieldInv(&iii, &xyzz->zzz);
  fieldMul(&i, &zero, &iii, &xyzz->zz, &zero, &zero);
  fieldMul(&ii, &zero, &i, &i, &zero, &zero);
  fieldMul(&r->x, &r->y, &xyzz->x, &ii, &xyzz->y, &iii);
}

