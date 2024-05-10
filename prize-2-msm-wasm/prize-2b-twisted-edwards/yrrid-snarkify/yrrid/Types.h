/***

Copyright (c) 2023-2024, Yrrid Software, Inc. and Snarkify Inc.  All rights reserved.
Dual licensed under the MIT License or the Apache License, Version 2.0.
See LICENSE for details.

Author(s):  Niall Emmart

***/

typedef struct {
  uint64_t f0, f1, f2, f3, f4;
} Field_t __attribute__ ((aligned (16)));

typedef struct {
  uint64_t f0, f1, f2, f3;
} Field256_t __attribute__ ((aligned (16)));

typedef struct {
  I128 v[5];
} FieldPair_t __attribute ((aligned (16)));

typedef struct {
  I128 v[2];
} PackedField_t;

typedef struct {
  I128 v[4];
} PackedPointXY_t;

typedef struct {
  I128 v[7];
} PackedPointXYT;

typedef struct {
  FieldPair_t x;
  FieldPair_t y;
  FieldPair_t t;
} PointPairXYT_t;

typedef struct {
  FieldPair_t x;
  FieldPair_t y;
  FieldPair_t t;
  FieldPair_t z;
} PointPairXYTZ_t;

typedef struct {
  uint32_t          lock;
  uint32_t          padding;
  volatile uint32_t workerCount;
  volatile uint32_t checkedIn;
} Barrier_t;

/*
typedef struct {
  PointXY_t bucket;
  Field_t   invert;
  Field_t   invertChain;
  uint32_t  pointIndex;
  uint32_t  bucketIndex;
  uint32_t  specialProcessing;
  uint32_t  padding;
} BatchBucket_t __attribute__ ((aligned (16)));

typedef struct {
  PointXY_t point;
  PointXY_t endomorphism;
  uint32_t  nextIndex;
  uint32_t  referenceCount;
  uint32_t  padding0;
  uint32_t  padding1;
} BatchPoint_t __attribute__ ((aligned (16)));

typedef struct {
  uint32_t pointIndex;
  uint32_t bucketIndex;
} BatchRetry_t __attribute__ ((aligned (8)));

#if defined(BATCH_COUNT) && defined(BATCH_POINT_COUNT) && defined(BATCH_RETRY_COUNT) && defined(BATCH_UNLOCK_COUNT)

typedef struct {
  BatchBucket_t buckets[BATCH_COUNT];
  BatchPoint_t  points[BATCH_POINT_COUNT];
  BatchRetry_t  retries[BATCH_RETRY_COUNT*2];
  uint32_t      unlocks[BATCH_UNLOCK_COUNT];
} Batch_t __attribute__ ((aligned (16)));

#endif
*/

