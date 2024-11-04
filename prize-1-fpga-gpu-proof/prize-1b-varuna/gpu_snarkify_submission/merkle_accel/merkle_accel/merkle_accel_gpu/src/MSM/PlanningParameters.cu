/***

Copyright (c) 2023-2024, Yrrid Software, Inc. and Snarkify Inc.  All rights reserved.
Dual licensed under the MIT License or the Apache License, Version 2.0.
See LICENSE for details.

Author(s):  Niall Emmart

***/

#include <stdio.h>
#include <stdint.h>

template<class Curve, uint32_t windowBits, uint32_t binBits, uint32_t safety>
class PlanningParameters {
  public:
  static const uint32_t EXPONENT_BITS=Curve::EXPONENT_BITS;
  static const uint32_t EXPONENT_HIGH_WORD=Curve::EXPONENT_HIGH_WORD;
  static const uint32_t WINDOW_BITS=windowBits;
  static const uint32_t BIN_BITS=binBits;
  static const uint32_t SAFETY=safety;

  static const uint32_t WINDOWS=(EXPONENT_BITS + WINDOW_BITS - 1) / WINDOW_BITS;
  static const uint32_t SIGN_BIT=31;

  // A bucket index decomposes into a BIN and a SEGMENT
  // A point index decomposes into a GROUP and an INDEX
  //
  // lg2(WINDOWS) + 1 (sign bit) + BIN_BITS + INDEX_BITS <= 32 bits.
  // Thus:
  //   BUCKET_BITS = BIN_BITS + SEGMENT_BITS
  //   N_BITS = GROUP_BITS + INDEX_BITS
  //   lg2(WINDOWS) + 1 + BIN_BITS + INDEX_BITS <= 32  ==>  INDEX_BITS = 32 - lg2(WINDOWS) - 1 - BUCKET_BITS + SEGMENT_BITS
  //
  // A good target is SEGMENT_BITS = 16

  static const uint32_t BUCKET_BITS=WINDOW_BITS-1;
  static const uint32_t SEGMENT_BITS=BUCKET_BITS-BIN_BITS;
  static const uint32_t INDEX_BITS=32 - BitSize<WINDOWS>::bitSize - 1 - BIN_BITS;

  static const uint32_t BUCKETS_PER_BIN=1u<<BIN_BITS;
  static const uint32_t BINS_PER_GROUP=1u<<SEGMENT_BITS;

  static const uint32_t BUCKET_COUNT=1u<<BUCKET_BITS;
  static const uint32_t BIN_OFFSET_COUNT=BINS_PER_GROUP;

  // these 5 can't be depend on pointCount and shouldn't be computed statically:
  //   static const uint32_t N_BITS=nBits;
  //   static const uint32_t GROUP_BITS=N_BITS-INDEX_BITS>0 ? N_BITS-INDEX_BITS : 0;
  //   static const uint32_t POINTS_PER_BIN=SAFETY*WINDOWS*(1u<<N_BITS - GROUP_BITS - SEGMENT_BITS);
  //   static const uint32_t GROUP_COUNT=1u<<GROUP_BITS;
  //   static const uint32_t BIN_COUNT=BINS_PER_GROUP*GROUP_COUNT;

  __device__ __forceinline__ static void setOrder(uint32_t* s) {
    Curve::setOrder(s);
  }

  __device__ __forceinline__ static void setN(uint32_t* f) {
    Curve::setN(f);
  }
};

