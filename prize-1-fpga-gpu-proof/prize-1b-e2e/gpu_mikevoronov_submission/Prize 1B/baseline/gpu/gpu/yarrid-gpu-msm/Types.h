/***

Copyright (c) 2023-2024, Yrrid Software, Inc. and Snarkify Inc.  All rights reserved.
Dual licensed under the MIT License or the Apache License, Version 2.0.
See LICENSE for details.

Author(s):  Niall Emmart

***/

typedef enum {
  ACCUMULATION_AFFINE,
  ACCUMULATION_EXTENDED_JACOBIAN,     // XYZZ
  ACCUMULATION_EXTENDED_JACOBIAN_ML,  // Use the Matter Labs XYZZ implementation
  ACCUMULATION_TWISTED_EDWARDS_XY,    // more modmuls, less memory bw + storage
  ACCUMULATION_TWISTED_EDWARDS_XYT,   // less modmuls, more memory bw + storage
} Accumulation;

enum {
  FIELDS_XY=2,
  FIELDS_XYZZ=4,
  FIELDS_XYT=3,
  FIELDS_XYTZ=4,
};

enum {
  FIELD_X=0,
  FIELD_Y=1,
  FIELD_ZZ=2,
  FIELD_ZZZ=3,
  FIELD_T=2,
  FIELD_Z=3,
};

typedef struct {
  uint32_t  maxPointCount;
  uint32_t  groupBits;
  uint32_t  groupCount;
  uint32_t  pointsPerBinGroup;
  uint32_t  binCount;

  uint32_t* pointIndexes;                    // size: N*WINDOWS
  uint32_t* sortedBucketIndexes;             // size: BUCKET_COUNT
  uint32_t* sortedBucketCounts;              // size: BUCKET_COUNT
  uint32_t* sortedBucketOffsets;             // size: BUCKET_COUNT

  uint32_t* binCounts;                       // size: BINS_PER_GROUP * GROUP_COUNT
  uint32_t* binOffsets;                      // size: BINS_PER_GROUP
  uint32_t* bigBinCounts;                    // size: 256
  uint32_t* binnedPointIndexes;              // size: N*WINDOWS*SAFETY

  uint32_t* overflowCount;                   // size: 1
  uint32_t* bucketOverflowCounts;            // size: BUCKET_COUNT
  uint2*    overflow;                        // size: N*WINDOWS

  uint32_t* bucketCounts;                    // size: BUCKET_COUNT
  uint32_t* bucketOffsets;                   // size: BUCKET_COUNT
  uint32_t* bucketSizeCounts;                // size: 256
  uint32_t* bucketSizeNexts;                 // size: 256
} PlanningLayout;

template<uint32_t n>
class BitSize {
  // number of bits required to store a number from 0 to n-1, e.g., BitSize<8>::bitSize is 3
  public:
  static const uint32_t lg=BitSize<n/2>::lg+1;
  static const uint32_t bitSize=BitSize<n-1>::lg;
};

template<>
class BitSize<1> {
  public:
  static const uint32_t lg=1;
  static const uint32_t bitSize=0;
};

template<class Field>
class PointXY {
  public:
  Field x, y;
};

template<class Field>
class PointXYZZ {
  public:
  Field x, y, zz, zzz;
};

template<class Field>
class PointXYT {
  public:
  Field x, y, t;
};

template<class Field>
class PointXYTZ {
  public:
  Field x, y, t, z;
};
