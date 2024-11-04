/***

Copyright (c) 2023-2024, Yrrid Software, Inc. and Snarkify Inc.  All rights reserved.
Dual licensed under the MIT License or the Apache License, Version 2.0.
See LICENSE for details.

Author(s):  Niall Emmart

***/

class BLS12381G1 {
  public:
  static const uint32_t limbs=12;
  static const uint32_t EXPONENT_BITS=255;
  static const uint32_t EXPONENT_HIGH_WORD=0x73EDA753;

//  typedef Host::BLS12381G1 HostField;

  typedef ML::msm<ff_dispatch_st<ML::FF_BLS12381, ML::FF_BLS12381_INV>> MLMSM;

  typedef uint32_t Field[limbs];
  typedef uint64_t EvenOdd[2*limbs];

  typedef PointXY<Field> PointXY;
  typedef PointXYZZ<Field> PointXYZZ;  
  typedef PointXYT<Field> PointXYT;
  typedef PointXYTZ<Field> PointXYTZ;

  __device__ __forceinline__ static uint32_t qTerm(uint32_t x);

  __device__ __forceinline__ static void setOrder(uint32_t* order);

  __device__ __forceinline__ static void initializeShared();
  __device__ __forceinline__ static void dump(const Field& f);
  __device__ __forceinline__ static void print(const Field& f);
  __device__ __forceinline__ static void load(Field& r, const void* ptr);
  __device__ __forceinline__ static void store(void* ptr, const Field& f);
  __device__ __forceinline__ static void loadUnaligned(Field& r, const void* ptr);
  __device__ __forceinline__ static void storeUnaligned(void* ptr, const Field& f);
  __device__ __forceinline__ static void loadShared(Field& r, const uint32_t addr);
  __device__ __forceinline__ static void storeShared(uint32_t addr, const Field& f);

  __device__ __forceinline__ static void setZero(Field& r);
  __device__ __forceinline__ static void setOne(Field& r);
  __device__ __forceinline__ static void setN(Field& r);

  __device__ __forceinline__ static void setR(Field& r);
  __device__ __forceinline__ static void setRInv(Field& r);
  __device__ __forceinline__ static void setRSquared(Field& r);
  __device__ __forceinline__ static void setGeneratorX(Field& r);
  __device__ __forceinline__ static void setGeneratorY(Field& r);

  __device__ __forceinline__ static bool isZero(const Field& f);
  __device__ __forceinline__ static void reduce(Field& r, const Field& f);
  __device__ __forceinline__ static void reduceFully(Field& r, const Field& f);

  __device__ __forceinline__ static void setField(Field& r, const Field& f);
  __device__ __forceinline__ static void warpShuffleField(Field& r, const Field& f, const uint32_t lane);
  __device__ __forceinline__ static void addN(Field& r, const Field& f);
  __device__ __forceinline__ static void add2N(Field& r, const Field& f);
  __device__ __forceinline__ static void add3N(Field& r, const Field& f);
  __device__ __forceinline__ static void add4N(Field& r, const Field& f);
  __device__ __forceinline__ static void add5N(Field& r, const Field& f);

  __device__ __forceinline__ static void add(Field& r, const Field& a, const Field& b);
  __device__ __forceinline__ static void sub(Field& r, const Field& a, const Field& b);

  // high level math
  __device__ __forceinline__ static void modmul(Field& r, const Field& a, const Field& b);
  __device__ __forceinline__ static void modsqr(Field& r, const Field& a);
  __device__ __forceinline__ static void modinv(Field& r, const Field& f);
  __device__ __forceinline__ static void toMontgomery(Field& r, const Field& f);
  __device__ __forceinline__ static void fromMontgomery(Field& r, const Field& f);

  // Extended Jacobian Representation
  __device__ __forceinline__ static void normalize(PointXY& r, const PointXYZZ& pt);
  __device__ __forceinline__ static void negateXY(PointXY& pt);
  __device__ __forceinline__ static void initializeXYZZ(PointXYZZ& acc);
  __device__ __forceinline__ static void negateXYZZ(PointXYZZ& pt);
  __device__ __forceinline__ static void setXY(PointXYZZ& acc, const PointXY& pt);
  __device__ __forceinline__ static void setXYZZ(PointXYZZ& acc, const PointXYZZ& pt);
  __device__ __forceinline__ static void warpShuffleXYZZ(PointXYZZ& r, const PointXYZZ& pt, const uint32_t lane);
  __device__ __forceinline__ static void doubleXY(PointXYZZ& acc, const PointXY& pt);
  __device__ __forceinline__ static void doubleXYZZ(PointXYZZ& acc, const PointXYZZ& pt);
  __device__ __forceinline__ static void accumulateXY(PointXYZZ& acc, const PointXY& pt); 
  __device__ __forceinline__ static void accumulateXYZZ(PointXYZZ& acc, const PointXYZZ& pt); 
};
