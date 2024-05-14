/***

Copyright (c) 2022, Yrrid Software, Inc.  All rights reserved.
Licensed under the Apache License, Version 2.0, see LICENSE for details.

Written by Niall Emmart.

***/

__launch_bounds__(256)
__global__ void precomputePointsKernel(void* pointsPtr, void* affinePointsPtr, uint32_t pointCount, int WINDOWS_NUM, int BITS_PER_WINDOW, bool is381) {
  typedef BLS12377::G1Montgomery           Field;
  typedef CurveXYZZ::HighThroughput<Field> AccumulatorXYZZ;
  typedef PointXY<Field>                   PointXY;

//    constexpr msm::field fd = msm::field();
//    AccumulatorXYZZ  acc;
    PointXY point, result;
  uint32_t         globalTID=blockIdx.x*blockDim.x+threadIdx.x, globalStride=blockDim.x*gridDim.x;

  for(uint32_t i=globalTID;i<pointCount;i+=globalStride) {
    point.loadUnaligned(byteOffset(affinePointsPtr, i, 104));
    #if 0   // FIX FIX FIX
      point.toInternal();
    #endif
    point.store(byteOffset(pointsPtr, i, 96));
      msm::point_xyzz acc = msm::point_at_infinity(is381);

      acc = msm::curve_add(acc, *(msm::point_affine *)&point, is381);

    #pragma unroll 1
    for(uint32_t j=1; j < WINDOWS_NUM; j++) {
        #pragma unroll 1
        for(uint32_t k=0; k < BITS_PER_WINDOW; k++)
            acc = msm::curve_dbl(acc, is381);
        auto aff = msm::to_affine(acc, is381);

        memory::store<msm::point_affine , memory::st_modifier::cs>(
                (msm::point_affine *)byteOffset(pointsPtr, pointCount * j + i, 96), aff);
    }
  }
}

