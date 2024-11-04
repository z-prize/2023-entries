/***

Copyright (c) 2023-2024, Yrrid Software, Inc. and Snarkify Inc.  All rights reserved.
Dual licensed under the MIT License or the Apache License, Version 2.0.
See LICENSE for details.

Author(s):  Niall Emmart

***/

template<class Curve>
__launch_bounds__(256)
__global__ void generatePoints(void* points, void* secrets, uint32_t pointCount) {
  uint32_t                  globalTID=blockIdx.x*blockDim.x+threadIdx.x, globalStride=blockDim.x*gridDim.x, bits;
  typename Curve::PointXYZZ scaled;
  typename Curve::PointXY   pt;

  Curve::initializeShared();
  for(uint32_t i=globalTID;i<pointCount;i+=globalStride) {
    Curve::initializeXYZZ(scaled);
    Curve::setGeneratorX(pt.x);
    Curve::setGeneratorY(pt.y);
    #pragma unroll 1
    for(int32_t j=7;j>=0;j--) {
      bits=((uint32_t*)secrets)[i*8+j];
      #pragma unroll 1
      for(int32_t k=31;k>=0;k--) {
        Curve::doubleXYZZ(scaled, scaled);
        if(((bits>>k) & 0x01)!=0)
          Curve::accumulateXY(scaled, pt);
      }
    }
    Curve::normalize(pt, scaled);
    Curve::reduceFully(pt.x, pt.x);
    Curve::reduceFully(pt.y, pt.y);
    Curve::store(fieldOffset(points, i*FIELDS_XY + FIELD_X), pt.x);
    Curve::store(fieldOffset(points, i*FIELDS_XY + FIELD_Y), pt.y);
  }
}

template<class Curve, uint32_t windowBits>
__launch_bounds__(256)
__global__ void scalePoints(void* scaledPoints, void* points, uint32_t pointCount, uint32_t maxPointCount) {
  uint32_t                  globalTID=blockIdx.x*blockDim.x+threadIdx.x, globalStride=blockDim.x*gridDim.x;
  typename Curve::PointXY   point;
  typename Curve::PointXYZZ scaled;

  Curve::initializeShared();
  for(uint32_t i=globalTID;i<pointCount;i+=globalStride) {
    Curve::load(point.x, fieldOffset(points, i*FIELDS_XY + FIELD_X));
    Curve::load(point.y, fieldOffset(points, i*FIELDS_XY + FIELD_Y));
    for(int window=0;window*windowBits<Curve::EXPONENT_BITS;window++) {
      Curve::reduceFully(point.x, point.x);
      Curve::reduceFully(point.y, point.y);
      Curve::store(fieldOffset(scaledPoints, (window*maxPointCount + i)*FIELDS_XY + FIELD_X), point.x);
      Curve::store(fieldOffset(scaledPoints, (window*maxPointCount + i)*FIELDS_XY + FIELD_Y), point.y);
      Curve::setXY(scaled, point);
      for(int32_t j=0;j<windowBits;j++)
        Curve::doubleXYZZ(scaled, scaled);
      Curve::normalize(point, scaled);
    }
  }   
}

template<class Curve, uint32_t windowBits>
__launch_bounds__(256)
__global__ void scaleTwistedEdwardsXYPoints(void* scaledPoints, void* points, uint32_t pointCount, uint32_t maxPointCount) {
  uint32_t                  globalTID=blockIdx.x*blockDim.x+threadIdx.x, globalStride=blockDim.x*gridDim.x;
  typename Curve::PointXY   point;
  typename Curve::PointXYZZ scaled;
  typename Curve::PointXYT  twistedPoint;

  Curve::initializeShared();
  for(uint32_t i=globalTID;i<pointCount;i+=globalStride) {
    Curve::load(point.x, fieldOffset(points, i*FIELDS_XY + FIELD_X));
    Curve::load(point.y, fieldOffset(points, i*FIELDS_XY + FIELD_Y));
    for(int window=0;window*windowBits<Curve::EXPONENT_BITS;window++) {
      Curve::toTwistedEdwards(twistedPoint, point);
      // throw away the T value
      Curve::reduceFully(twistedPoint.x, twistedPoint.x);
      Curve::reduceFully(twistedPoint.y, twistedPoint.y);
      Curve::store(fieldOffset(scaledPoints, (window*maxPointCount + i)*FIELDS_XY + FIELD_X), twistedPoint.x);
      Curve::store(fieldOffset(scaledPoints, (window*maxPointCount + i)*FIELDS_XY + FIELD_Y), twistedPoint.y);
      Curve::setXY(scaled, point);
      for(int32_t j=0;j<windowBits;j++)
        Curve::doubleXYZZ(scaled, scaled);
      Curve::normalize(point, scaled);
    }
  }   
}


template<class Curve, uint32_t windowBits>
__launch_bounds__(256)
__global__ void scaleTwistedEdwardsXYTPoints(void* scaledPoints, void* points, uint32_t pointCount, uint32_t maxPointCount) {
  uint32_t                  globalTID=blockIdx.x*blockDim.x+threadIdx.x, globalStride=blockDim.x*gridDim.x;
  typename Curve::PointXY   point;
  typename Curve::PointXYZZ scaled;
  typename Curve::PointXYT  twistedPoint;

  if(blockIdx.x==0 && threadIdx.x==0) printf("CALLED XYT SCALING!\n");

  Curve::initializeShared();
  for(uint32_t i=globalTID;i<pointCount;i+=globalStride) {
    Curve::load(point.x, fieldOffset(points, i*FIELDS_XY + FIELD_X));
    Curve::load(point.y, fieldOffset(points, i*FIELDS_XY + FIELD_Y));
    for(int window=0;window*windowBits<Curve::EXPONENT_BITS;window++) {
      Curve::toTwistedEdwards(twistedPoint, point);
      Curve::reduceFully(twistedPoint.x, twistedPoint.x);
      Curve::reduceFully(twistedPoint.y, twistedPoint.y);
      Curve::reduceFully(twistedPoint.t, twistedPoint.t);
      Curve::store(fieldOffset(scaledPoints, (window*maxPointCount + i)*FIELDS_XYT + FIELD_X), twistedPoint.x);
      Curve::store(fieldOffset(scaledPoints, (window*maxPointCount + i)*FIELDS_XYT + FIELD_Y), twistedPoint.y);
      Curve::store(fieldOffset(scaledPoints, (window*maxPointCount + i)*FIELDS_XYT + FIELD_T), twistedPoint.t);
      Curve::setXY(scaled, point);
      for(int32_t j=0;j<windowBits;j++)
        Curve::doubleXYZZ(scaled, scaled);
      Curve::normalize(point, scaled);
    }
  }   
}

