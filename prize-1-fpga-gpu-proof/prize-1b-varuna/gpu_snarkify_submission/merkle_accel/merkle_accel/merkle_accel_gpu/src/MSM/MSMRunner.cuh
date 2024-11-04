/***

Copyright (c) 2023-2024, Yrrid Software, Inc. and Snarkify Inc.  All rights reserved.
Dual licensed under the MIT License or the Apache License, Version 2.0.
See LICENSE for details.

Author(s):  Niall Emmart

***/

template<class BaseCurve, Accumulation accumulation, uint32_t windowBits, uint32_t binBits, uint32_t safety, bool uniformBuckets>
class MSMRunner {
  private:
  uint32_t _maxPointCount;
  uint32_t _pointCount;
  uint32_t _uniformBucketPointCount;
  void*    _uniformBucketMSM;

  public:
  typedef BaseCurve                                              Curve;
  typedef PlanningParameters<Curve, windowBits, binBits, safety> Planning;

  static const Accumulation ACCUMULATION=accumulation;
  static const uint32_t     WINDOW_BITS=windowBits;
  static const uint32_t     BUCKET_BITS=Planning::BUCKET_BITS;

  MSMRunner(uint32_t maxPointCount);
  ~MSMRunner();

  // memory management
  size_t pointBytesRequired();
  size_t planningBytesRequired();
  size_t bucketBytesRequired();
  size_t reduceBytesRequired();

  // MSM interfaces
  int32_t runPointGeneration(cudaStream_t stream, void* points, void* secretScalars, uint32_t pointCount);
  int32_t runPointPrecompute(cudaStream_t stream, void* scaledPoints, void* sourcePoints, uint32_t pointCount);
  int32_t runUniformBucketsSetup(cudaStream_t stream, void* reduceMemory, void* bucketMemory, void* planningMemory, void* pointMemory, uint32_t pointCount);
  int32_t runPlanning(cudaStream_t stream, void* planningMemory, void* scalars, uint32_t pointCount);
  int32_t runPlanning(cudaStream_t stream, void* planningMemory, void* scalars, uint32_t startPoint, uint32_t stopPoint);
  int32_t runAccumulate(cudaStream_t stream, void* bucketMemory, void* planningMemory, void* pointMemory, bool preloaded=false);
  int32_t runReduce(cudaStream_t stream, uint32_t* resultPoints, void* reduceMemory, void* bucketMemory);
  int32_t runFinalReduce(cudaStream_t stream, void* finalResult, void* reducedBuckets, uint32_t reducePointCount);
  int32_t runFinalReduceAndNormalize(cudaStream_t stream, void* finalResult, void* reducedBuckets, uint32_t reducePointCount, uint32_t batchCount);

  // debugging
  int32_t dumpOperations(const char* file, void* planningMemory, uint32_t pointCount); 
};
