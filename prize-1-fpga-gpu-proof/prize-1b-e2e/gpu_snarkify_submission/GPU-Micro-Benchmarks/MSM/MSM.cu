/***

Copyright (c) 2024, Snarkify, Inc. All rights reserved.
Dual licensed under the MIT License or the Apache License, Version 2.0.
See LICENSE for details.

Author(s):  Niall Emmart

***/

typedef MSMRunner<BLS12381G1, ACCUMULATION_EXTENDED_JACOBIAN_ML, 22, 6, 2, true> Best381;

class MSM {
  public:
  Best381*     runner;
  void*        scaledPointMemory;
  void*        planningMemory;
  void*        bucketMemory;
  void*        reduceMemory;
  void*        resultMemory;
  cudaStream_t stream;

  MSM() {
    runner=NULL;
    scaledPointMemory=NULL;
    planningMemory=NULL;
    bucketMemory=NULL;
    reduceMemory=NULL;
    resultMemory=NULL;
    stream=0;
  }

  ~MSM() {
    if(runner!=NULL)
      delete runner;
    if(scaledPointMemory!=NULL)
      $CUDA(cudaFree(scaledPointMemory));
    if(planningMemory!=NULL)
      $CUDA(cudaFree(planningMemory));
    if(bucketMemory!=NULL)
      $CUDA(cudaFree(bucketMemory));
    if(reduceMemory!=NULL)
      $CUDA(cudaFree(reduceMemory));
    if(resultMemory!=NULL)
      $CUDA(cudaFree(resultMemory));
  }

  void initialize(void* pointMemory, uint32_t pointCount, uint32_t maxBatchCount) {
    runner=new Best381(pointCount);
    $CUDA(cudaMalloc(&scaledPointMemory, runner->pointBytesRequired()));
    $CUDA(cudaMalloc(&planningMemory, runner->planningBytesRequired()));
    $CUDA(cudaMalloc(&bucketMemory, runner->bucketBytesRequired()));
    $CUDA(cudaMalloc(&reduceMemory, runner->reduceBytesRequired()*maxBatchCount));
    $CUDA(cudaMalloc(&resultMemory, 192*maxBatchCount));

    $CUDA(runner->runPointPrecompute(stream, scaledPointMemory, pointMemory, pointCount));
    $CUDA(runner->runUniformBucketsSetup(stream, reduceMemory, bucketMemory, planningMemory, scaledPointMemory, pointCount));
  }

  void* msm(void* scalars, uint32_t pointCount) {
    uint32_t resultPointCount;

    $CUDA(runner->runPlanning(stream, planningMemory, scalars, 0, pointCount));
    $CUDA(runner->runAccumulate(stream, bucketMemory, planningMemory, scaledPointMemory));
    $CUDA(runner->runReduce(stream, &resultPointCount, reduceMemory, bucketMemory));
    $CUDA(runner->runFinalReduceAndNormalize(stream, resultMemory, reduceMemory, resultPointCount, 1));
    return resultMemory;
  }

  void* msm(void** scalars, uint32_t pointCount, uint32_t batchCount) {
    uint32_t offset=0;
    uint32_t resultPointCount=0;

    for(int i=0;i<batchCount;i++) {
      $CUDA(runner->runPlanning(stream, planningMemory, scalars[i], 0, pointCount));
      $CUDA(runner->runAccumulate(stream, bucketMemory, planningMemory, scaledPointMemory));
      $CUDA(runner->runReduce(stream, &resultPointCount, fieldOffset(reduceMemory, offset*FIELDS_XYZZ), bucketMemory));
      offset+=resultPointCount;
    }

    $CUDA(runner->runFinalReduceAndNormalize(stream, resultMemory, reduceMemory, resultPointCount, batchCount));
    return resultMemory;
  }
};

