/***

Copyright (c) 2023-2024, Yrrid Software, Inc. and Snarkify Inc.  All rights reserved.
Dual licensed under the MIT License or the Apache License, Version 2.0.
See LICENSE for details.

Author(s):  Niall Emmart

***/

#include <stdio.h>
#include <stdint.h>
#include <stdbool.h>
#include "HostFields.cpp"
#include "HostCurves.cpp"
#include "asm.cu"
#include "Chain.cu"
#include "MP.cu"
#include "Types.h"
#include "Support.cu"
#include "MatterLabs.cuh"
#include "BLS12377.cuh"
#include "BLS12377.cu"
#include "BLS12381.cuh"
#include "BLS12381.cu"
#include "PlanningParameters.cu"
#include "PlanningKernels.cu"
#include "PrecomputeKernels.cu"
#include "AccumulateKernels.cu"
#include "ReduceKernels.cu"
#include "MSMRunner.cuh"
#include "MSMRunner.cu"

#define CUDA_CHECK(call) { int localEC=call; if(localEC!=cudaSuccess) { printf("\nCall \"" #call "\" failed from %s, line %d, error=%d\n", __FILE__, __LINE__, localEC); exit(1); } }

/********************************************************************************************
 * Possible run options for BLS12377G1:
 *
 *     typedef MSMRunner<BLS12377G1, ACCUMULATION_TWISTED_EDWARDS_XY, 23, 6, 2, false> Run377;
 *     typedef MSMRunner<BLS12377G1, ACCUMULATION_TWISTED_EDWARDS_XYT, 23, 6, 2, false> Run377;
 *     typedef MSMRunner<BLS12377G1, ACCUMULATION_EXTENDED_JACOBIAN, 23, 6, 2, false> Run377;
 *     typedef MSMRunner<BLS12377G1, ACCUMULATION_EXTENDED_JACOBIAN_ML, 23, 6, 2, false> Run377;
 *
 * Possible run options for BLS12381G1:
 *     typedef MSMRunner<BLS12381G1, ACCUMULATION_EXTENDED_JACOBIAN, 22, 5, 2, false> Run381;
 *     typedef MSMRunner<BLS12381G1, ACCUMULATION_EXTENDED_JACOBIAN_ML, 22, 5, 2, false> Run381;
 * 
 ********************************************************************************************/

typedef MSMRunner<BLS12377G1, ACCUMULATION_TWISTED_EDWARDS_XY, 23, 6, 2, true> Best377;
typedef MSMRunner<BLS12381G1, ACCUMULATION_EXTENDED_JACOBIAN_ML, 22, 5, 2, true> Best381;

class MSMBatch {
  public:
  uint32_t     curve;
  uint32_t     maxBatchCount;
  uint32_t     maxPointCount;

  Best377*     runner377;
  Best381*     runner381;

  void*        pointMemory;
  void*        scaledPointMemory;
  void*        scalarMemory;
  void*        planningMemory;
  void*        bucketMemory;
  void*        reduceMemory;

  cudaStream_t memoryStream;
  cudaStream_t runStream;
  cudaEvent_t* resultReadyEvents;

  // ok -- this isn't ideal to dispatch to one curve or the other... oh well

  MSMBatch(uint32_t inCurve, uint32_t inMaxBatchCount, uint32_t inMaxPointCount) {
    curve=inCurve;
    maxBatchCount=inMaxBatchCount;
    maxPointCount=inMaxPointCount;

    runner377=new Best377(inMaxPointCount);
    runner381=new Best381(inMaxPointCount);

    pointMemory=NULL;
    scaledPointMemory=NULL;
    scalarMemory=NULL;
    planningMemory=NULL;
    bucketMemory=NULL;
    reduceMemory=NULL;

    resultReadyEvents=NULL;
  }

  ~MSMBatch() {
    delete runner377;
    delete runner381;
  }

  size_t pointBytesRequired() {
    return curve==377 ? runner377->pointBytesRequired() : runner381->pointBytesRequired();
  }

  size_t planningBytesRequired() {
    return curve==377 ? runner377->planningBytesRequired() : runner381->planningBytesRequired();
  }

  size_t bucketBytesRequired() {
    return curve==377 ? runner377->bucketBytesRequired() : runner381->bucketBytesRequired();
  }

  size_t reduceBytesRequired() {
    return curve==377 ? runner377->reduceBytesRequired() : runner381->reduceBytesRequired();
  }

  int32_t runPointGeneration(cudaStream_t stream, void* pointMemory, void* secretScalars, uint32_t pointCount) {
    if(curve==377) 
      return runner377->runPointGeneration(stream, pointMemory, secretScalars, pointCount);
    else if(curve==381)
      return runner381->runPointGeneration(stream, pointMemory, secretScalars, pointCount);
    return -1;
  }

  int32_t runPointPrecompute(cudaStream_t stream, void* scaledPointMemory, void* pointMemory, uint32_t pointCount) {
    if(curve==377) 
      return runner377->runPointPrecompute(stream, scaledPointMemory, pointMemory, pointCount);
    else if(curve==381)
      return runner381->runPointPrecompute(stream, scaledPointMemory, pointMemory, pointCount);
    return -1;
  }

  int32_t runUniformBucketsSetup(cudaStream_t stream, void* reduceMemory, void* bucketMemory, void* planningMemory, void* pointMemory, uint32_t pointCount) {
    if(curve==377)
      return runner377->runUniformBucketsSetup(stream, reduceMemory, bucketMemory, planningMemory, pointMemory, pointCount);
    else if(curve==381)
      return runner381->runUniformBucketsSetup(stream, reduceMemory, bucketMemory, planningMemory, pointMemory, pointCount);
    return -1;
  }

  int32_t runPlanning(cudaStream_t stream, void* planningMemory, void* scalars, uint32_t startPoint, uint32_t stopPoint) {
    if(curve==377)
      return runner377->runPlanning(stream, planningMemory, scalars, startPoint, stopPoint);
    else if(curve==381)
      return runner381->runPlanning(stream, planningMemory, scalars, startPoint, stopPoint);
    return -1;
  }

  int32_t runPlanning(cudaStream_t stream, void* planningMemory, void* scalars, uint32_t pointCount) {
    if(curve==377)
      return runner377->runPlanning(stream, planningMemory, scalars, pointCount);
    else if(curve==381)
      return runner381->runPlanning(stream, planningMemory, scalars, pointCount);
    return -1;
  }

  int32_t runAccumulate(cudaStream_t stream, void* bucketMemory, void* planningMemory, void* scaledPointMemory, bool preloaded=false) {
    if(curve==377)
      return runner377->runAccumulate(stream, bucketMemory, planningMemory, scaledPointMemory, preloaded);
    else if(curve==381)
      return runner381->runAccumulate(stream, bucketMemory, planningMemory, scaledPointMemory, preloaded);
    return -1;
  }

  int32_t runReduce(cudaStream_t stream, uint32_t* resultPointCount, void* reduceMemory, void* bucketMemory) {
    if(curve==377)
      return runner377->runReduce(stream, resultPointCount, reduceMemory, bucketMemory);
    else if(curve==381)
      return runner381->runReduce(stream, resultPointCount, reduceMemory, bucketMemory);
    return -1;
  }

  int32_t runFinalReduce(void* finalResult, void* cpuReduceMemory, uint32_t resultPointCount) {
    if(curve==377) {
      runner377->runFinalReduce(finalResult, cpuReduceMemory, resultPointCount);
      return 0;
    }
    else if(curve==381) {
      runner381->runFinalReduce(finalResult, cpuReduceMemory, resultPointCount);
      return 0;
    }
    return -1;
  }
};

void* byteOffset(void* ptr, size_t bytes) {
  uint8_t* ptr8=(uint8_t*)ptr;

  return (void*)(ptr8+bytes);
}

void* scalarOffset(void* ptr, uint32_t scalarIndex) {
  size_t   offset=scalarIndex;
  uint8_t* ptr8=(uint8_t*)ptr;

  offset*=32;
  return (void*)(ptr8+offset);
}

void* reduceOffset(void* ptr, uint32_t reduceIndex, size_t reduceBytes) {
  uint8_t* ptr8=(uint8_t*)ptr;
  size_t   offset=reduceIndex;
  
  offset*=reduceBytes;
  return (void*)(ptr8+offset);
}

void* pointOffset(void* ptr, uint32_t pointIndex, uint32_t fields=2) {
  size_t   offset=pointIndex*fields;
  uint8_t* ptr8=(uint8_t*)ptr;

  offset*=48;
  return (void*)(ptr8+offset);
}

extern "C" {

void* createContext(uint32_t curve, uint32_t maxBatchCount, uint32_t maxPointCount) {
  MSMBatch* batch=new MSMBatch(curve, maxBatchCount, maxPointCount);

  CUDA_CHECK(cudaStreamCreate(&batch->memoryStream));
  CUDA_CHECK(cudaStreamCreate(&batch->runStream));

  batch->resultReadyEvents=(cudaEvent_t*)malloc(sizeof(cudaEvent_t)*maxBatchCount);

  for(int i=0;i<maxBatchCount;i++) 
    CUDA_CHECK(cudaEventCreate(&batch->resultReadyEvents[i]));
  
  return batch;
}

int32_t destroyContext(void* contextPtr) {
  MSMBatch* batch=(MSMBatch*)contextPtr;

  CUDA_CHECK(cudaStreamDestroy(batch->memoryStream));
  CUDA_CHECK(cudaStreamDestroy(batch->runStream));

  for(int i=0;i<batch->maxBatchCount;i++) 
    CUDA_CHECK(cudaEventDestroy(batch->resultReadyEvents[i]));

  // clean up as much memory as possible
  if(batch->pointMemory!=NULL) 
    CUDA_CHECK(cudaFree(batch->pointMemory));
  if(batch->scaledPointMemory!=NULL) 
    CUDA_CHECK(cudaFree(batch->scaledPointMemory));
  if(batch->scalarMemory!=NULL) 
    CUDA_CHECK(cudaFree(batch->scalarMemory));
  if(batch->planningMemory!=NULL)
    CUDA_CHECK(cudaFree(batch->planningMemory));
  if(batch->bucketMemory!=NULL)
    CUDA_CHECK(cudaFree(batch->bucketMemory));
  if(batch->reduceMemory!=NULL)
    CUDA_CHECK(cudaFree(batch->reduceMemory));

  delete batch;
  return 0;
}

int32_t preprocessPoints(void* contextPtr, void* pointData, uint32_t pointCount) {
  MSMBatch*    batch=(MSMBatch*)contextPtr;
  cudaStream_t stream;

  // clean up as much memory as possible
  if(batch->scalarMemory!=NULL) 
    CUDA_CHECK(cudaFree(batch->scalarMemory));
  batch->scalarMemory=NULL;
  if(batch->planningMemory!=NULL)
    CUDA_CHECK(cudaFree(batch->planningMemory));
  batch->planningMemory=NULL;
  if(batch->bucketMemory!=NULL)
    CUDA_CHECK(cudaFree(batch->planningMemory));
  batch->planningMemory=NULL;
  if(batch->reduceMemory!=NULL)
    CUDA_CHECK(cudaFree(batch->reduceMemory));
  batch->reduceMemory=NULL;

  CUDA_CHECK(cudaStreamCreate(&stream));
  CUDA_CHECK(cudaMalloc(&batch->pointMemory, 96*pointCount));
  CUDA_CHECK(cudaMemcpy(batch->pointMemory, pointData, 96*pointCount, cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMalloc(&batch->scaledPointMemory, batch->pointBytesRequired()));
  CUDA_CHECK(batch->runPointPrecompute(stream, batch->scaledPointMemory, batch->pointMemory, pointCount));

  // we're done with the point memory, clean it up
  CUDA_CHECK(cudaFree(batch->pointMemory));
  batch->pointMemory=NULL;

  // reallocate the required scalar memory and planning memory
  CUDA_CHECK(cudaMalloc(&batch->scalarMemory, 32*batch->maxBatchCount*batch->maxPointCount));
  CUDA_CHECK(cudaMalloc(&batch->planningMemory, batch->planningBytesRequired()));
  CUDA_CHECK(cudaMalloc(&batch->bucketMemory, batch->bucketBytesRequired()));
  CUDA_CHECK(cudaMalloc(&batch->reduceMemory, batch->maxBatchCount*batch->reduceBytesRequired()));

  // setup unform buckets
  CUDA_CHECK(batch->runUniformBucketsSetup(stream, batch->reduceMemory, batch->bucketMemory, batch->planningMemory, batch->scaledPointMemory, pointCount));

  CUDA_CHECK(cudaStreamDestroy(stream));
  return 0;
}

int32_t processBatches(void* contextPtr, void* resultData, void** scalarData, uint32_t batchCount, uint32_t pointCount) {
  MSMBatch*    batch=(MSMBatch*)contextPtr;
  size_t       reduceBytes=batch->reduceBytesRequired();
  void*        gpuScalars;
  void*        gpuReduce;
  void*        cpuScalars;
  void*        cpuReduce;
  uint32_t     oneEighth=pointCount>>3;
  uint32_t     threeEighths=3*oneEighth;
  uint32_t     oneHalf=pointCount>>1;
  uint32_t     resultPointCount;

  cpuReduce=malloc(reduceBytes);

  // copy and accumulate the first 1/8th of batch[0] scalars
  cpuScalars=scalarData[0];
  gpuScalars=batch->scalarMemory;
  gpuReduce=batch->reduceMemory;

  CUDA_CHECK(cudaMemcpyAsync(gpuScalars, cpuScalars, oneEighth*32, cudaMemcpyHostToDevice, batch->memoryStream));
  CUDA_CHECK(cudaStreamSynchronize(batch->memoryStream));
  CUDA_CHECK(batch->runPlanning(batch->runStream, batch->planningMemory, gpuScalars, 0, oneEighth));
  CUDA_CHECK(batch->runAccumulate(batch->runStream, batch->bucketMemory, batch->planningMemory, batch->scaledPointMemory));

  // copy and accumulate the next 3/8ths of batch[0] scalars
  CUDA_CHECK(cudaMemcpyAsync(scalarOffset(gpuScalars, oneEighth), scalarOffset(cpuScalars, oneEighth), threeEighths*32, cudaMemcpyHostToDevice, batch->memoryStream));
  CUDA_CHECK(cudaStreamSynchronize(batch->memoryStream));
  CUDA_CHECK(batch->runPlanning(batch->runStream, batch->planningMemory, gpuScalars, oneEighth, oneHalf));
  CUDA_CHECK(batch->runAccumulate(batch->runStream, batch->bucketMemory, batch->planningMemory, batch->scaledPointMemory, true));

  // copy and accumulate the last 1/2 of batch[0] scalars
  CUDA_CHECK(cudaMemcpyAsync(scalarOffset(gpuScalars, oneHalf), scalarOffset(cpuScalars, oneHalf), oneHalf*32, cudaMemcpyHostToDevice, batch->memoryStream));
  CUDA_CHECK(cudaStreamSynchronize(batch->memoryStream));
  CUDA_CHECK(batch->runPlanning(batch->runStream, batch->planningMemory, gpuScalars, oneHalf, pointCount));
  CUDA_CHECK(batch->runAccumulate(batch->runStream, batch->bucketMemory, batch->planningMemory, batch->scaledPointMemory, true));

  // reduce and announce results ready
  CUDA_CHECK(batch->runReduce(batch->runStream, &resultPointCount, gpuReduce, batch->bucketMemory));
  CUDA_CHECK(cudaEventRecord(batch->resultReadyEvents[0], batch->runStream));

  for(int i=1;i<batchCount;i++) {
    // copy and run the next set of scalars
    cpuScalars=scalarData[i];
    gpuScalars=scalarOffset(batch->scalarMemory, pointCount*i);
    gpuReduce=reduceOffset(batch->reduceMemory, i, reduceBytes);

    CUDA_CHECK(cudaMemcpyAsync(gpuScalars, cpuScalars, pointCount*32, cudaMemcpyHostToDevice, batch->memoryStream));
    CUDA_CHECK(cudaStreamSynchronize(batch->memoryStream));
    CUDA_CHECK(batch->runPlanning(batch->runStream, batch->planningMemory, gpuScalars, pointCount));
    CUDA_CHECK(batch->runAccumulate(batch->runStream, batch->bucketMemory, batch->planningMemory, batch->scaledPointMemory));
    CUDA_CHECK(batch->runReduce(batch->runStream, NULL, gpuReduce, batch->bucketMemory));
    CUDA_CHECK(cudaEventRecord(batch->resultReadyEvents[i], batch->runStream));

    // copy the results points of the prior run [run i-1] back and process them into a final result
    gpuReduce=reduceOffset(batch->reduceMemory, i-1, reduceBytes);

    CUDA_CHECK(cudaEventSynchronize(batch->resultReadyEvents[i-1]));
    CUDA_CHECK(cudaMemcpyAsync(cpuReduce, gpuReduce, reduceBytes, cudaMemcpyDeviceToHost, batch->memoryStream));
    CUDA_CHECK(cudaStreamSynchronize(batch->memoryStream));
    batch->runFinalReduce(pointOffset(resultData, i-1, FIELDS_XY), cpuReduce, resultPointCount); 
  }

  // copy the results points of the prior run [run i-1] back and process them into a final result
  gpuReduce=reduceOffset(batch->reduceMemory, batchCount-1, reduceBytes);
  CUDA_CHECK(cudaEventSynchronize(batch->resultReadyEvents[batchCount-1]));
  CUDA_CHECK(cudaMemcpyAsync(cpuReduce, gpuReduce, reduceBytes, cudaMemcpyDeviceToHost, batch->memoryStream));
  CUDA_CHECK(cudaStreamSynchronize(batch->memoryStream));
  batch->runFinalReduce(pointOffset(resultData, batchCount-1, FIELDS_XY), cpuReduce, resultPointCount); 

  free(cpuReduce);
  return 0;
}

}