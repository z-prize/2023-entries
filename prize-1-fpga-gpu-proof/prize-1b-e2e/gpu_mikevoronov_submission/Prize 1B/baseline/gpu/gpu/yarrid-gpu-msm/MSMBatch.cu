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
 *     typedef MSMRunner<BLS12377G1, ACCUMULATION_TWISTED_EDWARDS_XY, 23, 6, 2> Run377;
 *     typedef MSMRunner<BLS12377G1, ACCUMULATION_TWISTED_EDWARDS_XYT, 23, 6, 2> Run377;
 *     typedef MSMRunner<BLS12377G1, ACCUMULATION_EXTENDED_JACOBIAN, 23, 6, 2> Run377;
 *     typedef MSMRunner<BLS12377G1, ACCUMULATION_EXTENDED_JACOBIAN_ML, 23, 6, 2> Run377;
 *
 * Possible run options for BLS12381G1:
 *     typedef MSMRunner<BLS12381G1, ACCUMULATION_EXTENDED_JACOBIAN, 22, 5, 2> Run381;
 *     typedef MSMRunner<BLS12381G1, ACCUMULATION_EXTENDED_JACOBIAN_ML, 22, 5, 2> Run381;
 * 
 ********************************************************************************************/

typedef MSMRunner<BLS12377G1, ACCUMULATION_TWISTED_EDWARDS_XY, 23, 6, 2> Best377;
typedef MSMRunner<BLS12381G1, ACCUMULATION_EXTENDED_JACOBIAN_ML, 22, 6, 2> Best381;

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

  cudaEvent_t  eighthReadyEvent;
  cudaEvent_t  halfReadyEvent;
  cudaEvent_t* scalarReadyEvents;
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

    scalarReadyEvents=NULL;
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

void* scalarOffset(void* ptr, int32_t scalarCount) {
  int64_t  offset=scalarCount;
  uint8_t* ptr8=(uint8_t*)ptr;

  offset*=32;
  return (void*)(ptr8+offset);
}

void* resultOffset(void* ptr, int32_t bytes) {
  uint8_t* ptr8=(uint8_t*)ptr;

  return (void*)(ptr8+bytes);
}

extern "C" {

void* createContext(uint32_t curve, uint32_t maxBatchCount, uint32_t maxPointCount) {
    MSMBatch* batch=new MSMBatch(curve, maxBatchCount, maxPointCount);

    batch->scalarReadyEvents=(cudaEvent_t*)malloc(sizeof(cudaEvent_t)*maxBatchCount);
    batch->resultReadyEvents=(cudaEvent_t*)malloc(sizeof(cudaEvent_t)*maxBatchCount);

    CUDA_CHECK(cudaEventCreate(&batch->eighthReadyEvent));
    CUDA_CHECK(cudaEventCreate(&batch->halfReadyEvent));
    for(int i=0;i<maxBatchCount;i++) {
    CUDA_CHECK(cudaEventCreate(&batch->scalarReadyEvents[i]));
    CUDA_CHECK(cudaEventCreate(&batch->resultReadyEvents[i]));
    }

    return batch;
}

int32_t destroyContext(void* contextPtr) {
    MSMBatch* batch=(MSMBatch*)contextPtr;

    CUDA_CHECK(cudaEventDestroy(batch->eighthReadyEvent));
    CUDA_CHECK(cudaEventDestroy(batch->halfReadyEvent));
    for(int i=0;i<batch->maxBatchCount;i++) {
    CUDA_CHECK(cudaEventDestroy(batch->scalarReadyEvents[i]));
    CUDA_CHECK(cudaEventDestroy(batch->resultReadyEvents[i]));
    }

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

int32_t preprocessOnlyPoints(void* contextPtr, void* pointData, uint32_t pointCount, void* preprocessedContext) {
    MSMBatch*    batch=(MSMBatch*)contextPtr;
	MSMBatch*    batchPre= (MSMBatch*)preprocessedContext;
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
        
	if (batchPre == nullptr) {
       // printf("call first\n");
		CUDA_CHECK(cudaStreamCreate(&stream));
        
        if (batch->scaledPointMemory != NULL) {
            CUDA_CHECK(cudaFree(batch->scaledPointMemory));    
            //printf("free\n");
        }
		CUDA_CHECK(cudaMalloc(&batch->scaledPointMemory, batch->pointBytesRequired()));
        
        if (batch->pointMemory != NULL)
            CUDA_CHECK(cudaFree(batch->pointMemory));            
		CUDA_CHECK(cudaMalloc(&batch->pointMemory, 96*pointCount));
        
		CUDA_CHECK(cudaMemcpy(batch->pointMemory, pointData, 96*pointCount, cudaMemcpyHostToDevice));
		CUDA_CHECK(batch->runPointPrecompute(stream, batch->scaledPointMemory, batch->pointMemory, pointCount));
		CUDA_CHECK(cudaStreamDestroy(stream));

		// we're done with the point memory, clean it up
		CUDA_CHECK(cudaFree(batch->pointMemory));
		batch->pointMemory=NULL;
	}
	else {
		batch->scaledPointMemory = batchPre->scaledPointMemory;
	}

    // reallocate the required scalar memory and planning memory    
    CUDA_CHECK(cudaMalloc(&batch->planningMemory, batch->planningBytesRequired()));
    CUDA_CHECK(cudaMalloc(&batch->bucketMemory, batch->bucketBytesRequired()));
    CUDA_CHECK(cudaMalloc(&batch->reduceMemory, batch->maxBatchCount*batch->reduceBytesRequired()));
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
  CUDA_CHECK(cudaStreamDestroy(stream));

  // we're done with the point memory, clean it up
  CUDA_CHECK(cudaFree(batch->pointMemory));
  batch->pointMemory=NULL;

  // reallocate the required scalar memory and planning memory
  CUDA_CHECK(cudaMalloc(&batch->scalarMemory, 32*batch->maxBatchCount*batch->maxPointCount));
  CUDA_CHECK(cudaMalloc(&batch->planningMemory, batch->planningBytesRequired()));
  CUDA_CHECK(cudaMalloc(&batch->bucketMemory, batch->bucketBytesRequired()));
  CUDA_CHECK(cudaMalloc(&batch->reduceMemory, batch->maxBatchCount*batch->reduceBytesRequired()));
  return 0;
}

int32_t process(void* contextPtr, void* resultData, void* scalarData, uint32_t pointCount, 
			    void* tmpResult, cudaStream_t runStream) {
	MSMBatch*    batch=(MSMBatch*)contextPtr;

	void*        gpuScalars;
	void*        gpuResults;
	void*        cpuScalars;
	uint32_t     resultPointCount;

	gpuScalars=scalarData;
	gpuResults=batch->reduceMemory;

	CUDA_CHECK(batch->runPlanning(runStream, batch->planningMemory, gpuScalars, pointCount));
	CUDA_CHECK(batch->runAccumulate(runStream, batch->bucketMemory, batch->planningMemory, batch->scaledPointMemory));
	CUDA_CHECK(batch->runReduce(runStream, &resultPointCount, gpuResults, batch->bucketMemory));
	//printf("result count %d\n", resultPointCount);
	//tmpResult=(void*)malloc(resultPointCount*192);

	gpuResults=batch->reduceMemory;
	CUDA_CHECK(cudaMemcpyAsync(tmpResult, gpuResults, resultPointCount*192, cudaMemcpyDeviceToHost, runStream));
	CUDA_CHECK(cudaStreamSynchronize(runStream));
	batch->runFinalReduce(resultOffset(resultData, 0), tmpResult, resultPointCount);
	return 0;
}

int32_t processBatches(void* contextPtr, void* resultData, void** scalarData, uint32_t batchCount, uint32_t pointCount) {
  MSMBatch*    batch=(MSMBatch*)contextPtr;
  cudaStream_t memoryStream, runStream;
  void*        gpuScalars;
  void*        gpuResults;
  void*        cpuScalars;
  void*        cpuResult;
  uint32_t     oneEighth=pointCount>>3;
  uint32_t     threeEighths=3*oneEighth;
  uint32_t     oneHalf=pointCount>>1;
  uint32_t     resultPointCount;

  CUDA_CHECK(cudaStreamCreate(&memoryStream));
  CUDA_CHECK(cudaStreamCreate(&runStream));

  // QUEUE UP ALL THE MEMORY COPIES

  gpuScalars=batch->scalarMemory;
  cpuScalars=scalarData[0];
  CUDA_CHECK(cudaMemcpyAsync(scalarOffset(gpuScalars, 0), scalarOffset(cpuScalars, 0), oneEighth*32, cudaMemcpyHostToDevice, memoryStream));
  CUDA_CHECK(cudaEventRecord(batch->eighthReadyEvent, memoryStream));
  CUDA_CHECK(cudaMemcpyAsync(scalarOffset(gpuScalars, oneEighth), scalarOffset(cpuScalars, oneEighth), threeEighths*32, cudaMemcpyHostToDevice, memoryStream));
  CUDA_CHECK(cudaEventRecord(batch->halfReadyEvent, memoryStream));
  CUDA_CHECK(cudaMemcpyAsync(scalarOffset(gpuScalars, oneHalf), scalarOffset(cpuScalars, oneHalf), oneHalf*32, cudaMemcpyHostToDevice, memoryStream));
  CUDA_CHECK(cudaEventRecord(batch->scalarReadyEvents[0], memoryStream));

  for(int i=1;i<batchCount;i++) {
    gpuScalars=scalarOffset(batch->scalarMemory, pointCount*i);
    cpuScalars=scalarData[i];
    CUDA_CHECK(cudaMemcpyAsync(gpuScalars, cpuScalars, pointCount*32, cudaMemcpyHostToDevice, memoryStream));
    CUDA_CHECK(cudaEventRecord(batch->scalarReadyEvents[i], memoryStream));
  }

  // QUEUE UP ALL THE RUNS 

  gpuScalars=batch->scalarMemory;
  gpuResults=batch->reduceMemory;
  CUDA_CHECK(cudaStreamWaitEvent(runStream, batch->eighthReadyEvent));
  CUDA_CHECK(batch->runPlanning(runStream, batch->planningMemory, gpuScalars, 0, oneEighth));
  CUDA_CHECK(batch->runAccumulate(runStream, batch->bucketMemory, batch->planningMemory, batch->scaledPointMemory));

  CUDA_CHECK(cudaStreamWaitEvent(runStream, batch->halfReadyEvent));
  CUDA_CHECK(batch->runPlanning(runStream, batch->planningMemory, gpuScalars, oneEighth, oneHalf));
  CUDA_CHECK(batch->runAccumulate(runStream, batch->bucketMemory, batch->planningMemory, batch->scaledPointMemory, true));

  CUDA_CHECK(cudaStreamWaitEvent(runStream, batch->scalarReadyEvents[0]));
  CUDA_CHECK(batch->runPlanning(runStream, batch->planningMemory, gpuScalars, oneHalf, pointCount));
  CUDA_CHECK(batch->runAccumulate(runStream, batch->bucketMemory, batch->planningMemory, batch->scaledPointMemory, true));
  CUDA_CHECK(batch->runReduce(runStream, &resultPointCount, gpuResults, batch->bucketMemory));
  CUDA_CHECK(cudaEventRecord(batch->resultReadyEvents[0], runStream));

  for(int i=1;i<batchCount;i++) {
    gpuScalars=scalarOffset(batch->scalarMemory, pointCount*i);
    gpuResults=resultOffset(batch->reduceMemory, batch->reduceBytesRequired()*i); 
    CUDA_CHECK(cudaStreamWaitEvent(runStream, batch->scalarReadyEvents[i]));
    CUDA_CHECK(batch->runPlanning(runStream, batch->planningMemory, gpuScalars, pointCount));
    CUDA_CHECK(batch->runAccumulate(runStream, batch->bucketMemory, batch->planningMemory, batch->scaledPointMemory));
    CUDA_CHECK(batch->runReduce(runStream, &resultPointCount, gpuResults, batch->bucketMemory));
    CUDA_CHECK(cudaEventRecord(batch->resultReadyEvents[i], runStream));
  }

  // PROCESS THE RESULTS

  cpuResult=(void*)malloc(resultPointCount*192);

  for(int i=0;i<batchCount;i++) {
    gpuResults=resultOffset(batch->reduceMemory, batch->reduceBytesRequired()*i); 
    CUDA_CHECK(cudaStreamWaitEvent(memoryStream, batch->resultReadyEvents[i]));
    CUDA_CHECK(cudaMemcpyAsync(cpuResult, gpuResults, resultPointCount*192, cudaMemcpyDeviceToHost, memoryStream));
    CUDA_CHECK(cudaStreamSynchronize(memoryStream));
    batch->runFinalReduce(resultOffset(resultData, 96*i), cpuResult, resultPointCount); 
  }

  free(cpuResult);

  CUDA_CHECK(cudaStreamDestroy(runStream));
  CUDA_CHECK(cudaStreamDestroy(memoryStream));
  return 0;
}

uint32_t randomWord() {
  uint32_t r=rand() & 0xFFFF;

  return (r<<16) | (rand() & 0xFFFF);
}

void randomWords(uint32_t* data, uint32_t count) {
  for(int i=0;i<count;i++)
    data[i]=randomWord();
}


void* createRandomPoints(const int SIZE, void* context, uint32_t* secret) {
  MSMBatch* batch=(MSMBatch*)context;
  void*     gpuPointMemory;
  void*     cpuPointMemory;
  
  cpuPointMemory=malloc(SIZE*96);
  CUDA_CHECK(cudaMalloc(&gpuPointMemory, SIZE*96));
  batch->runPointGeneration(0, gpuPointMemory, secret, SIZE);
  CUDA_CHECK(cudaMemcpy(cpuPointMemory, gpuPointMemory, SIZE*96, cudaMemcpyDeviceToHost));
  CUDA_CHECK(cudaFree(gpuPointMemory));
  return cpuPointMemory;    
}

int test(int SIZE, int BATCHES) {
  void*     context;
  uint32_t* secret;
  uint32_t* scalars;
  uint32_t* scalarSet[BATCHES];
  void*     points;
  uint32_t  curve=381;
  uint32_t  results[12*2*BATCHES];
  uint32_t  top;

  context=createContext(curve, BATCHES, SIZE);

  secret=(uint32_t*)malloc(SIZE*32);
  scalars=(uint32_t*)malloc(SIZE*32*BATCHES);
  for(int i=0;i<BATCHES;i++) 
    scalarSet[i]=scalars+SIZE*8*i;

  printf("Creating random scalars\n");
  randomWords(scalars, SIZE*8*BATCHES);

  top=(curve==377) ? 0x12AB655E : 0x73EDA753;
  for(int i=0;i<SIZE*BATCHES;i++) 
    scalars[i*8+7]=scalars[i*8+7] % top;

  printf("Creating random points\n");
  points=createRandomPoints(SIZE, context, secret);

  preprocessPoints(context, points, SIZE);
  processBatches(context, results, (void**)scalarSet, BATCHES, SIZE);

  /*for(int i=0;i<BATCHES*2;i++) {
    for(int j=11;j>=0;j--)
      printf("%08X", results[i*12+j]);
    printf("\n");
  }*/
  printf("Done!\n");
  destroyContext(context);
  
  return 0;
}

}