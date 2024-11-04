/***

Copyright (c) 2023-2024, Yrrid Software, Inc. and Snarkify Inc.  All rights reserved.
Dual licensed under the MIT License or the Apache License, Version 2.0.
See LICENSE for details.

Author(s):  Niall Emmart

***/

/*************************************************************************************************
 *
 * Typical memory usage for 2^24 points:
 *
 * total bytes:  N * WINDOWS * (SAFETY + 3) * 4 = 264 * N --> 4224 MB
 *               BUCKET_COUNT * 6 * 4                     --> 96 MB
 *               BINS_PER_GROUP * (GROUP_COUNT + 1) * 4   --> 2.25 MB
 *               Misc crap < 0.5 MB
 *
 *               Scalars:                                       512 MB
 *               Bucket storage:                                384 MB
 *               Point Storage:                               18432 MB
 *                                                         ------------
 *               Total:                                      23,651 MB
 *
 *************************************************************************************************/

void* advancePointer(void*& ptr, size_t bytes) {
  size_t   add=(bytes + 0x3FFL) & 0xFFFFFFFFFFFFFC00L;
  uint8_t* ptr8=(uint8_t*)ptr;
  void*    prior=ptr;

  ptr=(void*)(ptr8+add);
  return prior;
}

void advanceOffset(size_t& current, size_t bytes) {
  size_t add=(bytes + 0x3FFL) & 0xFFFFFFFFFFFFFC00L;

  current=current + add;
}

template<class Curve, Accumulation accumulation, uint32_t windowBits, uint32_t binBits, uint32_t safety, bool uniformBuckets>
MSMRunner<Curve, accumulation, windowBits, binBits, safety, uniformBuckets>::MSMRunner(uint32_t maxPointCount) {
  _maxPointCount=maxPointCount;
  _uniformBucketPointCount=0xFFFFFFFF;
  _uniformBucketMSM=NULL;
}

template<class Curve, Accumulation accumulation, uint32_t windowBits, uint32_t binBits, uint32_t safety, bool uniformBuckets>
MSMRunner<Curve, accumulation, windowBits, binBits, safety, uniformBuckets>::~MSMRunner() {
  if(_uniformBucketMSM!=NULL)
    cudaFree(_uniformBucketMSM);
}

template<class Curve, Accumulation accumulation, uint32_t windowBits, uint32_t binBits, uint32_t safety, bool uniformBuckets>
size_t MSMRunner<Curve, accumulation, windowBits, binBits, safety, uniformBuckets>::pointBytesRequired() {
  const uint32_t WINDOWS=Planning::WINDOWS;
  const uint32_t FIELDS_PER_POINT=(accumulation==ACCUMULATION_TWISTED_EDWARDS_XYT) ? 3 : 2;

  return ((size_t)(WINDOWS*FIELDS_PER_POINT*Curve::limbs*4))*_maxPointCount;
}

template<class Curve, Accumulation accumulation, uint32_t windowBits, uint32_t binBits, uint32_t safety, bool uniformBuckets>
size_t MSMRunner<Curve, accumulation, windowBits, binBits, safety, uniformBuckets>::planningBytesRequired() {
  const uint32_t WINDOWS=Planning::WINDOWS;
  const uint32_t INDEX_BITS=Planning::INDEX_BITS;
  const uint32_t BINS_PER_GROUP=Planning::BINS_PER_GROUP;
  const uint32_t BUCKET_COUNT=Planning::BUCKET_COUNT;

  size_t         current=0;
  uint32_t       groupCount;

  groupCount=_maxPointCount>>INDEX_BITS;
  groupCount=(groupCount>0) ? groupCount : 1;

  advanceOffset(current, _maxPointCount*WINDOWS*4);             // pointIndexes
  advanceOffset(current, BUCKET_COUNT*4);                       // sortedBucketIndexes
  advanceOffset(current, BUCKET_COUNT*4);                       // sortedBucketCounts
  advanceOffset(current, BUCKET_COUNT*4);                       // sortedBucketOffsets

  if(uniformBuckets)
    advanceOffset(current, _maxPointCount*32);                  // randomScalars

  advanceOffset(current, BINS_PER_GROUP*groupCount*4);          // binCounts
  advanceOffset(current, BINS_PER_GROUP*4);                     // binOffsets
  advanceOffset(current, 256*4);                                // bigBinCounts
  advanceOffset(current, _maxPointCount*safety*WINDOWS*4);      // binnedPointIndexes

  advanceOffset(current, 4);                                    // overflowCount
  advanceOffset(current, BUCKET_COUNT*4);                       // bucketOverflowCounts
  advanceOffset(current, _maxPointCount*WINDOWS*8);             // overflowValues

  advanceOffset(current, BUCKET_COUNT*4);                       // bucketCounts
  advanceOffset(current, BUCKET_COUNT*4);                       // bucketOffsets
  advanceOffset(current, 256*4);                                // bucketSizeCounts
  advanceOffset(current, 256*4);                                // bucketSizeNexts

  return current;
}

template<class Curve, Accumulation accumulation, uint32_t windowBits, uint32_t binBits, uint32_t safety, bool uniformBuckets>
size_t MSMRunner<Curve, accumulation, windowBits, binBits, safety, uniformBuckets>::bucketBytesRequired() {
  const uint32_t BUCKET_BITS=Planning::BUCKET_BITS;
  const uint32_t FIELDS_PER_BUCKET=(accumulation==ACCUMULATION_AFFINE) ? 2 : 4;
  
  return ((size_t)FIELDS_PER_BUCKET*Curve::limbs*4)<<BUCKET_BITS;
}

template<class Curve, Accumulation accumulation, uint32_t windowBits, uint32_t binBits, uint32_t safety, bool uniformBuckets>
size_t MSMRunner<Curve, accumulation, windowBits, binBits, safety, uniformBuckets>::reduceBytesRequired() {
  const uint32_t MAX_SM_COUNT=256;
  const uint32_t MAX_WARP_COUNT=8;
  const uint32_t FIELDS_PER_BUCKET=4;

  return (size_t)(FIELDS_PER_BUCKET*Curve::limbs*4*MAX_SM_COUNT*MAX_WARP_COUNT); 
}

template<class Curve, Accumulation accumulation, uint32_t windowBits, uint32_t binBits, uint32_t safety, bool uniformBuckets>
int32_t MSMRunner<Curve, accumulation, windowBits, binBits, safety, uniformBuckets>::
runPointGeneration(cudaStream_t stream, void* points, void* secretScalars, uint32_t pointCount) {
  uint32_t* cpuScalars;
  void*     gpuScalars;
  int32_t   ec, smCount;

  if(secretScalars==NULL)
    cpuScalars=(uint32_t*)malloc(pointCount*32);
  else
    cpuScalars=(uint32_t*)secretScalars;

  for(int i=0;i<pointCount;i++) {
    for(int j=0;j<8;j++) 
      cpuScalars[i*8+j]=((rand() & 0xFFFF)<<16) + (rand() & 0xFFFF);
  }

  ec=cudaDeviceGetAttribute(&smCount, cudaDevAttrMultiProcessorCount, 0);
  if(ec!=cudaSuccess) return ec;
  ec=cudaMalloc(&gpuScalars, pointCount*32);
  if(ec!=cudaSuccess) return ec;
  ec=cudaMemcpy(gpuScalars, cpuScalars, pointCount*32, cudaMemcpyHostToDevice);
  if(ec!=cudaSuccess) return ec;
  generatePoints<Curve><<<smCount, 256, 1536, stream>>>(points, gpuScalars, pointCount);
  ec=cudaStreamSynchronize(stream);
  if(ec!=cudaSuccess) return ec;
  ec=cudaFree(gpuScalars);
  if(ec!=cudaSuccess) return ec;
  if(secretScalars==NULL)
    free(cpuScalars);
  return cudaSuccess;
}

template<class Curve, Accumulation accumulation, uint32_t windowBits, uint32_t binBits, uint32_t safety, bool uniformBuckets>
int32_t MSMRunner<Curve, accumulation, windowBits, binBits, safety, uniformBuckets>::
runPointPrecompute(cudaStream_t stream, void* scaledPoints, void* sourcePoints, uint32_t pointCount) {
  int32_t ec=0, smCount;

  ec=cudaDeviceGetAttribute(&smCount, cudaDevAttrMultiProcessorCount, 0);
  if(ec!=cudaSuccess) return ec;
  if constexpr (accumulation==ACCUMULATION_AFFINE || accumulation==ACCUMULATION_EXTENDED_JACOBIAN || accumulation==ACCUMULATION_EXTENDED_JACOBIAN_ML)
    scalePoints<Curve, windowBits><<<smCount, 256, 1536, stream>>>(scaledPoints, sourcePoints, pointCount, _maxPointCount);
  else if constexpr (accumulation==ACCUMULATION_TWISTED_EDWARDS_XY) 
    scaleTwistedEdwardsXYPoints<Curve, windowBits><<<smCount, 256, 1536, stream>>>(scaledPoints, sourcePoints, pointCount, _maxPointCount);
  else if constexpr (accumulation==ACCUMULATION_TWISTED_EDWARDS_XYT) 
    scaleTwistedEdwardsXYTPoints<Curve, windowBits><<<smCount, 256, 1536, stream>>>(scaledPoints, sourcePoints, pointCount, _maxPointCount);
  ec=cudaStreamSynchronize(stream);
  if(ec!=cudaSuccess) return ec;
  return 0;
}

template<class Curve, Accumulation accumulation, uint32_t windowBits, uint32_t binBits, uint32_t safety, bool uniformBuckets>
int32_t MSMRunner<Curve, accumulation, windowBits, binBits, safety, uniformBuckets>::
runUniformBucketsSetup(cudaStream_t stream, void* reduceMemory, void* bucketMemory, void* planningMemory, void* pointMemory, uint32_t pointCount) {
  const uint32_t FIELDS_PER_BUCKET=4;
  const uint32_t WINDOWS=Planning::WINDOWS;
  const uint32_t BUCKET_COUNT=Planning::BUCKET_COUNT;
  const uint32_t EXPONENT_HIGH_WORD=Planning::EXPONENT_HIGH_WORD;

  void*    randomScalars;
  uint32_t resultPointCount;
  size_t   reduceBytes=reduceBytesRequired();
  int32_t  ec;

  // UNIFORM BUCKET SUPPORT:
  // 1.  If this is the first call, or if the point size has changed since the prior call:
  // 2.    If this is the first call then:
  // 3.      Generate random scalars and copy them to the GPU
  // 4.    compute the MSM of the random scalars of size "pointCount"

  if(!uniformBuckets || _uniformBucketPointCount==pointCount)
    return cudaSuccess;

  printf("Running uniform setup code\n");

  randomScalars=planningMemory;
  advancePointer(randomScalars, _maxPointCount*WINDOWS*4);
  advancePointer(randomScalars, BUCKET_COUNT*4);
  advancePointer(randomScalars, BUCKET_COUNT*4);
  advancePointer(randomScalars, BUCKET_COUNT*4);

  if(_uniformBucketPointCount==0xFFFFFFFF) {
    uint32_t* cpuRandomScalars=(uint32_t*)malloc(_maxPointCount*32);

    // For a production system, this should be a cryptographic random number generated, with a 128 bit seed.
    // Also, note, this is not quite a uniform distribution, but it's pretty close.  Again, for production,
    // I'd go with a good quality uniform field random generator, rather than just modding the top word.

    for(int i=0;i<_maxPointCount;i++) {
      for(int j=0;j<8;j++) 
        cpuRandomScalars[i*8+j]=((rand() & 0xFFFF)<<16) + (rand() & 0xFFFF);
      cpuRandomScalars[i*8+7]=cpuRandomScalars[i*8+7] % EXPONENT_HIGH_WORD;
    }
    ec=cudaMemcpy(randomScalars, cpuRandomScalars, _maxPointCount*32, cudaMemcpyHostToDevice);
    if(ec!=cudaSuccess)
      return ec;
    free(cpuRandomScalars);
  }

  _uniformBucketPointCount=0xFFFFFFFF;

  ec=cudaMalloc(&_uniformBucketMSM, FIELDS_PER_BUCKET*Curve::limbs*4);
  if(ec!=cudaSuccess) return ec;
  // now we compute the MSM for the number of points that we have
  ec=runPlanning(stream, planningMemory, NULL, pointCount);
  if(ec!=cudaSuccess) return ec;
  ec=runAccumulate(stream, bucketMemory, planningMemory, pointMemory);
  if(ec!=cudaSuccess) return ec;
  ec=runReduce(stream, &resultPointCount, reduceMemory, bucketMemory);
  if(ec!=cudaSuccess) return ec;
  ec=runFinalReduce(stream, _uniformBucketMSM, reduceMemory, resultPointCount);
  if(ec!=cudaSuccess) return ec;
  _uniformBucketPointCount=pointCount;

  return cudaSuccess;
}

template<class Curve, Accumulation accumulation, uint32_t windowBits, uint32_t binBits, uint32_t safety, bool uniformBuckets>
int32_t MSMRunner<Curve, accumulation, windowBits, binBits, safety, uniformBuckets>::
runPlanning(cudaStream_t stream, void* planningMemory, void* scalars, uint32_t startPoint, uint32_t stopPoint) {
  const uint32_t WINDOWS=Planning::WINDOWS;
  const uint32_t INDEX_BITS=Planning::INDEX_BITS;
  const uint32_t SEGMENT_BITS=Planning::SEGMENT_BITS;
  const uint32_t BINS_PER_GROUP=Planning::BINS_PER_GROUP;
  const uint32_t BUCKET_COUNT=Planning::BUCKET_COUNT;

  PlanningLayout planning;
  uint32_t       groupBits, groupCount, pointsPerBinGroup;
  int32_t        ec, smCount;

  groupCount=_maxPointCount>>INDEX_BITS;
  groupCount=(groupCount>0) ? groupCount : 1;
  pointsPerBinGroup=safety*WINDOWS<<((INDEX_BITS-SEGMENT_BITS>0) ? INDEX_BITS-SEGMENT_BITS : 0);

  groupBits=0;
  for(int i=groupCount-1;i>0;i=i>>1)
    groupBits++;

  planning.maxPointCount=_maxPointCount;
  planning.groupBits=groupBits;
  planning.groupCount=groupCount;
  planning.pointsPerBinGroup=pointsPerBinGroup;
  planning.binCount=BINS_PER_GROUP*groupCount;

  planning.pointIndexes=(uint32_t*)advancePointer(planningMemory, _maxPointCount*WINDOWS*4);
  planning.sortedBucketIndexes=(uint32_t*)advancePointer(planningMemory, BUCKET_COUNT*4);
  planning.sortedBucketCounts=(uint32_t*)advancePointer(planningMemory, BUCKET_COUNT*4);
  planning.sortedBucketOffsets=(uint32_t*)advancePointer(planningMemory, BUCKET_COUNT*4);

  planning.randomScalars=NULL;
  if(uniformBuckets) 
    planning.randomScalars=(uint32_t*)advancePointer(planningMemory, _maxPointCount*32);

  planning.binCounts=(uint32_t*)advancePointer(planningMemory, BINS_PER_GROUP*groupCount*4);
  planning.binOffsets=(uint32_t*)advancePointer(planningMemory, BINS_PER_GROUP*4);
  planning.bigBinCounts=(uint32_t*)advancePointer(planningMemory, 256*4);
  planning.binnedPointIndexes=(uint32_t*)advancePointer(planningMemory, _maxPointCount*safety*WINDOWS*4);

  planning.overflowCount=(uint32_t*)advancePointer(planningMemory, 4);
  planning.bucketOverflowCounts=(uint32_t*)advancePointer(planningMemory, BUCKET_COUNT*4);
  planning.overflow=(uint2*)advancePointer(planningMemory, _maxPointCount*WINDOWS*8);

  planning.bucketCounts=(uint32_t*)advancePointer(planningMemory, BUCKET_COUNT*4);
  planning.bucketOffsets=(uint32_t*)advancePointer(planningMemory, BUCKET_COUNT*4);
  planning.bucketSizeCounts=(uint32_t*)advancePointer(planningMemory, 256*4);
  planning.bucketSizeNexts=(uint32_t*)advancePointer(planningMemory, 256*4);

  ec=cudaDeviceGetAttribute(&smCount, cudaDevAttrMultiProcessorCount, 0);
  if(ec!=0) return ec;

  zeroCounters<Planning><<<256, 256>>>(planning);
  if constexpr (!uniformBuckets)
    partitionIntoBins<Planning><<<smCount, 512>>>(planning, (uint4*)scalars, startPoint, stopPoint);
  else {
    if(scalars==NULL) 
      partitionIntoBins<Planning><<<smCount, 512>>>(planning, (uint4*)planning.randomScalars, startPoint, stopPoint);
    else 
      partitionIntoBins<Planning><<<smCount, 512>>>(planning, (uint4*)scalars, (uint4*)planning.randomScalars, startPoint, stopPoint);
  }
  computeBinOffsets<Planning><<<256, 256>>>(planning);
  sortBins<Planning><<<BINS_PER_GROUP/64, 256, pointsPerBinGroup*groupCount*4>>>(planning);
  processOverflows<Planning><<<smCount, 512>>>(planning);
  sortByCounts<Planning><<<smCount, 512>>>(planning);

  return 0;
}

template<class Curve, Accumulation accumulation, uint32_t windowBits, uint32_t binBits, uint32_t safety, bool uniformBuckets>
int32_t MSMRunner<Curve, accumulation, windowBits, binBits, safety, uniformBuckets>::runPlanning(cudaStream_t stream, void* planningMemory, void* scalars, uint32_t pointCount) {
  return runPlanning(stream, planningMemory, scalars, 0, pointCount);
}

template<class Curve, Accumulation accumulation, uint32_t windowBits, uint32_t binBits, uint32_t safety, bool uniformBuckets>
int32_t MSMRunner<Curve, accumulation, windowBits, binBits, safety, uniformBuckets>::
runAccumulate(cudaStream_t stream, void* bucketMemory, void* planningMemory, void* pointMemory, bool preloaded) {
  const uint32_t WINDOWS=Planning::WINDOWS;
  const uint32_t BUCKET_COUNT=Planning::BUCKET_COUNT;

  PlanningLayout planning;
  int32_t        ec;

  planning.pointIndexes=(uint32_t*)advancePointer(planningMemory, _maxPointCount*WINDOWS*4);
  planning.sortedBucketIndexes=(uint32_t*)advancePointer(planningMemory, BUCKET_COUNT*4);
  planning.sortedBucketCounts=(uint32_t*)advancePointer(planningMemory, BUCKET_COUNT*4);
  planning.sortedBucketOffsets=(uint32_t*)advancePointer(planningMemory, BUCKET_COUNT*4);

  if constexpr (accumulation==ACCUMULATION_TWISTED_EDWARDS_XYT) {
    ec=cudaFuncSetAttribute(accumulateTwistedEdwardsXYTBuckets<Curve, 2>, cudaFuncAttributePreferredSharedMemoryCarveout, cudaSharedmemCarveoutMaxShared);
    if(ec!=0) return ec;
  }

  // about ACCUMULATION_EXTENDED_JACOBIAN and ACCUMULATION_EXTENDED_JACOBIAN_ML are about 1.5% faster with 3 blocks instead of 2
  if constexpr (accumulation==ACCUMULATION_EXTENDED_JACOBIAN) 
    accumulateExtendedJacobianBuckets<Curve, 3><<<BUCKET_COUNT/128, 128, 19616, stream>>>(bucketMemory, planning, pointMemory, preloaded);
  else if constexpr (accumulation==ACCUMULATION_EXTENDED_JACOBIAN_ML) {
    accumulateExtendedJacobianMLBuckets<Curve, 3><<<BUCKET_COUNT/128, 128, 19616, stream>>>(bucketMemory, planning, pointMemory, preloaded);
  }
  else if constexpr (accumulation==ACCUMULATION_TWISTED_EDWARDS_XY) {
    accumulateTwistedEdwardsXYBuckets<Curve, 2><<<BUCKET_COUNT/128, 128, 32416, stream>>>(bucketMemory, planning, pointMemory, preloaded);
  }
  else if constexpr (accumulation==ACCUMULATION_TWISTED_EDWARDS_XYT) {
    accumulateTwistedEdwardsXYTBuckets<Curve, 2><<<BUCKET_COUNT/128, 128, 32416, stream>>>(bucketMemory, planning, pointMemory, preloaded);
  }
  return cudaSuccess;
}

template<class Curve, Accumulation accumulation, uint32_t windowBits, uint32_t binBits, uint32_t safety, bool uniformBuckets>
int32_t MSMRunner<Curve, accumulation, windowBits, binBits, safety, uniformBuckets>::
runReduce(cudaStream_t stream, uint32_t* reducePointCount, void* reduceMemory, void* bucketMemory) {
  int32_t ec, smCount;

  if(reducePointCount!=NULL)
    *reducePointCount=0;

  ec=cudaDeviceGetAttribute(&smCount, cudaDevAttrMultiProcessorCount, 0);
  if(ec!=0) return ec;

  if(reducePointCount!=NULL)
    *reducePointCount=smCount*8;

  if constexpr (accumulation==ACCUMULATION_EXTENDED_JACOBIAN || accumulation==ACCUMULATION_EXTENDED_JACOBIAN_ML) {
    reduceExtendedJacobianBuckets<Curve, windowBits-1><<<smCount, 256, 1536, stream>>>(reduceMemory, bucketMemory);
  }
  else if constexpr (accumulation==ACCUMULATION_TWISTED_EDWARDS_XY || accumulation==ACCUMULATION_TWISTED_EDWARDS_XYT) {
    reduceTwistedEdwardsBuckets<Curve, windowBits-1><<<smCount, 256, 0, stream>>>(reduceMemory, bucketMemory);
  }
  return cudaSuccess;
}

template<class Curve, Accumulation accumulation, uint32_t windowBits, uint32_t binBits, uint32_t safety, bool uniformBuckets>
int32_t MSMRunner<Curve, accumulation, windowBits, binBits, safety, uniformBuckets>::
runFinalReduce(cudaStream_t stream, void* finalResult, void* reducedBuckets, uint32_t reducePointCount) {
  if constexpr (accumulation==ACCUMULATION_EXTENDED_JACOBIAN || accumulation==ACCUMULATION_EXTENDED_JACOBIAN_ML) {
    sumExtendedJacobian<Curve><<<1, 256, 3072, stream>>>(finalResult, _uniformBucketPointCount==0xFFFFFFFF ? NULL : _uniformBucketMSM, reducedBuckets, reducePointCount);
  }
  else if constexpr (accumulation==ACCUMULATION_TWISTED_EDWARDS_XY || accumulation==ACCUMULATION_TWISTED_EDWARDS_XYT) {
    printf("ACK!! IMPLEMENT ME!\n");
  }
  return cudaSuccess;
}

template<class Curve, Accumulation accumulation, uint32_t windowBits, uint32_t binBits, uint32_t safety, bool uniformBuckets>
int32_t MSMRunner<Curve, accumulation, windowBits, binBits, safety, uniformBuckets>::
runFinalReduceAndNormalize(cudaStream_t stream, void* finalResult, void* reducedBuckets, uint32_t reducePointCount, uint32_t batchCount) {
  if constexpr (accumulation==ACCUMULATION_EXTENDED_JACOBIAN || accumulation==ACCUMULATION_EXTENDED_JACOBIAN_ML) {
    sumAndNormalizeExtendedJacobian<Curve><<<batchCount, 256, 3072, stream>>>(finalResult, _uniformBucketPointCount==0xFFFFFFFF ? NULL : _uniformBucketMSM, reducedBuckets, reducePointCount);
  }
  else if constexpr (accumulation==ACCUMULATION_TWISTED_EDWARDS_XY || accumulation==ACCUMULATION_TWISTED_EDWARDS_XYT) {
    printf("ACK!! IMPLEMENT ME!\n");
  }
  return cudaSuccess;
}


template<class Curve, Accumulation accumulation, uint32_t windowBits, uint32_t binBits, uint32_t safety, bool uniformBuckets>
int32_t MSMRunner<Curve, accumulation, windowBits, binBits, safety, uniformBuckets>::dumpOperations(const char* filename, void* planningMemory, uint32_t pointCount) {
  const uint32_t BUCKET_COUNT=Planning::BUCKET_COUNT, WINDOWS=Planning::WINDOWS;

  FILE*     f;
  uint32_t* operations;
  int32_t   ec;
  size_t    bytes=_maxPointCount;     
  uint32_t* pointIndexes;
  uint32_t* sortedBucketIndexes;
  uint32_t* sortedBucketCounts;
  uint32_t* sortedBucketOffsets;
  void*     current;

  f=fopen(filename, "w");
  bytes=bytes*WINDOWS*4 + BUCKET_COUNT*12;

  operations=(uint32_t*)malloc(bytes);
  ec=cudaMemcpy(operations, planningMemory, bytes, cudaMemcpyDeviceToHost);
  if(ec!=cudaSuccess) return ec;

  current=(void*)operations;
  pointIndexes=(uint32_t*)advancePointer(current, _maxPointCount*WINDOWS*4);
  sortedBucketIndexes=(uint32_t*)advancePointer(current, BUCKET_COUNT*4); 
  sortedBucketCounts=(uint32_t*)advancePointer(current, BUCKET_COUNT*4); 
  sortedBucketOffsets=(uint32_t*)advancePointer(current, BUCKET_COUNT*4);

  for(int i=0;i<BUCKET_COUNT;i++) {
    if(sortedBucketCounts[i]+sortedBucketOffsets[i]>=pointCount*WINDOWS) {
      fprintf(stderr, "Bad planning phase at index %d - aborting\n", i);
      exit(1);
    }
    for(int j=0;j<sortedBucketCounts[i];j++) 
      fprintf(f, "%07d %08X\n", sortedBucketIndexes[i], pointIndexes[sortedBucketOffsets[i] + j]);
  }

  fclose(f);
  free(operations);
  return cudaSuccess;   
}

