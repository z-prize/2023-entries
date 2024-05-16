/***

Copyright (c) 2023-2024, Yrrid Software, Inc. and Snarkify Inc.  All rights reserved.
Dual licensed under the MIT License or the Apache License, Version 2.0.
See LICENSE for details.

Author(s):  Niall Emmart

***/

#include <stdio.h>
#include <stdint.h>
#include <stdbool.h>
#include "Types.h"

#include "asm.cu"
#include "Chain.cu"
#include "MP.cu"

#include "BLS12377.cuh"
#include "BLS12377.cu"
#include "CurveParameters.cu"

#include "PlanningKernels.cu"

#include "MSM.cu"

#define TOP_377              0x12ab655e
#define TOP_381              0x73eda753

void zeroWords(uint32_t* data, uint32_t count) {
  for(int i=0;i<count;i++)
    data[i]=0;
}

uint32_t randomWord() {
  uint32_t r=rand() & 0xFFFF;

  return (r<<16) | (rand() & 0xFFFF);
}

void randomWords(uint32_t* data, uint32_t count) {
  for(int i=0;i<count;i++)
    data[i]=randomWord();
}

#define N (1<<24)

__managed__ uint4    scalars[N*2];

typedef MSM<BLS12377G1, REPRESENTATION_TWISTED_EDWARDS, 23, 6, 2> MSMBLS12377G1;

int main() {
  MSMBLS12377G1 msm=MSMBLS12377G1(N);
  uint64_t      bytes=msm.planningBytesRequired();
  void*         planningMemory;
  cudaStream_t  stream=(cudaStream_t)0;
  uint32_t*     scalarWords, *cpuOffsets, *cpuCounts, *cpuPointIndexes, *cpuBucketCounts, *cpuBucketOffsets;
  cudaEvent_t   start, stop;
  float         time;
  uint32_t      begin, special[100];

  begin=randomWord() & 0xFFFF;
  for(int i=0;i<50;i++)
    special[i]=begin+i;
  for(int i=50;i<100;i++)
    special[i]=randomWord() & 0x003FFFFF;

  scalarWords=(uint32_t*)scalars;
  randomWords(scalarWords, 8*N);
  for(int i=0;i<N;i++) {
    if(randomWord() % 100 == 0) 
      scalarWords[i*8]=(scalarWords[i*8] & 0xFFC00000) + special[randomWord() % 100];
    scalarWords[i*8+7]=scalarWords[i*8+7] % TOP_377;
  }

  cudaMalloc(&planningMemory, bytes);

  printf("start\n");

  cudaEventCreate(&start);
  cudaEventCreate(&stop);
  for(int i=0;i<20;i++) {
    cudaEventRecord(start, 0);
    msm.runPlanning(stream, planningMemory, scalars, N);
    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&time, start, stop);
    printf("%0.3f MS\n", time);
  }

  printf("sync: %d\n", cudaDeviceSynchronize()); 

  cpuCounts=(uint32_t*)malloc(65536*8*4);
  cpuOffsets=(uint32_t*)malloc(65536*4);

  cpuPointIndexes=(uint32_t*)malloc(4*11<<24);
  cpuBucketCounts=(uint32_t*)malloc(4<<22);
  cpuBucketOffsets=(uint32_t*)malloc(4<<22);

//  cudaMemcpy(cpuCounts, _binCounts, 65536*32, cudaMemcpyDeviceToHost);
//  cudaMemcpy(cpuOffsets, _binOffsets, 65536*4, cudaMemcpyDeviceToHost);

  cudaMemcpy(cpuPointIndexes, _pointIndexes, 4*11<<24, cudaMemcpyDeviceToHost);
  cudaMemcpy(cpuBucketCounts, _bucketCounts, 4<<22, cudaMemcpyDeviceToHost);
  cudaMemcpy(cpuBucketOffsets, _bucketOffsets, 4<<22, cudaMemcpyDeviceToHost);

printf("HERE! %d\n", cudaGetLastError());

  uint32_t next=0;
  for(int i=0;i<(1<<22);i++) {
    if(next!=cpuBucketOffsets[i]) printf("YIKES %d\n", i);
    for(int j=0;j<cpuBucketCounts[i];j++) {
      printf("%08d %08X\n", i, cpuPointIndexes[next]);
      next++;
    }      
  }


/*
  for(int i=0;i<20;i++)
    printf("... %d %d\n", cpuCounts[i], cpuOffsets[i]);

  uint32_t total=0;
  for(int i=0;i<20;i++) {
    total=0;
    for(int j=0;j<64;j++) 
      total+=cpuBucketCounts[i*64+j];
    printf("Total: %d\n", total);
  }
*/

/*
  for(int i=0;i<65536;i++) {
    if(cpuOffsets[i]!=total) printf("Mismatch at %d   %d %d\n", i, total, cpuOffsets[i]);
    for(int j=0;j<8;j++)
      total+=cpuCounts[i+j*65536];
  }
*/
}
