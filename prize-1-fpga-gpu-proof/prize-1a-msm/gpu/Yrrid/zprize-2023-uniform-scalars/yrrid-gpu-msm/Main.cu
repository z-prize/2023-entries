/***

Copyright (c) 2023-2024, Yrrid Software, Inc. and Snarkify Inc.  All rights reserved.
Dual licensed under the MIT License or the Apache License, Version 2.0.
See LICENSE for details.

Author(s):  Niall Emmart

***/

#include "MSMBatch.cu"

#define SIZE (1<<24)
#define BATCHES 4

uint32_t randomWord() {
  uint32_t r=rand() & 0xFFFF;

  return (r<<16) | (rand() & 0xFFFF);
}

void randomWords(uint32_t* data, uint32_t count) {
  for(int i=0;i<count;i++)
    data[i]=randomWord();
}

void dumpScalarsToFile(const char* name, uint32_t* scalars, uint32_t scalarCount) {
  FILE *file=fopen(name, "w");

  for(int i=0;i<scalarCount;i++) {
    for(int j=7;j>=0;j--)
      fprintf(file, "%08X", scalars[i*8+j]);
    fprintf(file, "\n");
  }
  fclose(file);
}

void* createRandomPoints(void* context, uint32_t* secret) {
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

void dumpHex256(const char* filename, uint32_t* values, uint32_t count) {
  FILE* f=fopen(filename, "w");

  for(int i=0;i<count;i++) {
    for(int j=7;j>=0;j--)
      fprintf(f, "%08X", values[i*8+j]);
    fprintf(f, "\n");
  }
  fclose(f);
}

#include <sys/time.h>
uint64_t clockMS() {
  struct timeval te;

  gettimeofday(&te, NULL); // get current time
  uint64_t milliseconds = ((uint64_t)te.tv_sec)*1000L + ((uint64_t)te.tv_usec)/1000L; // calculate milliseconds
  return milliseconds;
}

int main(int argc, char** argv) {
  void*     context;
  uint32_t* secret;
  uint32_t* scalars;
  uint32_t* scalarSet[BATCHES];
  void*     points;
  uint32_t  curve=377;
  uint32_t  results[12*2*BATCHES];
  uint32_t  top;

  if(argc>=2)
    curve=atoi(argv[1]);

  context=createContext(curve, BATCHES, SIZE);

  secret=(uint32_t*)malloc(SIZE*32ul);
  scalars=(uint32_t*)malloc(SIZE*32ul*BATCHES);
  for(int i=0;i<BATCHES;i++) 
    scalarSet[i]=scalars+SIZE*8*i;

  printf("Creating random scalars\n");
  randomWords(scalars, SIZE*8*BATCHES);

  top=(curve==377) ? 0x12AB655E : 0x73EDA753;
  for(int i=0;i<SIZE*BATCHES;i++) 
    scalars[i*8+7]=scalars[i*8+7] % top;

  printf("Creating random points\n");
  points=createRandomPoints(context, secret);

  dumpHex256("scalars.hex", scalars, SIZE*4);
  dumpHex256("secrets.hex", secret, SIZE);

  preprocessPoints(context, points, SIZE);

for(int i=0;i<10;i++) {
uint64_t timer=-clockMS();
  processBatches(context, results, (void**)scalarSet, BATCHES, SIZE);
timer+=clockMS();
printf("Run time: %ld ms\n", timer);
}

  for(int i=0;i<BATCHES*2;i++) {
    for(int j=11;j>=0;j--)
      printf("%08X", results[i*12+j]);
    printf("\n");
  }
  printf("Done!\n");
  destroyContext(context);
}

