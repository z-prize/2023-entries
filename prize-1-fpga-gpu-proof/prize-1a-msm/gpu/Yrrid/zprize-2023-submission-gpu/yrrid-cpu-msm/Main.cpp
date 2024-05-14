/***

Copyright (c) 2023-2024, Yrrid Software, Inc. All rights reserved.
Dual licensed under the MIT License or the Apache License, Version 2.0.
See LICENSE for details.

Author(s):  Niall Emmart

***/

#include <stdio.h>
#include <stdint.h>
#include <stdlib.h>
#include "HostFields.cpp"
#include "HostCurves.cpp"

extern "C" {

typedef struct {
  uint32_t curve;
  uint32_t maxBatchCount;
  uint32_t maxPointCount;
  void*    pointData;
} Context;

void* createContext(uint32_t curve, uint32_t maxBatchCount, uint32_t maxPointCount) {
  Context* context;

  context=(Context*)malloc(sizeof(Context));
  context->curve=curve;
  context->maxBatchCount=maxBatchCount;
  context->maxPointCount=maxPointCount;

  context->pointData=NULL;

  return (void*)context;
}

int32_t destroyContext(void* contextPtr) {
  Context* context=(Context*)contextPtr;

  if(context->pointData!=NULL)
    free(context->pointData);
  free(context);
  return 0;
}

int32_t preprocessPoints(void* contextPtr, void* pointData, uint32_t pointCount) {
  Context*  context=(Context*)contextPtr;
  uint64_t* from;
  uint64_t* to;

  if(context->pointData!=NULL)
    free(context->pointData);

  context->pointData=malloc(pointCount*96);

  // note -- 104 bytes per point, we ignore the top 8 bytes per point
  from=(uint64_t*)pointData;
  to=(uint64_t*)context->pointData;
  for(uint32_t i=0;i<pointCount;i++) {
    for(int j=0;j<12;j++)
      to[i*12+j]=from[i*13+j];
  }

  return 0;
}

int32_t processBatches(void* contextPtr, void* resultData, void** scalarData, uint32_t batchCount, uint32_t pointCount) {
  typedef Host::Curve<Host::BLS12377G1> C377;
  typedef Host::Curve<Host::BLS12381G1> C381;

  Context*  context=(Context*)contextPtr;
  uint64_t* results=(uint64_t*)resultData;
  uint64_t* points=(uint64_t*)context->pointData;

  for(uint32_t batch=0;batch<batchCount;batch++) {
    uint8_t*  scalars=(uint8_t*)scalarData[batch];
    if(context->curve==377) {
      C377::PointXYZZ acc;
      C377::PointXY   pt;

      for(uint32_t i=0;i<pointCount;i++) {
        pt.load(&points[i*12]);
        pt.fromDeviceMontgomeryToHostMontgomery();
        acc.addXYZZ(pt.scale(&scalars[i*32], 32));
      }
      pt=acc.normalize();
      pt.dump();
      fflush(stdout);
      pt.fromHostMontgomeryToDeviceMontgomery();
      pt.store(&results[batch*13]);                     // Note batch*13 for storing back to host
      results[batch*13 + 12]=0;                         // zero out 13th word
    }
    else if(context->curve==381) {
      C381::PointXYZZ acc;
      C381::PointXY   pt;

      for(uint32_t i=0;i<pointCount;i++) {
        pt.load(&points[i*12]);
        pt.fromDeviceMontgomeryToHostMontgomery();
        acc.addXYZZ(pt.scale(&scalars[i*32], 32));
      }
      pt=acc.normalize();
      pt.dump();
      fflush(stdout);
      pt.fromHostMontgomeryToDeviceMontgomery();
      pt.store(&results[batch*13]);                     // Note batch*13 for storing back to host
      results[batch*13 + 12]=0;                         // zero out 13th word
    }
    else
      return -1;
  }
  return 0;
}

}