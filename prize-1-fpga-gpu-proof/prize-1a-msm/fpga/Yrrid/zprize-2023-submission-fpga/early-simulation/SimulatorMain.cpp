/***

Copyright (c) 2023-2024, Yrrid Software, Inc. All rights reserved.
Dual licensed under the MIT License or the Apache License, Version 2.0.
See LICENSE for details.

Author(s):  Niall Emmart

***/

#include <stdio.h>
#include <stdint.h>
#include <map>

#define WINDOW_BITS        14
#define WINDOWS            19
#define CURVE              BLS12381
#define CYCLES             336
#define INVALID_BUCKET     0xFFFF

#include "core/core.h"
#include "HexReader.cpp"
#include "Types.h"
#include "Stage.cpp"
#include "Simulator.h"
#include "BatchCreatorStage.cpp"
#include "DelayStage.cpp"
#include "DMultiplierStage.cpp"
#include "InverterStage.cpp"
#include "DTreeStage.cpp"
#include "TreeCopierStage.cpp"
#include "DUntreeStage.cpp"
#include "ECStage.cpp"

#include "Simulator.cpp"

int main(int argc, const char** argv) {
  Simulator*          simulator;
  int                 msmCount=1024;
  PointXY<CURVE,true> result;

/*
  Inverter inverter(FF<BLS12381::MainField,false>::fromString("22B685ED3CE8D30A5F74073753931CF5932CED301ECACFBA16B429BBED32A7C8E01C2715C7171229E95B50E8E7CEC76"));
  uint32_t steps=0;
  CURVE::MainField inv;

  while(!inverter.done()) {
    steps++;
    inverter.runStep();
  }
  inverter.runStep();
  inverter.reduceFirst();
  while(true) {
    if(inverter.reduceStep())
      break;
  }
  inverter.reduceStep();
  inv=inverter.reduceLast();
  printf("%016llX%016llX%016llX%016llX%016llX%016llX\n", inv.l[5], inv.l[4], inv.l[3], inv.l[2], inv.l[1], inv.l[0]);

  printf("%d %d\n", steps, inverter.shiftCount);
*/

  if(argc>1) {
    msmCount=atoi(argv[1]);
  }

  simulator=new Simulator();
  simulator->run(msmCount);

  result=simulator->reduceBuckets();
  if(result.infinity)
    printf("Result: Inf\n");
  else {
    printf("X = %016llX%016llX%016llX%016llX%016llX%016llX\n", result.x.l[5], result.x.l[4], result.x.l[3], result.x.l[2], result.x.l[1], result.x.l[0]);
    printf("Y = %016llX%016llX%016llX%016llX%016llX%016llX\n", result.y.l[5], result.y.l[4], result.y.l[3], result.y.l[2], result.y.l[1], result.y.l[0]);
  }
}

