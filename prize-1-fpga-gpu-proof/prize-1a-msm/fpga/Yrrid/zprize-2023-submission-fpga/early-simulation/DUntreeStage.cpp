/***

Copyright (c) 2023-2024, Yrrid Software, Inc. All rights reserved.
Dual licensed under the MIT License or the Apache License, Version 2.0.
See LICENSE for details.

Author(s):  Niall Emmart

***/

class DUntreeStage {
  public:
  bool             needsWork;
  uint32_t         startCycle;
  uint32_t         batchSlot;
  uint32_t         sourceI;
  uint32_t         sourceOffset;
  uint32_t         destinationI;
  uint32_t         destinationOffset;
  uint32_t         count;
  CURVE::MainField root, a, b;

  typedef FF<CURVE::MainField,true> FF;

  DUntreeStage() {
    needsWork=true;
    batchSlot=0;
  }

  void advance(uint32_t& i, uint32_t& slot) {
    slot+=(i==2) ? 1 : 0;
    i=(i==2) ? 0 : i+1;
  }

  void run(Simulator& simulator, InverterStage& inverterStage, uint32_t cycle) {
    uint32_t i, offset;

    if(needsWork && simulator.acceptWork(stageDUntreeMultiplier)) {
      startCycle=cycle;
      needsWork=false;
    }
    if(needsWork)
      return;

    if(cycle-startCycle==0) {
      simulator.dUntree[0][85 + batchSlot*CYCLES]=inverterStage.getOutput(0);
      simulator.dUntree[1][85 + batchSlot*CYCLES]=inverterStage.getOutput(1);
      simulator.dUntree[2][85 + batchSlot*CYCLES]=inverterStage.getOutput(2);
      simulator.dUntree[0][86 + batchSlot*CYCLES]=inverterStage.getOutput(3);

      // 4+8 in, 8 out
      sourceI=0;
      sourceOffset=85;
      destinationI=0;
      destinationOffset=82;
      count=8;
    }
    else if(cycle-startCycle==43) {     // 0 + 43
      // 8+16 in, 16 out
      sourceI=0;
      sourceOffset=82;
      destinationI=0;
      destinationOffset=76;
      count=16;
    }
    else if(cycle-startCycle==86) {    // 43 + 43
      // 16+32 in, 32 out
      sourceI=0;
      sourceOffset=76;
      destinationI=0;
      destinationOffset=65;
      count=32;
    }
    else if(cycle-startCycle==129) {    // 86 + 43
      // 32+64 in, 64 out
      sourceI=0;
      sourceOffset=65;
      destinationI=0;
      destinationOffset=43;
      count=64;
    }
    else if(cycle-startCycle==193) {    // 129 + 64
      // 64+128 in, 128 out
      sourceI=0;
      sourceOffset=43;
      destinationI=0;
      destinationOffset=0;
      count=128;
    }
    else if(cycle-startCycle==321) {
      simulator.postWork(cycle+43, stageEC);
      needsWork=true;
      batchSlot=simulator.nextSlot(batchSlot);
    }

    if(count>0) {
      if((count & 0x01)==0) {
        i=destinationI;
        offset=destinationOffset;

        root=simulator.dUntree[sourceI][sourceOffset + batchSlot*CYCLES];
        advance(sourceI, sourceOffset);
        a=simulator.dUntree[i][offset + batchSlot*CYCLES];
        advance(i, offset);
        b=simulator.dUntree[i][offset + batchSlot*CYCLES];

        simulator.postRowUpdate(cycle+42, UpdateRow::dUntree(batchSlot, destinationI, destinationOffset, FF::mul(root, b)));
        advance(destinationI, destinationOffset);
        count--;
      }
      else {
        simulator.postRowUpdate(cycle+42, UpdateRow::dUntree(batchSlot, destinationI, destinationOffset, FF::mul(root, a)));
        advance(destinationI, destinationOffset);
        count--;
      }
    }
  }
};
