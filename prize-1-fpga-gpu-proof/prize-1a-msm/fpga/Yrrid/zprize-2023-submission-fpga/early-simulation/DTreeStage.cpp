/***

Copyright (c) 2023-2024, Yrrid Software, Inc. All rights reserved.
Dual licensed under the MIT License or the Apache License, Version 2.0.
See LICENSE for details.

Author(s):  Niall Emmart

***/

class DTreeStage {
  typedef enum {
    modeTree,
    modeRInv,
  } ModeEnum;

  public:
  bool             needsWork;
  uint32_t         startCycle;
  uint32_t         batchSlot;
  ModeEnum         mode;
  uint32_t         sourceI;
  uint32_t         sourceOffset;
  uint32_t         destinationI;
  uint32_t         destinationOffset;
  uint32_t         count;

  typedef FF<CURVE::MainField,false> FFF;
  typedef FF<CURVE::MainField,true> FF;

  DTreeStage(Simulator& simulator) {
    needsWork=true;
    batchSlot=0;

    // now we have a full 128-leaf tree
    for(int i=0;i<4096/CYCLES;i++) {
      simulator.dTree[0][42 + i*CYCLES]=FF::one();   // 129                // 126, 127, 128
      simulator.dTree[1][42 + i*CYCLES]=FF::one();
      simulator.dTree[2][42 + i*CYCLES]=FF::one();
      simulator.dTree[1][64 + i*CYCLES]=FF::one();   // 129 + 66 = 195     // 193, 194
      simulator.dTree[2][64 + i*CYCLES]=FF::one();
      simulator.dTree[2][75 + i*CYCLES]=FF::one();   // 195 + 33 = 228     // 227
      simulator.dTree[1][81 + i*CYCLES]=FF::one();   // 228 + 18 = 246     // 244, 245
      simulator.dTree[2][81 + i*CYCLES]=FF::one();
      simulator.dTree[2][84 + i*CYCLES]=FF::one();	 // 246 + 9 = 255      // 254
    }
  }

  void advance(uint32_t& i, uint32_t& slot) {
    slot+=(i==2) ? 1 : 0;
    i=(i==2) ? 0 : i+1;
  }

  void run(Simulator& simulator, InverterStage& inverterStage, uint32_t cycle) {
    CURVE::MainField a,  b, r;

    if(needsWork && simulator.acceptWork(stageDTreeMultiplier)) {
      startCycle=cycle;
      needsWork=false;
    }
    if(needsWork)
      return;

    if(cycle-startCycle==0) {
      // 128 in, 64 out
      mode=modeTree;
      sourceI=0;
      sourceOffset=0;
      destinationI=0;
      destinationOffset=43;
      count=64;
    }
    else if(cycle-startCycle==75) {    // 0 + 64 + 43 - 32
      // 62 in, 31 out
      sourceI=0;
      sourceOffset=43;
      destinationI=0;
      destinationOffset=65;
      count=32;
    }
    else if(cycle-startCycle==134) {    // 75 + 32 + 43 - 16
      // 32 in, 16 out
      sourceI=0;
      sourceOffset=65;
      destinationI=0;
      destinationOffset=76;
      count=16;
    }
    else if(cycle-startCycle==185) {    // 134 + 16 + 43 - 8
      // 16 in, 8 out
      sourceI=0;
      sourceOffset=76;
      destinationI=0;
      destinationOffset=82;
      count=8;
    }
    else if(cycle-startCycle==232) {    // 185 + 8 + 43 - 4
      // 8 in, 4 out
      sourceI=0;
      sourceOffset=82;
      destinationI=0;
      destinationOffset=85;
      count=4;
    }
    else if(cycle-startCycle==275) {    // 232 + 4 + 43 - 4
      mode=modeRInv;
      sourceI=0;
      sourceOffset=85;
      destinationI=0;
      destinationOffset=85;
      count=4;
    }
    else if(cycle-startCycle>=322 && inverterStage.readyForWork()) {   // 275 + 43 + 4
      inverterStage.setInput(0, simulator.dTree[0][85 + batchSlot*CYCLES]);
      inverterStage.setInput(1, simulator.dTree[1][85 + batchSlot*CYCLES]);
      inverterStage.setInput(2, simulator.dTree[2][85 + batchSlot*CYCLES]);
      inverterStage.setInput(3, simulator.dTree[0][86 + batchSlot*CYCLES]);

      simulator.postWork(cycle+2, stageInverter);
      simulator.postWork(cycle+2, stageTreeCopier);
      needsWork=true;
      batchSlot=simulator.nextSlot(batchSlot);
    }

    if(count>0) {
      a=simulator.dTree[sourceI][sourceOffset + batchSlot*CYCLES];
      advance(sourceI, sourceOffset);

      if(mode==modeTree) {
        b=simulator.dTree[sourceI][sourceOffset + batchSlot*CYCLES];
        advance(sourceI, sourceOffset);
      }
      else if(mode==modeRInv)
        b=CURVE::MainField(CURVE::MainField::rInvModP);

      r=FF::mul(a, b);

      simulator.postRowUpdate(cycle+42, UpdateRow::dTree(batchSlot, destinationI, destinationOffset, r));

      advance(destinationI, destinationOffset);
      count--;
    }
  }
};

