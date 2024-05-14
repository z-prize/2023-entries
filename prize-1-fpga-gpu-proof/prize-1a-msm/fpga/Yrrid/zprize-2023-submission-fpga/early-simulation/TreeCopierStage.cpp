/***

Copyright (c) 2023-2024, Yrrid Software, Inc. All rights reserved.
Dual licensed under the MIT License or the Apache License, Version 2.0.
See LICENSE for details.

Author(s):  Niall Emmart

***/

class TreeCopierStage {
  public:
  bool     needsWork;
  uint32_t startCycle;
  uint32_t batchSlot;

  TreeCopierStage() {
    needsWork=true;
    batchSlot=0;
  }

  void run(Simulator& simulator, uint32_t cycle) {
    UpdateEntry entries[3];
    uint32_t    location, batchLocation;

    if(needsWork && simulator.acceptWork(stageTreeCopier)) {
      startCycle=cycle;
      needsWork=false;
    }
    if(needsWork)
      return;

    location=cycle - startCycle;
    batchLocation=batchSlot*CYCLES + location;
    if(cycle-startCycle<85) {
      for(int i=0;i<3;i++)
        entries[i]=UpdateEntry::treeCopy(simulator.dTree[i][batchLocation]);
      simulator.postRowUpdate(cycle+16, UpdateRow::treeCopy(batchSlot, location, entries[0], entries[1], entries[2]));
    }

    if(cycle-startCycle==85) {
      // note, we don't actually kick anyone here!!
      needsWork=true;
      batchSlot=simulator.nextSlot(batchSlot);
    }
  }
};
