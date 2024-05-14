/***

Copyright (c) 2023-2024, Yrrid Software, Inc. All rights reserved.
Dual licensed under the MIT License or the Apache License, Version 2.0.
See LICENSE for details.

Author(s):  Niall Emmart

***/

class DelayStage {
  public:
  bool     needsWork;
  uint32_t startCycle;

  DelayStage() {
    needsWork=true;
  }

  void run(Simulator& simulator, uint32_t cycle) {
    if(needsWork && simulator.acceptWork(stageDelay)) {
      startCycle=cycle;
      needsWork=false;
    }
    if(needsWork)
      return;
    if(cycle-startCycle==109) {
      needsWork=true;
      simulator.postWork(cycle+2, stageDMultiplier);
    }
  }
};
