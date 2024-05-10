/***

Copyright (c) 2023-2024, Yrrid Software, Inc. and Snarkify Inc.  All rights reserved.
Dual licensed under the MIT License or the Apache License, Version 2.0.
See LICENSE for details.

Author(s):  Niall Emmart
            Antony Suresh
            Sougata Bhattacharya

***/

#if defined(WASM)

void barrier(uint32_t barrier, uint32_t workerID) {
  uint32_t prior, workers;
  bool     done=false;

  while(true) {
    thread_fence();
    prior=atomic_or(&sharedBarriers[barrier].lock, 0x01);
    if(prior==0) {
      if(sharedBarriers[barrier].workerCount==0) {
        sharedBarriers[barrier].workerCount=sharedWorkerIndex;
        sharedBarriers[barrier].checkedIn=1;
        done=sharedBarriers[barrier].checkedIn>=sharedBarriers[barrier].workerCount;
        break;
      }
      else {
        if(workerID<sharedBarriers[barrier].workerCount)
          sharedBarriers[barrier].checkedIn++;
        done=sharedBarriers[barrier].checkedIn>=sharedBarriers[barrier].workerCount;
        break;
      }
    }
  }
  thread_fence();
  atomic_and(&sharedBarriers[barrier].lock, 0xFFFFFFFE);

  if(done) {
    __builtin_wasm_memory_atomic_notify(&sharedNotification, 0xFFFF);                // wake all threads
  }
  else {
    while(true) {
      __builtin_wasm_memory_atomic_wait32(&sharedNotification, 0, 500000);           // 500 usec
      thread_fence();
      if(sharedBarriers[barrier].checkedIn>=sharedBarriers[barrier].workerCount)
        break;
    }
  }
}

#endif

#if defined(X86)

#include <stdio.h>
#include <unistd.h>

void barrier(uint32_t barrier, uint32_t workerID) {
  uint32_t prior, workers;

  while(true) {
    thread_fence();
    prior=atomic_or(&sharedBarriers[barrier].lock, 0x01);
    if(prior==0) {
      if(sharedBarriers[barrier].workerCount==0) {
        sharedBarriers[barrier].workerCount=sharedWorkerIndex;
        sharedBarriers[barrier].checkedIn=1;
        break;
      }
      else {
        if(workerID<sharedBarriers[barrier].workerCount)
          sharedBarriers[barrier].checkedIn++;
        break;
      }
    }
  }
  thread_fence();
  atomic_and(&sharedBarriers[barrier].lock, 0xFFFFFFFE);
  while(true) {
    thread_fence();
    if(sharedBarriers[barrier].checkedIn>=sharedBarriers[barrier].workerCount)
      break;
    usleep(500);
  }
}

#endif
