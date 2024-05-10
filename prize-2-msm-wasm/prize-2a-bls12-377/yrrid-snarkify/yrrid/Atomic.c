/***

Copyright (c) 2023-2024, Yrrid Software, Inc. and Snarkify Inc.  All rights reserved.
Dual licensed under the MIT License or the Apache License, Version 2.0.
See LICENSE for details.

Author(s):  Niall Emmart

***/

void thread_fence() {
  __atomic_thread_fence(__ATOMIC_SEQ_CST);
}

uint32_t atomic_add(uint32_t* address, uint32_t amount) {
  return __atomic_fetch_add(address, amount, __ATOMIC_ACQ_REL);
}

uint32_t atomic_or(uint32_t* address, uint32_t amount) {
  return __atomic_fetch_or(address, amount, __ATOMIC_ACQ_REL);
}

uint32_t atomic_and(uint32_t* address, uint32_t amount) {
  return __atomic_fetch_and(address, amount, __ATOMIC_ACQ_REL);
}

bool atomic_lock_bucket(I128* packedPoint) {
  uint32_t* address=(uint32_t*)packedPoint;
  uint32_t  prior;

  prior=__atomic_fetch_and(&address[23], 0x7FFFFFFF, __ATOMIC_ACQ_REL);
  return (prior & 0x80000000)!=0;
}

void atomic_unlock_bucket(I128* packedPoint) {
  uint32_t* address=(uint32_t*)packedPoint;
  uint32_t  prior;

  prior=__atomic_fetch_or(&address[23], 0x80000000, __ATOMIC_ACQ_REL);
}

