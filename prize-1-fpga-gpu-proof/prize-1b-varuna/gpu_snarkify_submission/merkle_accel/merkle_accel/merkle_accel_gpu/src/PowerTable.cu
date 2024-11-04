/***

Copyright (c) 2024, Snarkify, Inc. All rights reserved.
Dual licensed under the MIT License or the Apache License, Version 2.0.
See LICENSE for details.

Author(s):  Niall Emmart

***/

__device__ __forceinline__ bls12381_fr power(bls12381_fr* table, uint32_t index, uint32_t count) {
  return table[index & 0xFFF] * table[(index>>12) + 4096]; 
}

template<bool inv>
__global__ void power_table_kernel(bls12381_fr* table, bls12381_fr* value) {
  uint32_t         globalTID=blockIdx.x*blockDim.x + threadIdx.x;
  uint32_t         power;
  bls12381_fr current=bls12381_fr::r();
  bls12381_fr square;

  if(globalTID>=49*256)
    return;

  if(globalTID<4096)
    power=globalTID;
  else
    power=(globalTID-4096)*4096;

  if constexpr(!inv)
    square=value[0];
  else
    square=value[0].inv();

  while(power!=0) {
    if((power & 0x01)==1)
      current=current * square;
    square=square * square;
    power=power>>1;
  }

  table[globalTID]=current;
}

