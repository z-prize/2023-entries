/***

Copyright (c) 2024, Snarkify, Inc. All rights reserved.
Dual licensed under the MIT License or the Apache License, Version 2.0.
See LICENSE for details.

Author(s):  Niall Emmart

***/

/* We assume a geometry of 256 threads per block for this implementation */

template<uint32_t n_bits, uint32_t t_bits>
class NTT {
  public:
  bls12381_fr rf[8];

  // This class sucks!  Why does it suck?  It sucks because each NTT_7, NTT_8, NTT_9
  // all require tons of custom transposes. 
  // Question:  instead of placing small NTTs next to each other, i.e., for size 4,
  // threads 0+1 handle NTT0, thread 2+3 handle NTT1, etc, what if we stride them?
  // i.e., threads 0 + 128 handle NTT0, threads 1 + 129 handle NTT1, etc.
  // Intuition tells me that we can reuse the same mappings.  Needs testing!

  __device__ __forceinline__ void sync256() {
    __syncthreads();
  }

  __device__ __forceinline__ void load_stride_1(uint4* shared, bls12381_fr* in, uint32_t base, uint32_t advance) {
    uint32_t n, t, lsb=threadIdx.x & 0x01;
    uint4*   in4_ptr=(uint4*)in;

    for(int32_t vtid=threadIdx.x;vtid<512;vtid+=256) {
      n=vtid>>t_bits+1;
      t=vtid & (1<<t_bits+1)-1;
      #pragma unroll
      for(int32_t k=0;k<4;k++)
        shared[k*513 + (vtid>>1) + lsb*257] = in4_ptr[(base + n*advance + (k<<t_bits))*2 + t];
    }

    sync256();

    #pragma unroll
    for(int32_t k=0;k<4;k++) 
      rf[k] = bls12381_fr(shared[k*513 + threadIdx.x], shared[k*513 + threadIdx.x + 257]);

    sync256();

    for(int32_t vtid=threadIdx.x;vtid<512;vtid+=256) {
      n=vtid>>t_bits+1;
      t=vtid & (1<<t_bits+1)-1;
      #pragma unroll
      for(int32_t k=0;k<4;k++)
        shared[k*513 + (vtid>>1) + lsb*257] = in4_ptr[(base + n*advance + (k+4<<t_bits))*2 + t];
    }

    sync256();

    #pragma unroll
    for(int32_t k=0;k<4;k++) 
      rf[k+4] = bls12381_fr(shared[k*513 + threadIdx.x], shared[k*513 + threadIdx.x + 257]);

    sync256();
  }

  __device__ __forceinline__ void store_stride_1(uint4* shared, bls12381_fr* out, uint32_t base, uint32_t advance) {
    uint32_t n, t, lsb=threadIdx.x & 0x01;
    uint4*   out4_ptr=(uint4*)out;

    #pragma unroll
    for(int32_t k=0;k<4;k++) {
      shared[k*513 + threadIdx.x] = rf[k].get_low();
      shared[k*513 + threadIdx.x + 257] = rf[k].get_high();
    }

    sync256();

    for(int32_t vtid=threadIdx.x;vtid<512;vtid+=256) {
      n=vtid>>t_bits+1;
      t=vtid & (1<<t_bits+1)-1;
      #pragma unroll
      for(int32_t k=0;k<4;k++)
        out4_ptr[(base + n*advance + (k<<t_bits))*2 + t]=shared[k*513 + (vtid>>1) + lsb*257];
    }

    sync256();

    #pragma unroll
    for(int32_t k=0;k<4;k++) {
      shared[k*513 + threadIdx.x] = rf[k+4].get_low();
      shared[k*513 + threadIdx.x + 257] = rf[k+4].get_high();
    }

    sync256();

    for(int32_t vtid=threadIdx.x;vtid<512;vtid+=256) {
      n=vtid>>t_bits+1;
      t=vtid & (1<<t_bits+1)-1;
      #pragma unroll
      for(int32_t k=0;k<4;k++)
        out4_ptr[(base + n*advance + (k+4<<t_bits))*2 + t]=shared[k*513 + (vtid>>1) + lsb*257];
    }

    sync256();
  }

  __device__ __forceinline__ void load_advance_1(uint4* shared, bls12381_fr* in, uint32_t base, uint32_t stride) {
    uint32_t     n, t, size=(1024>>n_bits)+1;
    uint4*       in4_ptr=(uint4*)in;
 
    // load from:  base + n + t*stride

    for(int32_t vtid=threadIdx.x;vtid<512;vtid+=256) {
      t=vtid>>n_bits+1;
      n=vtid & (1<<n_bits+1)-1;
      #pragma unroll
      for(int32_t k=0;k<4;k++)
        shared[n*size + (k<<t_bits) + t] = in4_ptr[base*2 + ((k<<t_bits) + t)*stride*2 + n];
    }

    sync256();

    #pragma unroll
    for(int32_t k=0;k<4;k++) {
      t=threadIdx.x & (1<<t_bits)-1;
      n=threadIdx.x>>t_bits;
      rf[k]=bls12381_fr(shared[n*2*size + (k<<t_bits) + t], shared[n*2*size + size + (k<<t_bits) + t]);
    }

    sync256();

    for(int32_t vtid=threadIdx.x;vtid<512;vtid+=256) {
      t=vtid>>n_bits+1;
      n=vtid & (1<<n_bits+1)-1;
      #pragma unroll
      for(int32_t k=0;k<4;k++)
        shared[n*size + (k<<t_bits) + t] = in4_ptr[base*2 + ((k+4<<t_bits) + t)*stride*2 + n];
    }

    sync256();

    #pragma unroll
    for(int32_t k=0;k<4;k++) {
      t=threadIdx.x & (1<<t_bits)-1;
      n=threadIdx.x>>t_bits;
      rf[k+4]=bls12381_fr(shared[n*2*size + (k<<t_bits) + t], shared[n*2*size + size + (k<<t_bits) + t]);
    }

    sync256();
  }

  __device__ __forceinline__ void store_advance_1(uint4* shared, bls12381_fr* out, uint32_t base, uint32_t stride) {
    uint32_t     n, t, size=(1024>>n_bits)+1;
    uint4*       out4_ptr=(uint4*)out;
 
    #pragma unroll
    for(int32_t k=0;k<4;k++) {
      t=threadIdx.x & (1<<t_bits)-1;
      n=threadIdx.x>>t_bits;
      shared[n*2*size + (k<<t_bits) + t]=rf[k].get_low();
      shared[n*2*size + size + (k<<t_bits) + t]=rf[k].get_high();
    }

    sync256();

    for(int32_t vtid=threadIdx.x;vtid<512;vtid+=256) {
      t=vtid>>n_bits+1;
      n=vtid & (1<<n_bits+1)-1;
      #pragma unroll
      for(int32_t k=0;k<4;k++)
        out4_ptr[base*2 + ((k<<t_bits) + t)*stride*2 + n] = shared[n*size + (k<<t_bits) + t];
    }

    sync256();

    #pragma unroll
    for(int32_t k=0;k<4;k++) {
      t=threadIdx.x & (1<<t_bits)-1;
      n=threadIdx.x>>t_bits;
      shared[n*2*size + (k<<t_bits) + t]=rf[k+4].get_low();
      shared[n*2*size + size + (k<<t_bits) + t]=rf[k+4].get_high();
    }

    sync256();

    for(int32_t vtid=threadIdx.x;vtid<512;vtid+=256) {
      t=vtid>>n_bits+1;
      n=vtid & (1<<n_bits+1)-1;
      #pragma unroll
      for(int32_t k=0;k<4;k++)
        out4_ptr[base*2 + ((k+4<<t_bits) + t)*stride*2 + n] = shared[n*size + (k<<t_bits) + t];
    }

    sync256();
  }

  __device__ __forceinline__ void transpose_7_1(uint4* shared) {
    uint32_t n=threadIdx.x>>4;
    uint32_t t=threadIdx.x & 0x0F;

    #pragma unroll
    for(int32_t k=0;k<8;k++)
      shared[k + t*8 + n*128]=rf[k].get_low();    // bad bank conflicts

    sync256();

    #pragma unroll
    for(int32_t k=0;k<8;k++)
      rf[k].set_low(shared[t + k*16 + n*128]);

    sync256();
    
    #pragma unroll
    for(int32_t k=0;k<8;k++)
      shared[k + t*8 + n*128]=rf[k].get_high();    // bad bank conflicts

    sync256();

    #pragma unroll
    for(int32_t k=0;k<8;k++)
      rf[k].set_high(shared[t + k*16 + n*128]);

    sync256();
  }

  __device__ __forceinline__ void transpose_7_2(uint4* shared) {    
    uint32_t i=threadIdx.x & 0x07;
    uint32_t j=(threadIdx.x>>3) & 0x01;
    uint32_t n=threadIdx.x>>4;
 
    #pragma unroll
    for(int32_t k=0;k<8;k++)
      shared[i + k*8 + j*64 + n*128]=rf[k].get_low();

    sync256();

    #pragma unroll
    for(int32_t k=0;k<8;k++)
      rf[k].set_low(shared[i + j*8 + k*16 + n*128]);

    sync256();

    #pragma unroll
    for(int32_t k=0;k<8;k++)
      shared[i + k*8 + j*64 + n*128]=rf[k].get_high();

    sync256();

    #pragma unroll
    for(int32_t k=0;k<8;k++)
      rf[k].set_high(shared[i + j*8 + k*16 + n*128]);

    sync256();
  }

  __device__ __forceinline__ void transpose_8_1(uint4* shared) {
    uint32_t n=threadIdx.x>>5;
    uint32_t t=threadIdx.x & 0x1F;

    #pragma unroll
    for(int32_t k=0;k<8;k++)
      shared[k + t*8 + n*256]=rf[k].get_low();    // bad bank conflicts

    sync256();

    #pragma unroll
    for(int32_t k=0;k<8;k++)
      rf[k].set_low(shared[t + k*32 + n*256]);

    sync256();
    
    #pragma unroll
    for(int32_t k=0;k<8;k++)
      shared[k + t*8 + n*256]=rf[k].get_high();    // bad bank conflicts

    sync256();

    #pragma unroll
    for(int32_t k=0;k<8;k++)
      rf[k].set_high(shared[t + k*32 + n*256]);

    sync256();
  }

  __device__ __forceinline__ void transpose_8_2(uint4* shared) {
    uint32_t i=threadIdx.x & 0x07;
    uint32_t j=(threadIdx.x>>3) & 0x03;
    uint32_t n=threadIdx.x>>5;
 
    #pragma unroll
    for(int32_t k=0;k<8;k++)
      shared[i + k*8 + j*64 + n*256]=rf[k].get_low();

    sync256();

    #pragma unroll
    for(int32_t k=0;k<8;k++)
      rf[k].set_low(shared[i + j*8 + k*32 + n*256]);

    sync256();

    #pragma unroll
    for(int32_t k=0;k<8;k++)
      shared[i + k*8 + j*64 + n*256]=rf[k].get_high();

    sync256();

    #pragma unroll
    for(int32_t k=0;k<8;k++)
      rf[k].set_high(shared[i + j*8 + k*32 + n*256]);

    sync256();
  }

  __device__ __forceinline__ void transpose_9_1(uint4* shared) {
    uint32_t i=threadIdx.x & 0x07;
    uint32_t j=(threadIdx.x>>3) & 0x07;
    uint32_t n=threadIdx.x>>6;

    #pragma unroll
    for(int32_t k=0;k<8;k++)
      shared[threadIdx.x + k*257]=rf[k].get_low();

    sync256();

    #pragma unroll
    for(int32_t k=0;k<8;k++)
      rf[k].set_low(shared[j + k*8 + n*64 + i*257]);

    sync256();

    #pragma unroll
    for(int32_t k=0;k<8;k++)
      shared[threadIdx.x + k*257]=rf[k].get_high();

    sync256();

    #pragma unroll
    for(int32_t k=0;k<8;k++)
      rf[k].set_high(shared[j + k*8 + n*64 + i*257]);

    sync256();
  }

  __device__ __forceinline__ void transpose_9_2(uint4* shared) {
    uint32_t i=threadIdx.x & 0x07;
    uint32_t j=(threadIdx.x>>3) & 0x07;
    uint32_t n=threadIdx.x>>6;

    #pragma unroll
    for(int32_t k=0;k<8;k++)
      shared[i + k*8 + n*64 + j*256]=rf[k].get_low();

    sync256();

    #pragma unroll
    for(int32_t k=0;k<8;k++)
      rf[k].set_low(shared[i + j*8 + n*64 + k*256]);

    sync256();

    #pragma unroll
    for(int32_t k=0;k<8;k++)
      shared[i + k*8 + n*64 + j*256]=rf[k].get_high();

    sync256();

    #pragma unroll
    for(int32_t k=0;k<8;k++)
      rf[k].set_high(shared[i + j*8 + n*64 + k*256]);

    sync256();
  }

  __device__ __forceinline__ void twiddle_6_3x3(bls12381_fr* roots) {
    uint32_t group=threadIdx.x & 0x07;

    #pragma unroll
    for(int32_t k=1;k<8;k++)
      rf[k]=rf[k] * roots[ROOTS_6_3x3 + k*8 + group];
  }

  __device__ __forceinline__ void twiddle_7_1x6(bls12381_fr* roots) {
    uint32_t     group=threadIdx.x & 0x0F;

    #pragma unroll
    for(int32_t k=4;k<8;k++)
      rf[k]=rf[k] * roots[ROOTS_7 + (k & 0x03)*16 + group];
  }

  __device__ __forceinline__ void twiddle_8_2x6(bls12381_fr* roots) {
    uint32_t     group=threadIdx.x & 0x1F;

    #pragma unroll
    for(int32_t k=2;k<8;k++) 
      rf[k]=rf[k] * roots[ROOTS_8_2x6 + k*32 + group];

    __syncthreads();          // force separation of the two loops
  }

  __device__ __forceinline__ void twiddle_9_3x6(bls12381_fr* roots) {
    uint32_t     group=threadIdx.x & 0x3F;

    #pragma unroll
    for(int32_t k=1;k<8;k++)
      rf[k]=rf[k] * roots[ROOTS_9_3x6 + k*64 + group];

    __syncthreads();          // force separation of the two loops
  }

  __device__ __forceinline__ void twiddle_14_7x7(bls12381_fr* roots, uint32_t base, uint32_t advance) {
    uint32_t n=threadIdx.x>>t_bits;
    uint32_t t=threadIdx.x & (1<<t_bits)-1;

    #pragma unroll
    for(int32_t k=0;k<8;k++) 
      rf[k]=rf[k] * roots[ROOTS_14_7x7 + (base + n*advance)*128 + (k<<t_bits) + t];

    __syncthreads();          // force separation of the two loops
  }

  __device__ __forceinline__ void twiddle_16_7x9(bls12381_fr* roots, uint32_t base, uint32_t advance) {
    uint32_t n=threadIdx.x>>t_bits;
    uint32_t t=threadIdx.x & (1<<t_bits)-1;

    #pragma unroll
    for(int32_t k=0;k<8;k++) 
      rf[k]=rf[k] * roots[ROOTS_16_7x9 + (base + n*advance)*512 + (k<<t_bits) + t];

    __syncthreads();          // force separation of the two loops
  }

  __device__ __forceinline__ void twiddle_22(bls12381_fr* roots, uint32_t base, uint32_t advance) {
    uint32_t n=threadIdx.x>>t_bits;
    uint32_t t=threadIdx.x & (1<<t_bits)-1;

    // One weird facet of this code, if you have both multiplications in the same loop, then register
    // usage blows up.  So, we break it into two separate multiplies with syncthreads between them.

    #pragma unroll
    for(int32_t k=0;k<8;k++) 
      rf[k]=rf[k] * roots[ROOTS_12 + ((base + n*advance)*((k<<t_bits) + t)>>10)];

    __syncthreads();          // force separation of the two loops

    #pragma unroll
    for(int32_t k=0;k<8;k++) 
      rf[k]=rf[k] * roots[ROOTS_22_LOW_10 + ((base + n*advance)*((k<<t_bits) + t) & 0x3FF)];

    __syncthreads();          // force separation of the two loops
  }

  __device__ __forceinline__ void twiddle_25(bls12381_fr* roots, uint32_t base, uint32_t advance) {
    uint32_t n=threadIdx.x>>t_bits;
    uint32_t t=threadIdx.x & (1<<t_bits)-1;

    // One weird facet of this code, if you have both multiplications in the same loop, then register
    // usage blows up.  So, we break it into two separate multiplies with syncthreads between them.

    #pragma unroll
    for(int32_t k=0;k<8;k++) 
      rf[k]=rf[k] * roots[ROOTS_12 + ((base + n*advance)*((k<<t_bits) + t)>>13)];

    __syncthreads();          // force separation of the two loops

    #pragma unroll
    for(int32_t k=0;k<8;k++) 
      rf[k]=rf[k] * roots[ROOTS_25_LOW_13 + ((base + n*advance)*((k<<t_bits) + t) & 0x1FFF)];

    __syncthreads();          // force separation of the two loops
  }

  __device__ __forceinline__ void coset_25(bls12381_fr* roots, uint32_t base, uint32_t stride) {
    uint32_t n=threadIdx.x>>t_bits;
    uint32_t t=threadIdx.x & (1<<t_bits)-1;

    __syncthreads();          // force separation of the two loops

    #pragma unroll
    for(int32_t k=0;k<8;k++) 
      rf[k]=rf[k] * roots[COSET_25_HIGH_12 + (base + n + ((k<<t_bits) + t)*stride>>13)];

    __syncthreads();          // force separation of the two loops

    #pragma unroll
    for(int32_t k=0;k<8;k++) 
      rf[k]=rf[k] * roots[COSET_25_LOW_13 + (base + n + ((k<<t_bits) + t)*stride & 0x1FFF)];

    __syncthreads();          // force separation of the two loops
  }

  __device__ __forceinline__ void coset_25_to_montgomery(bls12381_fr* roots, uint32_t base, uint32_t stride) {
    uint32_t n=threadIdx.x>>t_bits;
    uint32_t t=threadIdx.x & (1<<t_bits)-1;

    __syncthreads();          // force separation of the two loops

    #pragma unroll
    for(int32_t k=0;k<8;k++) 
      rf[k]=rf[k] * roots[COSET_TO_MONTGOMERY_25_HIGH_12 + (base + n + ((k<<t_bits) + t)*stride>>13)];

    __syncthreads();          // force separation of the two loops

    #pragma unroll
    for(int32_t k=0;k<8;k++) 
      rf[k]=rf[k] * roots[COSET_25_LOW_13 + (base + n + ((k<<t_bits) + t)*stride & 0x1FFF)];

    __syncthreads();          // force separation of the two loops
  }

  __device__ __forceinline__ void scale(bls12381_fr factor) {
    #pragma unroll
    for(int32_t i=0;i<8;i++)
      rf[i]=rf[i] * factor;
  }

  __device__ __forceinline__ void ntt_1() {
    bls12381_fr temp;

    #pragma unroll
    for(int32_t base=0;base<4;base++) {
      temp       = rf[base+0] - rf[base+4];
      rf[base+0] = rf[base+0] + rf[base+4];
      rf[base+4] = temp;
    }
  }

  __device__ __forceinline__ void intt_1() {
    bls12381_fr temp;

    #pragma unroll
    for(int32_t base=0;base<4;base++) {
      temp       = rf[base+0] - rf[base+4];
      rf[base+0] = rf[base+0] + rf[base+4];
      rf[base+4] = temp;
    }
  }

  __device__ __forceinline__ void ntt_2() {
    bls12381_fr temp;

    // nice single temp NTT 4 schedule
    temp  = rf[0] + rf[4];
    rf[4] = rf[0] - rf[4];
    rf[0] = rf[2] + rf[6];
    rf[2] = rf[2] - rf[6];

    rf[2] = rf[2] * bls12381_fr::root_1_4();

    rf[6] = rf[4] - rf[2];
    rf[2] = rf[4] + rf[2];
    rf[4] = temp  - rf[0];
    rf[0] = temp  + rf[0];

    // copy of the above for odds
    temp  = rf[1] + rf[5];
    rf[5] = rf[1] - rf[5];
    rf[1] = rf[3] + rf[7];
    rf[3] = rf[3] - rf[7];

    rf[3] = rf[3] * bls12381_fr::root_1_4();

    rf[7] = rf[5] - rf[3];
    rf[3] = rf[5] + rf[3];
    rf[5] = temp  - rf[1];
    rf[1] = temp  + rf[1];
  }

  __device__ __forceinline__ void intt_2() {
    bls12381_fr temp;

    // nice single temp NTT 4 schedule
    temp  = rf[0] + rf[4];
    rf[4] = rf[0] - rf[4];
    rf[0] = rf[2] + rf[6];
    rf[2] = rf[2] - rf[6];

    rf[2] = rf[2] * bls12381_fr::root_3_4();

    rf[6] = rf[4] - rf[2];
    rf[2] = rf[4] + rf[2];
    rf[4] = temp  - rf[0];
    rf[0] = temp  + rf[0];

    // copy of the above for odds
    temp  = rf[1] + rf[5];
    rf[5] = rf[1] - rf[5];
    rf[1] = rf[3] + rf[7];
    rf[3] = rf[3] - rf[7];

    rf[3] = rf[3] * bls12381_fr::root_3_4();

    rf[7] = rf[5] - rf[3];
    rf[3] = rf[5] + rf[3];
    rf[5] = temp  - rf[1];
    rf[1] = temp  + rf[1];
  }

  __device__ __forceinline__ void ntt_3() {
    bls12381_fr temp;

    // here is a nice single temporary NTT 8 schedule

    temp  = rf[3] - rf[7];
    rf[7] = rf[3] + rf[7];
    rf[3] = rf[1] - rf[5];
    rf[5] = rf[1] + rf[5];
    rf[1] = rf[2] + rf[6];
    rf[2] = rf[2] - rf[6];
    rf[6] = rf[0] + rf[4];
    rf[0] = rf[0] - rf[4];

    temp  = temp  * bls12381_fr::root_1_4();
    rf[2] = rf[2] * bls12381_fr::root_1_4();  

    rf[4] = rf[6] + rf[1];
    rf[6] = rf[6] - rf[1];
    rf[1] = rf[3] + temp;
    rf[3] = rf[3] - temp;
    temp  = rf[5] + rf[7];
    rf[5] = rf[5] - rf[7]; 
    rf[7] = rf[0] + rf[2];
    rf[0] = rf[0] - rf[2];

    rf[1] = rf[1] * bls12381_fr::root_1_8();
    rf[5] = rf[5] * bls12381_fr::root_1_4();
    rf[3] = rf[3] * bls12381_fr::root_3_8();
  
    rf[2] = rf[6] + rf[5];
    rf[6] = rf[6] - rf[5];
    rf[5] = rf[7] - rf[1];
    rf[1] = rf[7] + rf[1];
    rf[7] = rf[0] - rf[3];
    rf[3] = rf[0] + rf[3];
    rf[0] = rf[4] + temp;
    rf[4] = rf[4] - temp;
  }

  __device__ __forceinline__ void intt_3() {
    bls12381_fr temp;

    // here is a nice single temporary NTT 8 schedule

    temp  = rf[3] - rf[7];
    rf[7] = rf[3] + rf[7];
    rf[3] = rf[1] - rf[5];
    rf[5] = rf[1] + rf[5];
    rf[1] = rf[2] + rf[6];
    rf[2] = rf[2] - rf[6];
    rf[6] = rf[0] + rf[4];
    rf[0] = rf[0] - rf[4];

    temp  = temp  * bls12381_fr::root_3_4();
    rf[2] = rf[2] * bls12381_fr::root_3_4();  

    rf[4] = rf[6] + rf[1];
    rf[6] = rf[6] - rf[1];
    rf[1] = rf[3] + temp;
    rf[3] = rf[3] - temp;
    temp  = rf[5] + rf[7];
    rf[5] = rf[5] - rf[7]; 
    rf[7] = rf[0] + rf[2];
    rf[0] = rf[0] - rf[2];

    rf[1] = rf[1] * bls12381_fr::root_7_8();
    rf[5] = rf[5] * bls12381_fr::root_3_4();
    rf[3] = rf[3] * bls12381_fr::root_5_8();
  
    rf[2] = rf[6] + rf[5];
    rf[6] = rf[6] - rf[5];
    rf[5] = rf[7] - rf[1];
    rf[1] = rf[7] + rf[1];
    rf[7] = rf[0] - rf[3];
    rf[3] = rf[0] + rf[3];
    rf[0] = rf[4] + temp;
    rf[4] = rf[4] - temp;
  }

  __device__ __forceinline__ void ntt_7(uint4* shared, bls12381_fr* roots) {
    ntt_3();
    transpose_7_1(shared);
    twiddle_6_3x3(roots);
    ntt_3();
    transpose_7_2(shared);
    twiddle_7_1x6(roots);
    ntt_1();
  }

  __device__ __forceinline__ void intt_7(uint4* shared, bls12381_fr* roots) {
    intt_3();
    transpose_7_1(shared);
    twiddle_6_3x3(roots);
    intt_3();
    transpose_7_2(shared);
    twiddle_7_1x6(roots);
    intt_1();
  }

  __device__ __forceinline__ void ntt_8(uint4* shared, bls12381_fr* roots) {
    ntt_3();
    transpose_8_1(shared);
    twiddle_6_3x3(roots);
    ntt_3();
    transpose_8_2(shared);
    twiddle_8_2x6(roots);
    ntt_2();
  }

  __device__ __forceinline__ void intt_8(uint4* shared, bls12381_fr* roots) {
    intt_3();
    transpose_8_1(shared);
    twiddle_6_3x3(roots);
    intt_3();
    transpose_8_2(shared);
    twiddle_8_2x6(roots);
    intt_2();
  }

  __device__ __forceinline__ void ntt_9(uint4* shared, bls12381_fr* roots) {
    ntt_3();
    transpose_9_1(shared);
    twiddle_6_3x3(roots);
    ntt_3();
    transpose_9_2(shared);
    twiddle_9_3x6(roots);
    ntt_3();
  }

  __device__ __forceinline__ void intt_9(uint4* shared, bls12381_fr* roots) {
    intt_3();
    transpose_9_1(shared);
    twiddle_6_3x3(roots);
    intt_3();
    transpose_9_2(shared);
    twiddle_9_3x6(roots);
    intt_3();
  }
};
