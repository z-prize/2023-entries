/***

Copyright (c) 2024, Snarkify, Inc. All rights reserved.
Dual licensed under the MIT License or the Apache License, Version 2.0.
See LICENSE for details.

Author(s):  Niall Emmart

***/

template<bool inv>
void generate_ntt_roots(st_t* roots) {
  uint32_t      next=0;
  st_t          st_seven={7, 0, 0, 0};
  impl_mont_t   r_squared=impl_mont_t::one();
  impl_mont_t   impl_seven=(impl_mont_t)st_seven;
  int32_t       sign=inv ? -1 : 1;
  ntt_context_t root_generator;

  root_generator.initialize_root_table(25);

  if(inv)
   impl_seven=impl_seven.inv();

  // we are missing some convert functionality in the host library, this is a cheap hack
  for(int i=0;i<256;i++)
    r_squared=r_squared + r_squared;

  roots[next++]=r_squared*root_generator.inverse_size(22);
  roots[next++]=r_squared*root_generator.inverse_size(25);
  roots[next++]=root_generator.inverse_size(22);
  roots[next++]=root_generator.inverse_size(25);

  next=32;
  // roots_6_3x3
  for(int row=0;row<8;row++) 
    for(int col=0;col<8;col++)
      roots[next++]=r_squared*root_generator.root(sign*row*col, 6);

  // roots_7
  for(int i=0;i<128;i++) 
    roots[next++]=r_squared*root_generator.root(sign*i, 7);

  // roots_8_2x6
  for(int row=0;row<4;row++) 
    for(int col=0;col<64;col++)
      roots[next++]=r_squared*root_generator.root(sign*row*col, 8);

  // roots_9_3x6
  for(int row=0;row<8;row++) 
    for(int col=0;col<64;col++)
      roots[next++]=r_squared*root_generator.root(sign*row*col, 9);

  // roots_14_7x7
  for(int row=0;row<128;row++) 
    for(int col=0;col<128;col++)
      roots[next++]=r_squared*root_generator.root(sign*row*col, 14);

  // roots_16_7x9
  for(int row=0;row<128;row++)
    for(int col=0;col<512;col++)
      roots[next++]=r_squared*root_generator.root(sign*row*col, 16);

  // root_12
  for(int i=0;i<4096;i++)
    roots[next++]=r_squared*root_generator.root(sign*i, 12);

  // root_22_low_10
  for(int i=0;i<1024;i++)
    roots[next++]=r_squared*root_generator.root(sign*i, 22);

  // root_25_low_13
  for(int i=0;i<8192;i++)
    roots[next++]=r_squared*root_generator.root(sign*i, 25);

  // coset_to_montgomery_25_high_12
  for(int i=0;i<4096;i++)
    roots[next++]=r_squared*r_squared*impl_seven.pow(i<<13);

  // coset_25_high_12
  for(int i=0;i<4096;i++)
    roots[next++]=r_squared*impl_seven.pow(i<<13);

  // icoset_25_low_13
  for(int i=0;i<8192;i++)
    roots[next++]=r_squared*impl_seven.pow(i);
}

__launch_bounds__(512)
__global__ void ntt_22_first_pass(bls12381_fr* out, bls12381_fr* in, bls12381_fr* roots) {
  const uint32_t   size1=128, size2=128, size3=256;
  NTT<4, 4>        ntt;
  __shared__ uint4 transpose_buffer[8*256+128];

  // Mixed radixes: first_pass=7 second_pass=7 third_pass=8
  ntt.load_advance_1(transpose_buffer, (bls12381_fr*)in, blockIdx.x*16, size2*size3);
  ntt.ntt_7(transpose_buffer, roots);
  ntt.store_stride_1(transpose_buffer, (bls12381_fr*)out, blockIdx.x*2048, size1);
}

__launch_bounds__(512)
__global__ void ntt_22_second_pass(bls12381_fr* out, bls12381_fr* in, bls12381_fr* roots) {
  const uint32_t   size1=128, size2=128, size3=256;
  NTT<4, 4>        ntt;
  uint32_t         i, j;
  __shared__ uint4 transpose_buffer[8*256+128];

  i=blockIdx.x & 0x07;
  j=blockIdx.x>>3;

  // Mixed radixes: first_pass=7 second_pass=7 third_pass=8
  ntt.load_advance_1(transpose_buffer, (bls12381_fr*)in, i*16 + j*size1, size1*size3);
  ntt.twiddle_14_7x7(roots, i*16, 1);
  ntt.ntt_7(transpose_buffer, roots);
  ntt.store_advance_1(transpose_buffer, (bls12381_fr*)out, i*16 + j*size1*size2, size1);
}

__launch_bounds__(512)
__global__ void ntt_22_third_pass(bls12381_fr* out, bls12381_fr* in, bls12381_fr* roots) {
  const uint32_t   size1=128, size2=128;
  NTT<3, 5>        ntt;
  __shared__ uint4 transpose_buffer[8*256+128];

  // Mixed radixes: first_pass=7 second_pass=7 third_pass=8
  ntt.load_advance_1(transpose_buffer, in, blockIdx.x*8, size1*size2);
  ntt.twiddle_22(roots, blockIdx.x*8, 1);
  ntt.ntt_8(transpose_buffer, roots);
  ntt.store_advance_1(transpose_buffer, out, blockIdx.x*8, size1*size2);
}

__launch_bounds__(512)
__global__ void ntt_25_first_pass(bls12381_fr* out, bls12381_fr* in, bls12381_fr* roots) {
  const uint32_t   size1=128, size2=512, size3=512;
  NTT<4, 4>        ntt;
  __shared__ uint4 transpose_buffer[8*256+128];

  // Mixed radixes: first_pass=7 second_pass=9 third_pass=9
  ntt.load_advance_1(transpose_buffer, in, blockIdx.x*16, size2*size3);
  ntt.ntt_7(transpose_buffer, roots);
  ntt.store_stride_1(transpose_buffer, out, blockIdx.x*2048, size1);
}

__launch_bounds__(512)
__global__ void ntt_25_second_pass(bls12381_fr* out, bls12381_fr* in, bls12381_fr* roots) {
  const uint32_t   size1=128, size2=512, size3=512;
  NTT<2, 6>        ntt;
  uint32_t         i, j;
  __shared__ uint4 transpose_buffer[8*256+128];

  i=blockIdx.x & 0x1F;
  j=blockIdx.x>>5;

  // Mixed radixes: first_pass=7 second_pass=9 third_pass=9
  ntt.load_advance_1(transpose_buffer, in, i*4 + j*size1, size1*size3);
  ntt.twiddle_16_7x9(roots, i*4, 1);
  ntt.ntt_9(transpose_buffer, roots);
  ntt.store_advance_1(transpose_buffer, out, i*4 + j*size1*size2, size1);
}

__launch_bounds__(512)
__global__ void ntt_25_third_pass(bls12381_fr* out, bls12381_fr* in, bls12381_fr* roots) {
  const uint32_t   size1=128, size2=512;
  NTT<2, 6>        ntt;
  __shared__ uint4 transpose_buffer[8*256+128];

  // Mixed radixes: first_pass=7 second_pass=9 third_pass=9
  ntt.load_advance_1(transpose_buffer, in, blockIdx.x*4, size1*size2);
  ntt.twiddle_25(roots, blockIdx.x*4, 1);
  ntt.ntt_9(transpose_buffer, roots);
  ntt.store_advance_1(transpose_buffer, out, blockIdx.x*4, size1*size2);
}

__launch_bounds__(512) 
__global__ void coset_ntt_25_first_pass(bls12381_fr* out, bls12381_fr* in, bls12381_fr* roots) {
  const uint32_t   size1=128, size2=512, size3=512;
  NTT<4, 4>        ntt;
  __shared__ uint4 transpose_buffer[8*256+128];

  // Mixed radixes: first_pass=7 second_pass=9 third_pass=9
  ntt.load_advance_1(transpose_buffer, in, blockIdx.x*16, size2*size3);
  ntt.coset_25(roots, blockIdx.x*16, size2*size3);
  ntt.ntt_7(transpose_buffer, roots);
  ntt.store_stride_1(transpose_buffer, out, blockIdx.x*2048, size1);
}

__launch_bounds__(512) 
__global__ void coset_ntt_25_to_montgomery_first_pass(bls12381_fr* out, bls12381_fr* in, bls12381_fr* roots) {
  const uint32_t   size1=128, size2=512, size3=512;
  NTT<4, 4>        ntt;
  __shared__ uint4 transpose_buffer[8*256+128];

  // Mixed radixes: first_pass=7 second_pass=9 third_pass=9
  ntt.load_advance_1(transpose_buffer, in, blockIdx.x*16, size2*size3);
  ntt.coset_25_to_montgomery(roots, blockIdx.x*16, size2*size3);
  ntt.ntt_7(transpose_buffer, roots);
  ntt.store_stride_1(transpose_buffer, out, blockIdx.x*2048, size1);
}

__launch_bounds__(512)
__global__ void intt_22_first_pass(bls12381_fr* out, bls12381_fr* in, bls12381_fr* roots) {
  const uint32_t   size1=128, size2=128, size3=256;
  NTT<4, 4>        ntt;
  bls12381_fr*     root_ptr=(bls12381_fr*)roots;
  __shared__ uint4 transpose_buffer[8*256+128];

  // Mixed radixes: first_pass=7 second_pass=7 third_pass=8
  ntt.load_advance_1(transpose_buffer, in, blockIdx.x*16, size2*size3);
  ntt.scale(root_ptr[0]);
  ntt.intt_7(transpose_buffer, roots);
  ntt.store_stride_1(transpose_buffer, out, blockIdx.x*2048, size1);
}

__launch_bounds__(512)
__global__ void intt_22_from_montgomery_first_pass(bls12381_fr* out, bls12381_fr* in, bls12381_fr* roots) {
  const uint32_t   size1=128, size2=128, size3=256;
  NTT<4, 4>        ntt;
  bls12381_fr*     root_ptr=(bls12381_fr*)roots;
  __shared__ uint4 transpose_buffer[8*256+128];

  // Mixed radixes: first_pass=7 second_pass=7 third_pass=8
  ntt.load_advance_1(transpose_buffer, in, blockIdx.x*16, size2*size3);
  ntt.scale(root_ptr[2]);
  ntt.intt_7(transpose_buffer, roots);
  ntt.store_stride_1(transpose_buffer, out, blockIdx.x*2048, size1);
}

__launch_bounds__(512)
__global__ void intt_22_second_pass(bls12381_fr* out, bls12381_fr* in, bls12381_fr* roots) {
  const uint32_t   size1=128, size2=128, size3=256;
  NTT<4, 4>        ntt;
  uint32_t         i, j;
  __shared__ uint4 transpose_buffer[8*256+128];

  i=blockIdx.x & 0x07;
  j=blockIdx.x>>3;

  // Mixed radixes: first_pass=7 second_pass=7 third_pass=8
  ntt.load_advance_1(transpose_buffer, in, i*16 + j*size1, size1*size3);
  ntt.twiddle_14_7x7(roots, i*16, 1);
  ntt.intt_7(transpose_buffer, roots);
  ntt.store_advance_1(transpose_buffer, out, i*16 + j*size1*size2, size1);
}

__launch_bounds__(512)
__global__ void intt_22_third_pass(bls12381_fr* out, bls12381_fr* in, bls12381_fr* roots) {
  const uint32_t   size1=128, size2=128;
  NTT<3, 5>        ntt;
  __shared__ uint4 transpose_buffer[8*256+128];

  // Mixed radixes: first_pass=7 second_pass=7 third_pass=8
  ntt.load_advance_1(transpose_buffer, in, blockIdx.x*8, size1*size2);
  ntt.twiddle_22(roots, blockIdx.x*8, 1);
  ntt.intt_8(transpose_buffer, roots);
  ntt.store_advance_1(transpose_buffer, out, blockIdx.x*8, size1*size2);
}

__launch_bounds__(512)
__global__ void intt_25_first_pass(bls12381_fr* out, bls12381_fr* in, bls12381_fr* roots) {
  const uint32_t   size1=128, size2=512, size3=512;
  NTT<4, 4>        ntt;
  bls12381_fr*     root_ptr=(bls12381_fr*)roots;
  __shared__ uint4 transpose_buffer[8*256+128];

  // Mixed radixes: first_pass=7 second_pass=9 third_pass=9
  ntt.load_advance_1(transpose_buffer, in, blockIdx.x*16, size2*size3);
  ntt.scale(root_ptr[1]);
  ntt.intt_7(transpose_buffer, roots);
  ntt.store_stride_1(transpose_buffer, out, blockIdx.x*2048, size1);
}

__launch_bounds__(512)
__global__ void intt_25_from_montgomery_first_pass(bls12381_fr* out, bls12381_fr* in, bls12381_fr* roots) {
  const uint32_t   size1=128, size2=512, size3=512;
  NTT<4, 4>        ntt;
  bls12381_fr*     root_ptr=(bls12381_fr*)roots;
  __shared__ uint4 transpose_buffer[8*256+128];

  // Mixed radixes: first_pass=7 second_pass=9 third_pass=9
  ntt.load_advance_1(transpose_buffer, in, blockIdx.x*16, size2*size3);
  ntt.scale(root_ptr[3]);
  ntt.intt_7(transpose_buffer, roots);
  ntt.store_stride_1(transpose_buffer, out, blockIdx.x*2048, size1);
}

__launch_bounds__(512)
__global__ void intt_25_second_pass(bls12381_fr* out, bls12381_fr* in, bls12381_fr* roots) {
  const uint32_t   size1=128, size2=512, size3=512;
  NTT<2, 6>        ntt;
  uint32_t         i, j;
  __shared__ uint4 transpose_buffer[8*256+128];

  i=blockIdx.x & 0x1F;
  j=blockIdx.x>>5;

  // Mixed radixes: first_pass=7 second_pass=9 third_pass=9
  ntt.load_advance_1(transpose_buffer, in, i*4 + j*size1, size1*size3);
  ntt.twiddle_16_7x9(roots, i*4, 1);
  ntt.intt_9(transpose_buffer, roots);
  ntt.store_advance_1(transpose_buffer, out, i*4 + j*size1*size2, size1);
}

__launch_bounds__(512)
__global__ void intt_25_third_pass(bls12381_fr* out, bls12381_fr* in, bls12381_fr* roots) {
  const uint32_t   size1=128, size2=512;
  NTT<2, 6>        ntt;
  __shared__ uint4 transpose_buffer[8*256+128];

  // Mixed radixes: first_pass=7 second_pass=9 third_pass=9
  ntt.load_advance_1(transpose_buffer, in, blockIdx.x*4, size1*size2);
  ntt.twiddle_25(roots, blockIdx.x*4, 1);
  ntt.intt_9(transpose_buffer, roots);
  ntt.store_advance_1(transpose_buffer, out, blockIdx.x*4, size1*size2);
}

__launch_bounds__(512)
__global__ void icoset_25_third_pass(bls12381_fr* out, bls12381_fr* in, bls12381_fr* roots) {
  const uint32_t   size1=128, size2=512;
  NTT<2, 6>        ntt;
  __shared__ uint4 transpose_buffer[8*256+128];

  // Mixed radixes: first_pass=7 second_pass=9 third_pass=9
  ntt.load_advance_1(transpose_buffer, in, blockIdx.x*4, size1*size2);
  ntt.twiddle_25(roots, blockIdx.x*4, 1);
  ntt.intt_9(transpose_buffer, roots);
  ntt.coset_25(roots, blockIdx.x*4, size1*size2);
  ntt.store_advance_1(transpose_buffer, out, blockIdx.x*4, size1*size2);
}
