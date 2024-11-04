/***

Copyright (c) 2024, Snarkify, Inc. All rights reserved.
Dual licensed under the MIT License or the Apache License, Version 2.0.
See LICENSE for details.

Author(s):  Niall Emmart

***/

void fp_array_to_montgomery(void* to, void* from, uint32_t count) {
  fp_array_to_montgomery_kernel<<<(count+255)/256, 256>>>(to, from, count);
}

void fp_array_from_montgomery(void* to, void* from, uint32_t count) {
  fp_array_from_montgomery_kernel<<<(count+255)/256, 256>>>(to, from, count);
}

void array_zero(bls12381_fr* to, uint32_t count) {
  array_zero_kernel<<<(count+255)/256, 256>>>(to, count);
}

void array_set_l1(bls12381_fr* to, uint32_t count) {
  array_set_l1_kernel<<<(count+255)/256, 256>>>(to, count);
}

void array_copy(bls12381_fr* to, bls12381_fr* from, uint32_t count) {
  array_copy_kernel<<<(count+255)/256, 256>>>(to, from, count);
}

void array_to_montgomery(bls12381_fr* to, bls12381_fr* from, uint32_t count) {
  array_to_montgomery_kernel<<<(count+255)/256, 256>>>(to, from, count);
}

void array_from_montgomery(bls12381_fr* to, bls12381_fr* from, uint32_t count) {
  array_from_montgomery_kernel<<<(count+255)/256, 256>>>(to, from, count);
}

template<operation_t op>
void reduce(bls12381_fr* to, bls12381_fr* from, bls12381_fr* temp, uint32_t count) {
  int32_t sm_count;

  if(count<1024) {
    reduce_1024_kernel<op><<<1, 1024>>>(to, from, count);
  }
  else {
    cudaDeviceGetAttribute(&sm_count, cudaDevAttrMultiProcessorCount, 0);

    reduce_kernel<op><<<sm_count, 1024>>>(temp, from, count);
    reduce_1024_kernel<op><<<1, 1024>>>(to, temp, sm_count);
  }
}

template<operation_t op>
void scan(bls12381_fr* to, bls12381_fr* from, bls12381_fr* temp, uint32_t count) {
  uint32_t blocks=(count+1023)>>10;

  if(count>1024)  {
    reduce_1024_kernel<op><<<blocks, 1024>>>(temp+blocks, from, count); 
    scan<op>(temp, temp+blocks, temp+blocks*2, blocks);
  }
  scan_kernel<op><<<blocks, 1024>>>(to, temp, from, count); 
}

void generate_power_table(void* void_table, void* void_value) {
  bls12381_fr* table=(bls12381_fr*)void_table;
  bls12381_fr* value=(bls12381_fr*)void_value;

  power_table_kernel<false><<<256, 49>>>(table, value);
}

void generate_inv_power_table(void* void_table, void* void_value) {
  bls12381_fr* table=(bls12381_fr*)void_table;
  bls12381_fr* value=(bls12381_fr*)void_value;

  power_table_kernel<true><<<256, 49>>>(table, value);
}

void array_batch_invert(bls12381_fr* to, bls12381_fr* from, bls12381_fr* temp, uint32_t count) {
  int32_t sm_count;

  $CUDA(cudaDeviceGetAttribute(&sm_count, cudaDevAttrMultiProcessorCount, 0));
  array_batch_invert_kernel<<<sm_count, 256>>>(to, from, temp, count);
}

void polynomial_evaluate(bls12381_fr* to, bls12381_fr* from, bls12381_fr* temp, bls12381_fr* point_pow_table, uint32_t count) {
  int32_t sm_count;

  $CUDA(cudaDeviceGetAttribute(&sm_count, cudaDevAttrMultiProcessorCount, 0));
  polynomial_evaluate_kernel<<<sm_count, 512>>>(temp, from, point_pow_table, count);
  reduce<reduce_add>(to, temp, temp + sm_count, sm_count);
}

void polynomial_divide(bls12381_fr* to, bls12381_fr* from, bls12381_fr* temp, bls12381_fr* pow_table, bls12381_fr* ipow_table, uint32_t count) {
  uint32_t blocks=(count+1023)>>10;

  // temp space required = count>>8

  // where p[i] are the input coefficients, a is an evaluation point, where a!=0, n is the number of coefficients:
  //    t[i] = p[i] * a^i
  //    s[i] = sum{i..n} t[i]     // inclusive suffix scan
  //    q[i] = (s[i]-t[i])*a^-(i+1)

  polynomial_divide_reduce_kernel<<<blocks, 1024>>>(temp, from, pow_table, count);            // read
  scan<suffix_scan_add>(temp, temp, temp+blocks, blocks);                                     // minimal   [size count/1024]
  polynomial_divide_kernel<<<blocks, 1024>>>(to, from, temp, pow_table, ipow_table, count);   // read + write
}

void polynomial_divide_from_montgomery(bls12381_fr* to, bls12381_fr* from, bls12381_fr* temp, bls12381_fr* pow_table, bls12381_fr* ipow_table, uint32_t count) {
  uint32_t blocks=(count+1023)>>10;

  // temp space required = count>>8

  // where p[i] are the input coefficients, a is an evaluation point, where a!=0, n is the number of coefficients:
  //    t[i] = p[i] * a^i
  //    s[i] = sum{i..n} t[i]     // inclusive suffix scan
  //    q[i] = (s[i]-t[i])*a^-(i+1)

  polynomial_divide_reduce_kernel<<<blocks, 1024>>>(temp, from, pow_table, count);                            // read
  scan<suffix_scan_add>(temp, temp, temp+blocks, blocks);                                                     // minimal   [size count/1024]
  polynomial_divide_from_montgomery_kernel<<<blocks, 1024>>>(to, from, temp, pow_table, ipow_table, count);   // read + write
}

void initialize_ntt_roots(bls12381_fr* roots, bls12381_fr* inv_roots) {
  st_t* mont_roots; 
  st_t* mont_inv_roots;

  mont_roots=(st_t*)malloc(32u*ROOT_COUNT);
  mont_inv_roots=(st_t*)malloc(32u*ROOT_COUNT);

  // it's probably worth noting -- these roots come back in the GPU Montgomery format, so they can be directly copied
  // to the GPU for consumption by the GPU NTT routines.
  generate_ntt_roots<false>(mont_roots);
  generate_ntt_roots<true>(mont_inv_roots);

  $CUDA(cudaMemcpy(roots, mont_roots, ROOT_COUNT*32, cudaMemcpyHostToDevice));
  $CUDA(cudaMemcpy(inv_roots, mont_inv_roots, ROOT_COUNT*32, cudaMemcpyHostToDevice));

  free(mont_roots);
  free(mont_inv_roots);
}

void ntt_22(bls12381_fr* out, bls12381_fr* temp2, bls12381_fr* temp1, bls12381_fr* in, bls12381_fr* roots) {
  ntt_22_first_pass<<<2048, 256>>>(temp1, in, roots);
  ntt_22_second_pass<<<2048, 256>>>(temp2, temp1, roots);
  ntt_22_third_pass<<<2048, 256>>>(out, temp2, roots);
}

void intt_22(bls12381_fr* out, bls12381_fr* temp2, bls12381_fr* temp1, bls12381_fr* in, bls12381_fr* roots) {
  intt_22_first_pass<<<2048, 256>>>(temp1, in, roots);
  intt_22_second_pass<<<2048, 256>>>(temp2, temp1, roots);
  intt_22_third_pass<<<2048, 256>>>(out, temp2, roots);
}

void intt_22_from_montgomery(bls12381_fr* out, bls12381_fr* temp2, bls12381_fr* temp1, bls12381_fr* in, bls12381_fr* roots) {
  intt_22_from_montgomery_first_pass<<<2048, 256>>>(temp1, in, roots);
  intt_22_second_pass<<<2048, 256>>>(temp2, temp1, roots);
  intt_22_third_pass<<<2048, 256>>>(out, temp2, roots);
}

void ntt_25(bls12381_fr* out, bls12381_fr* temp2, bls12381_fr* temp1, bls12381_fr* in, bls12381_fr* roots) {
  ntt_25_first_pass<<<16384, 256>>>(temp1, in, roots);
  ntt_25_second_pass<<<16384, 256>>>(temp2, temp1, roots);
  ntt_25_third_pass<<<16384, 256>>>(out, temp2, roots);
}

void intt_25(bls12381_fr* out, bls12381_fr* temp2, bls12381_fr* temp1, bls12381_fr* in, bls12381_fr* roots) {
  intt_25_first_pass<<<16384, 256>>>(temp1, in, roots);
  intt_25_second_pass<<<16384, 256>>>(temp2, temp1, roots);
  intt_25_third_pass<<<16384, 256>>>(out, temp2, roots);
}

void coset_25(bls12381_fr* out, bls12381_fr* temp2, bls12381_fr* temp1, bls12381_fr* in, bls12381_fr* roots) {
  coset_ntt_25_first_pass<<<16384, 256>>>(temp1, in, roots);
  ntt_25_second_pass<<<16384, 256>>>(temp2, temp1, roots);
  ntt_25_third_pass<<<16384, 256>>>(out, temp2, roots);
}

void coset_25_to_montgomery(bls12381_fr* out, bls12381_fr* temp2, bls12381_fr* temp1, bls12381_fr* in, bls12381_fr* roots) {
  coset_ntt_25_to_montgomery_first_pass<<<16384, 256>>>(temp1, in, roots);
  ntt_25_second_pass<<<16384, 256>>>(temp2, temp1, roots);
  ntt_25_third_pass<<<16384, 256>>>(out, temp2, roots);
}

void icoset_25_from_montgomery(bls12381_fr* out, bls12381_fr* temp2, bls12381_fr* temp1, bls12381_fr* in, bls12381_fr* roots) {
  intt_25_from_montgomery_first_pass<<<16384, 256>>>(temp1, in, roots);
  intt_25_second_pass<<<16384, 256>>>(temp2, temp1, roots);
  icoset_25_third_pass<<<16384, 256>>>(out, temp2, roots);
}

