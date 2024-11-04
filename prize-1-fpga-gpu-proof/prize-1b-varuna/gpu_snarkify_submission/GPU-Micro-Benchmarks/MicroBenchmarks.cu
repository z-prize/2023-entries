/***

Copyright (c) 2024, Snarkify, Inc. All rights reserved.
Dual licensed under the MIT License or the Apache License, Version 2.0.
See LICENSE for details.

Author(s):  Niall Emmart

***/

#include <stdio.h>
#include <stdint.h>
#include "asm.cu"
#include "Chain.cu"
#include "MP.cu"
#include "BLS12381_fr.cu"

#include "Common/core.h"
#include "Common/host.h"

typedef storage<ec::xy<ec::bls12381>,false> affine_t;

typedef storage<ff::bls12381_fr,false>      st_t;
typedef impl<ff::bls12381_fr,false>         impl_t;
typedef impl<ff::bls12381_fr,true>          impl_mont_t;

typedef host::ntt_context<ff::bls12381_fr>  ntt_context_t;
//typedef host::msm_context<ec::bls12381>     msm_context_t;
//typedef host::poly_context<ff::bls12381_fr> poly_context_t;

#define $CUDA(call) { int localEC=call; if(localEC!=cudaSuccess) { printf("\nCall \"" #call "\" failed from %s, line %d, error=%d\n", __FILE__, __LINE__, localEC); exit(1); } }

#include "ScanReduce.cu"
#include "PowerTable.cu"
#include "NTT/NTT.h"
#include "Montgomery.cu"
#include "Polynomial.cu"
#include "MSM/MSM.h"
#include "MathAPI.cu"

uint32_t generator[24]={
  0xDB22C6BB, 0xFB3AF00A, 0xF97A1AEF, 0x6C55E83F, 0x171BAC58, 0xA14E3A3F,  /* x */
  0x9774B905, 0xC3688C4F, 0x4FA9AC0F, 0x2695638C, 0x3197D794, 0x17F1D3A7,
  0x46C5E7E1, 0x0CAA2329, 0xA2888AE4, 0xD03CC744, 0x2C04B3ED, 0x00DB18CB,  /* y */
  0xD5D00AF6, 0xFCF5E095, 0x741D8AE4, 0xA09E30ED, 0xE3AAA0F1, 0x08B3F481,
};

#define SMALL (1<<22)
#define LARGE (1<<25)
#define POW_TAB 256*49

__managed__ uint32_t msm_points[SMALL*2*12];
__managed__ uint32_t roots[ROOT_COUNT*8];
__managed__ uint32_t inv_roots[ROOT_COUNT*8];

__managed__ uint32_t pow_tab[POW_TAB];
__managed__ uint32_t ipow_tab[POW_TAB];
__managed__ uint32_t eval_x[8];

__managed__ uint32_t s_res[SMALL*8];
__managed__ uint32_t s_a[SMALL*8];
__managed__ uint32_t s_b[SMALL*8];
__managed__ uint32_t s_t1[SMALL*8];
__managed__ uint32_t s_t2[SMALL*8];

__managed__ uint32_t l_res[LARGE*8];
__managed__ uint32_t l_a[LARGE*8];
__managed__ uint32_t l_b[LARGE*8];
__managed__ uint32_t l_t1[LARGE*8];
__managed__ uint32_t l_t2[LARGE*8];

MSM* msm=NULL;

uint32_t random_word() {
  uint32_t r=rand() & 0xFFFF;

  r=(r<<16) | (rand() & 0xFFFF);
  return r;
}

void random_words(uint32_t* data, uint32_t count) {
  for(int i=0;i<count;i++)
    data[i]=random_word();
}

void random_scalars(uint32_t* data, uint32_t count) {
  random_words(data, count*8);
  for(int i=0;i<count;i++)
    data[i*8+7]=data[i*8+7] % 0x73EDA753; 
}

void run_once(const char* op) {
  if(strcmp(op, "coset-ntt-25")==0) 
    coset_25((bls12381_fr*)l_res, (bls12381_fr*)l_t2, (bls12381_fr*)l_t1, (bls12381_fr*)l_a, (bls12381_fr*)roots);
  else if(strcmp(op, "ntt-25")==0) 
    ntt_25((bls12381_fr*)l_res, (bls12381_fr*)l_t2, (bls12381_fr*)l_t1, (bls12381_fr*)l_a, (bls12381_fr*)roots);
  else if(strcmp(op, "intt-25")==0) 
    intt_25((bls12381_fr*)l_res, (bls12381_fr*)l_t2, (bls12381_fr*)l_t1, (bls12381_fr*)l_a, (bls12381_fr*)roots);
  else if(strcmp(op, "ntt-22")==0) 
    ntt_22((bls12381_fr*)s_res, (bls12381_fr*)s_t2, (bls12381_fr*)s_t1, (bls12381_fr*)s_a, (bls12381_fr*)roots);
  else if(strcmp(op, "intt-22")==0) 
    intt_22((bls12381_fr*)s_res, (bls12381_fr*)s_t2, (bls12381_fr*)s_t1, (bls12381_fr*)s_a, (bls12381_fr*)roots);
  else if(strcmp(op, "polynomial-divide-22")==0)
    polynomial_divide((bls12381_fr*)s_res, (bls12381_fr*)s_a, (bls12381_fr*)s_t1, (bls12381_fr*)pow_tab, (bls12381_fr*)ipow_tab, SMALL);
  else if(strcmp(op, "polynomial-evaluate-22")==0)
    polynomial_evaluate((bls12381_fr*)s_res, (bls12381_fr*)s_a, (bls12381_fr*)s_t1, (bls12381_fr*)pow_tab, SMALL);
  else if(strcmp(op, "polynomial-evaluate-25")==0)
    polynomial_evaluate((bls12381_fr*)l_res, (bls12381_fr*)l_a, (bls12381_fr*)l_t1, (bls12381_fr*)pow_tab, LARGE);
  else if(strcmp(op, "batch-invert-22")==0)
    array_batch_invert((bls12381_fr*)s_res, (bls12381_fr*)s_a, (bls12381_fr*)s_t1, SMALL);
  else if(strcmp(op, "msm-22")==0) 
    msm->msm(s_a, SMALL);
  else
    printf("Unhandled op: %s\n", op);
}

void run(const char* op, uint32_t runs) {
  cudaEvent_t start, stop;
  float       time, total=0.0, min=0.0, max=0.0;

  run_once(op);
  $CUDA(cudaEventCreate(&start));
  $CUDA(cudaEventCreate(&stop));
  for(int run=0;run<runs;run++) {
    $CUDA(cudaEventRecord(start, 0));
    run_once(op);
    $CUDA(cudaEventRecord(stop, 0));
    $CUDA(cudaEventSynchronize(stop));
    $CUDA(cudaEventElapsedTime(&time, start, stop));
    if(run==0) {
      min=time; 
      max=time;
    }
    else {
      min=(min<time) ? min : time;
      max=(max>time) ? max : time;
    }       
    total+=time;
  }
  printf("%s: min=%0.3f max=%0.3f ave=%0.3f\n", op, min, max, total/((float)runs));
  $CUDA(cudaEventDestroy(start));
  $CUDA(cudaEventDestroy(stop));
}


int main() {
  random_scalars(eval_x, 1);
  random_scalars(s_res, SMALL);
  random_scalars(s_a, SMALL);
  random_scalars(s_b, SMALL);
  random_scalars(l_res, LARGE);
  random_scalars(l_a, LARGE);
  random_scalars(l_b, LARGE);

  for(int i=0;i<SMALL;i++) {
    for(int j=0;j<24;j++) {
      msm_points[i*24 + j]=generator[j];
    }
  }

  msm=new MSM();

  msm->initialize(msm_points, SMALL, 1);

  initialize_ntt_roots((bls12381_fr*)roots, (bls12381_fr*)inv_roots);

  generate_power_table(pow_tab, eval_x);
  generate_inv_power_table(ipow_tab, eval_x);

  run("coset-ntt-25", 20);
  run("coset-ntt-25", 20);
  run("ntt-25", 20);
  run("ntt-22", 20);
  run("intt-22", 20);
  run("polynomial-divide-22", 20);
  run("polynomial-evaluate-22", 20);
  run("polynomial-evaluate-25", 20);
  run("batch-invert-22", 20);
  run("msm-22", 20);
  $CUDA(cudaDeviceSynchronize());
}
