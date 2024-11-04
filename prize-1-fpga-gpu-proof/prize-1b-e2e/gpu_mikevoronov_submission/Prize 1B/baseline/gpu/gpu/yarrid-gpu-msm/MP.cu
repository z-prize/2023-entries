/***

Portions of this file originated in an NVIDIA Open Source Project.  The source can be
found here:  http://github.com/NVlabs/CGBN, from: include/cgbn/arith/mp.cu.

Copyright (c) 2022-2024, Yrrid Software, Inc.  All rights reserved.
Licensed under the Apache License, Version 2.0, see LICENSE for details.

Written by Niall Emmart.

***/

__device__ __constant__ uint32_t zc=0;

__device__ __forceinline__ uint32_t computeNP0(uint32_t x) {
  uint32_t inv=x;

  inv=inv*(inv*x+14);
  inv=inv*(inv*x+2);
  inv=inv*(inv*x+2);
  inv=inv*(inv*x+2);
  return inv;
}

template<uint32_t limbs>
__device__ __forceinline__ void mp_zero(uint32_t* r) {
  #pragma unroll
  for(int32_t i=0;i<limbs;i++)
    r[i]=0;
}

template<uint32_t limbs>
__device__ __forceinline__ void mp_one(uint32_t* r) {
  r[0]=1;
  #pragma unroll
  for(int32_t i=1;i<limbs;i++)
    r[i]=0;
}

template<uint32_t limbs>
__device__ __forceinline__ void mp_copy(uint32_t* dst, const uint32_t* src) {
  #pragma unroll
  for(int32_t i=0;i<limbs;i++)
    dst[i]=src[i];
}

template<uint32_t limbs>
__device__ __forceinline__ uint32_t mp_logical_or(const uint32_t* a) {
  uint32_t lor=a[0];
  
  #pragma unroll
  for(int32_t i=1;i<limbs;i++)
    lor=lor | a[i];
  return lor;
}

template<uint32_t limbs>
__device__ __forceinline__ uint32_t mp_shift_right(uint32_t* r, const uint32_t* x, uint32_t bits, uint32_t fill=0) {
  #pragma unroll
  for(int32_t i=0;i<limbs-1;i++)
    r[i]=__funnelshift_rc(x[i], x[i+1], bits);
  r[limbs-1]=__funnelshift_rc(x[limbs-1], fill, bits);
}

template<uint32_t limbs>
__device__ __forceinline__ uint32_t mp_shift_left(uint32_t* r, const uint32_t* x, uint32_t bits, uint32_t fill=0) {
  #pragma unroll
  for(int32_t i=limbs-1;i>0;i--)
    r[i]=__funnelshift_lc(x[i-1], x[i], bits);
  r[0]=__funnelshift_lc(fill, x[0], bits);
}

template<uint32_t limbs>
__device__ __forceinline__ void mp_add(uint32_t* r, const uint32_t* a, const uint32_t* b) {
  chain_t chain;
  
  #pragma unroll
  for(int32_t i=0;i<limbs;i++)
    r[i]=chain.add(a[i], b[i]);
}

template<uint32_t limbs>
__device__ __forceinline__ bool mp_add_carry(uint32_t* r, const uint32_t* a, const uint32_t* b) {
  chain_t chain;
  
  #pragma unroll
  for(int32_t i=0;i<limbs;i++)
    r[i]=chain.add(a[i], b[i]);
  return chain.getCarry();
}

template<uint32_t limbs>
__device__ __forceinline__ void mp_sub(uint32_t* r, const uint32_t* a, const uint32_t* b) {
  chain_t chain;
  
  #pragma unroll
  for(int32_t i=0;i<limbs;i++)
    r[i]=chain.sub(a[i], b[i]);
}

template<uint32_t limbs>
__device__ __forceinline__ bool mp_sub_carry(uint32_t* r, const uint32_t* a, const uint32_t* b) {
  chain_t chain;
  
  #pragma unroll
  for(int32_t i=0;i<limbs;i++)
    r[i]=chain.sub(a[i], b[i]);
  return chain.getCarry();
}

template<uint32_t limbs>
__device__ __forceinline__ void mp_neg(uint32_t* r, const uint32_t* a) {
  chain_t chain;

  #pragma unroll
  for(int32_t i=0;i<limbs;i++) 
    r[i]=chain.sub(0, a[i]);
}

template<uint32_t limbs>
__device__ __forceinline__ bool mp_neg_carry(uint32_t* r, const uint32_t* a) {
  chain_t chain;

  #pragma unroll
  for(int32_t i=0;i<limbs;i++) 
    r[i]=chain.sub(0, a[i]);
  return chain.getCarry();
}

template<uint32_t limbs>
__device__ __forceinline__ bool mp_comp_eq(const uint32_t* a, const uint32_t* b) {
  uint32_t match=a[0] ^ b[0];
  
  #pragma unroll
  for(int32_t i=1;i<limbs;i++)
    match=match | (a[i] ^ b[i]);
  return match==0;
}

template<uint32_t limbs>
__device__ __forceinline__ bool mp_comp_ge(const uint32_t* a, const uint32_t* b) {
  chain_t chain;
  
  #pragma unroll
  for(int32_t i=0;i<limbs;i++)
    chain.sub(a[i], b[i]);
  return chain.getCarry();
}

template<uint32_t limbs>
__device__ __forceinline__ bool mp_comp_gt(const uint32_t* a, const uint32_t* b) {
  chain_t chain;
  
  // a>b --> b-a is negative
  #pragma unroll
  for(int32_t i=0;i<limbs;i++)
    chain.sub(b[i], a[i]);
  return !chain.getCarry();
}

template<uint32_t limbs>
__device__ __forceinline__ void mp_swap(uint32_t* a, uint32_t* b) {
  uint32_t swap;

  #pragma unroll
  for(uint32_t i=0;i<limbs;i++) {
    swap=a[i];
    a[i]=b[i];
    b[i]=swap;
  }
}

template<uint32_t limbs>
__device__ __forceinline__ void mp_select(uint32_t* r, bool abSelect, const uint32_t* a, const uint32_t* b) {
  #pragma unroll
  for(int32_t i=0;i<limbs;i++) 
    r[i]=abSelect ? a[i] : b[i];
}

template<uint32_t limbs>
__device__ __forceinline__ void mp_scale(uint32_t* r, const uint32_t* a, const uint32_t scalar) {
  uint64_t wide=0;

  // this is not very fast, don't use it for performance critical computations
  #pragma unroll
  for(int32_t i=0;i<limbs;i++) {
    wide=madwide(a[i], scalar, wide>>32);
    r[i]=ulow(wide);
  }
}

template<class NP, uint32_t limbs>
__device__ __forceinline__ bool mp_mul_red_cl(uint64_t* evenOdd, const uint32_t* a, const uint32_t* b, const uint32_t* n) {
  uint64_t* even=evenOdd;
  uint64_t* odd=evenOdd + limbs/2;
  chain_t   chain;
  bool      carry=false;
  uint32_t  lo=0, q, c1, c2;
  
  // This routine can be used when max(a, b)+n < R (i.e. it doesn't carry out).  Hence the name cl for carryless.
  // Only works with an even number of limbs.
     
  #pragma unroll
  for(int32_t i=0;i<limbs/2;i++) {
    even[i]=make_wide(0, 0);
    odd[i]=make_wide(0, 0);
  }
  
  #pragma unroll
  for(int32_t i=0;i<limbs;i+=2) {
    if(i!=0) {
      // integrate lo
      chain.reset(carry);
      lo=chain.add(lo, ulow(even[0]));
      carry=chain.add(0, 0)!=0;
      even[0]=make_wide(lo, uhigh(even[0]));
    }

    chain.reset();
    #pragma unroll
    for(int j=0;j<limbs;j+=2)
      even[j/2]=chain.madwide(a[i], b[j], even[j/2]);
    c1=chain.add(0, 0);

    chain.reset();
    #pragma unroll
    for(int j=0;j<limbs;j+=2)
      odd[j/2]=chain.madwide(a[i], b[j+1], odd[j/2]);

    q=NP::qTerm(ulow(even[0]));

    chain.reset();
    #pragma unroll
    for(int j=0;j<limbs;j+=2)
      odd[j/2]=chain.madwide(q, n[j+1], odd[j/2]);

    chain.reset();
    even[0]=chain.madwide(q, n[0], even[0]);
    lo=uhigh(even[0]);
    #pragma unroll
    for(int j=2;j<limbs;j+=2)
      even[j/2-1]=chain.madwide(q, n[j], even[j/2]);
    c1=chain.add(c1, 0);
      
    // integrate lo
    
    chain.reset(carry);
    lo=chain.add(lo, ulow(odd[0]));
    carry=chain.add(0, 0)!=0;
    odd[0]=make_wide(lo, uhigh(odd[0]));

    chain.reset();
    #pragma unroll
    for(int j=0;j<limbs;j+=2)
      odd[j/2]=chain.madwide(a[i+1], b[j], odd[j/2]);
    c2=chain.add(0, 0);

    q=NP::qTerm(ulow(odd[0]));

    // shift odd by 64 bits

    chain.reset();
    odd[0]=chain.madwide(q, n[0], odd[0]);
    lo=uhigh(odd[0]);
    #pragma unroll
    for(int j=2;j<limbs;j+=2)
      odd[j/2-1]=chain.madwide(q, n[j], odd[j/2]);
    c2=chain.add(c2, 0);

    odd[limbs/2-1]=make_wide(0, 0);
    even[limbs/2-1]=make_wide(c1, c2);
    
    chain.reset();
    #pragma unroll
    for(int j=0;j<limbs;j+=2)
      even[j/2]=chain.madwide(a[i+1], b[j+1], even[j/2]);

    chain.reset();
    #pragma unroll
    for(int j=0;j<limbs;j+=2)
      even[j/2]=chain.madwide(q, n[j+1], even[j/2]);
  }

  chain.reset(carry);
  lo=chain.add(lo, ulow(even[0]));
  carry=chain.add(0, 0)!=0;
  even[0]=make_wide(lo, uhigh(even[0]));
  return carry;
}

template<class NP, uint32_t limbs>
__device__ __forceinline__ bool mp_sqr_red_cl(uint64_t* evenOdd, uint32_t* temp, const uint32_t* a, const uint32_t* n) {
  uint64_t* even=evenOdd;
  uint64_t* odd=evenOdd + limbs/2;
  chain_t   chain;
  bool      carry=false;
  uint32_t  lo=0, q, c1, c2, low, high;
  
  // This routine can be used when a+n < R (i.e. it doesn't carry out).  Hence the name cl for carryless.
  // Only works with an even number of limbs.
  
  mp_zero<limbs>(temp);
  
  #pragma unroll
  for(int32_t i=0;i<limbs/2;i++) {
    even[i]=make_wide(0, 0);
    odd[i]=make_wide(0, 0);
  }
  
  // do odds
  for(int32_t j=limbs-1;j>0;j-=2) {
    chain.reset();
    for(int i=0;i<limbs-j;i++)
      evenOdd[j/2+i+1]=chain.madwide(a[i], a[i+j], evenOdd[j/2+i+1]);
  }

  // shift right
  for(int32_t i=0;i<limbs-1;i++)
    evenOdd[i]=make_wide(uhigh(evenOdd[i]), ulow(evenOdd[i+1]));
  evenOdd[limbs-1]=make_wide(uhigh(evenOdd[limbs-1]), 0);
   
  // do evens
  for(int32_t j=limbs-2;j>0;j-=2) {
    chain.reset();
    for(int i=0;i<limbs-j;i++) 
      evenOdd[j/2+i]=chain.madwide(a[i], a[i+j], evenOdd[j/2+i]);
    temp[limbs-j]=(chain.add(0, 0)!=0) ? 2 : 0;
  }

  // double
  chain.reset();
  for(int32_t i=0;i<limbs;i++) {
    low=chain.add(ulow(evenOdd[i]), ulow(evenOdd[i]));
    high=chain.add(uhigh(evenOdd[i]), uhigh(evenOdd[i]));
    evenOdd[i]=make_wide(low, high);
  }

  // add diagonals
  chain.reset();
  for(int32_t i=0;i<limbs;i++) 
    evenOdd[i]=chain.madwide(a[i], a[i], evenOdd[i]);

  // add high part of wide to b...
  chain.reset();
  for(int32_t i=0;i<limbs;i+=2) {
    temp[i]=chain.add(ulow(evenOdd[limbs/2+i/2]), temp[i]);
    temp[i+1]=chain.add(uhigh(evenOdd[limbs/2+i/2]), temp[i+1]);
  }

  #pragma unroll
  for(int32_t i=0;i<limbs/2;i++) 
    odd[i]=make_wide(0, 0);

  // now we need to reduce
  #pragma unroll
  for(int i=0;i<limbs/2;i++) {
    if(i!=0) {
      // integrate lo
      chain.reset(carry);
      lo=chain.add(lo, ulow(even[0]));
      carry=chain.add(0, 0)!=0;
      even[0]=make_wide(lo, uhigh(even[0]));
    }
    
    q=NP::qTerm(ulow(even[0]));

    // shift even by 64 bits
    chain.reset();
    even[0]=chain.madwide(q, n[0], even[0]);
    lo=uhigh(even[0]);
    #pragma unroll
    for(int j=2;j<limbs;j+=2)
      even[j/2-1]=chain.madwide(q, n[j], even[j/2]);
    c1=chain.add(0, 0);

    chain.reset();
    #pragma unroll
    for(int j=0;j<limbs;j+=2)
      odd[j/2]=chain.madwide(q, n[j+1], odd[j/2]);
      
    // second half

    // integrate lo
    chain.reset(carry);
    lo=chain.add(lo, ulow(odd[0]));
    carry=chain.add(0, 0)!=0;
    odd[0]=make_wide(lo, uhigh(odd[0]));
    
    q=NP::qTerm(ulow(odd[0]));

    // shift odd by 64 bits
    chain.reset();
    odd[0]=chain.madwide(q, n[0], odd[0]);
    lo=uhigh(odd[0]);
    for(int j=2;j<limbs;j+=2)
      odd[j/2-1]=chain.madwide(q, n[j], odd[j/2]);
    odd[limbs/2-1]=0;
    c2=chain.add(0, 0);

    chain.reset();
    for(int j=0;j<limbs-2;j+=2)
      even[j/2]=chain.madwide(q, n[j+1], even[j/2]);
    even[limbs/2-1]=chain.madwide(q, n[limbs-1], make_wide(c1, c2));
  }
  
  chain.reset();
  for(int i=0;i<limbs;i+=2) {
    low=chain.add(ulow(even[i/2]), temp[i]);
    high=chain.add(uhigh(even[i/2]), temp[i+1]);
    even[i/2]=make_wide(low, high);
  }
  
  chain.reset(carry);
  lo=chain.add(lo, ulow(even[0]));
  carry=chain.add(0, 0)!=0;
  even[0]=make_wide(lo, uhigh(even[0]));
  return carry;
}

template<uint32_t limbs>
__device__ __forceinline__ void mp_merge_cl(uint32_t* r, const uint64_t* evenOdd, bool carry) {
  chain_t chain(carry);
 
  r[0]=ulow(evenOdd[0]);
  for(int i=0;i<limbs/2-1;i++) {
    r[2*i+1]=chain.add(uhigh(evenOdd[i]), ulow(evenOdd[limbs/2 + i]));
    r[2*i+2]=chain.add(ulow(evenOdd[i+1]), uhigh(evenOdd[limbs/2 + i]));
  }
  r[limbs-1]=chain.add(uhigh(evenOdd[limbs/2-1]), 0);
}

template<uint32_t limbs>
__device__ __forceinline__ void mp_inverse(uint32_t* r, const uint32_t* f, const uint32_t* n) {
  uint32_t A[limbs], B[limbs], U[limbs], V[limbs];

  // we assume that n is prime
  mp_copy<limbs>(A, f);
  mp_copy<limbs>(B, n);
  mp_one<limbs>(U);
  mp_zero<limbs>(V);
  
  while(mp_logical_or<limbs>(A)!=0) {
    if((A[0] & 0x01)!=0) {
      if(mp_comp_gt<12>(B, A)) {
        mp_swap<limbs>(A, B);
        mp_swap<limbs>(U, V);
      }
      mp_sub<limbs>(A, A, B);
      if(!mp_sub_carry<limbs>(U, U, V))
        mp_add<limbs>(U, U, n);
    }
    mp_shift_right<limbs>(A, A, 1);
    if((U[0] & 0x01)!=0)
      mp_add<limbs>(U, U, n);
    mp_shift_right<limbs>(U, U, 1);
  }
  mp_copy<limbs>(r, V);
}
