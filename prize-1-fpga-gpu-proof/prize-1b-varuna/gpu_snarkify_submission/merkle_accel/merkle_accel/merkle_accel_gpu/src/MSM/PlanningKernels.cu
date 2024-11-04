/***

Copyright (c) 2023-2024, Yrrid Software, Inc. and Snarkify Inc.  All rights reserved.
Dual licensed under the MIT License or the Apache License, Version 2.0.
See LICENSE for details.

Author(s):  Niall Emmart

***/

__device__ __forceinline__ void copy(uint32_t* destination, uint32_t* source, uint32_t count) {
  #pragma unroll 1
  for(uint32_t i=threadIdx.x;i<count;i+=blockDim.x)
    destination[i]=source[i];
}

__device__ __forceinline__ uint32_t warpSum(uint32_t value) {
  #pragma unroll
  for(uint32_t bit=1;bit<=16;bit=bit+bit) 
    value=value + __shfl_xor_sync(0xFFFFFFFF, value, bit);
  return value;
}

template<uint32_t THREADS>
__device__ __forceinline__ uint32_t prefixSum(uint32_t* warpSums, uint32_t value) {
  uint32_t sum, add;

  // THREADS must be a power of 2

  sum=0;
  if(THREADS<32) {
    const uint32_t shift=(THREADS<16 ? THREADS : 16);

    if(threadIdx.x<THREADS) {
      sum=value;
      #pragma unroll
      for(uint32_t amount=1;amount<THREADS;amount=amount+amount) {
        add=__shfl_up_sync((1u<<shift)-1, sum, amount);
        sum=(threadIdx.x>=amount) ? sum+add : sum;
      }
      if(threadIdx.x==THREADS-1)
        warpSums[0]=sum;
    }
    __syncthreads();
    if(threadIdx.x>=THREADS)
      sum=warpSums[0];
  }
  else {
    const uint32_t WARPS=THREADS>>5;
    uint32_t       warp=threadIdx.x>>5, warpThread=threadIdx.x & 0x1F;

    if(warp<WARPS) {
      sum=value;
      #pragma unroll
      for(uint32_t amount=1;amount<=16;amount=amount+amount) {
        add=__shfl_up_sync(0xFFFFFFFF, sum, amount);
        sum=(warpThread>=amount) ? sum+add : sum;
      }
      if(warpThread==31)
        warpSums[warp]=sum;
    }
    __syncthreads();
    #pragma unroll
    for(int32_t i=0;i<WARPS;i++) 
      sum+=i<warp ? warpSums[i] : 0;
  }
  return sum;    
}

__device__ __forceinline__ void slice16(uint32_t* buckets, uint32_t* scalar) {
  buckets[0]=scalar[0] & 0xFFFF;
  buckets[1]=scalar[0]>>16;
  buckets[2]=scalar[1] & 0xFFFF;
  buckets[3]=scalar[1]>>16;
  buckets[4]=scalar[2] & 0xFFFF;
  buckets[5]=scalar[2]>>16;
  buckets[6]=scalar[3] & 0xFFFF;
  buckets[7]=scalar[3]>>16;
  buckets[8]=scalar[4] & 0xFFFF;
  buckets[9]=scalar[4]>>16;
  buckets[10]=scalar[5] & 0xFFFF;
  buckets[11]=scalar[5]>>16;
  buckets[12]=scalar[6] & 0xFFFF;
  buckets[13]=scalar[6]>>16;
  buckets[14]=scalar[7] & 0xFFFF;
  buckets[15]=scalar[7]>>16;
}

__device__ __forceinline__ void slice22(uint32_t* buckets, uint32_t* scalar) {
  buckets[0]=scalar[0] & 0x3FFFFF;
  buckets[1]=__funnelshift_r(scalar[0], scalar[1], 22) & 0x3FFFFF;
  buckets[2]=__funnelshift_r(scalar[1], scalar[2], 12) & 0x3FFFFF;
  buckets[3]=(scalar[2]>>2) & 0x3FFFFF;
  buckets[4]=__funnelshift_r(scalar[2], scalar[3], 24) & 0x3FFFFF;
  buckets[5]=__funnelshift_r(scalar[3], scalar[4], 14) & 0x3FFFFF;
  buckets[6]=(scalar[4]>>4) & 0x3FFFFF;
  buckets[7]=__funnelshift_r(scalar[4], scalar[5], 26) & 0x3FFFFF;
  buckets[8]=__funnelshift_r(scalar[5], scalar[6], 16) & 0x3FFFFF;
  buckets[9]=(scalar[6]>>6) & 0x3FFFFF;
  buckets[10]=__funnelshift_r(scalar[6], scalar[7], 28) & 0x3FFFFF;
  buckets[11]=__funnelshift_r(scalar[7], scalar[8], 18) & 0x3FFFFF;
}

__device__ __forceinline__ void slice23(uint32_t* buckets, uint32_t* scalar) {
  buckets[0]=scalar[0] & 0x7FFFFF;
  buckets[1]=__funnelshift_r(scalar[0], scalar[1], 23) & 0x7FFFFF;
  buckets[2]=__funnelshift_r(scalar[1], scalar[2], 14) & 0x7FFFFF;
  buckets[3]=(scalar[2]>>5) & 0x7FFFFF;
  buckets[4]=__funnelshift_r(scalar[2], scalar[3], 28) & 0x7FFFFF;
  buckets[5]=__funnelshift_r(scalar[3], scalar[4], 19) & 0x7FFFFF;
  buckets[6]=__funnelshift_r(scalar[4], scalar[5], 10) & 0x7FFFFF;
  buckets[7]=(scalar[5]>>1) & 0x7FFFFF;
  buckets[8]=__funnelshift_r(scalar[5], scalar[6], 24) & 0x7FFFFF;
  buckets[9]=__funnelshift_r(scalar[6], scalar[7], 15) & 0x7FFFFF;
  buckets[10]=(scalar[7]>>6) & 0x7FFFFF;
  buckets[11]=__funnelshift_r(scalar[7], scalar[8], 29) & 0x7FFFFF;
}

template<class Planning>
__device__ __forceinline__ void signedDigits(uint32_t* buckets, const uint4& lo, const uint4& hi, uint32_t pointIndex) {
  const uint32_t WINDOWS=Planning::WINDOWS, SIGN_BIT=Planning::SIGN_BIT, SIGN_MASK=1u<<SIGN_BIT, WINDOW_BITS=Planning::WINDOW_BITS;
  const uint32_t WINDOW_MASK=(1u<<WINDOW_BITS)-1, BUCKET_BITS=Planning::BUCKET_BITS, BUCKET_MASK=(1u<<BUCKET_BITS)-1;
  const uint32_t EXPONENT_BITS=Planning::EXPONENT_BITS, EXPONENT_HIGH_WORD=Planning::EXPONENT_HIGH_WORD;
  const bool     EXACT=EXPONENT_BITS%WINDOW_BITS==0;

  uint32_t add=0, negate=0, order[9], scalar[9];

  Planning::setOrder(order);
  order[8]=0;

  scalar[0]=lo.x; scalar[1]=lo.y; scalar[2]=lo.z; scalar[3]=lo.w;
  scalar[4]=hi.x; scalar[5]=hi.y; scalar[6]=hi.z; scalar[7]=hi.w;
  scalar[8]=0;
  
  if constexpr (EXACT) {
    const uint32_t EXPONENT_HIGH_BIT=1U<<(EXPONENT_BITS-7*32-1);

    if((hi.w & EXPONENT_HIGH_BIT)!=0) {
      mp_sub<9>(scalar, order, scalar);
      negate=SIGN_MASK;
    }
  }
  else {
    const uint32_t EXPONENT_MOD=(0x80000000ull<<WINDOWS*WINDOW_BITS-256)/(EXPONENT_HIGH_WORD+1);

    mp_scale<9>(order, order, pointIndex % EXPONENT_MOD);
    mp_add<9>(scalar, scalar, order);
  }

  if constexpr (WINDOW_BITS==16) 
    slice16(buckets, scalar);
  else if constexpr (WINDOW_BITS==22) 
    slice22(buckets, scalar);
  else if constexpr (WINDOW_BITS==23) 
    slice23(buckets, scalar);
  else {
    #pragma unroll
    for(int32_t i=0;i<WINDOWS;i++)
      buckets[i]=0;
  }

  #pragma unroll
  for(int32_t i=0;i<WINDOWS;i++) {
    buckets[i]+=add;
    add=(buckets[i]>BUCKET_MASK+1) ? 1 : 0;
    if((buckets[i] & WINDOW_MASK)==0)
      buckets[i]=0xFFFFFFFF;
    else if(buckets[i]<=BUCKET_MASK+1)
      buckets[i]=buckets[i]-1 ^ negate;
    else
      buckets[i]=buckets[i] ^ (SIGN_MASK + WINDOW_MASK) ^ negate;
  }
}

template<class Planning>
__device__ __forceinline__ void signedDigits(uint32_t* buckets, const uint4& lo, const uint4& hi, const uint4& rLo, uint4& rHi, uint32_t pointIndex) {
  const uint32_t WINDOWS=Planning::WINDOWS, SIGN_BIT=Planning::SIGN_BIT, SIGN_MASK=1u<<SIGN_BIT, WINDOW_BITS=Planning::WINDOW_BITS;
  const uint32_t WINDOW_MASK=(1u<<WINDOW_BITS)-1, BUCKET_BITS=Planning::BUCKET_BITS, BUCKET_MASK=(1u<<BUCKET_BITS)-1;
  const uint32_t EXPONENT_BITS=Planning::EXPONENT_BITS, EXPONENT_HIGH_WORD=Planning::EXPONENT_HIGH_WORD;
  const bool     EXACT=EXPONENT_BITS%WINDOW_BITS==0;

  chain_t  chain;
  uint32_t add=0, negate=0, order[9], scalar[9];

  Planning::setOrder(order);
  order[8]=0;

  // UNIFORM BUCKET SUPPORT:
  // add the pregenerated random scalar to each MSM scalar term.  If the 
  // result is greater than the curve order, subtract the curve order.

  scalar[0]=chain.add(lo.x, rLo.x);
  scalar[1]=chain.add(lo.y, rLo.y);
  scalar[2]=chain.add(lo.z, rLo.z);
  scalar[3]=chain.add(lo.w, rLo.w);
  scalar[4]=chain.add(hi.x, rHi.x);
  scalar[5]=chain.add(hi.y, rHi.y);
  scalar[6]=chain.add(hi.z, rHi.z);
  scalar[7]=chain.add(hi.w, rHi.w);
  scalar[8]=0;
  if(scalar[7]>order[7])
    mp_sub<9>(scalar, scalar, order);
  
  if constexpr (EXACT) {
    const uint32_t EXPONENT_HIGH_BIT=1U<<(EXPONENT_BITS-7*32-1);

    if((scalar[7] & EXPONENT_HIGH_BIT)!=0) {
      mp_sub<9>(scalar, order, scalar);
      negate=SIGN_MASK;
    }
  }
  else {
    const uint32_t EXPONENT_MOD=(0x80000000ull<<WINDOWS*WINDOW_BITS-256)/(EXPONENT_HIGH_WORD+1);

    mp_scale<9>(order, order, pointIndex % EXPONENT_MOD);
    mp_add<9>(scalar, scalar, order);
  }

  if constexpr (WINDOW_BITS==16) 
    slice16(buckets, scalar);
  else if constexpr (WINDOW_BITS==22) 
    slice22(buckets, scalar);
  else if constexpr (WINDOW_BITS==23) 
    slice23(buckets, scalar);
  else {
    #pragma unroll
    for(int32_t i=0;i<WINDOWS;i++)
      buckets[i]=0;
  }

  #pragma unroll
  for(int32_t i=0;i<WINDOWS;i++) {
    buckets[i]+=add;
    add=(buckets[i]>BUCKET_MASK+1) ? 1 : 0;
    if((buckets[i] & WINDOW_MASK)==0)
      buckets[i]=0xFFFFFFFF;
    else if(buckets[i]<=BUCKET_MASK+1)
      buckets[i]=buckets[i]-1 ^ negate;
    else
      buckets[i]=buckets[i] ^ (SIGN_MASK + WINDOW_MASK) ^ negate;
  }
}

template<class Planning>
__global__ void zeroCounters(PlanningLayout layout) {
  uint32_t  globalTID=blockIdx.x*blockDim.x + threadIdx.x, globalStride=blockDim.x*gridDim.x;

  for(uint32_t i=globalTID;i<layout.binCount;i+=globalStride) 
    layout.binCounts[i]=0;

  if(globalTID<256) {
    layout.bigBinCounts[globalTID]=0;
    layout.overflowCount[0]=0;
    layout.bucketSizeCounts[globalTID]=0;
    layout.bucketSizeNexts[globalTID]=0;
  }

  for(uint32_t i=globalTID;i<Planning::BUCKET_COUNT;i+=globalStride)
    layout.bucketOverflowCounts[i]=0;
}

__device__ __forceinline__ void store_global_u32(uint32_t* ptr, uint32_t value) {
  asm volatile("st.global.cg.u32 [%0],%1;" : : "l"(ptr), "r"(value));
}

__device__ __forceinline__ void store_global_u2(uint2* ptr, uint2 value) {
  asm volatile("st.global.cg.v2.u32 [%0],{%1,%2};" : : "l"(ptr), "r"(value.x), "r"(value.y));
}

template<class Planning>
__global__ void partitionIntoBins(PlanningLayout layout, uint4* scalars, uint32_t startPoint, uint32_t stopPoint) {
  const uint32_t WINDOWS=Planning::WINDOWS, INDEX_BITS=Planning::INDEX_BITS, INDEX_MASK=(1u<<INDEX_BITS)-1, BINS_PER_GROUP=Planning::BINS_PER_GROUP;
  const uint32_t BIN_BITS=Planning::BIN_BITS, BIN_MASK=(1u<<BIN_BITS)-1, BUCKET_BITS=Planning::BUCKET_BITS, BUCKET_MASK=(1u<<BUCKET_BITS)-1;
  const uint32_t SIGN_BIT=Planning::SIGN_BIT;

  uint32_t globalTID=blockIdx.x*blockDim.x + threadIdx.x;
  uint4    lo, hi;
  uint32_t buckets[16];
  uint32_t group, groupIndex, bucket, append, bin, offset, groupCount=layout.groupCount, pointsPerBin=layout.pointsPerBinGroup, maxPointCount=layout.maxPointCount;
  
  __shared__ uint32_t localBigBinCounts[256];

  if(threadIdx.x<256)
    localBigBinCounts[threadIdx.x]=0;

  __syncthreads();

  for(uint32_t i=startPoint + globalTID;i<stopPoint;i+=blockDim.x*gridDim.x) {
    group=i>>INDEX_BITS;
    groupIndex=i & INDEX_MASK;

    lo=scalars[2*i+0];
    hi=scalars[2*i+1];
 
    signedDigits<Planning>(buckets, lo, hi, i);

    #pragma unroll
    for(int32_t j=0;j<WINDOWS;j++) {
      if(buckets[j]!=0xFFFFFFFF) {
        bucket=buckets[j] & BUCKET_MASK;
        bin=bucket>>BIN_BITS;
        atomicAdd(&localBigBinCounts[bucket>>BUCKET_BITS-8], 1);
        offset=atomicAdd(&layout.binCounts[group*BINS_PER_GROUP + bin], 1);
        if(offset<pointsPerBin) {
          append=(bucket & BIN_MASK)*2 + (buckets[j]>>SIGN_BIT) << 4;
          append=(append + j<<INDEX_BITS) + groupIndex;
          store_global_u32(&layout.binnedPointIndexes[(bin*groupCount + group)*pointsPerBin + offset], append);
        }
        else {
          offset=atomicAdd(layout.overflowCount, 1);
          store_global_u2(&layout.overflow[offset], make_uint2(buckets[j], i + j*maxPointCount));
          atomicAdd(&layout.bucketOverflowCounts[bucket], 1);
        }
      }
    }
  }   

  __syncthreads();

  if(threadIdx.x<256) 
    atomicAdd(&layout.bigBinCounts[threadIdx.x], localBigBinCounts[threadIdx.x]);
}

template<class Planning>
__global__ void partitionIntoBins(PlanningLayout layout, uint4* scalars, uint4* randomScalars, uint32_t startPoint, uint32_t stopPoint) {
  const uint32_t WINDOWS=Planning::WINDOWS, INDEX_BITS=Planning::INDEX_BITS, INDEX_MASK=(1u<<INDEX_BITS)-1, BINS_PER_GROUP=Planning::BINS_PER_GROUP;
  const uint32_t BIN_BITS=Planning::BIN_BITS, BIN_MASK=(1u<<BIN_BITS)-1, BUCKET_BITS=Planning::BUCKET_BITS, BUCKET_MASK=(1u<<BUCKET_BITS)-1;
  const uint32_t SIGN_BIT=Planning::SIGN_BIT;

  uint32_t globalTID=blockIdx.x*blockDim.x + threadIdx.x;
  uint4    lo, hi, rLo, rHi;
  uint32_t buckets[16];
  uint32_t group, groupIndex, bucket, append, bin, offset, groupCount=layout.groupCount, pointsPerBin=layout.pointsPerBinGroup, maxPointCount=layout.maxPointCount;
  
  __shared__ uint32_t localBigBinCounts[256];

  if(threadIdx.x<256)
    localBigBinCounts[threadIdx.x]=0;

  __syncthreads();

  for(uint32_t i=startPoint + globalTID;i<stopPoint;i+=blockDim.x*gridDim.x) {
    group=i>>INDEX_BITS;
    groupIndex=i & INDEX_MASK;

    lo=scalars[2*i+0];
    hi=scalars[2*i+1];
    rLo=randomScalars[2*i+0];
    rHi=randomScalars[2*i+1];
 
    signedDigits<Planning>(buckets, lo, hi, rLo, rHi, i);

    #pragma unroll
    for(int32_t j=0;j<WINDOWS;j++) {
      if(buckets[j]!=0xFFFFFFFF) {
        bucket=buckets[j] & BUCKET_MASK;
        bin=bucket>>BIN_BITS;
        atomicAdd(&localBigBinCounts[bucket>>BUCKET_BITS-8], 1);
        offset=atomicAdd(&layout.binCounts[group*BINS_PER_GROUP + bin], 1);
        if(offset<pointsPerBin) {
          append=(bucket & BIN_MASK)*2 + (buckets[j]>>SIGN_BIT) << 4;
          append=(append + j<<INDEX_BITS) + groupIndex;
          store_global_u32(&layout.binnedPointIndexes[(bin*groupCount + group)*pointsPerBin + offset], append);
        }
        else {
          offset=atomicAdd(layout.overflowCount, 1);
          store_global_u2(&layout.overflow[offset], make_uint2(buckets[j], i + j*maxPointCount));
          atomicAdd(&layout.bucketOverflowCounts[bucket], 1);
        }
      }
    }
  }   

  __syncthreads();

  if(threadIdx.x<256) 
    atomicAdd(&layout.bigBinCounts[threadIdx.x], localBigBinCounts[threadIdx.x]);
}

template<class Planning>
__global__ void computeBinOffsets(PlanningLayout layout) {
  const uint32_t BINS_PER_GROUP=Planning::BINS_PER_GROUP;

  uint32_t       warp=threadIdx.x>>5;
  uint32_t       indexes=BINS_PER_GROUP/256, base=indexes*blockIdx.x;
  uint32_t       blockBase=0, threadBase, sum, groupCount=layout.groupCount;

  __shared__ uint32_t warpSums[8];

  // fixed geometry, 256 blocks @ 256 threads

  sum=0;
  if(threadIdx.x<blockIdx.x)
    sum=layout.bigBinCounts[threadIdx.x];
  warpSums[warp]=warpSum(sum);

  __syncthreads();

  blockBase=0;
  for(uint32_t index=threadIdx.x;index<indexes;index+=blockDim.x) {
    #pragma unroll
    for(uint32_t i=0;i<8;i++) 
      blockBase=blockBase + warpSums[i];

    sum=0;
    for(uint32_t group=0;group<groupCount;group++) 
      sum=sum+layout.binCounts[group*BINS_PER_GROUP + base + index]; 

    __syncthreads();

    threadBase=prefixSum<256>(warpSums, sum);
    layout.binOffsets[base + index]=blockBase + threadBase - sum;
  }
}

template<class Planning>
__global__ void sortBins(PlanningLayout layout) {
  const uint32_t BIN_BITS=Planning::BIN_BITS, BIN_SIZE=1u<<BIN_BITS, BINS_PER_GROUP=Planning::BINS_PER_GROUP, WINDOWS=Planning::WINDOWS,
                 INDEX_BITS=Planning::INDEX_BITS, INDEX_MASK=(1u<<INDEX_BITS)-1;

  uint32_t  loaded, amount, offset, data, count, prefix, sign, window, bucketOverflowCount;
  uint32_t  groupCount=layout.groupCount, pointsPerBin=layout.pointsPerBinGroup, maxPointCount=layout.maxPointCount;
  uint32_t* sortTarget;

  __shared__ uint32_t localCounts[BIN_SIZE];
  __shared__ uint32_t localOffsets[BIN_SIZE];
  __shared__ uint32_t localBucketSizeCounts[256];
  __shared__ uint32_t destinationOffsets[BIN_SIZE];
  __shared__ uint32_t warpSums[4];
  __shared__ bool     overflow;

  extern __shared__ uint32_t localData[];    // required size: groupCount*PointsPerBinGroup*4

  // runs with a fixed geometry, BINS_PER_GROUP/64 blocks, at 256 threads per block

  // load bucket overflow counts
  // copy data to shared
  // load data into registers
  // add to counts
  // compute prefix sum
  // if no overflows, write data contiguously
  // else write data in 64 pieces

  localBucketSizeCounts[threadIdx.x]=0;

  __syncthreads();

  for(uint32_t bin=blockIdx.x*64;bin<blockIdx.x*64+64;bin++) {
    // initialize for the next bin
    overflow=false;
    if(threadIdx.x<BIN_SIZE) 
      localCounts[threadIdx.x]=0;

    __syncthreads();

    // load data and compute local bucket counts
    loaded=0;
    #pragma unroll 1
    for(int32_t group=0;group<groupCount;group++) {
      amount=umin(layout.binCounts[group*BINS_PER_GROUP + bin], pointsPerBin);
      #pragma unroll 1
      for(uint32_t i=threadIdx.x;i<amount;i+=blockDim.x) {
        data=layout.binnedPointIndexes[(bin*groupCount + group)*pointsPerBin + i];
        atomicAdd(&localCounts[data>>32-BIN_BITS], 1);
      }
      loaded+=amount;
    }

    if(threadIdx.x<BIN_SIZE) {
      bucketOverflowCount=layout.bucketOverflowCounts[bin*BIN_SIZE + threadIdx.x];
      if(bucketOverflowCount!=0)
        overflow=true;
    }

    __syncthreads();

    // prefix sum on the counts to get shared memory offsets
    count=0;
    if(threadIdx.x<BIN_SIZE)
      count=localCounts[threadIdx.x];
    prefix=prefixSum<BIN_SIZE>(warpSums, count);
    if(threadIdx.x<BIN_SIZE) {
      localOffsets[threadIdx.x]=prefix - count;
      offset=umin(255, count+bucketOverflowCount);
      atomicAdd(&localBucketSizeCounts[offset], 1);
    }

    __syncthreads();

    // load the data again (hopefully from L1) and sort
    #pragma unroll 1
    for(int32_t group=groupCount-1;group>=0;group--) {
      amount=umin(layout.binCounts[group*BINS_PER_GROUP + bin], pointsPerBin);
      #pragma unroll 1
      for(uint32_t i=threadIdx.x;i<amount;i+=blockDim.x) {
        data=layout.binnedPointIndexes[(bin*groupCount + group)*pointsPerBin + i];
        offset=atomicAdd(&localOffsets[data>>32-BIN_BITS], 1);
        sign=(data<<BIN_BITS) & 0x80000000;
        window=(data>>INDEX_BITS) & (1u<<BitSize<WINDOWS>::bitSize)-1;        
        localData[offset]=sign + window*maxPointCount + (group<<INDEX_BITS) + (data & INDEX_MASK);
      }
    }

    __syncthreads();

    // copy sorted data back to global memory
    sortTarget=layout.pointIndexes + layout.binOffsets[bin];
    if(!overflow) 
      copy(sortTarget, localData, loaded);
    else {
      // restore the offsets
      if(threadIdx.x<BIN_SIZE)
        localOffsets[threadIdx.x]=prefix - count;

      // if we have overflows, we copy the data in 64 pieces
      prefix=prefixSum<BIN_SIZE>(warpSums, count + bucketOverflowCount);
      if(threadIdx.x<BIN_SIZE)
        destinationOffsets[threadIdx.x]=prefix - count;

      __syncthreads();
      #pragma unroll 1
      for(uint32_t i=0;i<BIN_SIZE;i++) 
        copy(sortTarget + destinationOffsets[i], localData + localOffsets[i], localCounts[i]);
      __syncthreads();
    }

    if(threadIdx.x<BIN_SIZE) {
      layout.bucketCounts[bin*BIN_SIZE + threadIdx.x]=count + bucketOverflowCount;
      layout.bucketOffsets[bin*BIN_SIZE + threadIdx.x]=layout.binOffsets[bin] + prefix - (count + bucketOverflowCount);
    }
  }

  atomicAdd(&layout.bucketSizeCounts[threadIdx.x], localBucketSizeCounts[threadIdx.x]);
}

template<class Planning>
__global__ void processOverflows(PlanningLayout layout) {
  const uint32_t BUCKET_BITS=Planning::BUCKET_BITS, BUCKET_MASK=(1u<<BUCKET_BITS)-1, SIGN_BIT=Planning::SIGN_BIT, SIGN_MASK=1u<<SIGN_BIT;

  uint32_t globalTID=blockIdx.x*blockDim.x+threadIdx.x, globalStride=blockDim.x*gridDim.x;
  uint2    overflow;
  uint32_t bucket, bucketOffset, overflowOffset, overflowCount=layout.overflowCount[0];

  // overflow.x -> bucket
  // overflow.y -> point
  for(uint32_t i=globalTID;i<overflowCount;i+=globalStride) {
    overflow=layout.overflow[i];
    bucket=overflow.x & BUCKET_MASK;
    bucketOffset=layout.bucketOffsets[bucket];
    overflowOffset=atomicAdd(&layout.bucketOverflowCounts[bucket], 0xFFFFFFFF);
    store_global_u32(layout.pointIndexes + bucketOffset + overflowOffset - 1, overflow.y | (overflow.x & SIGN_MASK));
  }
}

template<class Planning>
__global__ void sortByCounts(PlanningLayout layout) {
  const uint32_t BUCKET_BITS=Planning::BUCKET_BITS, BUCKET_COUNT=1u<<BUCKET_BITS;

  uint32_t globalTID=blockIdx.x*blockDim.x+threadIdx.x, globalStride=blockDim.x*gridDim.x;
  uint32_t start, count, fullCount, prefix, base, offset;

  __shared__ uint32_t warpSums[8];
  __shared__ uint32_t bases[256];

  if(threadIdx.x<256) 
    count=layout.bucketSizeCounts[threadIdx.x];
  prefix=prefixSum<256>(warpSums, count); 
  if(threadIdx.x<256)
    bases[threadIdx.x]=prefix-count;

  for(uint32_t i=globalTID;i<BUCKET_COUNT;i+=globalStride) {
    start=layout.bucketOffsets[i];
    fullCount=layout.bucketCounts[i];
    count=umin(fullCount, 255);
    base=bases[count];
    offset=atomicAdd(&layout.bucketSizeNexts[count], 1);
    layout.sortedBucketIndexes[BUCKET_COUNT - base - offset - 1]=i;
    layout.sortedBucketOffsets[BUCKET_COUNT - base - offset - 1]=start;
    layout.sortedBucketCounts[BUCKET_COUNT - base - offset - 1]=fullCount;
  }
}
