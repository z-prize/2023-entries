/***

Copyright (c) 2023-2024, Yrrid Software, Inc. and Snarkify Inc.  All rights reserved.
Dual licensed under the MIT License or the Apache License, Version 2.0.
See LICENSE for details.

Author(s):  Niall Emmart

***/

__device__ __forceinline__ uint32_t BLS12381G1::qTerm(uint32_t x) {
  return x*0xFFFCFFFD;
}

__device__ __forceinline__ void BLS12381G1::setOrder(uint32_t* order) {
  order[0]=0x00000001;
  order[1]=0xFFFFFFFF;
  order[2]=0xFFFE5BFE;
  order[3]=0x53BDA402;
  order[4]=0x09A1D805;
  order[5]=0x3339D808;
  order[6]=0x299D7D48;
  order[7]=0x73EDA753;
}

__device__ __forceinline__ void BLS12381G1::initializeShared() {
  Field localN;

  setN(localN);
  if(threadIdx.x<10) {
    mp_scale<limbs>(localN, localN, threadIdx.x);
    storeShared(threadIdx.x*48, localN);
  }
  __syncthreads();
}

__device__ __forceinline__ void BLS12381G1::dump(const Field& f) {
  for(int32_t i=limbs-1;i>=0;i--) 
    printf("%08X", f[i]);
  printf("\n");
}

__device__ __forceinline__ void BLS12381G1::print(const Field& f) {
  Field local;

  fromMontgomery(local, f);
  for(int32_t i=limbs-1;i>=0;i--) 
    printf("%08X", local[i]);
  printf("\n");
}

__device__ __forceinline__ void BLS12381G1::load(Field& r, const void* ptr) {
  uint4* u4ptr=(uint4*)ptr;
  uint4  load0, load1, load2;

  load0=u4ptr[0];
  load1=u4ptr[1];
  load2=u4ptr[2];
  r[0]=load0.x; r[1]=load0.y; r[2]=load0.z; r[3]=load0.w;
  r[4]=load1.x; r[5]=load1.y; r[6]=load1.z; r[7]=load1.w;
  r[8]=load2.x; r[9]=load2.y; r[10]=load2.z; r[11]=load2.w;
}

__device__ __forceinline__ void BLS12381G1::store(void* ptr, const Field& f) {
  uint4* u4ptr=(uint4*)ptr;

  u4ptr[0]=make_uint4(f[0], f[1], f[2], f[3]);
  u4ptr[1]=make_uint4(f[4], f[5], f[6], f[7]);
  u4ptr[2]=make_uint4(f[8], f[9], f[10], f[11]);
}

__device__ __forceinline__ void BLS12381G1::loadUnaligned(Field& r, const void* ptr) {
  uint32_t* u32ptr=(uint32_t*)ptr;

  #pragma unroll
  for(uint32_t i=0;i<limbs;i++) 
    r[i]=u32ptr[i];
}

__device__ __forceinline__ void BLS12381G1::storeUnaligned(void* ptr, const Field& f) {
  uint32_t* u32ptr=(uint32_t*)ptr;

  #pragma unroll
  for(uint32_t i=0;i<limbs;i++) 
    u32ptr[i]=f[i];
}

__device__ __forceinline__ void BLS12381G1::loadShared(Field& r, const uint32_t addr) {
  uint4 load0, load1, load2;

  load0=load_shared_u4(addr);
  load1=load_shared_u4(addr+16);
  load2=load_shared_u4(addr+32);
  r[0]=load0.x; r[1]=load0.y; r[2]=load0.z; r[3]=load0.w;
  r[4]=load1.x; r[5]=load1.y; r[6]=load1.z; r[7]=load1.w;
  r[8]=load2.x; r[9]=load2.y; r[10]=load2.z; r[11]=load2.w;
}

__device__ __forceinline__ void BLS12381G1::storeShared(uint32_t addr, const Field& f) {
  store_shared_u4(addr, make_uint4(f[0], f[1], f[2], f[3]));
  store_shared_u4(addr+16, make_uint4(f[4], f[5], f[6], f[7]));
  store_shared_u4(addr+32, make_uint4(f[8], f[9], f[10], f[11]));
}

__device__ __forceinline__ void BLS12381G1::setZero(Field& r) {
  mp_zero<limbs>(r);
}

__device__ __forceinline__ void BLS12381G1::setOne(Field& r) {
  mp_one<limbs>(r);
}

__device__ __forceinline__ void BLS12381G1::setN(Field& r) {
  r[0]=0xFFFFAAAB; r[1]=0xB9FEFFFF; r[2]=0xB153FFFF; r[3]=0x1EABFFFE;
  r[4]=0xF6B0F624; r[5]=0x6730D2A0; r[6]=0xF38512BF; r[7]=0x64774B84;
  r[8]=0x434BACD7; r[9]=0x4B1BA7B6; r[10]=0x397FE69A; r[11]=0x1A0111EA;
}

__device__ __forceinline__ void BLS12381G1::setR(Field& r) {
  r[0]=0x0002FFFD; r[1]=0x76090000; r[2]=0xC40C0002; r[3]=0xEBF4000B;
  r[4]=0x53C758BA; r[5]=0x5F489857; r[6]=0x70525745; r[7]=0x77CE5853;
  r[8]=0xA256EC6D; r[9]=0x5C071A97; r[10]=0xFA80E493; r[11]=0x15F65EC3;
}

__device__ __forceinline__ void BLS12381G1::setRInv(Field& r) {
  r[0]=0x380B4820; r[1]=0xF4D38259; r[2]=0xD898FAFB; r[3]=0x7FE11274;
  r[4]=0x14956DC8; r[5]=0x343EA979; r[6]=0x58A88DE9; r[7]=0x1797AB14;
  r[8]=0x3C4F538B; r[9]=0xED5E6427; r[10]=0xE8FB0CE9; r[11]=0x14FEC701;
}

__device__ __forceinline__ void BLS12381G1::setRSquared(Field& r) {
  r[0]=0x1C341746; r[1]=0xF4DF1F34; r[2]=0x09D104F1; r[3]=0x0A76E6A6;
  r[4]=0x4C95B6D5; r[5]=0x8DE5476C; r[6]=0x939D83C0; r[7]=0x67EB88A9;
  r[8]=0xB519952D; r[9]=0x9A793E85; r[10]=0x92CAE3AA; r[11]=0x11988FE5;
}

__device__ __forceinline__ void BLS12381G1::setGeneratorX(Field& r) {
  r[0]=0xFD530C16; r[1]=0x5CB38790; r[2]=0x9976FFF5; r[3]=0x7817FC67;
  r[4]=0x143BA1C1; r[5]=0x154F95C7; r[6]=0xF3D0E747; r[7]=0xF0AE6ACD;
  r[8]=0x21DBF440; r[9]=0xEDCE6ECC; r[10]=0x9E0BFB75; r[11]=0x12017741;
}

__device__ __forceinline__ void BLS12381G1::setGeneratorY(Field& r) {
  r[0]=0x0CE72271; r[1]=0xBAAC93D5; r[2]=0x7918FD8E; r[3]=0x8C22631A;
  r[4]=0x570725CE; r[5]=0xDD595F13; r[6]=0x50405194; r[7]=0x51AC5829;
  r[8]=0xAD0059C0; r[9]=0x0E1C8C3F; r[10]=0x5008A26A; r[11]=0x0BBC3EFC;
}

__device__ __forceinline__ bool BLS12381G1::isZero(const Field& f) {
  uint32_t top;
  uint4    load;
  uint32_t mask=0;

  // Input assertion:  0 <= f < 2^384

  top=__funnelshift_r(f[0], f[1], 16)-1;
  if(((top & 0xFFFFFFF8) ^ 0xFFFFFFF8)!=0)
    return false;
  top=__umulhi(f[11], 0x0A)*48;
  load=load_shared_u4(top);
  mask=mask | (f[0] ^ load.x);
  mask=mask | (f[1] ^ load.y);
  mask=mask | (f[2] ^ load.z);
  mask=mask | (f[3] ^ load.w);
  load=load_shared_u4(top+16);
  mask=mask | (f[4] ^ load.x);
  mask=mask | (f[5] ^ load.y);
  mask=mask | (f[6] ^ load.z);
  mask=mask | (f[7] ^ load.w);
  load=load_shared_u4(top+32);
  mask=mask | (f[8] ^ load.x);
  mask=mask | (f[9] ^ load.y);
  mask=mask | (f[10] ^ load.z);
  mask=mask | (f[11] ^ load.w);
  return mask==0;

  // Output assertion: true, if and only if f mod N == 0
}

__device__ __forceinline__ void BLS12381G1::reduce(Field& r, const Field& f) {
  uint32_t top=__umulhi(f[11]>>12, 0x9D83)*48;
  chain_t  chain;
  uint4    load;

  // Input assertion:  0 <= f < 2^384

  // For any input value between 0 and 2^384-1, this routine will remove multiples
  // of N until the output is between 0 and 1.001*N.  In other words:
  //    r=f;
  //    while(r>1.001*N) 
  //      r=r-N;
  //    return r;

  load=load_shared_u4(top);
  r[0]=chain.sub(f[0], load.x);
  r[1]=chain.sub(f[1], load.y);
  r[2]=chain.sub(f[2], load.z);
  r[3]=chain.sub(f[3], load.w);
  load=load_shared_u4(top+16);
  r[4]=chain.sub(f[4], load.x);
  r[5]=chain.sub(f[5], load.y);
  r[6]=chain.sub(f[6], load.z);
  r[7]=chain.sub(f[7], load.w);
  load=load_shared_u4(top+32);
  r[8]=chain.sub(f[8], load.x);
  r[9]=chain.sub(f[9], load.y);
  r[10]=chain.sub(f[10], load.z);
  r[11]=chain.sub(f[11], load.w);

  // Output assertion: 0 <= r < 1.001*N
}

__device__ __forceinline__ void BLS12381G1::reduceFully(Field& r, const Field& f) {
  chain_t chain;
  Field   localN;

  // Input assertion:  0 <= f < 2^384
  
  reduce(r, f);
  if(r[11]>=0x1A0111EA) {
    setN(localN);
    if(!mp_sub_carry<limbs>(r, r, localN))
      mp_add<limbs>(r, r, localN);
  }

  // Output assertion: 0 <= r < N
}

__device__ __forceinline__ void BLS12381G1::setField(Field& r, const Field& f) {
  mp_copy<limbs>(r, f);
}

__device__ __forceinline__ void BLS12381G1::warpShuffleField(Field& r, const Field& f, const uint32_t lane) {
  #pragma unroll
  for(int32_t i=0;i<limbs;i++) 
    r[i]=__shfl_sync(0xFFFFFFFF, f[i], lane);
}

__device__ __forceinline__ void BLS12381G1::addN(Field& r, const Field& f) {
  chain_t chain;

  r[0]=chain.add(f[0], 0xFFFFAAAB);
  r[1]=chain.add(f[1], 0xB9FEFFFF);
  r[2]=chain.add(f[2], 0xB153FFFF);
  r[3]=chain.add(f[3], 0x1EABFFFE);
  r[4]=chain.add(f[4], 0xF6B0F624);
  r[5]=chain.add(f[5], 0x6730D2A0);
  r[6]=chain.add(f[6], 0xF38512BF);
  r[7]=chain.add(f[7], 0x64774B84);
  r[8]=chain.add(f[8], 0x434BACD7);
  r[9]=chain.add(f[9], 0x4B1BA7B6);
  r[10]=chain.add(f[10], 0x397FE69A);
  r[11]=chain.add(f[11], 0x1A0111EA);
}

__device__ __forceinline__ void BLS12381G1::add2N(Field& r, const Field& f) {
  chain_t chain;

  r[0]=chain.add(f[0], 0xFFFF5556);
  r[1]=chain.add(f[1], 0x73FDFFFF);
  r[2]=chain.add(f[2], 0x62A7FFFF);
  r[3]=chain.add(f[3], 0x3D57FFFD);
  r[4]=chain.add(f[4], 0xED61EC48);
  r[5]=chain.add(f[5], 0xCE61A541);
  r[6]=chain.add(f[6], 0xE70A257E);
  r[7]=chain.add(f[7], 0xC8EE9709);
  r[8]=chain.add(f[8], 0x869759AE);
  r[9]=chain.add(f[9], 0x96374F6C);
  r[10]=chain.add(f[10], 0x72FFCD34);
  r[11]=chain.add(f[11], 0x340223D4);
}

__device__ __forceinline__ void BLS12381G1::add3N(Field& r, const Field& f) {
  chain_t chain;

  r[0]=chain.add(f[0], 0xFFFF0001);
  r[1]=chain.add(f[1], 0x2DFCFFFF);
  r[2]=chain.add(f[2], 0x13FBFFFF);
  r[3]=chain.add(f[3], 0x5C03FFFC);
  r[4]=chain.add(f[4], 0xE412E26C);
  r[5]=chain.add(f[5], 0x359277E2);
  r[6]=chain.add(f[6], 0xDA8F383E);
  r[7]=chain.add(f[7], 0x2D65E28E);
  r[8]=chain.add(f[8], 0xC9E30686);
  r[9]=chain.add(f[9], 0xE152F722);
  r[10]=chain.add(f[10], 0xAC7FB3CE);
  r[11]=chain.add(f[11], 0x4E0335BE);
}

__device__ __forceinline__ void BLS12381G1::add4N(Field& r, const Field& f) {
  chain_t chain;

  r[0]=chain.add(f[0], 0xFFFEAAAC);
  r[1]=chain.add(f[1], 0xE7FBFFFF);
  r[2]=chain.add(f[2], 0xC54FFFFE);
  r[3]=chain.add(f[3], 0x7AAFFFFA);
  r[4]=chain.add(f[4], 0xDAC3D890);
  r[5]=chain.add(f[5], 0x9CC34A83);
  r[6]=chain.add(f[6], 0xCE144AFD);
  r[7]=chain.add(f[7], 0x91DD2E13);
  r[8]=chain.add(f[8], 0x0D2EB35D);
  r[9]=chain.add(f[9], 0x2C6E9ED9);
  r[10]=chain.add(f[10], 0xE5FF9A69);
  r[11]=chain.add(f[11], 0x680447A8);
}

__device__ __forceinline__ void BLS12381G1::add5N(Field& r, const Field& f) {
  chain_t chain;

  r[0]=chain.add(f[0], 0xFFFE5557);
  r[1]=chain.add(f[1], 0xA1FAFFFF);
  r[2]=chain.add(f[2], 0x76A3FFFE);
  r[3]=chain.add(f[3], 0x995BFFF9);
  r[4]=chain.add(f[4], 0xD174CEB4);
  r[5]=chain.add(f[5], 0x03F41D24);
  r[6]=chain.add(f[6], 0xC1995DBD);
  r[7]=chain.add(f[7], 0xF6547998);
  r[8]=chain.add(f[8], 0x507A6034);
  r[9]=chain.add(f[9], 0x778A468F);
  r[10]=chain.add(f[10], 0x1F7F8103);
  r[11]=chain.add(f[11], 0x82055993);
}

__device__ __forceinline__ void BLS12381G1::add(Field& r, const Field& a, const Field& b) {
  mp_add<limbs>(r, a, b);
}

__device__ __forceinline__ void BLS12381G1::sub(Field& r, const Field& a, const Field& b) {
  mp_sub<limbs>(r, a, b);
}

__device__ __forceinline__ void BLS12381G1::modmul(Field& r, const Field& a, const Field& b) {
  Field    localN;
  uint64_t evenOdd[limbs];
  bool     carry;

  setN(localN);
  carry=mp_mul_red_cl<BLS12381G1, limbs>(evenOdd, a, b, localN);
  mp_merge_cl<limbs>(r, evenOdd, carry);
}

__device__ __forceinline__ void BLS12381G1::modsqr(Field& r, const Field& a) {
  Field    localN;
  uint64_t evenOdd[limbs];
  uint32_t t[limbs];
  bool     carry;

  setN(localN);
  carry=mp_sqr_red_cl<BLS12381G1, limbs>(evenOdd, t, a, localN);
  mp_merge_cl<limbs>(r, evenOdd, carry);
}

__device__ __forceinline__ void BLS12381G1::modinv(Field& r, const Field& f) {
  Field localRInv;
  Field localN;

  setRInv(localRInv);
  modmul(r, f, localRInv);
  setN(localN);
  mp_inverse<limbs>(r, r, localN);
}

__device__ __forceinline__ void BLS12381G1::toMontgomery(Field& r, const Field& f) {
  Field localRSquared;

  setRSquared(localRSquared);
  modmul(r, f, localRSquared);

  // Output assertion:  0 <= r < 2N
}

__device__ __forceinline__ void BLS12381G1::fromMontgomery(Field& r, const Field& f) {
  Field localOne;

  setOne(localOne);
  modmul(r, f, localOne);

  // I believe the output assertion for this is R<N, but it's not used, so it's not necessary to prove
}

// Extended Jacobian Representation
__device__ __forceinline__ void BLS12381G1::normalize(PointXY& r, const PointXYZZ& pt) {
  Field t;

  if(isZero(pt.zz)) {
    setZero(r.x);
    setZero(r.y);
    return;
  }
  modinv(t, pt.zzz);                      // t = zzz^-1
  modmul(r.y, pt.y, t);                   // y' = y * t
  modmul(t, t, pt.zz);                    // t = zz * t
  modsqr(t, t);                           // t = t^2
  modmul(r.x, pt.x, t);                   // x' = x * t

  reduceFully(r.x, r.x);
  reduceFully(r.y, r.y);

  // Output assertion:  0 <= r.x, r.y < N
}

__device__ __forceinline__ void BLS12381G1::negateXY(PointXY& pt) {
  Field localN;

  setN(localN);
  mp_sub<limbs>(pt.y, localN, pt.y);
}

__device__ __forceinline__ void BLS12381G1::initializeXYZZ(PointXYZZ& acc) {
  setZero(acc.x);
  setZero(acc.y);
  setZero(acc.zz);
  setZero(acc.zzz);
}

__device__ __forceinline__ void BLS12381G1::negateXYZZ(PointXYZZ& pt) {
  // In/Out assertions: 0 <= acc.x < 1.001 * N
  //                    0 <= acc.y < 4.999 * N
  //                    0 <= acc.zz < 1.41 * N;
  //                    0 <= acc.zzz < 1.30 * N

  mp_neg<limbs>(pt.y, pt.y);
  add5N(pt.y, pt.y);
  reduce(pt.y, pt.y);
}

__device__ __forceinline__ void BLS12381G1::setXY(PointXYZZ& acc, const PointXY& pt) {
  mp_copy<limbs>(acc.x, pt.x);
  mp_copy<limbs>(acc.y, pt.y);
  setR(acc.zz);
  setR(acc.zzz);
}

__device__ __forceinline__ void BLS12381G1::setXYZZ(PointXYZZ& acc, const PointXYZZ& pt) {
  mp_copy<limbs>(acc.x, pt.x);
  mp_copy<limbs>(acc.y, pt.y);
  mp_copy<limbs>(acc.zz, pt.zz);
  mp_copy<limbs>(acc.zzz, pt.zzz);
}

__device__ __forceinline__ void BLS12381G1::warpShuffleXYZZ(PointXYZZ& r, const PointXYZZ& pt, const uint32_t lane) {
  warpShuffleField(r.x, pt.x, lane);
  warpShuffleField(r.y, pt.y, lane);
  warpShuffleField(r.zz, pt.zz, lane);
  warpShuffleField(r.zzz, pt.zzz, lane);
}

__device__ __forceinline__ void BLS12381G1::doubleXY(PointXYZZ& acc, const PointXY& pt) {
  Field T0, T1;

  // Input Assertion: 0 <= pt.x <= N
  //                  0 <= pt.y <= N

  add(T0, pt.y, pt.y);
  modsqr(acc.zz, T0);
  modmul(acc.zzz, acc.zz, T0);

  modmul(T0, pt.x, acc.zz);
  modsqr(acc.x, pt.x);
  modmul(acc.y, pt.y, acc.zzz);

  add(T1, acc.x, acc.x);
  add(T1, T1, acc.x);

  modsqr(acc.x, T1);
  sub(acc.x, acc.x, T0);
  sub(acc.x, acc.x, T0);
  add3N(acc.x, acc.x);
  reduce(acc.x, acc.x);             // ensure acc.x < 1.001N

  sub(T0, T0, acc.x);
  add2N(T0, T0);

  modmul(T1, T1, T0);
  sub(acc.y, T1, acc.y);
  add2N(acc.y, acc.y);

  // Output assertion:  0 <= acc.x < 1.001 * N
  //                    0 <= acc.y < 4.999 * N
  //                    0 <= acc.zz < 1.41 * N;
  //                    0 <= acc.zzz < 1.30 * N
}

__device__ __forceinline__ void BLS12381G1::doubleXYZZ(PointXYZZ& acc, const PointXYZZ& pt) {
  Field T0, T1;

  // Input assertion: 0 <= pt.x < 1.001 * N
  //                  0 <= pt.y < 4.999 * N
  //                  0 <= pt.zz < 1.41 * N;
  //                  0 <= pt.zzz < 1.30 * N

  if(isZero(pt.zz)) {
    setXYZZ(acc, pt);
    return;
  }

  add(T1, pt.y, pt.y);
  reduce(T1, T1);                   // ensure T1 < 1.001*N
  modsqr(T0, T1);
  modmul(T1, T0, T1);

  modmul(acc.zz, pt.zz, T0);
  modmul(acc.zzz, pt.zzz, T1);

  modmul(T0, pt.x, T0);
  modsqr(acc.x, pt.x);
  modmul(acc.y, pt.y, T1);

  add(T1, acc.x, acc.x);
  add(T1, acc.x, T1);

  modsqr(acc.x, T1);
  sub(acc.x, acc.x, T0);
  sub(acc.x, acc.x, T0);
  add3N(acc.x, acc.x);
  reduce(acc.x, acc.x);             // ensure acc.x < 1.001N
  
  sub(T0, T0, acc.x);
  add2N(T0, T0);

  modmul(T1, T1, T0);
  sub(acc.y, T1, acc.y);
  add2N(acc.y, acc.y);

  // Output assertion:  0 <= acc.x < 1.001 * N
  //                    0 <= acc.y < 4.999 * N
  //                    0 <= acc.zz < 1.41 * N;
  //                    0 <= acc.zzz < 1.30 * N
}

__device__ __forceinline__ void BLS12381G1::accumulateXY(PointXYZZ& acc, const PointXY& pt) {
  Field T0, T1, T2;

  // Input assertion:  0 <= acc.x < 1.001 * N
  //                   0 <= acc.y < 4.999 * N
  //                   0 <= acc.zz < 1.41 * N;
  //                   0 <= acc.zzz < 1.30 * N
  //
  //                   0 <= pt.x <= N
  //                   0 <= pt.y <= N

  if(isZero(acc.zz)) {
    setXY(acc, pt);
    return;
  }

  modmul(T0, pt.x, acc.zz);
  modmul(T1, pt.y, acc.zzz);
  sub(T0, T0, acc.x);
  add2N(T0, T0);
  sub(T1, T1, acc.y);
  add5N(T1, T1);

/*
printf("T0=");
for(int i=11;i>=0;i--) printf("%08X", T0[i]); printf("\n");
printf("T1=");
for(int i=11;i>=0;i--) printf("%08X", T1[i]); printf("\n");
*/

  if(isZero(T0) && isZero(T1)) {
    doubleXY(acc, pt);
    return;
  }
  
  modsqr(T2, T0);
  modmul(T0, T2, T0);

  modmul(acc.zz, acc.zz, T2);
  modmul(acc.zzz, acc.zzz, T0);
  modmul(T2, acc.x, T2);
  modmul(acc.y, acc.y, T0);

  modsqr(acc.x, T1);
  add(T0, T0, T2);
  add(T0, T0, T2);
  sub(acc.x, acc.x, T0);  
  add5N(acc.x, acc.x);
  reduce(acc.x, acc.x);             // ensure acc.x < 1.001N

  sub(T2, T2, acc.x);
  add2N(T2, T2);
  modmul(T2, T1, T2);
  sub(acc.y, T2, acc.y);
  add2N(acc.y, acc.y);

  // Output assertion:  0 <= acc.x < 1.001 * N
  //                    0 <= acc.y < 4.999 * N
  //                    0 <= acc.zz < 1.41 * N;
  //                    0 <= acc.zzz < 1.30 * N
}

__device__ __forceinline__ void BLS12381G1::accumulateXYZZ(PointXYZZ& acc, const PointXYZZ& pt) {
  Field T0, T1, T2;

  // Input assertion:  0 <= acc.x < 1.001 * N
  //                   0 <= acc.y < 4.999 * N
  //                   0 <= acc.zz < 1.41 * N;
  //                   0 <= acc.zzz < 1.30 * N
  //
  //                   0 <= pt.x < 1.001 * N
  //                   0 <= pt.y < 4.999 * N
  //                   0 <= pt.zz < 1.41 * N;
  //                   0 <= pt.zzz < 1.30 * N

  if(isZero(pt.zz))
    return;
  if(isZero(acc.zz)) {
    setXYZZ(acc, pt);
    return;
  }

  modmul(acc.x, acc.x, pt.zz);
  modmul(acc.y, acc.y, pt.zzz);
  modmul(T0, pt.x, acc.zz);
  modmul(T1, pt.y, acc.zzz);
  sub(T0, T0, acc.x);
  add2N(T0, T0);
  sub(T1, T1, acc.y);
  add2N(T1, T1);

  if(isZero(T0) && isZero(T1)) {
    doubleXYZZ(acc, pt);
    return;
  }
  
  modsqr(T2, T0);
  modmul(T0, T0, T2);

  modmul(acc.zz, acc.zz, T2);
  modmul(acc.zz, acc.zz, pt.zz);
  modmul(acc.zzz, acc.zzz, T0);
  modmul(acc.zzz, acc.zzz, pt.zzz);

  modmul(T2, acc.x, T2);
  modmul(acc.y, acc.y, T0);

  modsqr(acc.x, T1);
  add(T0, T0, T2);
  add(T0, T0, T2);
  sub(acc.x, acc.x, T0);
  add5N(acc.x, acc.x);
  reduce(acc.x, acc.x);             // ensure acc.x < 1.001N

  sub(T2, T2, acc.x);
  add2N(T2, T2);
  modmul(T2, T1, T2);
  sub(acc.y, T2, acc.y);
  add2N(acc.y, acc.y);

  // Output assertion:  0 <= acc.x < 1.001 * N
  //                    0 <= acc.y < 4.999 * N
  //                    0 <= acc.zz < 1.41 * N;
  //                    0 <= acc.zzz < 1.30 * N
}
