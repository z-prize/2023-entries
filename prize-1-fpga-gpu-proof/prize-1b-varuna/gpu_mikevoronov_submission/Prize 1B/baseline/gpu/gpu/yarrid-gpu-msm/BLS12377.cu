/***

Copyright (c) 2023-2024, Yrrid Software, Inc. and Snarkify Inc.  All rights reserved.
Dual licensed under the MIT License or the Apache License, Version 2.0.
See LICENSE for details.

Author(s):  Niall Emmart

***/

__constant__ __device__ uint32_t zeroConstant=0;

__device__ __forceinline__ uint32_t BLS12377G1::qTerm(uint32_t x) {
  return zeroConstant - x;
}

__device__ __forceinline__ void BLS12377G1::setOrder(uint32_t* order) {
  order[0]=0x00000001;
  order[1]=0x0A118000;
  order[2]=0xD0000001;
  order[3]=0x59AA76FE;
  order[4]=0x5C37B001;
  order[5]=0x60B44D1E;
  order[6]=0x9A2CA556;
  order[7]=0x12AB655E;
}

__device__ __forceinline__ void BLS12377G1::initializeShared() {
  Field localN;

  setN(localN);
  if(threadIdx.x<16) {
    mp_scale<limbs>(localN, localN, threadIdx.x);
    storeShared(threadIdx.x*48, localN);
  }
  __syncthreads();
}

__device__ __forceinline__ void BLS12377G1::print(const Field& f) {
  Field local;

  fromMontgomery(local, f);
  for(int32_t i=limbs-1;i>=0;i--) 
    printf("%08X", local[i]);
  printf("\n");
}

__device__ __forceinline__ void BLS12377G1::load(Field& r, const void* ptr) {
  uint4* u4ptr=(uint4*)ptr;
  uint4  load0, load1, load2;

  load0=u4ptr[0];
  load1=u4ptr[1];
  load2=u4ptr[2];
  r[0]=load0.x; r[1]=load0.y; r[2]=load0.z; r[3]=load0.w;
  r[4]=load1.x; r[5]=load1.y; r[6]=load1.z; r[7]=load1.w;
  r[8]=load2.x; r[9]=load2.y; r[10]=load2.z; r[11]=load2.w;
}

__device__ __forceinline__ void BLS12377G1::store(void* ptr, const Field& f) {
  uint4* u4ptr=(uint4*)ptr;

  u4ptr[0]=make_uint4(f[0], f[1], f[2], f[3]);
  u4ptr[1]=make_uint4(f[4], f[5], f[6], f[7]);
  u4ptr[2]=make_uint4(f[8], f[9], f[10], f[11]);
}

__device__ __forceinline__ void BLS12377G1::loadUnaligned(Field& r, const void* ptr) {
  uint32_t* u32ptr=(uint32_t*)ptr;

  #pragma unroll
  for(uint32_t i=0;i<limbs;i++) 
    r[i]=u32ptr[i];
}

__device__ __forceinline__ void BLS12377G1::storeUnaligned(void* ptr, const Field& f) {
  uint32_t* u32ptr=(uint32_t*)ptr;

  #pragma unroll
  for(uint32_t i=0;i<limbs;i++) 
    u32ptr[i]=f[i];
}

__device__ __forceinline__ void BLS12377G1::loadShared(Field& r, const uint32_t addr) {
  uint4 load0, load1, load2;

  load0=load_shared_u4(addr);
  load1=load_shared_u4(addr+16);
  load2=load_shared_u4(addr+32);
  r[0]=load0.x; r[1]=load0.y; r[2]=load0.z; r[3]=load0.w;
  r[4]=load1.x; r[5]=load1.y; r[6]=load1.z; r[7]=load1.w;
  r[8]=load2.x; r[9]=load2.y; r[10]=load2.z; r[11]=load2.w;
}

__device__ __forceinline__ void BLS12377G1::storeShared(uint32_t addr, const Field& f) {
  store_shared_u4(addr, make_uint4(f[0], f[1], f[2], f[3]));
  store_shared_u4(addr+16, make_uint4(f[4], f[5], f[6], f[7]));
  store_shared_u4(addr+32, make_uint4(f[8], f[9], f[10], f[11]));
}

__device__ __forceinline__ void BLS12377G1::setZero(Field& r) {
  mp_zero<limbs>(r);
}

__device__ __forceinline__ void BLS12377G1::setOne(Field& r) {
  mp_one<limbs>(r);
}

__device__ __forceinline__ void BLS12377G1::setN(Field& r) {
  r[0]=0x00000001; r[1]=0x8508C000; r[2]=0x30000000; r[3]=0x170B5D44;
  r[4]=0xBA094800; r[5]=0x1EF3622F; r[6]=0x00F5138F; r[7]=0x1A22D9F3;
  r[8]=0x6CA1493B; r[9]=0xC63B05C0; r[10]=0x17C510EA; r[11]=0x01AE3A46;
}

__device__ __forceinline__ void BLS12377G1::set2N(Field& r) {
  r[0]=0x00000002; r[1]=0x0A118000; r[2]=0x60000001; r[3]=0x2E16BA88;
  r[4]=0x74129000; r[5]=0x3DE6C45F; r[6]=0x01EA271E; r[7]=0x3445B3E6;
  r[8]=0xD9429276; r[9]=0x8C760B80; r[10]=0x2F8A21D5; r[11]=0x035C748C;
}

__device__ __forceinline__ void BLS12377G1::setR(Field& r) {
  r[0]=0xFFFFFF68; r[1]=0x02CDFFFF; r[2]=0x7FFFFFB1; r[3]=0x51409F83;
  r[4]=0x8A7D3FF2; r[5]=0x9F7DB3A9; r[6]=0x6E7C6305; r[7]=0x7B4E97B7;
  r[8]=0x803C84E8; r[9]=0x4CF495BF; r[10]=0xE2FDF49A; r[11]=0x008D6661;
}

__device__ __forceinline__ void BLS12377G1::setRInv(Field& r) {
  r[0]=0x451269E8; r[1]=0xEF129093; r[2]=0xE65839F5; r[3]=0x6E20BBCD;
  r[4]=0xA5582C93; r[5]=0x852E3C88; r[6]=0xF7F2E657; r[7]=0xEEAAF41D;
  r[8]=0xA4C49351; r[9]=0xEB89746C; r[10]=0x436B0736; r[11]=0x014212FC;
}

__device__ __forceinline__ void BLS12377G1::setRSquared(Field& r) {
  r[0]=0x9400CD22; r[1]=0xB786686C; r[2]=0xB00431B1; r[3]=0x0329FCAA;
  r[4]=0x62D6B46D; r[5]=0x22A5F111; r[6]=0x827DC3AC; r[7]=0xBFDF7D03;
  r[8]=0x41790BF9; r[9]=0x837E92F0; r[10]=0x1E914B88; r[11]=0x006DFCCB;
}

__device__ __forceinline__ void BLS12377G1::setGeneratorX(Field& r) {
  r[0]=0x772451F4; r[1]=0x260F33B9; r[2]=0x169D5658; r[3]=0xC54DD773; 
  r[4]=0x69A510DD; r[5]=0x5C1551C4; r[6]=0x425E1698; r[7]=0x761662E4;
  r[8]=0x6F065272; r[9]=0xC97D78CC; r[10]=0xB361FD4D; r[11]=0x00A41206;
}

__device__ __forceinline__ void BLS12377G1::setGeneratorY(Field& r) {
 r[0]=0xB8CB81F3; r[1]=0x8193961F; r[2]=0x5F44ADB8; r[3]=0x00638D4C;
 r[4]=0xD4DAF54A; r[5]=0xFAFAF3DA; r[6]=0xD655CD18; r[7]=0xC27849E2; 
 r[8]=0x01D52814; r[9]=0x2EC3DDB4; r[10]=0x26303C71; r[11]=0x007DA933;
}

__device__ __forceinline__ void BLS12377G1::setK(Field& r) {
  r[0]=0x44A706C6; r[1]=0xF24B7E84; r[2]=0x80FAA8FA; r[3]=0xEAE02375;
  r[4]=0x7EF38FA5; r[5]=0x0F4D7CF2; r[6]=0xC5F2BB26; r[7]=0x5597097D; 
  r[8]=0x0D95A93E; r[9]=0x8BF6C1DD; r[10]=0xFBFF628A; r[11]=0x01784602; 
}

__device__ __forceinline__ void BLS12377G1::setF(Field& r) {
  r[0]=0xEAD605E4; r[1]=0xFB7C0AB8; r[2]=0xE2752C7A; r[3]=0x363B93D1; 
  r[4]=0x144BE536; r[5]=0xD2246835; r[6]=0x429F2077; r[7]=0xEB8BF6DD; 
  r[8]=0x0234D423; r[9]=0x3A9EFDA2; r[10]=0xBA0C72A4; r[11]=0x013C7943;
}

__device__ __forceinline__ void BLS12377G1::setS(Field& r) {
  r[0]=0xD831A0A2; r[1]=0x6DE9CA9A; r[2]=0x0EB51C8E; r[3]=0xA8AFF422;
  r[4]=0x93607959; r[5]=0x5A76E6C5; r[6]=0xD3806339; r[7]=0x57F38956; 
  r[8]=0xA0C6FE30; r[9]=0xC50636F4; r[10]=0xF52BFD14; r[11]=0x018F4652;
}

__device__ __forceinline__ bool BLS12377G1::isZero(const Field& f) {
  uint32_t top;
  uint4    load;
  uint32_t mask=0;

  // Input assertion:  0 <= f <= 16N

  if((f[0] & 0xFFFFFFF0)!=0)
    return false;
  top=__umulhi(f[11], 0x99)*48;
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

__device__ __forceinline__ void BLS12377G1::reduce(Field& r, const Field& f) {
  uint32_t top=__umulhi(f[11]>>8, 0x9854)*48;
  chain_t  chain;
  uint4    load;

  // Input assertion: 0 <= f <= 16N

  // For any input value between 0 and 16N, this routine will remove multiples
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

  // Output assertion: 0 <= r <= 1.001 * N
}

__device__ __forceinline__ void BLS12377G1::reduceFully(Field& r, const Field& f) {
  chain_t chain;
  Field   localN;
  
  // Input assertion: 0 <= f <= 16N

  reduce(r, f);
  if(r[11]>=0x01AE3A46) {
    setN(localN);
    if(!mp_sub_carry<limbs>(r, r, localN))
      mp_add<limbs>(r, r, localN);
  }

  // Output assertion: 0 <= r < N
}

__device__ __forceinline__ void BLS12377G1::setField(Field& r, const Field& f) {
  mp_copy<limbs>(r, f);
}

__device__ __forceinline__ void BLS12377G1::warpShuffleField(Field& r, const Field& f, const uint32_t lane) {
  #pragma unroll
  for(int32_t i=0;i<limbs;i++) 
    r[i]=__shfl_sync(0xFFFFFFFF, f[i], lane);
}


__device__ __forceinline__ void BLS12377G1::addN(Field& r, const Field& f) {
  chain_t chain;

  r[0]=chain.add(f[0], 0x00000001);
  r[1]=chain.add(f[1], 0x8508C000);
  r[2]=chain.add(f[2], 0x30000000);
  r[3]=chain.add(f[3], 0x170B5D44);
  r[4]=chain.add(f[4], 0xBA094800);
  r[5]=chain.add(f[5], 0x1EF3622F);
  r[6]=chain.add(f[6], 0x00F5138F);
  r[7]=chain.add(f[7], 0x1A22D9F3);
  r[8]=chain.add(f[8], 0x6CA1493B);
  r[9]=chain.add(f[9], 0xC63B05C0);
  r[10]=chain.add(f[10], 0x17C510EA);
  r[11]=chain.add(f[11], 0x01AE3A46);
}

__device__ __forceinline__ void BLS12377G1::add2N(Field& r, const Field& f) {
  chain_t chain;

  r[0]=chain.add(f[0], 0x00000002);
  r[1]=chain.add(f[1], 0x0A118000);
  r[2]=chain.add(f[2], 0x60000001);
  r[3]=chain.add(f[3], 0x2E16BA88);
  r[4]=chain.add(f[4], 0x74129000);
  r[5]=chain.add(f[5], 0x3DE6C45F);
  r[6]=chain.add(f[6], 0x01EA271E);
  r[7]=chain.add(f[7], 0x3445B3E6);
  r[8]=chain.add(f[8], 0xD9429276);
  r[9]=chain.add(f[9], 0x8C760B80);
  r[10]=chain.add(f[10], 0x2F8A21D5);
  r[11]=chain.add(f[11], 0x035C748C);
}

__device__ __forceinline__ void BLS12377G1::add3N(Field& r, const Field& f) {
  chain_t chain;

  r[0]=chain.add(f[0], 0x00000003);
  r[1]=chain.add(f[1], 0x8F1A4000);
  r[2]=chain.add(f[2], 0x90000001);
  r[3]=chain.add(f[3], 0x452217CC);
  r[4]=chain.add(f[4], 0x2E1BD800);
  r[5]=chain.add(f[5], 0x5CDA268F);
  r[6]=chain.add(f[6], 0x02DF3AAD);
  r[7]=chain.add(f[7], 0x4E688DD9);
  r[8]=chain.add(f[8], 0x45E3DBB1);
  r[9]=chain.add(f[9], 0x52B11141);
  r[10]=chain.add(f[10], 0x474F32C0);
  r[11]=chain.add(f[11], 0x050AAED2);
}

__device__ __forceinline__ void BLS12377G1::add4N(Field& r, const Field& f) {
  chain_t chain;

  r[0]=chain.add(f[0], 0x00000004);
  r[1]=chain.add(f[1], 0x14230000);
  r[2]=chain.add(f[2], 0xC0000002);
  r[3]=chain.add(f[3], 0x5C2D7510);
  r[4]=chain.add(f[4], 0xE8252000);
  r[5]=chain.add(f[5], 0x7BCD88BE);
  r[6]=chain.add(f[6], 0x03D44E3C);
  r[7]=chain.add(f[7], 0x688B67CC);
  r[8]=chain.add(f[8], 0xB28524EC);
  r[9]=chain.add(f[9], 0x18EC1701);
  r[10]=chain.add(f[10], 0x5F1443AB);
  r[11]=chain.add(f[11], 0x06B8E918);
}

__device__ __forceinline__ void BLS12377G1::add5N(Field& r, const Field& f) {
  chain_t chain;

  r[0]=chain.add(f[0], 0x00000005);
  r[1]=chain.add(f[1], 0x992BC000);
  r[2]=chain.add(f[2], 0xF0000002);
  r[3]=chain.add(f[3], 0x7338D254);
  r[4]=chain.add(f[4], 0xA22E6800);
  r[5]=chain.add(f[5], 0x9AC0EAEE);
  r[6]=chain.add(f[6], 0x04C961CB);
  r[7]=chain.add(f[7], 0x82AE41BF);
  r[8]=chain.add(f[8], 0x1F266E27);
  r[9]=chain.add(f[9], 0xDF271CC2);
  r[10]=chain.add(f[10], 0x76D95495);
  r[11]=chain.add(f[11], 0x0867235E);  
}

__device__ __forceinline__ void BLS12377G1::add6N(Field& r, const Field& f) {
  chain_t chain;

  r[0]=chain.add(f[0], 0x00000006);
  r[1]=chain.add(f[1], 0x1E348000);
  r[2]=chain.add(f[2], 0x20000003);
  r[3]=chain.add(f[3], 0x8A442F99);
  r[4]=chain.add(f[4], 0x5C37B000);
  r[5]=chain.add(f[5], 0xB9B44D1E);
  r[6]=chain.add(f[6], 0x05BE755A);
  r[7]=chain.add(f[7], 0x9CD11BB2);
  r[8]=chain.add(f[8], 0x8BC7B762);
  r[9]=chain.add(f[9], 0xA5622282);
  r[10]=chain.add(f[10], 0x8E9E6580);
  r[11]=chain.add(f[11], 0x0A155DA4);  
}

__device__ __forceinline__ void BLS12377G1::add(Field& r, const Field& a, const Field& b) {
  mp_add<limbs>(r, a, b);
}

__device__ __forceinline__ void BLS12377G1::sub(Field& r, const Field& a, const Field& b) {
  mp_sub<limbs>(r, a, b);
}

__device__ __forceinline__ void BLS12377G1::modmul(Field& r, const Field& a, const Field& b) {
  Field    localN;
  uint64_t evenOdd[limbs];
  bool     carry;

  setN(localN);
  carry=mp_mul_red_cl<BLS12377G1, limbs>(evenOdd, a, b, localN);
  mp_merge_cl<limbs>(r, evenOdd, carry);
}

__device__ __forceinline__ void BLS12377G1::modsqr(Field& r, const Field& a) {
  Field    localN;
  uint64_t evenOdd[limbs];
  uint32_t t[limbs];
  bool     carry;

  setN(localN);
  carry=mp_sqr_red_cl<BLS12377G1, limbs>(evenOdd, t, a, localN);
  mp_merge_cl<limbs>(r, evenOdd, carry);
}

__device__ __forceinline__ void BLS12377G1::modinv(Field& r, const Field& f) {
  Field localRInv;
  Field localN;

  setRInv(localRInv);
  modmul(r, f, localRInv);
  setN(localN);
  mp_inverse<limbs>(r, r, localN);
}

__device__ __forceinline__ void BLS12377G1::toMontgomery(Field& r, const Field& f) {
  Field localRSquared;

  setRSquared(localRSquared);
  modmul(r, f, localRSquared);

  // Output assertion:  0 <= r < 2N
}

__device__ __forceinline__ void BLS12377G1::fromMontgomery(Field& r, const Field& f) {
  Field localOne;

  setOne(localOne);
  modmul(r, f, localOne);

  // I believe the output assertion for this is R<N, but it's not used, so it's not necessary to prove
}

// Extended Jacobian Representation
__device__ __forceinline__ void BLS12377G1::normalize(PointXY& r, const PointXYZZ& pt) {
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

__device__ __forceinline__ void BLS12377G1::negateXY(PointXY& pt) {
  Field localN;

  setN(localN);
  mp_sub<limbs>(pt.y, localN, pt.y);
}

__device__ __forceinline__ void BLS12377G1::initializeXYZZ(PointXYZZ& acc) {
  setZero(acc.x);
  setZero(acc.y);
  setZero(acc.zz);
  setZero(acc.zzz);
}

__device__ __forceinline__ void BLS12377G1::setXY(PointXYZZ& acc, const PointXY& pt) {
  mp_copy<limbs>(acc.x, pt.x);
  mp_copy<limbs>(acc.y, pt.y);
  setR(acc.zz);
  setR(acc.zzz);
}

__device__ __forceinline__ void BLS12377G1::setXYZZ(PointXYZZ& acc, const PointXYZZ& pt) {
  mp_copy<limbs>(acc.x, pt.x);
  mp_copy<limbs>(acc.y, pt.y);
  mp_copy<limbs>(acc.zz, pt.zz);
  mp_copy<limbs>(acc.zzz, pt.zzz);
}

__device__ __forceinline__ void BLS12377G1::warpShuffleXYZZ(PointXYZZ& r, const PointXYZZ& pt, const uint32_t lane) {
  warpShuffleField(r.x, pt.x, lane);
  warpShuffleField(r.y, pt.y, lane);
  warpShuffleField(r.zz, pt.zz, lane);
  warpShuffleField(r.zzz, pt.zzz, lane);
}

__device__ __forceinline__ void BLS12377G1::doubleXY(PointXYZZ& acc, const PointXY& pt) {
  Field T0, T1;

  // Input assertion:  0 <= pt.x < N, 0 <= pt.y < N

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
  sub(T0, T0, acc.x);
  add5N(T0, T0);

  modmul(T1, T1, T0);
  sub(acc.y, T1, acc.y);
  add2N(acc.y, acc.y);

  // Output assertion:  0 <= acc.x <= 6N, 0 <= acc.y <= 4N, 0 <= acc.zz, acc.zzz <= 2N
}

__device__ __forceinline__ void BLS12377G1::doubleXYZZ(PointXYZZ& acc, const PointXYZZ& pt) {
  Field T0, T1;

  // Input assertion:  0 <= pt.x <= 6N, 0 <= pt.y <= 4N, 0 <= pt.zz, pt.zzz <= 2N  

  if(isZero(pt.zz)) {
    setXYZZ(acc, pt);
    return;
  }

  add(T1, pt.y, pt.y);
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
  sub(T0, T0, acc.x);
  add5N(T0, T0);

  modmul(T1, T1, T0);
  sub(acc.y, T1, acc.y);
  add2N(acc.y, acc.y);

  // Output assertion:  0 <= acc.x <= 6N, 0 <= acc.y <= 4N, 0 <= acc.zz, acc.zzz <= 2N
}

__device__ __forceinline__ void BLS12377G1::accumulateXY(PointXYZZ& acc, const PointXY& pt) {
  Field T0, T1, T2;

  // Input assertion:  0 <= acc.x <= 6N, 0 <= acc.y <= 4N, 0 <= acc.zz, acc.zzz <= 2N
  //                   0 <= pt.x < N, 0 <= pt.y < N

  if(isZero(acc.zz)) {
    setXY(acc, pt);
    return;
  }

  modmul(T0, pt.x, acc.zz);
  modmul(T1, pt.y, acc.zzz);
  sub(T0, T0, acc.x);
  add6N(T0, T0);
  sub(T1, T1, acc.y);
  add4N(T1, T1);

  if(((T0[0] | T1[0]) & 0xFFFFFFF0)==0) {
    if(isZero(T0) && isZero(T1)) {
      doubleXY(acc, pt);
      return;
    }
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
  add4N(acc.x, acc.x);

  sub(T2, T2, acc.x);
  add6N(T2, T2);
  modmul(T2, T1, T2);
  sub(acc.y, T2, acc.y);
  add2N(acc.y, acc.y);

  // Output assertion:  0 <= acc.x <= 6N, 0 <= acc.y <= 4N, 0 <= acc.zz, acc.zzz <= 2N
}

__device__ __forceinline__ void BLS12377G1::accumulateXYZZ(PointXYZZ& acc, const PointXYZZ& pt) {
  Field T0, T1, T2;

  // Input assertion:  0 <= acc.x <= 6N, 0 <= acc.y <= 4N, 0 <= acc.zz, acc.zzz <= 2N
  //                   0 <= pt.x <= 6N, 0 <= pt.y <= 4N, 0 <= pt.zz, pt.zzz <= 2N

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
  add6N(T0, T0);
  sub(T1, T1, acc.y);
  add4N(T1, T1);

  if(((T0[0] | T1[0]) & 0xFFFFFFF0)==0) {
    if(isZero(T0) && isZero(T1)) {
      doubleXYZZ(acc, pt);
      return;
    }
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
  add4N(acc.x, acc.x);

  sub(T2, T2, acc.x);
  add6N(T2, T2);
  modmul(T2, T1, T2);
  sub(acc.y, T2, acc.y);
  add2N(acc.y, acc.y);

  // Output assertion:  0 <= acc.x <= 6N, 0 <= acc.y <= 4N, 0 <= acc.zz, acc.zzz <= 2N
}

__device__ __forceinline__ void BLS12377G1::toTwistedEdwards(PointXYT& r, const PointXY& pt) {  
  Field T0, T1;

  setR(T1);
  add(T0, pt.x, T1);
  modinv(T1, pt.y);
  modmul(r.x, T0, T1);
  setF(T1);
  modmul(r.x, r.x, T1);
  reduceFully(r.x, r.x);

  setS(T1);
  modmul(T0, T0, T1);
  setR(r.y);
  add(T1, T0, r.y);
  sub(T0, T0, r.y);
  addN(T0, T0);
  modinv(T1, T1);
  modmul(T0, T0, T1);
  reduceFully(r.y, T0);

  setK(T0);
  modmul(r.t, r.x, r.y);
  modmul(r.t, r.t, T0);
  reduceFully(r.t, r.t);

  // Output assertion:  0 <= r.x, r.y, r.t < N
}

__device__ __forceinline__ void BLS12377G1::negateXYT(PointXY& pt) {
  Field localN;

  // Input assertion:  0 <= pt.x, pt.y <= N

  // this is pretty terrible naming!  We can't call it negateXY because that conflicts with XYZZ negate

  setN(localN);
  sub(pt.x, localN, pt.x);

  // Output assertion:  0 <= pt.x, pt.y <= N
}

__device__ __forceinline__ void BLS12377G1::negateXYT(PointXYT& pt) {
  Field localN, local2N;

  // Input assertion:  0 <= pt.x, pt.y, pt.t <= 2N

  set2N(local2N);
  sub(pt.x, local2N, pt.x);
  set2N(localN);
  sub(pt.t, local2N, pt.t);

  // Output assertion:  0 <= pt.x, pt.y, pt.t <= 2N
}

__device__ __forceinline__ void BLS12377G1::initializeXYTZ(PointXYTZ& acc) {
  setZero(acc.x);
  setR(acc.y);
  setZero(acc.t);
  setR(acc.z);
}

__device__ __forceinline__ void BLS12377G1::setXYTZ(PointXYTZ& acc, const PointXYTZ& pt) {
  mp_copy<limbs>(acc.x, pt.x);
  mp_copy<limbs>(acc.y, pt.y);
  mp_copy<limbs>(acc.t, pt.t);
  mp_copy<limbs>(acc.z, pt.z);
}

__device__ __forceinline__ void BLS12377G1::warpShuffleXYTZ(PointXYTZ& r, const PointXYTZ& pt, const uint32_t lane) {
  warpShuffleField(r.x, pt.x, lane);
  warpShuffleField(r.y, pt.y, lane);
  warpShuffleField(r.t, pt.t, lane);
  warpShuffleField(r.z, pt.z, lane);
}

__device__ __forceinline__ void BLS12377G1::accumulateXY(PointXYTZ& acc, const PointXY& pt) {
  Field T0, T1, T2;

  // Input assertion:  0 <= acc.x, acc.y, acc.t, acc.z <= 2N
  //                   0 <= pt.x, pt.y <= N

  // Uses tries the non-unified equations first, if successful, it's 8 modmuls
  // if not, it falls back to unified equations, using a total of 11 modmuls

  sub(T1, acc.y, acc.x);
  add2N(T1, T1);
  add(T2, pt.y, pt.x);
  modmul(T0, T1, T2);              // A  (in T0)

  add(T1, acc.y, acc.x);
  sub(T2, pt.y, pt.x);
  addN(T2, T2);
  modmul(T1, T1, T2);              // B  (in T1)

  add(T2, T1, T0);                 // G  (in T2)
  sub(T1, T1, T0);
  add2N(T1, T1);                   // F  (in T1)

  modmul(T0, T1, T2);              // Z out

  if(isZero(T0)) {
    sub(T0, acc.x, acc.y);
    add2N(T0, T0);
    sub(T1, pt.x, pt.y);
    add2N(T1, T1);
    modmul(T0, T0, T1);            // A

    add(acc.x, acc.x, acc.y);
    add(acc.y, pt.x, pt.y);
    modmul(T1, acc.x, acc.y);      // B 

    setK(T2);
    modmul(acc.t, acc.t, pt.x);
    modmul(acc.t, acc.t, pt.y);
    modmul(acc.t, acc.t, T2);      // C

    add(acc.z, acc.z, acc.z);      // D

    sub(acc.x, T1, T0);            // E
    add2N(acc.x, acc.x);
    add(acc.y, T1, T0);            // H 
    sub(T0, acc.z, acc.t);         // F 
    add2N(T0, T0);
    add(T1, acc.z, acc.t);         // G 

    modmul(acc.t, acc.x, acc.y);
    modmul(acc.z, T0, T1);    
    modmul(acc.x, acc.x, T0);     
    modmul(acc.y, acc.y, T1);   
    return;
  }

  add(acc.z, acc.z, acc.z);
  modmul(acc.z, acc.z, pt.x);
  modmul(acc.z, acc.z, pt.y);      // C

  add(acc.x, acc.t, acc.t);        // D
  add(acc.t, acc.x, acc.z);        // E
  sub(acc.z, acc.x, acc.z);     
  add2N(acc.z, acc.z);             // H

  // E in acc.t, F in T1, G in T2, H in acc.z  (out Z is in T0)
  modmul(acc.x, acc.t, T1);
  modmul(acc.y, acc.z, T2);
  modmul(acc.t, acc.z, acc.t);
  setField(acc.z, T0);  
}

__device__ __forceinline__ void BLS12377G1::accumulateXYT(PointXYTZ& acc, const PointXYT& pt) {
  Field T0, T1;

  // Input assertion:  0 <= acc.x, acc.y, acc.t, acc.z <= 2N
  //                   0 <= pt.x, pt.y, pt.t <= N

  // 7 modmuls, requires pt.t = pt.x * pt.y * k
  sub(T0, acc.x, acc.y);
  add2N(T0, T0);
  sub(T1, pt.x, pt.y);
  add2N(T1, T1);
  modmul(T0, T0, T1);            // A

  add(acc.x, acc.x, acc.y);
  add(acc.y, pt.x, pt.y);
  modmul(T1, acc.x, acc.y);      // B 

  modmul(acc.t, acc.t, pt.t);    // C   Note, pt.t has k embedded
  add(acc.z, acc.z, acc.z);      // D

  sub(acc.x, T1, T0);            // E
  add2N(acc.x, acc.x);
  add(acc.y, T1, T0);            // H 
  sub(T0, acc.z, acc.t);         // F 
  add2N(T0, T0);
  add(T1, acc.z, acc.t);         // G 

  modmul(acc.t, acc.x, acc.y);
  modmul(acc.z, T0, T1);    
  modmul(acc.x, acc.x, T0);     
  modmul(acc.y, acc.y, T1);   

  // Output assertion: 0 <= acc.x, acc.y, acc.t, acc.z <= 2N
}

__device__ __forceinline__ void BLS12377G1::accumulateXYTZ(PointXYTZ& acc, const PointXYTZ& pt) {
  Field T0, T1;

  // Input assertion:  0 <= acc.x, acc.y, acc.t, acc.z <= 2N
  //                   0 <= pt.x, pt.y, pt.t, pt.z <= 2N

  setK(T0);
  modmul(acc.t, acc.t, pt.t);
  modmul(acc.t, acc.t, T0);      // C
  modmul(acc.z, acc.z, pt.z);
  add(acc.z, acc.z, acc.z);      // D

  sub(T0, acc.x, acc.y);
  add2N(T0, T0);
  sub(T1, pt.x, pt.y);
  add2N(T1, T1);
  modmul(T0, T0, T1);            // A

  add(T1, pt.x, pt.y);
  add(acc.x, acc.x, acc.y);
  modmul(T1, acc.x, T1);         // B 

  sub(acc.x, T1, T0);            // E
  add2N(acc.x, acc.x);
  add(acc.y, T1, T0);            // H
  sub(T0, acc.z, acc.t);         // F
  add2N(T0, T0);
  add(T1, acc.z, acc.t);         // G

  modmul(acc.t, acc.x, acc.y);
  modmul(acc.z, T0, T1);    
  modmul(acc.x, acc.x, T0);     
  modmul(acc.y, acc.y, T1);   

  // Output assertion: 0 <= acc.x, acc.y, acc.t, acc.z <= 2N
}