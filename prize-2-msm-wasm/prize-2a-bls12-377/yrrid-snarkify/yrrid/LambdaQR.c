/***

Copyright (c) 2023-2024, Yrrid Software, Inc. and Snarkify Inc.  All rights reserved.
Dual licensed under the MIT License or the Apache License, Version 2.0.
See LICENSE for details.

Author(s):  Niall Emmart

***/

static inline bool add32(uint32_t* r, uint32_t* a, uint32_t* b, uint32_t count) {
  int64_t carry=0;

  for(int i=0;i<count;i++) {
    carry=carry + a[i] + b[i];
    r[i]=(uint32_t)carry;
    carry=carry>>32;
  }
  return carry!=0;
}

static inline bool sub32(uint32_t* r, uint32_t* a, uint32_t* b, uint32_t count) {
  int64_t carry=0;

  for(int i=0;i<count;i++) {
    carry=carry + a[i] - b[i];
    r[i]=(uint32_t)carry;
    carry=carry>>32;
  }
  return carry!=0;
}

void lambdaQR(uint32_t* q, uint32_t* r, uint32_t* scalar) {
  uint32_t d0=scalar[0], d1=scalar[1], d2=scalar[2], d3=scalar[3], d4=scalar[4], d5=scalar[5], d6=scalar[6], d7=scalar[7];
  uint64_t l0=0x00000000, l1=0x28460000, l2=0x00000010, l3=0x0885F324, l4=0x00000045;     // <-- lambda
  uint64_t i0=0x1F203058, i1=0x3B976995, i2=0x3A6E2C0F, i3=0x3BD54B4F, i4=0x00000767;     // <-- inv (2^257 / lambda)
  uint64_t h0, h1, h2, h3, h4;
  uint64_t p0, p1, p2, p3, p4;
  uint64_t t, mask30=0x3FFFFFFF, mask32=0xFFFFFFFFu;
  int64_t  carry;

  // Similar to Barrett's method to compute q & r

  // shift the scalar right by 121 bits (i.e., extract the top 132 bits).
  // multiply by the inverse of lambda, i0-i4
  h0=(d3>>25) + (d4<<7) & mask30;
  h1=(d4>>23) + (d5<<9) & mask30;
  h2=(d5>>21) + (d6<<11) & mask30;
  h3=(d6>>19) + (d7<<13) & mask30;
  h4=d7>>17;

  t=h0*i3;
  t+=h1*i2;
  t+=h2*i1;
  t+=h3*i0;

  p0=h0*i4;
  p0+=h1*i3;
  p0+=h2*i2;
  p0+=h3*i1;
  p0+=h4*i0;
  p0+=(t>>30);

  p1=h1*i4;
  p1+=h2*i3;
  p1+=h3*i2;
  p1+=h4*i1;
  p1+=(p0>>30);

  p2=h2*i4;
  p2+=h3*i3;
  p2+=h4*i2;
  p2+=(p1>>30);

  p3=h3*i4;
  p3+=h4*i3;
  p3+=(p2>>30);

  p4=h4*i4;
  p4+=(p3>>30);

  p0=p0 & mask30;
  p1=p1 & mask30;
  p2=p2 & mask30;
  p3=p3 & mask30;
  p4=p4 & mask30;

  // q estimate is in p0-p4

  // next shift p0-p4 by 136 bits (note, we only computed the top 5 words of the product, so this is just a shift by 16 bits)

  h0=(p0>>16) + (p1<<14) & mask30;
  h1=(p1>>16) + (p2<<14) & mask30;
  h2=(p2>>16) + (p3<<14) & mask30;
  h3=(p3>>16) + (p4<<14) & mask30;
  h4=p4>>16;

  // pack the q estimate up into 32 bit words

  q[0]=(uint32_t)(h0 + (h1<<30));
  q[1]=(uint32_t)((h1>>2) + (h2<<28));
  q[2]=(uint32_t)((h2>>4) + (h3<<26));
  q[3]=(uint32_t)((h3>>6) + (h4<<24));

  // compute q estimate * lambda

  p0=0;

  p1=h0*l1;
  // p1+=(p0>>30);

  p2=h0*l2;
  p2+=h1*l1;
  p2+=(p1>>30);

  p3=h0*l3;
  p3+=h1*l2;
  p3+=h2*l1;
  p3+=(p2>>30);

  p4=h0*l4;
  p4+=h1*l3;
  p4+=h2*l2;
  p4+=h3*l1;
  p4+=(p3>>30);

  // p0=p0
  p1=p1 & mask30;
  p2=p2 & mask30;
  p3=p3 & mask30;
  p4=p4 & mask30;

  // pack q*lambda into h0-h4

  h0=p0 + (p1<<30) & mask32;
  h1=(p1>>2) + (p2<<28) & mask32;
  h2=(p2>>4) + (p3<<26) & mask32;
  h3=(p3>>6) + (p4<<24) & mask32;
  h4=p4>>8;

  // compute scalar - q*lambda

  carry=0 - h0 + d0;
  r[0]=(uint32_t)carry;
  carry=(carry>>32) - h1 + d1;
  r[1]=(uint32_t)carry;
  carry=(carry>>32) - h2 + d2;
  r[2]=(uint32_t)carry;
  carry=(carry>>32) - h3 + d3;
  r[3]=(uint32_t)carry;
  carry=(carry>>32) - h4 + d4;

  carry=carry & 0xFFFF;

  if(r[3]<0x80000000U && carry==0)
    return;

  // I think we literally never get here, but the proof is more trouble than it's worth

  while(carry!=0 || r[3]>=0x80000000U) {
    uint32_t lambda[4]={0x00000000, 0x0A118000, 0x90000001, 0x452217CC};
    uint32_t one[4]={1, 0, 0, 0};

    // untested code!
    if(!sub32(r, r, lambda, 4))
      carry--;
    add32(q, q, one, 4);
  }
}
