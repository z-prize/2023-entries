/***

Copyright (c) 2023-2024, Yrrid Software, Inc. and Snarkify Inc.  All rights reserved.
Dual licensed under the MIT License or the Apache License, Version 2.0.
See LICENSE for details.

Author(s):  Niall Emmart

***/

ExpandedField_t fieldMultiples[10]={
  {0x000000000000L, 0x000000000000L, 0x000000000000L, 0x000000000000L, 0x000000000000L, 0x000000000000L, 0x000000000000L, 0x000000000000L},
  {0xC00000000001L, 0x300000008508L, 0x4800170B5D44L, 0x1EF3622FBA09L, 0xD9F300F5138FL, 0x6CA1493B1A22L, 0x10EAC63B05C0L, 0x01AE3A4617C5L},
  {0x800000000002L, 0x600000010A11L, 0x90002E16BA88L, 0x3DE6C45F7412L, 0xB3E601EA271EL, 0xD94292763445L, 0x21D58C760B80L, 0x035C748C2F8AL},
  {0x400000000003L, 0x900000018F1AL, 0xD800452217CCL, 0x5CDA268F2E1BL, 0x8DD902DF3AADL, 0x45E3DBB14E68L, 0x32C052B11141L, 0x050AAED2474FL},
  {0x000000000004L, 0xC00000021423L, 0x20005C2D7510L, 0x7BCD88BEE825L, 0x67CC03D44E3CL, 0xB28524EC688BL, 0x43AB18EC1701L, 0x06B8E9185F14L},
  {0xC00000000005L, 0xF0000002992BL, 0x68007338D254L, 0x9AC0EAEEA22EL, 0x41BF04C961CBL, 0x1F266E2782AEL, 0x5495DF271CC2L, 0x0867235E76D9L},
  {0x800000000006L, 0x200000031E34L, 0xB0008A442F99L, 0xB9B44D1E5C37L, 0x1BB205BE755AL, 0x8BC7B7629CD1L, 0x6580A5622282L, 0x0A155DA48E9EL},
  {0x400000000007L, 0x50000003A33DL, 0xF800A14F8CDDL, 0xD8A7AF4E1640L, 0xF5A506B388E9L, 0xF869009DB6F3L, 0x766B6B9D2842L, 0x0BC397EAA663L},
  {0x000000000008L, 0x800000042846L, 0x4000B85AEA21L, 0xF79B117DD04AL, 0xCF9807A89C78L, 0x650A49D8D116L, 0x875631D82E03L, 0x0D71D230BE28L},
  {0xC00000000009L, 0xB0000004AD4EL, 0x8800CF664765L, 0x168E73AD8A53L, 0xA98B089DB008L, 0xD1AB9313EB39L, 0x9840F81333C3L, 0x0F200C76D5EDL}
};

void fieldLoad(Field_t* f, I128* data) {
  I128      v0, v1, v2, mask=i64x2_splat(0xFFFFFFFFFFFFL);
  uint64_t  f0, f1, f2, f3, f4, f5, f6;

  v0=data[0];
  v1=data[1];
  v2=data[2];

  f0=i64x2_extract_l(v0);
  f1=i64x2_extract_h(v0);
  f2=i64x2_extract_l(v1);
  f3=i64x2_extract_h(v1);
  f4=i64x2_extract_l(v2);
  f5=i64x2_extract_h(v2);

  f->v0=i64x2_make(f0, (f0>>48) | (f1<<16));
  f->v1=i64x2_make((f1>>32) | (f2<<32), f2>>16);
  f->v2=i64x2_make(f3, (f3>>48) | (f4<<16));
  f->v3=i64x2_make((f4>>32) | (f5<<32), f5>>16);

  f->v0=i64x2_and(f->v0, mask);
  f->v1=i64x2_and(f->v1, mask);
  f->v2=i64x2_and(f->v2, mask);
  f->v3=i64x2_and(f->v3, mask);
}

void fieldStore(I128* data, Field_t* f) {
  int64_t l[8]={i64x2_extract_l(f->v0), i64x2_extract_h(f->v0), i64x2_extract_l(f->v1), i64x2_extract_h(f->v1),
                i64x2_extract_l(f->v2), i64x2_extract_h(f->v2), i64x2_extract_l(f->v3), i64x2_extract_h(f->v3)};
  uint64_t mask16=0xFFFFL, mask32=0xFFFFFFFFL, mask48=0xFFFFFFFFFFFFL;

  l[1]+=l[0]>>48;
  l[2]+=l[1]>>48;
  l[3]+=l[2]>>48;
  l[4]+=l[3]>>48;
  l[5]+=l[4]>>48;
  l[6]+=l[5]>>48;
  l[7]+=l[6]>>48;

  data[0]=i64x2_make((l[0] & mask48) | (l[1]<<48), ((l[1]>>16) & mask32) | (l[2]<<32));
  data[1]=i64x2_make(((l[2]>>32) & mask16) | (l[3]<<16), (l[4] & mask48) | (l[5]<<48));
  data[2]=i64x2_make(((l[5]>>16) & mask32) | (l[6]<<32), ((l[6]>>32) & mask16) | (l[7]<<16));
}

void fieldResolve(Field_t* f) {
  int64_t l[8]={i64x2_extract_l(f->v0), i64x2_extract_h(f->v0), i64x2_extract_l(f->v1), i64x2_extract_h(f->v1),
                i64x2_extract_l(f->v2), i64x2_extract_h(f->v2), i64x2_extract_l(f->v3), i64x2_extract_h(f->v3)};
  I128    mask=i64x2_splat(0xFFFFFFFFFFFFL);

  l[1]+=l[0]>>48;
  l[2]+=l[1]>>48;
  l[3]+=l[2]>>48;
  l[4]+=l[3]>>48;
  l[5]+=l[4]>>48;
  l[6]+=l[5]>>48;
  l[7]+=l[6]>>48;

  f->v0=i64x2_and(i64x2_make(l[0], l[1]), mask);
  f->v1=i64x2_and(i64x2_make(l[2], l[3]), mask);
  f->v2=i64x2_and(i64x2_make(l[4], l[5]), mask);
  f->v3=i64x2_and(i64x2_make(l[6], l[7]), mask);
}

void fieldPartialResolve(Field_t* f) {
  I128 v0, v1, v2, v3, zero, mask;

  zero=i64x2_splat(0L);
  mask=i64x2_splat(0xFFFFFFFFFFFFL);

  v3=i64x2_shr(f->v3, 48);
  v2=i64x2_shr(f->v2, 48);
  v1=i64x2_shr(f->v1, 48);
  v0=i64x2_shr(f->v0, 48);

  v3=i64x2_extract_hl(v2, v3);
  v2=i64x2_extract_hl(v1, v2);
  v1=i64x2_extract_hl(v0, v1);
  v0=i64x2_extract_hl(zero, v0);

  f->v3=i64x2_add(i64x2_and(f->v3, mask), v3);
  f->v2=i64x2_add(i64x2_and(f->v2, mask), v2);
  f->v1=i64x2_add(i64x2_and(f->v1, mask), v1);
  f->v0=i64x2_add(i64x2_and(f->v0, mask), v0);
}

void fieldAdd(Field_t* r, Field_t* a, Field_t* b) {
  r->v0=i64x2_add(a->v0, b->v0);
  r->v1=i64x2_add(a->v1, b->v1);
  r->v2=i64x2_add(a->v2, b->v2);
  r->v3=i64x2_add(a->v3, b->v3);
}

void fieldSub(Field_t* r, Field_t* a, Field_t* b) {
  r->v0=i64x2_sub(a->v0, b->v0);
  r->v1=i64x2_sub(a->v1, b->v1);
  r->v2=i64x2_sub(a->v2, b->v2);
  r->v3=i64x2_sub(a->v3, b->v3);
}

bool fieldIsZero(Field_t* f) {
  uint64_t multiple, limbOr;
  Field_t  check;

  /*********************************************************************
   * Input Assertion:  0<=f<=9.99N
   *                   f can be in redundant form
   * Output Assertion: true, if and only if f mod N==0
   *
   * ceil(2^396 / N) = 0x98543
   *
   * One tricky thing here.  If i64x2_extract_h(f->v3) can be negative,
   * in which case, extract_h(resolved f) must be small.
   *********************************************************************/

  multiple=(i64x2_extract_h(f->v3) * 0x98543)>>60;
  if(multiple==15L) multiple=0L;
  if((i64x2_extract_l(f->v0) & 0x3FFFFFFFFFFFL)!=multiple)
    return false;

  // it should be very rare that we need to check this!
  fieldSub(&check, f, (Field_t*)&fieldMultiples[(uint32_t)multiple]);

  int64_t l[8]={i64x2_extract_l(check.v0), i64x2_extract_h(check.v0), i64x2_extract_l(check.v1), i64x2_extract_h(check.v1),
                i64x2_extract_l(check.v2), i64x2_extract_h(check.v2), i64x2_extract_l(check.v3), i64x2_extract_h(check.v3)};

  l[1]+=l[0]>>48;
  l[2]+=l[1]>>48;
  l[3]+=l[2]>>48;
  l[4]+=l[3]>>48;
  l[5]+=l[4]>>48;
  l[6]+=l[5]>>48;
  l[7]+=l[6]>>48;

  limbOr=l[0] | l[1] | l[2] | l[3] | l[4] | l[5] | l[6] | l[7];
  return (limbOr & 0xFFFFFFFFFFFFL)==0;
}

void fieldReduce(Field_t* f) {
  uint32_t multiple;

  /*********************************************************************
   * Input Assertion:  0<=f<=9.99N
   *                   f can be in redundant form
   * Output Assertion: 0<=f<=1.01N
   *
   * floor(2^396 / N) = 0x98542
   *
   * One tricky thing here.  If i64x2_extract_h(f->v3) can be negative,
   * in which case, extract_h(resolved f) must be small.
   *********************************************************************/

  multiple=(i64x2_extract_h(f->v3) * 0x98542)>>60;
  if(multiple==15) multiple=0;
  fieldSub(f, f, (Field_t*)&fieldMultiples[multiple]);
}

void fieldSetZero(Field_t* f) {
  f->v0=i64x2_splat(0L);
  f->v1=i64x2_splat(0L);
  f->v2=i64x2_splat(0L);
  f->v3=i64x2_splat(0L);
}

void fieldSetOne(Field_t* f) {
  f->v0=i64x2_make(0xFFFFFFFFFF68L, 0x7FFFFFB102CDL);
  f->v1=i64x2_make(0x3FF251409F83L, 0x9F7DB3A98A7DL);
  f->v2=i64x2_make(0x97B76E7C6305L, 0x803C84E87B4EL);
  f->v3=i64x2_make(0xF49A4CF495BFL, 0x008D6661E2FDL);
}

void fieldSetPhi(Field_t* f) {
  f->v0=i64x2_make(0x106DA5847973L, 0xBAC2A79ADACDL);
  f->v1=i64x2_make(0x2EDCD8FE2454L, 0x1ADA4FD6FD83L);
  f->v2=i64x2_make(0x68449D150908L, 0xEA32285EFB98L);
  f->v3=i64x2_make(0x3FD0D63EB8AEL, 0x0167D6A36F87L);
}

void fieldSetN(Field_t* f) {
  f->v0=i64x2_make(0xC00000000001L, 0x300000008508L);
  f->v1=i64x2_make(0x4800170B5D44L, 0x1EF3622FBA09L);
  f->v2=i64x2_make(0xD9F300F5138FL, 0x6CA1493B1A22L);
  f->v3=i64x2_make(0x10EAC63B05C0L, 0x01AE3A4617C5L);
}

void fieldSet2N(Field_t* f) {
  f->v0=i64x2_make(0x800000000002L, 0x600000010A11L);
  f->v1=i64x2_make(0x90002E16BA88L, 0x3DE6C45F7412L);
  f->v2=i64x2_make(0xB3E601EA271EL, 0xD94292763445L);
  f->v3=i64x2_make(0x21D58C760B80L, 0x035C748C2F8AL);
}

void fieldSet3N(Field_t* f) {
  f->v0=i64x2_make(0x400000000003L, 0x900000018F1AL);
  f->v1=i64x2_make(0xD800452217CCL, 0x5CDA268F2E1BL);
  f->v2=i64x2_make(0x8DD902DF3AADL, 0x45E3DBB14E68L);
  f->v3=i64x2_make(0x32C052B11141L, 0x050AAED2474FL);
}

void fieldSet4N(Field_t* f) {
  f->v0=i64x2_make(0x000000000004L, 0xC00000021423L);
  f->v1=i64x2_make(0x20005C2D7510L, 0x7BCD88BEE825L);
  f->v2=i64x2_make(0x67CC03D44E3CL, 0xB28524EC688BL);
  f->v3=i64x2_make(0x43AB18EC1701L, 0x06B8E9185F14L);
}

void fieldSet5N(Field_t* f) {
  f->v0=i64x2_make(0xC00000000005L, 0xF0000002992BL);
  f->v1=i64x2_make(0x68007338D254L, 0x9AC0EAEEA22EL);
  f->v2=i64x2_make(0x41BF04C961CBL, 0x1F266E2782AEL);
  f->v3=i64x2_make(0x5495DF271CC2L, 0x0867235E76D9L);
}

void fieldSet6N(Field_t* f) {
  f->v0=i64x2_make(0x800000000006L, 0x200000031E34L);
  f->v1=i64x2_make(0xB0008A442F99L, 0xB9B44D1E5C37L);
  f->v2=i64x2_make(0x1BB205BE755AL, 0x8BC7B7629CD1L);
  f->v3=i64x2_make(0x6580A5622282L, 0x0A155DA48E9EL);
}

void fieldSetRInverse(Field_t* f) {
  f->v0=i64x2_make(0x9093451269E8L, 0xE65839F5EF12L);
  f->v1=i64x2_make(0x2C936E20BBCDL, 0x852E3C88A558L);
  f->v2=i64x2_make(0xF41DF7F2E657L, 0xA4C49351EEAAL);
  f->v3=i64x2_make(0x0736EB89746CL, 0x014212FC436BL);
}

void fieldSetRSquared(Field_t* f) {
  f->v0=i64x2_make(0x686C9400CD22L, 0xB00431B1B786L);
  f->v1=i64x2_make(0xB46D0329FCAAL, 0x22A5F11162D6L);
  f->v2=i64x2_make(0x7D03827DC3ACL, 0x41790BF9BFDFL);
  f->v3=i64x2_make(0x4B88837E92F0L, 0x006DFCCB1E91L);
}
