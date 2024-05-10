/***

Copyright (c) 2023-2024, Yrrid Software, Inc. and Snarkify Inc.  All rights reserved.
Dual licensed under the MIT License or the Apache License, Version 2.0.
See LICENSE for details.

Author(s):  Niall Emmart

***/

Field256_t scalarMultiples[55]={
  {0x0000000000000000L, 0x0000000000000000L, 0x0000000000000000L, 0x0000000000000000L},
  {0xB95AEE9AC33FD9FFL, 0x5293A3AFC43C8AFEL, 0x982D1347970DEC00L, 0x04AAD957A68B2955L},
  {0x72B5DD35867FB3FEL, 0xA527475F887915FDL, 0x305A268F2E1BD800L, 0x0955B2AF4D1652ABL},
  {0x2C10CBD049BF8DFDL, 0xF7BAEB0F4CB5A0FCL, 0xC88739D6C529C400L, 0x0E008C06F3A17C00L},
  {0xE56BBA6B0CFF67FCL, 0x4A4E8EBF10F22BFAL, 0x60B44D1E5C37B001L, 0x12AB655E9A2CA556L},
  {0x9EC6A905D03F41FBL, 0x9CE2326ED52EB6F9L, 0xF8E16065F3459C01L, 0x17563EB640B7CEABL},
  {0x582197A0937F1BFAL, 0xEF75D61E996B41F8L, 0x910E73AD8A538801L, 0x1C01180DE742F801L},
  {0x117C863B56BEF5F9L, 0x420979CE5DA7CCF7L, 0x293B86F521617402L, 0x20ABF1658DCE2157L},
  {0xCAD774D619FECFF8L, 0x949D1D7E21E457F5L, 0xC1689A3CB86F6002L, 0x2556CABD34594AACL},
  {0x84326370DD3EA9F7L, 0xE730C12DE620E2F4L, 0x5995AD844F7D4C02L, 0x2A01A414DAE47402L},
  {0x3D8D520BA07E83F6L, 0x39C464DDAA5D6DF3L, 0xF1C2C0CBE68B3803L, 0x2EAC7D6C816F9D57L},
  {0xF6E840A663BE5DF5L, 0x8C58088D6E99F8F1L, 0x89EFD4137D992403L, 0x335756C427FAC6ADL},
  {0xB0432F4126FE37F4L, 0xDEEBAC3D32D683F0L, 0x221CE75B14A71003L, 0x3802301BCE85F003L},
  {0x699E1DDBEA3E11F3L, 0x317F4FECF7130EEFL, 0xBA49FAA2ABB4FC04L, 0x3CAD097375111958L},
  {0x22F90C76AD7DEBF2L, 0x8412F39CBB4F99EEL, 0x52770DEA42C2E804L, 0x4157E2CB1B9C42AEL},
  {0xDC53FB1170BDC5F1L, 0xD6A6974C7F8C24ECL, 0xEAA42131D9D0D404L, 0x4602BC22C2276C03L},
  {0x95AEE9AC33FD9FF0L, 0x293A3AFC43C8AFEBL, 0x82D1347970DEC005L, 0x4AAD957A68B29559L},
  {0x4F09D846F73D79EFL, 0x7BCDDEAC08053AEAL, 0x1AFE47C107ECAC05L, 0x4F586ED20F3DBEAFL},
  {0x0864C6E1BA7D53EEL, 0xCE61825BCC41C5E9L, 0xB32B5B089EFA9805L, 0x54034829B5C8E804L},
  {0xC1BFB57C7DBD2DEDL, 0x20F5260B907E50E7L, 0x4B586E5036088406L, 0x58AE21815C54115AL},
  {0x7B1AA41740FD07ECL, 0x7388C9BB54BADBE6L, 0xE3858197CD167006L, 0x5D58FAD902DF3AAFL},
  {0x347592B2043CE1EBL, 0xC61C6D6B18F766E5L, 0x7BB294DF64245C06L, 0x6203D430A96A6405L},
  {0xEDD0814CC77CBBEAL, 0x18B0111ADD33F1E3L, 0x13DFA826FB324807L, 0x66AEAD884FF58D5BL},
  {0xA72B6FE78ABC95E9L, 0x6B43B4CAA1707CE2L, 0xAC0CBB6E92403407L, 0x6B5986DFF680B6B0L},
  {0x60865E824DFC6FE8L, 0xBDD7587A65AD07E1L, 0x4439CEB6294E2007L, 0x700460379D0BE006L},
  {0x19E14D1D113C49E7L, 0x106AFC2A29E992E0L, 0xDC66E1FDC05C0C08L, 0x74AF398F4397095BL},
  {0xD33C3BB7D47C23E6L, 0x62FE9FD9EE261DDEL, 0x7493F5455769F808L, 0x795A12E6EA2232B1L},
  {0x8C972A5297BBFDE5L, 0xB5924389B262A8DDL, 0x0CC1088CEE77E408L, 0x7E04EC3E90AD5C07L},
  {0x45F218ED5AFBD7E4L, 0x0825E739769F33DCL, 0xA4EE1BD48585D009L, 0x82AFC5963738855CL},
  {0xFF4D07881E3BB1E3L, 0x5AB98AE93ADBBEDAL, 0x3D1B2F1C1C93BC09L, 0x875A9EEDDDC3AEB2L},
  {0xB8A7F622E17B8BE2L, 0xAD4D2E98FF1849D9L, 0xD5484263B3A1A809L, 0x8C057845844ED807L},
  {0x7202E4BDA4BB65E1L, 0xFFE0D248C354D4D8L, 0x6D7555AB4AAF9409L, 0x90B0519D2ADA015DL},
  {0x2B5DD35867FB3FE0L, 0x527475F887915FD7L, 0x05A268F2E1BD800AL, 0x955B2AF4D1652AB3L},
  {0xE4B8C1F32B3B19DFL, 0xA50819A84BCDEAD5L, 0x9DCF7C3A78CB6C0AL, 0x9A06044C77F05408L},
  {0x9E13B08DEE7AF3DEL, 0xF79BBD58100A75D4L, 0x35FC8F820FD9580AL, 0x9EB0DDA41E7B7D5EL},
  {0x576E9F28B1BACDDDL, 0x4A2F6107D44700D3L, 0xCE29A2C9A6E7440BL, 0xA35BB6FBC506A6B3L},
  {0x10C98DC374FAA7DCL, 0x9CC304B798838BD2L, 0x6656B6113DF5300BL, 0xA80690536B91D009L},
  {0xCA247C5E383A81DBL, 0xEF56A8675CC016D0L, 0xFE83C958D5031C0BL, 0xACB169AB121CF95EL},
  {0x837F6AF8FB7A5BDAL, 0x41EA4C1720FCA1CFL, 0x96B0DCA06C11080CL, 0xB15C4302B8A822B4L},
  {0x3CDA5993BEBA35D9L, 0x947DEFC6E5392CCEL, 0x2EDDEFE8031EF40CL, 0xB6071C5A5F334C0AL},
  {0xF635482E81FA0FD8L, 0xE7119376A975B7CCL, 0xC70B032F9A2CE00CL, 0xBAB1F5B205BE755FL},
  {0xAF9036C94539E9D7L, 0x39A537266DB242CBL, 0x5F381677313ACC0DL, 0xBF5CCF09AC499EB5L},
  {0x68EB25640879C3D6L, 0x8C38DAD631EECDCAL, 0xF76529BEC848B80DL, 0xC407A86152D4C80AL},
  {0x224613FECBB99DD5L, 0xDECC7E85F62B58C9L, 0x8F923D065F56A40DL, 0xC8B281B8F95FF160L},
  {0xDBA102998EF977D4L, 0x31602235BA67E3C7L, 0x27BF504DF664900EL, 0xCD5D5B109FEB1AB6L},
  {0x94FBF134523951D3L, 0x83F3C5E57EA46EC6L, 0xBFEC63958D727C0EL, 0xD20834684676440BL},
  {0x4E56DFCF15792BD2L, 0xD687699542E0F9C5L, 0x581976DD2480680EL, 0xD6B30DBFED016D61L},
  {0x07B1CE69D8B905D1L, 0x291B0D45071D84C4L, 0xF0468A24BB8E540FL, 0xDB5DE717938C96B6L},
  {0xC10CBD049BF8DFD0L, 0x7BAEB0F4CB5A0FC2L, 0x88739D6C529C400FL, 0xE008C06F3A17C00CL},
  {0x7A67AB9F5F38B9CFL, 0xCE4254A48F969AC1L, 0x20A0B0B3E9AA2C0FL, 0xE4B399C6E0A2E962L},
  {0x33C29A3A227893CEL, 0x20D5F85453D325C0L, 0xB8CDC3FB80B81810L, 0xE95E731E872E12B7L},
  {0xED1D88D4E5B86DCDL, 0x73699C04180FB0BEL, 0x50FAD74317C60410L, 0xEE094C762DB93C0DL},
  {0xA678776FA8F847CCL, 0xC5FD3FB3DC4C3BBDL, 0xE927EA8AAED3F010L, 0xF2B425CDD4446562L},
  {0x5FD3660A6C3821CBL, 0x1890E363A088C6BCL, 0x8154FDD245E1DC11L, 0xF75EFF257ACF8EB8L},
  {0x192E54A52F77FBCAL, 0x6B24871364C551BBL, 0x19821119DCEFC811L, 0xFC09D87D215AB80EL},
};

void modSplit14(uint32_t* buckets, I128* scalar) {
  Field256_t s;
  uint64_t   mask = 0x3FFFL;
  uint32_t   multiple, add=0, shifted=0, zeroAdd, pos, neg;
  bool       zero;

  s.f0=i64x2_extract_l(scalar[0]);
  s.f1=i64x2_extract_h(scalar[0]);
  s.f2=i64x2_extract_l(scalar[1]);
  s.f3=i64x2_extract_h(scalar[1]);

  multiple=(s.f3>>32)*0xDB65247AL>>58;

  // remove multiples of the curve order from the scalar
  field256Sub(&s, &s, &scalarMultiples[multiple]);

  scalar[0]=i64x2_make(s.f0, s.f1);
  scalar[1]=i64x2_make(s.f2, s.f3);

  buckets[0]= s.f0 & mask;
  buckets[1]=(s.f0 >> 14) & mask;
  buckets[2]=(s.f0 >> 28) & mask;
  buckets[3]=(s.f0 >> 42) & mask;
  buckets[4]=((s.f0 >> 56) | (s.f1 << 8)) & mask;
  buckets[5]=(s.f1 >> 6) & mask;
  buckets[6]=(s.f1 >> 20) & mask;
  buckets[7]=(s.f1 >> 34) & mask;
  buckets[8]=(s.f1 >> 48) & mask;
  buckets[9]=((s.f1>>62) | (s.f2 << 2)) & mask;
  buckets[10]=(s.f2 >> 12) & mask;
  buckets[11]=(s.f2 >> 26) & mask;
  buckets[12]=(s.f2 >> 40) & mask;
  buckets[13]=((s.f2 >> 54) | (s.f3 << 10)) & mask;
  buckets[14]=(s.f3 >> 4) & mask;
  buckets[15]=(s.f3 >> 18) & mask;
  buckets[16]=(s.f3 >> 32) & mask;
  buckets[17]=s.f3>>46;

  for (int i = 0; i < 18; i++) {
    buckets[i] = buckets[i] + add;
    pos = shifted + buckets[i] - 1;
    neg = shifted + (buckets[i] ^ 0x80003FFF);
    add = buckets[i] >> 13;
    zeroAdd = buckets[i] >> 14;
    zero = (buckets[i] & 0x3FFF) == 0;

    buckets[i] = zero ? 0xFFFFFFFF : (add != 0) ? neg : pos;
    shifted = shifted + 0x2000;
    add = add - zeroAdd;
  }
}

void split14(uint32_t* buckets, I128* scalar) {
  uint64_t l, h;
  uint64_t mask = 0x3FFFL;
  uint32_t add=0, shifted=0, zeroAdd, pos, neg;
  bool     zero;

  l=i64x2_extract_l(scalar[0]);
  h=i64x2_extract_h(scalar[0]);
  buckets[0]=l & mask;
  buckets[1]=(l >> 14) & mask;
  buckets[2]=(l >> 28) & mask;
  buckets[3]=(l >> 42) & mask;
  buckets[4]=((l >> 56) | (h << 8)) & mask;
  buckets[5]=(h >> 6) & mask;
  buckets[6]=(h >> 20) & mask;
  buckets[7]=(h >> 34) & mask;
  buckets[8]=(h >> 48) & mask;
  buckets[9]=h>>62;

  l=i64x2_extract_l(scalar[1]);
  h=i64x2_extract_h(scalar[1]);
  buckets[9]=(buckets[9] | (l << 2)) & mask;
  buckets[10]=(l >> 12) & mask;
  buckets[11]=(l >> 26) & mask;
  buckets[12]=(l >> 40) & mask;
  buckets[13]=((l >> 54) | (h << 10)) & mask;
  buckets[14]=(h >> 4) & mask;
  buckets[15]=(h >> 18) & mask;
  buckets[16]=(h >> 32) & mask;
  buckets[17]=h>>46;

  for (int i = 0; i < 18; i++) {
    buckets[i] = buckets[i] + add;
    pos = shifted + buckets[i] - 1;
    neg = shifted + (buckets[i] ^ 0x80003FFF);
    add = buckets[i] >> 13;
    zeroAdd = buckets[i] >> 14;
    zero = (buckets[i] & 0x3FFF) == 0;

    buckets[i] = zero ? 0xFFFFFFFF : (add != 0) ? neg : pos;
    shifted = shifted + 0x2000;
    add = add - zeroAdd;
  }
}

void split16(uint32_t* buckets, I128* scalar) {
  uint64_t l, h;
  uint64_t mask = 0xFFFFL;
  uint32_t add=0, shifted=0, zeroAdd, pos, neg;
  bool     zero;

  l=i64x2_extract_l(scalar[0]);
  h=i64x2_extract_h(scalar[0]);
  buckets[0] = l & mask;
  buckets[1] = (l >> 16) & mask;
  buckets[2] = (l >> 32) & mask;
  buckets[3] = l >> 48;
  buckets[4] = h & mask;
  buckets[5] = (h >> 16) & mask;
  buckets[6] = (h >> 32) & mask;
  buckets[7] = h >> 48;

  l=i64x2_extract_l(scalar[1]);
  h=i64x2_extract_h(scalar[1]);
  buckets[8] = l & mask;
  buckets[9] = (l >> 16) & mask;
  buckets[10] = (l >> 32) & mask;
  buckets[11] = l >> 48;
  buckets[12] = h & mask;
  buckets[13] = (h >> 16) & mask;
  buckets[14] = (h >> 32) & mask;
  buckets[15] = h >> 48;

  for (int i = 0; i < 16; i++) {
    buckets[i] = buckets[i] + add;
    pos = shifted + buckets[i] - 1;
    neg = shifted + (buckets[i] ^ 0x8000FFFF);
    add = buckets[i] >> 15;
    zeroAdd = buckets[i] >> 16;
    zero = (buckets[i] & 0xFFFF) == 0;

    buckets[i] = zero ? 0xFFFFFFFF : (add != 0) ? neg : pos;
    shifted = shifted + 0x8000;
    add = add - zeroAdd;
  }
}

