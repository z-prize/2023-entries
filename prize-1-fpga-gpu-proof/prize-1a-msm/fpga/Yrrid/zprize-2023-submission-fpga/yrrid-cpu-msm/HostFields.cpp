/***

Copyright (c) 2023-2024, Yrrid Software, Inc. All rights reserved.
Dual licensed under the MIT License or the Apache License, Version 2.0.
See LICENSE for details.

Author(s):  Niall Emmart

***/

#include <stdio.h>
#include <stdint.h>

/* Relatively fast for a portable library */

namespace Host {

typedef unsigned __int128 uint128_t;

uint128_t m(const uint64_t a, const uint64_t b) {
  uint128_t la=a, lb=b;

  return la*lb;
}

template<class Curve>
void load(typename Curve::Field& r, const uint64_t* data) {
  const uint64_t mask55=0x007FFFFFFFFFFFFFull;

  // hard coded for 7 limbs
  r[0]=data[0] & mask55;
  r[1]=((data[0]>>55) | (data[1]<<9)) & mask55;
  r[2]=((data[1]>>46) | (data[2]<<18)) & mask55;
  r[3]=((data[2]>>37) | (data[3]<<27)) & mask55;
  r[4]=((data[3]>>28) | (data[4]<<36)) & mask55;
  r[5]=((data[4]>>19) | (data[5]<<45)) & mask55;
  r[6]=data[5]>>10;
}

template<class Curve>
void store(uint64_t* data, const typename Curve::Field& f) {
  // hard coded for 7 limbs
  data[0]=f[0] | (f[1]<<55);
  data[1]=(f[1]>>9) | (f[2]<<46);
  data[2]=(f[2]>>18) | (f[3]<<37);
  data[3]=(f[3]>>27) | (f[4]<<28);
  data[4]=(f[4]>>36) | (f[5]<<19);
  data[5]=(f[5]>>45) | (f[6]<<10);
}

template<class Curve>
void zero(typename Curve::Field& r) {
  const uint32_t limbs=Curve::limbs;

  for(uint32_t i=0;i<limbs;i++)
    r[i]=0;
}

template<class Curve>
void copy(typename Curve::Field& r, const typename Curve::Field& f) {
  const uint32_t limbs=Curve::limbs;

  for(uint32_t i=0;i<limbs;i++)
    r[i]=f[i];
}

template<class Curve>
uint64_t limbOr(const typename Curve::Field& x) {
  const uint32_t limbs=Curve::limbs;

  uint64_t lor=x[0];

  for(uint32_t i=1;i<limbs;i++)
    lor=lor | x[i];
  return lor;
}

template<class Curve>
bool limbEq(const typename Curve::Field& a, const typename Curve::Field& b) {
  const uint32_t limbs=Curve::limbs;

  for(uint32_t i=0;i<limbs;i++)
    if(a[i]!=b[i])
      return false;
  return true;
}

template<class Curve>
bool add(typename Curve::Field& r, const typename Curve::Field& a, const typename Curve::Field& b) {
  const uint32_t limbs=Curve::limbs;
  const uint64_t mask55=0x007FFFFFFFFFFFFFull;

  int64_t acc=0;

  for(uint32_t i=0;i<limbs;i++) {
    acc=(acc>>55) + a[i] + b[i];
    r[i]=acc & mask55;
  }
  return (acc>>55)>=0;
}

template<class Curve>
bool sub(typename Curve::Field& r, const typename Curve::Field& a, const typename Curve::Field& b) {
  const uint32_t limbs=Curve::limbs;
  const uint64_t mask55=0x007FFFFFFFFFFFFFull;

  int64_t acc=0;

  for(uint32_t i=0;i<limbs;i++) {
    acc=(acc>>55) + a[i] - b[i];
    r[i]=acc & mask55;
  }
  return (acc>>55)>=0;
}

template<class Curve>
void swap(typename Curve::Field& a, typename Curve::Field& b) {
  const uint32_t limbs=Curve::limbs;

  uint64_t swap;

  for(uint32_t i=0;i<limbs;i++) {
    swap=a[i];
    a[i]=b[i];
    b[i]=swap;
  }
}

template<class Curve>
void shiftRightOne(typename Curve::Field r, const typename Curve::Field& f) {
  const uint32_t limbs=Curve::limbs;
  const uint64_t mask55=0x007FFFFFFFFFFFFFull;

  for(uint32_t i=0;i<limbs-1;i++)
    r[i]=((f[i]>>1) | (f[i+1]<<54)) & mask55;
  r[limbs-1]=f[limbs-1]>>1;
}

template<class Curve>
void add(typename Curve::Field& r, const typename Curve::Field& a, const typename Curve::Field& b, const typename Curve::Field& limit) {
  const uint32_t limbs=Curve::limbs;
  const uint64_t mask55=0x007FFFFFFFFFFFFFull;

  int64_t acc=0;

  for(uint32_t i=0;i<limbs;i++) {
    acc=(acc>>55) + a[i] + b[i] - limit[i];
    r[i]=acc & mask55;
  }

  if(acc>0) return;

  acc=0;
  for(uint32_t i=0;i<limbs;i++) {
    acc=(acc>>55) + r[i] + limit[i];
    r[i]=acc & mask55;
  }
}

template<class Curve>
void sub(typename Curve::Field& r, const typename Curve::Field& a, const typename Curve::Field& b, const typename Curve::Field& limit) {
  const uint32_t limbs=Curve::limbs;
  const uint64_t mask55=0x007FFFFFFFFFFFFFull;

  int64_t acc=0;

  for(uint32_t i=0;i<limbs;i++) {
    acc=(acc>>55) + a[i] - b[i];
    r[i]=acc & mask55;
  }

  if(acc>0) return;

  acc=0;
  for(uint32_t i=0;i<limbs;i++) {
    acc=(acc>>55) + r[i] + limit[i];
    r[i]=acc & mask55;
  }
}

template<class Curve>
void mul(typename Curve::Field& r, const typename Curve::Field& a, const typename Curve::Field& b, const typename Curve::Field& n) {
  const uint64_t mask55=0x007FFFFFFFFFFFFFull;

  uint64_t  q[7];
  uint128_t acc;

  // hard coded for 7 limbs
  acc=m(a[0], b[0]);
  q[0]=Curve::qTerm(acc);
  acc+=m(q[0], n[0]);

  acc=(acc>>55) + m(a[0], b[1]) + m(a[1], b[0])
                + m(q[0], n[1]);
  q[1]=Curve::qTerm(acc);
  acc+=m(q[1], n[0]);

  acc=(acc>>55) + m(a[0], b[2]) + m(a[1], b[1]) + m(a[2], b[0])
                + m(q[0], n[2]) + m(q[1], n[1]);
  q[2]=Curve::qTerm(acc);
  acc+=m(q[2], n[0]);

  acc=(acc>>55) + m(a[0], b[3]) + m(a[1], b[2]) + m(a[2], b[1]) + m(a[3], b[0])
                + m(q[0], n[3]) + m(q[1], n[2]) + m(q[2], n[1]);
  q[3]=Curve::qTerm(acc);
  acc+=m(q[3], n[0]);

  acc=(acc>>55) + m(a[0], b[4]) + m(a[1], b[3]) + m(a[2], b[2]) + m(a[3], b[1]) + m(a[4], b[0])
                + m(q[0], n[4]) + m(q[1], n[3]) + m(q[2], n[2]) + m(q[3], n[1]);
  q[4]=Curve::qTerm(acc);
  acc+=m(q[4], n[0]);

  acc=(acc>>55) + m(a[0], b[5]) + m(a[1], b[4]) + m(a[2], b[3]) + m(a[3], b[2]) + m(a[4], b[1]) + m(a[5], b[0])
                + m(q[0], n[5]) + m(q[1], n[4]) + m(q[2], n[3]) + m(q[3], n[2]) + m(q[4], n[1]);
  q[5]=Curve::qTerm(acc);
  acc+=m(q[5], n[0]);

  acc=(acc>>55) + m(a[0], b[6]) + m(a[1], b[5]) + m(a[2], b[4]) + m(a[3], b[3]) + m(a[4], b[2]) + m(a[5], b[1]) + m(a[6], b[0])
                + m(q[0], n[6]) + m(q[1], n[5]) + m(q[2], n[4]) + m(q[3], n[3]) + m(q[4], n[2]) + m(q[5], n[1]);
  q[6]=Curve::qTerm(acc);
  acc+=m(q[6], n[0]);

  acc=(acc>>55) + m(a[1], b[6]) + m(a[2], b[5]) + m(a[3], b[4]) + m(a[4], b[3]) + m(a[5], b[2]) + m(a[6], b[1])
                + m(q[1], n[6]) + m(q[2], n[5]) + m(q[3], n[4]) + m(q[4], n[3]) + m(q[5], n[2]) + m(q[6], n[1]);
  r[0]=acc & mask55;

  acc=(acc>>55) + m(a[2], b[6]) + m(a[3], b[5]) + m(a[4], b[4]) + m(a[5], b[3]) + m(a[6], b[2])
                + m(q[2], n[6]) + m(q[3], n[5]) + m(q[4], n[4]) + m(q[5], n[3]) + m(q[6], n[2]);
  r[1]=acc & mask55;

  acc=(acc>>55) + m(a[3], b[6]) + m(a[4], b[5]) + m(a[5], b[4]) + m(a[6], b[3])
                + m(q[3], n[6]) + m(q[4], n[5]) + m(q[5], n[4]) + m(q[6], n[3]);
  r[2]=acc & mask55;

  acc=(acc>>55) + m(a[4], b[6]) + m(a[5], b[5]) + m(a[6], b[4])
                + m(q[4], n[6]) + m(q[5], n[5]) + m(q[6], n[4]);
  r[3]=acc & mask55;

  acc=(acc>>55) + m(a[5], b[6]) + m(a[6], b[5])
                + m(q[5], n[6]) + m(q[6], n[5]);
  r[4]=acc & mask55;

  acc=(acc>>55) + m(a[6], b[6])
                + m(q[6], n[6]);
  r[5]=acc & mask55;
  r[6]=acc>>55;
}

template<class Curve>
void inv(typename Curve::Field& r, const typename Curve::Field& f, const typename Curve::Field& n) {
  typename Curve::Field A, B, U, V, ignore;

  // slow, but very easy to code
  copy<Curve>(A, f);
  copy<Curve>(B, n);
  copy<Curve>(U, Curve::constOne);
  zero<Curve>(V);

  while(limbOr<Curve>(A)!=0) {
    if((A[0] & 0x01)!=0) {
      if(!sub<Curve>(ignore, A, B)) {
        swap<Curve>(A, B);
        swap<Curve>(U, V);
      }
      sub<Curve>(A, A, B);
      if(!sub<Curve>(U, U, V))
        add<Curve>(U, U, n);
    }
    if((U[0] & 0x01)!=0)
      add<Curve>(U, U, n);

    shiftRightOne<Curve>(A, A);
    shiftRightOne<Curve>(U, U);
  }
  copy<Curve>(r, V);
}

class BLS12377G1 {
  public:
  typedef uint64_t Field[7];

  static const uint32_t limbs=7;
  static const Field    constN, const2N, constR, constRSquared, constRInv, constGeneratorX, constGeneratorY, constOne;
  static const Field    constTwistedEdwardsGeneratorX, constTwistedEdwardsGeneratorY, constTwistedEdwardsGeneratorT;
  static const Field    constTwistedEdwardsS, constTwistedEdwardsF, constTwistedEdwardsK, constTwistedEdwardsBInv;

  static uint64_t qTerm(uint128_t x) {
    const uint64_t mask55=0x007FFFFFFFFFFFFFull;

    uint64_t sx=(uint64_t)x;

    return sx*0x8BFFFFFFFFFFFull & mask55;
  }

  static void load(Field& r, const uint64_t* data) {
    Host::load<BLS12377G1>(r, data);
  }

  static void store(uint64_t* data, const Field& f) {
    Host::store<BLS12377G1>(data, f);
  }

  static void setZero(Field& r) {
    Host::zero<BLS12377G1>(r);
  }

  static void setOne(Field& r) {
    Host::copy<BLS12377G1>(r, constR);
  }

  static void setGeneratorX(Field& r) {
    Host::copy<BLS12377G1>(r, constGeneratorX);
  }

  static void setGeneratorY(Field& r) {
    Host::copy<BLS12377G1>(r, constGeneratorY);
  }

  static void setTwistedEdwardsGeneratorX(Field& r) {
    Host::copy<BLS12377G1>(r, constTwistedEdwardsGeneratorX);
  }

  static void setTwistedEdwardsGeneratorY(Field& r) {
    Host::copy<BLS12377G1>(r, constTwistedEdwardsGeneratorY);
  }

  static void setTwistedEdwardsGeneratorT(Field& r) {
    Host::copy<BLS12377G1>(r, constTwistedEdwardsGeneratorT);
  }

  static void setTwistedEdwardsS(Field& r) {
    Host::copy<BLS12377G1>(r, constTwistedEdwardsS);
  }

  static void setTwistedEdwardsF(Field& r) {
    Host::copy<BLS12377G1>(r, constTwistedEdwardsF);
  }

  static void setTwistedEdwardsK(Field& r) {
    Host::copy<BLS12377G1>(r, constTwistedEdwardsK);
  }

  static void setTwistedEdwardsBInv(Field& r) {
    Host::copy<BLS12377G1>(r, constTwistedEdwardsBInv);
  }

  static void set(Field& r, const Field& f) {
    Host::copy<BLS12377G1>(r, f);
  }

  static bool isZero(const Field& x) {
    if((x[0] & 0x00000FFFFFFFFFF0ull)!=0)
      return false;
    if(limbOr<BLS12377G1>(x)==0)
      return true;
    if(limbEq<BLS12377G1>(x, constN))
      return true;
    return limbEq<BLS12377G1>(x, const2N);
  }

  static void add(Field& r, const Field& a, const Field& b) {
    Host::add<BLS12377G1>(r, a, b, const2N);
  }

  static void sub(Field& r, const Field& a, const Field& b) {
    Host::sub<BLS12377G1>(r, a, b, const2N);
  }

  static void mul(Field& r, const Field& a, const Field& b) {
    Host::mul<BLS12377G1>(r, a, b, constN);
  }

  static void sqr(Field& r, const Field& f) {
    Host::mul<BLS12377G1>(r, f, f, constN);
  }

  static void inv(Field& r, const Field& f) {
    Host::mul<BLS12377G1>(r, f, constRInv, constN);
    Host::inv<BLS12377G1>(r, r, constN);
  }

  static void fromHostMontgomeryToDeviceMontgomery(Field& r, const Field& x) {
    Field reduced;

    // go from host Montgomery (R=2^385) to device Montgomery (R=2^384)
    if((x[0] & 0x01)!=0)
      Host::add<BLS12377G1>(r, x, constN);
    Host::shiftRightOne<BLS12377G1>(r, r);

    // remove any multiples of N
    while(Host::sub<BLS12377G1>(reduced, r, constN))
      set(r, reduced);
  }

  static void fromDeviceMontgomeryToHostMontgomery(Field& r, const Field& x) {
    // accepts values in the range of 0 to N-1
    // go from device Montgomery (R=2^384) to host Montgomery (R=2^385)
    Host::add<BLS12377G1>(r, x, x, constN);
  }

  static void toMontgomery(Field& r, const Field& x) {
    // from normal space to host Montgomery (R=2^385)
    Host::mul<BLS12377G1>(r, x, constRSquared, constN);
  }

  static void fromMontgomery(Field& r, const Field& x) {
    Field reduced;

    // go from host Montgomery (R=2^385) to normal space
    Host::mul<BLS12377G1>(r, x, constOne, constN);
    if(Host::sub<BLS12377G1>(reduced, r, constN))
      set(r, reduced);
  }
};

const uint64_t Host::BLS12377G1::constN[7] = {
  0x08C00000000001ull, 0x3A88600000010Aull, 0x3EE82520005C2Dull, 0x07A89C78F79B11ull, 0x1493B1A22D9F30ull, 0x1D58C760B80D94ull, 0x006B8E9185F144ull,
};

const uint64_t Host::BLS12377G1::const2N[7] = {
  0x11800000000002ull, 0x7510C000000214ull, 0x7DD04A4000B85Aull, 0x0F5138F1EF3622ull, 0x292763445B3E60ull, 0x3AB18EC1701B28ull, 0x00D71D230BE288ull,
};

const uint64_t Host::BLS12377G1::constR[7] = {
  0x1BFFFFFFFFFED0ull, 0x7E0DFFFFFEC40Bull, 0x4C53E9FF928A04ull, 0x67C63059F7DB3Aull, 0x109D0F69D2F6EDull, 0x26933D256FE00Full, 0x0046B330F17EFAull,
};

const uint64_t Host::BLS12377G1::constRSquared[7] = {
  0x10E1B250033487ull, 0x2ACD20218D8CB2ull, 0x57454626CFD672ull, 0x480FD90B5D2310ull, 0x2E6ABE55B1A1B0ull, 0x26E8F7E8C01328ull, 0x00026E3998A007ull,
};

const uint64_t Host::BLS12377G1::constRInv[7] = {
  0x094849A28934F4ull, 0x3BCDE65839F5EFull, 0x114AB05926DC41ull, 0x5FCB995E14B8F2ull, 0x249A8F7557A0EFull, 0x736EB89746CA4Cull, 0x0028425F886D60ull,
};

const uint64_t Host::BLS12377G1::constGeneratorX[7] = {
  0x1E6772EE48A3E8ull, 0x5DCC5A75596098ull, 0x234D2886EE2A6Eull, 0x25E16985C1551Cull, 0x4A4E4EC2CC5C88ull, 0x53725F5E331BC1ull, 0x0052090359B0FEull,
};

const uint64_t Host::BLS12377G1::constGeneratorY[7] = {
  0x272C3F719703E6ull, 0x35317D12B6E206ull, 0x56A6D7AA50031Cull, 0x655CD18FAFAF3Dull, 0x2502984F093C5Aull, 0x1C4BB0F76D0075ull, 0x003ED49993181Eull,
};

const uint64_t Host::BLS12377G1::constTwistedEdwardsGeneratorX[7] = {
  0x564366CA9682FEull, 0x7E2874F87F816Eull, 0x76F9CEE1B8217Full, 0x1426B02B702072ull, 0x0FA074AC4BCF07ull, 0x18EFCB70DA2D47ull, 0x0059D6C8145619ull,
};

const uint64_t Host::BLS12377G1::constTwistedEdwardsGeneratorY[7] = {
  0x092E79DE2A203Aull, 0x4846C7C51158E4ull, 0x32EFD64A75A2F6ull, 0x63945CD90A86B8ull, 0x6E12F9B6338AFFull, 0x5DE5CAA351BBD3ull, 0x00140D48D1ECE7ull,
};

const uint64_t Host::BLS12377G1::constTwistedEdwardsGeneratorT[7] = {
  0x21C943223B84B3ull, 0x6B7087D252C268ull, 0x19D0428A0F2419ull, 0x336B35BFF60315ull, 0x5D23B2904DF5C7ull, 0x49F7AB5D83F7BBull, 0x001652239F8771ull,
};

const uint64_t Host::BLS12377G1::constTwistedEdwardsS[7] = {
  0x4AD535B0634143ull, 0x15FFDAD47238ADull, 0x6DB2DEAACCE952ull, 0x305D971CAFD35Aull, 0x4B32595C438BAAull, 0x27D87A2D051A9Dull, 0x005C1497F4A4BAull,
};

const uint64_t Host::BLS12377G1::constTwistedEdwardsF[7] = {
  0x6F5571D5AC0BC7ull, 0x14BF29D4B1EAE3ull, 0x69BA3A09B155AFull, 0x22496B042AAB71ull, 0x05F0CBCF513C78ull, 0x0BB5E05EB072F9ull, 0x0032AE105714F5ull,
};

const uint64_t Host::BLS12377G1::constTwistedEdwardsK[7] = {
  0x0E3D08894E0D8Bull, 0x534DA3EAA3EABFull, 0x550F775D2EFAD3ull, 0x578315E7FD3CBDull, 0x20941910B39088ull, 0x054A364FBF35D1ull, 0x0050946FF80E6Dull,
};

const uint64_t Host::BLS12377G1::constTwistedEdwardsBInv[7] = {
  0x4EFFA11129C3C7ull, 0x4CEED07D56A7F4ull, 0x4B4851C066039Bull, 0x01C78C642043EDull, 0x386FA8D06F649Full, 0x3CD7DFC59F34B0ull, 0x003D20A4D20BA6ull,
};

const uint64_t Host::BLS12377G1::constOne[7] = {
  0x00000000000001ull, 0x00000000000000ull, 0x00000000000000ull, 0x00000000000000ull, 0x00000000000000ull, 0x00000000000000ull, 0x00000000000000ull,
};

class BLS12381G1 {
  public:
  typedef uint64_t Field[7];

  static const uint32_t limbs=7;
  static const Field    constN, const2N, constR, constRSquared, constRInv, constGeneratorX, constGeneratorY, constOne;

  static uint64_t qTerm(uint128_t x) {
    const uint64_t mask55=0x007FFFFFFFFFFFFFull;

    uint64_t sx=(uint64_t)x;

    return sx*0x73FFFCFFFCFFFDull & mask55;
  }

  static void load(Field& r, const uint64_t* data) {
    Host::load<BLS12381G1>(r, data);
  }

  static void store(uint64_t* data, const Field& f) {
    Host::store<BLS12381G1>(data, f);
  }

  static void setZero(Field& r) {
    Host::zero<BLS12381G1>(r);
  }

  static void setOne(Field& r) {
    Host::copy<BLS12381G1>(r, constR);
  }

  static void setGeneratorX(Field& r) {
    Host::copy<BLS12381G1>(r, constGeneratorX);
  }

  static void setGeneratorY(Field& r) {
    Host::copy<BLS12381G1>(r, constGeneratorY);
  }

  static void set(Field& r, const Field& f) {
    Host::copy<BLS12381G1>(r, f);
  }

  static bool isZero(const Field& x) {
    if(((x[0]-0x10000) & 0xFFFFFFFE0000ull)!=0xFFFFFFFE0000ull)
      return false;
    if(limbOr<BLS12381G1>(x)==0)
      return true;
    if(limbEq<BLS12381G1>(x, constN))
      return true;
    return limbEq<BLS12381G1>(x, const2N);
  }

  static void add(Field& r, const Field& a, const Field& b) {
    Host::add<BLS12381G1>(r, a, b, const2N);
  }

  static void sub(Field& r, const Field& a, const Field& b) {
    Host::sub<BLS12381G1>(r, a, b, const2N);
  }

  static void mul(Field& r, const Field& a, const Field& b) {
    Host::mul<BLS12381G1>(r, a, b, constN);
  }

  static void sqr(Field& r, const Field& f) {
    Host::mul<BLS12381G1>(r, f, f, constN);
  }

  static void inv(Field& r, const Field& f) {
    Host::mul<BLS12381G1>(r, f, constRInv, constN);
    Host::inv<BLS12381G1>(r, r, constN);
  }

  static void fromHostMontgomeryToDeviceMontgomery(Field& r, const Field& x) {
    Field reduced;

    // go from host Montgomery (R=2^385) to device Montgomery (R=2^384)
    if((x[0] & 0x01)!=0)
      Host::add<BLS12381G1>(r, x, constN);
    Host::shiftRightOne<BLS12381G1>(r, r);

    // remove any multiples of N
    while(Host::sub<BLS12381G1>(reduced, r, constN))
      set(r, reduced);
  }

  static void fromDeviceMontgomeryToHostMontgomery(Field& r, const Field& x) {
    // accepts values in the range of 0 to N-1
    // go from device Montgomery (R=2^384) to host Montgomery (R=2^385)
    Host::add<BLS12381G1>(r, x, x, constN);
  }

  static void toMontgomery(Field& r, const Field& x) {
    // from normal space to host Montgomery (R=2^385)
    Host::mul<BLS12381G1>(r, x, constRSquared, constN);
  }

  static void fromMontgomery(Field& r, const Field& x) {
    Field reduced;

    // go from host Montgomery (R=2^385) to normal space
    Host::mul<BLS12381G1>(r, x, constOne, constN);
    if(Host::sub<BLS12381G1>(reduced, r, constN))
      set(r, reduced);
  }
};

const uint64_t Host::BLS12381G1::constN[7] = {
  0x7EFFFFFFFFAAABull, 0x7FFD62A7FFFF73ull, 0x03DAC3D8907AAFull, 0x1C2895FB398695ull, 0x3ACD764774B84Full, 0x53496374F6C869ull, 0x0680447A8E5FF9ull,
};

const uint64_t Host::BLS12381G1::const2N[7] = {
  0x7DFFFFFFFF5556ull, 0x7FFAC54FFFFEE7ull, 0x07B587B120F55Full, 0x38512BF6730D2Aull, 0x759AEC8EE9709Eull, 0x2692C6E9ED90D2ull, 0x0D0088F51CBFF3ull,
};

const uint64_t Host::BLS12381G1::constR[7] = {
  0x1300000006554Full, 0x0031AD88000A64ull, 0x36C376ED46E4F0ull, 0x68FCDE5ABB02F0ull, 0x22C038B256521Eull, 0x518D9E51AF202Cull, 0x047AEAE76EE078ull,
};

const uint64_t Host::BLS12381G1::constRSquared[7] = {
  0x7E7CD070D107C2ull, 0x353589382790BEull, 0x3D13D3BC2FB20Eull, 0x3B5F4C1B499BC3ull, 0x6FB06D6BF8B9C6ull, 0x2EBA75B55549B9ull, 0x049806F0760AF0ull,
};

const uint64_t Host::BLS12381G1::constRInv[7] = {
  0x69C12C9C05A410ull, 0x1274D898FAFBF4ull, 0x72292ADB90FFC2ull, 0x62A237A4D0FAA5ull, 0x7A9C58BCBD58A2ull, 0x4E9ED5E64273C4ull, 0x029FD8E03D1F61ull,
};

const uint64_t Host::BLS12381G1::constGeneratorX[7] = {
  0x680F21FAA66D81ull, 0x71A10333FFD5FEull, 0x34C719357B460Full, 0x20E5DE761B72C7ull, 0x43BAA7CE58A16Full, 0x0A321026BC400Dull, 0x0280772640A604ull,
};

const uint64_t Host::BLS12381G1::constGeneratorY[7] = {
  0x5927AA19CE44E2ull, 0x0C69E463F63AEAull, 0x1AB8392E746113ull, 0x0405194DD595F1ull, 0x0B380A358B052Aull, 0x1A8387230FEB40ull, 0x05DE1F7E280451ull,
};

const uint64_t Host::BLS12381G1::constOne[7] = {
  0x00000000000001ull, 0x00000000000000ull, 0x00000000000000ull, 0x00000000000000ull, 0x00000000000000ull, 0x00000000000000ull, 0x00000000000000ull,
};

} /* namespace Host */