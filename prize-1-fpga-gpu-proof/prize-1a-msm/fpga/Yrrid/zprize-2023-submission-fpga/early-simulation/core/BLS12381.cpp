#include <stdio.h>
#include <stdint.h>

class BLS12381 {
  public:
  class MainField {
    public:
    typedef uint64_t Words[6];

    static const uint32_t bits=381;
    static const Words    prime;
    static const Words    np;
    static const Words    rModP;
    static const Words    rSquaredModP;
    static const Words    rInvModP;
    static const Words    rCubedModP;
    static const Words    g1GeneratorX;
    static const Words    g1GeneratorY;

    Words l;

    MainField() {
      for(int i=0;i<6;i++)
        l[i]=0;
    }

    MainField(const Words& limbs) {
      for(int i=0;i<6;i++)
        l[i]=limbs[i];
    }
  };

  class OrderField {
    public:
    typedef uint64_t Words[4];

    static const uint32_t bits=255;
    static const Words    prime;
    static const Words    np;
    static const Words    rModP;
    static const Words    rSquaredModP;
    static const Words    rCubedModP;

    Words l;

    OrderField() {
      for(int i=0;i<4;i++)
        l[i]=0;
    }

    OrderField(const Words& limbs) {
      for(int i=0;i<4;i++)
        l[i]=limbs[i];
    }
  };
};

const uint64_t BLS12381::MainField::prime[6]={
  0xB9FEFFFFFFFFAAAB, 0x1EABFFFEB153FFFF, 0x6730D2A0F6B0F624,
  0x64774B84F38512BF, 0x4B1BA7B6434BACD7, 0x1A0111EA397FE69A
};

const uint64_t BLS12381::MainField::np[6]={
  0x89F3FFFCFFFCFFFD, 0x286ADB92D9D113E8, 0x16EF2EF0C8E30B48,
  0x19ECCA0E8EB2DB4C, 0x68B316FEE268CF58, 0xCEB06106FEAAFC94
};

const uint64_t BLS12381::MainField::rModP[6]={
  0x760900000002FFFD, 0xEBF4000BC40C0002, 0x5F48985753C758BA,
  0x77CE585370525745, 0x5C071A97A256EC6D, 0x15F65EC3FA80E493
};

const uint64_t BLS12381::MainField::rSquaredModP[6]={
  0xF4DF1F341C341746, 0x0A76E6A609D104F1, 0x8DE5476C4C95B6D5,
  0x67EB88A9939D83C0, 0x9A793E85B519952D, 0x11988FE592CAE3AA
};

const uint64_t BLS12381::MainField::rInvModP[6]={
  0xF4D38259380B4820, 0x7FE11274D898FAFB, 0x343EA97914956DC8,
  0x1797AB1458A88DE9, 0xED5E64273C4F538B, 0x14FEC701E8FB0CE9
};

const uint64_t BLS12381::MainField::rCubedModP[6]={
  0xED48AC6BD94CA1E0, 0x315F831E03A7ADF8, 0x9A53352A615E29DD,
  0x34C04E5E921E1761, 0x2512D43565724728, 0x0AA6346091755D4D
};

const uint64_t BLS12381::MainField::g1GeneratorX[6]={
  0xFB3AF00ADB22C6BB, 0x6C55E83FF97A1AEF, 0xA14E3A3F171BAC58,
  0xC3688C4F9774B905, 0x2695638C4FA9AC0F, 0x17F1D3A73197D794
};

const uint64_t BLS12381::MainField::g1GeneratorY[6]={
  0x0CAA232946C5E7E1, 0xD03CC744A2888AE4, 0x00DB18CB2C04B3ED,
  0xFCF5E095D5D00AF6, 0xA09E30ED741D8AE4, 0x08B3F481E3AAA0F1
};

const uint64_t BLS12381::OrderField::prime[4]={
  0xFFFFFFFF00000001, 0x53BDA402FFFE5BFE, 0x3339D80809A1D805, 0x73EDA753299D7D48
};

const uint64_t BLS12381::OrderField::np[4]={
  0xFFFFFFFEFFFFFFFF, 0x53BA5BFFFFFE5BFD, 0x181B2C170004EC06, 0x3D443AB0D7BF2839
};

const uint64_t BLS12381::OrderField::rModP[4]={
  0x00000001FFFFFFFE, 0x5884B7FA00034802, 0x998C4FEFECBC4FF5, 0x01824B159ACC5056F
};

const uint64_t BLS12381::OrderField::rSquaredModP[4]={
  0xC999E990F3F29C6D, 0x2B6CEDCB87925C23, 0x05D314967254398F, 0x00748D9D99F59FF11
};

const uint64_t BLS12381::OrderField::rCubedModP[4]={
  0xC62C1807439B73AF, 0x1B3E0D188CF06990, 0x73D13C71C7B5F418, 0x06E2A5BB9C8DB33E9
};