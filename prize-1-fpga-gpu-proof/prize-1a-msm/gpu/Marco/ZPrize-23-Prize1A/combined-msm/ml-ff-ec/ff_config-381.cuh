// finite field definitions

#pragma once

#include "ff_storage.cuh"

namespace bls12_381 {

struct ff_config_base {
  // field structure size = 12 * 32 bit
  static constexpr unsigned limbs_count = 12;
//  static constexpr ff_storage<limbs_count> modulus = {0x00000001, 0x8508c000, 0x30000000, 0x170b5d44, 0xba094800, 0x1ef3622f,
//                                                      0x00f5138f, 0x1a22d9f3, 0x6ca1493b, 0xc63b05c0, 0x17c510ea, 0x01ae3a46};
//  // modulus*2 = 517328852025938188021305467389787067072787025509829321079768525333440936696681645549937776279146720248880642916354
//  static constexpr ff_storage<limbs_count> modulus_2 = {0x00000002, 0x0a118000, 0x60000001, 0x2e16ba88, 0x74129000, 0x3de6c45f,
//                                                        0x01ea271e, 0x3445b3e6, 0xd9429276, 0x8c760b80, 0x2f8a21d5, 0x035c748c};
//  // modulus*4 = 1034657704051876376042610934779574134145574051019658642159537050666881873393363291099875552558293440497761285832708
//  static constexpr ff_storage<limbs_count> modulus_4 = {0x00000004, 0x14230000, 0xc0000002, 0x5c2d7510, 0xe8252000, 0x7bcd88be,
//                                                        0x03d44e3c, 0x688b67cc, 0xb28524ec, 0x18ec1701, 0x5f1443ab, 0x06b8e918};

  // modulus = 4002409555221667393417789825735904156556882819939007885332058136124031650490837864442687629129015664037894272559787
        static constexpr ff_storage<limbs_count> modulus = {0xffffaaab, 0xb9feffff, 0xb153ffff, 0x1eabfffe,
                                                         0xf6b0f624, 0x6730d2a0, 0xf38512bf, 0x64774b84,
                                                         0x434bacd7, 0x4b1ba7b6, 0x397fe69a, 0x1a0111ea};
        static constexpr ff_storage<limbs_count> modulus_2 = {0xffff5556, 0x73fdffff, 0x62a7ffff, 0x3d57fffd,
                                                           0xed61ec48, 0xce61a541, 0xe70a257e, 0xc8ee9709,
                                                           0x869759ae, 0x96374f6c, 0x72ffcd34, 0x340223d4};
        static constexpr ff_storage<limbs_count> modulus_4 = {0xfffeaaac, 0xe7fbffff, 0xc54ffffe, 0x7aaffffa,
                                                           0xdac3d890, 0x9cc34a83, 0xce144afd, 0x91dd2e13,
                                                           0x0d2eb35d, 0x2c6e9ed9, 0xe5ff9a69, 0x680447a8};

  // modulus^2
  static constexpr ff_storage_wide<limbs_count> modulus_squared = {
                0x1c718e39, 0x26aa0000, 0x76382eab, 0x7ced6b1d, 0x62113cfd, 0x162c3383, 0x3e71b743, 0x66bf91ed,
                0x7091a049, 0x292e85a8, 0x86185c7b, 0x1d68619c, 0x0978ef01, 0xf5314933, 0x16ddca6e, 0x50a62cfd,
                0x349e8bd0, 0x66e59e49, 0x0e7046b4, 0xe2dc90e5, 0xa22f25e9, 0x4bd278ea, 0xb8c35fc7, 0x02a437a4};
  // 2*modulus^2
  static constexpr ff_storage_wide<limbs_count> modulus_squared_2 = {
                0x38e31c72, 0x4d540000, 0xec705d56, 0xf9dad63a, 0xc42279fa, 0x2c586706, 0x7ce36e86, 0xcd7f23da,
                0xe1234092, 0x525d0b50, 0x0c30b8f6, 0x3ad0c339, 0x12f1de02, 0xea629266, 0x2dbb94dd, 0xa14c59fa,
                0x693d17a0, 0xcdcb3c92, 0x1ce08d68, 0xc5b921ca, 0x445e4bd3, 0x97a4f1d5, 0x7186bf8e, 0x05486f49};
  // 4*modulus^2
  static constexpr ff_storage_wide<limbs_count> modulus_squared_4 = {
                0x71c638e4, 0x9aa80000, 0xd8e0baac, 0xf3b5ac75, 0x8844f3f5, 0x58b0ce0d, 0xf9c6dd0c, 0x9afe47b4,
                0xc2468125, 0xa4ba16a1, 0x186171ec, 0x75a18672, 0x25e3bc04, 0xd4c524cc, 0x5b7729bb, 0x4298b3f4,
                0xd27a2f41, 0x9b967924, 0x39c11ad1, 0x8b724394, 0x88bc97a7, 0x2f49e3aa, 0xe30d7f1d, 0x0a90de92};

  static constexpr ff_storage<limbs_count> r2 = {
//          0x9400cd22, 0xb786686c, 0xb00431b1, 0x0329fcaa, 0x62d6b46d, 0x22a5f111,
//                                                 0x827dc3ac, 0xbfdf7d03, 0x41790bf9, 0x837e92f0, 0x1e914b88, 0x006dfccb

//      0xf4df1f34, 0x1c341746, 0x0a76e6a6, 0x09d104f1,
//      0x8de5476c, 0x4c95b6d5, 0x67eb88a9, 0x939d83c0,
//      0x9a793e85, 0xb519952d, 0x11988fe5, 0x92cae3aa

          0x1c341746, 0xf4df1f34, 0x9d104f1, 0xa76e6a6, 0x4c95b6d5, 0x8de5476c, 0x939d83c0, 0x67eb88a9, 0xb519952d, 0x9a793e85, 0x92cae3aa, 0x11988fe5
  };


  // inv
  static constexpr uint32_t inv = 0xfffcfffd;
  // 1 in montgomery form
  static constexpr ff_storage<limbs_count> one = {0x0002fffd, 0x76090000, 0xc40c0002, 0xebf4000b,
                                                  0x53c758ba, 0x5f489857, 0x70525745, 0x77ce5853,
                                                  0xa256ec6d, 0x5c071a97, 0xfa80e493, 0x15f65ec3};
  static constexpr unsigned modulus_bits_count = 381;
  static constexpr unsigned B_VALUE = 1;
};

//    __device__ const __constant__ uint32_t inv_p = 0x7FFFFFFF;
//    __device__ const __constant__ uint32_t inv_p = 0xfffcfffd;
    extern __device__ __constant__ uint32_t inv_p;

//__device__ const __constant__ uint32_t inv_p = 0xffffffff;

struct ff_config_scalar {
  // field structure size = 8 * 32 bit
  static constexpr unsigned limbs_count = 8;
    static constexpr ff_storage<limbs_count> modulus = {0x00000001, 0xffffffff, 0xfffe5bfe, 0x53bda402,
                                                     0x09a1d805, 0x3339d808, 0x299d7d48, 0x73eda753};
    static constexpr ff_storage<limbs_count> modulus_2 = {0x00000002, 0xfffffffe, 0xfffcb7fd, 0xa77b4805,
                                                       0x1343b00a, 0x6673b010, 0x533afa90, 0xe7db4ea6};
    static constexpr ff_storage<limbs_count> modulus_4 = {0x00000004, 0xfffffffc, 0xfff96ffb, 0x4ef6900b,
                                                       0x26876015, 0xcce76020, 0xa675f520, 0xcfb69d4c};


    // modulus^2
    static constexpr ff_storage_wide<limbs_count> modulus_squared = {
            0x00000001, 0xfffffffe, 0xfffcb7fe, 0xa77e9007, 0x1cdbb005, 0x698ae002, 0x5433f7b8, 0x48aa415e,
            0x4aa9c661, 0xc2611f6f, 0x59934a1d, 0x0e9593f9, 0xef2cc20f, 0x520c13db, 0xf4bc2778, 0x347f60f3};
    // 4*modulus^2
    static constexpr ff_storage_wide<limbs_count> modulus_squared_2 = {
            0x00000002, 0xfffffffc, 0xfff96ffd, 0x4efd200f, 0x39b7600b, 0xd315c004, 0xa867ef70, 0x915482bc,
            0x95538cc2, 0x84c23ede, 0xb326943b, 0x1d2b27f2, 0xde59841e, 0xa41827b7, 0xe9784ef0, 0x68fec1e7};
    // 4*modulus^2
    static constexpr ff_storage_wide<limbs_count> modulus_squared_4 = {
            0x00000004, 0xfffffff8, 0xfff2dffb, 0x9dfa401f, 0x736ec016, 0xa62b8008, 0x50cfdee1, 0x22a90579,
            0x2aa71985, 0x09847dbd, 0x664d2877, 0x3a564fe5, 0xbcb3083c, 0x48304f6f, 0xd2f09de1, 0xd1fd83cf};


  static constexpr ff_storage<limbs_count> r2 = {
          0xf3f29c6d, 0xc999e990, 0x87925c23, 0x2b6cedcb,
          0x7254398f, 0x05d31496, 0x9f59ff11, 0x0748d9d9
  };
    // inv
  static constexpr uint32_t inv = 0xffffffff;
  // 1 in montgomery form
  static constexpr ff_storage<limbs_count> one = {0xfffffffe, 0x00000001, 0x00034802, 0x5884b7fa, 0xecbc4ff5, 0x998c4fef, 0xacc5056f, 0x1824b159};
  static constexpr unsigned modulus_bits_count = 253;
};

//__device__ const __constant__ uint32_t inv_q = 0xffffffff;
    extern __device__ __constant__ uint32_t inv_q;

} // namespace bls12_381

#define ff_config381_p bls12_381::ff_config_base
#define ff_config381_q bls12_381::ff_config_scalar
//#define inv381_p bls12_381::inv_p
//#define inv381_q bls12_381::inv_q
