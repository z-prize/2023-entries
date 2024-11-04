#include "ml-ff-ec/ff_storage.cuh"
#include "ml-ff-ec/common.cuh"
#include "ml-ff-ec/memory.cuh"
#include "ml-ff-ec/ptx.cuh"
#include "ml-ff-ec/carry_chain.cuh"
#include "ml-ff-ec/ec.cuh"
#include "ml-ff-ec/ff_dispatch_st.cuh"

namespace ML {
  struct FF_BLS12377 {
    static constexpr unsigned limbs_count = 12;

    static constexpr ff_storage<limbs_count> modulus = {0x00000001, 0x8508c000, 0x30000000, 0x170b5d44, 0xba094800, 0x1ef3622f,
                                                        0x00f5138f, 0x1a22d9f3, 0x6ca1493b, 0xc63b05c0, 0x17c510ea, 0x01ae3a46};

    static constexpr ff_storage<limbs_count> modulus_2 = {0x00000002, 0x0a118000, 0x60000001, 0x2e16ba88, 0x74129000, 0x3de6c45f,
                                                          0x01ea271e, 0x3445b3e6, 0xd9429276, 0x8c760b80, 0x2f8a21d5, 0x035c748c};

    static constexpr ff_storage<limbs_count> modulus_4 = {0x00000004, 0x14230000, 0xc0000002, 0x5c2d7510, 0xe8252000, 0x7bcd88be,
                                                          0x03d44e3c, 0x688b67cc, 0xb28524ec, 0x18ec1701, 0x5f1443ab, 0x06b8e918};

    static constexpr ff_storage_wide<limbs_count> modulus_squared = {
        0x00000001, 0x0a118000, 0xf0000001, 0x7338d254, 0x2e1bd800, 0x4ada268f, 0x35f1c09a, 0x6bcbfbd2, 0x58638c9d, 0x318324b9, 0x8bb70ae0, 0x460aaaaa,
        0x502a4d6c, 0xc014e712, 0xb90660cd, 0x09d018af, 0x3dda4d5c, 0x1f5e7141, 0xa4aee93f, 0x4bb8b87d, 0xb361263c, 0x2256913b, 0xd0bbaffb, 0x0002d307};
 
    static constexpr ff_storage_wide<limbs_count> modulus_squared_2 = {
        0x00000002, 0x14230000, 0xe0000002, 0xe671a4a9, 0x5c37b000, 0x95b44d1e, 0x6be38134, 0xd797f7a4, 0xb0c7193a, 0x63064972, 0x176e15c0, 0x8c155555,
        0xa0549ad8, 0x8029ce24, 0x720cc19b, 0x13a0315f, 0x7bb49ab8, 0x3ebce282, 0x495dd27e, 0x977170fb, 0x66c24c78, 0x44ad2277, 0xa1775ff6, 0x0005a60f};

    static constexpr ff_storage_wide<limbs_count> modulus_squared_4 = {
        0x00000004, 0x28460000, 0xc0000004, 0xcce34953, 0xb86f6001, 0x2b689a3c, 0xd7c70269, 0xaf2fef48, 0x618e3275, 0xc60c92e5, 0x2edc2b80, 0x182aaaaa,
        0x40a935b1, 0x00539c49, 0xe4198337, 0x274062be, 0xf7693570, 0x7d79c504, 0x92bba4fc, 0x2ee2e1f6, 0xcd8498f1, 0x895a44ee, 0x42eebfec, 0x000b4c1f};

    static constexpr ff_storage<limbs_count> r2 = {0x9400cd22, 0xb786686c, 0xb00431b1, 0x0329fcaa, 0x62d6b46d, 0x22a5f111,
                                                   0x827dc3ac, 0xbfdf7d03, 0x41790bf9, 0x837e92f0, 0x1e914b88, 0x006dfccb};

    static constexpr uint32_t inv = 0xffffffff;

    static constexpr ff_storage<limbs_count> one = {0xffffff68, 0x02cdffff, 0x7fffffb1, 0x51409f83, 0x8a7d3ff2, 0x9f7db3a9,
                                                    0x6e7c6305, 0x7b4e97b7, 0x803c84e8, 0x4cf495bf, 0xe2fdf49a, 0x008d6661};
    static constexpr unsigned modulus_bits_count = 377;
    static constexpr unsigned B_VALUE = 1;
  };

  __device__ __constant__ uint32_t FF_BLS12377_INV = 0xffffffff;

  struct FF_BLS12381 {
    static constexpr unsigned limbs_count = 12;

    static constexpr ff_storage<limbs_count> modulus = {0xFFFFAAAB, 0xB9FEFFFF, 0xB153FFFF, 0x1EABFFFE, 0xF6B0F624, 0x6730D2A0,
                                                        0xF38512BF, 0x64774B84, 0x434BACD7, 0x4B1BA7B6, 0x397FE69A, 0x1A0111EA};

    static constexpr ff_storage<limbs_count> modulus_2 = {0xFFFF5556, 0x73FDFFFF, 0x62A7FFFF, 0x3D57FFFD, 0xED61EC48, 0xCE61A541,
                                                          0xE70A257E, 0xC8EE9709, 0x869759AE, 0x96374F6C, 0x72FFCD34, 0x340223D4};

    static constexpr ff_storage<limbs_count> modulus_4 = {0xFFFEAAAC, 0xE7FBFFFF, 0xC54FFFFE, 0x7AAFFFFA, 0xDAC3D890, 0x9CC34A83,
                                                          0xCE144AFD, 0x91DD2E13, 0x0D2EB35D, 0x2C6E9ED9, 0xE5FF9A69, 0x680447A8};

    static constexpr ff_storage_wide<limbs_count> modulus_squared = {
        0x1C718E39, 0x26AA0000, 0x76382EAB, 0x7CED6B1D, 0x62113CFD, 0x162C3383, 0x3E71B743, 0x66BF91ED, 0x7091A049, 0x292E85A8, 0x86185C7B, 0x1D68619C,
        0x0978EF01, 0xF5314933, 0x16DDCA6E, 0x50A62CFD, 0x349E8BD0, 0x66E59E49, 0x0E7046B4, 0xE2DC90E5, 0xA22F25E9, 0x4BD278EA, 0xB8C35FC7, 0x02A437A4};

    static constexpr ff_storage_wide<limbs_count> modulus_squared_2 = {
        0x38E31C72, 0x4D540000, 0xEC705D56, 0xF9DAD63A, 0xC42279FA, 0x2C586706, 0x7CE36E86, 0xCD7F23DA, 0xE1234092, 0x525D0B50, 0x0C30B8F6, 0x3AD0C339,
        0x12F1DE02, 0xEA629266, 0x2DBB94DD, 0xA14C59FA, 0x693D17A0, 0xCDCB3C92, 0x1CE08D68, 0xC5B921CA, 0x445E4BD3, 0x97A4F1D5, 0x7186BF8E, 0x05486F49};

    static constexpr ff_storage_wide<limbs_count> modulus_squared_4 = {
        0x71C638E4, 0x9AA80000, 0xD8E0BAAC, 0xF3B5AC75, 0x8844F3F5, 0x58B0CE0D, 0xF9C6DD0C, 0x9AFE47B4, 0xC2468125, 0xA4BA16A1, 0x186171EC, 0x75A18672,
        0x25E3BC04, 0xD4C524CC, 0x5B7729BB, 0x4298B3F4, 0xD27A2F41, 0x9B967924, 0x39C11AD1, 0x8B724394, 0x88BC97A7, 0x2F49E3AA, 0xE30D7F1D, 0x0A90DE92};

    static constexpr ff_storage<limbs_count> r2 = {0x1C341746, 0xF4DF1F34, 0x09D104F1, 0x0A76E6A6, 0x4C95B6D5, 0x8DE5476C,
                                                   0x939D83C0, 0x67EB88A9, 0xB519952D, 0x9A793E85, 0x92CAE3AA, 0x11988FE5};

    static constexpr uint32_t inv = 0xFFFCFFFD;

    static constexpr ff_storage<limbs_count> one = {0x0002FFFD, 0x76090000, 0xC40C0002, 0xEBF4000B, 0x53C758BA, 0x5F489857,
                                                    0x70525745, 0x77CE5853, 0xA256EC6D, 0x5C071A97, 0xFA80E493, 0x15F65EC3};
 
    static constexpr unsigned modulus_bits_count = 381;
    static constexpr unsigned B_VALUE = 4;
  };

  __device__ __constant__ uint32_t FF_BLS12381_INV = 0xFFFCFFFD;

  template<class fd_p>
  class msm {
    public:
    typedef ml_ec<fd_p> curve;
    typedef typename curve::storage storage;
    typedef typename curve::field field;
    typedef typename curve::point_affine point_affine;
    typedef typename curve::point_jacobian point_jacobian;
    typedef typename curve::point_xyzz point_xyzz;
    typedef typename curve::point_projective point_projective;
  };

}  /* namespace ML */
