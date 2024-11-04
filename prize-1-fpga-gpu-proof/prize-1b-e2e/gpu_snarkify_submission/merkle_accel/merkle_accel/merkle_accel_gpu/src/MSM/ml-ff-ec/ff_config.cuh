// finite field definitions

#pragma once

namespace bls12_377 {

struct ff_config_base {
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

__device__ __constant__ uint32_t inv_p = 0xffffffff;

} // namespace bls12_377

//#define ff_config_p bls12_377::ff_config_base
//#define ff_config_q bls12_377::ff_config_scalar
//#define inv_p bls12_377::inv_p
//#define inv_q bls12_377::inv_q
