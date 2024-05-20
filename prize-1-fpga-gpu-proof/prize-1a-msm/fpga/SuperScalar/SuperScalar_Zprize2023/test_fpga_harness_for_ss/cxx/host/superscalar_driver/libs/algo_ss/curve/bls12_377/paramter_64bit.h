#ifndef __CURVE_BLS12_377_PARAMTER_64BIT_H__
#define __CURVE_BLS12_377_PARAMTER_64BIT_H__

#include "configuration.h"
#include "field.h"
#include "field_storage.h"

// 64 bit
namespace CURVE_BLS12_377 
{
  #define NLIMBS(bits)   (bits/LIMB_T_BITS)

  // fp
  struct fp_configuration_64bit
  {
    // field structure size = 6 * 64 bit
    static constexpr unsigned bits_count = 377;
    static constexpr FIELD_RADIX_TYPE radix_type = RADIX_TYPE_64BIT;
    static constexpr unsigned limbs_bits = 64;
    static constexpr unsigned limbs_count = 6;
    static constexpr unsigned bytes_count = 48;

    static constexpr field_storage<uint64_t, limbs_count> MODULAR = 
    {
        0x8508c00000000001,
        0x170b5d4430000000,
        0x1ef3622fba094800,
        0x1a22d9f300f5138f,
        0xc63b05c06ca1493b,
        0x01ae3a4617c510ea
    };

    // modulus inv
    static constexpr uint64_t MODULAR_INV = 0x8508bfffffffffff;
  
    static constexpr field_storage<uint64_t, limbs_count> R1 = 
    {
        0x02cdffffffffff68,
        0x51409f837fffffb1,
        0x9f7db3a98a7d3ff2,
        0x7b4e97b76e7c6305,
        0x4cf495bf803c84e8,
        0x008d6661e2fdf49a
    };

    static constexpr field_storage<uint64_t, limbs_count> ZERO = {0x0};
    static constexpr field_storage<uint64_t, limbs_count> ONE { R1 };

    static constexpr field_storage<uint64_t, limbs_count> R2 = 
    {
        0xb786686c9400cd22,
        0x0329fcaab00431b1,
        0x22a5f11162d6b46d,
        0xbfdf7d03827dc3ac,
        0x837e92f041790bf9,
        0x006dfccb1e914b88,
    };

    static constexpr field_storage<uint64_t, limbs_count> UNITY_ONE { 0x1 };

    static constexpr field_storage<uint64_t, limbs_count> EXED_ADD_K = {
        0x33f6817bbb58e892,
        0x3fd664caff054e5d,
        0x6ad7181ad6752eda,
        0x55eabefa47ff7cfe,
        0xacf1711193fe9656,
        0x008d1a81199ed6ff,
      };

    static constexpr field_storage<uint64_t, limbs_count> EXED_ALPHA = {
        0x5892506da58478da,
        0x133366940ac2a74b,
        0x9b64a150cdf726cf,
        0x5cc426090a9c587e,
        0x5cf848adfdcd640c,
        0x004702bf3ac02380,
      };

    static constexpr field_storage<uint64_t, limbs_count> EXED_BETA = {
        0x0320f2218b83be45,
        0x8c9ecfc2899abb8a,
        0x84b44205f494c488,
        0xc1a6e49a336eb9e7,
        0xa93aec49e2b475ca,
        0x01295b55161de008,
      };

    static constexpr field_storage<uint64_t, limbs_count> EXED_COEFF_SQRT_NEG_A = {
        0x90946f8df6255d75,
        0x74b2abaf9531916c,
        0x983c3c761f381c4b,
        0x50efb8d38ce958ab,
        0x10f5ee8891b3dd54,
        0x01167ec8f0a0b908,
      };

    static constexpr field_storage<uint64_t, limbs_count> EXED_COEFF_SQRT_NEG_A_INV = {
        0x5945fd4fd41bc23d,
        0xb11cec706559b338,
        0x853e6bcf722fdb39,
        0x53fa5979a887936b,
        0xfe77e8731018491a,
        0x00dee4cdb8c57a82,
      };
  };


  // fr
  struct fr_configuration_64bit
  {
    // field structure size = 4 * 64 bit
    static constexpr unsigned bits_count = 253;
    static constexpr FIELD_RADIX_TYPE radix_type = RADIX_TYPE_64BIT;
    static constexpr unsigned limbs_bits = 64;
    static constexpr unsigned limbs_count = 4;
    static constexpr unsigned bytes_count = 32;

    static constexpr field_storage<uint64_t, limbs_count> MODULAR = 
    {
        0x0A11800000000001,
        0x59AA76FED0000001,
        0x60B44D1E5C37B001,
        0x12AB655E9A2CA556
    };

    // modulus inv
    static constexpr uint64_t MODULAR_INV = 0x0A117FFFFFFFFFFF;

    static constexpr field_storage<uint64_t, limbs_count> R1 = 
    {
        0x7d1c7ffffffffff3,
        0x7257f50f6ffffff2,
        0x16d81575512c0fee,
        0x0d4bda322bbb9a9d
    };

    static constexpr field_storage<uint64_t, limbs_count> ZERO = {0x0};
    static constexpr field_storage<uint64_t, limbs_count> ONE { R1 };

    static constexpr field_storage<uint64_t, limbs_count> R2 = 
    {
        0x25D577BAB861857B,
        0xCC2C27B58860591F,
        0xA7CC008FE5DC8593,
        0x011FDAE7EFF1C939
    };
    static constexpr field_storage<uint64_t, limbs_count> UNITY_ONE { 0x1 };
  };

  static constexpr unsigned WEIERSTRASS_B = 3;
}


#endif