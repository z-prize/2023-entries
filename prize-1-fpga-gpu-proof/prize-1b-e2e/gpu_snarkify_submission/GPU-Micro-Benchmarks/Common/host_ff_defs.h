/***

Copyright (c) 2024, Snarkify, Inc. All rights reserved.
Dual licensed under the MIT License or the Apache License, Version 2.0.
See LICENSE for details.

Author(s):  Niall Emmart

***/

#pragma once

#include <stdio.h>
#include <stdint.h>

namespace ff {

class bls12381_fp {
  public:
  static const bool     defaultMontgomery=true;
  static const uint32_t bits=381;
  static const uint32_t bits_per_limb=55;
  static const uint64_t limb_mask=(1ul<<bits_per_limb)-1ul;
  static const uint32_t limbs=7;
  static const uint32_t packed_limbs=6;

  template<bool montgomery=defaultMontgomery>
  using impl = host::ff::impl<bls12381_fp,montgomery>;

  template<bool montgomery=defaultMontgomery>
  using storage = host::ff::storage<bls12381_fp,montgomery>;

  typedef storage<true> storage_const;
  typedef impl<true> impl_const;

  static constexpr storage_const const_storage_zero={
    0x0ull, 0x0ull, 0x0ull, 0x0ull, 0x0ull, 0x0ull,
  };

  static constexpr storage_const const_storage_one={
    0x00000001ull, 0x0ull, 0x0ull, 0x0ull, 0x0ull, 0x0ull,
  };

  static constexpr storage_const const_storage_n={
    0xB9FEFFFFFFFFAAABull, 0x1EABFFFEB153FFFFull, 0x6730D2A0F6B0F624ull, 0x64774B84F38512BFull, 0x4B1BA7B6434BACD7ull, 0x1A0111EA397FE69Aull,
  };

  static constexpr impl_const const_zero={
    0x0ull, 0x0ull, 0x0ull, 0x0ull, 0x0ull, 0x0ull, 0x0ull,
  };

  static constexpr impl_const const_n={
    0x7EFFFFFFFFAAABull, 0x7FFD62A7FFFF73ull, 0x03DAC3D8907AAFull, 0x1C2895FB398695ull, 0x3ACD764774B84Full, 0x53496374F6C869ull, 0x0680447A8E5FF9ull,
  };

  static constexpr impl_const const_2n={
    0x7DFFFFFFFF5556ull, 0x7FFAC54FFFFEE7ull, 0x07B587B120F55Full, 0x38512BF6730D2Aull, 0x759AEC8EE9709Eull, 0x2692C6E9ED90D2ull, 0x0D0088F51CBFF3ull,
  };

  static constexpr impl_const const_r={
    0x1300000006554Full, 0x0031AD88000A64ull, 0x36C376ED46E4F0ull, 0x68FCDE5ABB02F0ull, 0x22C038B256521Eull, 0x518D9E51AF202Cull, 0x047AEAE76EE078ull,
  };

  static constexpr impl_const const_r_inv={
    0x69C12C9C05A410ull, 0x1274D898FAFBF4ull, 0x72292ADB90FFC2ull, 0x62A237A4D0FAA5ull, 0x7A9C58BCBD58A2ull, 0x4E9ED5E64273C4ull, 0x029FD8E03D1F61ull,
  };

  static constexpr impl_const const_r_squared={
    0x7E7CD070D107C2ull, 0x353589382790BEull, 0x3D13D3BC2FB20Eull, 0x3B5F4C1B499BC3ull, 0x6FB06D6BF8B9C6ull, 0x2EBA75B55549B9ull, 0x049806F0760AF0ull,
  };

  static constexpr impl_const const_sr_squared={
    0x5F1F341C341746ull, 0x4D4C13A209E3E9ull, 0x313256DB5429DBull, 0x1CEC1E046F2A3Bull, 0x1952D67EB88A99ull, 0x75534F27D0B6A3ull, 0x046623F964B2B8ull,
  };

  static uint64_t qTerm(uint64_t x) {
    const uint64_t np0=0x73FFFCFFFCFFFDull;

    return x*np0 & limb_mask;
  }
};

class bls12381_fr {
  public:
  static const bool     defaultMontgomery=false;
  static const uint32_t bits=255;
  static const uint32_t limbs=5;
  static const uint32_t bits_per_limb=52;
  static const uint64_t limb_mask=(1ul<<bits_per_limb)-1ul;
  static const uint32_t packed_limbs=4;

  template<bool montgomery=defaultMontgomery>
  using impl = host::ff::impl<bls12381_fr,montgomery>;

  template<bool montgomery=defaultMontgomery>
  using storage = host::ff::storage<bls12381_fr,montgomery>;

  typedef storage<false> storage_const;
  typedef impl<false> impl_const;

  static constexpr storage_const const_storage_zero={
    0x0ull, 0x0ull, 0x0ull, 0x0ull,
  };

  static constexpr storage_const const_storage_one={
    0x00000001ull, 0x0ull, 0x0ull, 0x0ull,
  };

  static constexpr storage_const const_storage_n={
    0xFFFFFFFF00000001ull, 0x53BDA402FFFE5BFEull, 0x3339D80809A1D805ull, 0x73EDA753299D7D48ull,
  };

  static constexpr impl_const const_zero={
    0x0ull, 0x0ull, 0x0ull, 0x0ull, 0x0ull,
  };

  static constexpr impl_const const_n={
    0xFFFFF00000001ull, 0x02FFFE5BFEFFFull, 0x9A1D80553BDA4ull, 0x7D483339D8080ull, 0x073EDA753299Dull,
  };

  static constexpr impl_const const_2n={
    0xFFFFE00000002ull, 0x05FFFCB7FDFFFull, 0x343B00AA77B48ull, 0xFA906673B0101ull, 0x0E7DB4EA6533Aull,
  };

  static constexpr impl_const const_r={
    0x00022FFFFFFDDull, 0x9700396C23000ull, 0xEDF77458D1293ull, 0xDF20FF1776E6Aull, 0x026821FA14F77ull,
  };

  static constexpr impl_const const_r_inv={
    0xF75B69FE75C04ull, 0xA8F09DC705F13ull, 0x4F77266AAB6FCull, 0x09D577204078Aull, 0x001BBE8693300ull,
  };

  static constexpr impl_const const_r_squared={
    0x99103F29C6CF0ull, 0x57927663D999Eull, 0xA1C0ED631138Bull, 0x3C829F7715F1Bull, 0x009FF646CC027ull,
  };

  static constexpr impl_const const_sr_squared={
    0x9E990F3F29C6Dull, 0xCB87925C23C99ull, 0x254398F2B6CEDull, 0xFF1105D314967ull, 0x00748D9D99F59ull,
  };

  static constexpr impl_const const_generator={
    0x7ull, 0x0ull, 0x0ull, 0x0ull, 0x0ull,
  };

  static constexpr impl_const const_generator_montgomery={
    0x000F6FFFFFF09ull, 0x1B01953CF7000ull, 0x4D892DC3406C1ull, 0x1F569330903EBull, 0x025B38EC2D90Cull,
  };

  static uint64_t qTerm(uint64_t x) {
    const uint64_t np0=0xFFFFEFFFFFFFFull;

    return x*np0 & limb_mask;
  }
};

} // namespace ff
