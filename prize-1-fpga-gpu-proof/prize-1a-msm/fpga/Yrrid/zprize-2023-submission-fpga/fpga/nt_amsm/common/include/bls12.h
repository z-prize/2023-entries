/***

Copyright (c) 2023-2024, Yrrid Software, Inc. All rights reserved.
Dual licensed under the MIT License or the Apache License, Version 2.0.
See LICENSE for details.

Author(s):  Tony Wu

***/

#pragma once
#include "mp_int.h"

enum bls12_t { bls12_381, bls12_377 };

template<bls12_t CURVE>
class base_field_t : ap_uint<384> {
public:
    using ap_uint::ap_uint;

    inline static const ap_uint<259> CURVE_ORDER = CURVE == bls12_381 ? ap_uint<259>("0x73eda753299d7d483339d80809a1d80553bda402fffe5bfeffffffff00000001", 16) : ap_uint<259>("0x12ab655e9a2ca55660b44d1e5c37b00159aa76fed00000010a11800000000001", 16);
    inline static const ap_uint<384> P = CURVE == bls12_381 ? ap_uint<384>("0x1a0111ea397fe69a4b1ba7b6434bacd764774b84f38512bf6730d2a0f6b0f6241eabfffeb153ffffb9feffffffffaaab", 16) : ap_uint<384>("0x01ae3a4617c510eac63b05c06ca1493b1a22d9f300f5138f1ef3622fba094800170b5d44300000008508c00000000001", 16);
    inline static const ap_uint<384> P2 = CURVE == bls12_381 ? ap_uint<384>("0x340223d472ffcd3496374f6c869759aec8ee9709e70a257ece61a541ed61ec483d57fffd62a7ffff73fdffffffff5556", 16) : ap_uint<384>("0x35c748c2f8a21d58c760b80d94292763445b3e601ea271e3de6c45f741290002e16ba88600000010a11800000000002", 16);
    inline static const ap_uint<384> R_MOD_P = CURVE == bls12_381 ? ap_uint<384>("0x15f65ec3fa80e4935c071a97a256ec6d77ce5853705257455f48985753c758baebf4000bc40c0002760900000002fffd", 16) : ap_uint<384>("0x8d6661e2fdf49a4cf495bf803c84e87b4e97b76e7c63059f7db3a98a7d3ff251409f837fffffb102cdffffffffff68", 16);
    inline static const ap_uint<384> NP = CURVE == bls12_381 ? ap_uint<384>("0xceb06106feaafc9468b316fee268cf5819ecca0e8eb2db4c16ef2ef0c8e30b48286adb92d9d113e889f3fffcfffcfffd", 16) : ap_uint<384>("0xbfa5205feec82e3d22f80141806a3cec5b245b86cced7a1335ed1347970debffd1e94577a00000008508bfffffffffff", 16);
    inline static const ap_uint<384> R_INV_MOD_P = CURVE == bls12_381 ? ap_uint<384>("0x14fec701e8fb0ce9ed5e64273c4f538b1797ab1458a88de9343ea97914956dc87fe11274d898fafbf4d38259380b4820", 16) : ap_uint<384>("0x014212fc436b0736eb89746ca4c49351eeaaf41df7f2e657852e3c88a5582c936e20bbcde65839f5ef129093451269e8", 16);

    inline static const ap_uint<384> MONT_ONE = R_MOD_P;
};
