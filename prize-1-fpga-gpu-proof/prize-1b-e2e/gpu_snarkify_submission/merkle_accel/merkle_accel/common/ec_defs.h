/***

Copyright (c) 2024, Snarkify, Inc.  All rights reserved.
Dual licensed under the MIT License or the Apache License, Version 2.0.
See LICENSE for details.

Author(s):  Niall Emmart

***/

#pragma once
namespace ec {

class bls12381 {
  public:
  typedef ff::bls12381_fp fp;
  typedef ff::bls12381_fr fr;

  static constexpr host::ff::impl<ff::bls12381_fp,true> const_generator_x={
    0x680F21FAA66D81ull, 0x71A10333FFD5FEull, 0x34C719357B460Full, 0x20E5DE761B72C7ull, 0x43BAA7CE58A16Full, 0x0A321026BC400Dull, 0x0280772640A604ull,
  };

  static constexpr host::ff::impl<ff::bls12381_fp,true> const_generator_y={
    0x5927AA19CE44E2ull, 0x0C69E463F63AEAull, 0x1AB8392E746113ull, 0x0405194DD595F1ull, 0x0B380A358B052Aull, 0x1A8387230FEB40ull, 0x05DE1F7E280451ull,
  };
};

template<class Curve>
class xy {
  public:
  static const bool defaultMontgomery=fp<Curve>::defaultMontgomery;

  template<bool montgomery>
  using impl = impl_xy<Curve,montgomery>;

  template<bool montgomery>
  using storage = storage_xy<Curve,montgomery>;
};

template<class Curve>
class xyzz {
  public:
  static const bool defaultMontgomery=fp<Curve>::defaultMontgomery;

  template<bool montgomery>
  using impl = impl_xyzz<Curve,montgomery>;

  template<bool montgomery>
  using storage = storage_xyzz<Curve,montgomery>;
};

} // namespace ec
