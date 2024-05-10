/***

Copyright (c) 2023-2024, Yrrid Software, Inc. and Snarkify Inc.  All rights reserved.
Dual licensed under the MIT License or the Apache License, Version 2.0.
See LICENSE for details.

Author(s):  Niall Emmart

***/

#if !defined(WASM) && !defined(X86)
  #error Please compile with -DWASM or -DX86
#endif

#if defined(WASM)

  #include <wasm_simd128.h>

  typedef v128_t I128;
  typedef v128_t F128;

  inline F128 I2D(I128 x) {
    return x;
  }

  inline I128 D2I(F128 x) {
    return x;
  }

  inline I128 i64x2_splat(uint64_t x) {
    return wasm_i64x2_splat(x);
  }

  inline F128 f64x2_splat(double x) {
    return wasm_f64x2_splat(x);
  }

  inline I128 i64x2_make(uint64_t l0, uint64_t l1) {
    return wasm_u64x2_make(l0, l1);
  }

  inline I128 i32x4_make(uint32_t w0, uint32_t w1, uint32_t w2, uint32_t w3) {
    return wasm_u32x4_make(w0, w1, w2, w3);
  }

  inline I128 i64x2_and(I128 a, I128 b) {
    return wasm_v128_and(a, b);
  }

  inline I128 i64x2_xor(I128 a, I128 b) {
    return wasm_v128_xor(a, b);
  }

  inline I128 i64x2_add(I128 a, I128 b) {
    return wasm_i64x2_add(a, b);
  }

  inline I128 i64x2_sub(I128 a, I128 b) {
    return wasm_i64x2_sub(a, b);
  }

  inline I128 i64x2_shr(I128 x, int bits) {
    return wasm_i64x2_shr(x, bits);
  }

  inline F128 f64x2_sub(F128 a, F128 b) {
    return wasm_f64x2_sub(a, b);
  }

  inline F128 f64x2_fma(F128 a, F128 b, F128 c) {
    return wasm_f64x2_relaxed_madd(a, b, c);
  }

  inline uint64_t i64x2_extract_l(I128 x) {
    return wasm_i64x2_extract_lane(x, 0);
  }

  inline uint64_t i64x2_extract_h(I128 x) {
    return wasm_i64x2_extract_lane(x, 1);
  }

  inline I128 i64x2_extract_ll(I128 a, I128 b) {
    return wasm_v64x2_shuffle(a, b, 0, 2);
  }

  inline I128 i64x2_extract_hh(I128 a, I128 b) {
    return wasm_v64x2_shuffle(a, b, 1, 3);
  }

  inline I128 i64x2_extract_hl(I128 a, I128 b) {
    return wasm_v64x2_shuffle(a, b, 1, 2);
  }

#endif

#if defined(X86)

  #include <immintrin.h>

  typedef __m128i I128;
  typedef __m128d F128;

  static inline F128 I2D(I128 x) {
    return _mm_castsi128_pd(x);
  }

  static inline I128 D2I(F128 x) {
    return _mm_castpd_si128(x);
  }

  static inline I128 i64x2_splat(uint64_t x) {
    return _mm_set1_epi64x(x);
  }

  static inline F128 f64x2_splat(double x) {
    return _mm_set1_pd(x);
  }

  static inline I128 i64x2_make(uint64_t l0, uint64_t l1) {
    return _mm_set_epi64x(l1, l0);
  }

  static inline I128 i32x4_make(uint32_t l0, uint32_t l1, uint32_t l2, uint32_t l3) {
    return _mm_set_epi32(l3, l2, l1, l0);
  }

  static inline I128 i64x2_and(I128 a, I128 b) {
    return _mm_and_si128(a, b);
  }

  static inline I128 i64x2_xor(I128 a, I128 b) {
    return _mm_xor_si128(a, b);
  }

  static inline I128 i64x2_add(I128 a, I128 b) {
    return _mm_add_epi64(a, b);
  }

  static inline I128 i64x2_sub(I128 a, I128 b) {
    return _mm_sub_epi64(a, b);
  }

  static inline I128 i64x2_shr(I128 x, int bits) {
    int64_t x0=_mm_extract_epi64(x, 0), x1=_mm_extract_epi64(x, 1);

    return _mm_set_epi64x(x1>>bits, x0>>bits);
  }

  static inline F128 f64x2_sub(F128 a, F128 b) {
    return _mm_sub_pd(a, b);
  }

  static inline F128 f64x2_fma(F128 a, F128 b, F128 c) {
    return _mm_fmadd_pd(a, b, c);
  }

  static inline uint64_t i64x2_extract_l(I128 x) {
    return _mm_extract_epi64(x, 0);
  }

  static inline uint64_t i64x2_extract_h(I128 x) {
    return _mm_extract_epi64(x, 1);
  }

  static inline I128 i64x2_extract_ll(I128 a, I128 b) {
    return _mm_unpacklo_epi64(a, b);
  }

  static inline I128 i64x2_extract_hh(I128 a, I128 b) {
    return _mm_unpackhi_epi64(a, b);
  }

  static inline I128 i64x2_extract_hl(I128 a, I128 b) {
    return D2I(_mm_shuffle_pd(I2D(a), I2D(b), 0x01));
  }
#endif
