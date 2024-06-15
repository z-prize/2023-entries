/*
 * Module Name: Utilities
 *
 * Description:
 * This module provides various utility functions for cryptographic operations.
 */

import assert from "assert";
import * as bigintCryptoUtils from "bigint-crypto-utils";
import { BigIntPoint } from "../../../reference/types";
import { FieldMath } from "../../../reference/utils/FieldMath";
import { ExtPointType } from "@noble/curves/abstract/edwards";
import { EDWARDS_D } from "../../../reference/params/AleoConstants";

/**
 * Converts a BigIntPoint to an ExtPointType using field arithmetic.
 * @param {BigIntPoint} bip - A point with bigint coordinates.
 * @param {FieldMath} fieldMath - The field arithmetic utility.
 * @returns {ExtPointType} A point in the extended format.
 */
export const bigIntPointToExtPointType = (
  bip: BigIntPoint,
  fieldMath: FieldMath,
): ExtPointType => {
  return fieldMath.createPoint(bip.x, bip.y, bip.t, bip.z);
};

/**
 * Converts an ExtPointType to a BigIntPoint.
 * @param {ExtPointType} ept - A point in the extended format.
 * @returns {BigIntPoint} A point with bigint coordinates.
 */
export const extPointTypeToBigIntPoint = (ept: ExtPointType): BigIntPoint => {
  return { x: ept.ex, y: ept.ey, t: ept.et, z: ept.ez };
};

/**
 * Converts an array of BigIntPoints to a Uint8Array for GPU processing.
 * @param {BigIntPoint[]} points - An array of points with bigint coordinates.
 * @param {number} num_words - The number of words per bigint.
 * @param {number} word_size - The size of each word in bits.
 * @returns {Uint8Array} The resulting byte array for GPU use.
 */
export const points_to_u8s_for_gpu = (
  points: BigIntPoint[],
  num_words: number,
  word_size: number,
): Uint8Array => {
  const size = points.length * num_words * 4 * 4;
  const result = new Uint8Array(size);

  for (let i = 0; i < points.length; i++) {
    const x_bytes = bigint_to_u8_for_gpu(points[i].x, num_words, word_size);
    const y_bytes = bigint_to_u8_for_gpu(points[i].y, num_words, word_size);
    const t_bytes = bigint_to_u8_for_gpu(points[i].t, num_words, word_size);
    const z_bytes = bigint_to_u8_for_gpu(points[i].z, num_words, word_size);

    for (let j = 0; j < x_bytes.length; j++) {
      const i4l = i * 4 * x_bytes.length;
      result[i4l + j] = x_bytes[j];
      result[i4l + j + x_bytes.length] = y_bytes[j];
      result[i4l + j + x_bytes.length * 2] = t_bytes[j];
      result[i4l + j + x_bytes.length * 3] = z_bytes[j];
    }
  }

  return result;
};

/**
 * Converts a Uint8Array back into an array of BigIntPoints.
 * @param {Uint8Array} bytes - The byte array representing points.
 * @param {number} num_words - The number of words per bigint.
 * @param {number} word_size - The size of each word in bits.
 * @returns {BigIntPoint[]} An array of reconstructed points with bigint coordinates.
 */
export const u8s_to_points = (
  bytes: Uint8Array,
  num_words: number,
  word_size: number,
): BigIntPoint[] => {
  /// Since each limb is a u32, there are 4 u8s per limb.
  const num_u8s_per_coord = num_words * 4;
  const num_u8s_per_point = num_u8s_per_coord * 4;

  assert(bytes.length % num_u8s_per_point === 0);
  const result: BigIntPoint[] = [];
  for (let i = 0; i < bytes.length / num_u8s_per_point; i++) {
    const p = i * num_u8s_per_point;
    const x_i = p;
    const y_i = p + num_u8s_per_coord;
    const t_i = p + num_u8s_per_coord * 2;
    const z_i = p + num_u8s_per_coord * 3;

    const x_u8s = bytes.slice(x_i, x_i + num_u8s_per_coord);
    const y_u8s = bytes.slice(y_i, y_i + num_u8s_per_coord);
    const t_u8s = bytes.slice(t_i, t_i + num_u8s_per_coord);
    const z_u8s = bytes.slice(z_i, z_i + num_u8s_per_coord);

    const x = u8s_to_bigint(x_u8s, num_words, word_size);
    const y = u8s_to_bigint(y_u8s, num_words, word_size);
    const t = u8s_to_bigint(t_u8s, num_words, word_size);
    const z = u8s_to_bigint(z_u8s, num_words, word_size);

    result.push({ x, y, t, z });
  }

  return result;
};

/**
 * Converts a Uint8Array to an array of bigints, interpreting each segment as a separate bigint.
 * @param {Uint8Array} u8s - The byte array to convert.
 * @param {number} num_words - The number of words per bigint.
 * @param {number} word_size - The size of each word in bits.
 * @returns {bigint[]} An array of bigints.
 */
export const u8s_to_bigints = (
  u8s: Uint8Array,
  num_words: number,
  word_size: number,
): bigint[] => {
  const num_u8s_per_scalar = num_words * 4;
  const result = [];
  for (let i = 0; i < u8s.length / num_u8s_per_scalar; i++) {
    const p = i * num_u8s_per_scalar;
    const s = u8s.slice(p, p + num_u8s_per_scalar);
    result.push(u8s_to_bigint(s, num_words, word_size));
  }
  return result;
};

/**
 * Converts a Uint8Array to a single bigint.
 * @param {Uint8Array} u8s - The byte array to convert.
 * @param {number} num_words - The number of words per bigint.
 * @param {number} word_size - The size of each word in bits.
 * @returns {bigint} The reconstructed bigint.
 */
export const u8s_to_bigint = (
  u8s: Uint8Array,
  num_words: number,
  word_size: number,
): bigint => {
  const a = new Uint16Array(u8s.buffer);
  const limbs: number[] = [];
  for (let i = 0; i < a.length; i += 2) {
    limbs.push(a[i]);
  }

  return from_words_le(new Uint16Array(limbs), num_words, word_size);
};

/**
 * Converts an array of numbers to a Uint8Array for GPU processing without assertion checks.
 * @param {number[]} vals - An array of numbers to convert.
 * @returns {Uint8Array} The resulting byte array for GPU use.
 */
export const u8s_to_bigints_without_assertion = (
  u8s: Uint8Array,
  num_words: number,
  word_size: number,
): bigint[] => {
  const num_u8s_per_scalar = num_words * 4;
  const result = [];
  for (let i = 0; i < u8s.length / num_u8s_per_scalar; i++) {
    const p = i * num_u8s_per_scalar;
    const s = u8s.slice(p, p + num_u8s_per_scalar);
    result.push(u8s_to_bigint_without_assertion(s, num_words, word_size));
  }
  return result;
};

export const u8s_to_bigint_without_assertion = (
  u8s: Uint8Array,
  num_words: number,
  word_size: number,
): bigint => {
  const a = new Uint16Array(u8s.buffer);
  const limbs: number[] = [];
  for (let i = 0; i < a.length; i += 2) {
    limbs.push(a[i]);
  }

  return from_words_le_without_assertion(new Uint16Array(limbs), num_words, word_size);
};

export const numbers_to_u8s_for_gpu = (vals: number[]): Uint8Array => {
  /// Expect each val to be max 32 bits.
  const max = 2 ** 32;
  for (const val of vals) {
    assert(val < max);
  }
  const b = new Uint32Array(vals);
  return new Uint8Array(b.buffer);
};

/**
 * Converts a Uint8Array to an array of numbers, assuming each number occupies 16 bits.
 * @param {Uint8Array} u8s - The byte array to convert.
 * @returns {number[]} An array of reconstructed numbers.
 */
export const u8s_to_numbers = (u8s: Uint8Array): number[] => {
  const result: number[] = [];
  assert(u8s.length % 4 === 0);
  for (let i = 0; i < u8s.length / 4; i++) {
    const n0 = u8s[i * 4];
    const n1 = u8s[i * 4 + 1];
    result.push(n1 * 256 + n0);
  }
  return result;
};

/**
 * Converts a Uint8Array to an array of 32-bit numbers.
 * @param {Uint8Array} u8s - The byte array to convert.
 * @returns {number[]} An array of reconstructed 32-bit numbers.
 */
export const u8s_to_numbers_32 = (u8s: Uint8Array): number[] => {
  const result: number[] = [];
  assert(u8s.length % 4 === 0);
  for (let i = 0; i < u8s.length / 4; i++) {
    const n0 = u8s[i * 4];
    const n1 = u8s[i * 4 + 1];
    const n2 = u8s[i * 4 + 2];
    const n3 = u8s[i * 4 + 3];
    result.push(n3 * 16777216 + n2 * 65536 + n1 * 256 + n0);
  }
  return result;
};

/**
 * Converts an array of bigints to a Uint8Array for GPU processing (old version).
 * @param {bigint[]} vals - An array of bigints to convert.
 * @param {number} num_words - The number of words per bigint.
 * @param {number} word_size - The size of each word in bits.
 * @returns {Uint8Array} The resulting byte array for GPU use.
 */
export const bigints_to_u8_for_gpu_old = (
  vals: bigint[],
  num_words: number,
  word_size: number,
): Uint8Array => {
  const size = vals.length * num_words * 4;
  const result = new Uint8Array(size);

  for (let i = 0; i < vals.length; i++) {
    const bytes = bigint_to_u8_for_gpu(vals[i], num_words, word_size);
    for (let j = 0; j < bytes.length; j++) {
      result[i * bytes.length + j] = bytes[j];
    }
  }

  return result;
};

/**
 * Converts an array of 32-byte bigints to a Uint8Array for GPU processing.
 * @param {bigint[]} vals - An array of bigints to convert.
 * @returns {Uint8Array} The resulting byte array for GPU use.
 */
export const bigints_to_u8_for_gpu = (vals: bigint[]): Uint8Array => {
  const result = new Uint8Array(vals.length * 64);
  const mask = BigInt(255);

  for (let i = 0; i < vals.length; i++) {
    const val = vals[i];
    result[i * 64 + 60] = Number((val >> BigInt(240)) & mask);
    result[i * 64 + 61] = Number((val >> BigInt(248)) & mask);
    result[i * 64 + 56] = Number((val >> BigInt(224)) & mask);
    result[i * 64 + 57] = Number((val >> BigInt(232)) & mask);
    result[i * 64 + 52] = Number((val >> BigInt(208)) & mask);
    result[i * 64 + 53] = Number((val >> BigInt(216)) & mask);
    result[i * 64 + 48] = Number((val >> BigInt(192)) & mask);
    result[i * 64 + 49] = Number((val >> BigInt(200)) & mask);
    result[i * 64 + 44] = Number((val >> BigInt(176)) & mask);
    result[i * 64 + 45] = Number((val >> BigInt(184)) & mask);
    result[i * 64 + 40] = Number((val >> BigInt(160)) & mask);
    result[i * 64 + 41] = Number((val >> BigInt(168)) & mask);
    result[i * 64 + 36] = Number((val >> BigInt(144)) & mask);
    result[i * 64 + 37] = Number((val >> BigInt(152)) & mask);
    result[i * 64 + 32] = Number((val >> BigInt(128)) & mask);
    result[i * 64 + 33] = Number((val >> BigInt(136)) & mask);
    result[i * 64 + 28] = Number((val >> BigInt(112)) & mask);
    result[i * 64 + 29] = Number((val >> BigInt(120)) & mask);
    result[i * 64 + 24] = Number((val >> BigInt(96)) & mask);
    result[i * 64 + 25] = Number((val >> BigInt(104)) & mask);
    result[i * 64 + 20] = Number((val >> BigInt(80)) & mask);
    result[i * 64 + 21] = Number((val >> BigInt(88)) & mask);
    result[i * 64 + 16] = Number((val >> BigInt(64)) & mask);
    result[i * 64 + 17] = Number((val >> BigInt(72)) & mask);
    result[i * 64 + 12] = Number((val >> BigInt(48)) & mask);
    result[i * 64 + 13] = Number((val >> BigInt(56)) & mask);
    result[i * 64 + 8] = Number((val >> BigInt(32)) & mask);
    result[i * 64 + 9] = Number((val >> BigInt(40)) & mask);
    result[i * 64 + 4] = Number((val >> BigInt(16)) & mask);
    result[i * 64 + 5] = Number((val >> BigInt(24)) & mask);
    result[i * 64 + 0] = Number((val >> BigInt(0)) & mask);
    result[i * 64 + 1] = Number((val >> BigInt(8)) & mask);
  }

  return result;
};

/**
 * Converts a single bigint to a Uint8Array for GPU processing, breaking it down into words.
 * @param {bigint} val - The bigint to convert.
 * @param {number} num_words - The number of words per bigint.
 * @param {number} word_size - The size of each word in bits.
 * @returns {Uint8Array} The resulting byte array for GPU use.
 */
export const bigint_to_u8_for_gpu = (
  val: bigint,
  num_words: number,
  word_size: number,
): Uint8Array => {
  const result = new Uint8Array(num_words * 4);
  const limbs = to_words_le(BigInt(val), num_words, word_size);
  for (let i = 0; i < limbs.length; i++) {
    const i4 = i * 4;
    result[i4] = limbs[i] & 255;
    result[i4 + 1] = limbs[i] >> 8;
  }

  return result;
};

/**
 * Generates WGSL code to initialize a variable with bigint limbs.
 * @param {bigint} val - The bigint value to convert into limbs.
 * @param {string} var_name - The name of the WGSL variable to initialize.
 * @param {number} num_words - The number of words per bigint.
 * @param {number} word_size - The size of each word in bits.
 * @returns {string} The generated WGSL code snippet.
 */
export const gen_wgsl_limbs_code = (
  val: bigint,
  var_name: string,
  num_words: number,
  word_size: number,
): string => {
  const limbs = to_words_le(val, num_words, word_size);
  let r = "";
  for (let i = 0; i < limbs.length; i++) {
    r += `    ${var_name}.limbs[${i}]` + " = " + limbs[i].toString() + "u;\n";
  }
  return r;
};

/**
 * Generates WGSL code to initialize the Barrett-Domb 'm' parameter.
 * @param {bigint} m - The 'm' parameter value.
 * @param {number} num_words - The number of words for bigint representation.
 * @param {number} word_size - The size of each word in bits.
 * @returns {string} WGSL code for initializing the 'm' parameter.
 */
export const gen_barrett_domb_m_limbs = (
  m: bigint,
  num_words: number,
  word_size: number,
): string => {
  return gen_wgsl_limbs_code(m, "m", num_words, word_size);
};

/**
 * Generates WGSL code for initializing the prime modulus 'p' with its limbs.
 * @param {bigint} p - The prime modulus.
 * @param {number} num_words - The number of words for bigint representation.
 * @param {number} word_size - The size of each word in bits.
 * @returns {string} WGSL code for initializing 'p'.
 */
export const gen_p_limbs = (
  p: bigint,
  num_words: number,
  word_size: number,
): string => {
  return gen_wgsl_limbs_code(p, "p", num_words, word_size);
};

/**
 * Generates WGSL code for initializing the Montgomery radix 'r' with its limbs.
 * @param {bigint} r - The Montgomery radix.
 * @param {number} num_words - The number of words for bigint representation.
 * @param {number} word_size - The size of each word in bits.
 * @returns {string} WGSL code for initializing 'r'.
 */
export const gen_r_limbs = (
  r: bigint,
  num_words: number,
  word_size: number,
): string => {
  return gen_wgsl_limbs_code(r, "r", num_words, word_size);
};

/**
 * Generates WGSL code for initializing the curve constant 'd' with its limbs.
 * @param {bigint} d - The curve constant 'd'.
 * @param {number} num_words - The number of words for bigint representation.
 * @param {number} word_size - The size of each word in bits.
 * @returns {string} WGSL code for initializing 'd'.
 */
export const gen_d_limbs = (
  d: bigint,
  num_words: number,
  word_size: number,
): string => {
  return gen_wgsl_limbs_code(d, "d", num_words, word_size);
};

/**
 * Generates WGSL code for initializing the Barrett reduction parameter 'mu' with its limbs.
 * @param {bigint} p - The prime modulus used to compute 'mu'.
 * @param {number} num_words - The number of words for bigint representation.
 * @param {number} word_size - The size of each word in bits.
 * @returns {string} WGSL code for initializing 'mu'.
 */
export const gen_mu_limbs = (
  p: bigint,
  num_words: number,
  word_size: number,
): string => {
  /// precompute mu by choosing x such that x is the smallest 2 ** x > s
  /// https://www.nayuki.io/page/barrett-reduction-algorithm.
  let x = BigInt(1);
  while (BigInt(2) ** x < p) {
    x += BigInt(1);
  }

  const mu = BigInt(4) ** x / p;
  return gen_wgsl_limbs_code(mu, "mu", num_words, word_size);
};

/**
 * Converts a bigint into an array of 16-bit words in little-endian format.
 * @param {bigint} val - The bigint to convert.
 * @param {number} num_words - The number of words.
 * @param {number} word_size - The size of each word in bits.
 * @returns {Uint16Array} The array of 16-bit words.
 */
export const to_words_le = (
  val: bigint,
  num_words: number,
  word_size: number,
): Uint16Array => {
  const words = new Uint16Array(num_words);

  const mask = BigInt(2 ** word_size - 1);
  for (let i = 0; i < num_words; i++) {
    const idx = num_words - 1 - i;
    const shift = BigInt(idx * word_size);
    const w = (val >> shift) & mask;
    words[idx] = Number(w);
  }

  return words;
};

/**
 * Constructs a bigint from an array of 16-bit words in little-endian format without assertion checks.
 * @param {Uint16Array} words - The array of 16-bit words.
 * @param {number} num_words - The number of words.
 * @param {number} word_size - The size of each word in bits.
 * @returns {bigint} The reconstructed bigint.
 */
export const from_words_le_without_assertion = (
  words: Uint16Array,
  num_words: number,
  word_size: number,
): bigint => {
  let val = BigInt(0);
  for (let i = 0; i < num_words; i++) {
    val +=
      BigInt(2) ** BigInt((num_words - i - 1) * word_size) *
      BigInt(words[num_words - 1 - i]);
  }

  return val;
};

export const from_words_le = (
  words: Uint16Array,
  num_words: number,
  word_size: number,
): bigint => {
  assert(num_words == words.length);
  let val = BigInt(0);
  for (let i = 0; i < num_words; i++) {
    assert(words[i] < 2 ** word_size);
    assert(words[i] >= 0);
    val +=
      BigInt(2) ** BigInt((num_words - i - 1) * word_size) *
      BigInt(words[num_words - 1 - i]);
  }

  return val;
};

/**
 * Calculates the number of words needed to represent a bigint of a given bit width.
 * @param {number} word_size - The size of each word in bits.
 * @param {number} p_width - The width of the bigint in bits.
 * @returns {number} The number of words required.
 */
export const calc_num_words = (word_size: number, p_width: number): number => {
  let num_words = Math.floor(p_width / word_size);
  while (num_words * word_size < p_width) {
    num_words++;
  }
  return num_words;
};

/**
 * Computes miscellaneous parameters needed for cryptographic operations.
 * @param {bigint} p - The prime modulus.
 * @param {number} word_size - The size of each word in bits.
 * @returns {Object} An object containing various computed parameters.
 */
export const compute_misc_params = (
  p: bigint,
  word_size: number,
): {
  num_words: number;
  max_terms: number;
  k: number;
  nsafe: number;
  n0: bigint;
  r: bigint;
  edwards_d: bigint;
  rinv: bigint;
  barrett_domb_m: bigint;
} => {
  const max_int_width = 32;
  assert(word_size > 0);
  const p_width = p.toString(2).length;
  const num_words = calc_num_words(word_size, p_width);
  const max_terms = num_words * 2;

  const rhs = 2 ** max_int_width;
  let k = 1;
  while (k * 2 ** (2 * word_size) <= rhs) {
    k += 1;
  }

  /// Number of carry free terms.
  const nsafe = Math.floor(k / 2);

  /// The Montgomery radix.
  const r = BigInt(2) ** BigInt(num_words * word_size);

  /// Returns triple (g, rinv, pprime).
  const egcdResult: { g: bigint; x: bigint; y: bigint } =
    bigintCryptoUtils.eGcd(r, p);
  const rinv = egcdResult.x;
  const pprime = egcdResult.y;

  if (rinv < BigInt(0)) {
    assert(((r * rinv - p * pprime) % p) + p === BigInt(1));
    assert(((r * rinv) % p) + p == BigInt(1));
    assert((p * pprime) % r == BigInt(1));
  } else {
    assert((r * rinv - p * pprime) % p === BigInt(1));
    assert((r * rinv) % p == BigInt(1));
    assert(((p * pprime) % r) + r == BigInt(1));
  }

  const neg_n_inv = r - pprime;
  const n0 = neg_n_inv % BigInt(2) ** BigInt(word_size);

  /// The Barrett-Domb m value.
  const z = num_words * word_size - p_width;
  const barrett_domb_m = BigInt(2 ** (2 * p_width + z)) / p;
  /// m, _ = divmod(2 ** (2 * n + z), s) # prime approximation, n + 1 bits.
  const edwards_d = (EDWARDS_D * r) % p;

  return {
    num_words,
    max_terms,
    k,
    nsafe,
    n0,
    r: r % p,
    edwards_d,
    rinv,
    barrett_domb_m,
  };
};
