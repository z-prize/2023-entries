import { to_words_le } from "../implementation/cuzk/utils";
import assert from "assert";
import crypto from "crypto";
import { ExtPointType } from "@noble/curves/abstract/edwards";

// This is slower than bigints_to_u8_for_gpu, so don't use it.
// Converts the BigInts to byte arrays in the form of
// [b0, b1, 0, 0, b2, b3, 0, 0, ...]
export const bigints_to_16_bit_words_for_gpu = (vals: bigint[]): Uint8Array => {
  const result = new Uint8Array(64 * vals.length);
  for (let i = 0; i < vals.length; i++) {
    /// This code snippet is adapted from
    /// https://github.com/no2chem/bigint-buffer/blob/master/src/index.ts#L60
    const hex = vals[i].toString(16);
    const buf = Buffer.from(hex.padStart(128, "0").slice(0, 128), "hex");
    buf.reverse();

    for (let j = 0; j < buf.length; j += 2) {
      result[i * 64 + j * 2] = buf[j];
      result[i * 64 + j * 2 + 1] = buf[j + 1];
    }
  }
  return result;
};

//  if the scalars converted to limbs = [
//       [limb_a, limb_b],
//       [limb_c, limb_d]
//  ]
//  return: [
//       [limb_a, limb_c],
//       [limb_b, limb_d]
// ]
export const decompose_scalars = (
  scalars: bigint[],
  num_words: number,
  word_size: number,
): number[][] => {
  const as_limbs: number[][] = [];
  for (const scalar of scalars) {
    const limbs = to_words_le(scalar, num_words, word_size);
    as_limbs.push(Array.from(limbs));
  }
  const result: number[][] = [];
  for (let i = 0; i < num_words; i++) {
    const t = as_limbs.map((limbs) => limbs[i]);
    result.push(t);
  }
  return result;
};

export const decompose_scalars_signed = (
  scalars: bigint[],
  num_words: number,
  word_size: number,
): number[][] => {
  const l = 2 ** word_size;
  const shift = 2 ** (word_size - 1);

  const as_limbs: number[][] = [];

  for (const scalar of scalars) {
    const limbs = to_words_le(scalar, num_words, word_size);
    const signed_slices: number[] = Array(limbs.length).fill(0);

    let carry = 0;
    for (let i = 0; i < limbs.length; i++) {
      signed_slices[i] = limbs[i] + carry;
      if (signed_slices[i] >= l / 2) {
        signed_slices[i] = (l - signed_slices[i]) * -1;
        if (signed_slices[i] === -0) {
          signed_slices[i] = 0;
        }

        carry = 1;
      } else {
        carry = 0;
      }
    }

    if (carry === 1) {
      console.error(scalar);
      throw new Error("final carry is 1");
    }

    as_limbs.push(Array.from(signed_slices).map((x) => x + shift));
  }

  const result: number[][] = [];
  for (let i = 0; i < num_words; i++) {
    const t = as_limbs.map((limbs) => limbs[i]);
    result.push(t);
  }
  return result;
};

/**
 * Generates a random field element less than a given prime modulus.
 * @param {bigint} p - The prime modulus.
 * @returns {bigint} A random field element.
 */
export const genRandomFieldElement = (p: bigint): bigint => {
  /// Assume that p is < 32 bytes.
  const lim = BigInt(
    "0x10000000000000000000000000000000000000000000000000000000000000000",
  );
  assert(p < lim);
  const min = (lim - p) % p;

  let rand;
  while (true) {
    rand = BigInt("0x" + crypto.randomBytes(32).toString("hex"));
    if (rand >= min) {
      break;
    }
  }

  return rand % p;
};

export const are_point_arr_equal = (
  a: ExtPointType[],
  b: ExtPointType[],
): boolean => {
  if (a.length !== b.length) {
    return false;
  }

  for (let i = 0; i < a.length; i++) {
    const aa = a[i].toAffine();
    const ba = b[i].toAffine();
    if (aa.x !== ba.x || aa.y !== ba.y) {
      console.log(`mismatch at ${i}`);
      return false;
    }
  }

  return true;
};
