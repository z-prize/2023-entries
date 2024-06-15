/*
 * Module Name: Global parameters
 *
 * Description:
 * Global parameters to seed the multi-scalar multiplication program.
 */

import { compute_misc_params } from "./utils";

/// Twisted Edwards BLS12 (Base Field) prime order.
export const p = BigInt(
  "8444461749428370424248824938781546531375899335154063827935233455917409239041",
);

/// 13-bit limbs.
export const word_size = 13;

/// Compute miscellaneous parameters.
export const params = compute_misc_params(p, word_size);

/// 20 words (13-bit limbs * 20 words = 260-bits / coordinate)
export const num_words = params.num_words;
