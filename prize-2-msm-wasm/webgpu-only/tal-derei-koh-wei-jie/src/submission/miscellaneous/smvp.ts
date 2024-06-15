/// This module provides CPU-based operations, including Sparse Matrix-Vector Product (SMVP)
/// with signed bucket indices in typescript. It leverages the
/// `FieldMath` utility for mathematical operations on extended points.

import { FieldMath } from "../../reference/utils/FieldMath";
import { ExtPointType } from "@noble/curves/abstract/edwards";

export const cpu_smvp = (
  csc_col_ptr: number[],
  csc_val_idxs: number[],
  points: ExtPointType[],
  fieldMath: FieldMath,
) => {
  const zero = fieldMath.customEdwards.ExtendedPoint.ZERO;

  const result: ExtPointType[] = [];
  for (let i = 0; i < csc_col_ptr.length - 1; i++) {
    result.push(zero);
  }

  for (let i = 0; i < csc_col_ptr.length - 1; i++) {
    const row_begin = csc_col_ptr[i];
    const row_end = csc_col_ptr[i + 1];
    let sum = zero;
    for (let j = row_begin; j < row_end; j++) {
      sum = sum.add(points[csc_val_idxs[j]]);
    }
    result[i] = sum;
  }

  return result;
};

/**
 * Perform SMVP with signed bucket indices
 */
export const cpu_smvp_signed = (
  subtask_idx: number,
  input_size: number,
  num_columns: number,
  chunk_size: number,
  all_csc_col_ptr: number[],
  all_csc_val_idxs: number[],
  points: ExtPointType[],
  fieldMath: FieldMath,
) => {
  const l = 2 ** chunk_size;
  const h = l / 2;
  const zero = fieldMath.customEdwards.ExtendedPoint.ZERO;

  const buckets: ExtPointType[] = [];
  for (let i = 0; i < num_columns / 2; i++) {
    buckets.push(zero);
  }

  const rp_offset = subtask_idx * (num_columns + 1);

  // In a GPU implementation, each iteration of this loop should be performed by a thread.
  // Each thread handles two buckets.
  for (let thread_id = 0; thread_id < num_columns / 2; thread_id++) {
    for (let j = 0; j < 2; j++) {
      // row_idx is the index of the row in the CSR matrix. It is *not*
      // the same as the bucket index.
      let row_idx = thread_id + num_columns / 2;
      if (j === 1) {
        row_idx = num_columns / 2 - thread_id;
      }

      if (thread_id === 0 && j === 0) {
        row_idx = 0;
      }

      const row_begin = all_csc_col_ptr[rp_offset + row_idx];
      const row_end = all_csc_col_ptr[rp_offset + row_idx + 1];

      let sum = zero;
      for (let k = row_begin; k < row_end; k++) {
        sum = sum.add(points[all_csc_val_idxs[subtask_idx * input_size + k]]);
      }

      let bucket_idx;
      if (h > row_idx) {
        bucket_idx = h - row_idx;
        sum = sum.negate();
      } else {
        bucket_idx = row_idx - h;
      }

      if (bucket_idx > 0) {
        // Store the result in buckets[thread_id]. Each thread must use
        // a unique storage location (thread_id) to prevent race conditions.
        buckets[thread_id] = buckets[thread_id].add(sum);
      } else {
        buckets[thread_id] = buckets[thread_id].add(zero);
      }
      //console.log({ thread_id, bucket_idx })
    }
  }
  //console.log('----')

  return buckets;
};
