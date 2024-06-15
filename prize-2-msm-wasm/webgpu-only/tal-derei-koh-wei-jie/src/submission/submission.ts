/*
 * Copyright 2024 Tal Derei and Koh Wei Jie. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *     http://www.apache.org/licenses/LICENSE-2.0
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 *
 * Authors: Koh Wei Jie and Tal Derei
 */

import assert from "assert";
import { readBigIntsFromBufferLE } from "../reference/webgpu/utils";
import { BigIntPoint, U32ArrayPoint } from "../reference/types";
import { ExtPointType } from "@noble/curves/abstract/edwards";
import { ShaderManager } from "./implementation/cuzk/shader_manager";
import {
  get_device,
  create_and_write_sb,
  create_and_write_ub,
  create_bind_group,
  create_bind_group_layout,
  create_compute_pipeline,
  create_sb,
  read_from_gpu,
  execute_pipeline,
} from "./implementation/cuzk/gpu";
import {
  u8s_to_bigints,
  u8s_to_numbers,
  u8s_to_numbers_32,
  from_words_le,
  numbers_to_u8s_for_gpu,
  u8s_to_bigints_without_assertion,
} from "./implementation/cuzk/utils";
import { decompose_scalars_signed } from "./miscellaneous/utils";
import { cpu_transpose } from "./miscellaneous/transpose";
import { cpu_smvp_signed } from "./miscellaneous/smvp";
import {
  parallel_bucket_reduction_1,
  parallel_bucket_reduction_2,
} from "./miscellaneous/bpr";
import { FieldMath } from "../reference/utils/FieldMath";
import { params, p, word_size, num_words } from "./implementation/cuzk/params";

const fieldMath = new FieldMath();

/*
 * End-to-end implementation of the modified cuZK MSM algorithm by Lu et al,
 * 2022: https://eprint.iacr.org/2022/1321.pdf
 *
 * Many aspects of cuZK were adapted and modified for our submission, and some
 * aspects were omitted. As such, please refer to the documentation
 * (https://hackmd.io/HNH0DcSqSka4hAaIfJNHEA) we have written for a more accurate
 * description of our work. We also used techniques by previous ZPrize contestations.
 * In summary, we took the following approach:
 *
 * 1. Perform as much of the computation within the GPU as possible, in order
 *    to minimse CPU-GPU and GPU-CPU data transfer, which is slow.
 * 2. Use optimizations inspired by previous years' submissions, such as:
 *    - Montgomery multiplication and Barrett reduction with 13-bit limb sizes
 *    - Signed bucket indices
 * 3. Careful memory management to stay within WebGPU's default buffer size
 *    limits.
 * 4. Perform the final computation of (Horner's rule) in the CPU instead of the GPU,
 *    as the number of points is small, and the time taken to compile a shader to
 *    perform this computation is greater than the time it takes for the CPU to do so.
 */
export const compute_msm = async (
  bufferPoints: BigIntPoint[] | U32ArrayPoint[] | Buffer,
  bufferScalars: bigint[] | Uint32Array[] | Buffer,
  log_result = true,
  force_recompile = false,
): Promise<{ x: bigint; y: bigint }> => {
  const input_size = bufferScalars.length / 32;
  const chunk_size = input_size >= 65536 ? 16 : 4;
  const num_columns = 2 ** chunk_size;
  const num_rows = Math.ceil(input_size / num_columns);
  const num_subtasks = Math.ceil(256 / chunk_size);

  /// Intantiate a `ShaderManager` class for managing and invoking shaders.
  const shaderManager = new ShaderManager(
    params,
    word_size,
    chunk_size,
    input_size,
    force_recompile,
  );

  /// Shaders use the same `GPUDevice` and `GPUCommandEncoder, enabling
  /// storage buffers to be reused across compute passes.
  const device = await get_device();
  const commandEncoder = device.createCommandEncoder();

  ////////////////////////////////////////////////////////////////////////////////////////////
  /// 1. Point Coordinate Conversation and Scalar Decomposition                              /
  ///                                                                                        /
  /// (1) Convert elliptic curve points (ETE Affine coordinates) into 13-bit limbs,          /
  /// and represented internally in Montgomery form by using Barret Reduction.               /
  ///                                                                                        /
  /// (2) Decompose scalars into chunk_size windows using signed bucket indices.             /
  ////////////////////////////////////////////////////////////////////////////////////////////

  /// Total thread count = workgroup_size * #x workgroups * #y workgroups * #z workgroups.
  let c_workgroup_size = 64;
  let c_num_x_workgroups = 128;
  let c_num_y_workgroups = input_size / c_workgroup_size / c_num_x_workgroups;
  const c_num_z_workgroups = 1;

  if (input_size <= 256) {
    c_workgroup_size = input_size;
    c_num_x_workgroups = 1;
    c_num_y_workgroups = 1;
  } else if (input_size > 256 && input_size <= 32768) {
    c_workgroup_size = 64;
    c_num_x_workgroups = 4;
    c_num_y_workgroups = input_size / c_workgroup_size / c_num_x_workgroups;
  } else if (input_size > 32768 && input_size <= 65536) {
    c_workgroup_size = 256;
    c_num_x_workgroups = 8;
    c_num_y_workgroups = input_size / c_workgroup_size / c_num_x_workgroups;
  } else if (input_size > 65536 && input_size <= 131072) {
    c_workgroup_size = 256;
    c_num_x_workgroups = 8;
    c_num_y_workgroups = input_size / c_workgroup_size / c_num_x_workgroups;
  } else if (input_size > 131072 && input_size <= 262144) {
    c_workgroup_size = 256;
    c_num_x_workgroups = 32;
    c_num_y_workgroups = input_size / c_workgroup_size / c_num_x_workgroups;
  } else if (input_size > 262144 && input_size <= 524288) {
    c_workgroup_size = 256;
    c_num_x_workgroups = 32;
    c_num_y_workgroups = input_size / c_workgroup_size / c_num_x_workgroups;
  } else if (input_size > 524288 && input_size <= 1048576) {
    c_workgroup_size = 256;
    c_num_x_workgroups = 32;
    c_num_y_workgroups = input_size / c_workgroup_size / c_num_x_workgroups;
  }

  const c_shader = shaderManager.gen_convert_points_and_decomp_scalars_shader(
    c_workgroup_size,
    c_num_y_workgroups,
    num_subtasks,
    num_columns,
  );

  const { point_x_sb, point_y_sb, scalar_chunks_sb } =
    await convert_point_coords_and_decompose_shaders(
      c_shader,
      c_num_x_workgroups,
      c_num_y_workgroups,
      c_num_z_workgroups,
      device,
      commandEncoder,
      bufferPoints as Buffer,
      bufferScalars as Buffer,
      num_words,
      word_size,
      num_subtasks,
      chunk_size,
    );

  ////////////////////////////////////////////////////////////////////////////////////////////
  /// 2. Sparse Matrix Transposition                                                         /
  ///                                                                                        /
  /// Compute the indices of the points which share the same                                 /
  /// scalar chunks, enabling the parallel accumulation of points                            /
  /// into buckets. Transposing each subtask (CSR sparse matrix)                             /
  /// is a serial computation.                                                               /
  ///                                                                                        /
  /// The transpose step generates the CSR sparse matrix and                                 /
  /// transpoes the matrix simultaneously, resulting in a                                    /
  /// wide and flat matrix where width of the matrix (n) = 2 ^ chunk_size                    /
  /// and height of the matrix (m) = 1.                                                      /
  ////////////////////////////////////////////////////////////////////////////////////////////

  const t_num_x_workgroups = 1;
  const t_num_y_workgroups = 1;
  const t_num_z_workgroups = 1;

  const t_shader = shaderManager.gen_transpose_shader(num_subtasks);

  const { all_csc_col_ptr_sb, all_csc_val_idxs_sb } = await transpose_gpu(
    t_shader,
    device,
    commandEncoder,
    t_num_x_workgroups,
    t_num_y_workgroups,
    t_num_z_workgroups,
    input_size,
    num_columns,
    num_rows,
    num_subtasks,
    scalar_chunks_sb,
  );

  ////////////////////////////////////////////////////////////////////////////////////////////
  /// 3. Sparse Matrix Vector Product (SMVP)                                                 /
  ///                                                                                        /
  /// Each thread handles accumulating points in a single bucket.                            /
  /// The workgroup size and number of workgroups are designed around                        /
  /// minimizing shader invocations.                                                         /
  ////////////////////////////////////////////////////////////////////////////////////////////

  const half_num_columns = num_columns / 2;
  let s_workgroup_size = 256;
  let s_num_x_workgroups = 64;
  let s_num_y_workgroups =
    half_num_columns / s_workgroup_size / s_num_x_workgroups;
  let s_num_z_workgroups = num_subtasks;

  if (half_num_columns < 32768) {
    s_workgroup_size = 32;
    s_num_x_workgroups = 1;
    s_num_y_workgroups = Math.ceil(
      half_num_columns / s_workgroup_size / s_num_x_workgroups,
    );
  }

  if (num_columns < 256) {
    s_workgroup_size = 1;
    s_num_x_workgroups = half_num_columns;
    s_num_y_workgroups = 1;
    s_num_z_workgroups = 1;
  }

  /// This is a dynamic variable that determines the number of CSR
  /// matrices processed per invocation of the shader. A safe default is 1.
  const num_subtask_chunk_size = 4;

  /// Buffers that store the SMVP result, ie. bucket sums. They are
  /// overwritten per iteration.
  const bucket_sum_coord_bytelength =
    (num_columns / 2) * num_words * 4 * num_subtasks;
  const bucket_sum_x_sb = create_sb(device, bucket_sum_coord_bytelength);
  const bucket_sum_y_sb = create_sb(device, bucket_sum_coord_bytelength);
  const bucket_sum_t_sb = create_sb(device, bucket_sum_coord_bytelength);
  const bucket_sum_z_sb = create_sb(device, bucket_sum_coord_bytelength);

  const smvp_shader = shaderManager.gen_smvp_shader(
    s_workgroup_size,
    num_columns,
  );

  for (
    let offset = 0;
    offset < num_subtasks;
    offset += num_subtask_chunk_size
  ) {
    await smvp_gpu(
      smvp_shader,
      s_num_x_workgroups / (num_subtasks / num_subtask_chunk_size),
      s_num_y_workgroups,
      s_num_z_workgroups,
      offset,
      device,
      commandEncoder,
      num_columns,
      input_size,
      chunk_size,
      all_csc_col_ptr_sb,
      point_x_sb,
      point_y_sb,
      all_csc_val_idxs_sb,
      bucket_sum_x_sb,
      bucket_sum_y_sb,
      bucket_sum_t_sb,
      bucket_sum_z_sb,
    );
  }

  /////////////////////////////////////////////////////////////////////////////////////////////
  /// 4. Bucket Reduction                                                                     /
  ///                                                                                         /
  /// Performs a parallelized running-sum by computing a serieds of point additions,          /
  /// followed by a scalar multiplication (Algorithm 4 of the cuZK paper).                    /
  /////////////////////////////////////////////////////////////////////////////////////////////

  /// This is a dynamic variable that determines the number of CSR
  /// matrices processed per invocation of the BPR shader. A safe default is 1.
  const num_subtasks_per_bpr_1 = 16

  const b_num_x_workgroups = num_subtasks_per_bpr_1;
  const b_num_y_workgroups = 1;
  const b_num_z_workgroups = 1;
  const b_workgroup_size = 256;

  /// Buffers that store the bucket points reduction (BPR) output.
  const g_points_coord_bytelength =
    num_subtasks * b_workgroup_size * num_words * 4;
  const g_points_x_sb = create_sb(device, g_points_coord_bytelength);
  const g_points_y_sb = create_sb(device, g_points_coord_bytelength);
  const g_points_t_sb = create_sb(device, g_points_coord_bytelength);
  const g_points_z_sb = create_sb(device, g_points_coord_bytelength);

  const bpr_shader = shaderManager.gen_bpr_shader(b_workgroup_size);

  /// Stage 1: Bucket points reduction (BPR)
  for (let subtask_idx = 0; subtask_idx < num_subtasks; subtask_idx += num_subtasks_per_bpr_1) {
    await bpr_1(
      bpr_shader,
      subtask_idx,
      b_num_x_workgroups,
      b_num_y_workgroups,
      b_num_z_workgroups,
      b_workgroup_size,
      num_columns,
      device,
      commandEncoder,
      bucket_sum_x_sb,
      bucket_sum_y_sb,
      bucket_sum_t_sb,
      bucket_sum_z_sb,
      g_points_x_sb,
      g_points_y_sb,
      g_points_t_sb,
      g_points_z_sb,
    );
  }

  const num_subtasks_per_bpr_2 = 16;
  const b_2_num_x_workgroups = num_subtasks_per_bpr_2;

  /// Stage 2: Bucket points reduction (BPR).
  for (let subtask_idx = 0; subtask_idx < num_subtasks; subtask_idx += num_subtasks_per_bpr_2) {
    await bpr_2(
      bpr_shader,
      subtask_idx,
      b_2_num_x_workgroups,
      1,
      1,
      b_workgroup_size,
      num_columns,
      device,
      commandEncoder,
      bucket_sum_x_sb,
      bucket_sum_y_sb,
      bucket_sum_t_sb,
      bucket_sum_z_sb,
      g_points_x_sb,
      g_points_y_sb,
      g_points_t_sb,
      g_points_z_sb,
    );
  }

  /// Map results back from GPU to CPU.
  const data = await read_from_gpu(device, commandEncoder, [
    g_points_x_sb,
    g_points_y_sb,
    g_points_t_sb,
    g_points_z_sb,
  ]);

  /// Destroy the GPU device object.
  device.destroy();

  /// Storage buffer for performing the final accumulation on the CPU.
  const points: ExtPointType[] = [];
  const g_points_x_mont_coords = u8s_to_bigints_without_assertion(data[0], num_words, word_size);
  const g_points_y_mont_coords = u8s_to_bigints_without_assertion(data[1], num_words, word_size);
  const g_points_t_mont_coords = u8s_to_bigints_without_assertion(data[2], num_words, word_size);
  const g_points_z_mont_coords = u8s_to_bigints_without_assertion(data[3], num_words, word_size);

  for (let i = 0; i < num_subtasks; i++) {
    let point = fieldMath.customEdwards.ExtendedPoint.ZERO;
    for (let j = 0; j < b_workgroup_size; j++) {
      const reduced_point = fieldMath.createPoint(
        fieldMath.Fp.mul(
          g_points_x_mont_coords[i * b_workgroup_size + j],
          params.rinv,
        ),
        fieldMath.Fp.mul(
          g_points_y_mont_coords[i * b_workgroup_size + j],
          params.rinv,
        ),
        fieldMath.Fp.mul(
          g_points_t_mont_coords[i * b_workgroup_size + j],
          params.rinv,
        ),
        fieldMath.Fp.mul(
          g_points_z_mont_coords[i * b_workgroup_size + j],
          params.rinv,
        ),
      );
      point = point.add(reduced_point);
    }
    points.push(point);
  }

  ////////////////////////////////////////////////////////////////////////////////////////////
  /// 5. Horner's Method                                                                     /
  ///                                                                                        /
  /// Calculate the final result using Horner's method (Formula 3 of the cuZK paper)         /
  ////////////////////////////////////////////////////////////////////////////////////////////

  /// The last scalar chunk is the most significant digit (base m)
  const m = BigInt(2) ** BigInt(chunk_size);
  let result = points[points.length - 1];
  for (let i = points.length - 2; i >= 0; i--) {
    result = result.multiply(m);
    result = result.add(points[i]);
  }

  if (log_result) {
    console.log(result.toAffine());
  }
  return result.toAffine();
};

/****************************************************** WGSL Shader Invocations ******************************************************/

/*
 * Prepares and executes the shader for converting point coordinates to Montgomery form
 * and decomposing scalars into chunk_size windows using the signed bucket index technique.
 *
 * ASSUMPTION: the vast majority of WebGPU-enabled consumer devices have a
 * maximum buffer size of at least 268435456 bytes.
 *
 * The default maximum buffer size is 268435456 bytes. Since each point
 * consumes 320 bytes, a maximum of around 2 ** 19 points can be stored in a
 * single buffer. If, however, we use 2 buffers - one for each point coordinate
 * X and Y - we can support larger input sizes.
 * Our implementation, however, will only support up to 2 ** 20 points as that
 * is the maximum input size for the ZPrize competition.
 *
 * Furthremore, there is a limit of 8 storage buffers per shader. As such, we
 * do not calculate the T and Z coordinates in this shader. Rather, we do so in
 * the SMVP shader.
 *
 * Note that The test harness readme at
 * https://github.com/demox-labs/webgpu-msm states: "The submission should
 * produce correct outputs on input vectors with length up to 2^20. The
 * evaluation will be using input randomly sampled from size 2^16 ~ 2^20."
 */
export const convert_point_coords_and_decompose_shaders = async (
  shaderCode: string,
  num_x_workgroups: number,
  num_y_workgroups: number,
  num_z_workgroups: number,
  device: GPUDevice,
  commandEncoder: GPUCommandEncoder,
  points_buffer: Buffer,
  scalars_buffer: Buffer,
  num_words: number,
  word_size: number,
  num_subtasks: number,
  chunk_size: number,
  debug = false,
) => {
  assert(num_subtasks * chunk_size === 256);
  const input_size = scalars_buffer.length / 32;

  /// Input storage buffers.
  const points_sb = create_and_write_sb(device, points_buffer);
  const scalars_sb = create_and_write_sb(device, scalars_buffer);

  /// Output storage buffers.
  const point_x_sb = create_sb(device, input_size * num_words * 4);
  const point_y_sb = create_sb(device, input_size * num_words * 4);
  const scalar_chunks_sb = create_sb(device, input_size * num_subtasks * 4);

  /// Uniform storage buffer.
  const params_bytes = numbers_to_u8s_for_gpu([input_size]);
  const params_ub = create_and_write_ub(device, params_bytes);

  const bindGroupLayout = create_bind_group_layout(device, [
    "read-only-storage",
    "read-only-storage",
    "storage",
    "storage",
    "storage",
    "uniform",
  ]);

  const bindGroup = create_bind_group(device, bindGroupLayout, [
    points_sb,
    scalars_sb,
    point_x_sb,
    point_y_sb,
    scalar_chunks_sb,
    params_ub,
  ]);

  const computePipeline = await create_compute_pipeline(
    device,
    [bindGroupLayout],
    shaderCode,
    "main",
  );

  execute_pipeline(
    commandEncoder,
    computePipeline,
    bindGroup,
    num_x_workgroups,
    num_y_workgroups,
    num_z_workgroups,
  );

  /// Debug the output of the shader. This should **not** be run in production.
  if (debug) {
    await debug_point_conv_and_scalar_decomp_shader(
      device,
      commandEncoder,
      point_x_sb,
      point_y_sb,
      scalar_chunks_sb,
      input_size,
      points_buffer,
      scalars_buffer,
      num_subtasks,
      chunk_size,
    );
  }

  return { point_x_sb, point_y_sb, scalar_chunks_sb };
};

/*
 * Perform a modified version of CSR matrix transposition, which comes before
 * SMVP. Essentially, this step generates the point indices for each thread in
 * the SMVP step which corresponds to a particular bucket.
 */
export const transpose_gpu = async (
  shaderCode: string,
  device: GPUDevice,
  commandEncoder: GPUCommandEncoder,
  num_x_workgroups: number,
  num_y_workgroups: number,
  num_z_workgroups: number,
  input_size: number,
  num_columns: number,
  num_rows: number,
  num_subtasks: number,
  scalar_chunks_sb: GPUBuffer,
  debug = false,
): Promise<{
  all_csc_col_ptr_sb: GPUBuffer;
  all_csc_val_idxs_sb: GPUBuffer;
}> => {
  /// Input storage buffers.
  const all_csc_col_ptr_sb = create_sb(
    device,
    num_subtasks * (num_columns + 1) * 4,
  );
  const all_csc_val_idxs_sb = create_sb(device, scalar_chunks_sb.size);
  const all_curr_sb = create_sb(device, num_subtasks * num_columns * 4);

  /// Uniform storage buffer.
  const params_bytes = numbers_to_u8s_for_gpu([
    num_rows,
    num_columns,
    input_size,
  ]);
  const params_ub = create_and_write_ub(device, params_bytes);

  const bindGroupLayout = create_bind_group_layout(device, [
    "read-only-storage",
    "storage",
    "storage",
    "storage",
    "uniform",
  ]);

  const bindGroup = create_bind_group(device, bindGroupLayout, [
    scalar_chunks_sb,
    all_csc_col_ptr_sb,
    all_csc_val_idxs_sb,
    all_curr_sb,
    params_ub,
  ]);

  const computePipeline = await create_compute_pipeline(
    device,
    [bindGroupLayout],
    shaderCode,
    "main",
  );

  execute_pipeline(
    commandEncoder,
    computePipeline,
    bindGroup,
    num_x_workgroups,
    num_y_workgroups,
    num_z_workgroups,
  );

  /// Debug the output of the shader. This should **not** be run in production.
  if (debug) {
    await debug_transpose_shader(
      device,
      commandEncoder,
      scalar_chunks_sb,
      all_csc_col_ptr_sb,
      all_csc_val_idxs_sb,
      input_size,
      num_subtasks,
      num_rows,
      num_columns,
    );
  }

  return {
    all_csc_col_ptr_sb,
    all_csc_val_idxs_sb,
  };
};

/*
 * Compute the bucket sums and perform scalar multiplication with the bucket indices.
 */
export const smvp_gpu = async (
  shaderCode: string,
  num_x_workgroups: number,
  num_y_workgroups: number,
  num_z_workgroups: number,
  offset: number,
  device: GPUDevice,
  commandEncoder: GPUCommandEncoder,
  num_csr_cols: number,
  input_size: number,
  chunk_size: number,
  all_csc_col_ptr_sb: GPUBuffer,
  point_x_sb: GPUBuffer,
  point_y_sb: GPUBuffer,
  all_csc_val_idxs_sb: GPUBuffer,
  bucket_sum_x_sb: GPUBuffer,
  bucket_sum_y_sb: GPUBuffer,
  bucket_sum_t_sb: GPUBuffer,
  bucket_sum_z_sb: GPUBuffer,
  debug = false,
) => {
  /// Uniform Storage Buffer.
  const params_bytes = numbers_to_u8s_for_gpu([
    input_size,
    num_y_workgroups,
    num_z_workgroups,
    offset,
  ]);
  const params_ub = create_and_write_ub(device, params_bytes);

  const bindGroupLayout = create_bind_group_layout(device, [
    "read-only-storage",
    "read-only-storage",
    "read-only-storage",
    "read-only-storage",
    "storage",
    "storage",
    "storage",
    "storage",
    "uniform",
  ]);

  const bindGroup = create_bind_group(device, bindGroupLayout, [
    all_csc_col_ptr_sb,
    all_csc_val_idxs_sb,
    point_x_sb,
    point_y_sb,
    bucket_sum_x_sb,
    bucket_sum_y_sb,
    bucket_sum_t_sb,
    bucket_sum_z_sb,
    params_ub,
  ]);

  const computePipeline = await create_compute_pipeline(
    device,
    [bindGroupLayout],
    shaderCode,
    "main",
  );

  execute_pipeline(
    commandEncoder,
    computePipeline,
    bindGroup,
    num_x_workgroups,
    num_y_workgroups,
    num_z_workgroups,
  );

  /// Debug the output of the shader. This should **not** be run in production.
  if (debug) {
    await debug_smvp_shader(
      device,
      commandEncoder,
      all_csc_col_ptr_sb,
      all_csc_val_idxs_sb,
      point_x_sb,
      point_y_sb,
      bucket_sum_x_sb,
      bucket_sum_y_sb,
      bucket_sum_t_sb,
      bucket_sum_z_sb,
      input_size,
      num_csr_cols,
      offset,
      chunk_size,
    );
  }

  return {
    bucket_sum_x_sb,
    bucket_sum_y_sb,
    bucket_sum_t_sb,
    bucket_sum_z_sb,
  };
};

const bpr_1 = async (
  shaderCode: string,
  subtask_idx: number,
  num_x_workgroups: number,
  num_y_workgroups: number,
  num_z_workgroups: number,
  workgroup_size: number,
  num_columns: number,
  device: GPUDevice,
  commandEncoder: GPUCommandEncoder,
  bucket_sum_x_sb: GPUBuffer,
  bucket_sum_y_sb: GPUBuffer,
  bucket_sum_t_sb: GPUBuffer,
  bucket_sum_z_sb: GPUBuffer,
  g_points_x_sb: GPUBuffer,
  g_points_y_sb: GPUBuffer,
  g_points_t_sb: GPUBuffer,
  g_points_z_sb: GPUBuffer,
  debug = false,
) => {
  /// Uniform storage buffer.
  const params_bytes = numbers_to_u8s_for_gpu([
    subtask_idx, num_columns, num_x_workgroups
  ]);
  const params_ub = create_and_write_ub(device, params_bytes);

  const bindGroupLayout = create_bind_group_layout(device, [
    "storage",
    "storage",
    "storage",
    "storage",
    "storage",
    "storage",
    "storage",
    "storage",
    "uniform",
  ]);

  const bindGroup = create_bind_group(device, bindGroupLayout, [
    bucket_sum_x_sb,
    bucket_sum_y_sb,
    bucket_sum_t_sb,
    bucket_sum_z_sb,
    g_points_x_sb,
    g_points_y_sb,
    g_points_t_sb,
    g_points_z_sb,
    params_ub,
  ]);

  const computePipeline = await create_compute_pipeline(
    device,
    [bindGroupLayout],
    shaderCode,
    "stage_1",
  );

  execute_pipeline(
    commandEncoder,
    computePipeline,
    bindGroup,
    num_x_workgroups,
    num_y_workgroups,
    num_z_workgroups,
  );

  /// Debug the output of the shader. This should **not** be run in production.
  if (debug) {
    await debug_bpr_stage_one_shader(
      device,
      commandEncoder,
      bucket_sum_x_sb,
      bucket_sum_y_sb,
      bucket_sum_t_sb,
      bucket_sum_z_sb,
      g_points_x_sb,
      g_points_y_sb,
      g_points_t_sb,
      g_points_z_sb,
      num_columns,
      subtask_idx,
      num_x_workgroups,
      workgroup_size,
    );
  }
};

const bpr_2 = async (
  shaderCode: string,
  subtask_idx: number,
  num_x_workgroups: number,
  num_y_workgroups: number,
  num_z_workgroups: number,
  workgroup_size: number,
  num_columns: number,
  device: GPUDevice,
  commandEncoder: GPUCommandEncoder,
  bucket_sum_x_sb: GPUBuffer,
  bucket_sum_y_sb: GPUBuffer,
  bucket_sum_t_sb: GPUBuffer,
  bucket_sum_z_sb: GPUBuffer,
  g_points_x_sb: GPUBuffer,
  g_points_y_sb: GPUBuffer,
  g_points_t_sb: GPUBuffer,
  g_points_z_sb: GPUBuffer,
  debug = false,
) => {
  /// Uniform storage buffer.
  const params_bytes = numbers_to_u8s_for_gpu([
    subtask_idx, num_columns, num_x_workgroups
  ]);
  const params_ub = create_and_write_ub(device, params_bytes);

  const bindGroupLayout = create_bind_group_layout(device, [
    "storage",
    "storage",
    "storage",
    "storage",
    "storage",
    "storage",
    "storage",
    "storage",
    "uniform",
  ]);

  const bindGroup = create_bind_group(device, bindGroupLayout, [
    bucket_sum_x_sb,
    bucket_sum_y_sb,
    bucket_sum_t_sb,
    bucket_sum_z_sb,
    g_points_x_sb,
    g_points_y_sb,
    g_points_t_sb,
    g_points_z_sb,
    params_ub,
  ]);

  const computePipeline = await create_compute_pipeline(
    device,
    [bindGroupLayout],
    shaderCode,
    "stage_2",
  );

  execute_pipeline(
    commandEncoder,
    computePipeline,
    bindGroup,
    num_x_workgroups,
    num_y_workgroups,
    num_z_workgroups,
  );

  /// Debug the output of the shader. This should **not** be run in production.
  if (debug) {
    await debug_bpr_stage_two_shader(
      device,
      commandEncoder,
      bucket_sum_x_sb,
      bucket_sum_y_sb,
      bucket_sum_t_sb,
      bucket_sum_z_sb,
      g_points_x_sb,
      g_points_y_sb,
      g_points_t_sb,
      g_points_z_sb,
      num_columns,
      subtask_idx,
      num_x_workgroups,
      workgroup_size,
    );
  }
};

/****************************************************** WGSL Shader Debugging ******************************************************/

const debug_point_conv_and_scalar_decomp_shader = async (
  device: GPUDevice,
  commandEncoder: GPUCommandEncoder,
  point_x_sb: GPUBuffer,
  point_y_sb: GPUBuffer,
  scalar_chunks_sb: GPUBuffer,
  input_size: number,
  points_buffer: Buffer,
  scalars_buffer: Buffer,
  num_subtasks: number,
  chunk_size: number,
) => {
  const data = await read_from_gpu(device, commandEncoder, [
    point_x_sb,
    point_y_sb,
    scalar_chunks_sb,
  ]);

  /// Verify point coords.
  const x_coords: bigint[] = [];
  const y_coords: bigint[] = [];
  const computed_x_coords = u8s_to_bigints(data[0], num_words, word_size);
  const computed_y_coords = u8s_to_bigints(data[1], num_words, word_size);

  for (let i = 0; i < input_size; i++) {
    const x_slice = points_buffer.subarray(i * 64, i * 64 + 32);
    x_coords.push(from_words_le(new Uint16Array(x_slice), 32, 8));
    const y_slice = points_buffer.subarray(i * 64 + 32, i * 64 + 64);
    y_coords.push(from_words_le(new Uint16Array(y_slice), 32, 8));
  }

  for (let i = 0; i < input_size; i++) {
    const expected_x = (x_coords[i] * params.r) % p;
    const expected_y = (y_coords[i] * params.r) % p;
    if (
      !(
        expected_x === computed_x_coords[i] &&
        expected_y === computed_y_coords[i]
      )
    ) {
      console.log("mismatch at", i);
      break;
    }
  }

  /// Verify scalar chunks.
  const computed_chunks = u8s_to_numbers(data[2]);
  const scalars = readBigIntsFromBufferLE(scalars_buffer);
  const expected = decompose_scalars_signed(scalars, num_subtasks, chunk_size);

  for (let j = 0; j < expected.length; j++) {
    let z = 0;
    for (let i = j * input_size; i < (j + 1) * input_size; i++) {
      if (computed_chunks[i] !== expected[j][z]) {
        throw Error(`scalar decomp mismatch at ${i}`);
      }
      z++;
    }
  }
};

const debug_transpose_shader = async (
  device: GPUDevice,
  commandEncoder: GPUCommandEncoder,
  scalar_chunks_sb: GPUBuffer,
  all_csc_col_ptr_sb: GPUBuffer,
  all_csc_val_idxs_sb: GPUBuffer,
  input_size: number,
  num_subtasks: number,
  num_rows: number,
  num_columns: number,
) => {
  const data = await read_from_gpu(device, commandEncoder, [
    all_csc_col_ptr_sb,
    all_csc_val_idxs_sb,
    scalar_chunks_sb,
  ]);

  const all_csc_col_ptr_result = u8s_to_numbers_32(data[0]);
  const all_csc_val_idxs_result = u8s_to_numbers_32(data[1]);
  const new_scalar_chunks = u8s_to_numbers_32(data[2]);

  /// Verify the output of the shader.
  const expected = cpu_transpose(
    new_scalar_chunks,
    num_columns,
    num_rows,
    num_subtasks,
    input_size,
  );

  assert(
    expected.all_csc_col_ptr.toString() === all_csc_col_ptr_result.toString(),
    "all_csc_col_ptr mismatch",
  );
  assert(
    expected.all_csc_vals.toString() === all_csc_val_idxs_result.toString(),
    "all_csc_vals mismatch",
  );
};

const debug_smvp_shader = async (
  device: GPUDevice,
  commandEncoder: GPUCommandEncoder,
  all_csc_col_ptr_sb: GPUBuffer,
  all_csc_val_idxs_sb: GPUBuffer,
  point_x_sb: GPUBuffer,
  point_y_sb: GPUBuffer,
  bucket_sum_x_sb: GPUBuffer,
  bucket_sum_y_sb: GPUBuffer,
  bucket_sum_t_sb: GPUBuffer,
  bucket_sum_z_sb: GPUBuffer,
  input_size: number,
  num_csr_cols: number,
  offset: number,
  chunk_size: number,
) => {
  const data = await read_from_gpu(device, commandEncoder, [
    all_csc_col_ptr_sb,
    all_csc_val_idxs_sb,
    point_x_sb,
    point_y_sb,
    bucket_sum_x_sb,
    bucket_sum_y_sb,
    bucket_sum_t_sb,
    bucket_sum_z_sb,
  ]);

  const all_csc_col_ptr_sb_result = u8s_to_numbers_32(data[0]);
  const all_csc_val_idxs_result = u8s_to_numbers_32(data[1]);
  const point_x_sb_result = u8s_to_bigints(data[2], num_words, word_size);
  const point_y_sb_result = u8s_to_bigints(data[3], num_words, word_size);
  const bucket_sum_x_sb_result = u8s_to_bigints(data[4], num_words, word_size);
  const bucket_sum_y_sb_result = u8s_to_bigints(data[5], num_words, word_size);
  const bucket_sum_t_sb_result = u8s_to_bigints(data[6], num_words, word_size);
  const bucket_sum_z_sb_result = u8s_to_bigints(data[7], num_words, word_size);

  /// Convert GPU output out of Montgomery coordinates.
  const bigIntPointToExtPointType = (bip: BigIntPoint): ExtPointType => {
    return fieldMath.createPoint(bip.x, bip.y, bip.t, bip.z);
  };
  const output_points_gpu: ExtPointType[] = [];
  for (
    let i = offset * (num_csr_cols / 2);
    i < offset * (num_csr_cols / 2) + num_csr_cols / 2;
    i++
  ) {
    const non = {
      x: fieldMath.Fp.mul(bucket_sum_x_sb_result[i], params.rinv),
      y: fieldMath.Fp.mul(bucket_sum_y_sb_result[i], params.rinv),
      t: fieldMath.Fp.mul(bucket_sum_t_sb_result[i], params.rinv),
      z: fieldMath.Fp.mul(bucket_sum_z_sb_result[i], params.rinv),
    };
    output_points_gpu.push(bigIntPointToExtPointType(non));
  }

  const zero = fieldMath.customEdwards.ExtendedPoint.ZERO;

  const input_points: ExtPointType[] = [];
  for (let i = 0; i < input_size; i++) {
    const x = fieldMath.Fp.mul(point_x_sb_result[i], params.rinv);
    const y = fieldMath.Fp.mul(point_y_sb_result[i], params.rinv);
    const t = fieldMath.Fp.mul(x, y);
    const pt = fieldMath.createPoint(x, y, t, BigInt(1));
    if (!pt.equals(zero)) {
      pt.assertValidity();
    }
    input_points.push(pt);
  }

  // Calculate SMVP in the CPU.
  const output_points_cpu: ExtPointType[] = cpu_smvp_signed(
    offset,
    input_size,
    num_csr_cols,
    chunk_size,
    all_csc_col_ptr_sb_result,
    all_csc_val_idxs_result,
    input_points,
    fieldMath,
  );

  /// Transform results into affine representation.
  const output_points_affine_cpu = output_points_cpu.map((x) => x.toAffine());
  const output_points_affine_gpu = output_points_gpu.map((x) => x.toAffine());

  for (let i = 0; i < output_points_affine_gpu.length; i++) {
    assert(
      output_points_affine_gpu[i].x === output_points_affine_cpu[i].x &&
        output_points_affine_gpu[i].y === output_points_affine_cpu[i].y,
      `failed at offset ${offset}, i = ${i}`,
    );
  }
};

const debug_bpr_stage_one_shader = async (
  device: GPUDevice,
  commandEncoder: GPUCommandEncoder,
  bucket_sum_x_sb: GPUBuffer,
  bucket_sum_y_sb: GPUBuffer,
  bucket_sum_t_sb: GPUBuffer,
  bucket_sum_z_sb: GPUBuffer,
  g_points_x_sb: GPUBuffer,
  g_points_y_sb: GPUBuffer,
  g_points_t_sb: GPUBuffer,
  g_points_z_sb: GPUBuffer,
  num_columns: number,
  subtask_idx: number,
  num_x_workgroups: number,
  workgroup_size: number,
) => {
  const original_bucket_sum_x_sb = create_sb(device, bucket_sum_x_sb.size);
  const original_bucket_sum_y_sb = create_sb(device, bucket_sum_y_sb.size);
  const original_bucket_sum_t_sb = create_sb(device, bucket_sum_t_sb.size);
  const original_bucket_sum_z_sb = create_sb(device, bucket_sum_z_sb.size);

  commandEncoder.copyBufferToBuffer(
    bucket_sum_x_sb,
    0,
    original_bucket_sum_x_sb,
    0,
    bucket_sum_x_sb.size,
  );
  commandEncoder.copyBufferToBuffer(
    bucket_sum_y_sb,
    0,
    original_bucket_sum_y_sb,
    0,
    bucket_sum_y_sb.size,
  );
  commandEncoder.copyBufferToBuffer(
    bucket_sum_t_sb,
    0,
    original_bucket_sum_t_sb,
    0,
    bucket_sum_t_sb.size,
  );
  commandEncoder.copyBufferToBuffer(
    bucket_sum_z_sb,
    0,
    original_bucket_sum_z_sb,
    0,
    bucket_sum_z_sb.size,
  );

  const data = await read_from_gpu(device, commandEncoder, [
    bucket_sum_x_sb,
    bucket_sum_y_sb,
    bucket_sum_t_sb,
    bucket_sum_z_sb,
    g_points_x_sb,
    g_points_y_sb,
    g_points_t_sb,
    g_points_z_sb,
    original_bucket_sum_x_sb,
    original_bucket_sum_y_sb,
    original_bucket_sum_t_sb,
    original_bucket_sum_z_sb,
  ]);

  /// The number of buckets per subtask
  const n = num_columns / 2;
  const start = subtask_idx * n * num_words * 4;
  const end = (subtask_idx * n + n) * num_words * 4;
  //const num_threads = num_x_workgroups * workgroup_size;
  const num_threads = workgroup_size;

  const m_points_x_mont_coords = u8s_to_bigints(
    data[0].slice(start, end),
    num_words,
    word_size,
  );
  const m_points_y_mont_coords = u8s_to_bigints(
    data[1].slice(start, end),
    num_words,
    word_size,
  );
  const m_points_t_mont_coords = u8s_to_bigints(
    data[2].slice(start, end),
    num_words,
    word_size,
  );
  const m_points_z_mont_coords = u8s_to_bigints(
    data[3].slice(start, end),
    num_words,
    word_size,
  );

  const g_points_x_mont_coords = u8s_to_bigints(data[4], num_words, word_size);
  const g_points_y_mont_coords = u8s_to_bigints(data[5], num_words, word_size);
  const g_points_t_mont_coords = u8s_to_bigints(data[6], num_words, word_size);
  const g_points_z_mont_coords = u8s_to_bigints(data[7], num_words, word_size);

  const original_bucket_sum_x_mont_coords = u8s_to_bigints(
    data[8].slice(start, end),
    num_words,
    word_size,
  );
  const original_bucket_sum_y_mont_coords = u8s_to_bigints(
    data[9].slice(start, end),
    num_words,
    word_size,
  );
  const original_bucket_sum_t_mont_coords = u8s_to_bigints(
    data[10].slice(start, end),
    num_words,
    word_size,
  );
  const original_bucket_sum_z_mont_coords = u8s_to_bigints(
    data[11].slice(start, end),
    num_words,
    word_size,
  );

  const zero = fieldMath.customEdwards.ExtendedPoint.ZERO;

  /// Convert the bucket sums out of Montgomery form
  const original_bucket_sums: ExtPointType[] = [];
  for (let i = 0; i < n; i++) {
    const pt = fieldMath.createPoint(
      fieldMath.Fp.mul(original_bucket_sum_x_mont_coords[i], params.rinv),
      fieldMath.Fp.mul(original_bucket_sum_y_mont_coords[i], params.rinv),
      fieldMath.Fp.mul(original_bucket_sum_t_mont_coords[i], params.rinv),
      fieldMath.Fp.mul(original_bucket_sum_z_mont_coords[i], params.rinv),
    );
    if (!pt.equals(zero)) {
      pt.assertValidity();
    }
    original_bucket_sums.push(pt);
  }

  const m_points: ExtPointType[] = [];
  for (let i = 0; i < n; i++) {
    const pt = fieldMath.createPoint(
      fieldMath.Fp.mul(m_points_x_mont_coords[i], params.rinv),
      fieldMath.Fp.mul(m_points_y_mont_coords[i], params.rinv),
      fieldMath.Fp.mul(m_points_t_mont_coords[i], params.rinv),
      fieldMath.Fp.mul(m_points_z_mont_coords[i], params.rinv),
    );
    if (!pt.equals(zero)) {
      pt.assertValidity();
    }
    m_points.push(pt);
  }

  /// Convert the reduced buckets out of Montgomery form
  const g_points: ExtPointType[] = [];
  for (let i = 0; i < num_threads; i++) {
    const idx = subtask_idx * num_threads + i;
    const pt = fieldMath.createPoint(
      fieldMath.Fp.mul(g_points_x_mont_coords[idx], params.rinv),
      fieldMath.Fp.mul(g_points_y_mont_coords[idx], params.rinv),
      fieldMath.Fp.mul(g_points_t_mont_coords[idx], params.rinv),
      fieldMath.Fp.mul(g_points_z_mont_coords[idx], params.rinv),
    );
    if (!pt.equals(zero)) {
      pt.assertValidity();
    }
    g_points.push(pt);
  }

  const expected = parallel_bucket_reduction_1(
    original_bucket_sums,
    num_threads,
  );

  for (let i = 0; i < expected.g_points.length; i++) {
    assert(g_points[i].equals(expected.g_points[i]), `mismatch at ${i}`);
  }
};

const debug_bpr_stage_two_shader = async (
  device: GPUDevice,
  commandEncoder: GPUCommandEncoder,
  bucket_sum_x_sb: GPUBuffer,
  bucket_sum_y_sb: GPUBuffer,
  bucket_sum_t_sb: GPUBuffer,
  bucket_sum_z_sb: GPUBuffer,
  g_points_x_sb: GPUBuffer,
  g_points_y_sb: GPUBuffer,
  g_points_t_sb: GPUBuffer,
  g_points_z_sb: GPUBuffer,
  num_columns: number,
  subtask_idx: number,
  num_x_workgroups: number,
  workgroup_size: number,
) => {
  const data = await read_from_gpu(device, commandEncoder, [
    bucket_sum_x_sb,
    bucket_sum_y_sb,
    bucket_sum_t_sb,
    bucket_sum_z_sb,
    g_points_x_sb,
    g_points_y_sb,
    g_points_t_sb,
    g_points_z_sb,
  ]);

  /// The number of buckets per subtask.
  const n = num_columns / 2;
  const start = subtask_idx * n * num_words * 4;
  const end = (subtask_idx * n + n) * num_words * 4;
  const num_threads = num_x_workgroups * workgroup_size;

  const m_points_x_mont_coords = u8s_to_bigints(
    data[0].slice(start, end),
    num_words,
    word_size,
  );
  const m_points_y_mont_coords = u8s_to_bigints(
    data[1].slice(start, end),
    num_words,
    word_size,
  );
  const m_points_t_mont_coords = u8s_to_bigints(
    data[2].slice(start, end),
    num_words,
    word_size,
  );
  const m_points_z_mont_coords = u8s_to_bigints(
    data[3].slice(start, end),
    num_words,
    word_size,
  );

  const g_points_x_mont_coords = u8s_to_bigints(data[4], num_words, word_size);
  const g_points_y_mont_coords = u8s_to_bigints(data[5], num_words, word_size);
  const g_points_t_mont_coords = u8s_to_bigints(data[6], num_words, word_size);
  const g_points_z_mont_coords = u8s_to_bigints(data[7], num_words, word_size);

  const zero = fieldMath.customEdwards.ExtendedPoint.ZERO;

  const m_points: ExtPointType[] = [];
  for (let i = 0; i < n; i++) {
    const pt = fieldMath.createPoint(
      fieldMath.Fp.mul(m_points_x_mont_coords[i], params.rinv),
      fieldMath.Fp.mul(m_points_y_mont_coords[i], params.rinv),
      fieldMath.Fp.mul(m_points_t_mont_coords[i], params.rinv),
      fieldMath.Fp.mul(m_points_z_mont_coords[i], params.rinv),
    );
    if (!pt.equals(zero)) {
      pt.assertValidity();
    }
    m_points.push(pt);
  }

  /// Convert the reduced buckets out of Montgomery form.
  const g_points: ExtPointType[] = [];
  for (let i = 0; i < num_threads; i++) {
    const idx = subtask_idx * num_threads + i;
    const pt = fieldMath.createPoint(
      fieldMath.Fp.mul(g_points_x_mont_coords[idx], params.rinv),
      fieldMath.Fp.mul(g_points_y_mont_coords[idx], params.rinv),
      fieldMath.Fp.mul(g_points_t_mont_coords[idx], params.rinv),
      fieldMath.Fp.mul(g_points_z_mont_coords[idx], params.rinv),
    );
    if (!pt.equals(zero)) {
      pt.assertValidity();
    }
    g_points.push(pt);
  }

  const expected = parallel_bucket_reduction_2(
    g_points,
    m_points,
    n,
    num_threads,
  );
  for (let i = 0; i < expected.length; i++) {
    assert(g_points[i].equals(expected[i]), `mismatch at ${i}`);
  }
};
