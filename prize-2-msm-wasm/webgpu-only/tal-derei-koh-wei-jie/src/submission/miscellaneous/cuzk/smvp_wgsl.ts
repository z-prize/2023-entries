import { BigIntPoint, U32ArrayPoint } from "../../../reference/types";
import mustache from "mustache";
import { ExtPointType } from "@noble/curves/abstract/edwards";
import {
  get_device,
  create_and_write_sb,
  create_bind_group,
  create_bind_group_layout,
  create_compute_pipeline,
  create_sb,
  read_from_gpu,
  execute_pipeline,
} from "../../implementation/cuzk/gpu";
import {
  CSRSparseMatrix,
  ELLSparseMatrix,
  fieldMath,
} from "../matrices/matrices";
import smvp_shader from "../wgsl/smvp_benchmark.template.wgsl";
import structs from "../../implementation/wgsl/struct/structs.template.wgsl";
import bigint_functions from "../../implementation/wgsl/bigint/bigint.template.wgsl";
import field_functions from "../../implementation/wgsl/field/field.template.wgsl";
import curve_functions from "../../implementation/wgsl/curve/ec.template.wgsl";
import montgomery_product_funcs from "../../implementation/wgsl/montgomery/mont_pro_product.template.wgsl";
import {
  u8s_to_points,
  points_to_u8s_for_gpu,
  numbers_to_u8s_for_gpu,
  compute_misc_params,
  gen_p_limbs,
  gen_r_limbs,
  gen_d_limbs,
} from "../../implementation/cuzk/utils";
import assert from "assert";

// WGSL implementation of Sparse-Matrix Vector Multiplication
export const smvp_wgsl = async (
  baseAffinePoints: BigIntPoint[] | U32ArrayPoint[] | Buffer,
  scalars: bigint[] | Uint32Array[] | Buffer,
): Promise<{ x: bigint; y: bigint }> => {
  console.log("Starting WGSL SMVP!");

  const result = await smvp(
    baseAffinePoints as BigIntPoint[],
    scalars as bigint[],
  );
  return result;
};

export async function smvp(
  baseAffinePoints: BigIntPoint[],
  scalars: bigint[],
): Promise<{ x: bigint; y: bigint }> {
  // Scalar finite field
  const p = BigInt(
    "0x12ab655e9a2ca55660b44d1e5c37b00159aa76fed00000010a11800000000001",
  );

  // s-bit window size
  const word_size = 13;

  const params = compute_misc_params(p, word_size);

  // Number of limbs (ie. windows)
  const num_words = params.num_words;

  // λ-bit scalars (13 * 20 = 260)
  const lambda = word_size * num_words;

  // Thread count
  const threads = num_words;

  const n0 = params.n0;

  // Request GPU device
  const device = await get_device();
  //const limits = device.limits

  // Generate CSR sparse matrices
  const csr_sparse_matrices = await gen_csr_sparse_matrices(
    baseAffinePoints,
    scalars,
    lambda,
    word_size,
    threads,
  );

  // WGSL Shader invocations
  for (let i = 0; i < 2; i++) {
    console.log("CSR matrix:", i);
    // Perform Sparse-Matrix Tranpose and SMVP
    await smvp_gpu(
      device,
      csr_sparse_matrices[i],
      num_words,
      word_size,
      p,
      n0,
      params.r,
      params.rinv,
      params.edwards_d,
    );
  }

  device.destroy();

  return { x: BigInt(1), y: BigInt(0) };
}

export async function gen_csr_sparse_matrices(
  baseAffinePoints: BigIntPoint[],
  scalars: bigint[],
  lambda: number,
  s: number,
  threads: number,
): Promise<any> {
  // Number of rows and columns (ie. row-space)
  const num_rows = threads;
  const num_columns = Math.pow(2, s) - 1;

  // Intantiate empty array of sparse matrices
  const csr_sparse_matrix_array: CSRSparseMatrix[] = [];

  const ZERO_POINT = fieldMath.customEdwards.ExtendedPoint.ZERO;

  for (let i = 0; i < num_rows; i++) {
    // Instantiate empty ELL sparse matrix format
    const data = new Array(num_rows);
    for (let i = 0; i < num_rows; i++) {
      data[i] = new Array(num_columns).fill(ZERO_POINT);
    }

    const col_idx = new Array(num_rows);
    for (let i = 0; i < num_rows; i++) {
      col_idx[i] = new Array(num_columns).fill(0);
    }

    const row_length = Array(num_rows).fill(0);

    // Perform scalar decomposition
    const scalars_decomposed: number[][] = [];
    for (let j = Math.ceil(lambda / s); j > 0; j--) {
      const chunk: number[] = [];
      for (let i = 0; i < scalars.length; i++) {
        const mask = (BigInt(1) << BigInt(s)) - BigInt(1);
        const limb = (scalars[i] >> BigInt((j - 1) * s)) & mask; // Right shift and extract lower 32-bits
        chunk.push(Number(limb));
      }
      scalars_decomposed.push(chunk);
    }

    // Divide EC points into t parts
    for (let thread_idx = 0; thread_idx < num_rows; thread_idx++) {
      for (let j = 0; j < num_columns; j++) {
        const point_i = thread_idx + j * threads;
        data[thread_idx][j] = baseAffinePoints[point_i];
        col_idx[thread_idx][j] = scalars_decomposed[i][point_i];
        row_length[thread_idx] += 1;
      }
    }

    // Transform ELL sparse matrix to CSR sparse matrix
    const ell_sparse_matrix = new ELLSparseMatrix(data, col_idx, row_length);
    const csr_sparse_matrix = await new CSRSparseMatrix(
      [],
      [],
      [],
    ).ell_to_csr_sparse_matrix(ell_sparse_matrix);

    csr_sparse_matrix_array.push(csr_sparse_matrix);
  }

  return csr_sparse_matrix_array;
}

export async function smvp_gpu(
  device: GPUDevice,
  csr_sm: CSRSparseMatrix,
  num_words: number,
  word_size: number,
  p: bigint,
  n0: bigint,
  r: bigint,
  rinv: bigint,
  d: bigint,
) {
  const csr_sparse_matrix_transposed = await csr_sm.transpose();

  const start_1 = Date.now();

  console.log("sparse matrix before transpose", csr_sm);
  const vector_smvp: bigint[] = Array(
    csr_sparse_matrix_transposed.row_ptr.length - 1,
  ).fill(BigInt(1));
  console.log("csr_sparse_matrix_transposed: ", csr_sparse_matrix_transposed);
  const buckets_svmp: ExtPointType[] =
    await csr_sparse_matrix_transposed.smvp(vector_smvp);
  console.log("csr_sparse_matrix_transposed after smvp: ", buckets_svmp);

  const elapsed_1 = Date.now() - start_1;
  const output_points_non_mont_and_affine_cpu = buckets_svmp.map((x) =>
    x.toAffine(),
  );
  console.log(`CPU took ${elapsed_1}ms`);

  // Convert BigIntPoint to montgomery form
  const points_with_mont_coords: BigIntPoint[] = [];
  for (const pt of csr_sparse_matrix_transposed.data) {
    points_with_mont_coords.push({
      x: fieldMath.Fp.mul(pt.x, r),
      y: fieldMath.Fp.mul(pt.y, r),
      t: fieldMath.Fp.mul(pt.t, r),
      z: fieldMath.Fp.mul(pt.z, r),
    });
  }

  const row_ptr_bytes = numbers_to_u8s_for_gpu(
    csr_sparse_matrix_transposed.row_ptr,
  );

  const NUM_ROWS = csr_sparse_matrix_transposed.row_ptr.length - 1;

  let NUM_ROWS_GPU = 0;
  let numBlocks = 1;
  if (NUM_ROWS < 256) {
    NUM_ROWS_GPU = NUM_ROWS;
  } else {
    const blockSize = 256;
    const totalThreads = NUM_ROWS;
    numBlocks = Math.floor((totalThreads + blockSize - 1) / blockSize);
    NUM_ROWS_GPU = blockSize;
  }

  // Define number of workgroups
  const num_x_workgroups = numBlocks;

  const points_bytes = points_to_u8s_for_gpu(
    points_with_mont_coords,
    num_words,
    word_size,
  );

  const p_limbs = gen_p_limbs(p, num_words, word_size);
  const r_limbs = gen_r_limbs(r, num_words, word_size);
  const d_limbs = gen_d_limbs(d, num_words, word_size);
  const shaderCode = mustache.render(
    smvp_shader,
    {
      word_size,
      num_words,
      n0,
      p_limbs,
      r_limbs,
      d_limbs,
      mask: BigInt(2) ** BigInt(word_size) - BigInt(1),
      two_pow_word_size: BigInt(2) ** BigInt(word_size),
      NUM_ROWS_GPU,
    },
    {
      structs,
      bigint_functions,
      montgomery_product_funcs,
      field_functions,
      curve_functions,
    },
  );

  const commandEncoder = device.createCommandEncoder();

  const output_buffer_length = NUM_ROWS * num_words * 4 * 4;

  const start = Date.now();
  const output_storage_buffer = create_sb(device, output_buffer_length);
  const row_ptr_storage_buffer = create_and_write_sb(device, row_ptr_bytes);
  const points_storage_buffer = create_and_write_sb(device, points_bytes);

  const bindGroupLayout = create_bind_group_layout(device, [
    "storage",
    "read-only-storage",
    "read-only-storage",
  ]);

  const bindGroup = create_bind_group(device, bindGroupLayout, [
    output_storage_buffer,
    row_ptr_storage_buffer,
    points_storage_buffer,
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
  );

  const data = await read_from_gpu(device, commandEncoder, [
    output_storage_buffer,
  ]);

  const elapsed = Date.now() - start;
  console.log(`GPU took ${elapsed}ms`);

  // Transform results
  const data_as_uint8s = new Uint8Array(data[0]);
  const output_points = u8s_to_points(data_as_uint8s, num_words, word_size);

  const bigIntPointToExtPointType = (bip: BigIntPoint): ExtPointType => {
    return fieldMath.createPoint(bip.x, bip.y, bip.t, bip.z);
  };

  // Convert output_points out of Montgomery coords
  const output_points_non_mont: ExtPointType[] = [];
  for (const pt of output_points) {
    const non = {
      x: fieldMath.Fp.mul(pt.x, rinv),
      y: fieldMath.Fp.mul(pt.y, rinv),
      t: fieldMath.Fp.mul(pt.t, rinv),
      z: fieldMath.Fp.mul(pt.z, rinv),
    };
    output_points_non_mont.push(bigIntPointToExtPointType(non));
  }

  // Convert output_points_non_mont into affine
  const output_points_non_mont_and_affine_gpu = output_points_non_mont.map(
    (x) => x.toAffine(),
  );

  for (let i = 0; i < output_points_non_mont_and_affine_gpu.length; i++) {
    assert(
      output_points_non_mont_and_affine_gpu[i].x ===
        output_points_non_mont_and_affine_cpu[i].x,
    );
    assert(
      output_points_non_mont_and_affine_gpu[i].y ===
        output_points_non_mont_and_affine_cpu[i].y,
    );
  }
}
