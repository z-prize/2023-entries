import { BigIntPoint, U32ArrayPoint } from "../../../reference/types";
import {
  ELLSparseMatrix,
  CSRSparseMatrix,
  fieldMath,
} from "../matrices/matrices";
import { ExtPointType } from "@noble/curves/abstract/edwards";

// Typescript implementation of cuZK
export const cuzk_typescript_serial = async (
  baseAffinePoints: BigIntPoint[] | U32ArrayPoint[] | Buffer,
  scalars: bigint[] | Uint32Array[] | Buffer,
): Promise<any> => {
  console.log("Starting Serial cuZK!");

  const result = await execute_cuzk(
    baseAffinePoints as BigIntPoint[],
    scalars as bigint[],
  );

  const result_affine = result.toAffine();
  const x = result_affine.x;
  const y = result_affine.y;

  return { x, y };
};

export async function init(
  baseAffinePoints: BigIntPoint[],
  scalars: bigint[],
): Promise<CSRSparseMatrix[]> {
  // λ-bit scalars
  const lambda = 256;

  // s-bit window size
  const s = 16;

  // Thread count
  const threads = 16;

  // Number of rows and columns (ie. row-space)
  const num_rows = threads;
  const num_columns = Math.pow(2, s) - 1;

  const ZERO_POINT = fieldMath.customEdwards.ExtendedPoint.ZERO;

  const csr_sparse_matrix_array: CSRSparseMatrix[] = [];

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
    const scalars_decomposed: bigint[][] = [];
    for (let j = Math.ceil(lambda / s); j > 0; j--) {
      const chunk: bigint[] = [];
      for (let i = 0; i < scalars.length; i++) {
        const mask = (BigInt(1) << BigInt(s)) - BigInt(1);
        const limb = (scalars[i] >> BigInt((j - 1) * s)) & mask; // Right shift and extract lower 32-bits
        chunk.push(limb);
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

export async function transpose_and_spmv(
  csr_sparse_matrix: CSRSparseMatrix,
): Promise<ExtPointType> {
  // Transpose CSR sparse matrix
  const csr_sparse_matrix_transposed = await csr_sparse_matrix.transpose();

  // Derive partial bucket sums by performing SMVP
  const vector_smvp: bigint[] = Array(
    csr_sparse_matrix_transposed.row_ptr.length - 1,
  ).fill(BigInt(1));
  const buckets_svmp: ExtPointType[] =
    await csr_sparse_matrix_transposed.smvp(vector_smvp);

  // Aggregate SVMP buckets with running sum
  let aggregate_svmp: ExtPointType = fieldMath.customEdwards.ExtendedPoint.ZERO;
  for (const [i, bucket] of buckets_svmp.entries()) {
    if (i == 0) {
      continue;
    }
    aggregate_svmp = aggregate_svmp.add(bucket.multiply(BigInt(i)));
  }

  return aggregate_svmp;
}

export async function smtvp(
  csr_sparse_matrix: CSRSparseMatrix,
): Promise<ExtPointType> {
  // Derive partial bucket sums by performing SMTVP
  const vector_smtvp: bigint[] = Array(
    csr_sparse_matrix.row_ptr.length - 1,
  ).fill(BigInt(1));
  const buckets_svtmp: ExtPointType[] =
    await csr_sparse_matrix.smtvp(vector_smtvp);

  // Aggregate SVTMP buckets with running sum
  let aggregate_svtmp: ExtPointType =
    fieldMath.customEdwards.ExtendedPoint.ZERO;
  for (const [i, bucket] of buckets_svtmp.entries()) {
    if (i == 0) {
      continue;
    }
    aggregate_svtmp = aggregate_svtmp.add(bucket.multiply(BigInt(i)));
  }

  return aggregate_svtmp;
}

export async function execute_cuzk(
  baseAffinePoints: BigIntPoint[],
  scalars: bigint[],
): Promise<ExtPointType> {
  // Initialize instance
  const csr_sparse_matrix_array = await init(baseAffinePoints, scalars);

  // Perform Transpose and SPMV
  const G: ExtPointType[] = [];
  for (let i = 0; i < csr_sparse_matrix_array.length; i++) {
    const partial_sum = await transpose_and_spmv(csr_sparse_matrix_array[i]);
    G.push(partial_sum);
  }

  // Perform Honer's rule
  let T = G[0];
  for (let i = 1; i < G.length; i++) {
    T = T.multiply(BigInt(Math.pow(2, 16)));
    T = T.add(G[i]);
  }

  return T;
}
