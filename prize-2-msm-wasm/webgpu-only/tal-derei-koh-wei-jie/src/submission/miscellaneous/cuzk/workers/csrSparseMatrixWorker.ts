import { CSRSparseMatrix } from "../../matrices/matrices";
import { ExtPointType } from "@noble/curves/abstract/edwards";
import { expose } from "threads/worker";
import { fieldMath } from "../../matrices/matrices";

export async function transpose_and_spmv(
  data: any[],
  col_idx: any[],
  row_ptr: any[],
): Promise<ExtPointType> {
  // Reconstruct CSR sparse matrix object
  const csr_sparse_matrix = await new CSRSparseMatrix(data, col_idx, row_ptr);

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

// Expose function to web workers
expose(transpose_and_spmv);
