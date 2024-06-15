import { spawn, Thread, Worker } from "threads";
import { CSRSparseMatrix } from "../../matrices/matrices";
import { ExtPointType } from "@noble/curves/abstract/edwards";

export const webWorkers = async (
  csr_sparse_matrix: CSRSparseMatrix,
): Promise<ExtPointType> => {
  // Spawn web workers
  const worker = await spawn(new Worker("./webworkers.js"));
  const result: ExtPointType = await worker(
    csr_sparse_matrix.data,
    csr_sparse_matrix.col_idx,
    csr_sparse_matrix.row_ptr,
  );
  await Thread.terminate(worker);
  return result;
};
