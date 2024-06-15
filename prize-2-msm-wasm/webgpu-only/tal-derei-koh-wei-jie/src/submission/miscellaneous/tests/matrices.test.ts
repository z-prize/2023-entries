import {
  DenseMatrix,
  ELLSparseMatrix,
  CSRSparseMatrix,
} from "../matrices/matrices";

describe("matrices", () => {
  describe("dense matrix functions", () => {
    it("test_transpose", async () => {
      // Initialize dense matrix
      const dense_matrix = new DenseMatrix([
        [0, 1],
        [1, 0],
        [2, 0],
      ]);
      const expected = new DenseMatrix([
        [0, 1, 2],
        [1, 0, 0],
      ]);

      // Transpose dense matrix
      const dense_transposed: DenseMatrix = await dense_matrix.transpose();

      expect(dense_transposed).toEqual(expected);
    });
  });

  describe("sparse matrix functions", () => {
    it("test_convert_dense_matrix", async () => {
      // Define dense matrix, and expected ELL / CSR sparse matrices
      const dense_matrix = new DenseMatrix([
        [null, 5, null, 4, null],
        [7, 2, null, null, 1],
        [null, null, 3, null, null],
        [null, null, null, null, null],
        [null, 8, 2, null, null],
      ]);

      const ell_expected = new ELLSparseMatrix(
        [
          [5, 4, null],
          [7, 2, 1],
          [3, null, null],
          [null, null, null],
          [8, 2, null],
        ],
        [
          [1, 3, null],
          [0, 1, 4],
          [2, null, null],
          [null, null, null],
          [1, 2, null],
        ],
        [2, 3, 1, 0, 2],
      );

      const csr_expected = new CSRSparseMatrix(
        [5, 4, 7, 2, 1, 3, 8, 2],
        [1, 3, 0, 1, 4, 2, 1, 2],
        [0, 2, 5, 6, 6, 8],
      );

      const ell_sparse_matrix = await new ELLSparseMatrix(
        [],
        [],
        [],
      ).dense_to_sparse_matrix(dense_matrix);

      expect(ell_sparse_matrix).toEqual(ell_expected);

      // Transform ELL to CSR sparse matrix
      const csr_sparse_matrix = await new CSRSparseMatrix(
        [],
        [],
        [],
      ).ell_to_csr_sparse_matrix(ell_sparse_matrix);

      expect(csr_sparse_matrix).toEqual(csr_expected);
    });

    it("test_smtvp", async () => {
      const dense_matrix = new DenseMatrix([
        [null, 5, null, 4, null],
        [7, 2, null, null, 1],
        [null, null, 3, null, null],
        [null, null, null, null, null],
        [null, 8, 2, null, null],
      ]);

      // Transpose dense matrix
      const dense_transposed = await dense_matrix.transpose();

      // Initialize vector
      const vec: number[] = [];
      for (let i = 0; i < dense_matrix.num_rows; i++) {
        vec.push(Math.floor(Math.random() * 100));
      }

      // Calculate dense matrix-vector product (~SMVP)
      const matrix_vec_product = await dense_transposed.matrix_vec_mult(vec);

      // Transform dense matrix to CRS sparse matrix
      const ell_sparse_matrix = await new ELLSparseMatrix(
        [],
        [],
        [],
      ).dense_to_sparse_matrix(dense_matrix);
      const csr_sparse_matrix = await new CSRSparseMatrix(
        [],
        [],
        [],
      ).ell_to_csr_sparse_matrix(ell_sparse_matrix);

      // Perform sparse transpose and vector product
      const smtvp_result = await csr_sparse_matrix.smtvp_test(vec);

      expect(smtvp_result).toEqual(matrix_vec_product);
    });

    it("test_transpose", async () => {
      const dense_matrix = new DenseMatrix([
        [null, 5, null, 4, null],
        [7, 2, null, null, 1],
        [null, null, 3, null, null],
        [null, null, null, null, null],
        [null, 8, 2, null, null],
      ]);

      const csr_sparse_matrix = new CSRSparseMatrix(
        [5, 4, 7, 2, 1, 3, 8, 2],
        [1, 3, 0, 1, 4, 2, 1, 2],
        [0, 2, 5, 6, 6, 8],
      );

      // Transpose dense matrix
      const dense_transposed = await dense_matrix.transpose();

      // Transform transposed dense matrix to transposed CRS sparse matrix
      const ell_sparse_matrix_transposed = await new ELLSparseMatrix(
        [],
        [],
        [],
      ).dense_to_sparse_matrix(dense_transposed);
      const expected_csr_transposed = await new CSRSparseMatrix(
        [],
        [],
        [],
      ).ell_to_csr_sparse_matrix(ell_sparse_matrix_transposed);

      // Transpose sparse matrix in CSR representation
      const csr_transposed = await csr_sparse_matrix.transpose_test();

      expect(csr_transposed).toEqual(expected_csr_transposed);
    });
  });
});
