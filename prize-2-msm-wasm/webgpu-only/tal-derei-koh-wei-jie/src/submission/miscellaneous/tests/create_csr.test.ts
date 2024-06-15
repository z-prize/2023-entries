import {
  create_csr_cpu,
  precompute_with_gpu_simulated,
  pre_aggregate,
} from "../cuzk/create_csr";

describe("Create an CSR sparse matrix from the MSM input points and scalars", () => {
  describe("pre-aggregation using a simulated GPU", () => {
    const num_points = 8;
    const num_rows = 1;

    it("small test #0", () => {
      const points = [];
      for (let i = 0; i < num_points; i++) {
        points.push(`P${i}`);
      }

      const decomposed_scalars = [
        [3, 3, 2, 1, 2, 1, 4, 4],
        [4, 4, 4, 3, 3, 3, 3, 0],
        [4, 4, 4, 4, 3, 3, 3, 3],
      ];

      const expected_data = [
        ["P3P5", "P2P4", "P0P1", "P6P7"],
        ["P3P4P5P6", "P0P1P2"],
        ["P4P5P6P7", "P0P1P2P3"],
      ];

      for (let i = 0; i < 3; i++) {
        const scalar_chunks = decomposed_scalars[i];
        const r = precompute_with_gpu_simulated(scalar_chunks, 0, num_rows);
        const { new_points } = pre_aggregate(
          points,
          scalar_chunks,
          r.new_point_indices,
          r.cluster_start_indices,
          r.cluster_end_indices,
        );

        expect(new_points.toString()).toEqual(expected_data[i].toString());
      }
    });
  });

  describe("pre-aggregation using the cluster method", () => {
    describe("small tests", () => {
      const num_points = 8;
      const num_rows = 2;

      it("small test #1", () => {
        const points = [];
        for (let i = 0; i < num_points; i++) {
          points.push(`P${i}`);
        }

        const decomposed_scalars = [
          [4, 4, 4, 3, 3, 3, 3, 0],
          [3, 4, 4, 3, 4, 1, 0, 2],
        ];

        const expected_data = [
          ["P0P1P2", "P4P5P6", "P3"],
          ["P0P3", "P1P2", "P4", "P5", "P7"],
        ];

        const expected_col_idxs = [
          [4, 3, 3],
          [3, 4, 4, 1, 2],
        ];

        const expected_row_ptrs = [
          [0, 2, 3],
          [0, 2, 5],
        ];

        for (let i = 0; i < decomposed_scalars.length; i++) {
          const scalar_chunks = decomposed_scalars[i];
          const csr_sm = create_csr_cpu(
            points,
            scalar_chunks,
            num_rows,
            (a: string, b: string) => a + b,
          );
          expect(csr_sm.data.toString()).toEqual(expected_data[i].toString());
          expect(csr_sm.col_idx.toString()).toEqual(
            expected_col_idxs[i].toString(),
          );
          expect(csr_sm.row_ptr.toString()).toEqual(
            expected_row_ptrs[i].toString(),
          );
        }
      });
    });
  });
});
