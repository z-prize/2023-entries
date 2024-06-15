jest.setTimeout(10000000);
import { FieldMath } from "../../../reference/utils/FieldMath";
import { ExtPointType } from "@noble/curves/abstract/edwards";
import { decompose_scalars_signed } from "../utils";
import { cpu_transpose } from "../transpose";
import { cpu_smvp_signed } from "../smvp";
import {
  running_sum_bucket_reduction,
  parallel_bucket_reduction,
  parallel_bucket_reduction_1,
  parallel_bucket_reduction_2,
} from "../bpr";

const fieldMath = new FieldMath();
const x = BigInt(
  "2796670805570508460920584878396618987767121022598342527208237783066948667246",
);
const y = BigInt(
  "8134280397689638111748378379571739274369602049665521098046934931245960532166",
);
const t = BigInt(
  "3446088593515175914550487355059397868296219355049460558182099906777968652023",
);
const z = BigInt("1");
const pt = fieldMath.createPoint(x, y, t, z);

describe("cuzk", () => {
  it("cuzk with smaller inputs and parameters", () => {
    const p = BigInt(
      "0x12ab655e9a2ca55660b44d1e5c37b00159aa76fed00000010a11800000000001",
    );

    const input_size = 16;
    const chunk_size = 4;

    expect(input_size >= 2 ** chunk_size).toBeTruthy();

    const num_columns = 2 ** chunk_size;
    const num_rows = Math.ceil(input_size / num_columns);

    const num_chunks_per_scalar = Math.ceil(256 / chunk_size);
    const num_subtasks = num_chunks_per_scalar;

    const points: ExtPointType[] = [];
    const scalars: bigint[] = [];
    for (let i = 0; i < input_size; i++) {
      points.push(pt);
      const v = BigInt(
        "1111111111111111111111111111111111111111111111111111111111111111111111111111",
      );
      scalars.push((BigInt(i) * v) % p);
      points.push(fieldMath.createPoint(x, y, t, z).multiply(BigInt(i + 1)));
    }

    const decomposed_scalars = decompose_scalars_signed(
      scalars,
      num_subtasks,
      chunk_size,
    );

    const bucket_sums: ExtPointType[] = [];

    // Perform multiple transpositions "in parallel"
    const { all_csc_col_ptr, all_csc_vals } = cpu_transpose(
      decomposed_scalars.flat(),
      num_columns,
      num_rows,
      num_subtasks,
      input_size,
    );

    for (let subtask_idx = 0; subtask_idx < num_subtasks; subtask_idx++) {
      // Perform SMVP
      const buckets = cpu_smvp_signed(
        subtask_idx,
        input_size,
        num_columns,
        chunk_size,
        all_csc_col_ptr,
        all_csc_vals,
        points,
        fieldMath,
      );

      const bucket_sum_serial = serial_bucket_reduction(buckets);
      const bucket_sum_rs = running_sum_bucket_reduction(buckets);

      // Use the full pBucketPointReduciton algo
      let bucket_sum = fieldMath.customEdwards.ExtendedPoint.ZERO;
      for (const b of parallel_bucket_reduction(buckets)) {
        bucket_sum = bucket_sum.add(b);
      }

      expect(bucket_sum_rs.equals(bucket_sum)).toBeTruthy();
      expect(bucket_sum_serial.equals(bucket_sum)).toBeTruthy();

      bucket_sums.push(bucket_sum);

      // Use the pBucketPointReduciton algo in 2 stages
      const num_buckets = buckets.length;
      const { g_points, m_points } = parallel_bucket_reduction_1(buckets);

      const p_result = parallel_bucket_reduction_2(
        g_points,
        m_points,
        num_buckets,
      );
      let bucket_sum_2 = fieldMath.customEdwards.ExtendedPoint.ZERO;
      for (const b of p_result) {
        bucket_sum_2 = bucket_sum_2.add(b);
      }

      expect(bucket_sum_rs.equals(bucket_sum_2)).toBeTruthy();
      expect(bucket_sum_serial.equals(bucket_sum_2)).toBeTruthy();

      //console.log('-----')
    }

    // Horner's rule
    const m = BigInt(2) ** BigInt(chunk_size);
    // The last scalar chunk is the most significant digit (base m)
    let result = bucket_sums[bucket_sums.length - 1];
    for (let i = bucket_sums.length - 2; i >= 0; i--) {
      result = result.multiply(m);
      result = result.add(bucket_sums[i]);
    }

    const result_affine = result.toAffine();

    // Calculated expected result
    let expected = fieldMath.customEdwards.ExtendedPoint.ZERO;
    for (let i = 0; i < input_size; i++) {
      if (scalars[i] !== BigInt(0)) {
        const p = points[i].multiply(scalars[i]);
        expected = expected.add(p);
      }
    }
    const expected_affine = expected.toAffine();
    expect(result_affine.x).toEqual(expected_affine.x);
    expect(result_affine.y).toEqual(expected_affine.y);
  });
});

const serial_bucket_reduction = (buckets: ExtPointType[]) => {
  const indices = [];
  for (let i = 1; i < buckets.length; i++) {
    indices.push(i);
  }
  indices.push(0);

  let bucket_sum = fieldMath.customEdwards.ExtendedPoint.ZERO;

  for (let i = 1; i < buckets.length + 1; i++) {
    const b = buckets[indices[i - 1]].multiply(BigInt(i));
    bucket_sum = bucket_sum.add(b);
    //console.log(`serial: buckets[${indices[i - 1]}] * ${i}`)
  }
  return bucket_sum;
};

export {};
