import assert from "assert";
import mustache from "mustache";
import { ExtPointType } from "@noble/curves/abstract/edwards";
import { BigIntPoint, U32ArrayPoint } from "../../reference/types";
import { FieldMath } from "../../reference/utils/FieldMath";
import { createHash } from "crypto";
import {
  gen_p_limbs,
  gen_r_limbs,
  gen_d_limbs,
  to_words_le,
  from_words_le,
  u8s_to_points,
  compute_misc_params,
  points_to_u8s_for_gpu,
  numbers_to_u8s_for_gpu,
  calc_num_words,
} from "../implementation/cuzk/utils";
import {
  create_sb,
  get_device,
  read_from_gpu,
  create_bind_group,
  create_and_write_sb,
  create_compute_pipeline,
  create_bind_group_layout,
} from "../implementation/cuzk/gpu";
import structs from "../implementation/wgsl/struct/structs.template.wgsl";
import bigint_funcs from "../implementation/wgsl/bigint/bigint.template.wgsl";
import field_funcs from "../implementation/wgsl/field/field.template.wgsl";
import ec_funcs from "../implementation/wgsl/curve/ec.template.wgsl";
import montgomery_product_funcs from "../implementation/wgsl/montgomery/mont_pro_product.template.wgsl";
import scalar_mul_shader from "./wgsl/scalar_mul.template.wgsl";
import { are_point_arr_equal } from "./utils";

const fieldMath = new FieldMath();

const ZERO_POINT = fieldMath.createPoint(
  BigInt(0),
  BigInt(1),
  BigInt(0),
  BigInt(1),
);

const p = BigInt(
  "8444461749428370424248824938781546531375899335154063827935233455917409239041",
);
const word_size = 13;

/*
 * This benchmark measures the time taken to multiply each point in an array
 * with a random 16-bit scalar.
 */
export const scalar_mul_benchmarks = async (
  baseAffinePoints: BigIntPoint[] | U32ArrayPoint[] | Buffer,
  {}: bigint[] | Uint32Array[] | Buffer,
): Promise<{ x: bigint; y: bigint }> => {
  const cost = 1024;

  const num_scalars = 1;
  const scalars: number[] = [];

  // Use a PRNG instead of system randomness so that the inputs are
  // deterministic
  const hasher = createHash("sha256");
  for (let i = 0; i < num_scalars; i++) {
    hasher.update(new Uint8Array([i]));
    const d = Array.from(hasher.digest());
    const s = new Uint16Array(d.slice(0, 2));
    const u16 = from_words_le(s, 2, 8);
    scalars.push(Number(u16));
  }

  // Convert baseAffinePoints to ETE points
  const points = (baseAffinePoints as BigIntPoint[])
    .slice(0, num_scalars)
    .map((x) => {
      return fieldMath.createPoint(
        x.x,
        x.y,
        fieldMath.Fp.mul(x.x, x.y),
        BigInt(1),
      );
    });

  // Benchmark in CPU
  const expected = cpu_benchmark(points, scalars, cost);

  // Double-and-add method in TS
  const double_and_add_cpu_results = double_and_add_cpu(points, scalars, cost);

  assert(are_point_arr_equal(double_and_add_cpu_results, expected));

  // Double-and-add method in GPU
  const start = Date.now();
  const double_and_add_gpu_results = await double_and_add_gpu(
    points,
    scalars,
    cost,
  );
  const elapsed = Date.now() - start;
  console.log(
    `double-and-add method (cost = ${cost}) in GPU (including data transfer) took ${elapsed}ms`,
  );
  assert(are_point_arr_equal(double_and_add_gpu_results, expected));

  // 2^w-ary method in CPU
  const two_w_ary_results = two_w_ary_cpu(points, scalars, cost);
  assert(are_point_arr_equal(two_w_ary_results, expected));

  // TODO: 2^w-ary method in GPU

  // TODO: Sliding window method
  //const sliding_window_results = sliding_window_cpu(points, scalars, cost)
  //assert(are_point_arr_equal(sliding_window_results, expected))

  // TODO: NAF method

  // TODO: wNAF method

  // Booth encoding method in CPU
  const booth_cpu_results = await booth_cpu(points, scalars, cost);
  assert(are_point_arr_equal(booth_cpu_results, expected));

  // Booth encoding method in GPU
  const start_booth = Date.now();
  const booth_gpu_results = await booth_gpu(points, scalars, cost);
  const elapsed_booth = Date.now() - start_booth;
  console.log(
    `booth method (cost = ${cost}) in GPU (including data transfer) took ${elapsed_booth}ms`,
  );
  assert(are_point_arr_equal(booth_gpu_results, expected));

  const booth2_cpu_results = await booth_cpu(points, scalars, cost);
  assert(are_point_arr_equal(booth2_cpu_results, expected));

  return { x: BigInt(1), y: BigInt(0) };
};

const copyPoint = (point: ExtPointType): ExtPointType => {
  return fieldMath.createPoint(
    BigInt(point.ex),
    BigInt(point.ey),
    BigInt(point.et),
    BigInt(point.ez),
  );
};

const cpu_benchmark = (
  points: ExtPointType[],
  scalars: number[],
  cost: number,
): ExtPointType[] => {
  const results: ExtPointType[] = [];

  for (let i = 0; i < scalars.length; i++) {
    let result = copyPoint(points[i]);
    for (let j = 0; j < cost; j++) {
      result = result.multiply(BigInt(scalars[i]));
    }
    results.push(result);
  }

  return results;
};

// On average, requires (n−1) doublings and n/2 additions
// where n is the number of digits of the binary decomposition of the scalar
const double_and_add = (point: ExtPointType, scalar: number): ExtPointType => {
  let result = ZERO_POINT;
  let temp = fieldMath.createPoint(point.ex, point.ey, point.et, point.ez);

  let scalar_iter = scalar;
  while (scalar_iter !== 0) {
    if ((scalar_iter & 1) === 1) {
      result = result.add(temp);
    }
    temp = temp.double();
    scalar_iter = Number(BigInt(scalar_iter) >> BigInt(1));
  }
  return result;
};

const booth_cpu = (
  points: ExtPointType[],
  scalars: number[],
  cost: number,
): ExtPointType[] => {
  const results: ExtPointType[] = [];

  for (let i = 0; i < scalars.length; i++) {
    let result = copyPoint(points[i]);
    for (let j = 0; j < cost; j++) {
      result = booth2(result, scalars[i]);
    }
    results.push(result);
  }

  return results;
};

const booth2 = (point: ExtPointType, scalar: number): ExtPointType => {
  if (scalar === 0) {
    return point;
  }

  let s = scalar;

  // Store the Booth-encoded scalar in booth.
  let booth = 0;

  // The zeroth digit has to be stored separately because booth is limited to
  // 32 bits
  const zeroth = s & 1;

  // Use 2 digits per bit. e.g. 0b1111 should become 01, 01, 01, 01
  s = s >> 1;
  let i = 30;
  while (s !== 0) {
    const digit = s & 1;
    booth += digit << i;
    s = s >> 1;
    i -= 2;
  }

  // Perform Booth encoding
  for (let i = 0; i < 15; i++) {
    let pair = (booth >> (i * 2)) & 0b1111;
    if (pair === 0b0100) {
      pair = 0b0101;
    } else if (pair === 0b0001) {
      pair = 0b0010;
    } else if (pair === 0b0101) {
      pair = 0b0100;
    }

    const left = booth >> (i * 2 + 4);
    const right = booth & ((1 << (i * 2)) - 1);
    booth = (((left << 4) + pair) << (i * 2)) + right;
  }

  let result = ZERO_POINT;
  let temp = point;

  // Encode the last digit and the 0th digit
  let im = booth >> 30;
  if (im === 0 && zeroth === 1) {
    im = 1;
  } else if (im === 1 && zeroth === 0) {
    im = 2;
  } else if (im === 1 && zeroth === 1) {
    im = 0;
  }

  // Handle the zeroth and 1st digits
  if (zeroth === 1) {
    // Treat it as 2
    result = result.add(temp.negate());
  }
  temp = temp.double();

  const mask = (1 << 30) - 1;
  booth = (booth & mask) + Number(BigInt(im) << BigInt(30));

  // Handle the rest of the digits
  for (let idx = 0; idx < 15; idx++) {
    const i = 15 - idx;
    const x = (booth >> (i * 2)) & 0b11;

    if (x === 1) {
      result = result.add(temp);
    } else if (x === 2) {
      result = result.add(temp.negate());
    }
    temp = temp.double();
  }

  return result;
};

export const booth = (point: ExtPointType, scalar: number): ExtPointType => {
  if (scalar === 0) {
    return point;
  }
  const scalar_width = BigInt(scalar).toString(2).length;
  const num_digits = 2 ** Math.ceil(Math.log2(scalar_width)) + 1;

  // Binary decomp of the scalar
  const a = Array.from(to_words_le(BigInt(scalar), num_digits, 1));
  assert(a.length === num_digits);

  // Sanity check
  const b = Array(num_digits).fill(0);
  let s = scalar;
  let i = 0;
  while (s != 0) {
    b[i] = s & 1;
    s = Number(BigInt(s) >> BigInt(1));
    i++;
  }
  assert(a.toString() === b.toString());

  for (let i = a.length - 1; i >= 1; i--) {
    if (a[i] === 0 && a[i - 1] === 1) {
      a[i] = 1;
    } else if (a[i] === 1 && a[i - 1] === 0) {
      a[i] = 2;
    } else if (a[i] === 0 && a[i - 1] === 0) {
      //a[i] = 0
    } else if (a[i] === 1 && a[i - 1] === 1) {
      a[i] = 0;
    }
  }

  if (a[0] === 1) {
    a[0] = 2;
  }

  // Find the last 1
  let max_idx = a.length - 1;
  while (a[max_idx] === 0) {
    max_idx--;
  }

  let result = ZERO_POINT;
  let temp = point;
  for (let i = 0; i < max_idx + 1; i++) {
    if (a[i] === 1) {
      result = result.add(temp);
    } else if (a[i] === 2) {
      result = result.add(temp.negate());
    }
    temp = temp.double();
  }

  return result;
};

//const sliding_window_cpu = (
//points: ExtPointType[],
//scalars: number[],
//cost: number,
//w = 3,
//): ExtPointType[] => {
//const results: ExtPointType[] = []

//for (let i = 0; i < scalars.length; i ++) {
//let result = copyPoint(points[i])
//for (let j = 0; j < cost; j ++) {
//result = sliding_window(result, scalars[i], w)
//}
//results.push(result)
//}

//return results
//}

//const sliding_window = (
//point: ExtPointType,
//scalar: number,
//w: number,
//): ExtPointType => {
//const scalar_width = BigInt(scalar).toString(2).length
//const base = 2 ** w
//const num_digits = calc_num_words(w, scalar_width)
//const digits = to_words_le(BigInt(scalar), num_digits, w)
//const precomputed = [
//copyPoint(point),
//copyPoint(point).multiply(BigInt(2)),
//]
//for (let i = 1; i < (2 ** w) + 1; i ++) {
//precomputed.push(ZERO_POINT)
//}

//for (let i = 1; i < 2 ** (w - 1); i ++) {
//const pt = precomputed[2 * i - 1].add(precomputed[1])
//precomputed[2 * i - 1] = pt
//}

//let result = ZERO_POINT
//let i = num_digits - 1

//while (i >= 0) {
//let j
//if (digits[i] === 0) {
//j = i
//while (digits[j] === 0) {
//j --
//}
//result = result.multiply(BigInt(2 ** (i - j + 1)))
//} else {
//j = 0
//while (digits[j] !== 1) {
//j ++
//}
//result = result.multiply(BigInt(2 ** (i - j + 1)))
//// TODO: result = result.add(P_a[i:j])?????????
//}

//i = j - 1
//}

//return result
//}

const two_w_ary_cpu = (
  points: ExtPointType[],
  scalars: number[],
  cost: number,
  w = 3,
): ExtPointType[] => {
  const results: ExtPointType[] = [];

  for (let i = 0; i < scalars.length; i++) {
    let result = copyPoint(points[i]);
    for (let j = 0; j < cost; j++) {
      result = two_w_ary(result, scalars[i], w);
    }
    results.push(result);
  }

  return results;
};

// On average, requires (l - 1) doublings and
// 2^w - 2 + ((2^w -1) / 2^w) * l additions
// TODO: calculate the optimal w on a spreadsheet
const two_w_ary = (
  point: ExtPointType,
  scalar: number,
  w: number,
): ExtPointType => {
  const scalar_width = BigInt(scalar).toString(2).length;
  const base = 2 ** w;
  const num_digits = calc_num_words(w, scalar_width);
  const digits = to_words_le(BigInt(scalar), num_digits, w);
  const precomputed = [copyPoint(point)];

  for (let i = 1; i < base; i++) {
    const pt_prev = precomputed[i - 1];
    const pt = point;
    precomputed.push(pt_prev.add(pt));
  }

  let result = ZERO_POINT;
  for (let i = num_digits - 1; i >= 0; i--) {
    result = result.multiply(BigInt(base));
    if (digits[i] !== 0) {
      result = result.add(precomputed[digits[i] - 1]);
    }
  }
  return result;
};

const double_and_add_cpu = (
  points: ExtPointType[],
  scalars: number[],
  cost: number,
): ExtPointType[] => {
  const results: ExtPointType[] = [];

  for (let i = 0; i < scalars.length; i++) {
    let result = copyPoint(points[i]);
    for (let j = 0; j < cost; j++) {
      result = double_and_add(result, scalars[i]);
    }
    results.push(result);
  }

  return results;
};

const booth_gpu = async (
  points: ExtPointType[],
  scalars: number[],
  cost: number,
): Promise<ExtPointType[]> => {
  return run_in_gpu(points, scalars, cost, "booth_benchmark");
};

const double_and_add_gpu = async (
  points: ExtPointType[],
  scalars: number[],
  cost: number,
): Promise<ExtPointType[]> => {
  return run_in_gpu(points, scalars, cost, "double_and_add_benchmark");
};

const run_in_gpu = async (
  points: ExtPointType[],
  scalars: number[],
  cost: number,
  entrypoint: string,
): Promise<ExtPointType[]> => {
  const params = compute_misc_params(p, word_size);
  const n0 = params.n0;
  const num_words = params.num_words;
  const r = params.r;
  const rinv = params.rinv;
  const d = params.edwards_d;

  const input_size = scalars.length;
  const workgroup_size = 1;
  const num_x_workgroups = 256;
  const num_y_workgroups = Math.ceil(
    input_size / workgroup_size / num_x_workgroups,
  );

  const mont_points: BigIntPoint[] = [];
  for (const pt of points) {
    mont_points.push({
      x: fieldMath.Fp.mul(pt.ex, r),
      y: fieldMath.Fp.mul(pt.ey, r),
      t: fieldMath.Fp.mul(pt.et, r),
      z: fieldMath.Fp.mul(pt.ez, r),
    });
  }

  // Convert points to Montgomery form
  const points_bytes = points_to_u8s_for_gpu(mont_points, num_words, word_size);
  const scalars_bytes = numbers_to_u8s_for_gpu(scalars);

  const device = await get_device();
  const commandEncoder = device.createCommandEncoder();

  const scalars_sb = create_and_write_sb(device, scalars_bytes);
  const points_sb = create_and_write_sb(device, points_bytes);
  const results_sb = create_sb(device, points_sb.size);

  const bindGroupLayout = create_bind_group_layout(device, [
    "read-only-storage",
    "read-only-storage",
    "storage",
  ]);
  const bindGroup = create_bind_group(device, bindGroupLayout, [
    points_sb,
    scalars_sb,
    results_sb,
  ]);

  const p_limbs = gen_p_limbs(p, num_words, word_size);
  const r_limbs = gen_r_limbs(r, num_words, word_size);
  const d_limbs = gen_d_limbs(d, num_words, word_size);
  const shaderCode = mustache.render(
    scalar_mul_shader,
    {
      word_size,
      num_words,
      n0,
      p_limbs,
      r_limbs,
      d_limbs,
      mask: BigInt(2) ** BigInt(word_size) - BigInt(1),
      two_pow_word_size: BigInt(2) ** BigInt(word_size),
      workgroup_size,
      num_y_workgroups,
      cost,
      num_words_plus_one: num_words + 1,
      num_words_mul_two: num_words * 2,
    },
    {
      structs,
      bigint_funcs,
      field_funcs,
      ec_funcs,
      montgomery_product_funcs,
    },
  );
  const computePipeline = await create_compute_pipeline(
    device,
    [bindGroupLayout],
    shaderCode,
    entrypoint,
  );

  const passEncoder = commandEncoder.beginComputePass();
  passEncoder.setPipeline(computePipeline);
  passEncoder.setBindGroup(0, bindGroup);
  passEncoder.dispatchWorkgroups(num_x_workgroups, num_y_workgroups, 1);
  passEncoder.end();

  const results_data = await read_from_gpu(device, commandEncoder, [
    results_sb,
  ]);

  const data = u8s_to_points(results_data[0], num_words, word_size);
  const results_points = [];
  for (const p of data) {
    const pt = fieldMath.createPoint(
      fieldMath.Fp.mul(p.x, rinv),
      fieldMath.Fp.mul(p.y, rinv),
      fieldMath.Fp.mul(p.t, rinv),
      fieldMath.Fp.mul(p.z, rinv),
    );
    results_points.push(pt);
  }

  device.destroy();

  return results_points;
};
