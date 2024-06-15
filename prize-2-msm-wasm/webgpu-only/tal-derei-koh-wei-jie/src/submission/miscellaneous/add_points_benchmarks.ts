import mustache from "mustache";
import assert from "assert";
import { BigIntPoint, U32ArrayPoint } from "../../reference/types";
import { ExtPointType } from "@noble/curves/abstract/edwards";
import { FieldMath } from "../../reference/utils/FieldMath";
import {
  compute_misc_params,
  u8s_to_points,
  points_to_u8s_for_gpu,
  gen_p_limbs,
  gen_d_limbs,
} from "../implementation/cuzk/utils";
import { add_points_any_a } from "./add_points";
import {
  GPUExecution,
  IEntryInfo,
  IGPUInput,
  IGPUResult,
  IShaderCode,
  multipassEntryCreator,
} from "./entries/multipassEntryCreator";

import structs from "../implementation/wgsl/struct/structs.template.wgsl";
import bigint_funcs from "../implementation/wgsl/bigint/bigint.template.wgsl";
import field_funcs from "../implementation/wgsl/field/field.template.wgsl";
import ec_funcs from "../implementation/wgsl/curve/ec.template.wgsl";
import add_points_benchmark_shader from "./wgsl/add_points_benchmark.template.wgsl";
import montgomery_product_funcs from "../implementation/wgsl/montgomery/mont_pro_product.template.wgsl";
import { genRandomFieldElement } from "./utils";

const setup_shader_code = (
  shader: string,
  p: bigint,
  d: bigint,
  num_words: number,
  word_size: number,
  n0: bigint,
  cost: number,
) => {
  const p_limbs = gen_p_limbs(p, num_words, word_size);
  const d_limbs = gen_d_limbs(d, num_words, word_size);
  const shaderCode = mustache.render(
    shader,
    {
      word_size,
      num_words,
      n0,
      cost,
      p_limbs,
      d_limbs,
      mask: BigInt(2) ** BigInt(word_size) - BigInt(1),
      two_pow_word_size: BigInt(2) ** BigInt(word_size),
    },
    {
      structs,
      bigint_funcs,
      field_funcs,
      ec_funcs,
      montgomery_product_funcs,
    },
  );
  //console.log(shaderCode)
  return shaderCode;
};

const expensive_computation = (
  a: ExtPointType,
  b: ExtPointType,
  cost: number,
  fieldMath: FieldMath,
  add_points_func: any,
): ExtPointType => {
  let c = add_points_func(a, b, fieldMath);
  for (let i = 1; i < cost; i++) {
    c = add_points_func(c, a, fieldMath);
  }

  return c;
};
export const add_points_benchmarks = async (
  {}: BigIntPoint[] | U32ArrayPoint[] | Buffer,
  {}: bigint[] | Uint32Array[] | Buffer,
): Promise<{ x: bigint; y: bigint }> => {
  const cost = 2 ** 12;

  const fieldMath = new FieldMath();
  fieldMath.aleoD = BigInt(-1);
  const p = BigInt(
    "0x12ab655e9a2ca55660b44d1e5c37b00159aa76fed00000010a11800000000001",
  );

  // Ignore the input points and scalars. Generate two random Extended
  // Twisted Edwards points, perform many additions, and print the time taken.
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
  // Note that pt is not the generator of the group. it's just a valid curve point.
  const pt = fieldMath.createPoint(x, y, t, z);

  // Generate two random points by multiplying by random field elements
  const a = pt.multiply(genRandomFieldElement(p));
  const b = pt.multiply(genRandomFieldElement(p));

  // Uncomment to test a == b
  //const a = pt
  //const b = pt

  const num_x_workgroups = 1;
  const word_size = 13;

  const num_runs = 5;

  console.log("Cost:", cost);
  const print_avg_timings = (timings: any[]) => {
    if (timings.length === 1) {
      console.log(`CPU took ${timings[0].elapsed_cpu}`);
      console.log(`GPU took ${timings[0].elapsed_gpu}`);
      return {
        avg_cpu: timings[0].elapsed_cpu,
        avg_gpu: timings[0].elapsed_gpu,
      };
    } else {
      let total_gpu = 0;
      let total_cpu = 0;
      for (let i = 1; i < timings.length; i++) {
        total_cpu += timings[i].elapsed_cpu;
        total_gpu += timings[i].elapsed_gpu;
      }
      const avg_gpu = total_gpu / (timings.length - 1);
      const avg_cpu = total_cpu / (timings.length - 1);
      console.log(
        `Average timing for CPU over ${num_runs} runs (ignoring the first): ${avg_cpu}ms`,
      );
      console.log(
        `Average timing for GPU over ${num_runs} runs (ignoring the first): ${avg_gpu}ms`,
      );

      return { avg_cpu, avg_gpu };
    }
    console.log(timings);
  };

  console.log("Benchmarking add_points for add-2008-hwcd");
  const timings_any_a: any[] = [];
  for (let i = 0; i < num_runs; i++) {
    const r = await do_benchmark(
      add_points_benchmark_shader,
      a,
      b,
      p,
      cost,
      num_x_workgroups,
      word_size,
      fieldMath,
      add_points_any_a,
    );
    timings_any_a.push(r);
  }

  print_avg_timings(timings_any_a);

  return { x: BigInt(1), y: BigInt(0) };
};

const do_benchmark = async (
  shader: string,
  a: ExtPointType,
  b: ExtPointType,
  p: bigint,
  cost: number,
  num_x_workgroups: number,
  word_size: number,
  fieldMath: FieldMath,
  add_points_func: any,
): Promise<{ elapsed_cpu: number; elapsed_gpu: number }> => {
  // Compute in CPU
  // Start timer
  const start_cpu = Date.now();
  const expected_cpu = expensive_computation(
    a,
    b,
    cost,
    fieldMath,
    add_points_func,
  );
  const expected_cpu_affine = expected_cpu.toAffine();
  // End Timer
  const elapsed_cpu = Date.now() - start_cpu;

  const f = (a: ExtPointType, b: ExtPointType) => {
    return a.add(b);
  };
  const expected_cpu_noble_affine = expensive_computation(
    a,
    b,
    cost,
    fieldMath,
    f,
  ).toAffine();

  assert(expected_cpu_affine.x === expected_cpu_noble_affine.x);
  assert(expected_cpu_affine.y === expected_cpu_noble_affine.y);
  //console.log(`CPU took ${elapsed_cpu}ms`)

  const params = compute_misc_params(p, word_size);
  const n0 = params.n0;
  const num_words = params.num_words;
  const r = params.r;
  const rinv = params.rinv;
  const d = params.edwards_d;

  // Convert to Mont form
  const points_with_mont_coords: BigIntPoint[] = [];
  for (const pt of [a, b]) {
    points_with_mont_coords.push({
      x: fieldMath.Fp.mul(pt.ex, r),
      y: fieldMath.Fp.mul(pt.ey, r),
      t: fieldMath.Fp.mul(pt.et, r),
      z: fieldMath.Fp.mul(pt.ez, r),
    });
  }

  const points_bytes = points_to_u8s_for_gpu(
    points_with_mont_coords,
    num_words,
    word_size,
  );
  const shaderCode = setup_shader_code(
    shader,
    p,
    d,
    num_words,
    word_size,
    n0,
    cost,
  );

  const executionSteps: GPUExecution[] = [];
  const addPointsShader: IShaderCode = {
    code: shaderCode,
    entryPoint: "main",
  };
  const addPointsInputs: IGPUInput = {
    inputBufferTypes: ["storage"],
    inputBufferSizes: [points_bytes.length],
    inputBufferUsages: [
      GPUBufferUsage.STORAGE |
        GPUBufferUsage.COPY_DST |
        GPUBufferUsage.COPY_SRC,
    ],
    mappedInputs: new Map<number, Uint8Array>([[0, points_bytes]]),
  };

  // Note that the framework currently requires a separate output buffer
  const addPointsResult: IGPUResult = {
    resultBufferTypes: ["storage"],
    resultBufferSizes: [points_bytes.length],
    resultBufferUsages: [GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC],
  };
  const addPointsStep = new GPUExecution(
    addPointsShader,
    addPointsInputs,
    addPointsResult,
  );
  executionSteps.push(addPointsStep);

  const entryInfo: IEntryInfo = {
    numInputs: 1,
    outputSize: points_bytes.length,
  };

  const start_gpu = Date.now();
  const data = await multipassEntryCreator(
    executionSteps,
    entryInfo,
    num_x_workgroups,
  );
  const elapsed_gpu = Date.now() - start_gpu;

  const data_as_uint8s = new Uint8Array(data);

  const bigIntPointToExtPointType = (bip: BigIntPoint): ExtPointType => {
    return fieldMath.createPoint(bip.x, bip.y, bip.t, bip.z);
  };

  const output_points = u8s_to_points(data_as_uint8s, num_words, word_size);

  // convert output_points out of Montgomery coords
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

  // convert output_points_non_mont into affine
  const output_points_non_mont_and_affine = output_points_non_mont.map((x) =>
    x.toAffine(),
  );
  //console.log('result from gpu, in affine:', output_points_non_mont_and_affine[0])

  //console.log(expected_cpu.toAffine())

  assert(
    output_points_non_mont_and_affine[0].x === expected_cpu_affine.x,
    "point mismatch",
  );
  assert(
    output_points_non_mont_and_affine[0].y === expected_cpu_affine.y,
    "point mismatch",
  );

  return { elapsed_cpu, elapsed_gpu };
};
