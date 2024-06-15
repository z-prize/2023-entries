import assert from "assert";
import mustache from "mustache";
import { BigIntPoint, U32ArrayPoint } from "../../reference/types";
import { FieldMath } from "../../reference/utils/FieldMath";
import {
  get_device,
  create_and_write_sb,
  create_bind_group,
  create_bind_group_layout,
  create_compute_pipeline,
  read_from_gpu,
  execute_pipeline,
} from "../implementation/cuzk/gpu";
import structs from "../implementation/wgsl/struct/structs.template.wgsl";
import bigint_funcs from "../implementation/wgsl/bigint/bigint.template.wgsl";
import field_funcs from "../implementation/wgsl/field/field.template.wgsl";
import ec_funcs from "../implementation/wgsl/curve/ec.template.wgsl";
import montgomery_product_funcs from "../implementation/wgsl/montgomery/mont_pro_product.template.wgsl";
import horners_rule_shader from "./wgsl/horners_rule.template.wgsl";
import {
  compute_misc_params,
  u8s_to_bigints,
  gen_p_limbs,
  gen_r_limbs,
  gen_d_limbs,
  bigints_to_u8_for_gpu,
} from "../implementation/cuzk/utils";
import { are_point_arr_equal } from "./utils";

export const horners_rule_benchmark = async (
  baseAffinePoints: BigIntPoint[] | U32ArrayPoint[] | Buffer,
  {}: bigint[] | Uint32Array[] | Buffer,
): Promise<{ x: bigint; y: bigint }> => {
  const fieldMath = new FieldMath();
  const p = BigInt(
    "0x12ab655e9a2ca55660b44d1e5c37b00159aa76fed00000010a11800000000001",
  );
  const word_size = 13;
  const chunk_size = 16;
  const params = compute_misc_params(p, word_size);
  const n0 = params.n0;
  const num_words = params.num_words;
  const r = params.r;
  const rinv = params.rinv;
  const d = params.edwards_d;
  const p_limbs = gen_p_limbs(p, num_words, word_size);
  const r_limbs = gen_r_limbs(r, num_words, word_size);
  const d_limbs = gen_d_limbs(d, num_words, word_size);

  const num_subtasks = 16;
  assert(baseAffinePoints.length >= num_subtasks);

  // Take the first num_subtasks points
  const x_y_coords: bigint[] = [];
  const t_z_coords: bigint[] = [];
  for (const pt of (baseAffinePoints as BigIntPoint[]).slice(0, num_subtasks)) {
    x_y_coords.push(fieldMath.Fp.mul(pt.x, r));
    x_y_coords.push(fieldMath.Fp.mul(pt.y, r));
    t_z_coords.push(fieldMath.Fp.mul(pt.t, r));
    t_z_coords.push(fieldMath.Fp.mul(pt.z, r));
  }

  const points = (baseAffinePoints as BigIntPoint[])
    .slice(0, num_subtasks)
    .map((x) => fieldMath.createPoint(x.x, x.y, x.t, x.z));

  const start_cpu = Date.now();
  let expected = points[0];
  const m = BigInt(2) ** BigInt(chunk_size);
  for (let i = 1; i < num_subtasks; i++) {
    expected = expected.multiply(m);
    expected = expected.add(points[i]);
  }
  const elapsed_cpu = Date.now() - start_cpu;
  console.log(
    `CPU took ${elapsed_cpu}ms to compute Horner's rule for ${num_subtasks} points`,
  );

  const device = await get_device();
  const commandEncoder = device.createCommandEncoder();

  const x_y_coords_bytes = bigints_to_u8_for_gpu(x_y_coords);
  const t_z_coords_bytes = bigints_to_u8_for_gpu(t_z_coords);

  const x_y_coords_sb = create_and_write_sb(device, x_y_coords_bytes);
  const t_z_coords_sb = create_and_write_sb(device, t_z_coords_bytes);

  const shaderCode = mustache.render(
    horners_rule_shader,
    {
      word_size,
      num_words,
      n0,
      p_limbs,
      r_limbs,
      d_limbs,
      mask: BigInt(2) ** BigInt(word_size) - BigInt(1),
      two_pow_word_size: BigInt(2) ** BigInt(word_size),
      chunk_size,
    },
    {
      structs,
      bigint_funcs,
      field_funcs,
      ec_funcs,
      montgomery_product_funcs,
    },
  );

  const bindGroupLayout = create_bind_group_layout(device, [
    "storage",
    "storage",
  ]);
  const bindGroup = create_bind_group(device, bindGroupLayout, [
    x_y_coords_sb,
    t_z_coords_sb,
  ]);

  const start_gpu = Date.now();
  const computePipeline = await create_compute_pipeline(
    device,
    [bindGroupLayout],
    shaderCode,
    "main",
  );

  // Horner's rule has to be run serially because each step depends on the
  // result of the previous step
  const num_x_workgroups = 1;
  const num_y_workgroups = 1;

  execute_pipeline(
    commandEncoder,
    computePipeline,
    bindGroup,
    num_x_workgroups,
    num_y_workgroups,
    1,
  );
  const elapsed_gpu = Date.now() - start_gpu;

  const data = await read_from_gpu(device, commandEncoder, [
    x_y_coords_sb,
    t_z_coords_sb,
  ]);

  const x_y_mont_coords_result = u8s_to_bigints(data[0], num_words, word_size);
  const t_z_mont_coords_result = u8s_to_bigints(data[1], num_words, word_size);
  console.log(
    `GPU took ${elapsed_gpu}ms to compute Horner's rule for ${num_subtasks} points (including compilation and data transfer)`,
  );

  const result = fieldMath.createPoint(
    fieldMath.Fp.mul(x_y_mont_coords_result[0], rinv),
    fieldMath.Fp.mul(x_y_mont_coords_result[1], rinv),
    fieldMath.Fp.mul(t_z_mont_coords_result[0], rinv),
    fieldMath.Fp.mul(t_z_mont_coords_result[1], rinv),
  );

  //console.log('result.isAffine():', result.toAffine())
  //console.log('expected.isAffine():', expected.toAffine())
  assert(are_point_arr_equal([result], [expected]));

  device.destroy();

  return { x: BigInt(0), y: BigInt(0) };
};
