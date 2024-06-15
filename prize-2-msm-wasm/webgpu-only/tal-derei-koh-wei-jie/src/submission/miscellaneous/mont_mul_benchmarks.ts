import mustache from "mustache";
import { BigIntPoint, U32ArrayPoint } from "../../reference/types";
import {
  compute_misc_params,
  gen_p_limbs,
  bigints_to_u8_for_gpu_old,
  u8s_to_bigints,
} from "../implementation/cuzk/utils";
import {
  get_device,
  create_and_write_sb,
  create_bind_group,
  create_bind_group_layout,
  create_compute_pipeline,
  execute_pipeline,
  read_from_gpu,
} from "../implementation/cuzk/gpu";
import structs from "../implementation/wgsl/struct/structs.template.wgsl";
import bigint_functions from "../implementation/wgsl/bigint/bigint.template.wgsl";
import field_functions from "../implementation/wgsl/field/field.template.wgsl";
import mont_pro_optimised_shader from "../implementation/wgsl/montgomery/mont_pro_optimized.template.wgsl";
import mont_pro_modified_shader from "../implementation/wgsl/montgomery/mont_pro_modified.template.wgsl";
import mont_pro_cios_shader from "../implementation/wgsl/montgomery/mont_pro_cios.template.wgsl";
import montgomery_product_funcs from "../implementation/wgsl/montgomery/mont_pro_product.template.wgsl";
import { genRandomFieldElement } from "./utils";

export const mont_mul_benchmarks = async (
  {}: BigIntPoint[] | U32ArrayPoint[] | Buffer,
  {}: bigint[] | Uint32Array[] | Buffer,
): Promise<{ x: bigint; y: bigint }> => {
  const cost = 2 ** 17;
  const num_runs = 10;

  // Define and generate params
  const num_inputs = 1;
  const num_x_workgroups = 1;

  const p = BigInt(
    "0x12ab655e9a2ca55660b44d1e5c37b00159aa76fed00000010a11800000000001",
  );

  const expensive_computation = (
    a: bigint,
    b: bigint,
    r: bigint,
    cost: number,
  ): bigint => {
    const c = a ** BigInt(cost);
    return (c * b * r) % p;
  };

  const timings: any = {};

  for (let word_size = 13; word_size < 14; word_size++) {
    timings[word_size] = [];

    const misc_params = compute_misc_params(p, word_size);
    const num_words = misc_params.num_words;
    const n0 = misc_params.n0;
    const mask = BigInt(2) ** BigInt(word_size) - BigInt(1);
    const r = misc_params.r;
    const two_pow_word_size = 2 ** word_size;

    console.log(
      `Limb size: ${word_size}, Number of limbs: ${num_words}, ` +
        `N: ${word_size * num_words}, ` +
        `Max terms: ${misc_params.max_terms}, k: ${misc_params.k}, ` +
        `nSafe: ${misc_params.nsafe}`,
    );
    const p_limbs = gen_p_limbs(p, num_words, word_size);

    let shaderCode = "";

    if (word_size > 11 && word_size < 14) {
      console.log(
        `Performing ${num_inputs} (a ^ ${cost} * b * r) (using MontProOptimised) with ${word_size}-bit limbs over ${num_runs} runs on ${num_x_workgroups} workgroups`,
      );

      shaderCode = mustache.render(
        mont_pro_optimised_shader,
        {
          num_words,
          word_size,
          n0,
          mask,
          two_pow_word_size,
          cost,
          p_limbs,
        },
        {
          structs,
          bigint_functions,
          field_functions,
          montgomery_product_funcs,
        },
      );
      //console.log(shaderCode)
    } else if (word_size > 13 && word_size < 16) {
      console.log(
        `Performing ${num_inputs} (a ^ ${cost} * b * r) (using MontProModified) with ${word_size}-bit limbs over ${num_runs} runs on ${num_x_workgroups} workgroups`,
      );
      shaderCode = mustache.render(
        mont_pro_modified_shader,
        {
          num_words,
          word_size,
          n0,
          mask,
          two_pow_word_size,
          cost,
          p_limbs,
          nsafe: misc_params.nsafe,
        },
        {
          structs,
          bigint_functions,
        },
      );
    } else if (word_size === 16) {
      console.log(
        `Performing ${num_inputs} (a ^ ${cost} * b * r) (using MontProCios) with ${word_size}-bit limbs over ${num_runs} runs on ${num_x_workgroups} workgroups`,
      );
      shaderCode = mustache.render(
        mont_pro_cios_shader,
        {
          num_words,
          num_words_plus_one: num_words + 1,
          num_words_plus_two: num_words + 2,
          word_size,
          n0,
          mask,
          two_pow_word_size,
          cost,
          p_limbs,
          nsafe: misc_params.nsafe,
        },
        {
          structs,
        },
      );
    }

    for (let run = 0; run < num_runs; run++) {
      const expected: bigint[] = [];
      const inputs: bigint[] = [];

      // Generate random inputs
      for (let i = 0; i < num_inputs; i++) {
        const a = genRandomFieldElement(p);
        const b = genRandomFieldElement(p);
        const ar = (a * r) % p;
        const br = (b * r) % p;

        inputs.push(ar);
        inputs.push(br);

        expected.push(expensive_computation(a, b, r, cost));
      }

      const input_bytes = bigints_to_u8_for_gpu_old(
        inputs,
        num_words,
        word_size,
      );

      const device = await get_device();

      const a_storage_buffer = create_and_write_sb(device, input_bytes);

      const stagingBuffer = device.createBuffer({
        size: input_bytes.length,
        usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST,
      });

      const bindGroupLayout = create_bind_group_layout(device, ["storage"]);
      const bindGroup = create_bind_group(device, bindGroupLayout, [
        a_storage_buffer,
      ]);

      const computePipeline = await create_compute_pipeline(
        device,
        [bindGroupLayout],
        shaderCode,
        "main",
      );

      const commandEncoder = device.createCommandEncoder();

      const start = Date.now();

      await execute_pipeline(
        commandEncoder,
        computePipeline,
        bindGroup,
        num_x_workgroups,
      );

      commandEncoder.copyBufferToBuffer(
        a_storage_buffer,
        0, // Source offset
        stagingBuffer,
        0, // Destination offset
        input_bytes.length,
      );

      //device.queue.submit([commandEncoder.finish()]);
      //await device.queue.onSubmittedWorkDone()

      const data = await read_from_gpu(device, commandEncoder, [
        a_storage_buffer,
      ]);
      const elapsed = Date.now() - start;

      const results = u8s_to_bigints(data[0], num_words, word_size);

      timings[word_size].push(elapsed);

      for (let i = 0; i < num_inputs; i++) {
        if (results[i] !== expected[i]) {
          console.error(`Result mismatch at ${i}`);
          break;
        }
      }

      device.destroy();
    }

    if (num_runs < 2) {
      console.log(
        `Limb size: ${word_size}. Time taken for 1 run: ${timings[word_size][0]}ms`,
      );
    } else {
      const sum = timings[word_size].slice(1).reduce((a: number, b: number) => {
        return a + b;
      }, 0);
      const avg = Math.floor(sum / num_runs);
      console.log(`Limb size: ${word_size}. Average time taken: ${avg}ms`);
    }
  }

  return { x: BigInt(0), y: BigInt(0) };
};
