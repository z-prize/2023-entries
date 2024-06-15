import mustache from "mustache";
import { calc_m, barrett_domb_mul } from "./barrett_domb";
import {
  compute_misc_params,
  from_words_le,
  to_words_le,
  gen_p_limbs,
  gen_r_limbs,
  gen_barrett_domb_m_limbs,
  bigints_to_u8_for_gpu,
} from "../implementation/cuzk/utils";
import { get_device } from "../implementation/cuzk/gpu";
import structs from "../implementation/wgsl/struct/structs.template.wgsl";
import bigint_functions from "../implementation/wgsl/bigint/bigint.template.wgsl";
import field_functions from "../implementation/wgsl/field/field.template.wgsl";
import barrett_functions from "./wgsl/barrett_domb.template.wgsl";
import barrett_benchmarks_shader from "./wgsl/barrett_benchmarks_shader.template.wgsl";
import montgomery_product_funcs from "../implementation/wgsl/montgomery/mont_pro_product.template.wgsl";
import { genRandomFieldElement } from "./utils";

export const barrett_domb_mul_benchmarks = async (): Promise<{
  x: bigint;
  y: bigint;
}> => {
  // Define and generate params
  const num_inputs = 1;
  const num_x_workgroups = 1;
  const cost = 8192;

  const p = BigInt(
    "0x12ab655e9a2ca55660b44d1e5c37b00159aa76fed00000010a11800000000001",
  );

  const expensive_computation = (
    a: bigint,
    b: bigint,
    cost: number,
  ): bigint => {
    const c = a ** BigInt(cost);
    return (c * b) % p;
  };

  const num_runs = 1;
  const word_size = 13;
  const misc_params = compute_misc_params(p, word_size);
  const num_words = misc_params.num_words;
  const n0 = misc_params.n0;
  const mask = BigInt(2) ** BigInt(word_size) - BigInt(1);
  const r = misc_params.r;
  const two_pow_word_size = 2 ** word_size;
  const p_limbs = gen_p_limbs(p, num_words, word_size);
  const r_limbs = gen_r_limbs(r, num_words, word_size);
  const m_limbs = gen_barrett_domb_m_limbs(
    misc_params.barrett_domb_m,
    num_words,
    word_size,
  );
  const p_bitlength = p.toString(2).length;
  const slack = num_words * word_size - p_bitlength;

  console.log(
    `Performing ${num_inputs} (a ^ ${cost} * b) (using Barrett reduction) over ${num_runs} runs on ${num_x_workgroups} workgroups`,
  );

  const shaderCode = mustache.render(
    barrett_benchmarks_shader,
    {
      num_words,
      word_size,
      n0,
      mask,
      two_pow_word_size,
      cost,
      p_limbs,
      r_limbs,
      m_limbs,
      w_mask: (1 << word_size) - 1,
      slack,
      num_words_mul_two: num_words * 2,
      num_words_plus_one: num_words + 1,
      z: word_size * num_words - p_bitlength,
    },
    {
      structs,
      bigint_functions,
      field_functions,
      barrett_functions,
      montgomery_product_funcs,
    },
  );

  const expected: bigint[] = [];
  const expected_words: Uint16Array[] = [];
  const inputs: bigint[] = [];

  const p_words = to_words_le(p, num_words, word_size);
  // Generate random inputs
  for (let i = 0; i < num_inputs; i++) {
    const a = genRandomFieldElement(p);
    const b = genRandomFieldElement(p);
    //const a = BigInt('8341461749428370124248824931781546531375899335154063827935293455917439005792')
    //const b = BigInt('3444221749422370424228824238781524531374499335144063827945233455917409005733')

    inputs.push(a);
    inputs.push(b);

    const a_words = to_words_le(a, num_words, word_size);
    const b_words = to_words_le(b, num_words, word_size);

    const m = calc_m(p, word_size);
    const m_words = to_words_le(m, num_words, word_size);

    const result = barrett_domb_mul(
      a_words,
      b_words,
      p_words,
      p.toString(2).length,
      m_words,
      num_words,
      word_size,
    );
    expected_words.push(result);

    //expected.push(a * b % p)
    expected.push(expensive_computation(a, b, cost));
  }

  const input_bytes = bigints_to_u8_for_gpu(inputs);

  const device = await get_device();

  // 2: Create a shader module from the shader template literal
  const shaderModule = device.createShaderModule({
    code: shaderCode,
  });

  const a_storage_buffer = device.createBuffer({
    size: input_bytes.length,
    usage:
      GPUBufferUsage.STORAGE |
      GPUBufferUsage.COPY_SRC |
      GPUBufferUsage.COPY_DST,
  });
  device.queue.writeBuffer(a_storage_buffer, 0, input_bytes);

  const stagingBuffer = device.createBuffer({
    size: input_bytes.length,
    usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST,
  });

  const bindGroupLayout = device.createBindGroupLayout({
    entries: [
      {
        binding: 0,
        visibility: GPUShaderStage.COMPUTE,
        buffer: {
          type: "storage",
        },
      },
    ],
  });

  const bindGroup = device.createBindGroup({
    layout: bindGroupLayout,
    entries: [
      {
        binding: 0,
        resource: {
          buffer: a_storage_buffer,
        },
      },
    ],
  });

  const computePipeline = device.createComputePipeline({
    layout: device.createPipelineLayout({
      bindGroupLayouts: [bindGroupLayout],
    }),
    compute: {
      module: shaderModule,
      entryPoint: "main",
    },
  });

  // 5: Create GPUCommandEncoder to issue commands to the GPU
  const commandEncoder = device.createCommandEncoder();

  const start = Date.now();

  // 6: Initiate render pass
  const passEncoder = commandEncoder.beginComputePass();

  // 7: Issue commands
  passEncoder.setPipeline(computePipeline);
  passEncoder.setBindGroup(0, bindGroup);
  passEncoder.dispatchWorkgroups(num_x_workgroups);

  // End the render pass
  passEncoder.end();

  commandEncoder.copyBufferToBuffer(
    a_storage_buffer,
    0, // Source offset
    stagingBuffer,
    0, // Destination offset
    input_bytes.length,
  );

  // 8: End frame by passing array of command buffers to command queue for execution
  device.queue.submit([commandEncoder.finish()]);

  // map staging buffer to read results back to JS
  await stagingBuffer.mapAsync(
    GPUMapMode.READ,
    0, // Offset
    input_bytes.length,
  );

  const copyArrayBuffer = stagingBuffer.getMappedRange(0, input_bytes.length);
  const data = copyArrayBuffer.slice(0);
  stagingBuffer.unmap();

  const elapsed = Date.now() - start;
  console.log(`GPU took ${elapsed}ms`);

  const dataBuf = new Uint32Array(data);

  const results: bigint[] = [];
  for (let i = 0; i < num_inputs; i++) {
    const r: number[] = [];
    for (let j = 0; j < num_words; j++) {
      r.push(dataBuf[i * num_words + j]);
    }
    results.push(from_words_le(new Uint16Array(r), num_words, word_size));
  }

  for (let i = 0; i < num_inputs; i++) {
    if (results[i] !== expected[i]) {
      console.error(`Result mismatch at ${i}`);
      break;
    }
  }

  device.destroy();

  return { x: BigInt(0), y: BigInt(0) };
};
