import {
  get_device,
  create_and_write_sb,
  create_bind_group,
  create_bind_group_layout,
  create_compute_pipeline,
  read_from_gpu,
} from "../implementation/cuzk/gpu";
import { numbers_to_u8s_for_gpu } from "../implementation/cuzk/utils";
import simple_shader from "./wgsl/simple.wgsl";
import complex_shader from "./wgsl/complex.wgsl";

/*
 * Benchmark data transfer costs
 */
export const data_transfer_cost_benchmarks = async (): Promise<{
  x: bigint;
  y: bigint;
}> => {
  let num_bytes = 1 * 1024 * 1024;

  console.log("Simple shader benchmarks:");
  await shader_benchmark(simple_shader, num_bytes, true);
  await shader_benchmark(simple_shader, num_bytes, false);

  console.log("Complex shader benchmarks:");
  await shader_benchmark(complex_shader, num_bytes, true);
  await shader_benchmark(complex_shader, num_bytes, false);

  num_bytes = 32 * 1024 * 1024;

  console.log("Simple shader benchmarks:");
  await shader_benchmark(simple_shader, num_bytes, true);
  await shader_benchmark(simple_shader, num_bytes, false);

  console.log("Complex shader benchmarks:");
  await shader_benchmark(complex_shader, num_bytes, true);
  await shader_benchmark(complex_shader, num_bytes, false);

  return { x: BigInt(1), y: BigInt(0) };
};

const shader_benchmark = async (
  shaderCode: string,
  num_bytes: number,
  read: boolean,
) => {
  const device = await get_device();
  const commandEncoder = device.createCommandEncoder();

  const data = Array(num_bytes / 4).fill(2 ** 32 - 1);
  const data_bytes = numbers_to_u8s_for_gpu(data);

  const start = Date.now();

  const data_sb = create_and_write_sb(device, data_bytes);

  const bindGroupLayout = create_bind_group_layout(device, ["storage"]);
  const bindGroup = create_bind_group(device, bindGroupLayout, [data_sb]);

  const num_x_workgroups = 1;

  const computePipeline = await create_compute_pipeline(
    device,
    [bindGroupLayout],
    shaderCode,
    "main",
  );

  const passEncoder = commandEncoder.beginComputePass();
  passEncoder.setPipeline(computePipeline);
  passEncoder.setBindGroup(0, bindGroup);
  passEncoder.dispatchWorkgroups(num_x_workgroups);
  passEncoder.end();

  if (read) {
    await read_from_gpu(device, commandEncoder, [data_sb]);
  }
  const elapsed = Date.now() - start;
  if (read) {
    console.log(
      `Writing to and reading from GPU ${num_bytes} bytes (${
        num_bytes / 1024 / 1024
      } MB) took ${elapsed} ms`,
    );
  } else {
    console.log(
      `Writing to GPU (but not reading from) ${num_bytes} bytes (${
        num_bytes / 1024 / 1024
      } MB) took ${elapsed} ms`,
    );
  }

  device.destroy();
};
