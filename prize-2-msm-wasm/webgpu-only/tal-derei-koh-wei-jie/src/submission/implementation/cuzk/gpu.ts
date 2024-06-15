/*
 * Module Name: WebGPU Utility Functions
 *
 * Description:
 * This helper module provides utility functions for working with WebGPU.
 * It includes scaffolding functions for obtaining a high-performance GPU device,
 * creating and managing GPU buffers, executing compute pipelines, and more.
 */

/**
 * Requests a high-performance GPU device from the WebGPU API.
 * @returns {Promise<GPUDevice>} A promise that resolves to a GPUDevice.
 */
export const get_device = async (): Promise<GPUDevice> => {
  const gpuErrMsg = "Please use a browser that has WebGPU enabled.";
  const adapter = await navigator.gpu.requestAdapter({
    powerPreference: "high-performance",
  });
  if (!adapter) {
    console.log(gpuErrMsg);
    throw Error("Couldn't request WebGPU adapter.");
  }
  const device = await adapter.requestDevice();
  return device;
};

/**
 * Creates a GPUBuffer for storage, and initializes it with the provided data.
 * @param {GPUDevice} device - The GPU device to use.
 * @param {Uint8Array} buffer - The data to initialize the storage buffer with.
 * @returns {GPUBuffer} The created and initialized storage buffer.
 */
export const create_and_write_sb = (
  device: GPUDevice,
  buffer: Uint8Array,
): GPUBuffer => {
  const sb = device.createBuffer({
    size: buffer.length,
    usage:
      GPUBufferUsage.STORAGE |
      GPUBufferUsage.COPY_SRC |
      GPUBufferUsage.COPY_DST,
  });
  device.queue.writeBuffer(sb, 0, buffer);
  return sb;
};

/**
 * Creates a GPUBuffer for uniforms, and initializes it with the provided data.
 * @param {GPUDevice} device - The GPU device to use.
 * @param {Uint8Array} buffer - The data to initialize the uniform buffer with.
 * @returns {GPUBuffer} The created and initialized uniform buffer.
 */
export const create_and_write_ub = (
  device: GPUDevice,
  buffer: Uint8Array,
): GPUBuffer => {
  const ub = device.createBuffer({
    size: buffer.length,
    usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
  });
  device.queue.writeBuffer(ub, 0, buffer);
  return ub;
};

/**
 * Allocates memory for a storage buffer without initializing it.
 * @param {GPUDevice} device - The GPU device to use.
 * @param {number} size - The size of the buffer to create.
 * @returns {GPUBuffer} The created storage buffer.
 */
export const create_sb = (device: GPUDevice, size: number): GPUBuffer => {
  return device.createBuffer({
    size,
    usage:
      GPUBufferUsage.STORAGE |
      GPUBufferUsage.COPY_SRC |
      GPUBufferUsage.COPY_DST,
  });
};

/**
 * Reads data from the specified storage buffers.
 * @param {GPUDevice} device - The GPU device to use.
 * @param {GPUCommandEncoder} commandEncoder - The command encoder for GPU commands.
 * @param {GPUBuffer[]} storage_buffers - An array of storage buffers to read from.
 * @param {number} custom_size - The size of data to read from each buffer.
 * @param {number} source_offset - The offset in the source buffer to start reading from.
 * @param {number} dest_offset=0 - The offset in the destination buffer to start writing to.
 * @returns {Promise<Uint8Array[]>} A promise that resolves to an array of Uint8Array containing the read data.
 */
export const read_from_gpu = async (
  device: GPUDevice,
  commandEncoder: GPUCommandEncoder,
  storage_buffers: GPUBuffer[],
  custom_size = 0,
  source_offset = 0,
  dest_offset = 0,
) => {
  const staging_buffers: GPUBuffer[] = [];
  for (const storage_buffer of storage_buffers) {
    const size = custom_size === 0 ? storage_buffer.size : custom_size;
    const staging_buffer = device.createBuffer({
      size,
      usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST,
    });
    commandEncoder.copyBufferToBuffer(
      storage_buffer,
      source_offset,
      staging_buffer,
      dest_offset,
      size,
    );
    staging_buffers.push(staging_buffer);
  }

  /// Finish encoding commands and submit to GPU device command queue.
  device.queue.submit([commandEncoder.finish()]);

  /// Map staging buffers to read results back to JS.
  const data = [];
  for (let i = 0; i < staging_buffers.length; i++) {
    await staging_buffers[i].mapAsync(
      GPUMapMode.READ,
      0,
      staging_buffers[i].size,
    );
    const result_data = staging_buffers[i]
      .getMappedRange(0, staging_buffers[i].size)
      .slice(0);
    staging_buffers[i].unmap();
    data.push(new Uint8Array(result_data));
  }
  return data;
};

/**
 * Creates a GPUBindGroupLayout specifying the order and type of buffers expected by a shader.
 * @param {GPUDevice} device - The GPU device to use.
 * @param {string[]} types - An array of buffer types ("storage", "read-only-storage", or "uniform").
 * @returns {GPUBindGroupLayout} The created GPUBindGroupLayout.
 */
export const create_bind_group_layout = (
  device: GPUDevice,
  types: string[],
) => {
  const entries: any[] = [];
  for (let i = 0; i < types.length; i++) {
    entries.push({
      binding: i,
      visibility: GPUShaderStage.COMPUTE,
      buffer: { type: types[i] },
    });
  }
  return device.createBindGroupLayout({ entries });
};

/**
 * Creates a GPUBindGroup that specifies the storage and uniform buffers to be used by a shader.
 * @param {GPUDevice} device - The GPU device to use.
 * @param {GPUBindGroupLayout} layout - The bind group layout that defines the order and type of buffers.
 * @param {GPUBuffer[]} buffers - An array of GPUBuffers to include in the bind group.
 * @returns {GPUBindGroup} The created GPUBindGroup.
 */
export const create_bind_group = (
  device: GPUDevice,
  layout: GPUBindGroupLayout,
  buffers: GPUBuffer[],
) => {
  const entries: any[] = [];
  for (let i = 0; i < buffers.length; i++) {
    entries.push({
      binding: i,
      resource: { buffer: buffers[i] },
    });
  }
  return device.createBindGroup({ layout, entries });
};

/**
 * Asynchronously create a compute pipeline. It's recommended to use
 * createComputePipelineAsync instead of createComputePipeline to prevent stalls:
 * https://www.khronos.org/assets/uploads/developers/presentations/WebGPU_Best_Practices_Google.pdf.
 * @param {GPUDevice} device - The GPU device to use.
 * @param {GPUBindGroupLayout[]} bindGroupLayouts - An array of GPUBindGroupLayouts for the pipeline.
 * @param {string} code - The WGSL code for the shader.
 * @param {string} entryPoint - The entry point in the shader code.
 * @returns {Promise<GPUComputePipeline>} A promise that resolves to the created GPUComputePipeline.
 */
export const create_compute_pipeline = async (
  device: GPUDevice,
  bindGroupLayouts: GPUBindGroupLayout[],
  code: string,
  entryPoint: string,
) => {
  const m = device.createShaderModule({ code });
  return device.createComputePipelineAsync({
    layout: device.createPipelineLayout({ bindGroupLayouts }),
    compute: { module: m, entryPoint },
  });
};

/**
 * Encode pipeline commands.
 * @param {GPUCommandEncoder} commandEncoder - The command encoder for GPU commands.
 * @param {GPUComputePipeline} computePipeline - The compute pipeline to execute.
 * @param {GPUBindGroup} bindGroup - The bind group containing resources needed by the pipeline.
 * @param {number} num_x_workgroups - The number of workgroups to dispatch in the X dimension.
 * @param {number} num_y_workgroups - The number of workgroups to dispatch in the Y dimension.
 * @param {number} num_z_workgroups - The number of workgroups to dispatch in the Z dimension.
 */
export const execute_pipeline = async (
  commandEncoder: GPUCommandEncoder,
  computePipeline: GPUComputePipeline,
  bindGroup: GPUBindGroup,
  num_x_workgroups: number,
  num_y_workgroups = 1,
  num_z_workgroups = 1,
) => {
  const passEncoder = commandEncoder.beginComputePass();
  passEncoder.setPipeline(computePipeline);
  passEncoder.setBindGroup(0, bindGroup);
  passEncoder.dispatchWorkgroups(
    num_x_workgroups,
    num_y_workgroups,
    num_z_workgroups,
  );
  passEncoder.end();
};
