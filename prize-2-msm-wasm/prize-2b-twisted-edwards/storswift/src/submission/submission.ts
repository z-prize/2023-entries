import { BigIntPoint, U32ArrayPoint } from "../reference/types";
import { FieldMath } from "./FieldMath";
import { bigIntsToU32Array, gpuU32Inputs, u32ArrayToBigInts } from "./utils";


import { U256WGSL } from "./wgsl/U256";
import { FieldWGSL } from "./wgsl/Field";
import { CurveWGSL } from "./wgsl/Curve";



const FIELD_SIZE = 8; // 8 u32s to make 256 bits
const EXT_POINT_SIZE = 32; // 4 Fields to make one extended point
const SCALAR_BITS = 253; // 253 bits for scalar

/// Pippinger Algorithm Summary:
/// 
/// Great explanation of algorithm can be found here:
/// https://www.youtube.com/watch?v=Bl5mQA7UL2I
///
/// 1) Break down each 256bit scalar k into 256/c, c bit scalars 
///    ** Note: c = 16 seems to be optimal per source mentioned above
///
/// 2) Set up 256/c different MSMs T_1, T_2, ..., T_c where
///    T_1 = a_1_1(P_1) + a_2_1(P_2) + ... + a_n_1(P_n)
///    T_2 = a_1_2(P_1) + a_2_2(P_2) + ... + a_n_2(P_n)
///     .
///     .
///     .
///    T_c = a_1_c(P_1) + a_2_c(P_2) + ... + a_n_c(P_n)
///
/// 3) Use Bucket Method to efficiently compute each MSM
///    * Create 2^c - 1 buckets where each bucket represents a c-bit scalar
///    * In each bucket, keep a running sum of all the points that are mutliplied 
///      by the corresponding scalar
///    * T_i = 1(SUM(Points)) + 2(SUM(Points)) + ... + (2^c - 1)(SUM(Points))
///
/// 4) Once the result of each T_i is calculated, can compute the original
///    MSM (T) with the following formula:
///    T <- T_1
///    for j = 2,...,256/c:
///        T <- (2^c) * T
///        T <- T + T_j
/* eslint-disable @typescript-eslint/no-unused-vars */
export const compute_msm = async (
  baseAffinePoints: BigIntPoint[] | U32ArrayPoint[] | Uint32Array,
  basescalars: bigint[] | Uint32Array[] | Uint32Array
  ): Promise<{x: bigint, y: bigint}> => {

    const pointsAsU32s  = baseAffinePoints as Uint32Array;
    const scalarsAsU32s = basescalars as Uint32Array;

    const bufferResult = await pippinger_msm(
      { u32Inputs: pointsAsU32s, individualInputSize: EXT_POINT_SIZE }, 
      { u32Inputs: scalarsAsU32s, individualInputSize: FIELD_SIZE }
    );
    const fieldMath = new FieldMath();
    let acc = fieldMath.customEdwards.ExtendedPoint.ZERO;
    const bigIntResult = u32ArrayToBigInts(bufferResult || new Uint32Array(0));
    for (let i = 0; i < bigIntResult.length; i += 4) {
      const x = bigIntResult[i];
      const y = bigIntResult[i + 1];
      const t = bigIntResult[i + 2];
      const z = bigIntResult[i + 3];
      acc = acc.add(fieldMath.createPoint(x, y, t, z));
    }

    const affineResult = acc.toAffine();
    return { x: affineResult.x, y: affineResult.y };
}

const getDevice = async () => {
  if (!navigator.gpu) {
    console.log("WebGPU not supported.");
    return;
  }

  const adapter = await navigator.gpu.requestAdapter({powerPreference: "high-performance"});
  if (!adapter) { 
    console.log("Couldn't request WebGPU adapter.");
    return;
  }

  return await adapter.requestDevice({
    requiredLimits: {
        maxBufferSize: adapter.limits.maxBufferSize,
        maxStorageBufferBindingSize: adapter.limits.maxStorageBufferBindingSize,
        maxComputeWorkgroupStorageSize: adapter.limits.maxComputeWorkgroupStorageSize,
        maxComputeWorkgroupSizeX: 256,
    },
  });
}

async function pippinger_msm(
  points: gpuU32Inputs, scalars: gpuU32Inputs) {
    const PARTITION_SIZE = 6;
    const shaderEntry = `
      const PARTITION_SIZE = ${PARTITION_SIZE}u;
      const POW_PART = (1u << PARTITION_SIZE);

      
      @group(0) @binding(0)
      var<storage, read_write> points: array<Point>;
      @group(0) @binding(1)
      var<storage, read> scalars: array<Field>;
      @group(0) @binding(2)
      var<storage, read_write> powerset: array<Point>;
      @group(0) @binding(3)
      var<storage, read_write> result: array<Point>;

      @compute @workgroup_size(128)
      fn init(
        @builtin(global_invocation_id) global_id : vec3<u32>
      ) {
        let gidx = global_id.x;
        let ps_base = gidx * (POW_PART - 1u);
        let point_base = gidx * PARTITION_SIZE;
        let point_size = arrayLength(&scalars);
        if(point_base >= point_size) {
          return;
        }

        var idx = 0u;
        for(var j = 1u; j < POW_PART; j = j + 1u){
          if((i32(j) & -i32(j)) == i32(j)) {
            if(point_base + idx >= point_size) {
              break;
            }
            powerset[ps_base + j - 1u] = points[point_base + idx];
            idx = idx + 1u;
          } else {
            let mask = j & u32(j - 1u);
            let other_mask = j ^ mask;
            powerset[ps_base + j - 1u] = add_points(powerset[ps_base + mask - 1u], powerset[ps_base + other_mask - 1u]);
          }
        }
      }

      @compute @workgroup_size(${SCALAR_BITS})
      fn msm(
        @builtin(global_invocation_id) global_id : vec3<u32>,
        @builtin(local_invocation_id) local_id: vec3<u32>,
        @builtin(workgroup_id) workgroup_id : vec3<u32>,
        @builtin(num_workgroups) num_workgroups: vec3<u32>
      ) {
        let point_size = arrayLength(&scalars);
        let wbit = local_id.x + 256u - ${SCALAR_BITS}u;
        let mask_W = 1u << (31u - (wbit & 31u));
        let quotbW = wbit >> 5u;
        let ps_group = num_workgroups.x * (POW_PART - 1u);        
        let point_group = num_workgroups.x * PARTITION_SIZE;

        var sum = INF_POINT;
        var ps_base = workgroup_id.x * (POW_PART - 1u);
        var point_base = workgroup_id.x * PARTITION_SIZE;
        while (point_base < point_size) {
          var powerset_idx = 0u;
          for(var j = 0u; j < PARTITION_SIZE; j = j + 1u){
            var point_idx = point_base + j;
            if (point_idx < point_size && (scalars[point_idx][quotbW] & mask_W) > 0) {
                powerset_idx = powerset_idx | (1u << j);
            }
          }
          if (powerset_idx > 0u) {
            sum = add_points(sum,  powerset[ps_base + powerset_idx - 1u]);
          }
          ps_base = ps_base + ps_group;
          point_base = point_base + point_group;
        }
        points[workgroup_id.x + local_id.x * num_workgroups.x] = sum;
      }
      var<workgroup> sp: array<Point, 64>;
      @compute @workgroup_size(64)
      fn acc(
          @builtin(global_invocation_id) global_id: vec3<u32>,
          @builtin(local_invocation_id) local_id: vec3<u32>,
          @builtin(workgroup_id) workgroup_id : vec3<u32>,
      ) {
          let psize = arrayLength(&result);
          if (global_id.x < psize) {
            var acc = points[global_id.x];
            for(var i = 1u; i < ${SCALAR_BITS}u; i = i + 1u) {
              acc = double_point(acc);
              acc = add_points(acc, points[global_id.x + i * psize]);
            }
            sp[local_id.x] = acc;
          }
          workgroupBarrier();

          for (var offset: u32 = 32u; offset >= 2u; offset = offset / 2u) {
            if (local_id.x < offset && global_id.x + offset < psize) {
              sp[local_id.x]= add_points(sp[local_id.x], sp[local_id.x + offset]);
            }
            workgroupBarrier();
          }

          if (local_id.x < 2u && global_id.x < psize) { 
             result[workgroup_id.x * 2u + local_id.x] = sp[local_id.x];
          }
      }
      `;


    const device = await getDevice();
    if (!device) {
      return new Uint32Array(32);
    }

    const shaderCode = [U256WGSL, FieldWGSL, CurveWGSL, shaderEntry].join('');
    const numInputs = points.u32Inputs.length / points.individualInputSize;
    const groupSize = Math.ceil(numInputs / PARTITION_SIZE);  
    const resultPoints = Math.min(Math.ceil(groupSize / 32), 1024);
    const powersetPoints = ((1<<PARTITION_SIZE)-1) * groupSize;
    const inputPoints = Math.max(numInputs, resultPoints * SCALAR_BITS);

    const module = device.createShaderModule({code: shaderCode });
  
    const pointsBuffer = device.createBuffer({
      mappedAtCreation: true,
      size: Uint32Array.BYTES_PER_ELEMENT * 32 * inputPoints,
      usage: GPUBufferUsage.STORAGE| GPUBufferUsage.COPY_SRC
    });
    new Uint32Array(pointsBuffer.getMappedRange()).set(points.u32Inputs);
    pointsBuffer.unmap();

    const scalarsBuffer = device.createBuffer({
      mappedAtCreation: true,
      size: scalars.u32Inputs.byteLength,
      usage: GPUBufferUsage.STORAGE
    });
    new Uint32Array(scalarsBuffer.getMappedRange()).set(scalars.u32Inputs);
    scalarsBuffer.unmap();

    const powersetBuffer = device.createBuffer({
      size: Uint32Array.BYTES_PER_ELEMENT * 32 * powersetPoints,
      usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC
    });

    const resultBuffer = device.createBuffer({
      size: Uint32Array.BYTES_PER_ELEMENT * 32 * resultPoints,
      usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC
    });

    const readPoints = Math.trunc(resultPoints / 64) * 2 + Math.min(resultPoints % 64, 2);
    const gpuReadBuffer = device.createBuffer({
      size:  Uint32Array.BYTES_PER_ELEMENT * 32 * readPoints, 
      usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ
    });
  
    // Bind group layout and bind group
    const bindGroupLayout = device.createBindGroupLayout({
      entries: [
          { binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: "storage" } },
          { binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: "read-only-storage" } },
          { binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: "storage" } },
          { binding: 3, visibility: GPUShaderStage.COMPUTE, buffer: { type: "storage" } },
      ],
    });
  
    const bindGroup = device.createBindGroup({
      layout: bindGroupLayout,
      entries: [
          { binding: 0, resource: { buffer: pointsBuffer } },
          { binding: 1, resource: { buffer: scalarsBuffer } },
          { binding: 2, resource: { buffer: powersetBuffer} },
          { binding: 3, resource: { buffer: resultBuffer} },
      ],
    });
  
    // Pipeline setup
    const layout = device.createPipelineLayout({
      bindGroupLayouts: [bindGroupLayout]
    });
    const initPipeline = await device.createComputePipelineAsync({
      layout: layout,
      compute: { module: module, entryPoint: "init"}
    });
    const msmPipeline = await device.createComputePipelineAsync({
      layout: layout,
      compute: { module: module, entryPoint: "msm"}
    });
    const accPipeline = await device.createComputePipelineAsync({
      layout: layout,
      compute: { module: module, entryPoint: "acc"}
    });

    // Commands submission
    const commandEncoder = device.createCommandEncoder();
    const initEncoder = commandEncoder.beginComputePass();
    initEncoder.setPipeline(initPipeline);
    initEncoder.setBindGroup(0, bindGroup);
    initEncoder.dispatchWorkgroups(Math.ceil(groupSize / 128));
    initEncoder.end();

    const msmEncoder = commandEncoder.beginComputePass();
    msmEncoder.setPipeline(msmPipeline);
    msmEncoder.setBindGroup(0, bindGroup);
    msmEncoder.dispatchWorkgroups(resultPoints);
    msmEncoder.end();

    const accEncoder = commandEncoder.beginComputePass();
    accEncoder.setPipeline(accPipeline);
    accEncoder.setBindGroup(0, bindGroup);
    accEncoder.dispatchWorkgroups(Math.ceil(resultPoints / 64));
    accEncoder.end();

    // Encode commands for copying buffer to buffer.
    commandEncoder.copyBufferToBuffer(
      resultBuffer /* source buffer */,
      0 /* source offset */,
      gpuReadBuffer /* destination buffer */,
      0 /* destination offset */,
      gpuReadBuffer.size /* size */
    );
  
    // Submit GPU commands.
    const gpuCommands = commandEncoder.finish();
    device.queue.submit([gpuCommands]);

    // Read buffer.
    await gpuReadBuffer.mapAsync(GPUMapMode.READ);
    const arrayBuffer = gpuReadBuffer.getMappedRange();
    const result = new Uint32Array(arrayBuffer.slice(0));
    gpuReadBuffer.unmap();

    // Destroy all buffers
    pointsBuffer.destroy();
    scalarsBuffer.destroy();
    resultBuffer.destroy();
    powersetBuffer.destroy();
    gpuReadBuffer.destroy();
    device.destroy();
    
    return result;

}

