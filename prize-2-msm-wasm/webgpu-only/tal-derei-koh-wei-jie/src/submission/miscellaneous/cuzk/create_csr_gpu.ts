import assert from "assert";
import mustache from "mustache";
import { BigIntPoint, U32ArrayPoint } from "../../../reference/types";
import { CSRSparseMatrix } from "../../miscellaneous/matrices/matrices";
import { FieldMath } from "../../../reference/utils/FieldMath";
import {
  gen_p_limbs,
  u8s_to_points,
  points_to_u8s_for_gpu,
  numbers_to_u8s_for_gpu,
  bigIntPointToExtPointType,
  u8s_to_numbers,
  compute_misc_params,
} from "../../implementation/cuzk/utils";
import { ExtPointType } from "@noble/curves/abstract/edwards";
import { get_device, create_bind_group } from "../../implementation/cuzk/gpu";
import structs from "../../implementation/wgsl/struct/structs.template.wgsl";
import bigint_funcs from "../../implementation/wgsl/bigint/bigint.template.wgsl";
import field_funcs from "../../implementation/wgsl/field/field.template.wgsl";
import ec_funcs from "../../implementation/wgsl/curve/ec.template.wgsl";
import montgomery_product_funcs from "../../implementation/wgsl/montgomery/mont_pro_product.template.wgsl";
import create_csr_shader from "../wgsl/create_csr.template.wgsl";
import { all_precomputation, create_csr_cpu } from "./create_csr";
import { decompose_scalars } from "../utils";

const fieldMath = new FieldMath();

export const create_csr_sms_gpu = async (
  num_x_workgroups: number,
  num_y_workgroups: number,
  workgroup_size: number,
  points: ExtPointType[],
  decomposed_scalars: number[][],
  num_rows: number,
  device: GPUDevice,
  points_storage_buffer: GPUBuffer,
  p: bigint,
  n0: bigint,
  rinv: bigint,
  num_words: number,
  word_size: number,
) => {
  const p_limbs = gen_p_limbs(p, num_words, word_size);
  const shaderCode = mustache.render(
    create_csr_shader,
    {
      num_y_workgroups,
      workgroup_size,
      num_words,
      word_size,
      n0,
      mask: BigInt(2) ** BigInt(word_size) - BigInt(1),
      two_pow_word_size: BigInt(2) ** BigInt(word_size),
      p_limbs,
    },
    {
      structs,
      bigint_funcs,
      field_funcs,
      ec_funcs,
      montgomery_product_funcs,
    },
  );

  const shaderModule = device.createShaderModule({ code: shaderCode });
  const csr_sms_gpu: CSRSparseMatrix[] = [];
  for (const scalar_chunks of decomposed_scalars) {
    const csr_sm = await create_csr_gpu(
      num_x_workgroups,
      points,
      scalar_chunks,
      num_rows,
      device,
      shaderModule,
      points_storage_buffer,
      p,
      n0,
      rinv,
      num_words,
      word_size,
    );
    csr_sms_gpu.push(csr_sm);
  }

  return csr_sms_gpu;
};

export const create_csr_gpu = async (
  num_x_workgroups: number,
  points: ExtPointType[],
  scalar_chunks: number[],
  num_rows: number,
  device: GPUDevice,
  shaderModule: GPUShaderModule,
  points_storage_buffer: GPUBuffer,
  p: bigint,
  n0: bigint,
  rinv: bigint,
  num_words: number,
  word_size: number,
): Promise<CSRSparseMatrix> => {
  //const wasm_result = wasm.all_precomputation(
  //new Uint32Array(scalar_chunks),
  //num_rows,
  //)
  //const all_new_point_indices = Array.from(wasm_result.get_all_new_point_indices())
  //const all_cluster_start_indices = Array.from(wasm_result.get_all_cluster_start_indices())
  //const all_cluster_end_indices = Array.from(wasm_result.get_all_cluster_end_indices())
  //const all_single_point_indices = Array.from(wasm_result.get_all_single_point_indices())
  //const all_single_scalar_chunks = Array.from(wasm_result.get_all_single_scalar_chunks())
  //const row_ptr = Array.from(wasm_result.get_row_ptr())
  const {
    all_new_point_indices,
    all_cluster_start_indices,
    all_cluster_end_indices,
    all_single_point_indices,
    all_single_scalar_chunks,
    row_ptr,
  } = all_precomputation(scalar_chunks, num_rows);

  const scalar_chunks_bytes = numbers_to_u8s_for_gpu(scalar_chunks);
  const new_point_indices_bytes = numbers_to_u8s_for_gpu(all_new_point_indices);
  const cluster_start_indices_bytes = numbers_to_u8s_for_gpu(
    all_cluster_start_indices,
  );
  const cluster_end_indices_bytes = numbers_to_u8s_for_gpu(
    all_cluster_end_indices,
  );

  const scalar_chunks_storage_buffer = device.createBuffer({
    size: scalar_chunks_bytes.length,
    usage:
      GPUBufferUsage.STORAGE |
      GPUBufferUsage.COPY_DST |
      GPUBufferUsage.COPY_SRC,
  });
  device.queue.writeBuffer(
    scalar_chunks_storage_buffer,
    0,
    scalar_chunks_bytes,
  );

  const new_point_indices_storage_buffer = device.createBuffer({
    size: new_point_indices_bytes.length,
    usage:
      GPUBufferUsage.STORAGE |
      GPUBufferUsage.COPY_DST |
      GPUBufferUsage.COPY_SRC,
  });
  device.queue.writeBuffer(
    new_point_indices_storage_buffer,
    0,
    new_point_indices_bytes,
  );

  const cluster_start_indices_storage_buffer = device.createBuffer({
    size: cluster_start_indices_bytes.length,
    usage:
      GPUBufferUsage.STORAGE |
      GPUBufferUsage.COPY_DST |
      GPUBufferUsage.COPY_SRC,
  });
  device.queue.writeBuffer(
    cluster_start_indices_storage_buffer,
    0,
    cluster_start_indices_bytes,
  );

  const cluster_end_indices_storage_buffer = device.createBuffer({
    size: cluster_end_indices_bytes.length,
    usage:
      GPUBufferUsage.STORAGE |
      GPUBufferUsage.COPY_DST |
      GPUBufferUsage.COPY_SRC,
  });
  device.queue.writeBuffer(
    cluster_end_indices_storage_buffer,
    0,
    cluster_end_indices_bytes,
  );

  const num_new_points = all_cluster_start_indices.length;

  // Output buffers
  const new_points_storage_buffer = device.createBuffer({
    size: num_new_points * 320,
    usage:
      GPUBufferUsage.STORAGE |
      GPUBufferUsage.COPY_DST |
      GPUBufferUsage.COPY_SRC,
  });

  const new_scalar_chunks_storage_buffer = device.createBuffer({
    size: cluster_start_indices_bytes.length,
    usage:
      GPUBufferUsage.STORAGE |
      GPUBufferUsage.COPY_DST |
      GPUBufferUsage.COPY_SRC,
  });

  const bindGroupLayout = device.createBindGroupLayout({
    entries: [
      {
        binding: 0,
        visibility: GPUShaderStage.COMPUTE,
        buffer: { type: "read-only-storage" },
      },
      {
        binding: 1,
        visibility: GPUShaderStage.COMPUTE,
        buffer: { type: "read-only-storage" },
      },
      {
        binding: 2,
        visibility: GPUShaderStage.COMPUTE,
        buffer: { type: "read-only-storage" },
      },
      {
        binding: 3,
        visibility: GPUShaderStage.COMPUTE,
        buffer: { type: "read-only-storage" },
      },
      {
        binding: 4,
        visibility: GPUShaderStage.COMPUTE,
        buffer: { type: "read-only-storage" },
      },
      {
        binding: 5,
        visibility: GPUShaderStage.COMPUTE,
        buffer: { type: "storage" },
      },
      {
        binding: 6,
        visibility: GPUShaderStage.COMPUTE,
        buffer: { type: "storage" },
      },
    ],
  });

  const bindGroup = create_bind_group(device, bindGroupLayout, [
    points_storage_buffer,
    scalar_chunks_storage_buffer,
    new_point_indices_storage_buffer,
    cluster_start_indices_storage_buffer,
    cluster_end_indices_storage_buffer,
    new_points_storage_buffer,
    new_scalar_chunks_storage_buffer,
  ]);

  const computePipeline = device.createComputePipeline({
    layout: device.createPipelineLayout({
      bindGroupLayouts: [bindGroupLayout],
    }),
    compute: {
      module: shaderModule,
      entryPoint: "main",
    },
  });

  const start = Date.now();

  const commandEncoder = device.createCommandEncoder();
  const passEncoder = commandEncoder.beginComputePass();
  passEncoder.setPipeline(computePipeline);
  passEncoder.setBindGroup(0, bindGroup);
  passEncoder.dispatchWorkgroups(num_x_workgroups);
  passEncoder.end();

  const new_points_staging_buffer = device.createBuffer({
    size: new_points_storage_buffer.size,
    usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST,
  });
  const new_scalar_chunks_staging_buffer = device.createBuffer({
    size: new_scalar_chunks_storage_buffer.size,
    usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST,
  });

  commandEncoder.copyBufferToBuffer(
    new_points_storage_buffer,
    0,
    new_points_staging_buffer,
    0,
    new_points_storage_buffer.size,
  );
  commandEncoder.copyBufferToBuffer(
    new_scalar_chunks_storage_buffer,
    0,
    new_scalar_chunks_staging_buffer,
    0,
    new_scalar_chunks_storage_buffer.size,
  );

  device.queue.submit([commandEncoder.finish()]);

  // map staging buffers to read results back to JS
  await new_points_staging_buffer.mapAsync(
    GPUMapMode.READ,
    0,
    new_points_storage_buffer.size,
  );
  await new_scalar_chunks_staging_buffer.mapAsync(
    GPUMapMode.READ,
    0,
    new_scalar_chunks_storage_buffer.size,
  );

  const np = new_points_staging_buffer.getMappedRange(
    0,
    new_points_staging_buffer.size,
  );
  const new_points_data = np.slice(0);
  new_points_staging_buffer.unmap();

  const ns = new_scalar_chunks_staging_buffer.getMappedRange(
    0,
    new_scalar_chunks_staging_buffer.size,
  );
  const new_scalar_chunks_data = ns.slice(0);
  new_scalar_chunks_staging_buffer.unmap();

  const elapsed = Date.now() - start;
  console.log(`GPU took ${elapsed}ms (including data transfer)`);

  const new_scalar_chunks = u8s_to_numbers(
    new Uint8Array(new_scalar_chunks_data),
  );
  const new_points = u8s_to_points(
    new Uint8Array(new_points_data),
    num_words,
    word_size,
  );

  // Convert out of Mont form
  const new_points_non_mont: any[] = [];
  for (const pt of new_points) {
    const non = fieldMath.createPoint(
      fieldMath.Fp.mul(pt.x, rinv),
      fieldMath.Fp.mul(pt.y, rinv),
      fieldMath.Fp.mul(pt.t, rinv),
      fieldMath.Fp.mul(pt.z, rinv),
    );
    new_points_non_mont.push(non.toAffine());
  }

  // TODO: in the end-to-end version, write all_single_points,
  // all_single_scalar_chunks, and row_ptr to the GPU buffers at the
  // appropriate offsets.

  return new CSRSparseMatrix(
    new_points_non_mont.concat(
      all_single_point_indices.map((x: number) => points[x]),
    ),
    new_scalar_chunks.concat(all_single_scalar_chunks),
    row_ptr,
  );

  //const data: ExtPointType[] = []
  //const col_idx: number[] = []
  //const row_ptr: number[] = []
  //return new CSRSparseMatrix(data, col_idx, row_ptr)
};

export async function create_csr_precomputation_benchmark(
  baseAffinePoints: BigIntPoint[] | U32ArrayPoint[] | Buffer,
  scalars: bigint[] | Uint32Array[] | Buffer,
): Promise<{ x: bigint; y: bigint }> {
  const num_rows = 16;
  const num_words = 16;
  const word_size = 16;

  const start_decomposed = Date.now();
  const decomposed_scalars = decompose_scalars(
    scalars as bigint[],
    num_words,
    word_size,
  );
  const elapsed_decomposed = Date.now() - start_decomposed;
  console.log(`CPU took ${elapsed_decomposed}ms to decompose scalars`);

  const start = Date.now();
  for (const scalar_chunks of decomposed_scalars) {
    all_precomputation(scalar_chunks, num_rows);
  }
  const elapsed = Date.now() - start;
  console.log(`CPU took ${elapsed}ms`);
  return { x: BigInt(0), y: BigInt(1) };
}

export async function create_csr_sparse_matrices_from_points_benchmark(
  baseAffinePoints: BigIntPoint[] | U32ArrayPoint[] | Buffer,
  scalars: bigint[] | Uint32Array[] | Buffer,
): Promise<{ x: bigint; y: bigint }> {
  const num_rows = 8;
  const points = (baseAffinePoints as BigIntPoint[]).map((x) =>
    bigIntPointToExtPointType(x as BigIntPoint, fieldMath),
  );
  /* eslint-disable @typescript-eslint/no-unused-vars */
  const csr_sms = await create_csr_sparse_matrices_from_points(
    points,
    scalars as bigint[],
    num_rows,
  );
  return { x: BigInt(0), y: BigInt(1) };
}

export async function create_csr_sparse_matrices_from_points(
  points: ExtPointType[],
  scalars: bigint[],
  num_rows: number,
): Promise<CSRSparseMatrix[]> {
  const num_x_workgroups = 256;
  const workgroup_size = 256;
  const num_y_workgroups = points.length / workgroup_size / num_x_workgroups;

  // The number of threads is the number of rows of the matrix
  // As such the number of threads should divide the number of points
  assert(points.length % num_rows === 0);
  assert(points.length === scalars.length);

  const num_words = 20;
  const word_size = 13;

  // Decompose scalars
  const decomposed_scalars = decompose_scalars(scalars, num_words, word_size);

  // Compute CSR sparse matrices in CPU
  const start = Date.now();
  const expected_csr_sms: CSRSparseMatrix[] = [];
  for (const scalar_chunks of decomposed_scalars) {
    const csr_sm = create_csr_cpu(points, scalar_chunks, num_rows);
    expected_csr_sms.push(csr_sm);
  }
  const elapsed = Date.now() - start;
  console.log(`CPU took ${elapsed}ms`);

  const fieldMath = new FieldMath();
  const p = BigInt(
    "0x12ab655e9a2ca55660b44d1e5c37b00159aa76fed00000010a11800000000001",
  );
  const params = compute_misc_params(p, word_size);
  const r = params.r;
  const n0 = params.n0;
  const rinv = params.rinv;

  // Convert points to Montgomery coordinates
  const points_with_mont_coords: BigIntPoint[] = [];
  for (const pt of points) {
    points_with_mont_coords.push({
      x: fieldMath.Fp.mul(pt.ex, r),
      y: fieldMath.Fp.mul(pt.ey, r),
      t: fieldMath.Fp.mul(pt.et, r),
      z: fieldMath.Fp.mul(pt.ez, r),
    });
  }

  const device = await get_device();
  // Store all the points in a GPU buffer
  const points_bytes = points_to_u8s_for_gpu(
    points_with_mont_coords,
    num_words,
    word_size,
  );
  const points_storage_buffer = device.createBuffer({
    size: points_bytes.length,
    usage:
      GPUBufferUsage.STORAGE |
      GPUBufferUsage.COPY_DST |
      GPUBufferUsage.COPY_SRC,
  });
  device.queue.writeBuffer(points_storage_buffer, 0, points_bytes);

  const csr_sms_gpu = await create_csr_sms_gpu(
    num_x_workgroups,
    num_y_workgroups,
    workgroup_size,
    points,
    decomposed_scalars,
    num_rows,
    device,
    points_storage_buffer,
    p,
    n0,
    rinv,
    num_words,
    word_size,
  );

  assert(csr_sms_gpu.length === expected_csr_sms.length);
  for (let i = 0; i < expected_csr_sms.length; i++) {
    try {
      assert(csr_sms_gpu[i].data.length === expected_csr_sms[i].data.length);
      assert(
        csr_sms_gpu[i].col_idx.length === expected_csr_sms[i].col_idx.length,
      );
      assert(
        csr_sms_gpu[i].row_ptr.length === expected_csr_sms[i].row_ptr.length,
      );

      for (let j = 0; j < expected_csr_sms[i].data.length; j++) {
        assert(csr_sms_gpu[i].data[j].x === expected_csr_sms[i].data[j].x);
        assert(csr_sms_gpu[i].data[j].y === expected_csr_sms[i].data[j].y);
      }
      for (let j = 0; j < expected_csr_sms[i].col_idx.length; j++) {
        assert(csr_sms_gpu[i].col_idx[j] === expected_csr_sms[i].col_idx[j]);
      }
      for (let j = 0; j < expected_csr_sms[i].row_ptr.length; j++) {
        assert(csr_sms_gpu[i].row_ptr[j] === expected_csr_sms[i].row_ptr[j]);
      }
    } catch {
      console.log("assert fail at", i);
      debugger;
      break;
    }
  }

  device.destroy();

  return csr_sms_gpu;
}
