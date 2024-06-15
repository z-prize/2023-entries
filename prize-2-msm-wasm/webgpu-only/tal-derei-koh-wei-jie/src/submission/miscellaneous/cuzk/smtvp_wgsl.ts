import mustache from "mustache";
import {
  bigIntPointToExtPointType,
  compute_misc_params,
  u8s_to_points,
  points_to_u8s_for_gpu,
  numbers_to_u8s_for_gpu,
  gen_p_limbs,
  gen_r_limbs,
  gen_d_limbs,
  to_words_le,
} from "../../implementation/cuzk/utils";
import { BigIntPoint, U32ArrayPoint } from "../../../reference/types";
import { FieldMath } from "../../../reference/utils/FieldMath";
import { ELLSparseMatrix, CSRSparseMatrix } from "../matrices/matrices";
import smtvp_shader from "../wgsl/smtvp.template.wgsl";
import structs from "../../implementation/wgsl/struct/structs.template.wgsl";
import bigint_functions from "../../implementation/wgsl/bigint/bigint.template.wgsl";
import field_functions from "../../implementation/wgsl/field/field.template.wgsl";
import curve_functions from "../../implementation/wgsl/curve/ec.template.wgsl";
import montgomery_product_funcs from "../../implementation/wgsl/montgomery/mont_pro_product.template.wgsl";
import { ExtPointType } from "@noble/curves/abstract/edwards";
import { get_device, create_bind_group } from "../../implementation/cuzk/gpu";
import assert from "assert";

const fieldMath = new FieldMath();

// WGSL implementation of Sparse-Matrix Transpose
export const smtvp_wgsl = async (
  baseAffinePoints: BigIntPoint[] | U32ArrayPoint[] | Buffer,
  scalars: bigint[] | Uint32Array[] | Buffer,
): Promise<{ x: bigint; y: bigint }> => {
  console.log("Starting WGSL SMTVP!");

  const result = await smtvp(
    baseAffinePoints as BigIntPoint[],
    scalars as bigint[],
  );
  return result;
};

/*
 * Returns an array of CSR sparse matrices. Each sparse matrix contains the
 * same number of points, but different col_idx values.
 */
export async function gen_csr_sparse_matrices(
  baseAffinePoints: BigIntPoint[],
  scalars: bigint[],
  lambda: number, // λ-bit scalars
  window_size: number, // s-bit window size
  threads: number, // Thread count
): Promise<CSRSparseMatrix[]> {
  // Number of rows and columns (ie. row-space)
  const num_rows = threads;
  const num_columns = Math.pow(2, window_size) - 1;

  const ZERO_POINT = fieldMath.customEdwards.ExtendedPoint.ZERO;

  const csr_sparse_matrix_array: CSRSparseMatrix[] = [];
  const mask = (BigInt(1) << BigInt(window_size)) - BigInt(1);

  for (let i = 0; i < num_rows; i++) {
    // Instantiate empty ELL sparse matrix format
    const data = new Array(num_rows);
    for (let i = 0; i < num_rows; i++) {
      data[i] = new Array(num_columns).fill(ZERO_POINT);
    }

    const col_idx = new Array(num_rows);
    for (let i = 0; i < num_rows; i++) {
      col_idx[i] = new Array(num_columns).fill(0);
    }

    const row_length = Array(num_rows).fill(0);

    // Perform scalar decomposition. scalars_decomposed should contain (lambda / window_size) arrays of
    const scalars_decomposed: number[][] = [];
    for (let j = Math.ceil(lambda / window_size); j > 0; j--) {
      const chunk: number[] = [];

      // For each scalar, extract the j-th chunk
      for (const scalar of scalars) {
        const limb = (scalar >> BigInt((j - 1) * window_size)) & mask; // Right shift and extract lower window_size-bits
        chunk.push(Number(limb));
      }
      scalars_decomposed.push(chunk);
    }

    // Divide EC points into t parts
    for (let thread_idx = 0; thread_idx < num_rows; thread_idx++) {
      for (let j = 0; j < num_columns; j++) {
        const point_i = thread_idx + j * threads;
        data[thread_idx][j] = baseAffinePoints[point_i];
        col_idx[thread_idx][j] = scalars_decomposed[i][point_i];
        row_length[thread_idx] += 1;
      }
    }

    // Transform ELL sparse matrix to CSR sparse matrix
    const ell_sparse_matrix = new ELLSparseMatrix(data, col_idx, row_length);
    const csr_sparse_matrix = await new CSRSparseMatrix(
      [],
      [],
      [],
    ).ell_to_csr_sparse_matrix(ell_sparse_matrix);

    csr_sparse_matrix_array.push(csr_sparse_matrix);
  }

  return csr_sparse_matrix_array;
}

export const smtvp = async (
  baseAffinePoints: BigIntPoint[],
  scalars: bigint[],
): Promise<{ x: bigint; y: bigint }> => {
  // Decompose the scalars into windows. In the actual implementation, this
  // should be done by a shader.
  const p = BigInt(
    "0x12ab655e9a2ca55660b44d1e5c37b00159aa76fed00000010a11800000000001",
  );
  const word_size = 13;
  const params = compute_misc_params(p, word_size);
  const n0 = params.n0;
  const num_words = params.num_words;

  // Each scalar has λ bits. e.g. 13 * 20 = 260 bits
  const lambda = word_size * num_words;

  const window_size = word_size;

  // Thread count
  const threads = 1;

  const csr_sparse_matrices = await gen_csr_sparse_matrices(
    baseAffinePoints,
    scalars,
    lambda,
    window_size,
    threads,
  );

  const device = await get_device();

  const col_idxs: number[] = [];
  for (const csr_sm of csr_sparse_matrices) {
    for (const c of csr_sm.col_idx) {
      col_idxs.push(c);
    }
  }

  const timings: any[] = [];

  for (let i = 0; i < csr_sparse_matrices.length; i++) {
    let max_col_idx = 0;
    for (const c of col_idxs) {
      if (c > max_col_idx) {
        max_col_idx = c;
      }
    }

    const t = await smtvp_run(
      device,
      csr_sparse_matrices[i],
      fieldMath,
      max_col_idx,
      num_words,
      word_size,
      p,
      n0,
      params.r,
      params.rinv,
      params.edwards_d,
    );
    timings.push(t);
  }

  if (timings.length === 1) {
    console.log(`CPU took ${timings[0].cpu_elapsed}ms (1 CSR sparse matrix)`);
    console.log(`GPU took ${timings[0].gpu_elapsed}ms (1 CSR sparse matrix)`);
  } else {
    let cpu_total = 0;
    let gpu_total = 0;
    for (let i = 1; i < timings.length; i++) {
      cpu_total += timings[i].cpu_elapsed;
      gpu_total += timings[i].gpu_elapsed;
    }
    const cpu_average = cpu_total / (timings.length - 1);
    const gpu_average = gpu_total / (timings.length - 1);
    console.log(
      `CPU took an average of ${cpu_average}ms per CSR sparse matrix. Benchmark ignores the first)`,
    );
    console.log(
      `GPU took an average of ${gpu_average}ms per CSR sparse matrix. Benchmark ignores the first)`,
    );
  }

  device.destroy();

  return { x: BigInt(1), y: BigInt(0) };
};

export const smtvp_run = async (
  device: GPUDevice,
  csr_sm: CSRSparseMatrix,
  fieldMath: FieldMath,
  max_col_idx: number,
  num_words: number,
  word_size: number,
  p: bigint,
  n0: bigint,
  r: bigint,
  rinv: bigint,
  d: bigint,
): Promise<{ cpu_elapsed: number; gpu_elapsed: number }> => {
  let cpu_elapsed = 0;
  let gpu_elapsed = 0;
  const run_on_cpu = true;
  let smtvp_result_affine;

  if (run_on_cpu) {
    const cpu_start = Date.now();
    // Perform SMTVP in CPU
    const vec: bigint[] = [];
    for (let i = 0; i < csr_sm.row_ptr.length - 1; i++) {
      vec.push(BigInt(1));
    }
    const smtvp_result = await csr_sm.smtvp(vec);
    smtvp_result_affine = smtvp_result.map((x) => x.toAffine());
    //console.log('points from cpu, in affine:', smtvp_result_affine)
    cpu_elapsed = Date.now() - cpu_start;
    console.log(`CPU took ${cpu_elapsed}ms`);
  }

  // Create GPUCommandEncoder to issue commands to the GPU
  const commandEncoder = device.createCommandEncoder();

  const num_x_workgroups = 1;
  //const num_rows = csr_sm.row_ptr.length - 1

  const output_buffer_length = (max_col_idx + 1) * num_words * 4 * 4;

  const col_idx_bytes = numbers_to_u8s_for_gpu(csr_sm.col_idx);
  const row_ptr_bytes = numbers_to_u8s_for_gpu(csr_sm.row_ptr);

  // Convert each point coordinate for each point in the CSR matrix into
  // Montgomery form
  const points_with_mont_coords: BigIntPoint[] = [];
  for (const pt of csr_sm.data) {
    points_with_mont_coords.push({
      x: fieldMath.Fp.mul(pt.x, r),
      y: fieldMath.Fp.mul(pt.y, r),
      t: fieldMath.Fp.mul(pt.t, r),
      z: fieldMath.Fp.mul(pt.z, r),
    });
  }

  const points_bytes = points_to_u8s_for_gpu(
    points_with_mont_coords,
    num_words,
    word_size,
  );

  const shaderCode = setup_shader_code(p, r, d, num_words, word_size, n0);

  const shaderModule = device.createShaderModule({ code: shaderCode });

  let previous_output_storage_buffer: GPUBuffer | null = null;

  const start = Date.now();

  for (let i = 0; i < csr_sm.row_ptr.length - 1; i++) {
    const w = to_words_le(BigInt(i), 8, 8);
    const loop_index_bytes = new Uint8Array(w);

    // Create input buffers
    const col_idx_storage_buffer = device.createBuffer({
      size: col_idx_bytes.length,
      usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
    });

    const row_ptr_storage_buffer = device.createBuffer({
      size: row_ptr_bytes.length,
      usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
    });

    const points_storage_buffer = device.createBuffer({
      size: points_bytes.length,
      usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
    });

    const loop_index_storage_buffer = device.createBuffer({
      size: loop_index_bytes.length,
      usage:
        GPUBufferUsage.STORAGE |
        GPUBufferUsage.COPY_SRC |
        GPUBufferUsage.COPY_DST,
    });

    device.queue.writeBuffer(col_idx_storage_buffer, 0, col_idx_bytes);
    device.queue.writeBuffer(row_ptr_storage_buffer, 0, row_ptr_bytes);
    device.queue.writeBuffer(points_storage_buffer, 0, points_bytes);
    device.queue.writeBuffer(loop_index_storage_buffer, 0, loop_index_bytes);

    // Create result buffers
    const output_storage_buffer = device.createBuffer({
      size: output_buffer_length,
      usage:
        GPUBufferUsage.STORAGE |
        GPUBufferUsage.COPY_SRC |
        GPUBufferUsage.COPY_DST,
    });

    /*
        // Before the 0th pass, store the point at infinity at each element of the output buffer
        // TODO: benchmark whether initiating the paf values in the shader code is more efficient
        // since copying data from CPU to GPU is inefficient
        if (i === 0) {
            const paf_points_mont: BigIntPoint[] = []
            for (let j = 0; j < max_col_idx + 1; j ++) {
                paf_points_mont.push({
                    x: BigInt(0),
                    y: r,
                    t: BigInt(0),
                    z: r,
                })
            }
            const paf_bytes = points_to_u8s_for_gpu(paf_points_mont, num_words, word_size)
            device.queue.writeBuffer(output_storage_buffer, 0, paf_bytes);
        }
        */

    // Create bind group layout
    const bindGroupLayout = device.createBindGroupLayout({
      entries: [
        {
          binding: 0,
          visibility: GPUShaderStage.COMPUTE,
          buffer: { type: "storage" },
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
          buffer: { type: "storage" },
        },
      ],
    });

    // Create bind group
    const bindGroup = create_bind_group(device, bindGroupLayout, [
      output_storage_buffer,
      col_idx_storage_buffer,
      row_ptr_storage_buffer,
      points_storage_buffer,
      loop_index_storage_buffer,
    ]);

    // Create pipeline
    const computePipeline = device.createComputePipeline({
      layout: device.createPipelineLayout({
        bindGroupLayouts: [bindGroupLayout],
      }),
      compute: {
        module: shaderModule,
        entryPoint: "main",
      },
    });

    // Copy previous result buffer to input buffer
    if (previous_output_storage_buffer) {
      commandEncoder.copyBufferToBuffer(
        previous_output_storage_buffer, // source
        0, // sourceOffset
        output_storage_buffer, // destination
        0, // destinationOffset
        output_buffer_length, // size
      );
    }

    // Initiate compute pass
    const passEncoder = commandEncoder.beginComputePass();

    // Issue commands
    passEncoder.setPipeline(computePipeline);
    passEncoder.setBindGroup(0, bindGroup);
    passEncoder.dispatchWorkgroups(num_x_workgroups);

    // End the render pass
    passEncoder.end();

    previous_output_storage_buffer = output_storage_buffer;
  }

  // Create buffer to read result
  const stagingBuffer = device.createBuffer({
    size: output_buffer_length,
    usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST,
  });

  if (previous_output_storage_buffer != null) {
    commandEncoder.copyBufferToBuffer(
      previous_output_storage_buffer, // source
      0, // sourceOffset
      stagingBuffer, // destination
      0, // destinationOffset
      output_buffer_length,
    );
  }

  // 8: End frame by passing array of command buffers to command queue for execution
  device.queue.submit([commandEncoder.finish()]);

  // map staging buffer to read results back to JS
  await stagingBuffer.mapAsync(
    GPUMapMode.READ,
    0, // Offset
    output_buffer_length,
  );

  const copyArrayBuffer = stagingBuffer.getMappedRange(0, output_buffer_length);
  const data = copyArrayBuffer.slice(0);
  stagingBuffer.unmap();
  gpu_elapsed = Date.now() - start;

  console.log(`GPU took ${gpu_elapsed}ms`);

  const data_as_uint8s = new Uint8Array(data);

  const output_points = u8s_to_points(data_as_uint8s, num_words, word_size);
  console.log(output_points[0]);

  // convert output_points out of Montgomery coords
  const output_points_non_mont: ExtPointType[] = [];
  for (const pt of output_points) {
    const non = {
      x: fieldMath.Fp.mul(pt.x, rinv),
      y: fieldMath.Fp.mul(pt.y, rinv),
      t: fieldMath.Fp.mul(pt.t, rinv),
      z: fieldMath.Fp.mul(pt.z, rinv),
    };
    output_points_non_mont.push(bigIntPointToExtPointType(non, fieldMath));
  }

  // convert output_points_non_mont into affine
  const output_points_non_mont_and_affine = output_points_non_mont.map((x) =>
    x.toAffine(),
  );
  console.log("points from gpu, in affine:", output_points_non_mont_and_affine);

  if (run_on_cpu && smtvp_result_affine) {
    for (let i = 0; i < smtvp_result_affine.length; i++) {
      assert(
        smtvp_result_affine[i].x === output_points_non_mont_and_affine[i].x,
      );
      assert(
        smtvp_result_affine[i].y === output_points_non_mont_and_affine[i].y,
      );
    }
  }

  return { cpu_elapsed, gpu_elapsed };
};

const setup_shader_code = (
  p: bigint,
  r: bigint,
  d: bigint,
  num_words: number,
  word_size: number,
  n0: bigint,
) => {
  const p_limbs = gen_p_limbs(p, num_words, word_size);
  const r_limbs = gen_r_limbs(r, num_words, word_size);
  const d_limbs = gen_d_limbs(d, num_words, word_size);
  const shaderCode = mustache.render(
    smtvp_shader,
    {
      word_size,
      num_words,
      n0,
      p_limbs,
      r_limbs,
      d_limbs,
      mask: BigInt(2) ** BigInt(word_size) - BigInt(1),
      two_pow_word_size: BigInt(2) ** BigInt(word_size),
    },
    {
      structs,
      bigint_functions,
      montgomery_product_funcs,
      field_functions,
      curve_functions,
    },
  );

  return shaderCode;
};
