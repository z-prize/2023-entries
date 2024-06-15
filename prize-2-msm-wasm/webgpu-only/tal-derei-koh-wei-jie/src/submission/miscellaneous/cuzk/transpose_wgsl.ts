import mustache from "mustache";
import {
  u8s_to_numbers_32,
  numbers_to_u8s_for_gpu,
} from "../../implementation/cuzk/utils";
import {
  get_device,
  create_and_write_sb,
  create_bind_group,
  create_bind_group_layout,
  create_compute_pipeline,
  create_sb,
  read_from_gpu,
} from "../../implementation/cuzk/gpu";
import { cpu_transpose_classic } from "../transpose";
import assert from "assert";
import seedrandom from "seedrandom";

// WGSL implementation of Sparse-Matrix Transpose
export const transpose_wgsl = async (): Promise<{ x: bigint; y: bigint }> => {
  console.log("Starting WGSL Sparse-Matrix Transpose!");

  const result = await transpose();
  return result;
};

const transpose_dm = (
  num_cols: number, // of the original matrix
  num_rows: number, // of the original matrix
  data: number[],
) => {
  assert(data.length === num_cols * num_rows);
  const result = Array(data.length).fill(0);

  for (let i = 0; i < num_rows; i++) {
    for (let j = 0; j < num_cols; j++) {
      result[j * num_rows + i] = data[i * num_cols + j];
    }
  }

  return result;
};

const rng = seedrandom("hello");
const rand = () => {
  return rng();
};

const gen_random_dm = (num_cols: number, num_rows: number) => {
  const dense_size = num_cols * num_rows;
  const all_data = Array(dense_size).fill(0);
  for (let i = 0; i < dense_size; i++) {
    // Make the matrix sparse by only populating each cell with a nonzero
    // element half of the time on average
    if (rand() < 0.5) {
      all_data[i] = Math.ceil(rand() * 10);
    }
  }
  return all_data;
};

const gen_csr = (num_cols: number, num_rows: number, all_data: number[]) => {
  const csr_row_ptr = Array(num_rows + 1).fill(0);

  for (let i = 0; i < num_rows; i++) {
    let row_ct = 0;
    for (let j = 0; j < num_cols; j++) {
      const idx = i * num_cols + j;
      if (all_data[idx] !== 0) {
        row_ct++;
      }
    }
    csr_row_ptr[i + 1] = row_ct;
  }
  for (let i = 1; i < csr_row_ptr.length; i++) {
    csr_row_ptr[i] += csr_row_ptr[i - 1];
  }

  let nnz = 0;
  for (let i = 0; i < all_data.length; i++) {
    if (all_data[i] !== 0) {
      nnz++;
    }
  }

  const csr_col_idx = Array(nnz).fill(0);
  const csr_vals = Array(nnz).fill(0);

  let j = 0;
  for (let i = 0; i < all_data.length; i++) {
    if (all_data[i] !== 0) {
      csr_col_idx[j] = i % num_cols;
      csr_vals[j] = all_data[i];
      j++;
    }
  }

  return {
    csr_row_ptr,
    csr_col_idx,
    csr_vals,
  };
};

export async function transpose(): Promise<{ x: bigint; y: bigint }> {
  console.log(
    "warning: this is using a slightly outdated shader as our cuZK implementation arranges the elements in rows in a different fashion",
  );
  const num_rows = 16;
  const num_cols = 2 ** 20 / num_rows;

  const rand_dm = gen_random_dm(num_cols, num_rows);
  const rand_csr = gen_csr(num_cols, num_rows, rand_dm);
  const { csr_row_ptr, csr_col_idx } = rand_csr;

  const { csc_col_ptr, csc_row_idx } = cpu_transpose_classic(
    csr_row_ptr,
    csr_col_idx,
    num_cols,
  );

  const transposed_dm = transpose_dm(num_cols, num_rows, rand_dm);
  const transposed_csr = gen_csr(num_rows, num_cols, transposed_dm);

  assert(csc_col_ptr.toString() === transposed_csr.csr_row_ptr.toString());
  assert(csc_row_idx.toString() === transposed_csr.csr_col_idx.toString());

  // Check the sparse vals
  const dm_sparse_vals = [];
  for (let i = 0; i < rand_dm.length; i++) {
    if (rand_dm[i] !== 0) {
      dm_sparse_vals.push(rand_dm[i]);
    }
  }

  const transposed_dm_sparse_vals = [];
  for (let i = 0; i < transposed_dm.length; i++) {
    if (transposed_dm[i] !== 0) {
      transposed_dm_sparse_vals.push(transposed_dm[i]);
    }
  }

  const device = await get_device();
  const commandEncoder = device.createCommandEncoder();

  const csr_row_ptr_bytes = numbers_to_u8s_for_gpu(csr_row_ptr);
  const csr_row_ptr_sb = create_and_write_sb(device, csr_row_ptr_bytes);

  const csr_col_idx_bytes = numbers_to_u8s_for_gpu(csr_col_idx);
  const csr_col_idx_sb = create_and_write_sb(device, csr_col_idx_bytes);

  const csc_col_ptr_sb = create_sb(device, (num_cols + 1) * 4);
  const csc_row_idx_sb = create_sb(device, csr_col_idx.length * 4);
  const csc_val_idxs_sb = create_sb(device, csr_col_idx.length * 4);

  const start_gpu = Date.now();
  await transpose_serial_gpu(
    device,
    commandEncoder,
    num_cols,
    num_rows,
    csr_row_ptr_sb,
    csr_col_idx_sb,
    csc_col_ptr_sb,
    csc_row_idx_sb,
    csc_val_idxs_sb,
  );
  const elapsed_gpu = Date.now() - start_gpu;
  console.log(
    `transpose_serial_gpu took ${elapsed_gpu}ms on ${num_cols} x ${num_rows} elements`,
  );

  const data = await read_from_gpu(device, commandEncoder, [
    csc_col_ptr_sb,
    csc_row_idx_sb,
    csc_val_idxs_sb,
  ]);

  const csc_col_ptr_result = u8s_to_numbers_32(data[0]);
  const csc_row_idx_result = u8s_to_numbers_32(data[1]);
  const csc_val_idxs_result = u8s_to_numbers_32(data[2]);

  // Check the transposed col_ptr and row_idx
  for (let i = 0; i < csc_col_ptr_result.length; i++) {
    if (csc_col_ptr_result[i] !== csc_col_ptr[i]) {
      console.log("csc_col_ptr mismatch at", i);
      console.log("cpu:", csc_col_ptr);
      console.log("gpu:", csc_col_ptr_result);
      break;
    }
  }

  for (let i = 0; i < csc_row_idx_result.length; i++) {
    if (csc_row_idx_result[i] !== csc_row_idx[i]) {
      console.log("csc_row_idx mismatch at", i);
      break;
    }
  }

  assert(csc_col_ptr_result.toString() === csc_col_ptr.toString());
  assert(csc_row_idx_result.toString() === csc_row_idx.toString());

  // Check the new indices
  for (let i = 0; i < csc_val_idxs_result.length; i++) {
    assert(
      dm_sparse_vals[csc_val_idxs_result[i]] === transposed_dm_sparse_vals[i],
    );
  }

  device.destroy();

  return { x: BigInt(0), y: BigInt(0) };
}

const transpose_serial_gpu = async (
  device: GPUDevice,
  commandEncoder: GPUCommandEncoder,
  num_cols: number,
  num_rows: number,
  csr_row_ptr_sb: GPUBuffer,
  csr_col_idx_sb: GPUBuffer,

  csc_col_ptr_sb: GPUBuffer,
  csc_row_idx_sb: GPUBuffer,
  csc_val_idxs_sb: GPUBuffer,
) => {
  const curr_sb = create_sb(device, csc_col_ptr_sb.size - 4);
  const bindGroupLayout = create_bind_group_layout(device, [
    "read-only-storage",
    "read-only-storage",
    "storage",
    "storage",
    "storage",
    "storage",
  ]);

  const bindGroup = create_bind_group(device, bindGroupLayout, [
    csr_row_ptr_sb,
    csr_col_idx_sb,
    csc_col_ptr_sb,
    csc_row_idx_sb,
    csc_val_idxs_sb,
    curr_sb,
  ]);

  const num_x_workgroups = 1;
  const num_y_workgroups = 1;

  // Serial transpose algo from
  // https://synergy.cs.vt.edu/pubs/papers/wang-transposition-ics16.pdf
  const shaderCode = mustache.render(transpose_serial_shader, { num_cols }, {});
  const computePipeline = await create_compute_pipeline(
    device,
    [bindGroupLayout],
    shaderCode,
    "main",
  );

  const passEncoder = commandEncoder.beginComputePass();
  passEncoder.setPipeline(computePipeline);
  passEncoder.setBindGroup(0, bindGroup);
  passEncoder.dispatchWorkgroups(num_x_workgroups, num_y_workgroups, 1);
  passEncoder.end();
};

const transpose_serial_shader = `
// Input buffers
@group(0) @binding(0)
var<storage, read> csr_row_ptr: array<u32>;
@group(0) @binding(1)
var<storage, read> csr_col_idx: array<u32>;

// Output buffers
@group(0) @binding(2)
var<storage, read_write> csc_col_ptr: array<u32>;
@group(0) @binding(3)
var<storage, read_write> csc_row_idx: array<u32>;
@group(0) @binding(4)
var<storage, read_write> csc_val_idxs: array<u32>;

// Intermediate buffer
@group(0) @binding(5)
var<storage, read_write> curr: array<u32>;

@compute
@workgroup_size(1)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {    
    // Serial transpose algo from
    // https://synergy.cs.vt.edu/pubs/papers/wang-transposition-ics16.pdf

    // Number of rows
    let m = arrayLength(&csr_row_ptr) - 1u;

    // Number of columns
    let n = {{ num_cols }}u;

    for (var i = 0u; i < m; i ++) {
        for (var j = csr_row_ptr[i]; j < csr_row_ptr[i + 1u]; j ++) {
            csc_col_ptr[csr_col_idx[j] + 1u] += 1u;
        }
    }

    // Prefix sum, aka cumulative/incremental sum
    for (var i = 1u; i < n + 1u; i ++) {
        csc_col_ptr[i] += csc_col_ptr[i - 1u];
        storageBarrier();
    }

    var val = 0u;
    for (var i = 0u; i < m; i ++) {
        for (var j = csr_row_ptr[i]; j < csr_row_ptr[i + 1u]; j ++) {
            let loc = csc_col_ptr[csr_col_idx[j]] + curr[csr_col_idx[j]];
            csc_row_idx[loc] = i;
            curr[csr_col_idx[j]] ++;
            csc_val_idxs[loc] = val;
            val ++;
        }
    }
}
`;
