import {
  create_and_write_ub,
  create_bind_group,
  create_bind_group_layout,
  create_compute_pipeline,
  execute_pipeline,
} from "../implementation/cuzk/gpu";
import { numbers_to_u8s_for_gpu } from "../implementation/cuzk/utils";

export const shader_invocation = async (
  device: GPUDevice,
  commandEncoder: GPUCommandEncoder,
  shaderCode: string,
  x_coords_sb: GPUBuffer,
  y_coords_sb: GPUBuffer,
  t_coords_sb: GPUBuffer,
  z_coords_sb: GPUBuffer,
  out_x_sb: GPUBuffer,
  out_y_sb: GPUBuffer,
  out_t_sb: GPUBuffer,
  out_z_sb: GPUBuffer,
  num_buckets: number,
  num_words: number,
  num_subtasks: number,
) => {
  const compute_ideal_num_workgroups = (num_buckets: number) => {
    const m = Math.log2(num_buckets);
    const n = Math.ceil(m / 2);

    const num_x_workgroups = 2 ** n;
    const num_y_workgroups = Math.ceil(
      num_buckets / num_x_workgroups / num_subtasks / 2,
    );

    return { num_x_workgroups, num_y_workgroups };
  };

  const { num_x_workgroups, num_y_workgroups } =
    compute_ideal_num_workgroups(num_buckets);

  const num_z_workgroups = 1;

  const params_bytes = numbers_to_u8s_for_gpu([
    num_y_workgroups,
    num_z_workgroups,
  ]);
  const params_ub = create_and_write_ub(device, params_bytes);

  const bindGroupLayout = create_bind_group_layout(device, [
    "read-only-storage",
    "read-only-storage",
    "read-only-storage",
    "read-only-storage",
    "storage",
    "storage",
    "storage",
    "storage",
    "uniform",
  ]);
  const bindGroup = create_bind_group(device, bindGroupLayout, [
    x_coords_sb,
    y_coords_sb,
    t_coords_sb,
    z_coords_sb,
    out_x_sb,
    out_y_sb,
    out_t_sb,
    out_z_sb,
    params_ub,
  ]);

  const computePipeline = await create_compute_pipeline(
    device,
    [bindGroupLayout],
    shaderCode,
    "main",
  );

  //console.log(num_buckets)
  //console.log(num_x_workgroups, num_y_workgroups, num_z_workgroups)
  execute_pipeline(
    commandEncoder,
    computePipeline,
    bindGroup,
    num_x_workgroups,
    num_y_workgroups,
    num_z_workgroups,
  );

  const size = Math.ceil(num_buckets / 2) * 4 * num_words * num_subtasks;
  commandEncoder.copyBufferToBuffer(out_x_sb, 0, x_coords_sb, 0, size);
  commandEncoder.copyBufferToBuffer(out_y_sb, 0, y_coords_sb, 0, size);
  commandEncoder.copyBufferToBuffer(out_t_sb, 0, t_coords_sb, 0, size);
  commandEncoder.copyBufferToBuffer(out_z_sb, 0, z_coords_sb, 0, size);

  return {
    out_x_sb,
    out_y_sb,
    out_t_sb,
    out_z_sb,
    params_ub,
  };
};
