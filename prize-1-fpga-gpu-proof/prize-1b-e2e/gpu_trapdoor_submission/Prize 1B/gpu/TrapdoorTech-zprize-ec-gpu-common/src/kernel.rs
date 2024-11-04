use std::collections::HashMap;

pub use crypto_cuda::kernel::CudaKernel;
/// Creating kernels that can by statically initialized
use crypto_cuda::*;

use crate::{
    get_kernel_name, Context, CudaFunction, Device, GPUResult, Module,
    GPU_CUDA_DEVICES,
};

#[derive(Clone)]
pub struct GPUSourceCore {
    pub context: Context,
    pub device: Device,
    pub module: Module,
    pub dev_idx: usize,
    pub kernel_func_map: HashMap<String, CudaFunction>,
}

impl GPUSourceCore {
    pub fn get_kernel_func_list() -> Vec<&'static str> {
        vec![
            "Fr_poly_add_assign_scaled",
            "Fr_poly_add_assign",
            "Fr_poly_add_at_offset",
            "Fr_poly_add_constant",
            "Fr_poly_batch_inversion_part_1",
            "Fr_poly_batch_inversion_part_2",
            "Fr_poly_copy_from_offset_to",
            "Fr_poly_copy_from_to_offset",
            "Fr_poly_distribute_powers",
            "Fr_poly_evaluate_at",
            "Fr_poly_generate_powers",
            "Fr_poly_mul_assign",
            "Fr_poly_negate",
            "Fr_poly_scale",
            "Fr_poly_set_fe",
            "Fr_poly_sub_assign_scaled",
            "Fr_poly_sub_assign",
            "Fr_poly_sub_constant",
            "Fr_radix_fft",
            "Fr_poly_unmont",
            "Fr_poly_mont",
            "G1_bellman_multiexp",
            "G1_bellman_multiexp_precalc",
            "G1_multiexp_group_acc_iter",
            "G1_multiexp_ed_neg_one_a_precalc",
            "G1_multiexp_ed_neg_one_a_precalc_naf",
            "G1_multiexp_ed_neg_one_a_group_acc_iter",
            "G1_group_acc",
            "Fr_poly_square",
            "left_shift_kernel",
            "initialize_buckets_kernel",
            "compute_bucket_indexes_kernel",
            "aggregate_buckets_kernel",
            "reduce_buckets_kernel",
            "last_pass_gather_kernel",
            "split_windows_kernel_generic",
            "reduce_buckets_kernel_to_jacobian",
            "compute_top_win_bucket_indexes_kernel",
            "product_argument",
            "product_z_part1",
            "product_z_part2",
            "product_z_part3",
            "wires_to_single_gate",
        ]
    }
    /// create a new cuda context for every GPUSourceCore creation
    pub fn create_cuda(dev_idx: usize) -> GPUResult<Self> {
        cuda_init()?;

        let device_list = &GPU_CUDA_DEVICES;

        let context = Context::new(device_list[dev_idx])?;

        let module = Module::new(&get_kernel_name()?)?;
        let list = GPUSourceCore::get_kernel_func_list();
        let mut kernel_func_map = HashMap::new();
        for func_name in list.iter() {
            let func_name = func_name.to_owned();
            let func = module.get_func(func_name).unwrap();

            kernel_func_map
                .insert(func_name.to_owned(), CudaFunction { inner: func });
        }

        Ok(Self {
            context: context,
            device: device_list[dev_idx],
            module,
            dev_idx,
            kernel_func_map,
        })
    }

    pub fn get_context(&self) -> &Context {
        &self.context
    }

    pub fn get_device(&self) -> &Device {
        &self.device
    }

    pub fn get_module(&self) -> &Module {
        &self.module
    }
}
