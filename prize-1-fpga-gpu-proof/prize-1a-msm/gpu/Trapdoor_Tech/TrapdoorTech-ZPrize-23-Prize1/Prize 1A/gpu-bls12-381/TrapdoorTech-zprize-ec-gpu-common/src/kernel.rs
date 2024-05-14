use std::collections::HashMap;

pub use crypto_cuda::kernel::CudaKernel;
/// Creating kernels that can by statically initialized
use crypto_cuda::*;

use crate::{get_kernel_name, Context, Device, GPUResult, Module, GPU_CUDA_DEVICES, CudaFunction};

#[derive(Clone)]
pub struct GPUSourceCore {
    pub context: Context,
    pub device: Device,
    pub module: Module,
    pub dev_idx: usize,
    pub kernel_func_map: HashMap<String, CudaFunction>,
}

impl GPUSourceCore {
    pub fn get_kernel_func_list() ->Vec<&'static str>{
        vec![
            "left_shift_kernel",
            "initialize_buckets_kernel",
            "compute_bucket_indexes_kernel",
            "aggregate_buckets_kernel",
            "reduce_buckets_kernel",
            "last_pass_gather_kernel", 
            "split_windows_kernel_generic",
            "reduce_buckets_kernel_to_jacobian",
            "compute_top_win_bucket_indexes_kernel",
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

            kernel_func_map.insert(func_name.to_owned(), CudaFunction { inner: func });
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
