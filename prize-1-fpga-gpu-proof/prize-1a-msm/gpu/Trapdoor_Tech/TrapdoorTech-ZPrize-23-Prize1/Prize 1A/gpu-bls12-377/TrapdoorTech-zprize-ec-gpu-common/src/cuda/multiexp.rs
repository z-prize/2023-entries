use crate::errors::{GPUError, GPUResult};
use crate::structs::{
    AffineCurve, FromBytes, G1Affine, G1Projective, G2Affine, GpuAffine, GpuFr, GpuProjective,
    ProjectiveCurve, Zero, GPU_OP,
};
use crate::{utils::*, CudaFunction};
use crate::{GPUSourceCore, Stream, GPU_CUDA_CORES};
use crate::{GpuEdAffine, GpuEdProjective};

use crypto_cuda::DeviceMemory;
use ark_std::io::{BufReader, Read};
use std::any::TypeId;
use std::collections::HashMap;
use std::fs::File;
use std::marker::PhantomData;
use std::sync::Arc;

use crate::params::*;

pub struct MsmPrecalcContainer {
    // for Twisted Edwards Extended Curves
    ed_powers_of_g_precalc: DeviceMemory<GpuEdAffine>,
    ed_shifted_powers_of_g_precalc: DeviceMemory<GpuEdAffine>,
    ed_lagrange_g_precalc: DeviceMemory<GpuEdAffine>,
    ed_shifted_lagrange_g_precalc: DeviceMemory<GpuEdAffine>,

    window_size: usize,
}

impl MsmPrecalcContainer {
    pub fn create(dev_idx: usize, is_matter: bool) -> GPUResult<Self> {
        let source_core = &GPU_CUDA_CORES[dev_idx];

        let container = MsmPrecalcContainer::create_with_core(source_core, is_matter)?;

        Ok(container)
    }

    pub fn create_with_core(core: &GPUSourceCore, is_matter: bool) -> GPUResult<Self> {
        let core_count = core.device.get_cores()?;
        let mem = core.device.get_memory()?;

        let max_window_size = calc_window_size::<G1Affine>(MAX_SRS_G_LEN, mem, core_count);
        let max_window_num = 1 << max_window_size;

        let len = if is_matter {
            1
        } else {
            MAX_SRS_G_LEN * max_window_num
        };

        // for edwards curves
        let ed_powers_of_g_precalc = DeviceMemory::<GpuEdAffine>::new(&core.context, len)?;
        let ed_shifted_powers_of_g_precalc = DeviceMemory::<GpuEdAffine>::new(&core.context, len)?;
        let ed_lagrange_g_precalc = DeviceMemory::<GpuEdAffine>::new(&core.context, len)?;
        let ed_shifted_lagrange_g_precalc = DeviceMemory::<GpuEdAffine>::new(&core.context, len)?;

        Ok(MsmPrecalcContainer {
            ed_powers_of_g_precalc,
            ed_shifted_powers_of_g_precalc,
            ed_lagrange_g_precalc,
            ed_shifted_lagrange_g_precalc,

            window_size: max_window_size,
        })
    }

    pub fn get_ed_powers_of_g_precalc(&self) -> &DeviceMemory<GpuEdAffine> {
        &self.ed_powers_of_g_precalc
    }

    pub fn get_ed_shifted_powers_of_g_precalc(&self) -> &DeviceMemory<GpuEdAffine> {
        &self.ed_shifted_powers_of_g_precalc
    }

    pub fn get_ed_lagrange_g_precalc(&self) -> &DeviceMemory<GpuEdAffine> {
        &self.ed_lagrange_g_precalc
    }

    pub fn get_ed_shifted_lagrange_g_precalc(&self) -> &DeviceMemory<GpuEdAffine> {
        &self.ed_shifted_lagrange_g_precalc
    }

    pub fn get_window_size(&self) -> usize {
        self.window_size
    }
}

// Multiexp kernel for a single GPU
pub struct MultiexpKernel<'a, 'b, G>
where
    G: AffineCurve,
{
    pub core: &'a GPUSourceCore,
    pub stream: Stream,
    pub kernel_func_map: Arc<&'a HashMap<String, CudaFunction>>,
    precalc_container: Arc<&'b MsmPrecalcContainer>,

    tmp_base: DeviceMemory<GpuAffine>,
    bucket_buffer: DeviceMemory<GpuProjective>,

    result_buffer: DeviceMemory<GpuProjective>,
    exp_buffer: DeviceMemory<GpuFr>,

    ed_result: DeviceMemory<GpuEdProjective>,
    ed_result_2: DeviceMemory<GpuEdProjective>,

    ed_acc_points_g: Vec<G::Projective>,
    ed_acc_points_shifted_g: Vec<G::Projective>,
    ed_acc_points_lagrange_g: Vec<G::Projective>,
    ed_acc_points_shifted_lagrange_g: Vec<G::Projective>,

    core_count: usize,
    window_size: usize,
    max_window_size: usize,
    max_bucket_size: usize,
    sms_count: usize,

    group_name: String,

    _phantom: PhantomData<G>,
}

fn calc_num_groups(core_count: usize, num_windows: usize) -> usize {
    return CORE_N * core_count / num_windows;
}

fn calc_window_size<G>(n: usize, mem_bytes: u64, core_count: usize) -> usize
where
    G: AffineCurve,
{
    let size_affine = std::mem::size_of::<G>();
    let size_projective = std::mem::size_of::<G::Projective>();
    let size_scalar = std::mem::size_of::<G::ScalarField>();

    let size_ed_affine = std::mem::size_of::<GpuEdAffine>();
    let size_ed_projective = std::mem::size_of::<GpuEdProjective>();

    // TODO: not correct
    for i in (MIN_WINDOW_SIZE..=MAX_WINDOW_SIZE).rev() {
        let mem_needed = (n * (2i32.pow(i as u32)) as usize * size_ed_affine) * 4               // 4 ed precalc table
            + n * size_affine                                                                   // 1 bases table
            + CORE_N * core_count * (2i32.pow(MSM_MAX_BUCKET_SIZE as u32)) as usize * size_projective // buckets buffer
            + core_count * CORE_N * size_projective                                             // 1 result buffer
            + 2 * core_count * CORE_N * size_ed_projective                                      // 2 ed result buffer
            + n * size_scalar; // exp buffer

        if (mem_needed as u64) < mem_bytes {
            return i;
        }
    }

    return MIN_WINDOW_SIZE;
}

impl<'a, 'b, G> MultiexpKernel<'a, 'b, G>
where
    G: AffineCurve,
{
    pub fn create_with_core(
        core: &'a GPUSourceCore,
        container: &'b MsmPrecalcContainer,
    ) -> GPUResult<MultiexpKernel<'a, 'b, G>> {
        let kernel_func_map = Arc::new(&core.kernel_func_map);
        let precalc_container = Arc::new(container);

        let sms_count = core.device.get_multiprocessor_count()?;
        let core_count = core.device.get_cores()?;
        let mem = core.device.get_memory()?;

        let max_window_size = calc_window_size::<G>(MAX_SRS_G_LEN, mem, core_count);
        let max_bucket_size = MSM_MAX_BUCKET_SIZE;

        let n = MAX_SRS_G_LEN;
        let max_bucket_num = 1 << max_bucket_size;

        let exp_buf = DeviceMemory::<GpuFr>::new(&core.context, n)?;

        // keep these two buffers in case we need pippenger algorithm
        let tmp_base = DeviceMemory::<GpuAffine>::new(&core.context, n)?;
        let buck_buf = DeviceMemory::<GpuProjective>::new(
            &core.context,
            CORE_N * core_count * max_bucket_num,
        )?;
        let res_buf = DeviceMemory::<GpuProjective>::new(&core.context, CORE_N * core_count)?;

        let ed_result = DeviceMemory::<GpuEdProjective>::new(&core.context, CORE_N * core_count)?;
        let ed_result_2 = DeviceMemory::<GpuEdProjective>::new(&core.context, CORE_N * core_count)?;

        let ed_acc_points_g = Vec::new();
        let ed_acc_points_shifted_g = Vec::new();
        let ed_acc_points_lagrange_g = Vec::new();
        let ed_acc_points_shifted_lagrange_g = Vec::new();

        let group_name = if TypeId::of::<G>() == TypeId::of::<G1Affine>() {
            String::from("G1")
        } else if TypeId::of::<G>() == TypeId::of::<G2Affine>() {
            String::from("G2") // G2 MSM is not used
        } else {
            panic!("not supported elliptic curves!")
        };

        let stream = Stream::new_with_context(&core.context)?;

        Ok(MultiexpKernel {
            stream,
            kernel_func_map,
            precalc_container,

            tmp_base,

            bucket_buffer: buck_buf,
            result_buffer: res_buf,
            exp_buffer: exp_buf,

            ed_result,
            ed_result_2,

            ed_acc_points_g,
            ed_acc_points_shifted_g,
            ed_acc_points_lagrange_g,
            ed_acc_points_shifted_lagrange_g,

            core_count,
            window_size: max_window_size,
            max_window_size,
            max_bucket_size,
            sms_count,

            group_name,
            core,
            _phantom: PhantomData,
        })
    }

    pub fn create(
        dev_idx: usize,
        container: &'b MsmPrecalcContainer,
    ) -> GPUResult<MultiexpKernel<'a, 'b, G>> {
        let source_core = &GPU_CUDA_CORES[dev_idx];

        let kernel = MultiexpKernel::create_with_core(source_core, container)?;

        Ok(kernel)
    }

    pub fn sms_count(&self) -> usize {
        self.sms_count
    }
}
