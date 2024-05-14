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
use std::path::PathBuf;
use std::sync::Arc;
use rayon::prelude::*;
use crate::params::*;
use crate::twisted_edwards::*;

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
    pub fn setup_ed_acc_points<P>(
        &mut self,
        gpu_op: GPU_OP,
        acc_points: &Vec<P>,
        window_size: usize,
    ) -> GPUResult<()> {
        let acc_points = unsafe { std::mem::transmute::<_, &Vec<G::Projective>>(acc_points) };

        match gpu_op {
            GPU_OP::REUSE_G => {
                self.ed_acc_points_g = (*acc_points).clone();
            }
            GPU_OP::REUSE_SHIFTED_G => {
                self.ed_acc_points_shifted_g = (*acc_points).clone();
            }
            GPU_OP::REUSE_LAGRANGE_G => {
                self.ed_acc_points_lagrange_g = (*acc_points).clone();
            }
            GPU_OP::REUSE_SHIFTED_LAGRANGE_G => {
                self.ed_acc_points_shifted_lagrange_g = (*acc_points).clone();
            }
            _ => panic!("invalid gpu_op"),
        }

        self.window_size = window_size;

        Ok(())
    }

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

    /// naive multiexp, use pippenger algorithm
    pub fn multiexp<T, TS>(
        &mut self,
        bases: &[T],
        exps: &[TS],
        start: usize,
        n: usize,
        gpu_op: GPU_OP,
        is_mont_form: bool,
    ) -> GPUResult<G::Projective> {
        let bases = unsafe { std::mem::transmute::<_, &[G]>(bases) };
        let exps = unsafe { std::mem::transmute::<_, &[G::ScalarField]>(exps) };

        let exp_bits = std::mem::size_of::<G::ScalarField>() * 8;

        // `window_size` is designated by `bucket_size`
        let window_size = self.max_bucket_size;

        let num_windows = ((exp_bits as f64) / (window_size as f64)).ceil() as usize;
        let num_groups = std::cmp::min(n, calc_num_groups(self.core_count / 2, num_windows));

        let texps = unsafe { std::mem::transmute::<&[G::ScalarField], &[GpuFr]>(exps) };

        let mut total = num_windows * num_groups;
        total += (LOCAL_WORK_SIZE - (total % LOCAL_WORK_SIZE)) % LOCAL_WORK_SIZE;

        let (gws, lws) = calc_cuda_wg_threads(total);

        // prepare exps in non-montgomery form
        let exp_buf = self.exp_buffer.get_inner();
        self.exp_buffer
            .read_from_async(texps, exps.len(), &self.stream)?;

        if is_mont_form {
            let (gws, lws) = calc_cuda_wg_threads(n);
            let f = self.kernel_func_map.get("Fr_poly_unmont").unwrap().inner;

            let params = make_params!(exp_buf, n as u32);
            let kern = create_kernel_with_params!(f, <<<gws, lws, 0>>>(params));

            self.stream.launch(&kern)?;
        }

        if gpu_op == GPU_OP::LOAD_BASE {
            // GpuAffine does not have `inf` field, so we need to do a transform `G` -> `GpuAffine`
            let mut tbases = vec![GpuAffine::default(); bases.len()];
            for i in 0..bases.len() {
                // make sure z = 1
                let tmp_projective = bases[i].into_projective();

                let &tmp = unsafe { std::mem::transmute::<_, &GpuProjective>(&tmp_projective) };
                tbases[i].x = tmp.x;
                tbases[i].y = tmp.y;
            }

            self.tmp_base
                .read_from_async(&tbases, bases.len(), &self.stream)?;
        }

        let base_buf = match gpu_op {
            GPU_OP::LOAD_BASE => self.tmp_base.get_inner(),
            _ => panic!("invalid gpu_op"),
        };

        let base_start = if gpu_op == GPU_OP::LOAD_BASE {
            0 as u32
        } else {
            start as u32
        };

        let buck_buf = self.bucket_buffer.get_inner();
        let res_buf = self.result_buffer.get_inner();
        let exp_buf = self.exp_buffer.get_inner();

        let kernel_name = format!("{}_bellman_multiexp", self.group_name);
        let f = self.kernel_func_map.get(&kernel_name).unwrap().inner;

        let params = make_params!(
            base_buf,
            buck_buf,
            res_buf,
            exp_buf,
            base_start as u32,
            n as u32,
            num_groups as u32,
            num_windows as u32,
            window_size as u32
        );
        let kern = create_kernel_with_params!(f, <<<gws, lws, 0>>>(params));

        self.stream.launch(&kern)?;

        let (gws, lws) = (num_windows, CUDA_LWS);

        let kernel_name = format!("{}_group_acc", self.group_name);
        let f = self.kernel_func_map.get(&kernel_name).unwrap().inner;

        let params = make_params!(res_buf, num_groups as u32, num_windows as u32, lws as u32);
        let kern = create_kernel_with_params!(f, <<<gws as u32, lws as u32, 0>>>(params));

        self.stream.launch(&kern)?;

        let mut res = vec![G::Projective::zero(); num_windows];

        let tres =
            unsafe { &mut *(res.as_mut_slice() as *mut [G::Projective] as *mut [GpuProjective]) };

        self.result_buffer
            .write_to_async(tres, res.len(), &self.stream)?;

        self.stream.sync()?;

        // Using the algorithm below, we can calculate the final result by accumulating the results
        // of those `NUM_GROUPS` * `NUM_WINDOWS` threads.
        let mut acc = G::Projective::zero();
        let mut bits = 0;
        for i in 0..num_windows {
            let w = std::cmp::min(window_size, exp_bits - bits);
            for _ in 0..w {
                acc.double_in_place();
            }

            acc += &res[i];
            bits += w; // Process the next window
        }

        Ok(acc)
    }

    pub fn sms_count(&self) -> usize {
        self.sms_count
    }
}
