use crate::{
    api_local, calc_cuda_wg_threads, edwards_double, edwards_from_neg_one_a,
    edwards_proj_to_affine, edwards_proj_to_gpu_affine, edwards_to_neg_one_a, edwards_to_sw,
    sw_to_edwards, GPUResult, GpuAffine, GpuEdAffine, MultiexpKernel,
};
use crate::{
    edwards_add_with_neg_one_a, edwards_affine_to_proj, edwards_double_with_neg_one_a,
    GpuEdProjective,
};
use api_local::*;
use ark_ec::AffineCurve;
use ark_ff::BigInteger256;
use ark_std::log2;
use ark_std::Zero;
use crossbeam::thread;
use crypto_cuda::{api::*, CudaStream, DeviceMemory, HostMemory};
use std::cmp::min;
use std::ffi::c_void;
use std::rc::Weak;
use std::sync::{Arc, Mutex};

#[cfg(feature = "bls12_377")]
use ark_bls12_377::{Fq, G1Affine, G1Projective};

#[cfg(feature = "bls12_381")]
use ark_bls12_381::{Fq, G1Affine, G1Projective};

#[cfg(feature = "bls12_377")]
pub const FR_MODULUS_BITS: u32 = 253;

#[cfg(feature = "bls12_381")]
pub const FR_MODULUS_BITS: u32 = 255;

#[cfg(any(feature = "bls12_381", feature = "bls12_377"))]
#[derive(Copy, Clone, Debug, PartialEq)]
pub struct G1AffineNoInfinity {
    pub x: Fq,
    pub y: Fq,
}

#[cfg(any(feature = "bls12_381", feature = "bls12_377"))]
impl G1AffineNoInfinity {
    pub fn zero() -> Self {
        G1AffineNoInfinity {
            x: Fq::zero(),
            y: Fq::zero(),
        }
    }
}

#[cfg(any(feature = "bls12_381", feature = "bls12_377"))]
impl From<&G1Affine> for G1AffineNoInfinity {
    fn from(point: &G1Affine) -> Self {
        assert!(!point.infinity);
        G1AffineNoInfinity {
            x: point.x,
            y: point.y,
        }
    }
}

macro_rules! handle_cuda_error {
    ($x:expr) => {
        if $x != 0 {
            println!("File: {}, Line: {}", file!(), line!());
            panic!("Got value {}", $x);
        }
    };
}

fn get_window_bits_count(log_scalars_count: u32) -> u32 {
    match log_scalars_count {
        11 => return 9,
        12 | 13 => return 10,
        14 | 15 | 16 | 17 => return 12,
        18 | 19 => return 14,
        20 => return 15,
        21 | 22 | 23 | 24 => return 16,
        25 | 26 => return 20,
        _ => return 8,
    }
}

fn get_log_min_inputs_count(log_scalars_count: u32) -> u32 {
    match log_scalars_count {
        18 | 19 => return 17,
        20 => return 18,
        21 | 22 | 23 | 24 => return 19,
        25 | 26 => return 22,
        _ => return log_scalars_count,
    }
}

pub fn get_windows_count(window_bits_count: u32) -> u32 {
    (FR_MODULUS_BITS - 1) / window_bits_count + 1
}

#[inline(always)]
fn log2_floor(num: usize) -> u32 {
    assert!(num > 0);

    let mut pow = 0;

    while (1 << (pow + 1)) <= num {
        pow += 1;
    }

    pow
}

fn log2_ceiling(value: u32) -> u32 {
    if value <= 1 {
        0
    } else {
        log2_floor(value as usize - 1) + 1
    }
}
fn get_optimal_log_data_split(
    mpc: u32,
    source_window_bits: u32,
    target_window_bits: u32,
    target_windows_count: u32,
) -> u32 {
    const MAX_THREADS: u32 = 32;
    const MIN_BLOCKS: u32 = 12;
    let full_occupancy = mpc * MAX_THREADS * MIN_BLOCKS;
    let target = full_occupancy << 6;
    let unit_threads_count = target_windows_count << target_window_bits;
    let split_target = log2_ceiling(target / unit_threads_count);
    let split_limit = source_window_bits - target_window_bits - 1;
    return std::cmp::min(split_target, split_limit);
}

pub struct HostFn<'a> {
    arc: Arc<Box<dyn Fn() + Send + 'a>>,
}

impl<'a> HostFn<'a> {
    pub fn new(func: impl Fn() + Send + 'a) -> Self {
        Self {
            arc: Arc::new(Box::new(func) as Box<dyn Fn() + Send>),
        }
    }
}

unsafe extern "C" fn launch_host_fn_callback(data: *mut c_void) {
    let raw = data as *const Box<dyn Fn() + Send>;
    let weak = Weak::from_raw(raw);
    if let Some(func) = weak.upgrade() {
        func();
    }
}

pub fn get_raw_fn_and_data(host_fn: &HostFn) -> (CUhostFn, *mut c_void) {
    let weak = Arc::downgrade(&host_fn.arc);
    let raw = weak.into_raw();
    let data = raw as *mut c_void;
    (Some(launch_host_fn_callback), data)
}

pub fn launch_host_fn(stream: CUstream, host_fn: &HostFn) -> CUresult {
    let (func, data) = get_raw_fn_and_data(host_fn);
    unsafe { cuLaunchHostFunc(stream, func, data) }
}

#[cfg(any(feature = "bls12_381", feature = "bls12_377"))]
pub fn reduce_results(points: &[GpuEdProjective]) -> GpuEdProjective {
    let mut sum = GpuEdProjective::default();
    points.iter().rev().for_each(|p| {
        sum = edwards_double_with_neg_one_a(sum);
        sum = edwards_add_with_neg_one_a(sum, p.clone())
    });
    sum
}

#[cfg(any(feature = "bls12_381", feature = "bls12_377"))]
pub fn reduce_top_win_results(
    points: &[GpuEdProjective],
    win_count: u32,
    win_bits: u32,
    precomputed_windows_stride: u32,
) -> GpuEdProjective {
    let target_window_index = (win_count - 1) % precomputed_windows_stride;

    let points = &points[1..];
    let mut res = GpuEdProjective::default();
    let mut running_sum = GpuEdProjective::default();
    points.iter().rev().for_each(|p| {
        running_sum = edwards_add_with_neg_one_a(running_sum, p.clone());
        res = edwards_add_with_neg_one_a(res, running_sum);
    });

    for _ in 0..target_window_index * win_bits {
        res = edwards_double_with_neg_one_a(res);
    }

    res
}

#[cfg(any(feature = "bls12_381", feature = "bls12_377"))]
pub struct MSMContext<'a, 'b, G>
where
    G: AffineCurve,
{
    bases: DeviceMemory<GpuAffine>,
    buckets_pass_one: DeviceMemory<GpuEdProjective>,
    bucket_indexes: DeviceMemory<u32>,
    base_indexes: DeviceMemory<u32>,
    sorted_bucket_indexes: DeviceMemory<u32>,
    sorted_base_indexes: DeviceMemory<u32>,
    unique_bucket_indexes: DeviceMemory<u32>,
    bucket_run_lengths: DeviceMemory<u32>,
    bucket_runs_count: DeviceMemory<u32>,
    bucket_run_offsets: DeviceMemory<u32>,
    sorted_bucket_run_lengths: DeviceMemory<u32>,
    sorted_bucket_run_offsets: DeviceMemory<u32>,
    sorted_unique_bucket_indexes: DeviceMemory<u32>,
    results: DeviceMemory<GpuEdProjective>,
    input_indexes_sort_temp_storage: DeviceMemory<u8>,
    sort_offsets_temp_storage: DeviceMemory<u8>,
    target_buckets: DeviceMemory<GpuEdProjective>,
    inputs_scalars: DeviceMemory<BigInteger256>,

    pub cpu_results: HostMemory<GpuEdProjective>, //todo pub
    pub cpu_top_win_results: HostMemory<GpuEdProjective>,
    log_count: u32,
    window_bits_count: u32,
    precomputed_windows_stride: u32,
    precomputed_bases_stride: u32,
    top_window_unused_bits: u32,
    pub msm_kern: &'a MultiexpKernel<'a, 'b, G>, //todo pub
}

impl<G: AffineCurve> MSMContext<'_, '_, G> {
    pub fn read_bases_async(&self, stream: &CudaStream, bases: &mut [GpuAffine]) {
        self.bases
            .write_to_async(bases, bases.len(), stream)
            .unwrap();
    }
    pub fn bases_size(&self) -> usize {
        self.bases.size()
    }
    pub fn get_log_count(&self) -> u32 {
        self.log_count
    }
}

#[cfg(any(feature = "bls12_381", feature = "bls12_377"))]
pub struct ExecuteConfiguration<'a, 'b, 'c, G>
where
    G: AffineCurve,
{
    pub scalars: &'a [BigInteger256],
    pub h2d_copy_finished: Option<&'a CUevent>,
    pub h2d_copy_finished_callback: CUhostFn,
    pub h2d_copy_finished_callback_data: *mut c_void,
    pub d2h_copy_finished: Option<&'a CUevent>,
    pub d2h_copy_finished_callback: CUhostFn,
    pub d2h_copy_finished_callback_data: *mut c_void,
    pub force_min_chunk_size: bool,
    pub log_min_chunk_size: u32,
    pub force_max_chunk_size: bool,
    pub log_max_chunk_size: u32,
    pub scalars_not_montgomery: bool,
    pub msm_context: &'a mut MSMContext<'b, 'c, G>,
}

#[cfg(any(feature = "bls12_381", feature = "bls12_377"))]
pub struct MsmConfiguration<'a, 'b, 'c, G>
where
    G: AffineCurve,
{
    pub scalars: &'a [BigInteger256],
    pub h2d_copy_finished: Option<&'a CUevent>,
    pub h2d_copy_finished_callback: Option<&'a HostFn<'a>>,
    pub d2h_copy_finished: Option<&'a CUevent>,
    pub d2h_copy_finished_callback: Option<&'a HostFn<'a>>,
    pub force_min_chunk_size: bool,
    pub log_min_chunk_size: u32,
    pub force_max_chunk_size: bool,
    pub log_max_chunk_size: u32,
    pub scalars_not_montgomery: bool,
    pub msm_context: &'a mut MSMContext<'b, 'c, G>,
}

// extended_configuration
#[cfg(any(feature = "bls12_381", feature = "bls12_377"))]
struct ExtendedConfiguration<'a, 'b, 'c, G>
where
    G: AffineCurve,
{
    exec_cfg: ExecuteConfiguration<'a, 'b, 'c, G>,
    log_max_inputs_count: u32,
    multiprocessor_count: u32,
}

#[cfg(any(feature = "bls12_381", feature = "bls12_377"))]
impl<'a, 'b, G> MultiexpKernel<'a, 'b, G>
where
    G: AffineCurve,
{
    fn set_kernel_attributes(&self, func: *mut CUfunc_st) {
        handle_cuda_error!(unsafe {
            cuFuncSetCacheConfig(func, CUfunc_cache_enum_CU_FUNC_CACHE_PREFER_L1)
        });

        handle_cuda_error!(unsafe {
            cuFuncSetAttribute(
                func,
                CUfunction_attribute_enum_CU_FUNC_ATTRIBUTE_PREFERRED_SHARED_MEMORY_CARVEOUT,
                CUshared_carveout_enum_CU_SHAREDMEM_CARVEOUT_MAX_L1,
            )
        });
    }

    fn set_all_kernel_attributes(&self) {
        let f = self
            .kernel_func_map
            .get("aggregate_buckets_kernel")
            .unwrap()
            .inner;
        self.set_kernel_attributes(f);
    }

    pub fn left_shift(&self, values_addr: CUdeviceptr, shift: u32, count: u32) -> GPUResult<()> {
        let (gws, lws) = calc_cuda_wg_threads((count) as usize);
        let f = self.kernel_func_map.get("left_shift_kernel").unwrap().inner;

        let params = make_params!(values_addr, shift, count);
        let kern = create_kernel_with_params!(f, <<<gws, lws, 0>>>(params));

        self.stream.launch(&kern)?;

        Ok(())
    }

    fn precompute_ed_bases(bases: &[G1Affine], stride: u32, count: usize) -> Vec<GpuAffine> {
        let ed_projective_bases =
            Arc::new(Mutex::new(vec![GpuEdProjective::default(); bases.len()]));
        let count_parallel = 1 << 6;

        thread::scope(|s| {
            assert!(count_parallel >= 1);
            assert!(bases.len() % count_parallel == 0);
            let unit_count = bases.len() / count_parallel;
            let mut thread_handles = vec![];

            for i in 0..count_parallel {
                let start = i * unit_count;
                let end = (i + 1) * unit_count;
                let ed_projective_bases: Arc<Mutex<Vec<GpuEdProjective>>> =
                    Arc::clone(&ed_projective_bases);
                let bases_slice = &bases[start..end];
                let handle = s.spawn(move |_| {
                    let mut units = vec![];
                    for a in bases_slice.iter() {
                        assert_ne!(a.clone(), G1Affine::zero());
                        let ed_base = sw_to_edwards(GpuAffine {
                            x: a.x.clone(),
                            y: a.y.clone(),
                        });
                        let ed_base = edwards_to_neg_one_a(ed_base);
                        units.push(edwards_affine_to_proj(ed_base));
                    }
                    let mut ed_projective_bases = ed_projective_bases.lock().unwrap();
                    ed_projective_bases.splice(start..end, units);
                });
                thread_handles.push(handle);
            }

            for handle in thread_handles {
                handle.join().unwrap();
            }
        })
        .unwrap();

        let ed_projective_bases = ed_projective_bases.lock().unwrap().to_vec();
        let mut precomputed_bases = ed_projective_bases.clone();
        let mut accumulator = ed_projective_bases;
        for _ in 1..count {
            let temp_accumulator = Arc::new(Mutex::new(vec![
                GpuEdProjective::default();
                accumulator.len()
            ]));
            let unit_count = accumulator.len() / count_parallel;

            thread::scope(|s| {
                let mut thread_handles = vec![];

                for i in 0..count_parallel {
                    let start = i * unit_count;
                    let end = (i + 1) * unit_count;
                    let bases_slice = &accumulator[start..end];

                    let temp_accumulator = Arc::clone(&temp_accumulator);
                    let handle = s.spawn(move |_| {
                        let mut unit = vec![GpuEdProjective::default(); unit_count];
                        for (index, &x) in bases_slice.iter().enumerate() {
                            let mut y = x.clone();
                            for _ in 0..stride {
                                y = edwards_double_with_neg_one_a(y);
                            }

                            unit.as_mut_slice()[index] = y;
                        }

                        temp_accumulator
                            .lock()
                            .unwrap()
                            .splice(start..end, unit);
                    });

                    thread_handles.push(handle);

                }
                for handle in thread_handles {
                    handle.join().unwrap();
                }
            })
            .unwrap();

            let mut temp_accumulator = temp_accumulator.lock().unwrap().to_vec();
            std::mem::swap(&mut accumulator, &mut temp_accumulator);
            precomputed_bases.extend(accumulator.iter());
        }

        let ed_affine_bases = Arc::new(Mutex::new(vec![
            GpuAffine::default();
            precomputed_bases.len()
        ]));

        thread::scope(|s| {
            assert!(count_parallel >= 1);
            assert!(precomputed_bases.len() % count_parallel == 0);
            let unit_count = precomputed_bases.len() / count_parallel;
            let mut thread_handles = vec![];

            for i in 0..count_parallel {
                let start = i * unit_count;
                let end = (i + 1) * unit_count;
                let ed_affine_bases: Arc<Mutex<Vec<GpuAffine>>> = Arc::clone(&ed_affine_bases);
                let bases_slice = &precomputed_bases[start..end];
                let handle = s.spawn(move |_| {
                    let mut units = vec![];
                    for a in bases_slice.iter() {
                        units.push(edwards_proj_to_gpu_affine(a.clone()));
                    }
                    let mut ed_affine_bases = ed_affine_bases.lock().unwrap();
                    ed_affine_bases.splice(start..end, units);
                });

                thread_handles.push(handle);
            }
            for handle in thread_handles {
                handle.join().unwrap();
            }
        })
        .unwrap();

        let ret = ed_affine_bases.lock().unwrap().to_vec();
        ret
    }

    // todo G1Affine  ==>T
    pub fn multi_scalar_mult_init(&'a self, bases: &[G1Affine]) -> MSMContext<'a, 'b, G> {
        self.set_all_kernel_attributes();
        const WINDOW_BITS_COUNT: u32 = 23;
        const PRECOMPUTE_FACTOR: u32 = 4;
        const WINDOWS_COUNT: u32 = (FR_MODULUS_BITS - 1) / WINDOW_BITS_COUNT + 1;
        const PRECOMPUTE_WINDOWS_STRIDE: u32 = (WINDOWS_COUNT - 1) / PRECOMPUTE_FACTOR + 1;
        let result_bits_count: u32 = WINDOW_BITS_COUNT * PRECOMPUTE_WINDOWS_STRIDE;
        let precompute_factor: usize = PRECOMPUTE_FACTOR as usize;

        let bits_count_pass_one = WINDOW_BITS_COUNT;
        let mut total_windows_count = get_windows_count(bits_count_pass_one);
        let top_window_unused_bits = total_windows_count * bits_count_pass_one - FR_MODULUS_BITS;
        if top_window_unused_bits > 0 {
            total_windows_count -= 1;
        }
        let top_win_used_bits = bits_count_pass_one - top_window_unused_bits;

        assert!(WINDOW_BITS_COUNT == 23);

        let count = bases.len();
        let log_count = log2(count);
        assert_eq!(count, 1 << log_count);
        let ctx = self.core.get_context();

        //new for sort
        let windows_count_pass_one = PRECOMPUTE_WINDOWS_STRIDE;
        let buckets_count_pass_one = (windows_count_pass_one << bits_count_pass_one) as usize;
        let buckets_pass_one =
            DeviceMemory::<GpuEdProjective>::new(ctx, buckets_count_pass_one.try_into().unwrap())
                .unwrap();

        let inputs_scalars = DeviceMemory::<BigInteger256>::new(ctx, count).unwrap();

        let input_indexes_count = total_windows_count << log_count;
        let bucket_indexes =
            DeviceMemory::<u32>::new(ctx, input_indexes_count.try_into().unwrap()).unwrap();
        let base_indexes =
            DeviceMemory::<u32>::new(ctx, input_indexes_count.try_into().unwrap()).unwrap();
        let sorted_bucket_indexes =
            DeviceMemory::<u32>::new(ctx, input_indexes_count.try_into().unwrap()).unwrap();

        let sorted_base_indexes =
            DeviceMemory::<u32>::new(ctx, input_indexes_count.try_into().unwrap()).unwrap();

        let unique_bucket_indexes =
            DeviceMemory::<u32>::new(ctx, buckets_count_pass_one.try_into().unwrap()).unwrap();
        let bucket_run_lengths =
            DeviceMemory::<u32>::new(ctx, buckets_count_pass_one.try_into().unwrap()).unwrap();
        let bucket_runs_count = DeviceMemory::<u32>::new(ctx, 1).unwrap();
        let bucket_run_offsets =
            DeviceMemory::<u32>::new(ctx, buckets_count_pass_one.try_into().unwrap()).unwrap();
        let sorted_bucket_run_lengths =
            DeviceMemory::<u32>::new(ctx, buckets_count_pass_one.try_into().unwrap()).unwrap();
        let sorted_bucket_run_offsets =
            DeviceMemory::<u32>::new(ctx, buckets_count_pass_one.try_into().unwrap()).unwrap();
        let sorted_unique_bucket_indexes =
            DeviceMemory::<u32>::new(ctx, buckets_count_pass_one.try_into().unwrap()).unwrap();

        let result_windows_count = min(
            FR_MODULUS_BITS,
            windows_count_pass_one * bits_count_pass_one,
        );
        let results =
            DeviceMemory::<GpuEdProjective>::new(ctx, result_windows_count as usize).unwrap();

        let input_indexes_sort_temp_storage_bytes = 1499833855;
        let input_indexes_sort_temp_storage =
            DeviceMemory::<u8>::new(ctx, input_indexes_sort_temp_storage_bytes).unwrap();

        let sort_offsets_temp_storage_bytes = 204527103;
        let sort_offsets_temp_storage =
            DeviceMemory::<u8>::new(ctx, sort_offsets_temp_storage_bytes).unwrap();

        let target_buckets = DeviceMemory::<GpuEdProjective>::new(ctx, 3145728).unwrap();

        let bases_device = DeviceMemory::<GpuAffine>::new(ctx, count * precompute_factor).unwrap();

        let ed_bases_affine =
            Self::precompute_ed_bases(bases, result_bits_count, precompute_factor);
        println!("precomputer ed bases ok");
        bases_device
            .read_from_async(
                ed_bases_affine.as_slice(),
                ed_bases_affine.len(),
                &self.stream,
            )
            .unwrap();

        let cpu_results = HostMemory::<GpuEdProjective>::new(result_bits_count as usize).unwrap();
        let cpu_top_win_results =
            HostMemory::<GpuEdProjective>::new(1 << top_win_used_bits as usize).unwrap(); //todo

        MSMContext {
            log_count,
            bases: bases_device,
            buckets_pass_one,
            bucket_indexes,
            base_indexes,
            sorted_bucket_indexes,
            sorted_base_indexes,
            unique_bucket_indexes,
            bucket_run_lengths,
            bucket_runs_count,
            bucket_run_offsets,
            sorted_bucket_run_lengths,
            sorted_bucket_run_offsets,
            sorted_unique_bucket_indexes,
            window_bits_count: WINDOW_BITS_COUNT,
            results,
            msm_kern: self,
            input_indexes_sort_temp_storage,
            target_buckets,
            sort_offsets_temp_storage,
            inputs_scalars,
            precomputed_windows_stride: PRECOMPUTE_WINDOWS_STRIDE,
            precomputed_bases_stride: 1u32 << log_count,
            cpu_results,
            cpu_top_win_results,
            top_window_unused_bits,
        }
    }

    fn initialize_buckets(&self, gpu_addr: u64, count: u32) -> GPUResult<()> {
        const UINT4_COUNT: u32 = 3; //limbs: uint4 * 3

        let count_u4 = UINT4_COUNT * count;

        let (gws, lws) = calc_cuda_wg_threads((count_u4) as usize);
        let f = self
            .kernel_func_map
            .get("initialize_buckets_kernel")
            .unwrap()
            .inner;

        let params = make_params!(gpu_addr, count_u4);
        let kern = create_kernel_with_params!(f, <<<gws, lws, 0>>>(params));

        self.stream.launch(&kern)?;

        Ok(())
    }

    fn compute_bucket_indexes(
        &self,
        scalars: &DeviceMemory<BigInteger256>,
        windows_count: u32,
        window_bits: u32,
        precomputed_windows_stride: u32,
        precomputed_bases_stride: u32,
        bucket_indexes: &DeviceMemory<u32>,
        base_indexes: &DeviceMemory<u32>,
        count: u32,
        scalars_not_montgomery: bool,
    ) -> GPUResult<()> {
        let (gws, lws) = calc_cuda_wg_threads(count as usize);
        let f = self
            .kernel_func_map
            .get("compute_bucket_indexes_kernel")
            .unwrap()
            .inner;

        let params = make_params!(
            scalars.get_inner(),
            windows_count,
            window_bits,
            precomputed_windows_stride,
            precomputed_bases_stride,
            bucket_indexes.get_inner(),
            base_indexes.get_inner(),
            count,
            scalars_not_montgomery
        );
        let kern = create_kernel_with_params!(f, <<<gws, lws, 0>>>(params));

        self.stream.launch(&kern)?;

        Ok(())
    }

    fn compute_top_window_bucket_indexes(
        &self,
        scalars: &DeviceMemory<BigInteger256>,
        windows_count: u32,
        window_bits: u32,
        precomputed_windows_stride: u32,
        precomputed_bases_stride: u32,
        bucket_indexes: &DeviceMemory<u32>,
        base_indexes: &DeviceMemory<u32>,
        count: u32,
        scalars_not_montgomery: bool,
    ) -> GPUResult<()> {
        let (gws, lws) = calc_cuda_wg_threads(count as usize);
        let f = self
            .kernel_func_map
            .get("compute_top_win_bucket_indexes_kernel")
            .unwrap()
            .inner;

        let params = make_params!(
            scalars.get_inner(),
            windows_count,
            window_bits,
            precomputed_windows_stride,
            precomputed_bases_stride,
            bucket_indexes.get_inner(),
            base_indexes.get_inner(),
            count,
            scalars_not_montgomery
        );
        let kern = create_kernel_with_params!(f, <<<gws, lws, 0>>>(params));

        self.stream.launch(&kern)?;

        Ok(())
    }

    fn aggregate_buckets(
        &self,
        is_first: bool,
        base_indexes: &DeviceMemory<u32>,
        bucket_run_offsets: &DeviceMemory<u32>,
        bucket_run_lengths: &DeviceMemory<u32>,
        bucket_indexes: &DeviceMemory<u32>,
        bases: CUdeviceptr,
        buckets: &DeviceMemory<GpuEdProjective>,
        count: u32,
    ) -> GPUResult<()> {
        let (gws, lws) = calc_cuda_wg_threads(count as usize);
        let f = self
            .kernel_func_map
            .get("aggregate_buckets_kernel")
            .unwrap()
            .inner;

        let params = make_params!(
            is_first,
            base_indexes.get_inner(),
            bucket_run_offsets.get_inner(),
            bucket_run_lengths.get_inner(),
            bucket_indexes.get_inner(),
            bases,
            buckets.get_inner(),
            count
        );
        let kern = create_kernel_with_params!(f, <<<gws, lws, 0>>>(params));
        self.stream.launch(&kern)?;
        Ok(())
    }

    fn reduce_buckets(&self, buckets_addr: u64, count: u32) -> GPUResult<()> {
        let (gws, lws) = calc_cuda_wg_threads(count as usize);
        let f = self
            .kernel_func_map
            .get("reduce_buckets_kernel")
            .unwrap()
            .inner;

        let params = make_params!(buckets_addr, count);
        let kern = create_kernel_with_params!(f, <<<gws, lws, 0>>>(params));

        self.stream.launch(&kern)?;

        Ok(())
    }

    fn reduce_buckets_to_jacobian(
        &self,
        buckets_addr: u64,
        count: u32,
        target: &DeviceMemory<GpuEdProjective>,
    ) -> GPUResult<()> {
        let (gws, lws) = calc_cuda_wg_threads(count as usize);
        let f = self
            .kernel_func_map
            .get("reduce_buckets_kernel_to_jacobian")
            .unwrap()
            .inner;

        let params = make_params!(buckets_addr, count, target.get_inner());
        let kern = create_kernel_with_params!(f, <<<gws, lws, 0>>>(params));

        self.stream.launch(&kern)?;

        Ok(())
    }

    fn last_pass_gather(
        &self,
        bits_count_pass_one: u32,
        source: u64,
        target: &DeviceMemory<GpuEdProjective>,
        count: u32,
    ) -> GPUResult<()> {
        let (gws, lws) = calc_cuda_wg_threads(count.try_into().unwrap());

        let f = self
            .kernel_func_map
            .get("last_pass_gather_kernel")
            .unwrap()
            .inner;

        let params = make_params!(bits_count_pass_one, source, target.get_inner(), count);
        let kern = create_kernel_with_params!(f, <<<gws, lws, 0>>>(params));
        self.stream.launch(&kern)?;

        Ok(())
    }

    fn split_windows(
        &self,
        source_window_bits_count: u32,
        source_windows_count: u32,
        source_buckets: u64,
        target_buckets: u64,
        count: u32,
    ) -> GPUResult<()> {
        let (gws, lws) = calc_cuda_wg_threads(count as usize);
        let f = self
            .kernel_func_map
            .get("split_windows_kernel_generic")
            .unwrap()
            .inner;

        let params = make_params!(
            source_window_bits_count,
            source_windows_count,
            source_buckets,
            target_buckets,
            count
        );
        let kern = create_kernel_with_params!(f, <<<gws, lws, 0>>>(params));
        self.stream.launch(&kern)?;

        Ok(())
    }

    fn compute_top_window(&self, cfg: &mut ExtendedConfiguration<G>) {
        let ec = &mut cfg.exec_cfg.msm_context;
        let stream = &self.stream;
        let log_scalars_count = ec.log_count;
        let log_max_inputs_count = cfg.log_max_inputs_count;
        let bits_count_pass_one = match ec.window_bits_count > 0 {
            true => ec.window_bits_count,
            false => get_window_bits_count(log_scalars_count),
        };
        let precomputed_windows_stride = ec.precomputed_windows_stride;
        let precomputed_bases_stride = ec.precomputed_bases_stride;
        let total_windows_count = get_windows_count(bits_count_pass_one);
        let log_precomputations_count = 0;
        let top_window_unused_bits = total_windows_count * bits_count_pass_one - FR_MODULUS_BITS;
        if top_window_unused_bits == 0 {
            return;
        }
        let windows_count_pass_one = 1;
        let buckets_count_pass_one = windows_count_pass_one << bits_count_pass_one;

        let stream_sort_a = create_stream();
        let stream_sort_b = create_stream();

        let log_inputs_count = log_max_inputs_count;
        let inputs_count = 1 << log_inputs_count;

        let input_indexes_count = 1 << log_inputs_count;
        self.initialize_buckets(
            ec.buckets_pass_one.get_inner(),
            buckets_count_pass_one as u32,
        )
        .unwrap();

        self.compute_top_window_bucket_indexes(
            &ec.inputs_scalars,
            total_windows_count,
            bits_count_pass_one,
            precomputed_windows_stride,
            precomputed_bases_stride,
            &ec.bucket_indexes,
            &ec.base_indexes,
            inputs_count,
            cfg.exec_cfg.scalars_not_montgomery,
        )
        .unwrap();

        let mut input_indexes_sort_temp_storage_bytes: usize = 0;
        sort_pairs_local(
            std::ptr::null_mut(),
            &mut input_indexes_sort_temp_storage_bytes,
            &ec.bucket_indexes,
            &ec.sorted_bucket_indexes,
            &ec.base_indexes,
            &ec.sorted_base_indexes,
            input_indexes_count as i32,
            0,
            bits_count_pass_one as i32,
            stream.get_inner(),
        );

        let input_indexes_sort_temp_storage_void = unsafe {
            std::mem::transmute::<u64, *mut c_void>(ec.input_indexes_sort_temp_storage.get_inner())
        };

        sort_pairs_local(
            input_indexes_sort_temp_storage_void,
            &mut input_indexes_sort_temp_storage_bytes,
            &ec.bucket_indexes,
            &ec.sorted_bucket_indexes,
            &ec.base_indexes,
            &ec.sorted_base_indexes,
            input_indexes_count as i32,
            0,
            bits_count_pass_one as i32,
            stream.get_inner(),
        );

        memset_d32_async(
            ec.bucket_run_lengths.get_inner(),
            0,
            buckets_count_pass_one as u64,
            stream.get_inner(),
        );

        let mut encode_temp_storage_bytes: usize = 0;
        run_length_encode_local(
            std::ptr::null_mut(),
            &mut encode_temp_storage_bytes,
            &ec.sorted_bucket_indexes,
            &ec.unique_bucket_indexes,
            &ec.bucket_run_lengths,
            &ec.bucket_runs_count,
            input_indexes_count as i32,
            stream.get_inner(),
        );

        let encode_temp_storage_void = input_indexes_sort_temp_storage_void;
        run_length_encode_local(
            encode_temp_storage_void,
            &mut encode_temp_storage_bytes,
            &ec.sorted_bucket_indexes,
            &ec.unique_bucket_indexes,
            &ec.bucket_run_lengths,
            &ec.bucket_runs_count,
            input_indexes_count as i32,
            stream.get_inner(),
        );

        let mut scan_temp_storage_bytes: usize = 0;
        exclusive_sum_local(
            std::ptr::null_mut(),
            &mut scan_temp_storage_bytes,
            &ec.bucket_run_lengths,
            &ec.bucket_run_offsets,
            buckets_count_pass_one as i32,
            stream.get_inner(),
        );
        let scan_temp_storage_void = input_indexes_sort_temp_storage_void;
        exclusive_sum_local(
            scan_temp_storage_void,
            &mut scan_temp_storage_bytes,
            &ec.bucket_run_lengths,
            &ec.bucket_run_offsets,
            buckets_count_pass_one as i32,
            stream.get_inner(),
        );

        let mut sort_offsets_temp_storage_bytes: usize = 0;
        sort_pairs_descending_local(
            std::ptr::null_mut(),
            &mut sort_offsets_temp_storage_bytes,
            &ec.bucket_run_lengths,
            &ec.sorted_bucket_run_lengths,
            &ec.bucket_run_offsets,
            &ec.sorted_bucket_run_offsets,
            buckets_count_pass_one as i32,
            0,
            (log_inputs_count + 1 + log_precomputations_count) as i32,
            stream.get_inner(),
        );

        let mut sort_indexes_temp_storage_bytes: usize = 0;
        sort_pairs_descending_local(
            std::ptr::null_mut(),
            &mut sort_indexes_temp_storage_bytes,
            &ec.bucket_run_lengths,
            &ec.sorted_bucket_run_lengths,
            &ec.unique_bucket_indexes,
            &ec.sorted_unique_bucket_indexes,
            buckets_count_pass_one as i32,
            0,
            (log_inputs_count + 1 + log_precomputations_count) as i32,
            stream.get_inner(),
        );

        let event_sort_inputs_ready: CUevent =
            create_event(CUevent_flags_enum_CU_EVENT_DISABLE_TIMING);
        event_record(event_sort_inputs_ready, stream.get_inner());

        stream_wait_event(
            stream_sort_a,
            event_sort_inputs_ready,
            CUevent_wait_flags_enum_CU_EVENT_WAIT_DEFAULT,
        );
        stream_wait_event(
            stream_sort_b,
            event_sort_inputs_ready,
            CUevent_wait_flags_enum_CU_EVENT_WAIT_DEFAULT,
        );

        destroy_event(event_sort_inputs_ready);
        let sort_offsets_temp_storage_void = unsafe {
            std::mem::transmute::<u64, *mut c_void>(ec.sort_offsets_temp_storage.get_inner())
        };

        sort_pairs_descending_local(
            sort_offsets_temp_storage_void,
            &mut sort_offsets_temp_storage_bytes,
            &ec.bucket_run_lengths,
            &ec.sorted_bucket_run_lengths,
            &ec.bucket_run_offsets,
            &ec.sorted_bucket_run_offsets,
            buckets_count_pass_one as i32,
            0,
            (log_inputs_count + 1 + log_precomputations_count) as i32,
            stream_sort_a,
        );

        let sort_indexes_temp_storage_void = input_indexes_sort_temp_storage_void;
        sort_pairs_descending_local(
            sort_indexes_temp_storage_void,
            &mut sort_indexes_temp_storage_bytes,
            &ec.bucket_run_lengths,
            &ec.sorted_bucket_run_lengths,
            &ec.unique_bucket_indexes,
            &ec.sorted_unique_bucket_indexes,
            buckets_count_pass_one as i32,
            0,
            (log_inputs_count + 1 + log_precomputations_count) as i32,
            stream_sort_b,
        );

        let event_sort_a = create_event(CUevent_flags_enum_CU_EVENT_DISABLE_TIMING);
        let event_sort_b = create_event(CUevent_flags_enum_CU_EVENT_DISABLE_TIMING);
        event_record(event_sort_a, stream_sort_a);
        event_record(event_sort_b, stream_sort_b);

        stream_wait_event(
            stream.get_inner(),
            event_sort_a,
            CUevent_wait_flags_enum_CU_EVENT_WAIT_DEFAULT,
        );
        stream_wait_event(
            stream.get_inner(),
            event_sort_b,
            CUevent_wait_flags_enum_CU_EVENT_WAIT_DEFAULT,
        );
        destroy_event(event_sort_a);
        destroy_event(event_sort_b);

        // aggregate buckets
        self.aggregate_buckets(
            true,
            &ec.sorted_base_indexes,
            &ec.sorted_bucket_run_offsets,
            &ec.sorted_bucket_run_lengths,
            &ec.sorted_unique_bucket_indexes,
            ec.bases.get_inner(),
            &ec.buckets_pass_one,
            buckets_count_pass_one as u32,
        )
        .unwrap();

        for i in 0..(top_window_unused_bits - 1) {
            self.reduce_buckets(
                ec.buckets_pass_one.get_inner(),
                1 << (bits_count_pass_one - i - 1),
            )
            .unwrap();
        }

        let top_win_results_len = ec.cpu_top_win_results.len();

        self.reduce_buckets_to_jacobian(
            ec.buckets_pass_one.get_inner(),
            1 << (bits_count_pass_one - (top_window_unused_bits - 1) - 1),
            &ec.results,
        )
        .unwrap();

        ec.results
            .write_to_async(&mut ec.cpu_top_win_results, top_win_results_len, stream)
            .unwrap();

        stream.sync().unwrap();
    }

    fn compute_windows(&self, cfg: &mut ExtendedConfiguration<G>, event_scalars_loaded: &CUevent) {
        let ec = &mut cfg.exec_cfg.msm_context;
        let stream = &self.stream;
        let log_max_inputs_count = cfg.log_max_inputs_count;
        let bits_count_pass_one = ec.window_bits_count;
        let precomputed_windows_stride = ec.precomputed_windows_stride;
        let precomputed_bases_stride = ec.precomputed_bases_stride;
        let mut total_windows_count = get_windows_count(bits_count_pass_one);

        let precomputations_count = match precomputed_windows_stride > 0 {
            true => (total_windows_count - 1) / precomputed_windows_stride + 1,
            false => 1,
        };
        let log_precomputations_count = log2_ceiling(precomputations_count) + 1;
        let windows_count_pass_one = match precomputed_windows_stride > 0 {
            true => precomputed_windows_stride,
            false => total_windows_count,
        };

        let top_window_unused_bits = ec.top_window_unused_bits;
        if top_window_unused_bits > 0 {
            total_windows_count -= 1;
        }

        let start = std::time::Instant::now();

        let buckets_count_pass_one = (windows_count_pass_one << bits_count_pass_one) as usize;
        let stream_sort_a = create_stream();
        let stream_sort_b = create_stream();

        let log_inputs_count = log_max_inputs_count;
        let inputs_count = 1 << log_inputs_count;
        let is_first_loop = true;

        let input_indexes_count = total_windows_count << log_inputs_count;
        self.initialize_buckets(
            ec.buckets_pass_one.get_inner(),
            buckets_count_pass_one as u32,
        )
        .unwrap();

        stream_wait_event(
            stream.get_inner(),
            event_scalars_loaded.clone(),
            CUevent_wait_flags_enum_CU_EVENT_WAIT_DEFAULT,
        );

        self.compute_bucket_indexes(
            &ec.inputs_scalars,
            total_windows_count,
            bits_count_pass_one,
            precomputed_windows_stride,
            precomputed_bases_stride,
            &ec.bucket_indexes,
            &ec.base_indexes,
            inputs_count,
            cfg.exec_cfg.scalars_not_montgomery,
        )
        .unwrap();
        let mut input_indexes_sort_temp_storage_bytes: usize = 0;
        sort_pairs_local(
            std::ptr::null_mut(),
            &mut input_indexes_sort_temp_storage_bytes,
            &ec.bucket_indexes,
            &ec.sorted_bucket_indexes,
            &ec.base_indexes,
            &ec.sorted_base_indexes,
            input_indexes_count as i32,
            0,
            23 as i32,
            stream.get_inner(),
        );

        assert!(ec.input_indexes_sort_temp_storage.size() >= input_indexes_sort_temp_storage_bytes);
        let input_indexes_sort_temp_storage_void = unsafe {
            std::mem::transmute::<u64, *mut c_void>(ec.input_indexes_sort_temp_storage.get_inner())
        };

        sort_pairs_local(
            input_indexes_sort_temp_storage_void,
            &mut input_indexes_sort_temp_storage_bytes,
            &ec.bucket_indexes,
            &ec.sorted_bucket_indexes,
            &ec.base_indexes,
            &ec.sorted_base_indexes,
            input_indexes_count as i32,
            0,
            bits_count_pass_one as i32,
            stream.get_inner(),
        );
        memset_d32_async(
            ec.bucket_run_lengths.get_inner(),
            0,
            buckets_count_pass_one as u64,
            stream.get_inner(),
        );
        let mut encode_temp_storage_bytes: usize = 0;
        run_length_encode_local(
            std::ptr::null_mut(),
            &mut encode_temp_storage_bytes,
            &ec.sorted_bucket_indexes,
            &ec.unique_bucket_indexes,
            &ec.bucket_run_lengths,
            &ec.bucket_runs_count,
            input_indexes_count as i32,
            stream.get_inner(),
        );

        let encode_temp_storage_void = input_indexes_sort_temp_storage_void;
        run_length_encode_local(
            encode_temp_storage_void,
            &mut encode_temp_storage_bytes,
            &ec.sorted_bucket_indexes,
            &ec.unique_bucket_indexes,
            &ec.bucket_run_lengths,
            &ec.bucket_runs_count,
            input_indexes_count as i32,
            stream.get_inner(),
        );

        let mut scan_temp_storage_bytes: usize = 0;
        exclusive_sum_local(
            std::ptr::null_mut(),
            &mut scan_temp_storage_bytes,
            &ec.bucket_run_lengths,
            &ec.bucket_run_offsets,
            buckets_count_pass_one as i32,
            stream.get_inner(),
        );

        let scan_temp_storage_void = input_indexes_sort_temp_storage_void;
        exclusive_sum_local(
            scan_temp_storage_void,
            &mut scan_temp_storage_bytes,
            &ec.bucket_run_lengths,
            &ec.bucket_run_offsets,
            buckets_count_pass_one as i32,
            stream.get_inner(),
        );

        let mut sort_offsets_temp_storage_bytes: usize = 0;
        sort_pairs_descending_local(
            std::ptr::null_mut(),
            &mut sort_offsets_temp_storage_bytes,
            &ec.bucket_run_lengths,
            &ec.sorted_bucket_run_lengths,
            &ec.bucket_run_offsets,
            &ec.sorted_bucket_run_offsets,
            buckets_count_pass_one as i32,
            0,
            (log_inputs_count + 1 + log_precomputations_count) as i32,
            stream.get_inner(),
        );

        let mut sort_indexes_temp_storage_bytes: usize = 0;
        sort_pairs_descending_local(
            std::ptr::null_mut(),
            &mut sort_indexes_temp_storage_bytes,
            &ec.bucket_run_lengths,
            &ec.sorted_bucket_run_lengths,
            &ec.unique_bucket_indexes,
            &ec.sorted_unique_bucket_indexes,
            buckets_count_pass_one as i32,
            0,
            (log_inputs_count + 1 + log_precomputations_count) as i32,
            stream.get_inner(),
        );

        let event_sort_inputs_ready = create_event(CUevent_flags_enum_CU_EVENT_DISABLE_TIMING);
        event_record(event_sort_inputs_ready, stream.get_inner());
        stream_wait_event(
            stream_sort_a,
            event_sort_inputs_ready,
            CUevent_wait_flags_enum_CU_EVENT_WAIT_DEFAULT,
        );
        stream_wait_event(
            stream_sort_b,
            event_sort_inputs_ready,
            CUevent_wait_flags_enum_CU_EVENT_WAIT_DEFAULT,
        );
        destroy_event(event_sort_inputs_ready);

        let sort_offsets_temp_storage_void = unsafe {
            std::mem::transmute::<u64, *mut c_void>(ec.sort_offsets_temp_storage.get_inner())
        };

        sort_pairs_descending_local(
            sort_offsets_temp_storage_void,
            &mut sort_offsets_temp_storage_bytes,
            &ec.bucket_run_lengths,
            &ec.sorted_bucket_run_lengths,
            &ec.bucket_run_offsets,
            &ec.sorted_bucket_run_offsets,
            buckets_count_pass_one as i32,
            0,
            (log_inputs_count + 1 + log_precomputations_count) as i32,
            stream_sort_a,
        );

        let sort_indexes_temp_storage_void = input_indexes_sort_temp_storage_void;
        sort_pairs_descending_local(
            sort_indexes_temp_storage_void,
            &mut sort_indexes_temp_storage_bytes,
            &ec.bucket_run_lengths,
            &ec.sorted_bucket_run_lengths,
            &ec.unique_bucket_indexes,
            &ec.sorted_unique_bucket_indexes,
            buckets_count_pass_one as i32,
            0,
            (log_inputs_count + 1 + log_precomputations_count) as i32,
            stream_sort_b,
        );

        let event_sort_a = create_event(CUevent_flags_enum_CU_EVENT_DISABLE_TIMING);
        let event_sort_b = create_event(CUevent_flags_enum_CU_EVENT_DISABLE_TIMING);
        event_record(event_sort_a, stream_sort_a);
        event_record(event_sort_b, stream_sort_b);
        stream_wait_event(
            stream.get_inner(),
            event_sort_a,
            CUevent_wait_flags_enum_CU_EVENT_WAIT_DEFAULT,
        );
        stream_wait_event(
            stream.get_inner(),
            event_sort_b,
            CUevent_wait_flags_enum_CU_EVENT_WAIT_DEFAULT,
        );
        destroy_event(event_sort_a);
        destroy_event(event_sort_b);
        destroy_stream(stream_sort_a);
        destroy_stream(stream_sort_b);

        // aggregate buckets
        self.aggregate_buckets(
            is_first_loop,
            &ec.sorted_base_indexes,
            &ec.sorted_bucket_run_offsets,
            &ec.sorted_bucket_run_lengths,
            &ec.sorted_unique_bucket_indexes,
            ec.bases.get_inner(),
            &ec.buckets_pass_one,
            buckets_count_pass_one as u32,
        )
        .unwrap();

        let source_bits_count = bits_count_pass_one;
        let source_windows_count = windows_count_pass_one;
        let target_bits_count = (source_bits_count + 1) >> 1;
        let target_windows_count = source_windows_count << 1;
        let target_buckets_count = target_windows_count << target_bits_count;
        let log_data_split = get_optimal_log_data_split(
            cfg.multiprocessor_count,
            source_bits_count,
            target_bits_count,
            target_windows_count,
        );
        let total_buckets_count = target_buckets_count << log_data_split;

        assert!(total_buckets_count <= ec.target_buckets.size() as u32);
        self.reduce(
            source_bits_count,
            source_windows_count,
            &ec.buckets_pass_one,
            &ec.results,
            &ec.target_buckets,
            cfg.multiprocessor_count,
            bits_count_pass_one,
            &mut ec.cpu_results,
            stream,
        );
    }

    fn reduce(
        &self,
        mut source_bits_count: u32,
        mut source_windows_count: u32,
        source_buckets: &DeviceMemory<GpuEdProjective>,
        results: &DeviceMemory<GpuEdProjective>,
        target_buckets: &DeviceMemory<GpuEdProjective>,
        multiprocessor_count: u32,
        bits_count_pass_one: u32,
        mut cpu_results: &mut [GpuEdProjective],
        stream: &CudaStream,
    ) {
        let mut source_buckets = source_buckets.get_inner();
        let mut target_buckets = target_buckets.get_inner();
        loop {
            let target_bits_count = (source_bits_count + 1) >> 1;
            let target_windows_count = source_windows_count << 1;
            let target_buckets_count = target_windows_count << target_bits_count;
            let log_data_split = get_optimal_log_data_split(
                multiprocessor_count,
                source_bits_count,
                target_bits_count,
                target_windows_count,
            );

            let total_buckets_count = target_buckets_count << log_data_split;
            self.split_windows(
                source_bits_count,
                source_windows_count,
                source_buckets,
                target_buckets,
                total_buckets_count,
            )
            .unwrap();

            let mut j: u32 = 0;
            while j < log_data_split {
                self.reduce_buckets(target_buckets, total_buckets_count >> (j + 1))
                    .unwrap();
                j += 1;
            }

            if target_bits_count == 1 {
                self.last_pass_gather(
                    bits_count_pass_one,
                    target_buckets,
                    &results,
                    results.size() as u32,
                )
                .unwrap();

                results
                    .write_to_async(&mut cpu_results, results.size() as usize, &stream)
                    .unwrap();
                break;
            }

            (source_buckets, target_buckets) = (target_buckets, source_buckets);
            source_bits_count = target_bits_count;
            source_windows_count = target_windows_count;
        }
    }
    fn schedule_execution(&self, cfg: &mut ExtendedConfiguration<G>) {
        let ec = &cfg.exec_cfg;
        let stream = &self.stream;
        let top_window_unused_bits = ec.msm_context.top_window_unused_bits;

        let stream_copy_scalars: CUstream = create_stream();
        let execution_started_event = create_event(CUevent_flags_enum_CU_EVENT_DISABLE_TIMING);
        let event_scalars_loaded = create_event(CUevent_flags_enum_CU_EVENT_DISABLE_TIMING);

        event_record(execution_started_event, stream.get_inner());

        stream_wait_event(
            stream_copy_scalars,
            execution_started_event,
            CUevent_wait_flags_enum_CU_EVENT_WAIT_DEFAULT,
        );

        ec.msm_context
            .inputs_scalars
            .read_from_async_cu(&ec.scalars, ec.scalars.len(), stream_copy_scalars)
            .unwrap();

        assert_eq!(ec.scalars.len(), 1 << cfg.log_max_inputs_count);
        event_record(event_scalars_loaded, stream_copy_scalars);

        self.compute_windows(cfg, &event_scalars_loaded);
        destroy_stream(stream_copy_scalars);
        destroy_event(event_scalars_loaded);
        destroy_event(execution_started_event);

        if top_window_unused_bits > 0 {
            self.compute_top_window(cfg);
            self.stream.sync().unwrap();
        }
    }

    pub fn execute_msm(&self, exec_cfg: ExecuteConfiguration<G>) {
        let log_max_chunk_size = exec_cfg.log_max_chunk_size.clone();
        let mut cfg = ExtendedConfiguration {
            exec_cfg,
            log_max_inputs_count: log_max_chunk_size,
            multiprocessor_count: self.sms_count() as u32,
        };
        self.schedule_execution(&mut cfg);
    }
}

// for plonk
#[cfg(any(feature = "bls12_381", feature = "bls12_377"))]
fn execute_msm<'a, 'b, 'c, G>(configuration: &'a mut MsmConfiguration<'a, 'b, 'c, G>)
where
    G: AffineCurve,
{
    let kern = configuration.msm_context.msm_kern;

    let conf = ExecuteConfiguration {
        scalars: configuration.scalars,
        h2d_copy_finished: configuration
            .h2d_copy_finished
            .map_or(unsafe { std::mem::zeroed() }, |e| e.into()),
        h2d_copy_finished_callback: None,
        h2d_copy_finished_callback_data: std::ptr::null_mut(),
        d2h_copy_finished: None,
        d2h_copy_finished_callback: None,
        d2h_copy_finished_callback_data: std::ptr::null_mut(),
        force_min_chunk_size: configuration.force_min_chunk_size,
        log_min_chunk_size: configuration.log_min_chunk_size,
        force_max_chunk_size: configuration.force_max_chunk_size,
        log_max_chunk_size: configuration.log_max_chunk_size,
        scalars_not_montgomery: true,
        msm_context: configuration.msm_context,
    };
    kern.execute_msm(conf);
}

#[cfg(any(feature = "bls12_381", feature = "bls12_377"))]
pub fn gpu_msm<'a, 'b, G>(
    scalars: &[BigInteger256],
    msm_context: &mut MSMContext<'a, 'b, G>,
) -> G1Projective
where
    G: AffineCurve,
{
    let windows_count: u32 = (FR_MODULUS_BITS - 1) / msm_context.window_bits_count + 1;
    let start = std::time::Instant::now();

    let mut cfg = MsmConfiguration {
        scalars,
        h2d_copy_finished: None,
        h2d_copy_finished_callback: None,
        d2h_copy_finished: None,
        d2h_copy_finished_callback: None,
        force_min_chunk_size: false,
        log_min_chunk_size: 0,
        force_max_chunk_size: true,
        log_max_chunk_size: msm_context.log_count,
        scalars_not_montgomery: true,

        msm_context,
    };
    execute_msm(&mut cfg);
    msm_context.msm_kern.stream.sync().unwrap();

    let mut results_gpu = reduce_results(&msm_context.cpu_results);

    if msm_context.top_window_unused_bits != 0 {
        let results_gpu_top_win = reduce_top_win_results(
            &msm_context.cpu_top_win_results,
            windows_count,
            msm_context.window_bits_count,
            msm_context.precomputed_windows_stride,
        );
        results_gpu = edwards_add_with_neg_one_a(results_gpu, results_gpu_top_win);
    }
    let results_gpu = edwards_proj_to_affine(results_gpu);
    let results_gpu = edwards_from_neg_one_a(results_gpu);
    let results_gpu = edwards_to_sw(results_gpu);
    let mut results_gpu = G1Affine::new(results_gpu.x, results_gpu.y, false);

    results_gpu.into_projective()
}
