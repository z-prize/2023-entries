use crate::{api_local, calc_cuda_wg_threads, GPUResult, MultiexpKernel};
use api_local::*;
use ark_bls12_381::Fq;
use ark_bls12_381::{G1Affine, G1Projective};
use ark_ec::{AffineCurve, ProjectiveCurve};
use ark_ff::BigInteger256;
use ark_std::{log2, Zero};
use crypto_cuda::{api::*, CudaStream, DeviceMemory, HostMemory};
use std::cmp::min;
use std::ffi::c_void;
use std::ops::{AddAssign, DerefMut};
use std::rc::Weak;
use std::{mem::MaybeUninit, sync::Arc};
pub const BLS_381_FR_MODULUS_BITS: u32 = 255;

// const NUM_BITS :u32 =
// <<ark_ec::short_weierstrass_jacobian::GroupAffine<<ark_bls12_381::Parameters
// as Bls12Parameters>::<ark_ec::short_weierstrass_jacobian::GroupParameters> as
// AffineCurve<<ark_bls12_381::Parameters as Bls12Paramete>rs>::G1Parameters> as
// Group>::ScalarField as PrimeField>::Params::MODULUS_BITS as usize;

#[derive(Copy, Clone, Debug, PartialEq)]
pub struct G1AffineNoInfinity {
    pub x: Fq,
    pub y: Fq,
}

impl G1AffineNoInfinity {
    pub fn zero() -> Self {
        G1AffineNoInfinity {
            x: Fq::zero(),
            y: Fq::zero(),
        }
    }
}

impl From<&G1Affine> for G1AffineNoInfinity {
    fn from(point: &G1Affine) -> Self {
        assert!(!point.infinity);
        G1AffineNoInfinity {
            x: point.x,
            y: point.y,
        }
    }
}

//  https://hyperelliptic.org/EFD/g1p/auto-shortw-xyzz.html
//  x=X/ZZ
//  y=Y/ZZZ
//  ZZ^3=ZZZ^2
#[derive(Clone)]
#[warn(dead_code)]
struct PointXyzz {
    _x: Fq,
    _y: Fq,
    _zz: Fq,
    _zzz: Fq,
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
    (BLS_381_FR_MODULUS_BITS - 1) / window_bits_count + 1
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

pub fn reduce_results(points: &[G1Projective]) -> G1Projective {
    let mut sum = G1Projective::zero();
    points.iter().rev().for_each(|p| {
        sum.double_in_place().add_assign(p);
    });
    sum
}

pub fn reduce_top_win_results(
    points: &[G1Projective],
    win_count: u32,
    win_bits: u32,
    precomputed_windows_stride: u32,
) -> G1Projective {
    // const unsigned target_window_index = precomputed_windows_stride ?
    // source_window_index % precomputed_windows_stride : source_window_index;
    let target_window_index = (win_count - 1) % precomputed_windows_stride;

    let affines = G1Projective::batch_normalization_into_affine(points);
    let points = &points[1..];
    let mut res = G1Projective::zero();
    let mut running_sum = G1Projective::zero();
    points.iter().rev().for_each(|p| {
        running_sum.add_assign(p);
        res.add_assign(&running_sum);
    });

    for _ in 0..target_window_index * win_bits {
        res.double_in_place();
    }

    res
}

pub struct MSMContext<'a, 'b, G>
where
    G: AffineCurve,
{
    bases: DeviceMemory<G1AffineNoInfinity>,
    buckets_pass_one: DeviceMemory<PointXyzz>,
    pub input_scalars: DeviceMemory<BigInteger256>,

    pub cpu_results: HostMemory<G1Projective>,
    pub cpu_top_win_results: HostMemory<G1Projective>,
    gpu_results: DeviceMemory<G1Projective>,
    gpu_top_results: DeviceMemory<G1Projective>,
    target_buckets: DeviceMemory<PointXyzz>,

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

    input_indexes_sort_temp_storage: DeviceMemory<u8>,
    sort_offsets_temp_storage: DeviceMemory<u8>,
    encode_temp_storage: DeviceMemory<u8>,
    scan_temp_storage: DeviceMemory<u8>,
    sort_indexes_temp_storage: DeviceMemory<u8>,

    log_count: u32,
    window_bits_count: u32,
    precomputed_windows_stride: u32,
    precomputed_bases_stride: u32,
    top_window_unused_bits: u32,
    pub msm_kern: &'a MultiexpKernel<'a, 'b, G>,
}

impl<G: AffineCurve> MSMContext<'_, '_, G> {
    pub fn read_bases_async(
        &self,
        stream: &CudaStream,
        bases: &mut [G1AffineNoInfinity],
    ) {
        self.bases
            .write_to_async(bases, bases.len(), stream)
            .unwrap();
    }

    pub fn get_log_count(&self) -> u32 {
        self.log_count
    }

    pub fn reset(&mut self) {
        self.input_scalars
            .memset_async(0, self.input_scalars.size(), &self.msm_kern.stream)
            .unwrap();
        self.msm_kern.stream.sync().unwrap();
    }
}

pub struct ExecuteConfiguration<'a, 'b, 'c, G>
where
    G: AffineCurve,
{
    pub scalars_cpu: &'a [BigInteger256],
    pub scalars_gpu: Option<(CUdeviceptr, usize)>,

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
pub struct MsmConfiguration<'a, 'b, 'c, G>
where
    G: AffineCurve,
{
    pub scalars_cpu: &'a [BigInteger256],
    pub scalars_gpu: Option<(CUdeviceptr, usize)>,

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
struct ExtendedConfiguration<'a, 'b, 'c, G>
where
    G: AffineCurve,
{
    exec_cfg: ExecuteConfiguration<'a, 'b, 'c, G>,
    log_max_inputs_count: u32,
    multiprocessor_count: u32,
}

impl<'a, 'b, G> MultiexpKernel<'a, 'b, G>
where
    G: AffineCurve,
{
    pub fn product_argument(
        &self,
        gate_roots: &DeviceMemory<BigInteger256>,
        gatewise_sigmas: &DeviceMemory<BigInteger256>,
        gatewise_wires: &DeviceMemory<BigInteger256>,
        ks: &DeviceMemory<BigInteger256>,
        beta: &DeviceMemory<BigInteger256>,
        gamma: &DeviceMemory<BigInteger256>,
        dest: &DeviceMemory<BigInteger256>,
        d_dest: &DeviceMemory<BigInteger256>,
    ) -> GPUResult<()> {
        let count: u32 = gate_roots.size() as u32;
        let (gws, lws) = calc_cuda_wg_threads((count) as usize);
        let f = self.kernel_func_map.get("product_argument").unwrap().inner;

        let params = make_params!(
            gate_roots.get_inner(),
            gatewise_sigmas.get_inner(),
            gatewise_wires.get_inner(),
            ks.get_inner(),
            beta.get_inner(),
            gamma.get_inner(),
            dest.get_inner(),
            d_dest.get_inner(),
            count
        );
        let kern = create_kernel_with_params!(f, <<<gws, lws, 0>>>(params));
        self.stream.launch(&kern)?;
        //self.stream.sync().unwrap();
        Ok(())
    }

    pub fn product_z_part1(
        &self,
        dest: &DeviceMemory<BigInteger256>,
        z_iv: &DeviceMemory<BigInteger256>,
        chunk_size: usize,
        count: usize,
    ) -> GPUResult<()> {
        let chunks = count / chunk_size;
        let (gws, lws) = calc_cuda_wg_threads(chunks);
        let f = self.kernel_func_map.get("product_z_part1").unwrap().inner;

        let params =
            make_params!(dest.get_inner(), z_iv.get_inner(), chunk_size, count);
        let kern = create_kernel_with_params!(f, <<<gws, lws, 0>>>(params));
        self.stream.launch(&kern)?;
        // self.stream.sync().unwrap();

        Ok(())
    }

    pub fn product_z_part2(
        &self,
        z_iv: &DeviceMemory<BigInteger256>,
        z_piv: &DeviceMemory<BigInteger256>,
        chunk_size: usize,
        count: usize,
    ) -> GPUResult<()> {
        let chunks = count / chunk_size;
        let (gws, lws) = calc_cuda_wg_threads(chunks);
        let f = self.kernel_func_map.get("product_z_part2").unwrap().inner;

        let params = make_params!(
            z_iv.get_inner(),
            z_piv.get_inner(),
            chunk_size,
            count
        );
        let kern = create_kernel_with_params!(f, <<<gws, lws, 0>>>(params));
        self.stream.launch(&kern)?;
        //self.stream.sync().unwrap();

        Ok(())
    }

    pub fn product_z_part3(
        &self,
        dest: &DeviceMemory<BigInteger256>,
        z_gpu: CUdeviceptr,
        z_piv: &DeviceMemory<BigInteger256>,
        chunk_size: usize,
        count: usize,
    ) -> GPUResult<()> {
        let chunks = count / chunk_size;
        let (gws, lws) = calc_cuda_wg_threads(chunks);
        let f = self.kernel_func_map.get("product_z_part3").unwrap().inner;

        let params = make_params!(
            dest.get_inner(),
            z_gpu,
            z_piv.get_inner(),
            chunk_size,
            count
        );
        let kern = create_kernel_with_params!(f, <<<gws, lws, 0>>>(params));
        self.stream.launch(&kern)?;
        self.stream.sync().unwrap();

        Ok(())
    }

    pub fn left_shift(
        &self,
        values_addr: CUdeviceptr,
        shift: u32,
        count: u32,
    ) -> GPUResult<()> {
        let (gws, lws) = calc_cuda_wg_threads((count) as usize);
        let f = self.kernel_func_map.get("left_shift_kernel").unwrap().inner;

        let params = make_params!(values_addr, shift, count);
        let kern = create_kernel_with_params!(f, <<<gws, lws, 0>>>(params));

        self.stream.launch(&kern)?;

        Ok(())
    }

    // todo G1Affine  ==>T
    pub fn multi_scalar_mult_init(
        &'a self,
        bases: &[G1Affine],
    ) -> MSMContext<'a, 'b, G> {
        let ctx = self.core.get_context();

        const WINDOW_BITS_COUNT: u32 = 21; //18 - 24 passed
        const PRECOMPUTE_FACTOR: u32 = 4;
        const WINDOWS_COUNT: u32 =
            (BLS_381_FR_MODULUS_BITS - 1) / WINDOW_BITS_COUNT + 1;
        const PRECOMPUTE_WINDOWS_STRIDE: u32 =
            (WINDOWS_COUNT - 1) / PRECOMPUTE_FACTOR + 1;
        let result_bits_count: u32 =
            WINDOW_BITS_COUNT * PRECOMPUTE_WINDOWS_STRIDE;
        let precompute_factor: usize = PRECOMPUTE_FACTOR as usize;

        let bits_count_pass_one = WINDOW_BITS_COUNT;
        let mut total_windows_count = get_windows_count(bits_count_pass_one);
        let top_window_unused_bits =
            total_windows_count * bits_count_pass_one - BLS_381_FR_MODULUS_BITS;
        if top_window_unused_bits > 0 {
            total_windows_count -= 1;
        }
        let top_win_used_bits = bits_count_pass_one - top_window_unused_bits;

        let count = bases.len();
        let log_count = log2(count);

        assert_eq!(count, 1 << log_count);
        let windows_count_pass_one = PRECOMPUTE_WINDOWS_STRIDE;
        let buckets_count_pass_one =
            (windows_count_pass_one << bits_count_pass_one) as usize;
        let buckets_pass_one = DeviceMemory::<PointXyzz>::new(
            ctx,
            buckets_count_pass_one.try_into().unwrap(),
        )
        .unwrap();
        let result_windows_count = min(
            BLS_381_FR_MODULUS_BITS,
            windows_count_pass_one * bits_count_pass_one,
        );
        let gpu_results = DeviceMemory::<G1Projective>::new(
            ctx,
            result_windows_count as usize,
        )
        .unwrap();

        let input_indexes_count = total_windows_count << log_count;
        let bucket_indexes = DeviceMemory::<u32>::new(
            ctx,
            input_indexes_count.try_into().unwrap(),
        )
        .unwrap();
        let base_indexes = DeviceMemory::<u32>::new(
            ctx,
            input_indexes_count.try_into().unwrap(),
        )
        .unwrap();

        let sorted_bucket_indexes = DeviceMemory::<u32>::new(
            ctx,
            input_indexes_count.try_into().unwrap(),
        )
        .unwrap();
        let sorted_base_indexes = DeviceMemory::<u32>::new(
            ctx,
            input_indexes_count.try_into().unwrap(),
        )
        .unwrap();

        let unique_bucket_indexes = DeviceMemory::<u32>::new(
            ctx,
            buckets_count_pass_one.try_into().unwrap(),
        )
        .unwrap();
        let bucket_run_lengths = DeviceMemory::<u32>::new(
            ctx,
            buckets_count_pass_one.try_into().unwrap(),
        )
        .unwrap();
        let bucket_runs_count = DeviceMemory::<u32>::new(ctx, 1).unwrap();

        let bucket_run_offsets = DeviceMemory::<u32>::new(
            ctx,
            buckets_count_pass_one.try_into().unwrap(),
        )
        .unwrap();
        let sorted_bucket_run_lengths = DeviceMemory::<u32>::new(
            ctx,
            buckets_count_pass_one.try_into().unwrap(),
        )
        .unwrap();
        let sorted_bucket_run_offsets = DeviceMemory::<u32>::new(
            ctx,
            buckets_count_pass_one.try_into().unwrap(),
        )
        .unwrap();
        let sorted_unique_bucket_indexes = DeviceMemory::<u32>::new(
            ctx,
            buckets_count_pass_one.try_into().unwrap(),
        )
        .unwrap();

        let target_buckets =
            DeviceMemory::<PointXyzz>::new(ctx, 4194304).unwrap();
        let inputs_scalars =
            DeviceMemory::<BigInteger256>::new(ctx, count).unwrap();
        let bases_device = DeviceMemory::<G1AffineNoInfinity>::new(
            ctx,
            count * precompute_factor,
        )
        .unwrap();
        let bases_no_infinity = bases
            .iter()
            .map(G1AffineNoInfinity::from)
            .collect::<Vec<_>>();

        bases_device
            .read_from_async(&bases_no_infinity, count, &self.stream)
            .unwrap();

        let cpu_results =
            HostMemory::<G1Projective>::new(result_bits_count as usize)
                .unwrap();
        let cpu_top_win_results =
            HostMemory::<G1Projective>::new(1 << top_win_used_bits as usize)
                .unwrap(); //todo
        let gpu_top_results = DeviceMemory::<G1Projective>::new(
            ctx,
            1 << top_win_used_bits as usize,
        )
        .unwrap();

        let input_indexes_sort_temp_storage_bytes = 407034367; // 1763801599 19; //1628124671 ->20 21;//1492447743 ->22 23;
        let input_indexes_sort_temp_storage =
            DeviceMemory::<u8>::new(ctx, input_indexes_sort_temp_storage_bytes)
                .unwrap();
        let sort_offsets_temp_storage_bytes = 67843583; //203520511;
        let sort_offsets_temp_storage =
            DeviceMemory::<u8>::new(ctx, sort_offsets_temp_storage_bytes)
                .unwrap();
        let encode_temp_storage_bytes = 1049343; // 4544767 19;//4195071 ->21 20;//3845631 ->22 23;
        let encode_temp_storage =
            DeviceMemory::<u8>::new(ctx, encode_temp_storage_bytes).unwrap();

        let scan_temp_storage_bytes = 35583;
        let scan_temp_storage =
            DeviceMemory::<u8>::new(ctx, scan_temp_storage_bytes).unwrap();

        let sort_indexes_temp_storage_bytes = 67843583;
        let sort_indexes_temp_storage =
            DeviceMemory::<u8>::new(ctx, sort_indexes_temp_storage_bytes)
                .unwrap();

        let stride =
            count as u64 * std::mem::size_of::<G1AffineNoInfinity>() as u64;
        for i in 1..precompute_factor as usize {
            let dst_offset = (i as u64 * stride) + bases_device.get_inner();
            let src_offset =
                ((i - 1) as u64 * stride) + bases_device.get_inner();

            DeviceMemory::<u8>::memcpy_from_to_raw_async(
                src_offset as u64,
                dst_offset as u64,
                stride,
                &self.stream,
            )
            .unwrap();
            self.left_shift(dst_offset, result_bits_count, count as u32)
                .unwrap();
        }

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
            gpu_results,     //full wins bits
            gpu_top_results, // top win
            msm_kern: self,
            input_indexes_sort_temp_storage,
            sort_offsets_temp_storage,
            encode_temp_storage,
            scan_temp_storage,
            sort_indexes_temp_storage,

            target_buckets,
            input_scalars: inputs_scalars,
            precomputed_windows_stride: PRECOMPUTE_WINDOWS_STRIDE,
            precomputed_bases_stride: 1u32 << log_count,
            cpu_results,
            cpu_top_win_results,
            top_window_unused_bits,
        }
    }

    fn initialize_buckets(&self, gpu_addr: u64, count: u32) -> GPUResult<()> {
        const UINT4_COUNT: u32 = 3; //limbs: uint4 * 3

        // assert_eq!(buckets_buf.size(), count as usize);
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
        scalars: CUdeviceptr,
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
            scalars,
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
        scalars: CUdeviceptr,
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
            scalars,
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
        buckets: &DeviceMemory<PointXyzz>,
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
        target: &DeviceMemory<G1Projective>,
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
        target: &DeviceMemory<G1Projective>,
        count: u32,
    ) -> GPUResult<()> {
        let (gws, lws) = calc_cuda_wg_threads(count.try_into().unwrap());

        let f = self
            .kernel_func_map
            .get("last_pass_gather_kernel")
            .unwrap()
            .inner;

        let params = make_params!(
            bits_count_pass_one,
            source,
            target.get_inner(),
            count
        );
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

    fn to_umont(&self, gpu_ptr: CUdeviceptr, count: usize) -> GPUResult<()> {
        let (gws, lws) = calc_cuda_wg_threads(count as usize);

        let f = self.kernel_func_map.get("Fr_poly_unmont").unwrap().inner;

        let params = make_params!(gpu_ptr, count);

        let kern = create_kernel_with_params!(f, <<<gws, lws, 0>>>(params));
        self.stream.launch(&kern)?;

        Ok(())
    }

    fn to_mont(&self, gpu_ptr: CUdeviceptr, count: usize) -> GPUResult<()> {
        let (gws, lws) = calc_cuda_wg_threads(count as usize);

        let f = self.kernel_func_map.get("Fr_poly_mont").unwrap().inner;

        let params = make_params!(gpu_ptr, count);

        let kern = create_kernel_with_params!(f, <<<gws, lws, 0>>>(params));
        self.stream.launch(&kern)?;

        Ok(())
    }

    fn wires_to_single_gate(
        &self,
        w_l_gpu: &DeviceMemory<BigInteger256>,
        w_r_gpu: &DeviceMemory<BigInteger256>,
        w_o_gpu: &DeviceMemory<BigInteger256>,
        w_4_gpu: &DeviceMemory<BigInteger256>,
        gatewise_wires_gpu: &DeviceMemory<BigInteger256>,
    ) -> GPUResult<()> {
        let count = w_l_gpu.size();
        let (gws, lws) = calc_cuda_wg_threads(count);

        let f = self
            .kernel_func_map
            .get("wires_to_single_gate")
            .unwrap()
            .inner;

        let params = make_params!(
            w_l_gpu.get_inner(),
            w_r_gpu.get_inner(),
            w_o_gpu.get_inner(),
            w_4_gpu.get_inner(),
            gatewise_wires_gpu.get_inner(),
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
        let log_precomputations_count = 0; // todo del?
        let top_window_unused_bits =
            total_windows_count * bits_count_pass_one - BLS_381_FR_MODULUS_BITS;
        if top_window_unused_bits == 0 {
            return;
        }
        let windows_count_pass_one = 1;
        let buckets_count_pass_one =
            windows_count_pass_one << bits_count_pass_one;

        let stream_sort_a = create_stream();
        let stream_sort_b = create_stream();
        let buckets_pass_one = &ec.buckets_pass_one;
        let log_inputs_count = log_max_inputs_count;
        let inputs_count = 1 << log_inputs_count;

        let input_indexes_count = 1 << log_inputs_count;
        self.initialize_buckets(
            buckets_pass_one.get_inner(),
            buckets_count_pass_one as u32,
        )
        .unwrap();

        let input_scalars = if cfg.exec_cfg.scalars_cpu.len() > 0 {
            ec.input_scalars.get_inner()
        } else {
            let (gpu_ptr, _n) = cfg.exec_cfg.scalars_gpu.unwrap();
            gpu_ptr
        };

        self.compute_top_window_bucket_indexes(
            input_scalars,
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

        // sort base indexes by bucket indexes
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
            // 32,
            stream.get_inner(),
        );

        let input_indexes_sort_temp_storage_void = unsafe {
            std::mem::transmute::<u64, *mut c_void>(
                ec.input_indexes_sort_temp_storage.get_inner(),
            )
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
            // 32,
            stream.get_inner(),
        );

        memset_d32_async(
            ec.bucket_run_lengths.get_inner(),
            0,
            buckets_count_pass_one as u64,
            stream.get_inner(),
        );

        // temp_storage encode_temp_storage;
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

        let encode_temp_storage_void = unsafe {
            std::mem::transmute::<u64, *mut c_void>(
                ec.encode_temp_storage.get_inner(),
            )
        };
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

        // compute bucket run offsets
        let mut scan_temp_storage_bytes: usize = 0;
        exclusive_sum_local(
            std::ptr::null_mut(),
            &mut scan_temp_storage_bytes,
            &ec.bucket_run_lengths,
            &ec.bucket_run_offsets,
            buckets_count_pass_one as i32,
            stream.get_inner(),
        );

        let scan_temp_storage_void = unsafe {
            std::mem::transmute::<u64, *mut c_void>(
                ec.scan_temp_storage.get_inner(),
            )
        };
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

        let event_sort_inputs_ready =
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
            std::mem::transmute::<u64, *mut c_void>(
                ec.sort_offsets_temp_storage.get_inner(),
            )
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

        let sort_indexes_temp_storage_void = unsafe {
            std::mem::transmute::<u64, *mut c_void>(
                ec.sort_indexes_temp_storage.get_inner(),
            )
        };
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

        //cuEventCreate
        let event_sort_a =
            create_event(CUevent_flags_enum_CU_EVENT_DISABLE_TIMING);
        let event_sort_b =
            create_event(CUevent_flags_enum_CU_EVENT_DISABLE_TIMING);
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

        destroy_stream(stream_sort_a);
        destroy_stream(stream_sort_b);
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
                buckets_pass_one.get_inner(),
                1 << (bits_count_pass_one - i - 1),
            )
            .unwrap();
        }

        let top_win_results_len = ec.cpu_top_win_results.len();
        self.reduce_buckets_to_jacobian(
            buckets_pass_one.get_inner(),
            1 << (bits_count_pass_one - (top_window_unused_bits - 1) - 1),
            &ec.gpu_top_results,
        )
        .unwrap();

        ec.gpu_top_results
            .write_to_async(
                &mut ec.cpu_top_win_results,
                top_win_results_len,
                stream,
            )
            .unwrap();
    }

    fn compute_windows(
        &self,
        cfg: &mut ExtendedConfiguration<G>,
        event_scalars_loaded: &CUevent,
    ) {
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

        let buckets_count_pass_one =
            (windows_count_pass_one << bits_count_pass_one) as usize;
        let buckets_pass_one = &ec.buckets_pass_one;

        let stream_sort_a = create_stream();
        let stream_sort_b = create_stream();

        let log_inputs_count = log_max_inputs_count;
        let inputs_count = 1 << log_inputs_count;

        let input_indexes_count = total_windows_count << log_inputs_count;
        self.initialize_buckets(
            buckets_pass_one.get_inner(),
            buckets_count_pass_one as u32,
        )
        .unwrap();

        let input_scalars = if cfg.exec_cfg.scalars_cpu.len() > 0 {
            // stream.sync().unwrap();
            stream_wait_event(
                stream.get_inner(),
                event_scalars_loaded.clone(),
                CUevent_wait_flags_enum_CU_EVENT_WAIT_DEFAULT,
            );
            ec.input_scalars.get_inner()
        } else {
            let (gpu_ptr, n) = cfg.exec_cfg.scalars_gpu.unwrap();
            assert!(n == inputs_count);
            gpu_ptr
        };
        self.compute_bucket_indexes(
            input_scalars,
            total_windows_count,
            bits_count_pass_one,
            precomputed_windows_stride,
            precomputed_bases_stride,
            &ec.bucket_indexes,
            &ec.base_indexes,
            inputs_count as u32,
            cfg.exec_cfg.scalars_not_montgomery,
        )
        .unwrap();
        stream.sync().unwrap();

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
            // 32,  //Generic version
            stream.get_inner(),
        );

        assert!(
            ec.input_indexes_sort_temp_storage.size()
                >= input_indexes_sort_temp_storage_bytes
        );
        let input_indexes_sort_temp_storage_void = unsafe {
            std::mem::transmute::<u64, *mut c_void>(
                ec.input_indexes_sort_temp_storage.get_inner(),
            )
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
            // 32, //Generic version
            stream.get_inner(),
        );

        memset_d32_async(
            ec.bucket_run_lengths.get_inner(),
            0,
            buckets_count_pass_one as u64,
            stream.get_inner(),
        );

        // temp_storage encode_temp_storage;
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

        assert!(ec.encode_temp_storage.size() >= encode_temp_storage_bytes);

        let encode_temp_storage_void = unsafe {
            std::mem::transmute::<u64, *mut c_void>(
                ec.encode_temp_storage.get_inner(),
            )
        };
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

        // compute bucket run offsets
        let mut scan_temp_storage_bytes: usize = 0;
        exclusive_sum_local(
            std::ptr::null_mut(),
            &mut scan_temp_storage_bytes,
            &ec.bucket_run_lengths,
            &ec.bucket_run_offsets,
            buckets_count_pass_one as i32,
            stream.get_inner(),
        );
        assert!(ec.scan_temp_storage.size() >= scan_temp_storage_bytes);
        let scan_temp_storage_void = unsafe {
            std::mem::transmute::<u64, *mut c_void>(
                ec.scan_temp_storage.get_inner(),
            )
        };
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

        assert!(
            ec.sort_offsets_temp_storage.size()
                >= sort_offsets_temp_storage_bytes
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

        assert!(
            ec.sort_indexes_temp_storage.size()
                >= sort_indexes_temp_storage_bytes
        );

        let event_sort_inputs_ready =
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

        let sort_offsets_temp_storage_void = unsafe {
            std::mem::transmute::<u64, *mut c_void>(
                ec.sort_offsets_temp_storage.get_inner(),
            )
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

        let sort_indexes_temp_storage_void = unsafe {
            std::mem::transmute::<u64, *mut c_void>(
                ec.sort_indexes_temp_storage.get_inner(),
            )
        };
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

        //cuEventCreate
        let event_sort_a =
            create_event(CUevent_flags_enum_CU_EVENT_DISABLE_TIMING);
        let event_sort_b =
            create_event(CUevent_flags_enum_CU_EVENT_DISABLE_TIMING);
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
        destroy_event(event_sort_inputs_ready);

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
        // stream.sync().unwrap();

        // windows 累加
        let source_bits_count = bits_count_pass_one;
        let source_windows_count = windows_count_pass_one;
        let target_bits_count = (source_bits_count + 1) >> 1;
        let target_windows_count = source_windows_count << 1;
        let target_buckets_count = target_windows_count << target_bits_count;
        let log_data_split = get_optimal_log_data_split(
            cfg.multiprocessor_count, //???
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
            &ec.gpu_results,
            &ec.target_buckets,
            cfg.multiprocessor_count,
            bits_count_pass_one,
            &mut ec.cpu_results,
            stream,
        );

        stream.sync().unwrap();
    }

    fn reduce(
        &self,
        mut source_bits_count: u32,
        mut source_windows_count: u32,
        source_buckets: &DeviceMemory<PointXyzz>,
        results: &DeviceMemory<G1Projective>,
        target_buckets: &DeviceMemory<PointXyzz>,
        multiprocessor_count: u32,
        bits_count_pass_one: u32,
        mut cpu_results: &mut [G1Projective],
        stream: &CudaStream,
    ) {
        let mut source_buckets = source_buckets.get_inner();
        let mut target_buckets = target_buckets.get_inner();
        loop {
            let target_bits_count = (source_bits_count + 1) >> 1;
            let target_windows_count = source_windows_count << 1;
            let target_buckets_count =
                target_windows_count << target_bits_count;
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
                self.reduce_buckets(
                    target_buckets,
                    total_buckets_count >> (j + 1),
                )
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
                    .write_to_async(
                        &mut cpu_results,
                        results.size() as usize,
                        &stream,
                    )
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
        let execution_started_event =
            create_event(CUevent_flags_enum_CU_EVENT_DISABLE_TIMING);
        let event_scalars_loaded =
            create_event(CUevent_flags_enum_CU_EVENT_DISABLE_TIMING);

        event_record(execution_started_event, stream.get_inner());

        stream_wait_event(
            stream_copy_scalars,
            execution_started_event,
            CUevent_wait_flags_enum_CU_EVENT_WAIT_DEFAULT,
        );

        assert!(ec.scalars_cpu.len() <= 1 << cfg.log_max_inputs_count);

        if ec.scalars_cpu.len() > 0 {
            ec.msm_context
                .input_scalars
                .read_from_async_cu(
                    &ec.scalars_cpu,
                    ec.scalars_cpu.len(),
                    stream_copy_scalars,
                )
                .unwrap();
            event_record(event_scalars_loaded, stream_copy_scalars);
        }

        let start = std::time::Instant::now();
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

fn execute_msm<'a, 'b, 'c, G>(
    configuration: &'a mut MsmConfiguration<'a, 'b, 'c, G>,
    kern: &MultiexpKernel<'a, 'b, G>,
) where
    G: AffineCurve,
{
    let conf = unsafe {
        ExecuteConfiguration {
            scalars_cpu: configuration.scalars_cpu,
            scalars_gpu: configuration.scalars_gpu,
            h2d_copy_finished: configuration
                .h2d_copy_finished
                .map_or(std::mem::zeroed(), |e| e.into()),
            h2d_copy_finished_callback: None,
            h2d_copy_finished_callback_data: std::ptr::null_mut(),
            d2h_copy_finished: None,
            d2h_copy_finished_callback: None,
            d2h_copy_finished_callback_data: std::ptr::null_mut(),
            force_min_chunk_size: configuration.force_min_chunk_size,
            log_min_chunk_size: configuration.log_min_chunk_size,
            force_max_chunk_size: configuration.force_max_chunk_size,
            log_max_chunk_size: configuration.log_max_chunk_size,
            scalars_not_montgomery: configuration.scalars_not_montgomery,
            msm_context: configuration.msm_context,
        }
    };
    kern.execute_msm(conf);
}

pub fn gpu_msm<'a, 'b, G>(
    scalars_cpu: &[BigInteger256],
    scalars_gpu: Option<(CUdeviceptr, usize)>,
    msm_context: &mut MSMContext<'a, 'b, G>,
    not_montgomery: bool,
) -> G1Projective
where
    G: AffineCurve,
{
    let windows_count: u32 =
        (BLS_381_FR_MODULUS_BITS - 1) / msm_context.window_bits_count + 1;
    let precompute_windows_stride: u32 = msm_context.precomputed_windows_stride;
    let start = std::time::Instant::now();

    let kern = msm_context.msm_kern;
    let scalar_len = scalars_cpu.len();

    if scalar_len != 0 {
        assert!(scalars_gpu.is_none());
    } else {
        assert!(!scalars_gpu.is_none());
    }
    assert!(scalar_len <= msm_context.input_scalars.size());

    if scalar_len != 0 && scalar_len != msm_context.input_scalars.size() {
        println!("scalar len:{}", scalar_len);
        msm_context.reset();
        println!("memset spent:{:?}", start.elapsed());
    }

    let mut cfg = MsmConfiguration {
        scalars_cpu,
        scalars_gpu,
        h2d_copy_finished: None,
        h2d_copy_finished_callback: None,
        d2h_copy_finished: None,
        d2h_copy_finished_callback: None,
        force_min_chunk_size: false,
        log_min_chunk_size: 0,
        force_max_chunk_size: true,
        log_max_chunk_size: msm_context.log_count,
        scalars_not_montgomery: not_montgomery,
        msm_context,
    };
    let start = std::time::Instant::now();
    execute_msm(&mut cfg, &kern);
    // kern.stream.sync().unwrap();
    let mut results_gpu = reduce_results(&msm_context.cpu_results);

    if msm_context.top_window_unused_bits > 0 {
        let results_gpu_top_win = reduce_top_win_results(
            &msm_context.cpu_top_win_results,
            windows_count,
            msm_context.window_bits_count,
            precompute_windows_stride,
        );

        results_gpu.add_assign(results_gpu_top_win);
    }

    results_gpu
}

pub fn product_argument<'a, 'b, G>(
    gate_roots: &DeviceMemory<BigInteger256>,
    gatewise_sigmas: &DeviceMemory<BigInteger256>,
    gatewise_wires: &DeviceMemory<BigInteger256>,
    ks: &DeviceMemory<BigInteger256>,
    beta: &DeviceMemory<BigInteger256>,
    gamma: &DeviceMemory<BigInteger256>,
    dest: &DeviceMemory<BigInteger256>,
    d_dest: &DeviceMemory<BigInteger256>,
    msm_context: &MSMContext<'a, 'b, G>,
) where
    G: AffineCurve,
{
    msm_context
        .msm_kern
        .product_argument(
            gate_roots,
            gatewise_sigmas,
            gatewise_wires,
            ks,
            beta,
            gamma,
            dest,
            d_dest,
        )
        .unwrap();
}

pub fn product_z_parts<'a, 'b, G>(
    dest: &DeviceMemory<BigInteger256>,
    z_gpu: CUdeviceptr,
    z_iv: &DeviceMemory<BigInteger256>,
    z_piv: &DeviceMemory<BigInteger256>,
    chunk_size: usize,
    count: usize,
    msm_context: &MSMContext<'a, 'b, G>,
) where
    G: AffineCurve,
{
    msm_context
        .msm_kern
        .product_z_part1(dest, z_iv, chunk_size, count)
        .unwrap();
    msm_context
        .msm_kern
        .product_z_part2(z_iv, z_piv, chunk_size, count)
        .unwrap();
    msm_context
        .msm_kern
        .product_z_part3(dest, z_gpu, z_piv, chunk_size, count)
        .unwrap();
}

pub fn product_argument_test<'a, 'b, G>(
    gate_roots: &DeviceMemory<BigInteger256>,
    sigmas: &DeviceMemory<BigInteger256>,
    gatewise_wires_gpu: &DeviceMemory<BigInteger256>,
    // ks: &Vec<BigInteger256>,
    // beta: BigInteger256,
    // gamma: BigInteger256,
    z_gpu: CUdeviceptr,
    msm_context: &MSMContext<'a, 'b, G>,
    dest_gpu: &DeviceMemory<BigInteger256>,
    d_dest_gpu: &DeviceMemory<BigInteger256>,
    ks_gpu: &DeviceMemory<BigInteger256>,
    beta_gpu: &DeviceMemory<BigInteger256>,
    gamma_gpu: &DeviceMemory<BigInteger256>,
    z_iv: &DeviceMemory<BigInteger256>,
    z_piv: &DeviceMemory<BigInteger256>,
) where
    G: AffineCurve,
{
    let count = gate_roots.size();
    let ctx = &msm_context.msm_kern.core.context;
    // let ks_gpu = DeviceMemory::<BigInteger256>::new(ctx, ks.len()).unwrap();
    // let beta_gpu = DeviceMemory::<BigInteger256>::new(ctx, 1).unwrap();
    // let gamma_gpu = DeviceMemory::<BigInteger256>::new(ctx, 1).unwrap();
    // let dest_gpu = DeviceMemory::<BigInteger256>::new(ctx, count).unwrap();
    // let d_dest_gpu = DeviceMemory::<BigInteger256>::new(ctx, count).unwrap();

    // ks_gpu.read_from(&ks, ks.len()).unwrap();
    // beta_gpu.read_from(vec![beta].as_slice(), 1).unwrap();
    // gamma_gpu.read_from(vec![gamma].as_slice(), 1).unwrap();

    product_argument(
        &gate_roots,
        &sigmas,
        &gatewise_wires_gpu,
        &ks_gpu,
        &beta_gpu,
        &gamma_gpu,
        &dest_gpu,
        &d_dest_gpu,
        msm_context,
    );

    //z product calculation
    let chunk_size = 512;
    // let chunks = count / chunk_size;
    // let z_iv = DeviceMemory::<BigInteger256>::new(ctx, chunks).unwrap();
    // let z_piv = DeviceMemory::<BigInteger256>::new(ctx, chunks).unwrap();

    product_z_parts(
        &dest_gpu,
        z_gpu,
        &z_iv,
        &z_piv,
        chunk_size,
        count,
        msm_context,
    );
}

pub fn to_mont<'a, 'b, G>(
    msm_context: &MSMContext<'a, 'b, G>,
    scalars_gpu: CUdeviceptr,
    count: usize,
) where
    G: AffineCurve,
{
    msm_context.msm_kern.to_mont(scalars_gpu, count).unwrap();
}

pub fn wires_to_single_gate<'a, 'b, G>(
    w_l_gpu: &DeviceMemory<BigInteger256>,
    w_r_gpu: &DeviceMemory<BigInteger256>,
    w_o_gpu: &DeviceMemory<BigInteger256>,
    w_4_gpu: &DeviceMemory<BigInteger256>,
    gatewise_wires_gpu: &DeviceMemory<BigInteger256>,
    msm_context: &MSMContext<'a, 'b, G>,
) where
    G: AffineCurve,
{
    msm_context
        .msm_kern
        .wires_to_single_gate(
            w_l_gpu,
            w_r_gpu,
            w_o_gpu,
            w_4_gpu,
            gatewise_wires_gpu,
        )
        .unwrap();
}
