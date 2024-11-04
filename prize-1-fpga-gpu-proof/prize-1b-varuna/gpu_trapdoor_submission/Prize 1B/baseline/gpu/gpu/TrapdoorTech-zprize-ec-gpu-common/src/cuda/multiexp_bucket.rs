use crate::{api_local, calc_cuda_wg_threads, GPUResult, MultiexpKernel};
use api_local::*;
use ark_bls12_381::Fq;
use ark_bls12_381::{G1Affine, G1Projective};
use ark_ec::{AffineCurve, ProjectiveCurve};
use ark_ff::BigInteger256;
use ark_std::{log2, Zero};
use crypto_cuda::api::*;
use crypto_cuda::DeviceMemory;
use crypto_cuda::{CudaContext, CudaStream};
use std::cmp::min;
use std::ffi::c_void;
use std::ops::AddAssign;
use std::rc::Weak;
use std::{mem::MaybeUninit, sync::Arc};
const BLS_381_FR_MODULUS_BITS: u32 = 255;

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

fn get_windows_count(window_bits_count: u32) -> u32 {
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
    let mut i = 68;
    points.iter().rev().for_each(|p| {
        i -= 1;
        sum.double_in_place().add_assign(p);
    });
    sum
}

pub struct MSMContext<'a, 'b, G>
where
    G: AffineCurve,
{
    log_count: u32,
    bases: DeviceMemory<G1AffineNoInfinity>,
    window_bits_count: u32,
    precompute_factor: u32,
    msm_kern: &'a MultiexpKernel<'a, 'b, G>,
}

impl<G: AffineCurve> MSMContext<'_, '_, G> {
    pub fn read_bases_async(&self, stream: &CudaStream, bases: &mut [G1AffineNoInfinity]) {
        self.bases
            .write_to_async(bases, bases.len(), stream)
            .unwrap();
    }

    pub fn get_log_count(&self) -> u32 {
        self.log_count
    }
}

pub struct ExecuteConfiguration<'a> {
    pub bases: CUdeviceptr,
    pub scalars: &'a Vec<BigInteger256>,
    pub results: &'a mut [G1Projective],
    pub log_scalars_count: u32,
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
    pub window_bits_count: u32,
    pub precomputed_windows_stride: u32,
    pub precomputed_bases_stride: u32,
    pub scalars_not_montgomery: bool,
}
pub struct MsmConfiguration<'a> {
    pub bases: CUdeviceptr,
    pub scalars: &'a Vec<BigInteger256>,
    pub results: &'a mut [G1Projective],
    pub log_scalars_count: u32,
    pub h2d_copy_finished: Option<&'a CUevent>,
    pub h2d_copy_finished_callback: Option<&'a HostFn<'a>>,
    pub d2h_copy_finished: Option<&'a CUevent>,
    pub d2h_copy_finished_callback: Option<&'a HostFn<'a>>,
    pub force_min_chunk_size: bool,
    pub log_min_chunk_size: u32,
    pub force_max_chunk_size: bool,
    pub log_max_chunk_size: u32,
    pub window_bits_count: u32,
    pub precomputed_windows_stride: u32,
    pub precomputed_bases_stride: u32,
    pub scalars_not_montgomery: bool,
}

// extended_configuration
struct ExtendedConfiguration<'a> {
    exec_cfg: ExecuteConfiguration<'a>,
    log_min_inputs_count: u32,
    log_max_inputs_count: u32,
    multiprocessor_count: u32,
}

impl<'a, 'b, G> MultiexpKernel<'a, 'b, G>
where
    G: AffineCurve,
{
    pub fn left_shift(&self, values_addr: CUdeviceptr, shift: u32, count: u32) -> GPUResult<()> {
        let (gws, lws) = calc_cuda_wg_threads((count) as usize);
        let f = self.kernel_func_map.get("left_shift_kernel").unwrap().inner;

        let params = make_params!(values_addr, shift, count);
        let kern = create_kernel_with_params!(f, <<<gws, lws, 0>>>(params));

        self.stream.launch(&kern)?;

        Ok(())
    }

    // todo G1Affine  ==>T
    pub fn multi_scalar_mult_init<'c, 'd>(
        &self,
        bases: &'c [G1Affine],
        msm_kern: &'d MultiexpKernel<'a, 'b, G>,
    ) -> MSMContext<'c, 'd, G> {
        const WINDOW_BITS_COUNT: u32 = 17;
        const PRECOMPUTE_FACTOR: u32 = 4;
        const WINDOWS_COUNT: u32 = (BLS_381_FR_MODULUS_BITS - 1) / WINDOW_BITS_COUNT + 1;
        const PRECOMPUTE_WINDOWS_STRIDE: u32 = (WINDOWS_COUNT - 1) / PRECOMPUTE_FACTOR + 1;
        let result_bits_count: u32 = WINDOW_BITS_COUNT * PRECOMPUTE_WINDOWS_STRIDE;

        let precompute_factor: usize = PRECOMPUTE_FACTOR as usize;

        let count = bases.len();
        let log_count = log2(count);

        assert_eq!(count, 1 << log_count);

        let bases_device = DeviceMemory::<G1AffineNoInfinity>::new(
            msm_kern.core.get_context(),
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

        let stride = count as u64 * std::mem::size_of::<G1AffineNoInfinity>() as u64;
        for i in 1..precompute_factor as usize {
            let dst_offset = (i as u64 * stride) + bases_device.get_inner();
            let src_offset = ((i - 1) as u64 * stride) + bases_device.get_inner();

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
            window_bits_count: WINDOW_BITS_COUNT,
            precompute_factor: PRECOMPUTE_FACTOR,
            msm_kern,
        }
    }

    fn initialize_buckets(
        &self,
        buckets_buf: &DeviceMemory<PointXyzz>,
        count: u32,
    ) -> GPUResult<()> {
        const UINT4_COUNT: u32 = 3; //todo check

        assert_eq!(buckets_buf.size(), count as usize);
        let count_u4 = UINT4_COUNT * count;

        let (gws, lws) = calc_cuda_wg_threads((count_u4) as usize);
        let f = self
            .kernel_func_map
            .get("initialize_buckets_kernel")
            .unwrap()
            .inner;

        let params = make_params!(buckets_buf.get_inner(), count_u4);
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

    fn reduce_buckets(&self, buckets: &DeviceMemory<PointXyzz>, count: u32) -> GPUResult<()> {
        let (gws, lws) = calc_cuda_wg_threads(count as usize);
        let f = self
            .kernel_func_map
            .get("reduce_buckets_kernel")
            .unwrap()
            .inner;

        let params = make_params!(buckets.get_inner(), count);
        let kern = create_kernel_with_params!(f, <<<gws, lws, 0>>>(params));

        self.stream.launch(&kern)?;

        Ok(())
    }

    fn last_pass_gather(
        &self,
        bits_count_pass_one: u32,
        source: &DeviceMemory<PointXyzz>,
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
            source.get_inner(),
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
        source_buckets: &DeviceMemory<PointXyzz>,
        target_buckets: &mut DeviceMemory<PointXyzz>,
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
            source_buckets.get_inner(),
            target_buckets.get_inner(),
            count
        );
        let kern = create_kernel_with_params!(f, <<<gws, lws, 0>>>(params));
        self.stream.launch(&kern)?;

        Ok(())
    }

    #[allow(dead_code)]
    fn test_kernel_local(&self, addr: u64, addr2: u64, size: u32) -> GPUResult<()> {
        //
        let (gws, lws) = (1, 1);
        let f = self.kernel_func_map.get("test_kernel").unwrap().inner;
        let params = make_params!(addr, addr2, size);
        let kern = create_kernel_with_params!(f, <<<gws,lws, 0>>>(params));

        self.stream.launch(&kern)?;
        Ok(())
    }

    fn schedule_execution(&self, cfg: &mut ExtendedConfiguration, dry_run: bool) {
        let core = self.core;
        let ec = &mut cfg.exec_cfg;
        let stream = &self.stream;
        let log_scalars_count = ec.log_scalars_count;
        let log_max_inputs_count = cfg.log_max_inputs_count;
        let bits_count_pass_one = match ec.window_bits_count > 0 {
            true => ec.window_bits_count,
            false => get_window_bits_count(log_scalars_count),
        };
        let precomputed_windows_stride = ec.precomputed_windows_stride;
        let precomputed_bases_stride = ec.precomputed_bases_stride;
        let total_windows_count = get_windows_count(bits_count_pass_one);
        let top_window_unused_bits =
            total_windows_count * bits_count_pass_one - BLS_381_FR_MODULUS_BITS;
        let precomputations_count = match precomputed_windows_stride > 0 {
            true => (total_windows_count - 1) / precomputed_windows_stride + 1,
            false => 1,
        };
        let log_precomputations_count = log2_ceiling(precomputations_count);
        let windows_count_pass_one = match precomputed_windows_stride > 0 {
            true => precomputed_windows_stride,
            false => total_windows_count,
        };
        let buckets_count_pass_one = windows_count_pass_one << bits_count_pass_one;
        let buckets_pass_one = DeviceMemory::<PointXyzz>::new(
            &core.context,
            buckets_count_pass_one.try_into().unwrap(),
        )
        .unwrap();
        let copy_scalars = true; 
        let copy_bases = false;

        let mut execution_started_event: CUevent = std::ptr::null_mut();
        if !dry_run {
            cu_event_create(
                &mut execution_started_event,
                CUevent_flags_enum_CU_EVENT_DISABLE_TIMING,
            );

            handle_cuda_error!(unsafe {
                cuEventRecord(execution_started_event, stream.get_inner())
            });
        }

        let mut stream_copy_scalars: CUstream = std::ptr::null_mut();
        let mut stream_copy_finished: CUstream = std::ptr::null_mut();
        let mut stream_sort_a: CUstream = std::ptr::null_mut();
        let mut stream_sort_b: CUstream = std::ptr::null_mut();

        if !dry_run {
            if copy_scalars {
                handle_cuda_error!(unsafe {
                    cuStreamCreate(
                        &mut stream_copy_scalars,
                        CUstream_flags_enum_CU_STREAM_DEFAULT,
                    )
                });
                handle_cuda_error!(unsafe {
                    cuStreamWaitEvent(
                        stream_copy_scalars,
                        execution_started_event,
                        CUevent_wait_flags_enum_CU_EVENT_WAIT_DEFAULT,
                    )
                });
            }
            if copy_scalars || copy_bases {
                handle_cuda_error!(unsafe {
                    cuStreamCreate(
                        &mut stream_copy_finished,
                        CUstream_flags_enum_CU_STREAM_DEFAULT,
                    )
                });
            }

            handle_cuda_error!(unsafe {
                cuStreamCreate(&mut stream_sort_a, CUstream_flags_enum_CU_STREAM_DEFAULT)
            });
            handle_cuda_error!(unsafe {
                cuStreamWaitEvent(
                    stream_sort_a,
                    execution_started_event,
                    CUevent_wait_flags_enum_CU_EVENT_WAIT_DEFAULT,
                )
            });
            handle_cuda_error!(unsafe {
                cuStreamCreate(&mut stream_sort_b, CUstream_flags_enum_CU_STREAM_DEFAULT)
            });
            handle_cuda_error!(unsafe {
                cuStreamWaitEvent(
                    stream_sort_b,
                    execution_started_event,
                    CUevent_wait_flags_enum_CU_EVENT_WAIT_DEFAULT,
                )
            });
            handle_cuda_error!(unsafe { cuEventDestroy_v2(execution_started_event) });
        }

        // scalars inputs_scalars;
        let mut inputs_scalars = MaybeUninit::<DeviceMemory<BigInteger256>>::uninit();
        let mut event_scalars_free: CUevent = std::ptr::null_mut();
        let mut event_scalars_loaded: CUevent = std::ptr::null_mut();

        if copy_scalars {
            unsafe {
                inputs_scalars.as_mut_ptr().write(
                    DeviceMemory::<BigInteger256>::new(
                        &core.context,
                        ec.scalars.len(), //todo
                    )
                    .unwrap(),
                )
            };
            cu_event_create(
                &mut event_scalars_free,
                CUevent_flags_enum_CU_EVENT_DISABLE_TIMING,
            );
            cu_event_create(
                &mut event_scalars_loaded,
                CUevent_flags_enum_CU_EVENT_DISABLE_TIMING,
            );
        }

        let log_inputs_count = log_max_inputs_count;
        let mut inputs_scalars = unsafe { inputs_scalars.assume_init() };
        let inputs_count = 1 << log_inputs_count;
        let is_first_loop = true;

        let input_indexes_count = total_windows_count << log_inputs_count;
        if !dry_run {
            if is_first_loop {
                self.initialize_buckets(&buckets_pass_one, buckets_count_pass_one)
                    .unwrap();
            }

            if copy_scalars {
                cu_stream_wait_event(
                    stream_copy_scalars,
                    event_scalars_free,
                    CUevent_wait_flags_enum_CU_EVENT_WAIT_DEFAULT,
                );

                let temp_stream = CudaStream::new_from_cu(stream_copy_scalars).unwrap();
                inputs_scalars
                    .read_from_async(&ec.scalars, ec.scalars.len(), &temp_stream)
                    .unwrap();

                assert_eq!(ec.scalars.len(), 1 << log_inputs_count);

                handle_cuda_error!(unsafe {
                    cuEventRecord(event_scalars_loaded, stream_copy_scalars)
                });
            }

            if copy_scalars {
                handle_cuda_error!(unsafe {
                    cuStreamWaitEvent(
                        stream_copy_finished,
                        event_scalars_loaded,
                        CUevent_wait_flags_enum_CU_EVENT_WAIT_DEFAULT,
                    )
                });
            }
            if ec.h2d_copy_finished.is_some() {
                handle_cuda_error!(unsafe {
                    cuEventRecord(ec.h2d_copy_finished.unwrap().clone(), stream_copy_finished)
                });
            }

            if ec.h2d_copy_finished_callback.is_some() {
                handle_cuda_error!(unsafe {
                    cuLaunchHostFunc(
                        stream_copy_finished,
                        ec.h2d_copy_finished_callback,
                        ec.h2d_copy_finished_callback_data,
                    )
                });
            }
        }

        let mut bucket_indexes =
            DeviceMemory::<u32>::new(&core.context, input_indexes_count.try_into().unwrap())
                .unwrap();

        let mut base_indexes =
            DeviceMemory::<u32>::new(&core.context, input_indexes_count.try_into().unwrap())
                .unwrap();

        if !dry_run {
            if copy_scalars {
                handle_cuda_error!(unsafe {
                    cuStreamWaitEvent(
                        stream.get_inner(),
                        event_scalars_loaded,
                        CUevent_wait_flags_enum_CU_EVENT_WAIT_DEFAULT,
                    )
                });
            }

            self.compute_bucket_indexes(
                &inputs_scalars,
                total_windows_count,
                bits_count_pass_one,
                precomputed_windows_stride,
                precomputed_bases_stride,
                &bucket_indexes,
                &base_indexes,
                inputs_count,
                ec.scalars_not_montgomery,
            )
            .unwrap();
            if copy_scalars {
                handle_cuda_error!(unsafe {
                    cuEventRecord(event_scalars_free, stream.get_inner())
                });
            }
        }

        inputs_scalars.free().unwrap();
        // sort base indexes by bucket indexes
        let mut sorted_bucket_indexes =
            DeviceMemory::<u32>::new(&core.context, input_indexes_count.try_into().unwrap())
                .unwrap();

        let mut sorted_base_indexes =
            DeviceMemory::<u32>::new(&core.context, input_indexes_count.try_into().unwrap())
                .unwrap();

        let mut input_indexes_sort_temp_storage_bytes: usize = 0;
        sort_pairs_local(
            std::ptr::null_mut(),
            &mut input_indexes_sort_temp_storage_bytes,
            &bucket_indexes,
            &sorted_bucket_indexes,
            &base_indexes,
            &sorted_base_indexes,
            input_indexes_count as i32,
            0,
            bits_count_pass_one as i32,
            stream.get_inner(),
        );

        let mut input_indexes_sort_temp_storage =
            DeviceMemory::<u8>::new(&core.context, input_indexes_sort_temp_storage_bytes).unwrap();

        let input_indexes_sort_temp_storage_void = unsafe {
            std::mem::transmute::<u64, *mut c_void>(input_indexes_sort_temp_storage.get_inner())
        };

        if !dry_run {
            sort_pairs_local(
                input_indexes_sort_temp_storage_void,
                &mut input_indexes_sort_temp_storage_bytes,
                &bucket_indexes,
                &sorted_bucket_indexes,
                &base_indexes,
                &sorted_base_indexes,
                input_indexes_count as i32,
                0,
                bits_count_pass_one as i32,
                stream.get_inner(),
            );
        }

        input_indexes_sort_temp_storage.free().unwrap();
        bucket_indexes.free().unwrap();
        base_indexes.free().unwrap();

        // run length encode bucket runs
        let mut unique_bucket_indexes =
            DeviceMemory::<u32>::new(&core.context, buckets_count_pass_one.try_into().unwrap())
                .unwrap();

        let mut bucket_run_lengths =
            DeviceMemory::<u32>::new(&core.context, buckets_count_pass_one.try_into().unwrap())
                .unwrap();

        let mut bucket_runs_count = DeviceMemory::<u32>::new(&core.context, 1).unwrap();

        if !dry_run {
            handle_cuda_error!(unsafe {
                cuMemsetD32Async(
                    bucket_run_lengths.get_inner(),
                    0,
                    (buckets_count_pass_one).try_into().unwrap(),
                    stream.get_inner(),
                )
            });
        }

        // temp_storage encode_temp_storage;
        let mut encode_temp_storage_bytes: usize = 0;
        run_length_encode_local(
            std::ptr::null_mut(),
            &mut encode_temp_storage_bytes,
            &sorted_bucket_indexes,
            &unique_bucket_indexes,
            &bucket_run_lengths,
            &bucket_runs_count,
            input_indexes_count as i32,
            stream.get_inner(),
        );

        let mut encode_temp_storage =
            DeviceMemory::<u8>::new(&core.context, encode_temp_storage_bytes).unwrap();
        let encode_temp_storage_void =
            unsafe { std::mem::transmute::<u64, *mut c_void>(encode_temp_storage.get_inner()) };
        if !dry_run {
            run_length_encode_local(
                encode_temp_storage_void,
                &mut encode_temp_storage_bytes,
                &sorted_bucket_indexes,
                &unique_bucket_indexes,
                &bucket_run_lengths,
                &bucket_runs_count,
                input_indexes_count as i32,
                stream.get_inner(),
            );
        }

        encode_temp_storage.free().unwrap();
        bucket_runs_count.free().unwrap();
        sorted_bucket_indexes.free().unwrap();

        // compute bucket run offsets
        let mut bucket_run_offsets =
            DeviceMemory::<u32>::new(&core.context, buckets_count_pass_one.try_into().unwrap())
                .unwrap();

        let mut scan_temp_storage_bytes: usize = 0;
        exclusive_sum_local(
            std::ptr::null_mut(),
            &mut scan_temp_storage_bytes,
            &bucket_run_lengths,
            &bucket_run_offsets,
            buckets_count_pass_one as i32,
            stream.get_inner(),
        );
        let mut scan_temp_storage =
            DeviceMemory::<u8>::new(&core.context, scan_temp_storage_bytes).unwrap();
        let scan_temp_storage_void =
            unsafe { std::mem::transmute::<u64, *mut c_void>(scan_temp_storage.get_inner()) };
        if !dry_run {
            exclusive_sum_local(
                scan_temp_storage_void,
                &mut scan_temp_storage_bytes,
                &bucket_run_lengths,
                &bucket_run_offsets,
                buckets_count_pass_one as i32,
                stream.get_inner(),
            );
        }
        scan_temp_storage.free().unwrap();

        let mut sorted_bucket_run_lengths =
            DeviceMemory::<u32>::new(&core.context, buckets_count_pass_one.try_into().unwrap())
                .unwrap();

        let mut sorted_bucket_run_offsets =
            DeviceMemory::<u32>::new(&core.context, buckets_count_pass_one.try_into().unwrap())
                .unwrap();

        let mut sorted_unique_bucket_indexes =
            DeviceMemory::<u32>::new(&core.context, buckets_count_pass_one.try_into().unwrap())
                .unwrap();

        let mut sort_offsets_temp_storage_bytes: usize = 0;
        sort_pairs_descending_local(
            std::ptr::null_mut(),
            &mut sort_offsets_temp_storage_bytes,
            &bucket_run_lengths,
            &sorted_bucket_run_lengths,
            &bucket_run_offsets,
            &sorted_bucket_run_offsets,
            buckets_count_pass_one as i32,
            0,
            (log_inputs_count + 1 + log_precomputations_count) as i32,
            stream.get_inner(),
        );

        let mut sort_offsets_temp_storage = DeviceMemory::<u8>::new(
            &core.context,
            sort_offsets_temp_storage_bytes.try_into().unwrap(),
        )
        .unwrap();

        let mut sort_indexes_temp_storage_bytes: usize = 0;
        sort_pairs_descending_local(
            std::ptr::null_mut(),
            &mut sort_indexes_temp_storage_bytes,
            &bucket_run_lengths,
            &sorted_bucket_run_lengths,
            &unique_bucket_indexes,
            &sorted_unique_bucket_indexes,
            buckets_count_pass_one as i32,
            0,
            (log_inputs_count + 1 + log_precomputations_count) as i32,
            stream.get_inner(),
        );

        let mut sort_indexes_temp_storage =
            DeviceMemory::<u8>::new(&core.context, sort_indexes_temp_storage_bytes as usize)
                .unwrap();

        if !dry_run {
            let mut event_sort_inputs_ready: CUevent = std::ptr::null_mut();

            cu_event_create(
                &mut event_sort_inputs_ready,
                CUevent_flags_enum_CU_EVENT_DISABLE_TIMING,
            );
            handle_cuda_error!(unsafe {
                cuEventRecord(event_sort_inputs_ready, stream.get_inner())
            });
            handle_cuda_error!(unsafe {
                cuStreamWaitEvent(
                    stream_sort_a,
                    event_sort_inputs_ready,
                    CUevent_wait_flags_enum_CU_EVENT_WAIT_DEFAULT,
                )
            });
            handle_cuda_error!(unsafe {
                cuStreamWaitEvent(
                    stream_sort_b,
                    event_sort_inputs_ready,
                    CUevent_wait_flags_enum_CU_EVENT_WAIT_DEFAULT,
                )
            });
            handle_cuda_error!(unsafe { cuEventDestroy_v2(event_sort_inputs_ready) });

            let sort_offsets_temp_storage_void = unsafe {
                std::mem::transmute::<u64, *mut c_void>(sort_offsets_temp_storage.get_inner())
            };

            sort_pairs_descending_local(
                sort_offsets_temp_storage_void,
                &mut sort_offsets_temp_storage_bytes,
                &bucket_run_lengths,
                &sorted_bucket_run_lengths,
                &bucket_run_offsets,
                &sorted_bucket_run_offsets,
                buckets_count_pass_one as i32,
                0,
                (log_inputs_count + 1 + log_precomputations_count) as i32,
                stream_sort_a,
            );

            let sort_indexes_temp_storage_void = unsafe {
                std::mem::transmute::<u64, *mut c_void>(sort_indexes_temp_storage.get_inner())
            };
            sort_pairs_descending_local(
                sort_indexes_temp_storage_void,
                &mut sort_indexes_temp_storage_bytes,
                &bucket_run_lengths,
                &sorted_bucket_run_lengths,
                &unique_bucket_indexes,
                &sorted_unique_bucket_indexes,
                buckets_count_pass_one as i32,
                0,
                (log_inputs_count + 1 + log_precomputations_count) as i32,
                stream_sort_b,
            );

            let mut event_sort_a: CUevent = std::ptr::null_mut();
            let mut event_sort_b: CUevent = std::ptr::null_mut();

            //cuEventCreate
            cu_event_create(
                &mut event_sort_inputs_ready,
                CUevent_flags_enum_CU_EVENT_DISABLE_TIMING,
            );

            cu_event_create(
                &mut event_sort_a,
                CUevent_flags_enum_CU_EVENT_DISABLE_TIMING,
            );
            cu_event_create(
                &mut event_sort_b,
                CUevent_flags_enum_CU_EVENT_DISABLE_TIMING,
            );
            handle_cuda_error!(unsafe { cuEventRecord(event_sort_a, stream_sort_a) });
            handle_cuda_error!(unsafe { cuEventRecord(event_sort_b, stream_sort_b) });
            handle_cuda_error!(unsafe {
                cuStreamWaitEvent(
                    stream.get_inner(),
                    event_sort_a,
                    CUevent_wait_flags_enum_CU_EVENT_WAIT_DEFAULT,
                )
            });
            handle_cuda_error!(unsafe {
                cuStreamWaitEvent(
                    stream.get_inner(),
                    event_sort_b,
                    CUevent_wait_flags_enum_CU_EVENT_WAIT_DEFAULT,
                )
            });
            handle_cuda_error!(unsafe { cuEventDestroy_v2(event_sort_a) });
            handle_cuda_error!(unsafe { cuEventDestroy_v2(event_sort_b) });
        }

        sort_offsets_temp_storage.free().unwrap();
        sort_indexes_temp_storage.free().unwrap();
        bucket_run_lengths.free().unwrap();
        bucket_run_offsets.free().unwrap();
        unique_bucket_indexes.free().unwrap();

        // aggregate buckets
        if !dry_run {
            self.aggregate_buckets(
                is_first_loop,
                &sorted_base_indexes,
                &sorted_bucket_run_offsets,
                &sorted_bucket_run_lengths,
                &sorted_unique_bucket_indexes,
                ec.bases,
                &buckets_pass_one,
                buckets_count_pass_one,
            )
            .unwrap();
        }
        sorted_base_indexes.free().unwrap();
        sorted_bucket_run_offsets.free().unwrap();
        sorted_bucket_run_lengths.free().unwrap();
        sorted_unique_bucket_indexes.free().unwrap();

        let mut source_bits_count = bits_count_pass_one;
        let mut source_windows_count = windows_count_pass_one;
        let mut source_buckets = buckets_pass_one;
        loop {
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

            let mut target_buckets = DeviceMemory::<PointXyzz>::new(
                &core.context,
                total_buckets_count.try_into().unwrap(),
            )
            .unwrap();
            if !dry_run {
                self.split_windows(
                    source_bits_count,
                    source_windows_count,
                    &source_buckets,
                    &mut target_buckets,
                    total_buckets_count,
                )
                .unwrap();
            }

            if !dry_run {
                let mut j: u32 = 0;
                while j < log_data_split {
                    self.reduce_buckets(&target_buckets, total_buckets_count >> (j + 1))
                        .unwrap();
                    j += 1;
                }
            }

            if target_bits_count == 1 {
                let result_windows_count = min(
                    BLS_381_FR_MODULUS_BITS,
                    windows_count_pass_one * bits_count_pass_one,
                );
                let results = {
                    DeviceMemory::<G1Projective>::new(&core.context, result_windows_count as usize)
                        .unwrap()
                };
                if !dry_run {
                    self.last_pass_gather(
                        bits_count_pass_one,
                        &target_buckets,
                        &results,
                        result_windows_count,
                    )
                    .unwrap();

                    results
                        .write_to_async(&mut ec.results, result_windows_count as usize, &stream)
                        .unwrap();

                    if ec.d2h_copy_finished.is_some() {
                        handle_cuda_error!(unsafe {
                            cuEventRecord(ec.d2h_copy_finished.unwrap().clone(), stream.get_inner())
                        });
                    }

                    if ec.d2h_copy_finished_callback.is_some() {
                        handle_cuda_error!(unsafe {
                            cuLaunchHostFunc(
                                stream.get_inner(),
                                ec.d2h_copy_finished_callback,
                                ec.d2h_copy_finished_callback_data,
                            )
                        });
                    }
                }
                break;
            }

            std::mem::swap(&mut source_buckets, &mut target_buckets);
            target_buckets.free().unwrap();
            source_bits_count = target_bits_count;
            source_windows_count = target_windows_count;
        }
    }

    pub fn execute_msm(&self, exec_cfg: ExecuteConfiguration) {
        let log_scalars_count = exec_cfg.log_scalars_count;
        let copy_scalars = true;
        let log_min_inputs_count = match exec_cfg.force_min_chunk_size {
            true => exec_cfg.log_min_chunk_size,
            false => match copy_scalars {
                true => get_log_min_inputs_count(log_scalars_count),
                false => log_scalars_count,
            },
        };
        let log_max_chunk_size = exec_cfg.log_max_chunk_size.clone();
        let mut cfg = ExtendedConfiguration {
            exec_cfg,
            log_min_inputs_count: min(log_min_inputs_count, log_max_chunk_size),
            log_max_inputs_count: log_max_chunk_size,
            multiprocessor_count: self.core_count as u32,
        };
        self.schedule_execution(&mut cfg, false);
    }
}

// for plonk
fn execute_msm<'a, 'b, G>(
    configuration: &mut MsmConfiguration<'_>,
    kern: &MultiexpKernel<'a, 'b, G>,
) where
    G: AffineCurve,
{
    let conf = unsafe {
        ExecuteConfiguration {
            bases: configuration.bases,
            scalars: configuration.scalars,
            results: configuration.results,
            log_scalars_count: configuration.log_scalars_count,
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
            window_bits_count: configuration.window_bits_count,
            precomputed_windows_stride: configuration.precomputed_windows_stride,
            precomputed_bases_stride: configuration.precomputed_bases_stride,
            scalars_not_montgomery: true,
        }
    };
    kern.execute_msm(conf);
}

pub fn gpu_msm<'a, 'b, G>(
    scalars: &Vec<BigInteger256>,
    msm_context: &MSMContext<'a, 'b, G>,
) -> G1Projective
where
    G: AffineCurve,
{
    let windows_count: u32 = (BLS_381_FR_MODULUS_BITS - 1) / msm_context.window_bits_count + 1;
    let precompute_windows_stride: u32 = (windows_count - 1) / msm_context.precompute_factor + 1;
    let result_bits_count: u32 = msm_context.window_bits_count * precompute_windows_stride;

    let kern = msm_context.msm_kern;

    let mut results = vec![G1Projective::zero(); result_bits_count as usize];
    let mut cfg = MsmConfiguration {
        bases: msm_context.bases.get_inner(),
        scalars: &scalars,
        results: &mut results,
        log_scalars_count: msm_context.log_count,
        h2d_copy_finished: None,
        h2d_copy_finished_callback: None,
        d2h_copy_finished: None,
        d2h_copy_finished_callback: None,
        force_min_chunk_size: false,
        log_min_chunk_size: 0,
        force_max_chunk_size: true,
        log_max_chunk_size: msm_context.log_count,
        window_bits_count: msm_context.window_bits_count,
        precomputed_windows_stride: precompute_windows_stride,
        precomputed_bases_stride: 1u32 << msm_context.log_count,
        scalars_not_montgomery: true,
    };
    let _start = std::time::Instant::now();
    execute_msm(&mut cfg, &kern);

    kern.stream.sync().unwrap();
    let results_gpu = reduce_results(&results);
    // println!(
    //     "msm len:2^{} spent, {:?}",
    //     log2(scalars.len()),
    //     _start.elapsed()
    // );

    results_gpu
}
