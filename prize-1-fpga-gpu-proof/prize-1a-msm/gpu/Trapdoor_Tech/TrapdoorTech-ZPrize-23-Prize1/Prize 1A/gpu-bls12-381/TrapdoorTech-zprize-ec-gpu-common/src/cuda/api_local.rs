use crypto_cuda::api::{self, *};
use crypto_cuda::DeviceMemory;
use std::ffi::{c_int, c_uint};

pub fn create_event(flags: ::std::os::raw::c_uint) -> CUevent {
    let mut event: CUevent = std::ptr::null_mut();
    let ret = unsafe { cuEventCreate(&mut event, flags) };
    assert_eq!(ret, 0);
    event
}

pub fn stream_wait_event(h_stream: CUstream, h_event: CUevent, flags: ::std::os::raw::c_uint) {
    let ret = unsafe { cuStreamWaitEvent(h_stream, h_event, flags) };
    assert_eq!(ret, 0)
}

pub fn create_stream() -> CUstream {
    let mut stream: CUstream = std::ptr::null_mut();
    let ret = unsafe { cuStreamCreate(&mut stream, CUstream_flags_enum_CU_STREAM_DEFAULT) };
    assert_eq!(ret, 0);

    stream
}

pub fn destroy_stream(stream: CUstream) {
    let ret = unsafe { cuStreamDestroy_v2(stream) };
    assert_eq!(ret, 0);
}

pub fn destroy_event(event: CUevent) {
    let ret = unsafe { cuEventDestroy_v2(event) };
    assert_eq!(ret, 0);
}

pub fn event_record(h_event: CUevent, h_stream: CUstream) {
    let ret = unsafe { cuEventRecord(h_event, h_stream) };
    assert_eq!(ret, 0)
}

pub fn memset_d32_async(
    dst_device: CUdeviceptr,
    ui: ::std::os::raw::c_uint,
    n: size_t,
    stream: CUstream,
) {
    let ret = unsafe { cuMemsetD32Async(dst_device, ui, n, stream) };
    assert_eq!(ret, 0)
}

pub fn sort_pairs_local(
    d_temp_storage: *mut ::std::os::raw::c_void,
    temp_storage_bytes: &mut usize,
    d_keys_in: &DeviceMemory<u32>,
    d_keys_out: &DeviceMemory<u32>,
    d_values_in: &DeviceMemory<u32>,
    d_values_out: &DeviceMemory<u32>,
    num_items: i32,
    begin_bit: i32,
    end_bit: i32,
    stream: CUstream,
) {
    let d_keys_in = unsafe { std::mem::transmute::<u64, *const c_uint>(d_keys_in.get_inner()) };
    let d_keys_out = unsafe { std::mem::transmute::<u64, *mut c_uint>(d_keys_out.get_inner()) };
    let d_values_in = unsafe { std::mem::transmute::<u64, *const c_uint>(d_values_in.get_inner()) };
    let d_values_out = unsafe { std::mem::transmute::<u64, *mut c_uint>(d_values_out.get_inner()) };
    let stream = unsafe { std::mem::transmute::<CUstream, *mut api::cub::CUstream_st>(stream) };

    let ret = unsafe {
        sort_pairs(
            d_temp_storage,
            temp_storage_bytes,
            d_keys_in,
            d_keys_out,
            d_values_in,
            d_values_out,
            num_items as c_int,
            begin_bit as c_int,
            end_bit as c_int,
            stream,
        )
    };

    assert_eq!(ret, 0);
}

pub fn sort_pairs_descending_local(
    d_temp_storage: *mut ::std::os::raw::c_void,
    temp_storage_bytes: &mut usize,
    d_keys_in: &DeviceMemory<u32>,
    d_keys_out: &DeviceMemory<u32>,
    d_values_in: &DeviceMemory<u32>,
    d_values_out: &DeviceMemory<u32>,
    num_items: i32,
    begin_bit: i32,
    end_bit: i32,
    stream: CUstream,
) {
    let d_keys_in = unsafe { std::mem::transmute::<u64, *const c_uint>(d_keys_in.get_inner()) };
    let d_keys_out = unsafe { std::mem::transmute::<u64, *mut c_uint>(d_keys_out.get_inner()) };
    let d_values_in = unsafe { std::mem::transmute::<u64, *const c_uint>(d_values_in.get_inner()) };
    let d_values_out = unsafe { std::mem::transmute::<u64, *mut c_uint>(d_values_out.get_inner()) };
    let stream = unsafe { std::mem::transmute::<CUstream, *mut api::cub::CUstream_st>(stream) };

    let ret = unsafe {
        sort_pairs_descending(
            d_temp_storage,
            temp_storage_bytes,
            d_keys_in,
            d_keys_out,
            d_values_in,
            d_values_out,
            num_items as c_int,
            begin_bit as c_int,
            end_bit as c_int,
            stream,
        )
    };
    assert_eq!(ret, 0);
}

pub fn run_length_encode_local(
    d_temp_storage: *mut ::std::os::raw::c_void,
    temp_storage_bytes: &mut usize,
    d_in: &DeviceMemory<u32>,
    d_unique_out: &DeviceMemory<u32>,
    d_counts_out: &DeviceMemory<u32>,
    d_num_runs_out: &DeviceMemory<u32>,
    num_items: i32,
    stream: CUstream,
) {
    let d_in = unsafe { std::mem::transmute::<u64, *mut c_uint>(d_in.get_inner()) };
    let d_unique_out = unsafe { std::mem::transmute::<u64, *mut c_uint>(d_unique_out.get_inner()) };
    let d_counts_out = unsafe { std::mem::transmute::<u64, *mut c_uint>(d_counts_out.get_inner()) };
    let d_num_runs_out =
        unsafe { std::mem::transmute::<u64, *mut c_uint>(d_num_runs_out.get_inner()) };
    let stream = unsafe { std::mem::transmute::<CUstream, *mut api::cub::CUstream_st>(stream) };

    let ret = unsafe {
        run_length_encode(
            d_temp_storage,
            temp_storage_bytes,
            d_in,
            d_unique_out,
            d_counts_out,
            d_num_runs_out,
            num_items as c_int,
            stream,
        )
    };
    assert_eq!(ret, 0);
}

pub fn exclusive_sum_local(
    d_temp_storage: *mut ::std::os::raw::c_void,
    temp_storage_bytes: &mut usize,
    d_in: &DeviceMemory<u32>,
    d_out: &DeviceMemory<u32>,
    num_items: i32,
    stream: CUstream,
) {
    let d_in = unsafe { std::mem::transmute::<u64, *mut c_uint>(d_in.get_inner()) };
    let d_out = unsafe { std::mem::transmute::<u64, *mut c_uint>(d_out.get_inner()) };
    let stream = unsafe { std::mem::transmute::<CUstream, *mut api::cub::CUstream_st>(stream) };

    let ret = unsafe {
        exclusive_sum(
            d_temp_storage,
            temp_storage_bytes,
            d_in,
            d_out,
            num_items as c_int,
            stream,
        )
    };
    assert_eq!(ret, 0);
}
