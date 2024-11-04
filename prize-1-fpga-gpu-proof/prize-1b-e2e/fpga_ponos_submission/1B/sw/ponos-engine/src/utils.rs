pub fn bench<R>(what: &str, f: impl FnOnce() -> R) -> R {
    use std::time::Instant;

    println!("Starting '{}'", what);
    let start = Instant::now();
    let result = f();
    let duration = start.elapsed().as_micros();
    println!("Finished '{}', duration: {} us", what, duration);
    return result;
}

pub fn no_bench<R>(_: &str, f: impl FnOnce() -> R) -> R {
    return f();
}

pub unsafe fn transmute_ref<Dst, Src>(src: &Src) -> &Dst {
    &*(src as *const _ as *const Dst)
}

pub unsafe fn transmute_ref_mut<Dst, Src>(src: &mut Src) -> &mut Dst {
    &mut *(src as *mut _ as *mut Dst)
}

pub unsafe fn transmute_slice<Dst: Sized, Src: Sized>(src: &[Src]) -> &[Dst] {
    let transmuted_size =
        src.len() * std::mem::size_of::<Src>() / std::mem::size_of::<Dst>();
    core::slice::from_raw_parts(src.as_ptr() as *const Dst, transmuted_size)
}

pub unsafe fn transmute_slice_mut<Dst: Sized, Src: Sized>(src: &mut [Src]) -> &mut [Dst] {
    let transmuted_size =
        src.len() * std::mem::size_of::<Src>() / std::mem::size_of::<Dst>();
    core::slice::from_raw_parts_mut(src.as_ptr() as *mut Dst, transmuted_size)
}
