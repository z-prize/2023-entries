typedef ec<fd_p> curve;
typedef curve::storage storage;
typedef curve::field field;
typedef curve::point_affine point_affine;
typedef curve::point_jacobian point_jacobian;
typedef curve::point_xyzz point_xyzz;
typedef curve::point_projective point_projective;
typedef curve::point_ed_xytz point_ed_xytz;
typedef curve::point_ed_affine point_ed_affine;


extern "C" __global__  void left_shift_kernel(point_ed_xytz *values, const unsigned shift, const unsigned count)
{
  const field f = field();
  const unsigned gid = blockIdx.x * blockDim.x + threadIdx.x;
  if (gid >= count)
    return;

  point_ed_xytz value = values[gid];

  for (unsigned i = 0; i < shift; i++)
    value = curve::point_ed_xytz_add(value, value, f);

  values[gid] = value;
}

#define MAX_THREADS 32
#define MIN_BLOCKS 16
#define UINT4_COUNT (sizeof(storage) / sizeof(uint4))
extern "C" __launch_bounds__(MAX_THREADS, MIN_BLOCKS) __global__ void initialize_buckets_kernel(point_ed_xytz *buckets, const unsigned count)
{
  const unsigned gid = blockIdx.x * blockDim.x + threadIdx.x;

  if (gid >= count)
    return;

  const auto bucket_index = gid / UINT4_COUNT;
  const auto element_index = gid % UINT4_COUNT;
  auto elements = reinterpret_cast<uint4 *>(&buckets[bucket_index].x);
  memory::store<uint4, memory::st_modifier::cs>(elements + element_index, {});

  elements = reinterpret_cast<uint4 *>(&buckets[bucket_index].t);
  memory::store<uint4, memory::st_modifier::cs>(elements + element_index, {});

  auto elements_y = reinterpret_cast<uint4 *>(&buckets[bucket_index].y);
  auto elements_z = reinterpret_cast<uint4 *>(&buckets[bucket_index].z);
  auto one = FIELD_ONE;
  auto value = make_uint4(one.val[element_index * 4], one.val[element_index * 4 + 1], one.val[element_index * 4 + 2], one.val[element_index * 4 + 3]);
  memory::store<uint4, memory::st_modifier::cs>(elements_y + element_index, value);
  memory::store<uint4, memory::st_modifier::cs>(elements_z + element_index, value);
}

#undef UINT4_COUNT
#undef MAX_THREADS
#undef MIN_BLOCKS

#define MAX_THREADS 64
#define MIN_BLOCKS 16
extern "C" __launch_bounds__(MAX_THREADS, MIN_BLOCKS) __global__ void compute_bucket_indexes_kernel(const fd_q::storage *__restrict__ scalars, unsigned source_windows_count, const unsigned window_bits,
                                                                                                    const unsigned precomputed_windows_stride, const unsigned precomputed_bases_stride,
                                                                                                    unsigned *__restrict__ bucket_indexes, unsigned *__restrict__ base_indexes, const unsigned count, bool isMont)
{
  const unsigned scalar_index = blockIdx.x * blockDim.x + threadIdx.x;
  if (scalar_index >= count)
    return;
  const fd_q::storage scalar = isMont ? memory::load(scalars + scalar_index) : fd_q::from_montgomery(memory::load(scalars + scalar_index));
  const unsigned precomputations_count = precomputed_windows_stride ? (source_windows_count - 1) / precomputed_windows_stride + 1 : 1;

  for (unsigned i = 0; i < source_windows_count; i++)
  {
    const unsigned source_window_index = i;
    const unsigned precomputed_index = precomputed_windows_stride ? source_window_index / precomputed_windows_stride : 0;
    const unsigned target_window_index = precomputed_windows_stride ? source_window_index % precomputed_windows_stride : source_window_index;
    const unsigned window_mask = target_window_index << window_bits;
    const unsigned bucket_index = fd_q::extract_bits(scalar, source_window_index * window_bits, window_bits);

    const unsigned bucket_index_offset = 0;
    const unsigned output_index = target_window_index * precomputations_count * count + precomputed_index * count + scalar_index;

    bucket_indexes[output_index] = window_mask | bucket_index_offset | bucket_index;
    base_indexes[output_index] = scalar_index + precomputed_bases_stride * precomputed_index;
  }
}
#undef MAX_THREADS
#undef MIN_BLOCKS

extern "C" __global__ void compute_top_win_bucket_indexes_kernel(const fd_q::storage *__restrict__ scalars, const unsigned source_windows_count, const unsigned window_bits,
                                                                                                            const unsigned precomputed_windows_stride, const unsigned precomputed_bases_stride,
                                                                                                            unsigned *__restrict__ bucket_indexes, unsigned *__restrict__ base_indexes, const unsigned count, bool isMont)
{
  const unsigned scalar_index = blockIdx.x * blockDim.x + threadIdx.x;
  if (scalar_index >= count)
    return;
  const fd_q::storage scalar = isMont ? memory::load(scalars + scalar_index) : fd_q::from_montgomery(memory::load(scalars + scalar_index));
  const unsigned precomputations_count = precomputed_windows_stride ? (source_windows_count - 1) / precomputed_windows_stride + 1 : 1;
  const unsigned source_window_index = source_windows_count - 1;
  const unsigned precomputed_index = precomputed_windows_stride ? source_window_index / precomputed_windows_stride : 0;
  const unsigned target_window_index = precomputed_windows_stride ? source_window_index % precomputed_windows_stride : source_window_index;
  const unsigned bucket_index = fd_q::extract_bits(scalar, source_window_index * window_bits, window_bits);
  const unsigned top_window_unused_bits = source_windows_count * window_bits - fd_q::MBC;
  const unsigned top_window_unused_mask = (1 << top_window_unused_bits) - 1;
  const unsigned top_window_used_bits = window_bits - top_window_unused_bits;

  const unsigned bucket_index_offset = (scalar_index & top_window_unused_mask) << top_window_used_bits;
  const unsigned output_index = scalar_index;

  bucket_indexes[output_index] = bucket_index_offset | bucket_index;
  base_indexes[output_index] = scalar_index + precomputed_bases_stride * precomputed_index;
}

#define MAX_THREADS 256
#define MIN_BLOCKS 16
extern "C" __launch_bounds__(MAX_THREADS, MIN_BLOCKS) __global__ void aggregate_buckets_kernel(bool IS_FIRST, const unsigned *__restrict__ base_indexes, const unsigned *__restrict__ bucket_run_offsets,
                                                                                               const unsigned *__restrict__ bucket_run_lengths, const unsigned *__restrict__ bucket_indexes,
                                                                                               const point_affine *__restrict__ bases,
                                                                                               point_ed_xytz *__restrict__ buckets, const unsigned count)
{
  const field f = field();
  const unsigned gid = blockIdx.x * blockDim.x + threadIdx.x;
  if (gid >= count)
    return;

  const unsigned length = bucket_run_lengths[gid];
  if (length == 0)
    return;
  const unsigned base_indexes_offset = bucket_run_offsets[gid];
  const unsigned *indexes = base_indexes + base_indexes_offset;
  const unsigned bucket_index = bucket_indexes[gid];
  point_ed_xytz bucket;
  if (IS_FIRST)
  {
    const unsigned base_index = *indexes++;
    auto base = memory::load<point_affine, memory::ld_modifier::g>(bases + base_index);
    point_ed_affine ed_base;
    ed_base.x = base.x;
    ed_base.y = base.y;
    ed_base.t = f.template mul<0>(ed_base.x, ed_base.y);
    bucket = point_ed_xytz::to_ed_xytz(ed_base, f);
  }
  else
  {
    bucket = memory::load<point_ed_xytz, memory::ld_modifier::cs>(buckets + bucket_index);
  }
#pragma unroll 1
  for (unsigned i = IS_FIRST ? 1 : 0; i < length; i++)
  {
    const unsigned base_index = *indexes++;
    const point_affine base = memory::load<point_affine, memory::ld_modifier::g>((point_affine *)bases + base_index);

    point_ed_affine ed_base;
    ed_base.x = base.x;
    ed_base.y = base.y;
    ed_base.t = f.template mul<0>(ed_base.x, ed_base.y);

    bucket = curve::point_ed_add_mix(bucket, ed_base, f);
  }
  memory::store<point_ed_xytz, memory::st_modifier::cs>(buckets + bucket_index, bucket);
}
#undef MAX_THREADS
#undef MIN_BLOCKS

#define MAX_THREADS 64
#define MIN_BLOCKS 12
__device__ __forceinline__ void split_windows_kernel_inner(const unsigned source_window_bits_count, const unsigned source_windows_count,
                                                           const point_ed_xytz *__restrict__ source_buckets, point_ed_xytz *__restrict__ target_buckets,
                                                           const unsigned count)
{
  const field f = field();
  const unsigned gid = blockIdx.x * blockDim.x + threadIdx.x;
  if (gid >= count)
    return;
  const unsigned target_window_bits_count = (source_window_bits_count + 1) >> 1;
  const unsigned target_windows_count = source_windows_count << 1;
  const unsigned target_partition_buckets_count = target_windows_count << target_window_bits_count;
  const unsigned target_partitions_count = count / target_partition_buckets_count;
  const unsigned target_partition_index = gid / target_partition_buckets_count;
  const unsigned target_partition_tid = gid % target_partition_buckets_count;
  const unsigned target_window_buckets_count = 1 << target_window_bits_count;
  const unsigned target_window_index = target_partition_tid / target_window_buckets_count;
  const unsigned target_window_tid = target_partition_tid % target_window_buckets_count;
  const unsigned split_index = target_window_index & 1;
  const unsigned source_window_buckets_per_target = source_window_bits_count & 1
                                                        ? split_index ? (target_window_tid >> (target_window_bits_count - 1) ? 0 : target_window_buckets_count)
                                                                      : 1 << (source_window_bits_count - target_window_bits_count)
                                                        : target_window_buckets_count;
  const unsigned source_window_index = target_window_index >> 1;
  const unsigned source_offset = source_window_index << source_window_bits_count;
  const unsigned target_shift = target_window_bits_count * split_index;
  const unsigned target_offset = target_window_tid << target_shift;
  const unsigned global_offset = source_offset + target_offset;
  const unsigned index_mask = (1 << target_shift) - 1;
  point_ed_xytz target_bucket = point_ed_xytz::point_at_infinity(f);

#pragma unroll 1
  for (unsigned i = target_partition_index; i < source_window_buckets_per_target; i += target_partitions_count)
  {
    const unsigned index_offset = i & index_mask | (i & ~index_mask) << target_window_bits_count;
    const unsigned load_offset = global_offset + index_offset;
    const auto source_bucket = memory::load<point_ed_xytz, memory::ld_modifier::g>(source_buckets + load_offset);

    target_bucket = i == target_partition_index ? source_bucket : curve::point_ed_xytz_add(target_bucket, source_bucket, f);
  }
  memory::store<point_ed_xytz, memory::st_modifier::cs>(target_buckets + gid, target_bucket);
}
#undef MAX_THREADS
#undef MIN_BLOCKS

extern "C"  __global__ void split_windows_kernel_generic(const unsigned source_window_bits_count, const unsigned source_windows_count,
                                                                                                   const point_ed_xytz *__restrict__ source_buckets, point_ed_xytz *__restrict__ target_buckets, const unsigned count)
{
  split_windows_kernel_inner(source_window_bits_count, source_windows_count, source_buckets, target_buckets, count);
}

extern "C" __global__ void split_windows_kernel_specialized(unsigned source_window_bits_count, unsigned source_windows_count,
                                                                                                       const point_ed_xytz *__restrict__ source_buckets, point_ed_xytz *__restrict__ target_buckets, const unsigned count)
{
  split_windows_kernel_inner(source_window_bits_count, source_windows_count, source_buckets, target_buckets, count);
}

#define MAX_THREADS 32
#define MIN_BLOCKS 12
extern "C" __launch_bounds__(MAX_THREADS, MIN_BLOCKS) __global__ void reduce_buckets_kernel(point_ed_xytz *buckets, const unsigned count)
{
  const field f = field();
  const unsigned gid = blockIdx.x * blockDim.x + threadIdx.x;
  if (gid >= count)
    return;
  buckets += gid;
  const auto a = memory::load<point_ed_xytz, memory::ld_modifier::g>(buckets);
  const auto b = memory::load<point_ed_xytz, memory::ld_modifier::g>(buckets + count);

  const point_ed_xytz result = curve::point_ed_xytz_add(a, b, f);

  memory::store<point_ed_xytz, memory::st_modifier::cs>(buckets, result);
}
#undef MAX_THREADS
#undef MIN_BLOCKS

extern "C" __global__  void reduce_buckets_kernel_to_jacobian(point_ed_xytz *buckets, const unsigned count, point_ed_xytz *__restrict__ target)
{
  const field f = field();
  const unsigned gid = blockIdx.x * blockDim.x + threadIdx.x;
  if (gid >= count)
    return;
  buckets += gid;
  const auto a = memory::load<point_ed_xytz, memory::ld_modifier::g>(buckets);
  const auto b = memory::load<point_ed_xytz, memory::ld_modifier::g>(buckets + count);
  const point_ed_xytz result = curve::point_ed_xytz_add(a, b, f);

  memory::store<point_ed_xytz, memory::st_modifier::cs>(target + gid, result);
}

#undef MAX_THREADS
#undef MIN_BLOCKS

extern "C" __global__  void last_pass_gather_kernel(const unsigned bits_count_pass_one, const point_ed_xytz *__restrict__ source, point_ed_xytz *__restrict__ target,
                                    const unsigned count)
{
  const unsigned gid = blockIdx.x * blockDim.x + threadIdx.x;
  if (gid >= count)
    return;
  unsigned window_index = gid / bits_count_pass_one;
  unsigned window_tid = gid % bits_count_pass_one;
  for (unsigned bits_count = bits_count_pass_one; bits_count > 1;)
  {
    bits_count = (bits_count + 1) >> 1;
    window_index <<= 1;
    if (window_tid >= bits_count)
    {
      window_index++;
      window_tid -= bits_count;
    }
  }
  const unsigned sid = (window_index << 1) + 1;
  const auto pz = memory::load<point_ed_xytz, memory::ld_modifier::g>(source + sid);

  memory::store<point_ed_xytz, memory::st_modifier::cs>(target + gid, pz);
}
