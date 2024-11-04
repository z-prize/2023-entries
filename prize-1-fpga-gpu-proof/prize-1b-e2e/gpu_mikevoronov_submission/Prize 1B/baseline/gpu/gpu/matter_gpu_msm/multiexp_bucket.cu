#include "ec.cuh"
#include "ff_dispatch_st.cuh"
#include "typedefs.cuh"

#include "msm_config.cuh"
#include "cub_sorting.cuh"

static int get_bl(const int count, const int th)
{
    return (count / th) + ((count % th) != 0);
}

#define MAX_THREADS 32
#define MIN_BLOCKS 12
__global__ void left_shift_kernel(point_affine *values, const unsigned shift, const unsigned count)
{
  const unsigned gid = blockIdx.x * blockDim.x + threadIdx.x;
  if (gid >= count)
    return;
  auto fd = fd_p();
  auto value = point_affine::to_projective(values[gid], fd);
  for (unsigned i = 0; i < shift; i++)
    value = curve::dbl(value, fd);
  const auto inverse = fd_p::inverse(value.z);
  const auto x = fd_p::mul(value.x, inverse);
  const auto y = fd_p::mul(value.y, inverse);
  values[gid] = {x, y};
}

#undef MAX_THREADS
#undef MIN_BLOCKS

extern "C" void bls12_381_left_shift_kernel(void *values, const unsigned shift, const unsigned count, cudaStream_t st)
{
    left_shift_kernel<<<get_bl(count, 32), 32, 0, st>>>((point_affine*)values, shift, count);
}

#define MAX_THREADS 32
#define MIN_BLOCKS 16
#define UINT4_COUNT (sizeof(storage) / sizeof(uint4))

__global__ void initialize_buckets_kernel(point_xyzz *buckets, const unsigned count)
{
  const unsigned gid = blockIdx.x * blockDim.x + threadIdx.x;

  if (gid >= count)
    return;
  const auto bucket_index = gid / UINT4_COUNT;
  const auto element_index = gid % UINT4_COUNT;
  auto elements = reinterpret_cast<uint4 *>(&buckets[bucket_index].zz);
  memory::store<uint4, memory::st_modifier::cs>(elements + element_index, {});
}

#undef UINT4_COUNT
#undef MAX_THREADS
#undef MIN_BLOCKS

#define MAX_THREADS 64
#define MIN_BLOCKS 16
__global__ void compute_bucket_indexes_kernel(const fd_q::storage *__restrict__ scalars, const unsigned source_windows_count, const unsigned window_bits,
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
    //const unsigned top_window_unused_bits = source_windows_count * window_bits - fd_q::MBC;
    //const unsigned top_window_unused_mask = (1 << top_window_unused_bits) - 1;
    //const unsigned top_window_used_bits = window_bits - top_window_unused_bits;

    const unsigned bucket_index_offset = 0; // source_window_index == source_windows_count - 1 ? (scalar_index & top_window_unused_mask) << top_window_used_bits : 0;
    const unsigned output_index = target_window_index * precomputations_count * count + precomputed_index * count + scalar_index;
    bucket_indexes[output_index] = window_mask | bucket_index_offset | bucket_index;
    base_indexes[output_index] = scalar_index + precomputed_bases_stride * precomputed_index; 
  }
}

#undef MAX_THREADS
#undef MIN_BLOCKS

#define MAX_THREADS 32
#define MIN_BLOCKS 8
__global__ void aggregate_buckets_kernel(bool IS_FIRST, const unsigned *__restrict__ base_indexes, const unsigned *__restrict__ bucket_run_offsets,
                                         const unsigned *__restrict__ bucket_run_lengths, const unsigned *__restrict__ bucket_indexes,
                                         const point_affine *__restrict__ bases,
                                         point_xyzz *__restrict__ buckets, const unsigned count)
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
  point_xyzz bucket;
  if (IS_FIRST)
  {
    const unsigned base_index = *indexes++;
    const auto base = memory::load<point_affine, memory::ld_modifier::g>(bases + base_index);

    bucket = point_affine::to_xyzz(base, f);

  }
  else
  {
    bucket = memory::load<point_xyzz, memory::ld_modifier::cs>(buckets + bucket_index);
  }
#pragma unroll 1
  for (unsigned i = IS_FIRST ? 1 : 0; i < length; i++)
  {
    const unsigned base_index = *indexes++;
    const auto base = memory::load<point_affine, memory::ld_modifier::g>(bases + base_index);

    bucket = curve::add(bucket, base, f);
  }
  memory::store<point_xyzz, memory::st_modifier::cs>(buckets + bucket_index, bucket);
}

#undef MAX_THREADS
#undef MIN_BLOCKS

#define MAX_THREADS 32
#define MIN_BLOCKS 12
__device__ __forceinline__ void split_windows_kernel_inner(const unsigned source_window_bits_count, const unsigned source_windows_count,
                                                           const point_xyzz *__restrict__ source_buckets, point_xyzz *__restrict__ target_buckets,
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
  point_xyzz target_bucket = point_xyzz::point_at_infinity(f);
#pragma unroll 1
  for (unsigned i = target_partition_index; i < source_window_buckets_per_target; i += target_partitions_count)
  {
    const unsigned index_offset = i & index_mask | (i & ~index_mask) << target_window_bits_count;
    const unsigned load_offset = global_offset + index_offset;
    const auto source_bucket = memory::load<point_xyzz, memory::ld_modifier::g>(source_buckets + load_offset);
    target_bucket = i == target_partition_index ? source_bucket : curve::add(target_bucket, source_bucket, f);
  }
  memory::store<point_xyzz, memory::st_modifier::cs>(target_buckets + gid, target_bucket);
}

__global__ void split_windows_kernel_generic(const unsigned source_window_bits_count, const unsigned source_windows_count,
                                             const point_xyzz *__restrict__ source_buckets, point_xyzz *__restrict__ target_buckets, const unsigned count)
{
  split_windows_kernel_inner(source_window_bits_count, source_windows_count, source_buckets, target_buckets, count);
}

__global__ void split_windows_kernel_specialized(unsigned source_window_bits_count, unsigned source_windows_count,
                                                 const point_xyzz *__restrict__ source_buckets, point_xyzz *__restrict__ target_buckets, const unsigned count)
{
  split_windows_kernel_inner(source_window_bits_count, source_windows_count, source_buckets, target_buckets, count);
}

#undef MAX_THREADS
#undef MIN_BLOCKS

#define MAX_THREADS 32
#define MIN_BLOCKS 12
__global__ void reduce_buckets_kernel(point_xyzz *buckets, const unsigned count)
{
  const field f = field();
  const unsigned gid = blockIdx.x * blockDim.x + threadIdx.x;
  if (gid >= count)
    return;
  buckets += gid;
  const auto a = memory::load<point_xyzz, memory::ld_modifier::g>(buckets);
  const auto b = memory::load<point_xyzz, memory::ld_modifier::g>(buckets + count);
  const point_xyzz result = curve::add(a, b, f);
  memory::store<point_xyzz, memory::st_modifier::cs>(buckets, result);
}

#undef MAX_THREADS
#undef MIN_BLOCKS

__global__ void last_pass_gather_kernel(const unsigned bits_count_pass_one, const point_xyzz *__restrict__ source, point_jacobian *__restrict__ target,
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
  const field f = field();
  const unsigned sid = (window_index << 1) + 1;
  const auto pz = memory::load<point_xyzz, memory::ld_modifier::g>(source + sid);
  const point_jacobian pj = point_xyzz::to_jacobian(pz, f);

  memory::store<point_jacobian, memory::st_modifier::cs>(target + gid, pj);
}

__global__ void last_pass_gather_kernel_xyzz(const unsigned bits_count_pass_one, const point_xyzz *__restrict__ source, 
                                             point_xyzz *__restrict__ target, const unsigned count)
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
  const auto pz = memory::load<point_xyzz, memory::ld_modifier::g>(source + sid);
  memory::store<point_xyzz, memory::st_modifier::cs>(target + gid, pz);
}

__global__ void test_kernel(point_xyzz *__restrict__ buckets, point_jacobian *__restrict__ target, const unsigned count)
{

  if (target != nullptr)
  {
    for (int i = 0; i < count; i++)
    {
      if (target[i].x.limbs[0] != 0)
        printf("target x:%x,%x,%x,%x\n", target[i].x.limbs[0], target[i].x.limbs[1], target[i].x.limbs[2], target[i].x.limbs[3]);
    }
  }
}

static unsigned get_windows_count(unsigned window_bits_count) {
    return (BLS_381_FR_MODULUS_BITS - 1) / window_bits_count + 1;
}

template<typename T>
struct Storage {
    T *tmp = NULL;
    size_t size = 0;
    
    //in bytes
    T* get(const size_t len) {
        //printf("get %lld\n", len);
        if (tmp == NULL) {
            SAFE_CALL(cudaMalloc((void**)&tmp, len));
            size = len;
            //printf("  ==> allocates msm tmp %f MB\n", len / 1024. / 1024.);
        }
        else if (size < len) {
            SAFE_CALL(cudaFree(tmp));
            
            SAFE_CALL(cudaMalloc((void**)&tmp, len));
            size = len;
        }
        
        return tmp;
    }
};

struct MsmArrays {
    unsigned* bucket_indexes = NULL;
    unsigned* base_indexes = NULL;
    unsigned* sorted_bucket_indexes = NULL;
    unsigned* sorted_base_indexes = NULL;
    unsigned* unique_bucket_indexes = NULL;
    unsigned* bucket_run_offsets = NULL;
    unsigned* bucket_runs_count = NULL;
    unsigned* bucket_run_lengths = NULL;
    unsigned* sorted_bucket_run_lengths = NULL;
    unsigned* sorted_bucket_run_offsets = NULL;
    unsigned* sorted_unique_bucket_indexes = NULL;
    Storage<unsigned> tmp_buf1;

    void* buckets_pass_one = NULL;
    void* target_buckets = NULL;
};

static unsigned log2_floor(unsigned num) 
{
    unsigned pow = 0;

    while ((1 << (pow + 1)) <= num) {
        pow += 1;
    }
    return pow;
}

static unsigned log2_ceiling(unsigned value) 
{
    if (value <= 1)
        return 0;
    else 
        return log2_floor(value - 1) + 1;
    
}

//TODO!!!
static unsigned get_optimal_log_data_split(
    unsigned mpc,
    unsigned source_window_bits,
    unsigned target_window_bits,
    unsigned target_windows_count)
{
    const unsigned MAX_THREADS = 32;
    const unsigned MIN_BLOCKS = 12;
    unsigned full_occupancy = mpc * MAX_THREADS * MIN_BLOCKS;
    unsigned target = full_occupancy << 6;
    unsigned unit_threads_count = target_windows_count << target_window_bits;
    unsigned split_target = log2_ceiling(target / unit_threads_count);
    unsigned split_limit = source_window_bits - target_window_bits - 1;
    return std::min(split_target, split_limit);
}

static std::map<cudaStream_t, MsmArrays> cachedArrays;
static int cores = 0;
static int getCudaCores() 
{
    if (cores == 0) 
    {
        struct cudaDeviceProp props;
        cudaGetDeviceProperties(&props, 0);
        
        cores = props.multiProcessorCount;
    }
    return cores;
}

extern "C" void execute_msm(MsmConfiguration cfg, bool get_memory)
{
    auto st = cfg.st;
    auto& cache = cachedArrays[cfg.st];
    
    unsigned*& bucket_indexes = cache.bucket_indexes;
    unsigned*& base_indexes = cache.base_indexes;
    unsigned*& sorted_bucket_indexes = cache.sorted_bucket_indexes;
    unsigned*& sorted_base_indexes = cache.sorted_base_indexes;
    unsigned*& unique_bucket_indexes = cache.unique_bucket_indexes;
    unsigned*& bucket_run_offsets = cache.bucket_run_offsets;
    unsigned*& bucket_runs_count = cache.bucket_runs_count;
    unsigned*& bucket_run_lengths = cache.bucket_run_lengths;
    unsigned*& sorted_bucket_run_lengths = cache.sorted_bucket_run_lengths;
    unsigned*& sorted_bucket_run_offsets = cache.sorted_bucket_run_offsets;
    unsigned*& sorted_unique_bucket_indexes = cache.sorted_unique_bucket_indexes;
    Storage<unsigned>& tmp_buf1 = cache.tmp_buf1;

    void*& buckets_pass_one = cache.buckets_pass_one;
    void*& target_buckets = cache.target_buckets;

    void* inputs_scalars = cfg.scalars;    

    const int log_scalars_count = cfg.log_scalars_count;
    const int log_min_inputs_count = log_scalars_count;
    const int log_max_chunk_size = cfg.log_max_chunk_size;
    
    cfg.log_min_inputs_count = std::min(log_min_inputs_count, log_max_chunk_size);
    cfg.log_max_inputs_count = log_max_chunk_size;
    cfg.multiprocessor_count = getCudaCores() * 64; // TODO!!!
    
    const int log_max_inputs_count = cfg.log_max_inputs_count;
    if (cfg.window_bits_count <= 0) {
        printf("cfg.window_bits_count must be positive!\n");
        exit(-1);
    }
    const int bits_count_pass_one = cfg.window_bits_count;
    
    const int precomputed_windows_stride = cfg.precomputed_windows_stride;
    const int precomputed_bases_stride = cfg.precomputed_bases_stride;
    const int total_windows_count = get_windows_count(bits_count_pass_one);
    //const int top_window_unused_bits = total_windows_count * bits_count_pass_one - BLS_381_FR_MODULUS_BITS;
    const int precomputations_count = (precomputed_windows_stride > 0) ? ((total_windows_count - 1) / precomputed_windows_stride + 1) : 1;
    
    const int log_precomputations_count = log2_ceiling(precomputations_count);
    const int windows_count_pass_one = (precomputed_windows_stride > 0) ? precomputed_windows_stride : total_windows_count;
    
    const int buckets_count_pass_one = windows_count_pass_one << bits_count_pass_one;
    
    const int log_inputs_count = log_max_inputs_count;
    const int inputs_count = 1 << log_inputs_count;
    const int input_indexes_count = total_windows_count << log_inputs_count;
    
    if (buckets_pass_one == NULL) {
        //printf("alloc with flag %d\n", get_memory);
        SAFE_CALL(cudaMalloc((void**)&buckets_pass_one, sizeof(point_xyzz) * buckets_count_pass_one));
        SAFE_CALL(cudaMalloc((void**)&target_buckets, sizeof(point_xyzz) * buckets_count_pass_one));

        SAFE_CALL(cudaMalloc((void**)&bucket_indexes, sizeof(int) * input_indexes_count));
        SAFE_CALL(cudaMalloc((void**)&base_indexes, sizeof(int) * input_indexes_count));
        
        SAFE_CALL(cudaMalloc((void**)&sorted_bucket_indexes, sizeof(int) * input_indexes_count));
        SAFE_CALL(cudaMalloc((void**)&sorted_base_indexes, sizeof(int) * input_indexes_count));
        
        SAFE_CALL(cudaMalloc((void**)&unique_bucket_indexes, sizeof(int) * buckets_count_pass_one));
        SAFE_CALL(cudaMalloc((void**)&bucket_run_lengths, sizeof(int) * buckets_count_pass_one));
        SAFE_CALL(cudaMalloc((void**)&bucket_run_offsets, sizeof(int) * buckets_count_pass_one));
        
        SAFE_CALL(cudaMalloc((void**)&bucket_runs_count, sizeof(int) * 1));
        
        
        SAFE_CALL(cudaMalloc((void**)&sorted_bucket_run_lengths, sizeof(int) * buckets_count_pass_one));
        SAFE_CALL(cudaMalloc((void**)&sorted_bucket_run_offsets, sizeof(int) * buckets_count_pass_one));
        SAFE_CALL(cudaMalloc((void**)&sorted_unique_bucket_indexes, sizeof(int) * buckets_count_pass_one));
    }

    if (!get_memory) {
        SAFE_CALL(cudaMemsetAsync(bucket_runs_count, 0, sizeof(int) * 1, st));
        //SAFE_CALL(cudaMemsetAsync(buckets_pass_one, 0, sizeof(point_xyzz) * buckets_count_pass_one, st));
        initialize_buckets_kernel<<<get_bl(3 * buckets_count_pass_one, 32), 32, 0, st>>>((point_xyzz*)buckets_pass_one, 3 * buckets_count_pass_one);

        SAFE_CALL(cudaMemsetAsync(bucket_run_lengths, 0, sizeof(int) * buckets_count_pass_one, st));
        compute_bucket_indexes_kernel<<<get_bl(inputs_count, 32), 32, 0, st>>>((fd_q::storage*)inputs_scalars, total_windows_count, bits_count_pass_one, precomputed_windows_stride,
                                                                                precomputed_bases_stride, bucket_indexes, base_indexes, inputs_count, cfg.scalars_not_montgomery);
    }

    size_t len_bytes = 0;
    size_t max_bytes = 0;
    //first call only for getting storage size!
    SAFE_CALL(sort_pairs(NULL, &len_bytes, bucket_indexes, sorted_bucket_indexes, base_indexes, sorted_base_indexes, input_indexes_count, 0, bits_count_pass_one, st));
    max_bytes = std::max(max_bytes, len_bytes);

    if (!get_memory)
        SAFE_CALL(sort_pairs(tmp_buf1.get(len_bytes), &len_bytes, bucket_indexes, sorted_bucket_indexes, base_indexes, sorted_base_indexes, input_indexes_count, 0, bits_count_pass_one, st));
    
    len_bytes = 0;
    SAFE_CALL(run_length_encode(NULL, &len_bytes, sorted_bucket_indexes, unique_bucket_indexes, bucket_run_lengths, bucket_runs_count, input_indexes_count, 0));
    max_bytes = std::max(max_bytes, len_bytes);
    
    if (!get_memory)
        SAFE_CALL(run_length_encode(tmp_buf1.get(len_bytes), &len_bytes, sorted_bucket_indexes, unique_bucket_indexes, bucket_run_lengths, bucket_runs_count, input_indexes_count, st));
        
    len_bytes = 0;
    SAFE_CALL(exclusive_sum(NULL, &len_bytes, bucket_run_lengths, bucket_run_offsets, buckets_count_pass_one, 0));
    max_bytes = std::max(max_bytes, len_bytes);
    
    if (!get_memory)
        SAFE_CALL(exclusive_sum(tmp_buf1.get(len_bytes), &len_bytes, bucket_run_lengths, bucket_run_offsets, buckets_count_pass_one, st));
    
    len_bytes = 0;
    SAFE_CALL(sort_pairs_descending(NULL, &len_bytes, bucket_run_lengths, sorted_bucket_run_lengths, bucket_run_offsets, sorted_bucket_run_offsets, buckets_count_pass_one, 
                                    0, (log_inputs_count + 1 + log_precomputations_count), 0));
    max_bytes = std::max(max_bytes, len_bytes);
                                    
    if (!get_memory)
        SAFE_CALL(sort_pairs_descending(tmp_buf1.get(len_bytes), &len_bytes, bucket_run_lengths, sorted_bucket_run_lengths, bucket_run_offsets, sorted_bucket_run_offsets, buckets_count_pass_one, 
                                        0, (log_inputs_count + 1 + log_precomputations_count), st));
                                    
    len_bytes = 0;
    SAFE_CALL(sort_pairs_descending(NULL, &len_bytes, bucket_run_lengths, sorted_bucket_run_lengths, unique_bucket_indexes, sorted_unique_bucket_indexes, buckets_count_pass_one,
                                    0, (log_inputs_count + 1 + log_precomputations_count), 0));
    max_bytes = std::max(max_bytes, len_bytes);

    if (!get_memory)
        SAFE_CALL(sort_pairs_descending(tmp_buf1.get(len_bytes), &len_bytes, bucket_run_lengths, sorted_bucket_run_lengths, unique_bucket_indexes, sorted_unique_bucket_indexes, buckets_count_pass_one,
                                        0, (log_inputs_count + 1 + log_precomputations_count), st));

    if (get_memory) {
        tmp_buf1.get(max_bytes);
        return;
    }

    aggregate_buckets_kernel<<<get_bl(buckets_count_pass_one, 32), 32, 0, st>>>(true, sorted_base_indexes, sorted_bucket_run_offsets, 
                                                                                sorted_bucket_run_lengths, sorted_unique_bucket_indexes, 
                                                                                (const point_affine*)cfg.bases, (point_xyzz*)buckets_pass_one, buckets_count_pass_one);
        
    int source_bits_count = bits_count_pass_one;
    int source_windows_count = windows_count_pass_one;
    auto source_buckets = buckets_pass_one;
    while (true) {
        const int target_bits_count = (source_bits_count + 1) >> 1;
        const int target_windows_count = source_windows_count << 1;
        const int target_buckets_count = target_windows_count << target_bits_count;
        const int log_data_split = get_optimal_log_data_split(cfg.multiprocessor_count, source_bits_count, target_bits_count, target_windows_count);

        const int total_buckets_count = target_buckets_count << log_data_split;
        split_windows_kernel_generic<<<get_bl(total_buckets_count, 32), 32, 0, st>>>(source_bits_count, source_windows_count, 
                                                                                     (point_xyzz*)source_buckets, (point_xyzz*)target_buckets, total_buckets_count);

        for (int j = 0; j < log_data_split; ++j) {
            const int count = total_buckets_count >> (j + 1);
            reduce_buckets_kernel<<<get_bl(count, 32), 32, 0, st>>>((point_xyzz*)target_buckets, count);
        }
        
        if (target_bits_count == 1) {
            const int result_windows_count = std::min(BLS_381_FR_MODULUS_BITS, windows_count_pass_one * bits_count_pass_one);

            len_bytes = sizeof(point_xyzz) * result_windows_count;
            point_xyzz* gpu_results = (point_xyzz*)tmp_buf1.get(len_bytes);

            last_pass_gather_kernel_xyzz<<<get_bl(result_windows_count, 32), 32, 0, st>>>(bits_count_pass_one, (point_xyzz*)target_buckets, gpu_results, result_windows_count);
            
            auto cpu_tmp = (point_xyzz*)cfg.results_tmp;

            SAFE_CALL(cudaMemcpyAsync(cpu_tmp, gpu_results, sizeof(point_xyzz) * result_windows_count, cudaMemcpyDeviceToHost, st));
            SAFE_CALL(cudaStreamSynchronize(st));

            const point_xyzz *results = (const point_xyzz *)cpu_tmp;
            const unsigned N = result_windows_count;
            
            point_xyzz full = results[N - 1];    
            auto f = fd_p();
            for (int z = N - 2; z >= 0; --z)
            {
                full = curve::dbl(full, f);
                full = curve::add(full, results[z], f);
            }
            
            point_jacobian* total = (point_jacobian*)cfg.results;
            total[0] = point_xyzz::to_jacobian(full, f);

            break;
        }

        std::swap(source_buckets, target_buckets);
        source_bits_count = target_bits_count;
        source_windows_count = target_windows_count;
    }
}