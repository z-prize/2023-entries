//  #include <stddef.h>
 #include <cuda_runtime.h>

cudaError_t sort_pairs(void *d_temp_storage, size_t *temp_storage_bytes, const unsigned *d_keys_in, unsigned *d_keys_out, const unsigned *d_values_in,
                       unsigned *d_values_out, int num_items, int begin_bit, int end_bit, cudaStream_t stream);

cudaError_t sort_pairs_descending(void *d_temp_storage, size_t *temp_storage_bytes, const unsigned *d_keys_in, unsigned *d_keys_out,
                                  const unsigned *d_values_in, unsigned *d_values_out, int num_items, int begin_bit, int end_bit, cudaStream_t stream);

cudaError_t run_length_encode(void *d_temp_storage, size_t *temp_storage_bytes, const unsigned *d_in, unsigned *d_unique_out, unsigned *d_counts_out,
                              unsigned *d_num_runs_out, int num_items, cudaStream_t stream);

cudaError_t exclusive_sum(void *d_temp_storage, size_t *temp_storage_bytes, const unsigned *d_in, unsigned *d_out, int num_items, cudaStream_t stream);